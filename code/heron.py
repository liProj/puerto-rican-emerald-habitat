"""HERON -- Hierarchical Effort-disentangled Representation for Occurrence Networks.

A checklist-level reporting-probability model for semi-structured citizen-science data.

    P(report | x) = psi(env, space, period) * p_det(effort, period, season)

The factorisation carries the occupancy model's structure, but is fitted at checklist
resolution on every record instead of a 1044-site x 9-occasion reduction. Identification
rests on four ingredients:

 1. Exclusion restriction. Sampling effort (duration, distance) enters only p_det;
    environment (elevation, greenness, geography) enters only psi. This is exactly the
    assumption the published dynamic occupancy model makes (psi, eps ~ elev + NDVI; p ~ Period),
    but here it is imposed inside a nonlinear learner.
 2. Hard monotonicity. p_det is monotonically non-decreasing in duration and distance by
    construction (non-negative weights over monotone basis functions), so the model cannot
    explain habitat signal away as effort.
 3. Grouped occupancy likelihood. For every (site, period) cell with repeat visits, the
    probability of at least one report is 1 - prod_i (1 - psi_i p_i). Supervising this against
    the observed "detected at least once" indicator is the classical repeat-visit information
    that identifies psi from p -- supplied as an auxiliary loss.
 4. Missingness-native inputs. The vegetation index enters with an explicit mask, so the
    23.1% of checklists with no cloud-free Landsat pixel are used rather than dropped.

Spatial generalisation is handled by a variance-across-blocks (V-REx) penalty computed over
the geographic blocks present in each training batch.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

# The model is small and the step count is high, so per-kernel launch overhead dominates.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def mlp(i, h, o, depth=3, p=0.1):
    L, d = [], i
    for _ in range(depth):
        L += [nn.Linear(d, h), nn.GELU(), nn.Dropout(p)]; d = h
    return nn.Sequential(*L, nn.Linear(d, o))

class MonotoneEffort(nn.Module):
    """Non-decreasing map from (duration, distance) to a detection logit contribution.
    Fixed monotone bases with non-negative (softplus) weights => guaranteed monotone."""
    def __init__(self, n_basis=8):
        super().__init__()
        self.a = nn.Parameter(torch.full((2, n_basis), -2.0))
        self.register_buffer("knots", torch.linspace(0.05, 0.95, n_basis))
        self.n_basis = n_basis
    def bases(self, u):                        # u in [0,1]; each basis is increasing in u
        u = u.unsqueeze(-1)
        return torch.cat([u, torch.sqrt(u.clamp_min(0)), torch.log1p(9 * u) / np.log(10.0),
                          torch.sigmoid((u - self.knots[:5]) * 8.0)], dim=-1)[..., :self.n_basis]
    def forward(self, e):                      # e: (N,2) scaled to [0,1]
        w = F.softplus(self.a)                 # non-negative => monotone increasing
        return (self.bases(e) * w).sum(dim=(-1, -2))

class HERON(nn.Module):
    def __init__(self, d_psi, d_det, hid=128, depth=3, dropout=0.1,
                 factorised=True, monotone=True):
        super().__init__()
        self.factorised, self.monotone = factorised, monotone
        if factorised:
            self.psi = mlp(d_psi, hid, 1, depth, dropout)
            self.det = mlp(d_det, hid, 1, depth, dropout)
            self.eff = MonotoneEffort() if monotone else mlp(2, 32, 1, 2, 0.0)
        else:                                   # ablation: one joint head over everything
            self.joint = mlp(d_psi + d_det + 2, hid, 1, depth, dropout)
    def forward(self, xp, xd, xe):
        if not self.factorised:
            return self.joint(torch.cat([xp, xd, xe], 1)).squeeze(-1), None, None
        lo_psi = self.psi(xp).squeeze(-1)
        lo_det = self.det(xd).squeeze(-1)
        lo_det = lo_det + (self.eff(xe) if self.monotone else self.eff(xe).squeeze(-1))
        # log of a product of two sigmoids, in a numerically safe form
        logit = -(F.softplus(-lo_psi) + F.softplus(-lo_det))          # = log(psi*p)
        log1m = torch.log1p(-torch.exp(logit).clamp(max=1 - 1e-7))
        return logit - log1m, lo_psi, lo_det                          # logit of the product

def focal_terms(logit, y, gamma=1.0, pos_weight=1.0):
    """Per-sample weighted focal BCE numerator and weight, so group means are one scatter away."""
    p = torch.sigmoid(logit)
    w = torch.where(y > 0.5, pos_weight * (1 - p) ** gamma, p ** gamma)
    return w * F.binary_cross_entropy_with_logits(logit, y, reduction="none"), w

def focal_bce(logit, y, gamma=1.0, pos_weight=1.0):
    num, w = focal_terms(logit, y, gamma, pos_weight)
    return num.sum() / w.sum().clamp_min(1e-8)

def vrex(num, w, blk, nb, min_n=64):
    """Variance of per-block risks via scatter ops. nb is passed in: calling .item() here would
    force a host sync on every optimisation step and dominated the runtime."""
    sn = torch.zeros(nb, device=num.device).scatter_add_(0, blk, num)
    sw = torch.zeros(nb, device=num.device).scatter_add_(0, blk, w)
    cn = torch.zeros(nb, device=num.device).scatter_add_(0, blk, torch.ones_like(w))
    keep = (cn >= min_n).to(num.dtype)            # dense mask: no host sync
    k = keep.sum()
    r = sn / sw.clamp_min(1e-8)
    mu = (r * keep).sum() / k.clamp_min(1.0)
    var = ((r - mu) ** 2 * keep).sum() / k.clamp_min(1.0)
    return var * (k >= 2).to(num.dtype)           # zero unless at least two blocks qualify

class Scaler:
    def fit(self, X):
        self.m = X.mean(0); self.s = X.std(0); self.s[self.s < 1e-8] = 1.0; return self
    def __call__(self, X): return (X - self.m) / self.s

def fit_heron(Xp, Xd, Xe, y, grp, blk, *,
              hid=128, depth=3, dropout=0.1, lr=3e-3, wd=1e-4, epochs=400, batch=16384,
              gamma=1.0, pos_weight=3.0, lam_group=0.1, lam_vrex=1.0,
              factorised=True, monotone=True, val_frac=0.18, eval_every=10,
              seed=0, device="cuda", verbose=False):
    """grp: integer (site,period) group id (-1 = not in a repeat-visit cell). blk: spatial block id.

    Model selection uses an INNER SPATIAL split: whole blocks are withheld from the training
    fold and the epoch with the best inner average precision is restored at the end. Selecting
    on a random inner split would be measured under a leakier regime than the outer test, and
    would favour epochs at which the network has memorised regional signatures. Set
    val_frac=0 to disable (the ablation row "- inner spatial early stop").
    """
    torch.manual_seed(seed); np.random.seed(seed)
    sp, sd = Scaler().fit(Xp), Scaler().fit(Xd)
    emax = np.maximum(Xe.max(0), 1e-6)
    T = lambda A: torch.as_tensor(A, dtype=torch.float32, device=device)
    xp, xd, xe = T(sp(Xp)), T(sd(Xd)), T(np.clip(Xe / emax, 0, 1))
    yy = T(y); gg = torch.as_tensor(grp, device=device); bb = torch.as_tensor(blk, device=device)
    # inner spatial validation split: whole blocks are withheld
    is_val = np.zeros(len(y), bool)
    if val_frac and val_frac > 0:
        rs = np.random.default_rng(seed)
        ub = np.unique(blk); rs.shuffle(ub)
        hold = ub[:max(1, int(round(val_frac * len(ub))))]
        cand = np.isin(blk, hold)
        if cand.any() and (~cand).any() and y[cand].sum() >= 5 and y[~cand].sum() >= 20:
            is_val = cand
    tr_i = torch.as_tensor(np.where(~is_val)[0], device=device)
    va_i = torch.as_tensor(np.where(is_val)[0], device=device)
    y_va = y[is_val]
    NB = int(blk.max()) + 1                    # hoisted: avoids a per-step device->host sync
    m = HERON(Xp.shape[1], Xd.shape[1], hid, depth, dropout, factorised, monotone).to(device)
    try:        # fused AdamW folds the per-tensor update into one kernel: a large win here
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd, fused=True)
    except (RuntimeError, ValueError):
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd, foreach=True)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs, pct_start=0.25)
    n = int(tr_i.numel())
    if batch <= 0 or batch >= n:      # full batch: fits comfortably in GB10 memory and makes the
        batch = n                     # V-REx and grouped-occupancy terms exact over all blocks/cells
    # repeat-visit cells: "detected at least once" target per (site,period) group
    ng = 0
    if lam_group > 0 and factorised:
        ng = int(grp.max()) + 1 if (grp >= 0).any() else 0
        any_y = torch.zeros(max(ng, 1), device=device)
        cnt = torch.zeros(max(ng, 1), device=device)
        if ng:
            any_y.scatter_reduce_(0, gg.clamp_min(0), yy, reduce="amax", include_self=False)
            cnt.scatter_add_(0, gg.clamp_min(0), torch.ones_like(yy))
    best = (-1.0, None)
    from sklearn.metrics import average_precision_score as _ap
    for ep in range(epochs):
        m.train(); perm = tr_i[torch.randperm(n, device=device)]
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            logit, lo_psi, lo_det = m(xp[idx], xd[idx], xe[idx])
            num, w = focal_terms(logit, yy[idx], gamma, pos_weight)
            loss = num.sum() / w.sum().clamp_min(1e-8)
            if lam_vrex > 0:                                  # V-REx over spatial blocks in-batch
                loss = loss + lam_vrex * vrex(num, w, bb[idx], NB)
            if lam_group > 0 and factorised and ng:           # grouped occupancy likelihood
                # Kept fully dense: boolean-mask indexing here (`logit[sel]`, `acc[act]`) forces a
                # device->host sync on every optimisation step and dominated the runtime. Mask by
                # multiplication instead, which is numerically identical.
                g = gg[idx].clamp_min(0)
                keep = (gg[idx] >= 0).to(logit.dtype)
                log1mq = -F.softplus(logit) * keep            # log(1 - psi*p), zeroed outside cells
                acc = torch.zeros(ng, device=device).scatter_add_(0, g, log1mq)
                cn = torch.zeros(ng, device=device).scatter_add_(0, g, keep)
                valid = (cn > 1).to(logit.dtype)              # cells with >=2 visits in this batch
                p_any = (1 - torch.exp(acc.clamp(min=-30))).clamp(1e-6, 1 - 1e-6)
                bce_g = F.binary_cross_entropy(p_any, any_y, reduction="none")
                loss = loss + lam_group * (bce_g * valid).sum() / valid.sum().clamp_min(1.0)
            opt.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        sch.step()
        if va_i.numel() and ((ep + 1) % eval_every == 0 or ep == epochs - 1):
            m.eval()
            with torch.no_grad():
                pv = torch.sigmoid(m(xp[va_i], xd[va_i], xe[va_i])[0]).float().cpu().numpy()
            sc = _ap(y_va, pv)
            if sc > best[0]:
                best = (sc, {k: v.detach().clone() for k, v in m.state_dict().items()})
    if best[1] is not None:
        m.load_state_dict(best[1])                 # restore the best inner-validation epoch
    m.eval()
    def predict(Xp2, Xd2, Xe2, parts=False):
        with torch.no_grad():
            lg, lp, ld = m(T(sp(Xp2)), T(sd(Xd2)), T(np.clip(Xe2 / emax, 0, 1)))
            if parts:
                return (torch.sigmoid(lg).cpu().numpy(),
                        None if lp is None else torch.sigmoid(lp).cpu().numpy(),
                        None if ld is None else torch.sigmoid(ld).cpu().numpy())
            return torch.sigmoid(lg).cpu().numpy()
    return m, predict
