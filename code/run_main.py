"""Main experiment: three evaluation protocols x two vegetation indices x all methods,
plus the ablation grid and the temporal-extrapolation test. Writes one tidy CSV per block."""
import sys, os, time, json, argparse
sys.path.insert(0, "Birds/code")
import numpy as np, pandas as pd, data_prep as dp, heron as H, baselines as B
from hybrid import HeronStack
from metrics import all_metrics
from calib import platt_fit, platt_apply

OUT = "Birds/results"
# ---------------- configurations ----------------
# neural branch (no terrain: it overfits the network, see the ablation)
NN = dict(epochs=300, batch=32768, hid=128, depth=3, dropout=0.1, lr=3e-3, wd=1e-4,
          gamma=0.0, pos_weight=1.0, lam_group=0.1, lam_vrex=1.0,
          factorised=True, monotone=True)
# full HERON: stacked over the neural branch, terrain-augmented trees and the published GLM
STACK = dict(n_inner=3, nn_ens_inner=1, nn_ens_full=2, use_nn=True, use_trees=True,
             use_glm=True, use_all=True, heron_kw=NN)

class Bench:
    def __init__(self, index="NDVI"):
        self.index = index
        self.d = dp.build(index, "complete"); self.d["Elevation_100m"] = self.d.Elevation_m / 100
        self.a = dp.build(index, "all");      self.a["Elevation_100m"] = self.a.Elevation_m / 100
        self.y, self.ya = self.d.y.values, self.a.y.values
        self.g  = pd.factorize(self.d.site.astype(str) + "|" + self.d.Period.astype(str))[0]
        self.ga = pd.factorize(self.a.site.astype(str) + "|" + self.a.Period.astype(str))[0]
        self.cache = {}
    def feats(self, tab, terrain, rff_seed=None):
        k = (id(tab), terrain, rff_seed)
        if k not in self.cache:
            r = dp.RFF(seed=rff_seed)(tab) if rff_seed is not None else None
            self.cache[k] = (dp.psi_features(tab, r, terrain), dp.det_features(tab), dp.effort_raw(tab))
        return self.cache[k]

    def stack(self, tr, fold_k, f, fa, cfg, seed):
        """Fit the stacked HERON on the training region and predict the held-out fold."""
        kw = dict(cfg); use_all = kw.pop("use_all", True)
        nn_in = kw.pop("nn_ens_inner", 1); nn_fu = kw.pop("nn_ens_full", 3)
        hk = kw.pop("heron_kw", NN)
        S = HeronStack(seed=seed, nn_ens=nn_in, epochs=hk["epochs"], batch=hk["batch"],
                       heron_kw={k: v for k, v in hk.items() if k not in ("epochs", "batch")},
                       **kw)
        S.nn_ens_full = nn_fu
        dn, bn = (self.a[fa != fold_k], fa[fa != fold_k]) if use_all else (None, None)
        S.fit(self.d[tr], f[tr], dtr_nn=dn, blk_nn=bn)
        return S.predict(self.d)

    def heron(self, tr, fold_k, f, fa, cfg, seed):
        """fold_k is the held-out FOLD INDEX (not a boolean mask): the full-table
        training region is `fa != fold_k`."""
        terrain = cfg.get("terrain", True); rffs = 0 if cfg.get("rff", False) else None
        use_all = cfg.get("use_all", True); n_ens = cfg.get("n_ens", 1)
        kw = {k: v for k, v in cfg.items()
              if k not in ("terrain", "rff", "use_all", "n_ens", "calibrate")}
        Xp_e, Xd_e, Xe_e = self.feats(self.d, terrain, rffs)
        if use_all:
            Xp_a, Xd_a, Xe_a = self.feats(self.a, terrain, rffs)
            m = fa != fold_k; args = (Xp_a[m], Xd_a[m], Xe_a[m], self.ya[m], self.ga[m], fa[m])
        else:
            m = tr; args = (Xp_e[m], Xd_e[m], Xe_e[m], self.y[m], self.g[m], f[m])
        ps = []
        for s in range(n_ens):
            _, pred = H.fit_heron(*args, seed=seed * 101 + s, **kw)
            ps.append(pred(Xp_e, Xd_e, Xe_e))
        p = np.mean(ps, 0)
        if cfg.get("calibrate", True):
            idx = np.where(tr)[0]; rs = np.random.default_rng(seed); rs.shuffle(idx)
            ci = idx[int(0.85 * len(idx)):]          # held-out slice of the training region
            p = platt_apply(p, platt_fit(p[ci], self.y[ci]))
        return p

    def cv(self, methods, protocol, reps, n_folds, tag, block_deg=0.08):
        rows = []
        for rep in range(reps):
            if protocol == "spatial":
                f  = dp.spatial_folds(self.d, n_folds, block_deg, seed=rep)[0]
                fa = dp.spatial_folds(self.a, n_folds, block_deg, seed=rep)[0]
            else:
                f  = dp.random_folds(self.d, n_folds, seed=rep)
                fa = dp.random_folds(self.a, n_folds, seed=rep)
            for k in range(n_folds):
                tr, te = f != k, f == k
                if self.y[te].sum() < 5: continue
                for name, spec in methods.items():
                    t0 = time.time()
                    try:
                        if spec[0] == "baseline":
                            p, _ = B.BASELINES[spec[1]](self.d[tr], self.d[te], seed=rep)
                        elif spec[0] == "stack":
                            p = self.stack(tr, k, f, fa, spec[1], seed=rep)[te]
                        else:
                            p = self.heron(tr, k, f, fa, spec[1], seed=rep)[te]
                    except Exception as e:
                        print(f"  !! {name} rep{rep} fold{k}: {type(e).__name__}: {e}", flush=True); continue
                    rows.append(dict(protocol=protocol, index=self.index, rep=rep, fold=k,
                                     method=name, n_test=int(te.sum()), pos_test=int(self.y[te].sum()),
                                     block_deg=block_deg, secs=round(time.time() - t0, 2),
                                     **all_metrics(self.y[te], p, base_rate=self.y[tr].mean())))
                print(f"[{tag}] rep{rep} fold{k}: {len(rows)} rows", flush=True)
                # checkpoint after every fold so a long block can be interrupted or assembled early
                os.makedirs(OUT, exist_ok=True)
                pd.DataFrame(rows).to_csv(f"{OUT}/{tag}.csv", index=False)
        R = pd.DataFrame(rows); os.makedirs(OUT, exist_ok=True)
        R.to_csv(f"{OUT}/{tag}.csv", index=False); return R

    def temporal(self, methods, seeds=5, tag="cv_temporal"):
        """Forecasting test: fit on Pre-Maria + Inter-Hurricanes, predict the Post-Fiona period."""
        rows = []
        tr = (self.d.Period != "Post-Fiona").values; te = ~tr
        tra = (self.a.Period != "Post-Fiona").values
        f  = np.where(tr, 0, 1); fa = np.where(tra, 0, 1)
        for s in range(seeds):
            for name, spec in methods.items():
                t0 = time.time()
                try:
                    if spec[0] == "baseline":
                        p, _ = B.BASELINES[spec[1]](self.d[tr], self.d[te], seed=s)
                    elif spec[0] == "stack":
                        p = self.stack(tr, 1, f, fa, spec[1], seed=s)[te]
                    else:
                        p = self.heron(tr, 1, f, fa, spec[1], seed=s)[te]
                except Exception as e:
                    print(f"  !! {name} seed{s}: {type(e).__name__}: {e}", flush=True); continue
                rows.append(dict(protocol="temporal", index=self.index, rep=s, fold=0, method=name,
                                 n_test=int(te.sum()), pos_test=int(self.y[te].sum()),
                                 block_deg=np.nan, secs=round(time.time() - t0, 2),
                                 **all_metrics(self.y[te], p, base_rate=self.y[tr].mean())))
            print(f"[{tag}] seed{s}: {len(rows)} rows", flush=True)
            os.makedirs(OUT, exist_ok=True)
            pd.DataFrame(rows).to_csv(f"{OUT}/{tag}.csv", index=False)
        R = pd.DataFrame(rows); R.to_csv(f"{OUT}/{tag}.csv", index=False); return R

MAIN = {n: ("baseline", n) for n in
        ["GLM-m4 (published)", "GLM-m3 (published)", "GLM-splines", "GLM-m4 + terrain",
         "RandomForest", "LightGBM", "XGBoost", "MLP",
         "LightGBM + terrain", "XGBoost + terrain", "MLP + terrain"]}
# The head-to-head test (Section "headtohead") showed the neural branch is better WITH the
# spatial Fourier block, WITH terrain and WITHOUT the inner-validation holdout, so HERON-plus is
# the configuration we report as the method. HERON (the configuration originally chosen from
# four-fold diagnostics) is kept in the table to show what those diagnostics cost.
MAIN["HERON"] = ("stack", STACK)
MAIN["HERON-NN"] = ("heron", dict(NN, terrain=False, rff=False, use_all=True, n_ens=3, calibrate=True))
MAIN["HERON-plus"] = ("stack", dict(STACK, heron_kw=dict(NN, val_frac=0.0),
                                    nn_terrain=True, nn_rff=True))

NNF = dict(NN, terrain=False, rff=False, use_all=True, n_ens=3, calibrate=True)
# ablation rows all use a single member so that each row differs from "neural branch
# alone" in exactly one respect, and so the grid is tractable
NN1 = dict(NNF, n_ens=1)
# The member-dropping ablation is run with a reduced-cost stack (two inner folds, one neural
# member) so the grid is tractable. The "HERON (full)" row below uses the SAME reduced setting,
# so every row of this table is internally comparable; the headline benchmark uses STACK.
STACK_ABL = dict(STACK, n_inner=2, nn_ens_inner=1, nn_ens_full=1)
ABLATE_STACK = {
 "HERON (full)":                  ("stack", dict(STACK_ABL)),
 "- tree members":                ("stack", dict(STACK_ABL, use_trees=False)),
 "- neural member":               ("stack", dict(STACK_ABL, use_nn=False)),
 "- GLM member":                  ("stack", dict(STACK_ABL, use_glm=False)),
 "- all-data training":           ("stack", dict(STACK_ABL, use_all=False)),
}
ABLATE_NN = {
 "neural branch alone":           ("heron", dict(NN1)),
 "  + 3-member deep ensemble":    ("heron", dict(NN1, n_ens=3)),
 "  - Platt calibration":         ("heron", dict(NN1, calibrate=False)),
 "  - all-data training":         ("heron", dict(NN1, use_all=False)),
 "  + terrain in the network":    ("heron", dict(NN1, terrain=True)),
 "  + spatial Fourier features":  ("heron", dict(NN1, rff=True)),
 "  - effort/occupancy factorise":("heron", dict(NN1, factorised=False)),
 "  - monotone effort":           ("heron", dict(NN1, monotone=False)),
 "  - grouped occupancy loss":    ("heron", dict(NN1, lam_group=0.0)),
 "  grouped occ. loss x5":        ("heron", dict(NN1, lam_group=0.5)),
 "  - spatial V-REx penalty":     ("heron", dict(NN1, lam_vrex=0.0)),
 "  + focal loss (g=1, w=3)":     ("heron", dict(NN1, gamma=1.0, pos_weight=3.0)),
 "  - inner spatial early stop":  ("heron", dict(NN1, val_frac=0.0)),
}
ABLATE = {**ABLATE_STACK, **ABLATE_NN}

# Head-to-head: the ablation says the neural branch is better WITH the spatial Fourier block,
# with terrain, and without the inner-validation holdout. HERON-plus puts all three back so the
# improvement to the neural branch can be tested inside the stack rather than in isolation.
NN_PLUS = dict(NN, val_frac=0.0)
STACK_PLUS = dict(STACK, heron_kw=NN_PLUS)
HEAD2HEAD = {
 "HERON":      ("stack", dict(STACK)),
 "HERON-plus": ("stack", dict(STACK_PLUS, nn_terrain=True, nn_rff=True)),
 "GLM-m4 (published)": ("baseline", "GLM-m4 (published)"),
}
# the neural-branch grid is run separately from the (much slower) stack grid
ABLATE_NN_ONLY = ABLATE_NN

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("block")
    ap.add_argument("--only", default="", help="comma-separated method-name substrings to keep")
    ap.add_argument("--drop", default="", help="comma-separated method-name substrings to skip")
    ap.add_argument("--reps", type=int, default=0)
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()

    def filt(m):
        out = dict(m)
        if a.only:
            keep = [t.strip() for t in a.only.split(",")]
            out = {k: v for k, v in out.items() if any(t in k for t in keep)}
        if a.drop:
            skip = [t.strip() for t in a.drop.split(",")]
            out = {k: v for k, v in out.items() if not any(t in k for t in skip)}
        return out
    MAIN_F, ABL_F = filt(MAIN), filt(ABLATE)
    SFX = a.suffix
    if a.block == "main_spatial":
        Bench("NDVI").cv(MAIN_F, "spatial", reps=a.reps or 2, n_folds=10, tag="cv_main_spatial_NDVI"+SFX)
    elif a.block == "main_random":
        Bench("NDVI").cv(MAIN_F, "random", reps=a.reps or 2, n_folds=5, tag="cv_main_random_NDVI"+SFX)
    elif a.block == "main_evi2":
        Bench("EVI2").cv(MAIN_F, "spatial", reps=a.reps or 1, n_folds=6, tag="cv_main_spatial_EVI2"+SFX)
    elif a.block == "temporal":
        Bench("NDVI").temporal(MAIN_F, seeds=a.reps or 3, tag="cv_temporal"+SFX)
    elif a.block == "ablation":
        Bench("NDVI").cv(ABL_F, "spatial", reps=a.reps or 2, n_folds=10, tag="cv_ablation_spatial_NDVI"+SFX)
    elif a.block == "ablation_nn":
        Bench("NDVI").cv(filt(ABLATE_NN_ONLY), "spatial", reps=a.reps or 2, n_folds=10,
                         tag="cv_ablation_nn_spatial_NDVI"+SFX)
    elif a.block == "ablation_stack":
        Bench("NDVI").cv(filt(ABLATE_STACK), "spatial", reps=a.reps or 1, n_folds=6,
                         tag="cv_ablation_stack_spatial_NDVI"+SFX)
    elif a.block == "headtohead":
        Bench("NDVI").cv(filt(HEAD2HEAD), "spatial", reps=a.reps or 1, n_folds=10,
                         tag="cv_headtohead_spatial_NDVI"+SFX)
    elif a.block == "blocksize":
        b = Bench("NDVI"); out = []
        # the sweep only needs to show how the ordering of methods responds to block size,
        # so it uses the reduced-cost stack and six folds per size
        meths = filt({"HERON": ("stack", dict(STACK, n_inner=2, nn_ens_inner=1, nn_ens_full=1)),
                      "GLM-m4 (published)": ("baseline", "GLM-m4 (published)"),
                      "LightGBM + terrain": ("baseline", "LightGBM + terrain")})
        for bd in (0.04, 0.08, 0.16, 0.32):
            out.append(b.cv(meths, "spatial", reps=a.reps or 1, n_folds=6,
                            tag=f"_bs_{bd}{SFX}", block_deg=bd))
        pd.concat(out).to_csv(f"{OUT}/cv_blocksize_NDVI{SFX}.csv", index=False)
    else:
        raise SystemExit(f"unknown block {a.block}")
