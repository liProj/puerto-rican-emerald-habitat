"""Cross-validated benchmark driver: protocols x datasets x methods, 16 metrics per fold."""
import numpy as np, pandas as pd, sys, os, time, json
sys.path.insert(0, "Birds/code")
import data_prep as dp, heron as H, baselines as B
from metrics import all_metrics
from sklearn.isotonic import IsotonicRegression

def prep(index="NDVI", subset="complete", rff_seed=0, rff_scales=(0.02,0.05,0.12,0.35,1.0), n_per=32):
    d = dp.build(index, subset)
    d["Elevation_100m"] = d.Elevation_m / 100.0
    r = dp.RFF(rff_scales, n_per, rff_seed)
    return d, r(d)

def group_ids(d):
    return pd.factorize(d.site.astype(str) + "|" + d.Period.astype(str))[0]

def folds_for(d, protocol, rep, n_folds):
    if protocol == "spatial": return dp.spatial_folds(d, n_folds, 0.08, seed=rep)[0]
    if protocol == "random":  return dp.random_folds(d, n_folds, seed=rep)
    raise ValueError(protocol)

def heron_predict(d, RFF, tr, te, cfg, seed, d_all=None, RFF_all=None, tr_all=None, calibrate=True):
    """Fit HERON on the training split and predict the test split.
    If d_all is given, the model trains on the FULL (missingness-included) table restricted to the
    same spatial/temporal training region -- the 23.1% of checklists the published GLM must drop."""
    use_rff = cfg.get("rff", True)
    src, R, msk = (d_all, RFF_all, tr_all) if d_all is not None else (d, RFF, tr)
    Xp_tr = dp.psi_features(src, R if use_rff else None)[msk]
    Xd_tr, Xe_tr = dp.det_features(src)[msk], dp.effort_raw(src)[msk]
    y_tr, g_tr = src.y.values[msk], group_ids(src)[msk]
    blk_tr = cfg["_blk_all"][msk] if d_all is not None else cfg["_blk"][tr]
    kw = {k: v for k, v in cfg.items() if not k.startswith("_") and k != "rff" and k != "n_ens"}
    Xp_ev = dp.psi_features(d, RFF if use_rff else None)
    Xd_ev, Xe_ev = dp.det_features(d), dp.effort_raw(d)
    ps = []
    for s in range(cfg.get("n_ens", 1)):
        _, pred = H.fit_heron(Xp_tr, Xd_tr, Xe_tr, y_tr, g_tr, blk_tr, seed=seed * 97 + s, **kw)
        ps.append(pred(Xp_ev, Xd_ev, Xe_ev))
    p = np.mean(ps, 0)
    if calibrate:                       # isotonic fit on a held-out slice of the training region
        idx = np.where(tr)[0]
        rs = np.random.default_rng(seed); rs.shuffle(idx)
        cut = int(0.85 * len(idx)); ci = idx[cut:]
        iso = IsotonicRegression(out_of_bounds="clip").fit(p[ci], d.y.values[ci])
        p = np.clip(iso.predict(p), 1e-7, 1 - 1e-7)
    return p[te]

def run(methods, protocol="spatial", index="NDVI", subset="complete", reps=3, n_folds=10,
        out=None, use_all_for_heron=True, verbose=True):
    d, RFF = prep(index, subset)
    d_all, RFF_all = (prep(index, "all") if use_all_for_heron else (None, None))
    rows = []
    for rep in range(reps):
        f = folds_for(d, protocol, rep, n_folds)
        if d_all is not None:
            # map every full-table row to the fold of its geographic block / random partner
            f_all = (dp.spatial_folds(d_all, n_folds, 0.08, seed=rep)[0] if protocol == "spatial"
                     else dp.random_folds(d_all, n_folds, seed=rep))
        for k in range(n_folds):
            tr, te = f != k, f == k
            if d.y.values[te].sum() < 5 or d.y.values[tr].sum() < 20: continue
            for name, spec in methods.items():
                t0 = time.time()
                try:
                    if spec[0] == "baseline":
                        p, _ = B.BASELINES[name if name in B.BASELINES else spec[1]](d[tr], d[te], seed=rep)
                    else:
                        cfg = dict(spec[1]); cfg["_blk"] = f
                        ua = cfg.pop("_use_all", False) and d_all is not None
                        if ua: cfg["_blk_all"] = f_all
                        p = heron_predict(d, RFF, tr, te, cfg, seed=rep,
                                          d_all=d_all if ua else None,
                                          RFF_all=RFF_all if ua else None,
                                          tr_all=(f_all != k) if ua else None,
                                          calibrate=cfg.pop("_calibrate", True))
                except Exception as e:
                    print(f"  !! {name} rep{rep} fold{k}: {type(e).__name__}: {e}"); continue
                m = all_metrics(d.y.values[te], p, base_rate=d.y.values[tr].mean())
                rows.append(dict(protocol=protocol, index=index, subset=subset, rep=rep, fold=k,
                                 method=name, n_test=int(te.sum()), pos_test=int(d.y.values[te].sum()),
                                 secs=round(time.time() - t0, 2), **m))
            if verbose:
                cur = pd.DataFrame(rows)
                print(f"[{protocol}/{index}] rep{rep} fold{k} done "
                      f"({len(cur)} rows)", flush=True)
    R = pd.DataFrame(rows)
    if out: os.makedirs(os.path.dirname(out), exist_ok=True); R.to_csv(out, index=False)
    return R
