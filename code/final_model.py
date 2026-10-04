"""Fit the final HERON on the whole dataset and export the ecological read-outs:
the learned detection-effort curve, psi partial dependences, the island-wide psi map,
elevational centroids, and permutation importance. These are the analogues of the
published Figures 3-6, produced by the new model."""
import sys, os; sys.path.insert(0, "Birds/code")
import numpy as np, pandas as pd, torch
import data_prep as dp, heron as H, terrain as TR
from common import PERIODS, PER_EN
from calib import platt_fit, platt_apply
from sklearn.metrics import average_precision_score, roc_auc_score

OUT = "Birds/results/final_model"; os.makedirs(OUT, exist_ok=True)
N_ENS = 3

def main():
    d  = dp.build("NDVI", "complete")
    da = dp.build("NDVI", "all")
    y, ya = d.y.values, da.y.values
    g  = pd.factorize(d.site.astype(str) + "|" + d.Period.astype(str))[0]
    ga = pd.factorize(da.site.astype(str) + "|" + da.Period.astype(str))[0]
    blk_a = dp.spatial_folds(da, 10, 0.08, seed=0)[0]      # blocks only drive V-REx here
    PF = lambda t: dp.psi_features(t, None, False)   # terrain excluded: matches the
    Xp_a, Xd_a, Xe_a = PF(da), dp.det_features(da), dp.effort_raw(da)  # deployed branch
    Xp_e, Xd_e, Xe_e = PF(d),  dp.det_features(d),  dp.effort_raw(d)

    models, preds = [], []
    for s in range(N_ENS):
        m, pred = H.fit_heron(Xp_a, Xd_a, Xe_a, ya, ga, blk_a, seed=s,
                              epochs=300, batch=32768, gamma=0.0, pos_weight=1.0,
                              lam_group=0.1, lam_vrex=1.0)
        models.append((m, pred)); preds.append(pred(Xp_e, Xd_e, Xe_e))
        print(f"  ensemble member {s+1}/{N_ENS} fitted", flush=True)
    p_in = np.mean(preds, 0)
    print(f"in-sample AUC={roc_auc_score(y, p_in):.4f} AP={average_precision_score(y, p_in):.4f}")

    def ens_parts(Xp, Xd, Xe):
        P, PS, PD = [], [], []
        for _, pred in models:
            a, b, c = pred(Xp, Xd, Xe, parts=True); P.append(a); PS.append(b); PD.append(c)
        return np.mean(P, 0), np.mean(PS, 0), np.mean(PD, 0)

    # ---- 1. learned detection-effort curve (the published model has no analogue) ----
    rows = []
    for var, grid in (("Duration_min", np.linspace(5, 240, 120)),
                      ("Distance_km", np.linspace(0, 5, 120))):
        base = d.iloc[[0]].copy()
        rep = pd.concat([base] * len(grid), ignore_index=True)
        for c in ("Duration_min", "Distance_km"):
            rep[c] = d[c].median()
        rep[var] = grid
        for c in ("per_0", "per_1", "per_2"): rep[c] = d[c].mean()
        rep["doy_sin"], rep["doy_cos"] = d.doy_sin.mean(), d.doy_cos.mean()
        rep["days_veg"] = 0.0
        _, _, pd_det = ens_parts(PF(rep), dp.det_features(rep), dp.effort_raw(rep))
        rows += [dict(variable=var, value=v, p_det=q) for v, q in zip(grid, pd_det)]
    pd.DataFrame(rows).to_csv(f"{OUT}/detection_effort_curve.csv", index=False)

    # ---- 2. psi partial dependence over elevation x period, and over NDVI ----
    med = {c: d[c].median() for c in TR.FEATS}
    def mk(n, **over):
        r = pd.DataFrame({c: np.full(n, med[c]) for c in TR.FEATS})
        r["Elevation_m"] = d.Elevation_m.median(); r["veg_f"] = d.veg_f.median()
        r["veg_missing"] = 0.0
        r["Duration_min"] = d.Duration_min.median(); r["Distance_km"] = d.Distance_km.median()
        r["doy_sin"], r["doy_cos"], r["days_veg"] = d.doy_sin.mean(), d.doy_cos.mean(), 0.0
        for i in range(3): r[f"per_{i}"] = 0.0
        for k, v in over.items(): r[k] = v
        return r
    elev = np.arange(0, 1301, 10); rows = []
    for i, per in enumerate(PERIODS):
        r = mk(len(elev), Elevation_m=elev)
        r[f"per_{i}"] = 1.0
        # terrain co-varies with elevation in reality; use the local median terrain at each height
        for c in TR.FEATS:
            bins = np.clip(np.digitize(d.Elevation_m.values, elev) - 1, 0, len(elev) - 1)
            mvals = pd.Series(d[c].values).groupby(bins).median()
            r[c] = pd.Series(np.arange(len(elev))).map(mvals).ffill().bfill().values
        pr, psi, pdet = ens_parts(PF(r), dp.det_features(r), dp.effort_raw(r))
        rows += [dict(period=PER_EN[per], Elevation_m=e, p_report=a, psi=b, p_det=c)
                 for e, a, b, c in zip(elev, pr, psi, pdet)]
    PDp = pd.DataFrame(rows); PDp.to_csv(f"{OUT}/pd_elevation_by_period.csv", index=False)

    ndvi = np.linspace(d.veg_f.quantile(.01), d.veg_f.quantile(.99), 120); rows = []
    for i, per in enumerate(PERIODS):
        r = mk(len(ndvi), veg_f=ndvi); r[f"per_{i}"] = 1.0
        pr, psi, pdet = ens_parts(PF(r), dp.det_features(r), dp.effort_raw(r))
        rows += [dict(period=PER_EN[per], NDVI=v, p_report=a, psi=b) for v, a, b in zip(ndvi, pr, psi)]
    pd.DataFrame(rows).to_csv(f"{OUT}/pd_ndvi_by_period.csv", index=False)

    # ---- 3. elevational centroid, the published Figure 4 statistic, from HERON ----
    cen = []
    for per in PDp.period.unique():
        s = PDp[PDp.period == per]
        cen.append(dict(period=per,
                        centroid_report=float((s.Elevation_m * s.p_report).sum() / s.p_report.sum()),
                        centroid_psi=float((s.Elevation_m * s.psi).sum() / s.psi.sum())))
    C = pd.DataFrame(cen)
    for c in ("centroid_report", "centroid_psi"):
        C[f"delta_{c}"] = C[c] - C.loc[C.period == "Pre-Maria", c].iloc[0]
    C.to_csv(f"{OUT}/elevational_centroids_heron.csv", index=False)
    print(C.round(2).to_string(index=False))

    # ---- 4. island-wide psi surface on the occupancy grid ----
    sites = d.groupby("site").agg(Longitude=("Longitude", "mean"), Latitude=("Latitude", "mean"),
                                  Elevation_m=("Elevation_m", "mean"), veg_f=("veg_f", "mean"),
                                  n=("y", "size")).reset_index()
    T = TR.build(sites.Longitude.values, sites.Latitude.values)
    r = sites.join(T); r["veg_missing"] = 0.0
    r["Duration_min"] = d.Duration_min.median(); r["Distance_km"] = d.Distance_km.median()
    r["doy_sin"], r["doy_cos"], r["days_veg"] = d.doy_sin.mean(), d.doy_cos.mean(), 0.0
    out = []
    for i, per in enumerate(PERIODS):
        rr = r.copy()
        for j in range(3): rr[f"per_{j}"] = float(i == j)
        pr, psi, pdet = ens_parts(PF(rr), dp.det_features(rr), dp.effort_raw(rr))
        out.append(rr.assign(period=PER_EN[per], p_report=pr, psi=psi, p_det=pdet))
    pd.concat(out).to_csv(f"{OUT}/site_psi_surface.csv", index=False)

    # ---- 5. permutation importance on a held-out spatial fold ----
    f = dp.spatial_folds(d, 10, 0.08, seed=0)[0]
    te = f == 0
    names = ["Elevation_m", "veg_f", "veg_missing", "per_0", "per_1", "per_2"]
    Xp0 = PF(d)[te]; Xd0 = Xd_e[te]; Xe0 = Xe_e[te]
    base = average_precision_score(y[te], ens_parts(Xp0, Xd0, Xe0)[0])
    rng = np.random.default_rng(0); imp = []
    for j, nm in enumerate(names):
        drops = []
        for _ in range(5):
            Z = Xp0.copy(); Z[:, j] = rng.permutation(Z[:, j])
            drops.append(base - average_precision_score(y[te], ens_parts(Z, Xd0, Xe0)[0]))
        imp.append(dict(feature=nm, delta_AP=float(np.mean(drops)), sd=float(np.std(drops))))
    I = pd.DataFrame(imp).sort_values("delta_AP", ascending=False)
    I.to_csv(f"{OUT}/permutation_importance_nn.csv", index=False)
    print(I.round(4).to_string(index=False))

    # ---- 6. permutation importance for the terrain-carrying tree member ----
    import baselines as B
    ptree, mdl = B.BASELINES["LightGBM + terrain"](d[~te], d[te], seed=0)
    Xt = B.tab_features(d[te], terrain=True)
    tnames = ["Elevation_m", "veg_f", "veg_missing", "Duration_min", "Distance_km",
              "per_0", "per_1", "per_2", "doy_sin", "doy_cos", "Longitude", "Latitude"] + list(TR.FEATS)
    base_t = average_precision_score(y[te], ptree)
    rows = []
    for j, nm in enumerate(tnames):
        drops = []
        for _ in range(5):
            Z = Xt.copy(); Z[:, j] = rng.permutation(Z[:, j])
            drops.append(base_t - average_precision_score(y[te], mdl.predict_proba(Z)[:, 1]))
        rows.append(dict(feature=nm, delta_AP=float(np.mean(drops)), sd=float(np.std(drops))))
    T2 = pd.DataFrame(rows).sort_values("delta_AP", ascending=False)
    T2.to_csv(f"{OUT}/permutation_importance_tree.csv", index=False)
    print("\nLightGBM + terrain member:"); print(T2.head(14).round(4).to_string(index=False))
    print(f"\nartifacts -> {OUT}")

if __name__ == "__main__":
    main()
