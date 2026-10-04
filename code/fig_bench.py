"""Figures 5-10, 17, 18: benchmark, protocols, paired folds, curves, calibration, ablation."""
import sys, os; sys.path.insert(0, "Birds/code")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import plotstyle as ps
from metrics import HIGHER_BETTER
import stats_tests as st
ps.setup(); R = "Birds/results"
ORDER = ["GLM-m3 (published)", "GLM-m4 (published)", "GLM-splines", "GLM-m4 + terrain",
         "RandomForest", "MLP", "MLP + terrain", "LightGBM", "LightGBM + terrain",
         "XGBoost", "XGBoost + terrain", "HERON-NN", "HERON", "HERON-plus"]

def common_folds(d, ref="GLM-m4 (published)", new="HERON-plus"):
    """Restrict to folds that both the reference and the new method completed, so every bar in a
    panel is an average over the same set of held-out regions."""
    if new not in set(d.method) or ref not in set(d.method): return d
    keep = (set(map(tuple, d[d.method == new][["rep", "fold"]].values))
            & set(map(tuple, d[d.method == ref][["rep", "fold"]].values)))
    return d[[(r, f) in keep for r, f in zip(d.rep, d.fold)]]
def _load(n):
    p = f"{R}/{n}.csv"
    return pd.read_csv(p) if os.path.exists(p) else None

def fig05():
    d = _load("cv_main_spatial_NDVI")
    if d is None: return print("  (skip fig05: no spatial results yet)")
    d = common_folds(d)
    nf = d[["rep", "fold"]].drop_duplicates().shape[0]
    ms = [m for m in ORDER if m in set(d.method)]
    keys = [("AUC", "AUC-ROC"), ("AP", "Average precision"), ("pseudoR2", "McFadden pseudo-$R^2$"),
            ("Brier", "Brier score (lower better)"), ("LogLoss", "Log loss (lower better)"),
            ("Recall_at_10pct", "Recall @ top 10% flagged")]
    fig, ax = plt.subplots(2, 3, figsize=(10.4, 5.6), layout="constrained")
    for a, (k, lab) in zip(ax.ravel(), keys):
        g = d.groupby("method")[k].agg(["mean", "std", "count"]).loc[ms]
        se = g["std"] / np.sqrt(g["count"])
        cols = [ps.col(m) for m in ms]
        a.barh(range(len(ms)), g["mean"], xerr=se, color=cols, height=.72,
               error_kw=dict(lw=.9, capsize=2, ecolor="0.3"))
        a.set_yticks(range(len(ms)))
        a.set_yticklabels([m.replace(" (published)", "*") for m in ms], fontsize=7)
        a.invert_yaxis(); a.set_xlabel(lab, fontsize=8)
        ref = g["mean"].get("GLM-m4 (published)", np.nan)
        if np.isfinite(ref): a.axvline(ref, ls=":", color="#B03A2E", lw=1)
        lo, hi = float(g["mean"].min()), float(g["mean"].max()); pad = .18 * (hi - lo + 1e-9)
        a.set_xlim(lo - pad * 2.2, hi + pad)        # no clipping at 0: pseudo-R2 can be negative
        if lo < 0: a.axvline(0, color="0.55", lw=.8)
    fig.suptitle(f"Spatially blocked cross-validation, NDVI subset ({nf} folds completed by every "
                 "method shown). Dotted line = the published GLM; bars are mean ± SE across folds.",
                 fontsize=8.6)
    ps.save(fig, "fig05_main_benchmark_spatial")

def fig06():
    frames = {"Spatial blocks": _load("cv_main_spatial_NDVI"),
              "Random folds": _load("cv_main_random_NDVI"),
              "Temporal (forecast Post-Fiona)": _load("cv_temporal")}
    frames = {k: v for k, v in frames.items() if v is not None}
    if not frames: return print("  (skip fig06)")
    sel = ["GLM-m4 (published)", "GLM-splines", "RandomForest", "XGBoost + terrain",
           "HERON", "HERON-plus"]
    fig, ax = plt.subplots(1, len(frames), figsize=(3.4 * len(frames), 3.2), layout="constrained",
                           squeeze=False)
    for a, (nm, d) in zip(ax[0], frames.items()):
        ms = [m for m in sel if m in set(d.method)]
        x = np.arange(len(ms))
        for i, (k, o) in enumerate([("AUC", -.2), ("AP", .2)]):
            g = d.groupby("method")[k].agg(["mean", "std", "count"]).loc[ms]
            a.bar(x + o, g["mean"], .38, yerr=g["std"] / np.sqrt(g["count"]),
                  color=["#2E86C1", "#0B6E4F"][i], label=k, error_kw=dict(lw=.8, capsize=2))
        a.set_xticks(x); a.set_xticklabels([m.replace(" (published)", "*").replace(" + ", "\n+ ")
                                            for m in ms], fontsize=7, rotation=30, ha="right")
        a.set_title(nm, fontsize=9); a.set_ylabel("score"); a.legend(fontsize=7)
    fig.suptitle("Generalisation under three evaluation protocols", fontsize=9)
    ps.save(fig, "fig06_protocol_comparison")

def fig07():
    d = _load("cv_main_spatial_NDVI")
    if d is None: return print("  (skip fig07)")
    d = common_folds(d); ref = "GLM-m4 (published)"
    fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.2), layout="constrained")
    for a, k in zip(ax, ["AUC", "AP", "pseudoR2"]):
        A = d[d.method == ref].set_index(["rep", "fold"])[k]
        B = d[d.method == "HERON-plus"].set_index(["rep", "fold"])[k]
        ix = A.index.intersection(B.index); A, B = A.loc[ix], B.loc[ix]
        lim = [min(A.min(), B.min()) * .97, max(A.max(), B.max()) * 1.03]
        a.plot(lim, lim, ls="--", color="0.6", lw=.9)
        a.scatter(A, B, s=26, c=np.where(B > A, "#0B6E4F", "#B03A2E"), alpha=.8, lw=0)
        w = int((B > A).sum())
        a.set_xlabel(f"{ref}"); a.set_ylabel("HERON-plus")
        a.set_title(f"{k}: HERON-plus wins {w}/{len(A)} folds", fontsize=9)
        a.set_xlim(lim); a.set_ylim(lim); a.set_aspect("equal")
    fig.suptitle("Paired per-fold comparison under spatial blocking (points above the line favour HERON-plus)",
                 fontsize=8.8)
    ps.save(fig, "fig07_paired_folds")

def fig10():
    d = _load("cv_ablation_nn_spatial_NDVI")
    if d is None: return print("  (skip fig10)")
    full = "neural branch alone"
    rows = []
    for m in d.method.unique():
        if m == full: continue
        for k in ("AUC", "AP", "pseudoR2"):
            A = d[d.method == full].set_index(["rep", "fold"])[k]
            B = d[d.method == m].set_index(["rep", "fold"])[k]
            ix = A.index.intersection(B.index)
            rows.append(dict(variant=m, metric=k, delta=(B.loc[ix] - A.loc[ix]).mean(),
                             se=(B.loc[ix] - A.loc[ix]).std(ddof=1) / np.sqrt(len(ix))))
    t = pd.DataFrame(rows)
    order = t[t.metric == "AP"].sort_values("delta").variant.tolist()
    fig, ax = plt.subplots(1, 3, figsize=(10.2, 4.0), layout="constrained", sharey=True)
    for a, k in zip(ax, ["AUC", "AP", "pseudoR2"]):
        g = t[t.metric == k].set_index("variant").loc[order]
        a.barh(range(len(order)), g.delta, xerr=g.se, height=.7,
               color=np.where(g.delta < 0, "#0B6E4F", "#B03A2E"),
               error_kw=dict(lw=.8, capsize=2, ecolor="0.3"))
        a.axvline(0, color="0.3", lw=1)
        a.set_xlabel(f"$\\Delta$ {k} vs the reference row", fontsize=8)
        a.set_title(k, fontsize=9)
    ax[0].set_yticks(range(len(order))); ax[0].set_yticklabels(order, fontsize=7.5)
    ax[0].invert_yaxis()
    fig.suptitle("Neural-branch ablation, spatially blocked folds. Each row differs from "
                 "\u201cneural branch alone\u201d in exactly one respect.\n"
                 "Green = lower score after the change; red = higher score after the change.",
                 fontsize=8.4)
    ps.save(fig, "fig10_ablation")

def fig17():
    d = _load("cv_blocksize_NDVI")
    if d is None: return print("  (skip fig17)")
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0), layout="constrained")
    for a, k in zip(ax, ["AUC", "AP"]):
        for m in d.method.unique():
            g = d[d.method == m].groupby("block_deg")[k].agg(["mean", "std", "count"])
            a.errorbar(g.index * 111, g["mean"], yerr=g["std"] / np.sqrt(g["count"]),
                       fmt="o-", ms=4, lw=1.3, color=ps.col(m), label=m.replace(" (published)", "*"),
                       capsize=2)
        a.set_xscale("log"); a.set_xlabel("spatial block side (km)"); a.set_ylabel(k)
        a.set_title(f"{k} vs spatial-block size", fontsize=9)
    ax[0].legend(fontsize=7)
    fig.suptitle("Sensitivity to geographic block width", fontsize=9)
    ps.save(fig, "fig17_blocksize_sensitivity")

def fig18():
    d = _load("cv_main_spatial_NDVI")
    if d is None: return print("  (skip fig18)")
    d = common_folds(d); ref = "GLM-m4 (published)"
    ms = [m for m in ORDER if m in set(d.method) and m != ref]
    mets = list(HIGHER_BETTER)
    Z = np.full((len(ms), len(mets)), np.nan); S = np.empty((len(ms), len(mets)), object)
    for i, m in enumerate(ms):
        c = st.compare(d, ref, m, metrics=mets, group_cols=("protocol",))
        for j, k in enumerate(mets):
            r = c[c.metric == k]
            if not len(r): continue
            r = r.iloc[0]
            sd = r.delta_sd if r.delta_sd > 0 else np.nan
            Z[i, j] = r.delta / sd if np.isfinite(sd) else 0.0
            S[i, j] = ("+" if r.better else "-")
    fig, a = plt.subplots(figsize=(11.0, 4.6), layout="constrained")
    v = np.nanpercentile(np.abs(Z), 96)
    im = a.imshow(Z, cmap="RdYlGn", vmin=-v, vmax=v, aspect="auto")
    a.set_xticks(range(len(mets))); a.set_xticklabels(mets, rotation=45, ha="right", fontsize=7.5)
    a.set_yticks(range(len(ms))); a.set_yticklabels([m.replace(" (published)", "*") for m in ms], fontsize=7.5)
    for i in range(len(ms)):
        for j in range(len(mets)):
            if S[i, j]: a.text(j, i, S[i, j], ha="center", va="center", fontsize=6.5, color="0.15")
    a.grid(False)
    fig.colorbar(im, ax=a, fraction=.02, pad=.01, label="standardised gain vs published GLM")
    a.set_title("Per-metric comparison against the published GLM across 16 metrics\n"
                "(+ = better, - = worse; color = mean paired gain / SD of paired gains)", fontsize=9)
    ps.save(fig, "fig18_metric_winloss_heatmap")


def fig08():
    p = f"{R}/oof_spatial_NDVI.csv"
    if not os.path.exists(p): return print("  (skip fig08)")
    from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
    d = pd.read_csv(p)
    ms = [c for c in d.columns if c not in ("y", "fold") and d[c].notna().any()]
    y = d.y.values
    fig, ax = plt.subplots(1, 2, figsize=(7.8, 3.4), layout="constrained")
    for m in ms:
        s = d[m].values; ok = np.isfinite(s)
        fpr, tpr, _ = roc_curve(y[ok], s[ok])
        ax[0].plot(fpr, tpr, lw=1.5, color=ps.col(m),
                   label=f"{m.replace(' (published)','*')} ({roc_auc_score(y[ok],s[ok]):.3f})")
        pr, rc, _ = precision_recall_curve(y[ok], s[ok])
        ax[1].plot(rc, pr, lw=1.5, color=ps.col(m),
                   label=f"{m.replace(' (published)','*')} ({average_precision_score(y[ok],s[ok]):.3f})")
    ax[0].plot([0, 1], [0, 1], ls="--", color="0.6", lw=.9)
    ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate")
    ax[0].set_title("(a) ROC (AUC)", fontsize=9); ax[0].legend(fontsize=6.8, loc="lower right")
    ax[1].axhline(y.mean(), ls="--", color="0.6", lw=.9)
    ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision")
    ax[1].set_title("(b) Precision–recall (average precision)", fontsize=9)
    ax[1].legend(fontsize=6.8); ax[1].set_ylim(0, None)
    fig.suptitle("Pooled out-of-fold predictions, spatially blocked cross-validation", fontsize=8.8)
    ps.save(fig, "fig08_roc_pr_curves")

def fig09():
    p = f"{R}/oof_spatial_NDVI.csv"
    if not os.path.exists(p): return print("  (skip fig09)")
    from metrics import ece
    d = pd.read_csv(p); y = d.y.values
    ms = [c for c in d.columns if c not in ("y", "fold") and d[c].notna().any()]
    fig, ax = plt.subplots(1, 2, figsize=(7.8, 3.4), layout="constrained",
                           gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax[0].plot([0, .6], [0, .6], ls="--", color="0.6", lw=.9)
    eces = {}
    for m in ms:
        s = d[m].values; ok = np.isfinite(s)
        q = np.quantile(s[ok], np.linspace(0, 1, 13)); q[0], q[-1] = -np.inf, np.inf
        b = np.digitize(s[ok], q[1:-1])
        xs = [s[ok][b == k].mean() for k in np.unique(b)]
        ys = [y[ok][b == k].mean() for k in np.unique(b)]
        ax[0].plot(xs, ys, "o-", ms=3.2, lw=1.2, color=ps.col(m),
                   label=m.replace(" (published)", "*"))
        eces[m] = ece(y[ok], s[ok])
    ax[0].set_xlabel("Mean predicted probability"); ax[0].set_ylabel("Observed frequency")
    ax[0].set_title("(a) Reliability diagram (12 equal-count bins)", fontsize=9)
    ax[0].legend(fontsize=6.8); ax[0].set_xscale("log"); ax[0].set_yscale("log")
    k = list(eces)
    ax[1].barh(range(len(k)), [eces[m] for m in k], color=[ps.col(m) for m in k], height=.7)
    ax[1].set_yticks(range(len(k)))
    ax[1].set_yticklabels([m.replace(" (published)", "*") for m in k], fontsize=7.5)
    ax[1].invert_yaxis(); ax[1].set_xlabel("Expected calibration error (lower is better)")
    ax[1].set_title("(b) Calibration error", fontsize=9)
    ps.save(fig, "fig09_calibration")

if __name__ == "__main__":
    for w in (sys.argv[1:] or ["5", "6", "7", "8", "9", "10", "17", "18"]):
        print(f"-> fig{w}"); globals()[f"fig{int(w):02d}"]()
