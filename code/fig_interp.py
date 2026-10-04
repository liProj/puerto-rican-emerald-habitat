"""Figures 11-15, 19, 20: what HERON learned, and the ecological read-outs."""
import sys, os; sys.path.insert(0, "Birds/code")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import plotstyle as ps
ps.setup()
FM = "Birds/results/final_model"; R = "Birds/results"
PCOL = {"Pre-Maria": "#2E86C1", "Inter-Hurricanes": "#C0392B", "Post-Fiona": "#0B6E4F"}
def _ok(p): return os.path.exists(p)

def fig11():
    f = f"{FM}/detection_effort_curve.csv"
    if not _ok(f): return print("  (skip fig11)")
    d = pd.read_csv(f)
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9), layout="constrained")
    for a, (v, lab) in zip(ax, [("Duration_min", "Checklist duration (min)"),
                                ("Distance_km", "Distance travelled (km)")]):
        s = d[d.variable == v]
        a.plot(s.value, 100 * s.p_det, lw=2, color="#0B6E4F")
        a.set_xlabel(lab); a.set_ylabel("Effort-branch score (%)")
        a.set_title(f"Fitted effort response to {lab.split(' (')[0].lower()}", fontsize=9)
    fig.suptitle("HERON's effort branch is monotone in sampling effort by construction; "
                 "its shape is learned from the data", fontsize=8.4)
    ps.save(fig, "fig11_detection_effort_curve")

def fig12():
    fe, fn = f"{FM}/pd_elevation_by_period.csv", f"{FM}/pd_ndvi_by_period.csv"
    if not _ok(fe): return print("  (skip fig12)")
    E, N = pd.read_csv(fe), pd.read_csv(fn)
    gl = pd.read_csv("Birds/results/baseline/glm_predictions_by_elevation.csv")
    fig, ax = plt.subplots(1, 3, figsize=(10.2, 3.1), layout="constrained")
    for per, c in PCOL.items():
        s = E[E.period == per]
        ax[0].plot(s.Elevation_m, 100 * s.p_report, lw=1.9, color=c, label=per)
        ax[0].plot(gl.Elevation_m, 100 * gl[per], lw=1.2, ls="--", color=c, alpha=.75)
        ax[1].plot(s.Elevation_m, 100 * s.psi, lw=1.9, color=c, label=per)
        t = N[N.period == per]
        ax[2].plot(t.NDVI, 100 * t.psi, lw=1.9, color=c, label=per)
    ax[0].set_xlabel("Elevation (m)"); ax[0].set_ylabel("Reporting probability (%)")
    ax[0].set_title("(a) Reporting probability\nsolid = HERON, dashed = published GLM", fontsize=8.6)
    ax[1].set_xlabel("Elevation (m)"); ax[1].set_ylabel("Environmental-branch score (%)")
    ax[1].set_title("(b) HERON's environmental branch\nvs elevation", fontsize=8.6)
    ax[2].set_xlabel("NDVI"); ax[2].set_ylabel("Environmental-branch score (%)")
    ax[2].set_title("(c) HERON's environmental branch\nvs vegetation greenness", fontsize=8.6)
    ax[0].legend(fontsize=7)
    ps.save(fig, "fig12_psi_partial_dependence")

def fig13():
    fh = f"{FM}/elevational_centroids_heron.csv"
    gb = "Birds/results/baseline/glm_elevational_centroids.csv"
    if not (_ok(fh) and _ok(gb)): return print("  (skip fig13)")
    H, G = pd.read_csv(fh), pd.read_csv(gb)
    fig, a = plt.subplots(figsize=(5.4, 3.0), layout="constrained")
    x = np.arange(3); ords = ["Pre-Maria", "Inter-Hurricanes", "Post-Fiona"]
    g = G.set_index("period").loc[ords]; h = H.set_index("period").loc[ords]
    a.bar(x - .19, g.centroid_m, .36, color="#B03A2E", label="published GLM (reproduced)")
    a.bar(x + .19, h.centroid_report, .36, color="#0B6E4F", label="HERON")
    for i in x:
        a.text(i - .19, g.centroid_m.iloc[i] + 4, f"{g.delta_vs_PreMaria.iloc[i]:+.1f} m",
               ha="center", fontsize=6.8, color="#B03A2E")
        a.text(i + .19, h.centroid_report.iloc[i] + 4, f"{h.delta_centroid_report.iloc[i]:+.1f} m",
               ha="center", fontsize=6.8, color="#0B6E4F")
    a.set_xticks(x); a.set_xticklabels(ords, fontsize=8)
    a.set_ylabel("Predicted elevational centroid (m)")
    a.set_title("Elevational centroid by period (published Figure 4 statistic)\n"
                "labels show the shift relative to each model's own Pre-María baseline", fontsize=8.6)
    a.text(.5, -.30, "Absolute heights are not comparable between model classes: the GLM's linear "
           "logit keeps rising to 1300 m\nwhile HERON saturates above ~700 m. The within-model "
           "shifts are the comparable quantity.",
           transform=a.transAxes, ha="center", va="top", fontsize=6.6, color="0.35")
    a.legend(fontsize=7); a.set_ylim(0, max(g.centroid_m.max(), h.centroid_report.max()) * 1.25)
    ps.save(fig, "fig13_elevational_centroid")

def fig14():
    fc = "Birds/results/author_channel/outputs/tables/corrected_ndvi_dynamic_local_extinction_by_elevation.csv"
    if not _ok(fc): return print("  (skip fig14)")
    c = pd.read_csv(fc)
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.0), layout="constrained")
    ax[0].plot(c.Elevation_m, c.predicted_percent, lw=2, color="#B03A2E")
    ax[0].fill_between(c.Elevation_m, c.lower_percent, c.upper_percent, color="#B03A2E", alpha=.18)
    ax[0].set_xlabel("Site mean elevation (m)"); ax[0].set_ylabel("Local extinction $\\varepsilon$ (%)")
    ax[0].set_title("(a) Published dynamic occupancy model\n(reproduces Figure 5)", fontsize=8.6)
    fe = f"{FM}/pd_elevation_by_period.csv"
    if _ok(fe):
        E = pd.read_csv(fe)
        for per, col in PCOL.items():
            s = E[E.period == per]
            ax[1].plot(s.Elevation_m, 100 * (1 - s.psi), lw=1.8, color=col, label=per)
        ax[1].set_xlabel("Elevation (m)"); ax[1].set_ylabel("1 - environmental-branch score (%)")
        ax[1].set_title("(b) HERON's environmental branch\n(complement of fitted score)", fontsize=8.6)
        ax[1].legend(fontsize=7)
    fig.suptitle("Elevation gradients in a transition probability and a branch score", fontsize=8.8)
    ps.save(fig, "fig14_extinction_vs_elevation")

def fig15():
    fa = "Birds/results/author_channel/outputs/tables/corrected_ndvi_dynamic_local_extinction_by_site.csv"
    fh = f"{FM}/site_psi_surface.csv"
    if not (_ok(fa) and _ok(fh)): return print("  (skip fig15)")
    A = pd.read_csv(fa); Hs = pd.read_csv(fh)
    fig, ax = plt.subplots(2, 1, figsize=(8.0, 4.8), layout="constrained")
    s = ax[0].scatter(A.Grid_Longitude, A.Grid_Latitude, c=A.predicted_percent, s=9,
                      cmap="magma_r", vmin=0, vmax=100, lw=0)
    ax[0].set_title("(a) Published model: predicted local extinction probability per grid cell "
                    "(reproduces Figure 6)", fontsize=8.6)
    fig.colorbar(s, ax=ax[0], fraction=.02, pad=.01, label="$\\varepsilon$ (%)")
    pf = Hs[Hs.period == "Post-Fiona"]
    s2 = ax[1].scatter(pf.Longitude, pf.Latitude, c=100 * pf.psi, s=9, cmap="viridis", lw=0)
    ax[1].set_title("(b) HERON: environmental-branch score per site, Post-Fiona", fontsize=8.6)
    fig.colorbar(s2, ax=ax[1], fraction=.02, pad=.01, label="Branch score (%)")
    for a in ax:
        a.set_xlim(-67.32, -65.18); a.set_ylim(17.88, 18.58); a.set_aspect("equal")
        a.set_xlabel("Longitude (°)"); a.set_ylabel("Latitude (°)")
    ps.save(fig, "fig15_spatial_surfaces")

def fig19():
    fn, ft = f"{FM}/permutation_importance_nn.csv", f"{FM}/permutation_importance_tree.csv"
    if not (_ok(fn) and _ok(ft)): return print("  (skip fig19)")
    N, T = pd.read_csv(fn), pd.read_csv(ft).head(16)
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 4.2), layout="constrained",
                           gridspec_kw=dict(width_ratios=[1, 1.25]))
    n = N.sort_values("delta_AP")
    ax[0].barh(range(len(n)), n.delta_AP, xerr=n.sd, color="#0B6E4F", height=.7,
               error_kw=dict(lw=.8, capsize=2, ecolor="0.3"))
    ax[0].set_yticks(range(len(n))); ax[0].set_yticklabels(n.feature, fontsize=8)
    ax[0].axvline(0, color="0.4", lw=.8)
    ax[0].set_xlabel("drop in average precision when permuted")
    ax[0].set_title("(a) HERON environmental branch\n(terrain deliberately excluded)", fontsize=8.8)
    t = T.sort_values("delta_AP")
    import terrain as TR
    cols = ["#0B6E4F" if f in set(TR.FEATS) else "#B03A2E" for f in t.feature]
    ax[1].barh(range(len(t)), t.delta_AP, xerr=t.sd, color=cols, height=.7,
               error_kw=dict(lw=.8, capsize=2, ecolor="0.3"))
    ax[1].set_yticks(range(len(t))); ax[1].set_yticklabels(t.feature, fontsize=7.5)
    ax[1].axvline(0, color="0.4", lw=.8)
    ax[1].set_xlabel("drop in average precision when permuted")
    from matplotlib.patches import Patch
    ax[1].legend(handles=[Patch(fc="#B03A2E", label="Observation / location covariate"),
                          Patch(fc="#0B6E4F", label="SRTM terrain descriptor (new)")], fontsize=7,
                 loc="lower right")
    ax[1].set_title("(b) LightGBM + terrain member\n(where terrain enters the stack)", fontsize=8.8)
    fig.suptitle("Permutation importance: neural fitting source (left), tree holdout (right)", fontsize=9)
    ps.save(fig, "fig19_permutation_importance")

def fig20():
    d = pd.read_csv(f"{R}/cv_ablation_spatial_NDVI.csv") if _ok(f"{R}/cv_ablation_spatial_NDVI.csv") else None
    if d is None: return print("  (skip fig20)")
    pair = ["HERON (full)", "- all-data training"]
    if not set(pair) <= set(d.method): return print("  (skip fig20: missing variants)")
    fig, ax = plt.subplots(1, 3, figsize=(8.4, 2.9), layout="constrained")
    for a, k in zip(ax, ["AUC", "AP", "pseudoR2"]):
        A = d[d.method == pair[1]].set_index(["rep", "fold"])[k]
        B = d[d.method == pair[0]].set_index(["rep", "fold"])[k]
        ix = A.index.intersection(B.index)
        for i in ix: a.plot([0, 1], [A.loc[i], B.loc[i]], color="0.7", lw=.7, zorder=1)
        a.scatter([0] * len(ix), A.loc[ix], s=22, color="#B03A2E", zorder=2)
        a.scatter([1] * len(ix), B.loc[ix], s=22, color="#0B6E4F", zorder=2)
        a.set_xticks([0, 1]); a.set_xticklabels(["complete cases\n(n = 68,976)",
                                                 "all checklists\n(n = 89,682)"], fontsize=7.5)
        a.set_ylabel(k); a.set_xlim(-.4, 1.4)
        a.set_title(f"{k}: {int((B.loc[ix] > A.loc[ix]).sum())}/{len(ix)} folds improve", fontsize=8.6)
    fig.suptitle("Using the 20,706 checklists the published complete-case GLM must discard", fontsize=8.8)
    ps.save(fig, "fig20_missingness_gain")

if __name__ == "__main__":
    for w in (sys.argv[1:] or ["11", "12", "13", "14", "15", "19", "20"]):
        print(f"-> fig{w}"); globals()[f"fig{int(w):02d}"]()
