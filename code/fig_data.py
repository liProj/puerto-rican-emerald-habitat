"""Figures 1-4 and 16: study area, data verification, encounter rates, architecture, ordering test."""
import sys, os; sys.path.insert(0, "Birds/code")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import matplotlib.patches as mp
from matplotlib.lines import Line2D
import plotstyle as ps, data_prep as dp, terrain as TR
from common import load, PER_EN
ps.setup()
OUT = "Birds/figures"

def fig01():
    import rasterio
    # Figure 1 counts come from the pre-GEE export (7-dp coordinates), which is what the
    # authors' own Figure 1 script uses; the GEE round-trip in the NDVI export shifts
    # coordinates by <1 m and merges 20 location pairs.
    P = ("Birds/supp/birds-07-00055-s001/Riccordia_maugaeus_reproducibility_package/"
         "01 Riccordia_preprocessing_environmental_covariates/01_data_preparation/output/"
         "Datos_eBird_filtrados_pre_GEE.csv")
    d = pd.read_csv(P); d["Date"] = pd.to_datetime(d.Date)
    d = d[(d.Date >= "2013-01-01") & (d.Date <= "2025-12-31")].reset_index(drop=True)
    with rasterio.open(TR.TIF) as s:
        z = s.read(1, out_shape=(s.height // 6, s.width // 6)).astype(float)
        ext = [s.bounds.left, s.bounds.right, s.bounds.bottom, s.bounds.top]
    z[z <= 0] = np.nan                       # mask ocean so the land relief carries the colour
    fig, ax = plt.subplots(2, 1, figsize=(8.4, 5.0), layout="constrained")
    cmap = plt.get_cmap("terrain").copy(); cmap.set_bad("#EAF2F8")
    for a in ax:
        im = a.imshow(z, extent=ext, origin="upper", cmap=cmap, vmin=0, vmax=1300)
        a.set_xlim(-67.32, -65.18); a.set_ylim(17.88, 18.58); a.set_aspect("equal")
        a.set_xlabel("Longitude (°)"); a.set_ylabel("Latitude (°)"); a.grid(alpha=.15)
    u = d.drop_duplicates(subset=["Longitude", "Latitude"])
    det = d[d.Presence == 1].drop_duplicates(subset=["Longitude", "Latitude"])
    ax[0].scatter(u.Longitude, u.Latitude, s=.7, c="0.25", alpha=.35, lw=0, label=f"sampling locations (n={len(u):,})")
    ax[0].scatter(det.Longitude, det.Latitude, s=2.2, c="#C0392B", alpha=.85, lw=0,
                  label=f"$\\geq$1 report of the species (n={len(det):,})")
    ax[0].set_title("(a) SRTM topography and filtered eBird checklist coverage, 2013–2025", fontsize=9)
    ax[0].legend(loc="lower left", markerscale=5, fontsize=7, ncol=2)
    f, _ = dp.spatial_folds(d.assign(y=d.Presence), 10, 0.08, 0)
    ax[1].images[0].set_alpha(0.28)          # fade the hillshade behind the fold colours
    sc = ax[1].scatter(d.Longitude, d.Latitude, s=1.3, c=f, cmap="tab10", alpha=.85, lw=0, vmin=-.5, vmax=9.5)
    ax[1].set_title("(b) Spatial blocking for cross-validation: 0.08° blocks assigned to 10 folds",
                    fontsize=9)
    cb = fig.colorbar(sc, ax=ax[1], fraction=.018, pad=.008, ticks=range(10))
    cb.set_label("CV fold", fontsize=7.5); cb.ax.tick_params(labelsize=7)
    cb2 = fig.colorbar(im, ax=ax[0], fraction=.018, pad=.008); cb2.set_label("Elevation (m)", fontsize=7.5)
    cb2.ax.tick_params(labelsize=7)
    ps.save(fig, "fig01_study_area_and_spatial_blocks")

def fig02():
    d = pd.read_csv("Birds/results/data_check.csv")
    g = pd.read_csv("Birds/results/baseline/reproduction_check_glm.csv")
    o = pd.read_csv("Birds/results/baseline/occupancy_vs_paper.csv")
    o = o[o["mode"] == "as_published"]
    fig, ax = plt.subplots(1, 3, figsize=(8.6, 3.4), gridspec_kw=dict(width_ratios=[.7, .7, 1.8]),
                           layout="constrained")
    for a, (lab, ok, tot) in zip(ax[:2], [("Dataset\nverification", int(d.match.sum()), len(d)),
                                          ("GLM\nreproduction", int(g.match.sum()), len(g))]):
        a.bar([0], [ok], color="#0B6E4F", label="reproduces exactly")
        if tot - ok: a.bar([0], [tot - ok], bottom=[ok], color="#C0392B", label="deviates")
        a.set_xticks([]); a.set_ylim(0, tot * 1.18); a.set_ylabel("number of checks")
        a.set_title(f"{lab}\n{ok}/{tot}", fontsize=8.5)
        a.text(0, ok / 2, f"{ok}", ha="center", va="center", color="w", fontweight="bold")
        if tot - ok: a.text(0, ok + (tot - ok) / 2, f"{tot-ok}", ha="center", va="center", color="w", fontsize=7)
    ax[0].legend(loc="upper center", fontsize=7)
    a = ax[2]
    yy = np.arange(len(o))
    a.errorbar(o.paper_est, yy + .16, xerr=o.paper_SE, fmt="s", ms=4, color="#B03A2E",
               capsize=2, lw=1, label="published Table 2")
    a.errorbar(o.est, yy - .16, xerr=o.SE, fmt="o", ms=4, color="#0B6E4F",
               capsize=2, lw=1, label="our run of the authors' script")
    a.set_yticks(yy); a.set_yticklabels([f"{c}: {p}" for c, p in zip(o.component, o.parameter)], fontsize=7)
    a.axvline(0, color="0.5", lw=.7, ls=":"); a.invert_yaxis()
    a.set_xlabel("logit-scale estimate ± SE")
    a.set_title(f"Dynamic occupancy model: {int(o.est_match.sum())}/11 estimates\nand {int(o.se_match.sum())}/11 SEs reproduce exactly", fontsize=8.5)
    a.legend(loc="lower left", fontsize=7, frameon=True, framealpha=.92,
             edgecolor="0.8", handlelength=1.4)
    a.set_xlim(a.get_xlim()[0] - .25, a.get_xlim()[1])
    ps.save(fig, "fig02_verification_and_reproduction")

def fig03():
    d = load("NDVI")
    a = d.groupby(d.Date.dt.year).Presence.agg(["sum", "count"])
    a["p"] = a["sum"] / a["count"]
    lo = np.array([_ci(k, n)[0] for k, n in zip(a["sum"], a["count"])])
    hi = np.array([_ci(k, n)[1] for k, n in zip(a["sum"], a["count"])])
    fig, ax = plt.subplots(1, 2, figsize=(8.0, 3.1), layout="constrained")
    ax[0].errorbar(a.index, 100 * a.p, yerr=[100 * (a.p - lo), 100 * (hi - a.p)],
                   fmt="o-", ms=4, lw=1.2, color="#0B6E4F", capsize=2.5)
    for x, lab in [(2017.72, "Hurricane María"), (2022.72, "Hurricane Fiona")]:
        ax[0].axvline(x, ls="--", color="0.35", lw=1)
        ax[0].text(x + .07, ax[0].get_ylim()[1] * .96, lab, rotation=90, va="top", fontsize=7, color="0.3")
    ax[0].set_xlabel("Year"); ax[0].set_ylabel("Observed encounter rate (%)")
    ax[0].set_title("(a) Annual observed encounter rate\n(reproduces published Figure 2)", fontsize=9)
    per = ["Pre-Maria", "Inter-Huracanes", "Post-Fiona"]
    t = d.groupby("Period", observed=True).Presence.agg(["sum", "count"]).loc[per]
    ax[1].bar(range(3), 100 * t["sum"] / t["count"], color=["#2E86C1", "#C0392B", "#0B6E4F"], width=.6)
    for i, (k, n) in enumerate(zip(t["sum"], t["count"])):
        ax[1].text(i, 100 * k / n + .12, f"{100*k/n:.2f}%\nn={k:,}/{n:,}", ha="center", fontsize=7)
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels([PER_EN[p] for p in per], fontsize=8)
    ax[1].set_ylabel("Observed encounter rate (%)"); ax[1].set_ylim(0, 9.2)
    ax[1].set_title("(b) Encounter rate by hurricane period\n(reproduces published Table 1)", fontsize=9)
    ps.save(fig, "fig03_encounter_rates")

def _ci(k, n):
    from scipy.stats import beta
    lo = 0 if k == 0 else beta.ppf(.025, k, n - k + 1)
    hi = 1 if k == n else beta.ppf(.975, k + 1, n - k)
    return lo, hi

def fig16():
    t = pd.read_csv("Birds/results/baseline/occupancy_ordering_test.csv")
    lab = {"psi": "$\\psi$ initial occupancy", "col": "$\\gamma$ colonisation",
           "ext": "$\\varepsilon$ local extinction", "det": "$p$ detection"}
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.8), gridspec_kw=dict(width_ratios=[1.5, 1]),
                           layout="constrained")
    a = t[t["mode"] == "as_published"].reset_index(drop=True)
    b = t[t["mode"] == "site_major"].reset_index(drop=True)
    yy = np.arange(len(a))
    ax[0].errorbar(a.estimate, yy + .17, xerr=a.SE, fmt="s", ms=4.5, color="#B03A2E", capsize=2, lw=1,
                   label="as published (obsCovs stacked `each = M`)")
    ax[0].errorbar(b.estimate, yy - .17, xerr=b.SE, fmt="o", ms=4.5, color="#0B6E4F", capsize=2, lw=1,
                   label="corrected (site-major, `times = M`)")
    flip = [(a.p[i] < .05) != (b.p[i] < .05) for i in yy]
    for i in yy:
        if flip[i]:
            ax[0].axhspan(i - .42, i + .42, color="#FDEBD0", zorder=0)
    ax[0].set_yticks(yy)
    ax[0].set_yticklabels([f"{lab[c].split(' ')[0]} {p}" for c, p in zip(a.component, a.parameter)], fontsize=7.5)
    ax[0].axvline(0, color="0.5", lw=.7, ls=":"); ax[0].invert_yaxis()
    ax[0].set_xlabel("logit-scale estimate ± SE")
    h, l = ax[0].get_legend_handles_labels()
    h.append(mp.Patch(fc="#FDEBD0", ec="0.7")); l.append("significance changes")
    ax[0].legend(h, l, loc="lower left", fontsize=6.8)
    ax[0].set_title("(a) Observation-covariate stacking order", fontsize=9)
    det = ["(Intercept)", "PeriodPost-Fiona", "PeriodPre-Maria"]
    il = lambda x: 1 / (1 + np.exp(-x))
    ad = a[a.component == "det"].set_index("parameter").loc[det, "estimate"].values
    bd = b[b.component == "det"].set_index("parameter").loc[det, "estimate"].values
    pa = [il(ad[0]), il(ad[0] + ad[1]), il(ad[0] + ad[2])]
    pb = [il(bd[0]), il(bd[0] + bd[1]), il(bd[0] + bd[2])]
    x = np.arange(3)
    ax[1].bar(x - .19, 100 * np.array(pa), .36, color="#B03A2E", label="as published")
    ax[1].bar(x + .19, 100 * np.array(pb), .36, color="#0B6E4F", label="corrected")
    ax[1].set_xticks(x); ax[1].set_xticklabels(["Inter-\nHurricanes", "Post-\nFiona", "Pre-\nMaría"], fontsize=7.5)
    ax[1].set_ylabel("Detection probability $p$ (%)"); ax[1].legend(fontsize=7)
    ax[1].set_title("(b) Detection probability by period", fontsize=9)
    pv = a[(a.component == "det") & (a.parameter == "PeriodPre-Maria")].p.iloc[0]
    pv2 = b[(b.component == "det") & (b.parameter == "PeriodPre-Maria")].p.iloc[0]
    ax[1].text(.5, .02, f"Pre-María vs reference:\npublished p = {pv:.3f} (significant)\ncorrected p = {pv2:.2f} (not significant)",
               transform=ax[1].transAxes, fontsize=6.8, ha="center", va="bottom",
               bbox=dict(fc="#FDEBD0", ec="0.7", lw=.5))
    ax[1].set_ylim(0, 44)
    ps.save(fig, "fig16_occupancy_obscovs_ordering")

if __name__ == "__main__":
    import argparse
    which = sys.argv[1:] or ["1", "2", "3", "16"]
    for w in which:
        print(f"-> fig{w}"); globals()[f"fig{int(w):02d}"]()
