"""Figure 4: the HERON architecture."""
import sys; sys.path.insert(0, "Birds/code")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import plotstyle as ps
ps.setup()

def box(ax, x, y, w, h, text, fc, ec="0.35", fs=7.4, tc="black", lw=1.0, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.02",
                                fc=fc, ec=ec, lw=lw, linestyle=ls, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3,
            color=tc, linespacing=1.35)

def arrow(ax, p, q, style="-|>", color="0.35", lw=1.1, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=11, color=color,
                                 lw=lw, linestyle=ls, zorder=1,
                                 connectionstyle=f"arc3,rad={rad}"))

fig, ax = plt.subplots(figsize=(9.6, 5.5), layout="constrained")
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off"); ax.grid(False)

C_IN, C_NN, C_TR, C_OUT, C_LOSS = "#D6EAF8", "#D4EFDF", "#FCF3CF", "#E8DAEF", "#FADBD8"

ax.text(.015, .965, "(a)  Neural branch: factorized reporting scores", fontsize=9,
        fontweight="bold", va="center")
box(ax, .02, .74, .17, .14, "Environment\nelevation, NDVI,\nNDVI-missing mask,\nperiod", C_IN)
box(ax, .02, .555, .17, .125, "Observation context\nperiod, day-of-year,\nLandsat lag", C_IN)
box(ax, .02, .40, .17, .11, "Sampling effort\nduration, distance", C_IN)
box(ax, .235, .755, .155, .11, "Environment branch\n3$\\times$128 MLP", C_NN)
box(ax, .235, .575, .155, .10, "Effort branch\n3$\\times$128 MLP", C_NN)
box(ax, .235, .405, .155, .095, "Monotone effort\nbasis expansion", C_NN)
for y0, y1 in ((.81, .81), (.617, .625), (.455, .452)):
    arrow(ax, (.19, y0), (.235, y1))
arrow(ax, (.3125, .405), (.3125, .575))
box(ax, .44, .60, .135, .115, "$s \\times d$\nproduct head", C_NN, lw=1.6)
arrow(ax, (.39, .81), (.44, .70), rad=-.2)
arrow(ax, (.39, .625), (.44, .657))
box(ax, .625, .755, .33, .105,
    "Grouped reporting loss\n$1-\\prod_i\\,(1-s_i d_i)$  vs  'detected at least once'\n"
    "per (site, period) cell; depends on branch products", C_LOSS, fs=7.0)
box(ax, .625, .615, .33, .10,
    "Spatial V-REx penalty\nvariance of per-block risks across\ngeographic blocks in the batch", C_LOSS, fs=7.0)
box(ax, .625, .475, .33, .10,
    "Inner spatially blocked early stopping\nwhole blocks withheld; best inner-AP\nepoch retained", C_LOSS, fs=7.0)
for y in (.807, .665, .525):
    arrow(ax, (.575, .657), (.625, y), rad=.1, ls=(0, (3, 2)))
ax.text(.44, .565, "Training source: 89,682 checklists\nNDVI missingness retained",
        ha="left", va="top", fontsize=6.8, style="italic", color="#1A5276")

ax.text(.015, .335, "(b)  Stacking and distillation", fontsize=9, fontweight="bold", va="center")
box(ax, .02, .155, .155, .125, "neural branch\n(a), 3-member\ndeep ensemble", C_NN)
box(ax, .195, .155, .155, .125, "LightGBM\n+ 20 SRTM terrain\ndescriptors", C_TR)
box(ax, .37, .155, .155, .125, "XGBoost\n+ 20 SRTM terrain\ndescriptors", C_TR)
box(ax, .545, .155, .135, .125, "published\nGLM m4", "#F5CBA7")
box(ax, .725, .17, .245, .095,
    "logistic meta-learner\nfitted on INNER spatially blocked\nout-of-fold predictions", C_OUT, fs=7.0)
for x in (.0975, .2725, .4475, .6125):
    arrow(ax, (x, .155), (x + .04, .095), rad=-.15)
ax.plot([.09, .70], [.085, .085], color="0.45", lw=1.1, zorder=1)
arrow(ax, (.70, .085), (.8475, .17))
box(ax, .725, .035, .245, .085, "Platt scaling (strictly monotone,\nso ranking metrics unchanged)\n"
    "$\\rightarrow$ distilled student network", C_OUT, fs=7.0)
arrow(ax, (.8475, .17), (.8475, .12))
ax.text(.02, -.025, "Blue = inputs · green = learned neural components · yellow = tree members · "
        "orange = the published baseline · red = training losses · purple = combination",
        fontsize=6.0, color="0.35")
ps.save(fig, "fig04_heron_architecture")
