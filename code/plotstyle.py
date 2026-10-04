"""Shared figure style. Colour-blind-safe, consistent across all panels, print-ready."""
import matplotlib as mpl, matplotlib.pyplot as plt
mpl.use("Agg")
PAL = {"HERON": "#0B6E4F", "HERON-student": "#3FA06B", "GLM-m4 (published)": "#B03A2E",
       "GLM-m3 (published)": "#D98880", "GLM-splines": "#E67E22", "LightGBM": "#2E86C1",
       "XGBoost": "#5B2C6F", "RandomForest": "#7F8C8D", "MLP": "#A6ACAF",
       "LightGBM + terrain": "#1B4F72", "XGBoost + terrain": "#8E44AD",
       "MLP + terrain": "#616A6B", "GLM-m4 + terrain": "#CA6F1E",
       "HERON-plus": "#117A65",
       "HERON (teacher ens.)": "#0E8A63"}
SEQ = "viridis"
def setup():
    mpl.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 330, "savefig.bbox": "tight",
        "font.family": "DejaVu Sans", "font.size": 9,
        "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "legend.frameon": False, "legend.fontsize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
    })
def col(m): return PAL.get(m, "#95A5A6")
def save(fig, name, outdir="Birds/figures"):
    import os; os.makedirs(outdir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{outdir}/{name}.{ext}")
    plt.close(fig); print(f"  figure -> {outdir}/{name}.png")
