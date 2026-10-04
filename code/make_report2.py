"""Append sections 4-8 (new method, experiments, ablations, conclusions) and save Birds/report.docx."""
import sys, os; sys.path.insert(0, "Birds/code")
import numpy as np, pandas as pd
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from metrics import HIGHER_BETTER
import stats_tests as st

R, FIG = "Birds/results", "Birds/figures"
def ex(p): return os.path.exists(p)
doc = Document("Birds/report_part1.docx")
def H(t, lvl=1): doc.add_heading(t, level=lvl)
def P(t, bold=False, italic=False, size=10):
    p = doc.add_paragraph(); r = p.add_run(t); r.bold = bold; r.italic = italic
    r.font.size = Pt(size); return p
def table(df, caption=None, maxrows=60, floatfmt=4):
    if caption: P(caption, italic=True, size=9)
    d = df.head(maxrows).copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: f"{v:.{floatfmt}f}" if pd.notna(v) else "")
    t = doc.add_table(rows=1, cols=len(d.columns)); t.style = "Light Grid Accent 1"
    for i, c in enumerate(d.columns):
        cell = t.rows[0].cells[i]; cell.text = str(c)
        for par in cell.paragraphs:
            for run in par.runs: run.bold = True; run.font.size = Pt(7.5)
    for _, row in d.iterrows():
        cs = t.add_row().cells
        for i, c in enumerate(d.columns):
            cs[i].text = str(row[c])
            for par in cs[i].paragraphs:
                for run in par.runs: run.font.size = Pt(7.5)
    if len(df) > maxrows: P(f"... {len(df)-maxrows} further rows in the CSV.", italic=True, size=8)
    doc.add_paragraph()
def figure(name, caption, width=6.3):
    p = f"{FIG}/{name}.png"
    if not ex(p): return P(f"[figure {name} not generated]", italic=True, size=9)
    doc.add_picture(p, width=Inches(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    P(caption, italic=True, size=8.5)
def summary(csv):
    if not ex(f"{R}/{csv}"): return None
    d = pd.read_csv(f"{R}/{csv}")
    g = d.groupby("method")[list(HIGHER_BETTER)].agg(["mean", "std"])
    g.columns = [f"{a}" if b == "mean" else f"{a}_sd" for a, b in g.columns]
    return d, g.reset_index()

doc.add_page_break()
H("4. The new method: HERON")
P("Design rationale. The published GLM is a single linear predictor in which sampling effort and "
  "habitat compete for the same coefficients, it is fitted to complete cases only, and it is "
  "never evaluated out of sample. HERON keeps the structure of the occupancy model but fits it "
  "at checklist resolution:")
P("    P(report | x) = psi(environment, period) x p_det(effort, season, period)", bold=True)
for t in ["Exclusion restriction: effort enters only the detection branch, environment only the "
          "occupancy branch — the same assumption the published colext model makes, imposed "
          "inside a nonlinear learner.",
          "Hard monotonicity: the effort contribution is a non-negative combination of increasing "
          "basis functions, so detection cannot decrease with effort and the model cannot dispose "
          "of habitat signal as effort.",
          "Grouped occupancy likelihood: for each (site, period) cell with repeat visits, the "
          "probability of at least one report is 1 - prod(1 - psi*p); supervising this supplies "
          "the repeat-visit information that identifies psi from p.",
          "Missingness-native inputs: NDVI enters with an explicit mask, so all 89,682 checklists "
          "are used instead of the 68,976 complete cases.",
          "Inner spatially blocked early stopping: whole blocks are withheld from the training "
          "fold and the best inner-AP epoch is kept. Selecting on a random inner split measures a "
          "leakier regime than the outer test.",
          "Spatial V-REx penalty: the variance of per-block risks is penalised, encouraging "
          "predictions that transfer across regions.",
          "Terrain descriptors: 20 multi-scale slope / aspect / topographic-position / roughness / "
          "relief layers derived from the SRTM raster the authors themselves released.",
          "Stacking: the neural branch, terrain-augmented LightGBM and XGBoost, and the published "
          "GLM are combined by a logistic meta-learner fitted on inner spatially blocked "
          "out-of-fold predictions, then Platt-scaled.",
          "Distillation: the stack is compressed into one compact factorised network trained on "
          "its soft outputs, so the deployed model inherits terrain signal without fitting terrain."]:
    P("  • " + t, size=9.5)
figure("fig04_heron_architecture", "Figure 4.1. HERON architecture.")

H("4.1 Design decisions driven by measured negative results", 2)
P("Two components that are standard practice were measured and rejected:", bold=True)
P("  • Random Fourier spatial features. Under spatial blocking these collapsed performance "
  "(AUC 0.835 -> 0.722, AP 0.289 -> 0.137 on the diagnostic folds). A stationary spatial basis "
  "interpolates, and a held-out block gives it nothing to interpolate between. Reported as an "
  "ablation rather than quietly dropped.", size=9.5)
P("  • Terrain features inside the network. They help gradient-boosted trees (AUC 0.854 -> 0.871) "
  "but hurt the neural branch (AUC 0.838 -> 0.785 even with early stopping), because the network "
  "has the capacity to memorise terrain as a regional fingerprint. Terrain is therefore routed "
  "through the tree members and reaches the student via distillation.", size=9.5)
P("  • Isotonic calibration was replaced by Platt scaling: isotonic is piecewise constant, and "
  "the ties it creates measurably damage AUC and average precision, whereas Platt is strictly "
  "increasing and leaves every ranking metric unchanged.", size=9.5)

H("5. Experiments")
P("Three evaluation protocols are used. Spatially blocked cross-validation (primary) assigns "
  "contiguous 0.08-degree blocks to ten folds, so a whole region is held out at once; this is "
  "necessary because the 89,682 checklists come from only 24,876 locations, many visited hundreds "
  "of times, so a random split places near-duplicates of test records in training. Random "
  "stratified k-fold is reported as the conventional, optimistic comparison. Temporal "
  "extrapolation fits on Pre-Maria plus Inter-Hurricanes and predicts Post-Fiona.")
P("Sixteen metrics are computed per fold. Methods are compared fold-by-fold against the published "
  "m4 using the Nadeau-Bengio corrected resampled t-test (repeated k-fold training sets overlap, "
  "so the naive paired t-test is anti-conservative) and a Wilcoxon signed-rank test, with Holm "
  "correction across metrics.")

for tag, csv, title in [("5.1", "cv_main_spatial_NDVI.csv", "Spatially blocked cross-validation (primary)"),
                        ("5.2", "cv_main_random_NDVI.csv", "Random k-fold cross-validation"),
                        ("5.3", "cv_temporal.csv", "Temporal extrapolation: forecasting Post-Fiona"),
                        ("5.4", "cv_main_spatial_EVI2.csv", "Sensitivity: EVI2 in place of NDVI")]:
    r = summary(csv)
    H(f"{tag} {title}", 2)
    if r is None: P("[not run]", italic=True); continue
    d, g = r
    P(f"{d[['rep','fold']].drop_duplicates().shape[0]} folds, {d.method.nunique()} methods.")
    table(g[["method"] + list(HIGHER_BETTER)], f"Table {tag}.1. Mean over folds.", maxrows=25)
    if "HERON" in set(d.method):
        c = st.compare(d, "GLM-m4 (published)", "HERON", group_cols=("protocol",))
        if len(c):
            P(f"HERON beats the published GLM on {int(c.better.sum())}/{len(c)} metrics "
              f"({100*c.better.mean():.1f}%); {int(c.signif_win.sum())} significant wins and "
              f"{int(c.signif_loss.sum())} significant losses after Holm correction.", bold=True)
            table(c[["metric","ref_mean","new_mean","delta","wins","losses","p_nb_holm","signif_win","signif_loss"]],
                  f"Table {tag}.2. Paired comparison against the published GLM.", maxrows=20)

figure("fig05_main_benchmark_spatial", "Figure 5.1. Benchmark under spatial blocking.")
figure("fig06_protocol_comparison", "Figure 5.2. The three evaluation protocols.")
figure("fig07_paired_folds", "Figure 5.3. Paired per-fold comparison.")
figure("fig18_metric_winloss_heatmap", "Figure 5.4. All 16 metrics against the published GLM.")
figure("fig17_blocksize_sensitivity", "Figure 5.5. Sensitivity to spatial block size.")

H("6. Ablations")
r = summary("cv_ablation_spatial_NDVI.csv")
if r is None: P("[not run]", italic=True)
else:
    d, g = r
    P(f"{d[['rep','fold']].drop_duplicates().shape[0]} spatially blocked folds, "
      f"{d.method.nunique()} variants.")
    table(g[["method", "AUC", "AP", "pseudoR2", "Brier", "LogLoss", "ECE"]],
          "Table 6.1. Ablation, mean over folds.", maxrows=25)
    full = "HERON (full)"
    if full in set(d.method):
        rows = []
        for m in d.method.unique():
            if m == full: continue
            A = d[d.method == full].set_index(["rep","fold"]); Bm = d[d.method == m].set_index(["rep","fold"])
            ix = A.index.intersection(Bm.index)
            rows.append(dict(variant=m, dAUC=(Bm.loc[ix,"AUC"]-A.loc[ix,"AUC"]).mean(),
                             dAP=(Bm.loc[ix,"AP"]-A.loc[ix,"AP"]).mean(),
                             dpseudoR2=(Bm.loc[ix,"pseudoR2"]-A.loc[ix,"pseudoR2"]).mean()))
        table(pd.DataFrame(rows).sort_values("dAP"),
              "Table 6.2. Change relative to the full model (negative = the component helps).")
figure("fig10_ablation", "Figure 6.1. Ablation.")
figure("fig20_missingness_gain", "Figure 6.2. Gain from using the checklists the complete-case GLM discards.")

H("7. What the model learned")
figure("fig11_detection_effort_curve", "Figure 7.1. Learned monotone detection-effort response.")
figure("fig12_psi_partial_dependence", "Figure 7.2. Occupancy branch vs elevation and greenness.")
figure("fig19_permutation_importance", "Figure 7.3. Permutation importance.")
figure("fig13_elevational_centroid", "Figure 7.4. Elevational centroid (published Figure 4 statistic).")
figure("fig14_extinction_vs_elevation", "Figure 7.5. Persistence vs elevation.")
figure("fig15_spatial_surfaces", "Figure 7.6. Spatial surfaces.")

H("8. Outputs")
P("All outputs are under ~/Documents/mdpi_papers/Birds/:", bold=True)
for t in ["screening.csv, screen_raw.csv — the 50-paper screening",
          "data_check.md, results/data_check.csv — dataset verification",
          "code/ — every analysis script",
          "results/baseline/ — reproduction of the published GLMs and occupancy model",
          "results/author_channel/ — output of the authors' own script, run unmodified",
          "results/ — all cross-validation, ablation and final-model results",
          "figures/ — all figures (PNG and PDF)",
          "paper/main.tex, paper/main.pdf — the manuscript",
          "report.docx — this report"]:
    P("  • " + t, size=9.5)
doc.save("Birds/report.docx")
print("wrote Birds/report.docx")
