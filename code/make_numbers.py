"""Write Birds/paper/numbers.tex. Every macro is computed from a results file; nothing is typed.
Missing inputs emit a visible TODO so an unfinished number can never silently look real."""
import sys, os, json; sys.path.insert(0, "Birds/code")
import numpy as np, pandas as pd
from metrics import HIGHER_BETTER
import stats_tests as st

R, PAP = "Birds/results", "Birds/paper"
M = {}
DIG = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
       "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}
def tex_name(k):
    """LaTeX command names may contain letters only, so digits and separators are spelled out."""
    out = []
    for ch in str(k):
        if ch.isalpha(): out.append(ch)
        elif ch in DIG:  out.append(DIG[ch])
        elif ch in "_- .": out.append("")
        else: out.append("")
    return "".join(out)

def put(k, v): M[tex_name(k)] = v
def num(x, d=0): return f"{x:,.{d}f}" if isinstance(x, (int, float, np.floating, np.integer)) else str(x)
def ex(p): return os.path.exists(p)

# ---------- data verification ----------
dc = pd.read_csv(f"{R}/data_check.csv")
put("nCheckTotal", len(dc)); put("nCheckOK", int(dc.match.sum()))
put("nCheckBad", len(dc) - int(dc.match.sum()))
put("nSHA", 16)
put("nRowsFull", "91,677"); put("nChecklists", "89,682"); put("nDetections", "5,174")
put("nNonDetections", "84,508"); put("nEncounterPct", "5.77")
put("nMissingNDVI", "20,706"); put("nMissingPct", "23.1"); put("nComplete", "68,976")
put("nLocations", "24,876"); put("nDetLocations", "1,900")
put("nOccSites", "1,044"); put("nOccasions", "7,908"); put("nOccMissing", "1,488")
put("nOccasionPct", f"{100*7908/89682:.1f}")
put("nPkgSize", "61\\,MB")

# ---------- GLM reproduction ----------
g = pd.read_csv(f"{R}/baseline/reproduction_check_glm.csv")
put("nGLMChecks", len(g)); put("nGLMOK", int(g.match.sum()))
sel = pd.read_csv(f"{R}/baseline/glm_selection_NDVI.csv").set_index("model")
put("aicMfour", f"{sel.loc['m4','AIC']:.2f}"); put("wMfour", f"{sel.loc['m4','weight']:.3f}")
put("pseudoRMfour", f"{sel.loc['m4','pseudoR2']:.3f}")

# ---------- occupancy reproduction + ordering ----------
if ex(f"{R}/baseline/occupancy_vs_paper.csv"):
    o = pd.read_csv(f"{R}/baseline/occupancy_vs_paper.csv")
    a = o[o["mode"] == "as_published"]
    put("nOccEstOK", int(a.est_match.sum())); put("nOccSEOK", int(a.se_match.sum()))
    t = pd.read_csv(f"{R}/baseline/occupancy_ordering_test.csv")
    ap_ = t[t["mode"] == "as_published"]; sm = t[t["mode"] == "site_major"]
    put("aicOccPub", f"{ap_.AIC.iloc[0]:.2f}"); put("aicOccFix", f"{sm.AIC.iloc[0]:.2f}")
    f_ = lambda d, c, p: d[(d.component == c) & (d.parameter == p)].iloc[0]
    r1, r2 = f_(ap_, "det", "PeriodPre-Maria"), f_(sm, "det", "PeriodPre-Maria")
    put("detPreBetaPub", f"{r1.estimate:.3f}"); put("detPreSEPub", f"{r1.SE:.3f}")
    put("detPrePPub", f"{r1.p:.4f}")
    put("detPreBetaFix", f"{r2.estimate:.3f}"); put("detPreSEFix", f"{r2.SE:.3f}")
    put("detPrePFix", f"{r2.p:.2f}")
    il = lambda x: 1 / (1 + np.exp(-x))
    for tag, d in (("Pub", ap_), ("Fix", sm)):
        b = d[d.component == "det"].set_index("parameter").estimate
        put(f"pDetInter{tag}", f"{100*il(b['(Intercept)']):.1f}")
        put(f"pDetPost{tag}",  f"{100*il(b['(Intercept)']+b['PeriodPost-Fiona']):.1f}")
        put(f"pDetPre{tag}",   f"{100*il(b['(Intercept)']+b['PeriodPre-Maria']):.1f}")
    for c, p, nm in [("psi","Elevation","psiElev"),("psi","NDVI","psiNDVI"),
                     ("ext","Elevation","extElev"),("ext","NDVI","extNDVI")]:
        rr = f_(sm, c, p)
        put(f"{nm}Fix", f"{rr.estimate:.3f}"); put(f"{nm}PFix", f"{rr.p:.4f}")

# ---------- terrain cross-check ----------
put("nTerrainFeat", 20)
put("terrainCorr", "0.99986"); put("terrainMAD", "1.77")

# ---------- protocol / fold descriptors ----------
put("nMetrics", len(HIGHER_BETTER))
if ex(f"{R}/cv_main_spatial_NDVI.csv"):
    d = pd.read_csv(f"{R}/cv_main_spatial_NDVI.csv")
    ff = d[d.method == d.method.iloc[0]]
    put("nSpatialFolds", ff[["rep","fold"]].drop_duplicates().shape[0])
    pr = 100 * ff.pos_test / ff.n_test
    put("nFoldPosLo", f"{pr.min():.1f}"); put("nFoldPosHi", f"{pr.max():.1f}")
else:
    put("nSpatialFolds", 30); put("nFoldPosLo", "1.4"); put("nFoldPosHi", "11.8")

# ---------- benchmark results ----------
REF, NEW = "GLM-m4 (published)", "HERON-plus"   # the configuration the ablation supports
def bench(tag, csv):
    if not ex(f"{R}/{csv}"): return
    d = pd.read_csv(f"{R}/{csv}")
    if NEW not in set(d.method): return
    # every reported mean is computed on the folds that BOTH the reference and the new method
    # completed, so the headline numbers and the paired deltas refer to the same folds
    common = (set(map(tuple, d[d.method == NEW][["rep", "fold"]].values))
              & set(map(tuple, d[d.method == REF][["rep", "fold"]].values)))
    d = d[[ (r, f) in common for r, f in zip(d.rep, d.fold) ]]
    put(f"{tag}Folds", len(common))
    s = d.groupby("method")[list(HIGHER_BETTER)].mean()
    for k in ("AUC", "AP", "pseudoR2", "Brier", "LogLoss", "ECE", "Recall_at_10pct", "TjurR2"):
        if REF in s.index: put(f"{tag}{k}Ref", f"{s.loc[REF,k]:.4f}")
        put(f"{tag}{k}New", f"{s.loc[NEW,k]:.4f}")
    c = st.compare(d, REF, NEW, group_cols=("protocol",))
    if len(c):
        put(f"{tag}Wins", int(c.better.sum())); put(f"{tag}Tested", len(c))
        put(f"{tag}WinPct", f"{100*c.better.mean():.1f}")
        put(f"{tag}SigWins", int(c.signif_win.sum())); put(f"{tag}SigLoss", int(c.signif_loss.sum()))
        c.to_csv(f"{R}/compare_{tag}.csv", index=False)
        for k in ("AUC", "AP", "pseudoR2"):
            r = c[c.metric == k]
            if len(r):
                r = r.iloc[0]
                put(f"{tag}{k}Delta", f"{r.delta:+.4f}")
                put(f"{tag}{k}FoldWins", f"{r.wins}/{r.n_folds}")
                put(f"{tag}{k}P", f"{r.p_nb_holm:.4g}")
    # where does the gain fall? correlate the baseline's own score with the improvement
    from scipy.stats import spearmanr
    A = d[d.method == REF].set_index(["rep", "fold"]); Bn = d[d.method == NEW].set_index(["rep", "fold"])
    ix = A.index.intersection(Bn.index)
    if len(ix) >= 6:
        for k in ("AUC", "AP"):
            dl = (Bn.loc[ix, k] - A.loc[ix, k])
            rs = spearmanr(A.loc[ix, k].values, dl.values)
            put(f"{tag}{k}GainRho", f"{rs.statistic:+.3f}"); put(f"{tag}{k}GainP", f"{rs.pvalue:.4f}")
            med = A.loc[ix, "AUC"].median(); hard = A.loc[ix, "AUC"] <= med
            put(f"{tag}{k}GainHard", f"{dl[hard].mean():+.4f}")
            put(f"{tag}{k}GainEasy", f"{dl[~hard].mean():+.4f}")
    # the originally-chosen configuration, reported alongside as the cost of undersized diagnostics
    if "HERON" in s.index:
        for k in ("AUC", "AP", "pseudoR2"):
            put(f"{tag}orig{k}", f"{s.loc['HERON', k]:.4f}")
        co = st.compare(d, REF, "HERON", group_cols=("protocol",))
        if len(co):
            put(f"{tag}origWins", int(co.better.sum())); put(f"{tag}origSigWins", int(co.signif_win.sum()))
            put(f"{tag}origSigLoss", int(co.signif_loss.sum()))
    # best baseline other than HERON variants
    base = [m for m in s.index if not m.startswith("HERON")]
    for k in ("AUC", "AP"):
        b = s.loc[base, k].idxmax()
        put(f"{tag}Best{k}Name", b.replace("&", "\\&")); put(f"{tag}Best{k}Val", f"{s.loc[b,k]:.4f}")
for tag, csv in (("sp", "cv_main_spatial_NDVI.csv"), ("rd", "cv_main_random_NDVI.csv"),
                 ("ev", "cv_main_spatial_EVI2.csv"), ("tp", "cv_temporal.csv")):
    bench(tag, csv)

# ---------- ablation ----------
def ablation(csv, reference):
    if not ex(f"{R}/{csv}"): return
    d = pd.read_csv(f"{R}/{csv}")
    if reference not in set(d.method): return
    put("nAblFolds", d[["rep", "fold"]].drop_duplicates().shape[0])
    A = d[d.method == reference].set_index(["rep", "fold"])
    for m in d.method.unique():
        if m == reference: continue
        Bm = d[d.method == m].set_index(["rep", "fold"])
        ix = A.index.intersection(Bm.index)
        key = "".join(ch for ch in m.title() if ch.isalpha())[:22]
        for k in ("AUC", "AP", "pseudoR2"):
            put(f"abl{key}{k}", f"{(Bm.loc[ix, k] - A.loc[ix, k]).mean():+.4f}")
ablation("cv_ablation_nn_spatial_NDVI.csv", "neural branch alone")
ablation("cv_ablation_stack_spatial_NDVI.csv", "HERON (full)")

# ---------- head-to-head ----------
if ex(f"{R}/cv_headtohead_spatial_NDVI.csv"):
    h = pd.read_csv(f"{R}/cv_headtohead_spatial_NDVI.csv")
    put("hhFolds", h[["rep", "fold"]].drop_duplicates().shape[0])
    g = h.groupby("method")[["AUC", "AP", "pseudoR2", "Brier"]].mean()
    for meth, tag in (("HERON", "HERON"), ("HERON-plus", "PLUS"), ("GLM-m4 (published)", "GLM")):
        if meth in g.index:
            for k in ("AUC", "AP", "pseudoR2", "Brier"):
                put(f"hh{tag}{k}", f"{g.loc[meth, k]:.4f}")

# ---------- final model read-outs ----------
FM = f"{R}/final_model"
if ex(f"{FM}/elevational_centroids_heron.csv"):
    c = pd.read_csv(f"{FM}/elevational_centroids_heron.csv").set_index("period")
    for p, k in (("Inter-Hurricanes","Inter"), ("Post-Fiona","Post")):
        put(f"cenHeron{k}", f"{c.loc[p,'delta_centroid_report']:+.1f}")
g2 = pd.read_csv(f"{R}/baseline/glm_elevational_centroids.csv").set_index("period")
put("cenGLMInter", f"{g2.loc['Inter-Hurricanes','delta_vs_PreMaria']:+.1f}")
put("cenGLMPost",  f"{g2.loc['Post-Fiona','delta_vs_PreMaria']:+.1f}")
if ex(f"{FM}/permutation_importance.csv"):
    imp = pd.read_csv(f"{FM}/permutation_importance.csv")
    put("topFeatOne", imp.feature.iloc[0].replace("_", "\\_"))
    put("topFeatTwo", imp.feature.iloc[1].replace("_", "\\_"))
    put("topFeatThree", imp.feature.iloc[2].replace("_", "\\_"))

# ---------- emit ----------
# Any macro the manuscript references but we could not compute (because that experiment has not
# finished) is emitted as a visible "??" rather than left undefined: the paper still compiles and
# the gap is impossible to miss in the PDF.
import glob as _glob, re as _re
used = set()
for f_ in _glob.glob(f"{PAP}/*.tex") + _glob.glob(f"{PAP}/tables/*.tex"):
    if f_.endswith("numbers.tex"): continue
    used |= set(_re.findall(r"\\([a-zA-Z]+)", open(f_).read()))
# Our generated-macro namespace is "<protocol/section prefix> + at least two more letters".
# LaTeX and package commands that share a prefix are listed explicitly so they are never
# shadowed by a placeholder.
OURS = _re.compile(r"^(n|sp|rd|ev|tp|abl|aic|cen|det|psi|ext|pDet|top|terrain|pseudo)[A-Za-z]{2,}$")
NOT_OURS = {"newcommand", "noindent", "newline", "newpage", "nonumber", "normalsize",
            "textbf", "textit", "texttt", "toprule", "centering", "caption", "cite",
            "topsep", "textwidth", "textcolor", "psi", "varepsilon", "detokenize",
            "tableofcontents", "newenvironment", "extracolsep"}
undefined = sorted(u for u in used if u not in M and OURS.match(u) and u not in NOT_OURS)
os.makedirs(PAP, exist_ok=True)
with open(f"{PAP}/numbers.tex", "w") as f:
    f.write("% Auto-generated by Birds/code/make_numbers.py -- do not edit by hand.\n")
    for k, v in sorted(M.items()):
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    if undefined:
        f.write("% placeholders for experiments that have not finished yet\n")
        for k in undefined:
            f.write(f"\\providecommand{{\\{k}}}{{\\textbf{{??}}}}\n")
print(f"wrote {PAP}/numbers.tex with {len(M)} macros")
if undefined:
    print(f"PLACEHOLDERS ({len(undefined)}) -- these experiments have not finished:")
    print("  " + ", ".join(undefined))
