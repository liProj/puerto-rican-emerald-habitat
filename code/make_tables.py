"""Generate the paper's LaTeX tables from the results CSVs."""
import sys, os; sys.path.insert(0, "Birds/code")
import numpy as np, pandas as pd
from metrics import HIGHER_BETTER
import stats_tests as st

R, T = "Birds/results", "Birds/paper/tables"
os.makedirs(T, exist_ok=True)
def ex(p): return os.path.exists(p)
def esc(s):
    t = str(s).replace("&", "\\&").replace("_", "\\_").replace("%", "\\%")
    t = t.replace("~", "$\\sim$").replace("^", "\\^{}")
    # a cell beginning with "[" would be read as the optional argument of the preceding \\
    return ("{}" + t) if t.startswith("[") else t
ORDER = ["GLM-m3 (published)", "GLM-m4 (published)", "GLM-splines", "GLM-m4 + terrain",
         "RandomForest", "MLP", "MLP + terrain", "LightGBM", "LightGBM + terrain",
         "XGBoost", "XGBoost + terrain", "HERON-NN", "HERON", "HERON-plus"]

def tab_verify():
    d = pd.read_csv(f"{R}/data_check.csv")
    L = [r"\begin{longtable}{p{7.4cm}rrc}", r"\caption{Item-by-item verification of the released "
         r"dataset against every count reported by \cite{ortiz2026} and against the authors' own "
         r"\texttt{validation\_targets.csv}.\label{tab:verify}}\\", r"\toprule",
         r"Check & Reported & Recomputed & Match \\ \midrule", r"\endfirsthead",
         r"\toprule Check & Reported & Recomputed & Match \\ \midrule \endhead"]
    for _, r in d.iterrows():
        L.append(f"{esc(r['item'])} & {esc(r['reported'])} & {esc(r['observed'])} & "
                 f"{'yes' if r['match'] else r'\textbf{no}'} \\\\")
    L += [r"\bottomrule", r"\end{longtable}"]
    open(f"{T}/verify.tex", "w").write("\n".join(L))

def tab_glm():
    d = pd.read_csv(f"{R}/baseline/glm_selection_NDVI.csv")
    L = [r"\begin{table}[h]\centering\small", r"\caption{Reproduced checklist-level model "
         r"selection (published Section~3.3). All five models are fitted to the same "
         r"\nComplete{} NDVI-complete checklists.\label{tab:glmsel}}",
         r"\begin{tabular}{lp{7.6cm}rrrrr}\toprule",
         r"Model & Structure & $k$ & AIC & $\Delta$AIC & $w$ & pseudo-$R^2$ \\ \midrule"]
    SHORT = {"C(Period)": "Period", "Elevation_100m": "Elev", "veg": "NDVI",
             "Duration_min": "Dur", "Distance_km": "Dist", "Presence ~ ": ""}
    for _, r in d.iterrows():
        fm = str(r["formula"])
        for a_, b_ in SHORT.items(): fm = fm.replace(a_, b_)
        L.append(f"\\texttt{{{r['model']}}} & \\scriptsize\\texttt{{{esc(fm)}}} & "
                 f"{int(r['k'])} & {r['AIC']:.2f} & {r['dAIC']:.2f} & {r['weight']:.3f} & {r['pseudoR2']:.4f} \\\\")
    L += [r"\bottomrule\end{tabular}\end{table}"]
    open(f"{T}/glmsel.tex", "w").write("\n".join(L))

def tab_occ():
    if not ex(f"{R}/baseline/occupancy_ordering_test.csv"): return
    t = pd.read_csv(f"{R}/baseline/occupancy_ordering_test.csv")
    o = pd.read_csv(f"{R}/baseline/occupancy_vs_paper.csv")
    a = o[o["mode"] == "as_published"].reset_index(drop=True)
    f = t[t["mode"] == "site_major"].reset_index(drop=True)
    lab = {"psi": r"$\psi$", "col": r"$\gamma$", "ext": r"$\varepsilon$", "det": "$p$"}
    L = [r"\begin{table}[h]\centering\small",
         r"\caption{Dynamic occupancy model: the published Table~2, our run of the authors' own "
         r"script, and a refit that changes only the observation-covariate stacking order "
         r"(Section~\ref{sec:ordering}). Everything else---detection matrix, site covariates, "
         r"formulas, optimiser---is held at the authors' values.\label{tab:occ}}",
         r"\begin{tabular}{llrrrrrr}\toprule",
         r"& & \multicolumn{2}{c}{Published} & \multicolumn{2}{c}{Authors' script (ours)} & "
         r"\multicolumn{2}{c}{Corrected order} \\ \cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
         r"Comp. & Parameter & est. & SE & est. & SE & est. & $p$ \\ \midrule"]
    for i in range(len(a)):
        chg = r"\,$^\dagger$" if (a.p[i] < .05) != (f.p[i] < .05) else ""
        L.append(f"{lab[a.component[i]]} & {esc(a.parameter[i])}{chg} & {a.paper_est[i]:.3f} & "
                 f"{a.paper_SE[i]:.3f} & {a.est[i]:.4f} & {a.SE[i]:.4f} & "
                 f"{f.estimate[i]:.4f} & {f.p[i]:.3g} \\\\")
    ap_aic = t[t['mode']=='as_published'].AIC.iloc[0]; sm_aic = t[t['mode']=='site_major'].AIC.iloc[0]
    L += [r"\midrule", f"\\multicolumn{{2}}{{l}}{{AIC}} & \\multicolumn{{2}}{{c}}{{2878.81}} & "
          f"\\multicolumn{{2}}{{c}}{{{ap_aic:.4f}}} & \\multicolumn{{2}}{{c}}{{{sm_aic:.4f}}} \\\\",
          r"\bottomrule\end{tabular}", r"\\[2pt]\footnotesize $^\dagger$ statistical significance "
          r"changes between the published stacking order and the corrected one.", r"\end{table}"]
    open(f"{T}/occ.tex", "w").write("\n".join(L))

def tab_bench(csv="cv_main_spatial_NDVI.csv", out="bench.tex", proto="spatially blocked"):
    if not ex(f"{R}/{csv}"): return
    d = pd.read_csv(f"{R}/{csv}")
    ms = [m for m in ORDER if m in set(d.method)]
    keys = ["AUC", "AP", "pseudoR2", "TjurR2", "Brier", "LogLoss", "ECE", "Recall_at_10pct"]
    ref = "GLM-m4 (published)"
    sig = {}
    for m in ms:
        if m == ref: continue
        c = st.compare(d, ref, m, metrics=keys, group_cols=("protocol",))
        sig[m] = {r.metric: ("*" if r.signif_win else ("$-$" if r.signif_loss else ""))
                  for _, r in c.iterrows()}
    g = d.groupby("method")
    L = [r"\begin{table}[h]\centering\footnotesize",
         rf"\caption{{Benchmark under {proto} cross-validation ({d[['rep','fold']].drop_duplicates().shape[0]} folds). "
         r"Mean $\pm$ SD across folds. $*$ marks a Holm-corrected Nadeau--Bengio improvement over "
         r"the published \texttt{m4} at $p<0.05$; $-$ marks a significant loss."
         rf"\label{{tab:{out[:-4]}}}}}",
         r"\begin{tabular}{l" + "r" * len(keys) + r"}\toprule",
         "Method & " + " & ".join(k.replace("_", " ").replace("pseudoR2", "pseudo-$R^2$")
                                   .replace("TjurR2", "Tjur $R^2$") for k in keys) + r" \\",
         r"& " + " & ".join(("$\\uparrow$" if HIGHER_BETTER[k] > 0 else "$\\downarrow$") for k in keys) + r" \\ \midrule"]
    for m in ms:
        cells = []
        for k in keys:
            v, s = g[k].mean()[m], g[k].std()[m]
            cells.append(f"{v:.4f}{sig.get(m, {}).get(k, '')}")
        nm = esc(m)
        if m == "HERON-plus": nm = r"\textbf{HERON-plus}"
        if m == ref: nm = esc(m) + r"\,$\ddagger$"
        L.append(nm + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule\end{tabular}",
          r"\\[2pt]\footnotesize $\ddagger$ the model published by \cite{ortiz2026}.", r"\end{table}"]
    open(f"{T}/{out}", "w").write("\n".join(L))

def tab_ablation(csv="cv_ablation_nn_spatial_NDVI.csv", full="neural branch alone",
                 out="ablation.tex", what="neural branch"):
    if not ex(f"{R}/{csv}"): return
    d = pd.read_csv(f"{R}/{csv}")
    if full not in set(d.method): return
    keys = ["AUC", "AP", "pseudoR2", "Brier"]
    rows = []
    for m in d.method.unique():
        A = d[d.method == full].set_index(["rep", "fold"]); Bm = d[d.method == m].set_index(["rep", "fold"])
        ix = A.index.intersection(Bm.index)
        r = dict(variant=m)
        for k in keys:
            r[k] = Bm.loc[ix, k].mean()
            r[f"d{k}"] = (Bm.loc[ix, k] - A.loc[ix, k]).mean()
        rows.append(r)
    t = pd.DataFrame(rows)
    t = pd.concat([t[t.variant == full], t[t.variant != full].sort_values("dAP")])
    L = [r"\begin{table}[h]\centering\footnotesize",
         rf"\caption{{Ablation of the {what} under spatially blocked cross-validation "
         rf"({d[['rep','fold']].drop_duplicates().shape[0]} folds). Each row differs from the "
         r"reference row in exactly one respect; $\Delta$ is the change relative to it, so a "
         r"negative $\Delta$ means the change hurts and the component therefore helps."
         rf"\label{{tab:{out[:-4]}}}}}",
         r"\begin{tabular}{lrrrrrrrr}\toprule",
         r"Variant & AUC & $\Delta$ & AP & $\Delta$ & pseudo-$R^2$ & $\Delta$ & Brier & $\Delta$ \\ \midrule"]
    for _, r in t.iterrows():
        nm = r"\textbf{" + esc(r.variant) + "}" if r.variant == full else esc(r.variant)
        L.append(f"{nm} & {r.AUC:.4f} & {'' if r.variant==full else f'{r.dAUC:+.4f}'} & "
                 f"{r.AP:.4f} & {'' if r.variant==full else f'{r.dAP:+.4f}'} & "
                 f"{r.pseudoR2:.4f} & {'' if r.variant==full else f'{r.dpseudoR2:+.4f}'} & "
                 f"{r.Brier:.5f} & {'' if r.variant==full else f'{r.dBrier:+.5f}'} \\\\")
    L += [r"\bottomrule\end{tabular}\end{table}"]
    open(f"{T}/{out}", "w").write("\n".join(L))

def tab_screen():
    s = pd.read_csv("Birds/screening.csv").head(10)
    L = [r"\begin{table}[h]\centering\footnotesize",
         r"\caption{Top ten of the 50 most recent \textit{Birds} articles after screening. "
         r"Score $=0.4\,s_{\mathrm{data}}+0.3\,s_{\mathrm{time}}+0.2\,s_{\mathrm{ease}}+0.1\,s_{\mathrm{figs}}$."
         r"\label{tab:screen}}",
         r"\begin{tabular}{rlp{5.4cm}p{4.2cm}r}\toprule",
         r"\# & DOI suffix & Title (truncated) & Data availability & Score \\ \midrule"]
    for _, r in s.iterrows():
        L.append(f"{int(r['rank'])} & {esc(str(r['doi']).split('/')[-1])} & "
                 f"{esc(str(r['title'])[:78])} & {esc(str(r['data_availability'])[:58])} & {r['score']:.1f} \\\\")
    L += [r"\bottomrule\end{tabular}\end{table}"]
    open(f"{T}/screen.tex", "w").write("\n".join(L))

if __name__ == "__main__":
    tab_verify(); tab_glm(); tab_occ(); tab_screen()
    tab_bench("cv_main_spatial_NDVI.csv", "bench.tex", "spatially blocked")
    tab_bench("cv_main_random_NDVI.csv", "bench_random.tex", "random $k$-fold")
    tab_bench("cv_main_spatial_EVI2.csv", "bench_evi2.tex", "spatially blocked (EVI2 index)")
    tab_bench("cv_temporal.csv", "bench_temporal.tex", "temporal extrapolation")
    tab_ablation()
    tab_ablation("cv_ablation_stack_spatial_NDVI.csv", "HERON (full)",
                 "ablation_stack.tex", "stack members")
    tab_bench("cv_headtohead_spatial_NDVI.csv", "bench_headtohead.tex", "spatially blocked (head-to-head)")
    print("tables ->", sorted(os.listdir(T)))
