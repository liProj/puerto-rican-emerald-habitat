"""Reproduce the checklist-level binomial GLMs of Ortiz-Andrade & Rojas-Perez (Birds 2026, 7, 55).
Targets: Table 1, Sec 3.3 AIC/weights/pseudo-R2, Sec 3.3 odds ratios, Sec 3.4 predictions,
Fig. 4 elevational centroids, Sec 3.5 secondary temporal GLMs, Sec 3.7 EVI2 sensitivity."""
import pandas as pd, numpy as np, statsmodels.api as sm, statsmodels.formula.api as smf
import json, os, sys
sys.path.insert(0, "Birds/code"); from common import load, complete, PER_EN, MARIA, FIONA

os.makedirs("Birds/results/baseline", exist_ok=True)
OUT, rep = {}, []
def cmp_(name, reported, observed, tol=None):
    ok = abs(reported - observed) <= (tol if tol is not None else max(1e-9, abs(reported) * 1e-4))
    rep.append(dict(quantity=name, paper=reported, reproduced=round(float(observed), 6), match=bool(ok)))

FORMS = {
 "m0": "Presence ~ 1",
 "m1": "Presence ~ C(Period)",
 "m2": "Presence ~ C(Period) + Duration_min + Distance_km",
 "m3": "Presence ~ C(Period) + Elevation_100m + veg + Duration_min + Distance_km",
 "m4": "Presence ~ C(Period) * Elevation_100m + veg + Duration_min + Distance_km",
}

def fit_set(d, tag):
    a = complete(d)
    fits = {k: smf.glm(f, data=a, family=sm.families.Binomial()).fit() for k, f in FORMS.items()}
    aic = pd.Series({k: v.aic for k, v in fits.items()})
    dl = aic - aic.min(); w = np.exp(-dl / 2); w = w / w.sum()
    ll0 = fits["m0"].llf
    tab = pd.DataFrame(dict(model=aic.index, formula=[FORMS[k] for k in aic.index],
                            k=[fits[m].df_model + 1 for m in aic.index], logLik=[fits[m].llf for m in aic.index],
                            AIC=aic.values, dAIC=dl.values, weight=w.values,
                            pseudoR2=[1 - fits[m].llf / ll0 for m in aic.index])).sort_values("AIC")
    tab.to_csv(f"Birds/results/baseline/glm_selection_{tag}.csv", index=False)
    return a, fits, tab

# ---------------- Table 1 ----------------
d = load("NDVI")
t1 = []
for p in ["Pre-Maria", "Inter-Huracanes", "Post-Fiona"]:
    g = d[d.Period == p]
    t1.append(dict(period=PER_EN[p], N=len(g), n=int(g.Presence.sum()), pct=round(100 * g.Presence.mean(), 2)))
t1.append(dict(period="Total dataset", N=len(d), n=int(d.Presence.sum()), pct=round(100 * d.Presence.mean(), 2)))
pd.DataFrame(t1).to_csv("Birds/results/baseline/table1.csv", index=False)
for r, (N, n, pc) in zip(t1, [(14728,1118,7.59),(37701,1742,4.62),(37253,2314,6.21),(89682,5174,5.77)]):
    cmp_(f"Table 1 {r['period']} N", N, r["N"]); cmp_(f"Table 1 {r['period']} n", n, r["n"])
    cmp_(f"Table 1 {r['period']} %", pc, r["pct"], tol=0.005)

# ---------------- GLM selection (NDVI) ----------------
a, fits, tab = fit_set(d, "NDVI")
OUT["n_analytical"] = len(a)
cmp_("Sec 2.6 analytical subset N", 68976, len(a))
print(tab.to_string(index=False))
cmp_("Sec 3.3 best model AIC (m4)", 21373.31, fits["m4"].aic, tol=0.02)
cmp_("Sec 3.3 m4 Akaike weight", 0.776, float(tab.set_index("model").loc["m4","weight"]), tol=0.002)
cmp_("Sec 3.3 m3 dAIC", 2.49, float(tab.set_index("model").loc["m3","dAIC"]), tol=0.02)
cmp_("Sec 3.3 m3 Akaike weight", 0.224, float(tab.set_index("model").loc["m3","weight"]), tol=0.002)
cmp_("Sec 3.3 m4 McFadden pseudo-R2", 0.207, float(tab.set_index("model").loc["m4","pseudoR2"]), tol=0.0006)

m4 = fits["m4"]
co = pd.DataFrame(dict(term=m4.params.index, beta=m4.params.values, se=m4.bse.values,
                       z=m4.tvalues.values, p=m4.pvalues.values, OR=np.exp(m4.params.values)))
co.to_csv("Birds/results/baseline/glm_m4_coefficients.csv", index=False)
print("\n", co.to_string(index=False))

# total elevation effect per 100 m by period (Sec 3.3: +43%, +48.3%, +50.0%)
b = m4.params
slope = {"Pre-Maria": b["Elevation_100m"],
         "Inter-Huracanes": b["Elevation_100m"] + b["C(Period)[T.Inter-Huracanes]:Elevation_100m"],
         "Post-Fiona": b["Elevation_100m"] + b["C(Period)[T.Post-Fiona]:Elevation_100m"]}
for p, tgt in [("Pre-Maria", 43.0), ("Inter-Huracanes", 48.3), ("Post-Fiona", 50.0)]:
    cmp_(f"Sec 3.3 odds change per 100 m, {PER_EN[p]} (%)", tgt, 100 * (np.exp(slope[p]) - 1), tol=0.6)

# ---------------- Sec 3.4 predictions + Fig. 4 centroids ----------------
grid = np.arange(0, 1301, 10)
pred = {}
for p in ["Pre-Maria", "Inter-Huracanes", "Post-Fiona"]:
    nd = pd.DataFrame(dict(Period=pd.Categorical([p]*len(grid), categories=d.Period.cat.categories),
                           Elevation_100m=grid/100.0, veg=a.veg.mean(),
                           Duration_min=a.Duration_min.mean(), Distance_km=a.Distance_km.mean()))
    pred[p] = m4.predict(nd).values
pdf = pd.DataFrame({"Elevation_m": grid, **{PER_EN[k]: v for k, v in pred.items()}})
pdf.to_csv("Birds/results/baseline/glm_predictions_by_elevation.csv", index=False)
for p, e, tgt in [("Pre-Maria",100,3.77),("Pre-Maria",900,40.58),("Inter-Huracanes",100,2.40),
                  ("Inter-Huracanes",900,36.52),("Post-Fiona",100,3.08),("Post-Fiona",900,44.89)]:
    cmp_(f"Sec 3.4 predicted reporting prob, {PER_EN[p]} @{e} m (%)", tgt,
         100*pred[p][list(grid).index(e)], tol=0.02)
cen = {p: float((grid*pred[p]).sum()/pred[p].sum()) for p in pred}
pd.DataFrame([dict(period=PER_EN[p], centroid_m=round(cen[p],2),
                   delta_vs_PreMaria=round(cen[p]-cen["Pre-Maria"],2)) for p in pred]
            ).to_csv("Birds/results/baseline/glm_elevational_centroids.csv", index=False)
cmp_("Fig. 4 centroid shift, Inter-Hurricanes (m)", 28.5, cen["Inter-Huracanes"]-cen["Pre-Maria"], tol=0.06)
cmp_("Fig. 4 centroid shift, Post-Fiona (m)", 13.9, cen["Post-Fiona"]-cen["Pre-Maria"], tol=0.06)

# ---------------- Sec 3.5 secondary temporal GLMs ----------------
sec = []
for lab, per, t0, tgt in [("Inter-Hurricanes","Inter-Huracanes",MARIA,(28707,7716,7724,7.98,0.982,0.0265,0.00839,0.0016)),
                          ("Post-Fiona","Post-Fiona",FIONA,(29580,9497,9498,1.02,0.376,0.0457,0.0294,0.121))]:
    s = a[a.Period == per].copy()
    s["yrs"] = (s.Date - t0).dt.days / 365.25
    f_add = smf.glm("Presence ~ yrs + Elevation_100m + veg + Duration_min + Distance_km",
                    data=s, family=sm.families.Binomial()).fit()
    f_int = smf.glm("Presence ~ yrs * Elevation_100m + veg + Duration_min + Distance_km",
                    data=s, family=sm.families.Binomial()).fit()
    dd = f_int.aic - f_add.aic; wts = np.exp(-np.array([0, abs(dd)])/2)
    sec.append(dict(period=lab, N=len(s), AIC_additive=f_add.aic, AIC_interaction=f_int.aic,
                    dAIC=abs(dd), beta_yrs_add=f_add.params["yrs"], se_yrs_add=f_add.bse["yrs"],
                    p_yrs_add=f_add.pvalues["yrs"], beta_int=f_int.params["yrs:Elevation_100m"],
                    se_int=f_int.bse["yrs:Elevation_100m"], p_int=f_int.pvalues["yrs:Elevation_100m"]))
    cmp_(f"Sec 3.5 {lab} N", tgt[0], len(s))
    if lab == "Inter-Hurricanes":
        cmp_(f"Sec 3.5 {lab} AIC interaction", tgt[1], f_int.aic, tol=0.6)
        cmp_(f"Sec 3.5 {lab} AIC additive", tgt[2], f_add.aic, tol=0.6)
        cmp_(f"Sec 3.5 {lab} dAIC", tgt[3], abs(dd), tol=0.06)
        cmp_(f"Sec 3.5 {lab} interaction beta", tgt[5], f_int.params["yrs:Elevation_100m"], tol=6e-4)
        cmp_(f"Sec 3.5 {lab} interaction SE", tgt[6], f_int.bse["yrs:Elevation_100m"], tol=6e-5)
        cmp_(f"Sec 3.5 {lab} interaction p", tgt[7], f_int.pvalues["yrs:Elevation_100m"], tol=6e-4)
    else:
        cmp_(f"Sec 3.5 {lab} AIC additive", tgt[1], f_add.aic, tol=0.6)
        cmp_(f"Sec 3.5 {lab} AIC interaction", tgt[2], f_int.aic, tol=0.6)
        cmp_(f"Sec 3.5 {lab} additive beta(yrs)", tgt[5], f_add.params["yrs"], tol=6e-4)
        cmp_(f"Sec 3.5 {lab} additive SE(yrs)", tgt[6], f_add.bse["yrs"], tol=6e-5)
        cmp_(f"Sec 3.5 {lab} additive p(yrs)", tgt[7], f_add.pvalues["yrs"], tol=6e-4)
pd.DataFrame(sec).to_csv("Birds/results/baseline/glm_secondary_temporal.csv", index=False)

# ---------------- Sec 3.7 EVI2 sensitivity ----------------
de = load("EVI2"); ae, fe, te = fit_set(de, "EVI2")
cmp_("Sec 3.7 EVI2 checklist AIC", 21462.08, fe["m4"].aic, tol=0.02)
cmp_("Sec 3.7 checklist dAIC NDVI vs EVI2", 88.76, fe["m4"].aic - fits["m4"].aic, tol=0.03)

R = pd.DataFrame(rep)
R.to_csv("Birds/results/baseline/reproduction_check_glm.csv", index=False)
print("\n" + R.to_string(index=False))
print(f"\n=== GLM reproduction: {int(R.match.sum())}/{len(R)} quantities match ===")
if (~R.match).any(): print(R[~R.match].to_string(index=False))
