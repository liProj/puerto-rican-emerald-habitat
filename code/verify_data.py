"""Hard verification of the birds-07-00055 public dataset against every count
reported in the paper and in the authors' own validation_targets.csv."""
import pandas as pd, numpy as np, json, os, sys

PKG = ("Birds/supp/birds-07-00055-s001/Riccordia_maugaeus_reproducibility_package")
F_PRE  = f"{PKG}/01 Riccordia_preprocessing_environmental_covariates/01_data_preparation/output/Datos_eBird_filtrados_pre_GEE.csv"
F_NDVI = f"{PKG}/02 Riccordia_main_analysis/data/Riccordia_maugaeus_eBird_NDVI_corrected_nearest.csv"
F_EVI2 = f"{PKG}/03_Riccordia_NDVI_EVI2_sensitivity_analysis/Riccordia_maugaeus_eBird_EVI2_nearest.csv"

pre  = pd.read_csv(F_PRE)
ndvi = pd.read_csv(F_NDVI)
evi2 = pd.read_csv(F_EVI2)
for d in (pre, ndvi, evi2):
    d["Date"] = pd.to_datetime(d["Date"])

# analysis window used throughout the paper
def win(d): return d[(d.Date >= "2013-01-01") & (d.Date <= "2025-12-31")].copy()
S, N, E = win(pre), win(ndvi), win(evi2)

checks = []
def ck(item, reported, observed, src):
    ok = (reported == observed) if not isinstance(reported, float) else abs(reported - observed) < 5e-3
    checks.append(dict(item=item, reported=reported, observed=observed, match=bool(ok), source=src))

# ---- authors' validation_targets.csv (full table, no date window) ----
vt = pd.read_csv(f"{PKG}/01 Riccordia_preprocessing_environmental_covariates/documentation/validation_targets.csv")
for nm, d in [("Filtered pre-GEE table", pre), ("Corrected NDVI export", ndvi), ("EVI2 export", evi2)]:
    row = vt[vt.Dataset == nm].iloc[0]
    ck(f"[VT] {nm}: total rows", int(row.Total_rows), len(d), "validation_targets.csv")
    ck(f"[VT] {nm}: rows 2013-2025", int(row.Rows_2013_2025), len(win(d)), "validation_targets.csv")
    ck(f"[VT] {nm}: detections (full)", int(row.Detections_full), int(d.Presence.sum()), "validation_targets.csv")
    ck(f"[VT] {nm}: non-detections (full)", int(row.Non_detections_full), int((d.Presence == 0).sum()), "validation_targets.csv")
    if not pd.isna(row.Nonmissing_index_full):
        idx = "NDVI_corrected" if "NDVI" in nm else "EVI2"
        ck(f"[VT] {nm}: non-missing index (full)", int(row.Nonmissing_index_full), int(d[idx].notna().sum()), "validation_targets.csv")

# ---- paper Section 3.1 + Table 1 ----
ck("paper 3.1: checklists analysed 2013-2025", 89682, len(S), "Sec 3.1 / Abstract")
ck("paper 3.1: checklists reporting species", 5174, int(S.Presence.sum()), "Sec 3.1")
ck("paper 3.1: checklists not reporting", 84508, int((S.Presence == 0).sum()), "Sec 3.1")
ck("paper 3.1: overall encounter frequency (%)", 5.77, round(100 * S.Presence.mean(), 2), "Sec 3.1")
PLAB = {"Pre-Maria": "Pre-Maria", "Inter-Hurricanes": "Inter-Huracanes", "Post-Fiona": "Post-Fiona"}
T1 = {"Pre-Maria": (14728, 1118, 7.59), "Inter-Hurricanes": (37701, 1742, 4.62), "Post-Fiona": (37253, 2314, 6.21)}
for per, (n, k, pc) in T1.items():
    g = S[S.Period == PLAB[per]]
    ck(f"Table 1 {per}: total checklists", n, len(g), "Table 1")
    ck(f"Table 1 {per}: reporting species", k, int(g.Presence.sum()), "Table 1")
    ck(f"Table 1 {per}: encounter frequency (%)", pc, round(100 * g.Presence.mean(), 2), "Table 1")
ck("paper 3.1: Pre-Maria share (%)", 16.4, round(100 * len(S[S.Period == "Pre-Maria"]) / len(S), 1), "Sec 3.1")
ck("paper 3.1: Inter-Hurricanes share (%)", 42.0, round(100 * len(S[S.Period == "Inter-Huracanes"]) / len(S), 1), "Sec 3.1")
ck("paper 3.1: Post-Fiona share (%)", 41.5, round(100 * len(S[S.Period == "Post-Fiona"]) / len(S), 1), "Sec 3.1")

# ---- NDVI availability (Sec 2.4.2 + 3.1) ----
ck("paper 3.1: checklists with missing NDVI", 20706, int(N.NDVI_corrected.isna().sum()), "Sec 3.1")
ck("paper 3.1: missing-NDVI share (%)", 23.1, round(100 * N.NDVI_corrected.isna().mean(), 1), "Sec 3.1")
ck("paper 2.4.2/3.1: NDVI-complete analytical subset", 68976, int(N.NDVI_corrected.notna().sum()), "Sec 2.4.2, 3.1")
ck("EVI2 subset size (parity with NDVI)", 68976, int(E.EVI2.notna().sum()), "Sec 2.8 parity")

# ---- Figure 1 spatial counts (Sec 2.1) ----
loc = S.drop_duplicates(subset=["Longitude", "Latitude"])
ck("paper 2.1/Fig1: unique sampling locations", 24876, len(loc), "Sec 2.1 / Fig. 1")
det_loc = S[S.Presence == 1].drop_duplicates(subset=["Longitude", "Latitude"])
ck("paper 2.1/Fig1: unique locations with >=1 report", 1900, len(det_loc), "Sec 2.1 / Fig. 1")

# ---- integrity: duplicates / IDs / missingness ----
ck("integrity: unique sampling-event IDs == rows (full)", len(ndvi), ndvi.SAMPLINGEVENTIDENTIFIER.nunique(), "data integrity")
ck("integrity: fully duplicated rows (full)", 0, int(ndvi.duplicated().sum()), "data integrity")
ck("integrity: duplicated (date,lon,lat) triples", 0, int(ndvi.duplicated(subset=["Date","Longitude","Latitude"]).sum()), "Sec 2.2 dedup rule")
core = ["SAMPLINGEVENTIDENTIFIER","Date","Period","Presence","Elevation_m","Longitude","Latitude","Duration_min","Distance_km"]
ck("integrity: missing values in core covariates", 0, int(ndvi[core].isna().sum().sum()), "data integrity")
ck("integrity: Presence coded {0,1} only", True, set(ndvi.Presence.unique()) <= {0,1}, "data dictionary")
ck("integrity: Duration within 5-240 min filter", True, bool(S.Duration_min.between(5,240).all()), "Sec 2.2 filter")
ck("integrity: Distance <= 5 km filter", True, bool(S.Distance_km.le(5).all()), "Sec 2.2 filter")
ck("integrity: Elevation_m >= 0 (negatives set to 0)", True, bool((ndvi.Elevation_m >= 0).all()), "Sec 2.4.1")
ck("integrity: NDVI within [-1,1]", True, bool(N.NDVI_corrected.dropna().between(-1,1).all()), "Sec 2.4.2")
ck("integrity: |Days_from_eBird| < 17 (fractional days)", True, bool(N.Days_from_eBird.dropna().abs().lt(17).all()), "Sec 2.4.2")
ck("integrity: NDVI/EVI2 exports row-aligned on IDs", True,
   bool((ndvi.SAMPLINGEVENTIDENTIFIER.values == evi2.SAMPLINGEVENTIDENTIFIER.values).all()), "Sec 2.8 parity")

# ---- occupancy design (Sec 2.7) ----
N["site"] = N.Longitude.round(2).astype(str) + "_" + N.Latitude.round(2).astype(str)
per_site = N.groupby("site").Period.nunique()
elig = per_site[per_site == 3].index
ck("paper 2.7: initially eligible sites (>=1 checklist per period)", 1048, len(elig), "Sec 2.7")
nd_site = N[N.site.isin(elig)].groupby("site").NDVI_corrected.apply(lambda s: s.notna().sum())
final = sorted(nd_site[nd_site > 0].index)
ck("paper 2.7: sites dropped for no valid NDVI", 4, len(elig) - len(final), "Sec 2.7")
ck("paper 2.7: final occupancy sites", 1044, len(final), "Sec 2.7")
sub = N[N.site.isin(final)].copy()
occ_rows = 0
for (_, _), g in sub.groupby(["site", "Period"]):
    occ_rows += min(len(g), 3)
ck("paper 2.7: observed survey occasions", 7908, occ_rows, "Sec 2.7")
ck("paper 2.7: missing occasions in 1044x9 matrix", 1488, 1044 * 9 - occ_rows, "Sec 2.7")

# ---- secondary temporal analysis sample sizes (Sec 3.5) ----
ni = N[(N.Period == "Inter-Huracanes") & N.NDVI_corrected.notna()]
nf = N[(N.Period == "Post-Fiona") & N.NDVI_corrected.notna() & (N.Date <= "2025-12-31")]
ck("paper 3.5: Inter-Hurricanes time-since-Maria N", 28707, len(ni), "Sec 3.5")
ck("paper 3.5: Post-Fiona time-since-Fiona N", 29580, len(nf), "Sec 3.5")

out = pd.DataFrame(checks)
os.makedirs("Birds/results", exist_ok=True)
out.to_csv("Birds/results/data_check.csv", index=False)
n_ok = int(out.match.sum())
print(out.to_string(index=False))
print(f"\n=== {n_ok}/{len(out)} checks match ===")
if n_ok < len(out):
    print("\nMISMATCHES:"); print(out[~out.match].to_string(index=False))
