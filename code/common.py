"""Shared data loading for the birds-07-00055 reproduction and the new method."""
import pandas as pd, numpy as np, os

PKG = "Birds/supp/birds-07-00055-s001/Riccordia_maugaeus_reproducibility_package"
F_NDVI = f"{PKG}/02 Riccordia_main_analysis/data/Riccordia_maugaeus_eBird_NDVI_corrected_nearest.csv"
F_EVI2 = f"{PKG}/03_Riccordia_NDVI_EVI2_sensitivity_analysis/Riccordia_maugaeus_eBird_EVI2_nearest.csv"
# the released data labels the middle period in Spanish; the paper renders it in English
PERIODS = ["Pre-Maria", "Inter-Huracanes", "Post-Fiona"]
PER_EN = {"Pre-Maria": "Pre-Maria", "Inter-Huracanes": "Inter-Hurricanes", "Post-Fiona": "Post-Fiona"}
MARIA = pd.Timestamp("2017-09-20")
FIONA = pd.Timestamp("2022-09-18")

def load(index="NDVI"):
    """Checklist table restricted to the paper's 2013-2025 analysis window."""
    f = F_NDVI if index == "NDVI" else F_EVI2
    d = pd.read_csv(f)
    d["Date"] = pd.to_datetime(d["Date"])
    d = d[(d.Date >= "2013-01-01") & (d.Date <= "2025-12-31")].reset_index(drop=True)
    d["veg"] = d["NDVI_corrected"] if index == "NDVI" else d["EVI2"]
    d["Elevation_100m"] = d["Elevation_m"] / 100.0
    d["Period"] = pd.Categorical(d["Period"], categories=PERIODS, ordered=False)
    d["year"] = d.Date.dt.year
    d["site"] = d.Longitude.round(2).astype(str) + "_" + d.Latitude.round(2).astype(str)
    return d

def complete(d):
    """The paper's analytical subset: complete cases on the vegetation index."""
    return d[d.veg.notna()].reset_index(drop=True)
