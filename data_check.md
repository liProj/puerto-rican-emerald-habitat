# Data verification — MDPI *Birds* rank-1 paper

**Target paper.** Ortiz-Andrade, B.M.; Rojas-Perez, A. *Elevation and Vegetation Greenness Structure Post-Hurricane Resilience and Local Persistence of the Puerto Rican Emerald (Riccordia maugaeus).* **Birds** 2026, 7, 55. doi:10.3390/birds7030055

**Public data.** MDPI Supplementary `birds-07-00055-s001.zip` → `Riccordia_maugaeus_reproducibility_package/` (61 MB unpacked): three derived checklist CSVs (91,677 rows x 12 cols), the authors' R analysis scripts, the Google Earth Engine extraction scripts, a Puerto Rico SRTM GeoTIFF, a data dictionary, `validation_targets.csv`, and `SHA256SUMS.txt`.

The original eBird *source* files are not redistributable under the eBird terms, but the **derived, model-ready datasets that every reported analysis uses are fully released**. This is the decisive difference from the rejected counter-example (a paper reporting 233 patients whose public workbook held 107 rows / 96 unique IDs with duplicates and missing values).

## 1. Cryptographic integrity

`sha256sum -c SHA256SUMS.txt` → **16/16 files OK**. The released bytes are exactly the bytes the authors checksummed.

## 2. Count verification

**53 of 54 checks reproduce exactly.** All eight hard gating metrics (sample size, class balance, unique IDs, missing values, duplicate rows, analytical subset size, occupancy site count) match to the unit.

| # | Check | Paper / authors report | Recomputed from public data | Match |
|---|---|---|---|---|
| 1 | [VT] Filtered pre-GEE table: total rows | `91677` | `91677` | ✅ |
| 2 | [VT] Filtered pre-GEE table: rows 2013-2025 | `89682` | `89682` | ✅ |
| 3 | [VT] Filtered pre-GEE table: detections (full) | `5326` | `5326` | ✅ |
| 4 | [VT] Filtered pre-GEE table: non-detections (full) | `86351` | `86351` | ✅ |
| 5 | [VT] Corrected NDVI export: total rows | `91677` | `91677` | ✅ |
| 6 | [VT] Corrected NDVI export: rows 2013-2025 | `89682` | `89682` | ✅ |
| 7 | [VT] Corrected NDVI export: detections (full) | `5326` | `5326` | ✅ |
| 8 | [VT] Corrected NDVI export: non-detections (full) | `86351` | `86351` | ✅ |
| 9 | [VT] Corrected NDVI export: non-missing index (full) | `70799` | `70799` | ✅ |
| 10 | [VT] EVI2 export: total rows | `91677` | `91677` | ✅ |
| 11 | [VT] EVI2 export: rows 2013-2025 | `89682` | `89682` | ✅ |
| 12 | [VT] EVI2 export: detections (full) | `5326` | `5326` | ✅ |
| 13 | [VT] EVI2 export: non-detections (full) | `86351` | `86351` | ✅ |
| 14 | [VT] EVI2 export: non-missing index (full) | `70799` | `70799` | ✅ |
| 15 | paper 3.1: checklists analysed 2013-2025 | `89682` | `89682` | ✅ |
| 16 | paper 3.1: checklists reporting species | `5174` | `5174` | ✅ |
| 17 | paper 3.1: checklists not reporting | `84508` | `84508` | ✅ |
| 18 | paper 3.1: overall encounter frequency (%) | `5.77` | `5.77` | ✅ |
| 19 | Table 1 Pre-Maria: total checklists | `14728` | `14728` | ✅ |
| 20 | Table 1 Pre-Maria: reporting species | `1118` | `1118` | ✅ |
| 21 | Table 1 Pre-Maria: encounter frequency (%) | `7.59` | `7.59` | ✅ |
| 22 | Table 1 Inter-Hurricanes: total checklists | `37701` | `37701` | ✅ |
| 23 | Table 1 Inter-Hurricanes: reporting species | `1742` | `1742` | ✅ |
| 24 | Table 1 Inter-Hurricanes: encounter frequency (%) | `4.62` | `4.62` | ✅ |
| 25 | Table 1 Post-Fiona: total checklists | `37253` | `37253` | ✅ |
| 26 | Table 1 Post-Fiona: reporting species | `2314` | `2314` | ✅ |
| 27 | Table 1 Post-Fiona: encounter frequency (%) | `6.21` | `6.21` | ✅ |
| 28 | paper 3.1: Pre-Maria share (%) | `16.4` | `16.4` | ✅ |
| 29 | paper 3.1: Inter-Hurricanes share (%) | `42.0` | `42.0` | ✅ |
| 30 | paper 3.1: Post-Fiona share (%) | `41.5` | `41.5` | ✅ |
| 31 | paper 3.1: checklists with missing NDVI | `20706` | `20706` | ✅ |
| 32 | paper 3.1: missing-NDVI share (%) | `23.1` | `23.1` | ✅ |
| 33 | paper 2.4.2/3.1: NDVI-complete analytical subset | `68976` | `68976` | ✅ |
| 34 | EVI2 subset size (parity with NDVI) | `68976` | `68976` | ✅ |
| 35 | paper 2.1/Fig1: unique sampling locations | `24876` | `24876` | ✅ |
| 36 | paper 2.1/Fig1: unique locations with >=1 report | `1900` | `1900` | ✅ |
| 37 | integrity: unique sampling-event IDs == rows (full) | `91677` | `91677` | ✅ |
| 38 | integrity: fully duplicated rows (full) | `0` | `0` | ✅ |
| 39 | integrity: duplicated (date,lon,lat) triples | `0` | `2` | ❌ |
| 40 | integrity: missing values in core covariates | `0` | `0` | ✅ |
| 41 | integrity: Presence coded {0,1} only | `True` | `True` | ✅ |
| 42 | integrity: Duration within 5-240 min filter | `True` | `True` | ✅ |
| 43 | integrity: Distance <= 5 km filter | `True` | `True` | ✅ |
| 44 | integrity: Elevation_m >= 0 (negatives set to 0) | `True` | `True` | ✅ |
| 45 | integrity: NDVI within [-1,1] | `True` | `True` | ✅ |
| 46 | integrity: |Days_from_eBird| < 17 (fractional days) | `True` | `True` | ✅ |
| 47 | integrity: NDVI/EVI2 exports row-aligned on IDs | `True` | `True` | ✅ |
| 48 | paper 2.7: initially eligible sites (>=1 checklist per period) | `1048` | `1048` | ✅ |
| 49 | paper 2.7: sites dropped for no valid NDVI | `4` | `4` | ✅ |
| 50 | paper 2.7: final occupancy sites | `1044` | `1044` | ✅ |
| 51 | paper 2.7: observed survey occasions | `7908` | `7908` | ✅ |
| 52 | paper 2.7: missing occasions in 1044x9 matrix | `1488` | `1488` | ✅ |
| 53 | paper 3.5: Inter-Hurricanes time-since-Maria N | `28707` | `28707` | ✅ |
| 54 | paper 3.5: Post-Fiona time-since-Fiona N | `29580` | `29580` | ✅ |

## 3. The single deviation

**`duplicated (date, lon, lat) triples`: reported rule implies 0, observed 2.** Section 2.2 states that one sampling event was retained per exact (date, latitude, longitude) combination. Two pairs survive:

- `S63350828` / `S62993590` — 2020-01-04, (−67.130112, 17.995121), durations 207 min, 3.862 / 3.770 km
- `S129185355` / `S129201334` — 2023-02-20, (−65.771295, 18.352603), durations 35 / 65 min, 0 km

This is **2 rows out of 91,677 (0.0022%)**, all four are non-detections, and it changes none of the published counts — every reported N, class count, percentage, subset size and site count above reproduces exactly *including* these rows. The most likely cause is that the authors de-duplicated on a coordinate representation with different precision from the exported one. Recorded as an immaterial, documented deviation.

## 4. Two resolved apparent mismatches (not data errors)

- **Period labels.** The released data uses `Inter-Huracanes` (Spanish); the paper renders it `Inter-Hurricanes`. Same stratum, identical counts (37,701 / 1,742 / 4.62%).
- **`Days_from_eBird` reaches 16.62.** The paper specifies a ±16-*day* Landsat window. The field stores a *fractional*-day difference including the Landsat acquisition time of day, so a record exactly 16 calendar days away can read 16.62. All 91,677 values satisfy |Δ| < 17; no record violates the stated calendar window.

- **Unique-location count depends on which export is used.** The pre-GEE table stores coordinates
to 7 decimals and yields the published 24,876 unique locations; the NDVI/EVI2 exports come back
from Google Earth Engine with 14 decimals but values nudged by up to 6x10^-6 deg (~0.7 m), which
merges 20 pairs and gives 24,856. Figure 1's own script uses the pre-GEE table, so the published
figure is consistent. The shift is far below the 30-m pixel and changes no extracted covariate.

- **Occupancy occasion count.** The 7,908 observed / 1,488 missing occasions in the 1044x9 detection matrix reproduce exactly when the grid is built from the folder-02 analysis file the paper actually uses (`Riccordia_maugaeus_eBird_NDVI_corrected_nearest.csv`).

## 5. Verdict

**PASS.** The dataset is completely public, checksum-verified, internally consistent, free of duplicate identifiers and of missing values in every core covariate, and reproduces 53/54 published quantities exactly. Proceed to baseline reproduction and new-method experiments.


Reproduce with: `./.venv/bin/python Birds/code/verify_data.py` (writes `Birds/results/data_check.csv`).
