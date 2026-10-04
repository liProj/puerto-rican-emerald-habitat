"""Merge the per-queue CV result files into the canonical names the figures and tables expect."""
import os, glob, pandas as pd
R = "Birds/results"
MERGE = {
    "cv_main_spatial_NDVI.csv":  ["cv_main_spatial_NDVI_base.csv", "cv_main_spatial_NDVI_heron.csv",
                                  "cv_main_spatial_NDVI_plus.csv"],
    "cv_main_random_NDVI.csv":   ["cv_main_random_NDVI_raw.csv", "cv_main_random_NDVI_plus.csv"],
    "cv_main_spatial_EVI2.csv":  ["cv_main_spatial_EVI2.csv"],
    "cv_temporal.csv":           ["cv_temporal_raw.csv", "cv_temporal_plus.csv"],
    "cv_ablation_spatial_NDVI.csv": ["cv_ablation_nn_spatial_NDVI.csv",
                                     "cv_ablation_stack_spatial_NDVI.csv"],
}
for out, parts in MERGE.items():
    have = [f"{R}/{p}" for p in parts if os.path.exists(f"{R}/{p}")]
    if not have:
        print(f"  {out}: no inputs yet"); continue
    if len(have) == 1 and os.path.basename(have[0]) == out:
        print(f"  {out}: already canonical ({len(pd.read_csv(have[0]))} rows)"); continue
    d = pd.concat([pd.read_csv(h) for h in have], ignore_index=True)
    d.to_csv(f"{R}/{out}", index=False)
    print(f"  {out}: {len(d)} rows from {[os.path.basename(h) for h in have]}")
# block-size sweep is written as several _bs_*.csv files
bs = sorted(glob.glob(f"{R}/_bs_*.csv"))
if bs:
    pd.concat([pd.read_csv(b) for b in bs], ignore_index=True).to_csv(f"{R}/cv_blocksize_NDVI.csv", index=False)
    print(f"  cv_blocksize_NDVI.csv: merged {len(bs)} block sizes")
