"""Multi-scale terrain descriptors from the SRTM raster shipped inside the authors' package.

The published model uses point elevation only. Puerto Rican montane forest -- the habitat this
hummingbird depends on -- is structured by terrain form as much as by height: slope controls
soil moisture and forest type, local relief controls exposure to hurricane wind, and topographic
position separates ridges from sheltered valleys. These descriptors are derived entirely from
the raster the authors released, and unlike raw coordinates they transfer to unseen regions,
so they remain usable under spatial blocking.
"""
import numpy as np, pandas as pd, rasterio, os
from scipy.ndimage import uniform_filter, maximum_filter, minimum_filter

TIF = ("Birds/supp/birds-07-00055-s001/Riccordia_maugaeus_reproducibility_package/"
       "01 Riccordia_preprocessing_environmental_covariates/03_study_area_topography_map/"
       "data/Puerto_Rico_SRTM_30m.tif")
CACHE = "Birds/results/terrain_features.csv.gz"
RADII = [3, 9, 27, 81]          # pixels ~ 90 m, 270 m, 810 m, 2.4 km

def _layers(z, px_m):
    gy, gx = np.gradient(z.astype(np.float32), px_m[0], px_m[1])
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    aspect = np.arctan2(-gx, gy)
    out = {"srtm_elev": z.astype(np.float32), "slope_deg": slope,
           "northness": np.cos(aspect).astype(np.float32),
           "eastness": np.sin(aspect).astype(np.float32)}
    zf = z.astype(np.float32)
    for r in RADII:
        k = 2 * r + 1
        mu = uniform_filter(zf, k, mode="nearest")
        mu2 = uniform_filter(zf * zf, k, mode="nearest")
        out[f"tpi_{r}"] = (zf - mu)                                  # ridge (+) vs valley (-)
        out[f"sd_{r}"] = np.sqrt(np.clip(mu2 - mu * mu, 0, None))    # roughness
        out[f"relief_{r}"] = (maximum_filter(zf, k, mode="nearest")
                              - minimum_filter(zf, k, mode="nearest"))
        out[f"slope_mu_{r}"] = uniform_filter(slope, k, mode="nearest")
    return out

def build(lon, lat, force=False):
    """Sample terrain descriptors at the given coordinates (nearest 30-m cell)."""
    with rasterio.open(TIF) as s:
        z = s.read(1).astype(np.float32)
        z[z < -100] = np.nan
        z = np.where(np.isnan(z), np.nanmedian(z), z)
        tr = s.transform
        # metres per pixel at this latitude (EPSG:4326 raster)
        lat0 = float((s.bounds.top + s.bounds.bottom) / 2)
        py = abs(tr.e) * 111320.0
        px = abs(tr.a) * 111320.0 * np.cos(np.radians(lat0))
        L = _layers(z, (py, px))
        rows, cols = rasterio.transform.rowcol(tr, np.asarray(lon), np.asarray(lat))
        rows = np.clip(np.asarray(rows), 0, s.height - 1)
        cols = np.clip(np.asarray(cols), 0, s.width - 1)
    return pd.DataFrame({k: v[rows, cols] for k, v in L.items()})

FEATS = (["srtm_elev", "slope_deg", "northness", "eastness"]
         + [f"{p}_{r}" for r in RADII for p in ("tpi", "sd", "relief", "slope_mu")])

def attach(d, cache=CACHE):
    """Attach terrain features to a checklist table, caching by sampling-event identifier."""
    if os.path.exists(cache):
        t = pd.read_csv(cache)
        if set(d.SAMPLINGEVENTIDENTIFIER) <= set(t.SAMPLINGEVENTIDENTIFIER):
            return d.merge(t, on="SAMPLINGEVENTIDENTIFIER", how="left")
    return None

if __name__ == "__main__":
    import sys; sys.path.insert(0, "Birds/code"); from common import load
    d = load("NDVI")                       # full 89,682; cache covers every subset
    T = build(d.Longitude.values, d.Latitude.values)
    T.insert(0, "SAMPLINGEVENTIDENTIFIER", d.SAMPLINGEVENTIDENTIFIER.values)
    os.makedirs("Birds/results", exist_ok=True); T.to_csv(CACHE, index=False)
    r = np.corrcoef(d.Elevation_m.values, T.srtm_elev.values)[0, 1]
    print(f"cached {len(T)} rows x {len(FEATS)} terrain features -> {CACHE}")
    print(f"cross-check: raster-sampled elevation vs the authors' GEE Elevation_m  r = {r:.5f}")
    print(f"  mean abs difference = {np.abs(d.Elevation_m.values - T.srtm_elev.values).mean():.2f} m")
    print(T[FEATS].describe().T[["mean","std","min","max"]].round(2).to_string())
