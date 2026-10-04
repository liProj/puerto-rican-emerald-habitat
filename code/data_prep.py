"""Feature construction and evaluation protocols for the Birds 7,55 benchmark.

Three protocols:
  spatial : GroupKFold over contiguous geographic blocks -- held-out regions are unseen.
            This is the honest protocol for a species-distribution model; random splits
            leak because the same locality is revisited hundreds of times.
  random  : repeated stratified K-fold (the conventional, optimistic protocol).
  temporal: train on Pre-Maria + Inter-Hurricanes, test on Post-Fiona (forecasting).
"""
import numpy as np, pandas as pd, sys, os
sys.path.insert(0, "Birds/code"); from common import load, complete, PERIODS
import terrain as TR

EFFORT = ["Duration_min", "Distance_km"]
ENV    = ["Elevation_m", "veg", "veg_missing"]

def build(index="NDVI", subset="complete", with_terrain=True):
    d = load(index)
    if with_terrain and os.path.exists(TR.CACHE):
        d = d.merge(pd.read_csv(TR.CACHE), on="SAMPLINGEVENTIDENTIFIER", how="left")
    d["veg_missing"] = d.veg.isna().astype(float)
    if subset == "complete":
        d = complete(d).copy()                 # the paper's analytical subset
    d = d.reset_index(drop=True)
    doy = d.Date.dt.dayofyear.values
    d["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    d["days_veg"] = d.Days_from_eBird.fillna(0.0).values
    d["veg_f"] = d.veg.fillna(d.veg.mean()).values   # mean-fill; the mask carries the information
    for i, p in enumerate(PERIODS):
        d[f"per_{i}"] = (d.Period == p).astype(float)
    d["y"] = d.Presence.astype(float).values
    return d

# ---- feature blocks ----
def psi_features(d, rff=None, terrain=True):
    """Occupancy branch: environment + terrain form + period. No effort covariates."""
    base = [d.Elevation_m.values / 1000.0, d.veg_f.values, d.veg_missing.values,
            d.per_0.values, d.per_1.values, d.per_2.values]
    if terrain and TR.FEATS[0] in d.columns:
        base += [d[c].values / sc for c, sc in zip(TR.FEATS, TERRAIN_SCALE)]
    X = np.column_stack(base)
    if rff is not None:
        X = np.column_stack([X, rff])
    return np.nan_to_num(X).astype(np.float32)

# fixed divisors keep every terrain column on a comparable scale before standardisation
TERRAIN_SCALE = [1000.0, 30.0, 1.0, 1.0] + [v for _ in TR.RADII for v in (50.0, 40.0, 200.0, 30.0)]

def det_features(d):
    """Detection branch: period + seasonality + vegetation-match quality. Effort is separate."""
    return np.column_stack([d.per_0.values, d.per_1.values, d.per_2.values,
                            d.doy_sin.values, d.doy_cos.values,
                            d.days_veg.values / 16.0]).astype(np.float32)

def effort_raw(d):
    """Raw effort, passed through a hard-monotone path so detection increases with effort."""
    return np.column_stack([d.Duration_min.values, d.Distance_km.values]).astype(np.float32)

class RFF:
    """Multi-scale random Fourier features of (lon, lat): a stationary GP-like spatial prior.
    Scales are in degrees; small scales capture local clustering, large ones regional gradients."""
    def __init__(self, scales=(0.02, 0.05, 0.12, 0.35, 1.0), n_per_scale=32, seed=0):
        rng = np.random.default_rng(seed)
        self.W = np.concatenate([rng.normal(0, 1.0 / s, size=(2, n_per_scale)) for s in scales], axis=1)
        self.b = rng.uniform(0, 2 * np.pi, size=self.W.shape[1])
        self.scales, self.n_per_scale = scales, n_per_scale
    def __call__(self, d):
        X = np.column_stack([d.Longitude.values, d.Latitude.values])
        z = X @ self.W + self.b
        return np.column_stack([np.cos(z), np.sin(z)]).astype(np.float32) * np.sqrt(2.0 / self.W.shape[1])

# ---- protocols ----
def spatial_folds(d, n_folds=10, block_deg=0.08, seed=0):
    """Assign contiguous geographic blocks to folds; all checklists in a block share a fold."""
    rng = np.random.default_rng(seed)
    off_x, off_y = rng.uniform(0, block_deg, 2)
    bx = np.floor((d.Longitude.values + off_x) / block_deg).astype(int)
    by = np.floor((d.Latitude.values + off_y) / block_deg).astype(int)
    key = pd.Series([f"{a}_{b}" for a, b in zip(bx, by)])
    blocks = key.unique(); rng.shuffle(blocks)
    # greedy balancing: assign blocks (largest first) to the currently smallest fold
    size = key.value_counts()
    order = sorted(blocks, key=lambda b: -size[b])
    load_ = np.zeros(n_folds); assign = {}
    for b in order:
        j = int(np.argmin(load_)); assign[b] = j; load_[j] += size[b]
    f = key.map(assign).values
    return f, key.values

def random_folds(d, n_folds=5, seed=0):
    from sklearn.model_selection import StratifiedKFold
    f = np.empty(len(d), int)
    for j, (_, te) in enumerate(StratifiedKFold(n_folds, shuffle=True, random_state=seed).split(d, d.y)):
        f[te] = j
    return f

def temporal_split(d):
    tr = (d.Period != "Post-Fiona").values
    return tr, ~tr

if __name__ == "__main__":
    for ix in ("NDVI", "EVI2"):
        for sub in ("complete", "all"):
            d = build(ix, sub)
            print(f"{ix:5s} {sub:9s} n={len(d):6d} pos={int(d.y.sum()):5d} ({100*d.y.mean():.2f}%) "
                  f"veg_missing={int(d.veg_missing.sum()):5d} sites={d.site.nunique():5d}")
    d = build("NDVI", "complete")
    f, key = spatial_folds(d, 10, 0.08, 0)
    sz = pd.Series(f).value_counts().sort_index()
    print(f"\nspatial blocks: {len(np.unique(key))} | fold sizes {sz.min()}-{sz.max()} "
          f"| positive rate per fold {[round(100*d.y[f==j].mean(),2) for j in range(10)]}")
