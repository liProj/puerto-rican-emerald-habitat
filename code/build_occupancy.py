"""Build the 1044 x 9 dynamic-occupancy detection history exactly as Sec 2.7 specifies."""
import pandas as pd, numpy as np, sys, os
sys.path.insert(0, "Birds/code"); from common import load, PERIODS

os.makedirs("Birds/results/baseline", exist_ok=True)
def build(index="NDVI", order="chrono"):
    d = load(index)
    d["row"] = np.arange(len(d))                       # original row order in the analytical dataset
    ps = d.groupby("site", observed=True).Period.nunique()
    elig = set(ps[ps == 3].index)                      # >=1 checklist in every period
    e = d[d.site.isin(elig)].copy()
    # site-level means over ALL eligible checklists 2013-2025, missing excluded
    sm = e.groupby("site").agg(elev=("Elevation_m", "mean"), veg=("veg", "mean"),
                               n_veg=("veg", lambda s: s.notna().sum()))
    final = sorted(sm[sm.n_veg > 0].index)             # drop sites with no valid index value
    sm = sm.loc[final]
    e = e[e.site.isin(final)].copy()
    # within site & period: order by year then original row order; keep first 3 as secondary occasions
    e = e.sort_values(["site", "Period", "year", "row"], kind="mergesort")
    e["occ"] = e.groupby(["site", "Period"], observed=True).cumcount()
    e = e[e.occ < 3]
    Y = np.full((len(final), 9), np.nan)
    # "chrono" = Pre-Maria, Inter-Huracanes, Post-Fiona (temporal).
    # "alpha"  = R's default factor ordering of the Spanish labels, which is what the
    #           authors' Inter-Hurricanes detection reference level implies.
    cols = PERIODS if order == "chrono" else sorted(PERIODS)
    pi = {p: i for i, p in enumerate(cols)}
    si = {s: i for i, s in enumerate(final)}
    for r in e.itertuples():
        Y[si[r.site], pi[str(r.Period)] * 3 + r.occ] = r.Presence
    z = lambda v: (v - v.mean()) / v.std(ddof=1)
    sc = pd.DataFrame(dict(site=final, elev_z=z(sm.elev.values), veg_z=z(sm.veg.values),
                           elev_m=sm.elev.values, veg_raw=sm.veg.values))
    print(f"[{index}/{order}] eligible={len(elig)} dropped={len(elig)-len(final)} final={len(final)} "
          f"observed={int(np.isfinite(Y).sum())} missing={int(np.isnan(Y).sum())}")
    pd.DataFrame(Y, columns=[f"y{i+1}" for i in range(9)]).to_csv(f"Birds/results/baseline/occ_y_{index}_{order}.csv", index=False)
    sc.to_csv(f"Birds/results/baseline/occ_sitecovs_{index}_{order}.csv", index=False)
    return Y, sc

for ix in ("NDVI", "EVI2"):
    for od in ("chrono", "alpha"): build(ix, od)
print("column orders:", "chrono =", PERIODS, "| alpha =", sorted(PERIODS))
