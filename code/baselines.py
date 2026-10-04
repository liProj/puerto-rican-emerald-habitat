"""Competing models for the checklist-level reporting-probability benchmark.

B1/B2 are the published models (m4 / m3): exactly the specification of Birds 2026, 7, 55.
B3 adds the natural statistical upgrade (splines) a careful reviewer would ask for.
B4-B7 are strong modern tabular learners. All see the same features a GLM could see;
only HERON adds the spatial Fourier block, the factorised head and the mask.
"""
import numpy as np, pandas as pd, statsmodels.api as sm, statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, SplineTransformer
import lightgbm as lgb, xgboost as xgb
import sys; sys.path.insert(0, "Birds/code")
import terrain as TR

PAPER_M4 = "y ~ C(Period) * Elevation_100m + veg_f + Duration_min + Distance_km"
PAPER_M3 = "y ~ C(Period) + Elevation_100m + veg_f + Duration_min + Distance_km"
# Splines are built with sklearn's SplineTransformer (linear extrapolation) rather than
# patsy bs(), which refuses to predict outside the training knots -- a real problem under
# spatial blocking, where a held-out region can exceed the training elevation range.
SPLINE_VARS = [("Elevation_100m", 6), ("veg_f", 6), ("Duration_min", 5), ("Distance_km", 4)]

def _glm(form):
    def f(dtr, dte, seed=0):
        m = smf.glm(form, data=dtr, family=sm.families.Binomial()).fit()
        return np.clip(m.predict(dte).values, 1e-7, 1 - 1e-7), m
    return f

def _glm_terrain(dtr, dte, seed=0):
    """Published m4 plus the SRTM terrain block, entered linearly. Isolates how much of the
    improvement is available to the authors' own model class once terrain is supplied."""
    ts = " + ".join(TR.FEATS)
    return _glm(PAPER_M4 + " + " + ts)(dtr, dte, seed)

def _glm_spline(dtr, dte, seed=0):
    """Published m4 structure with smooth terms: the natural statistical upgrade of the baseline.
    Period x elevation interaction is retained by multiplying the elevation basis by period dummies."""
    def design(d, sts):
        cols = [np.ones((len(d), 1))]
        for (v, _), st in zip(SPLINE_VARS, sts):
            Bv = st.transform(d[[v]].values)
            cols.append(Bv)
            if v == "Elevation_100m":                      # Period x smooth(elevation)
                for pc in ("per_1", "per_2"):
                    cols.append(Bv * d[pc].values[:, None])
        cols += [d.per_1.values[:, None], d.per_2.values[:, None]]
        return np.hstack(cols)
    sts = [SplineTransformer(n_knots=k, degree=3, extrapolation="linear",
                             include_bias=False).fit(dtr[[v]].values) for v, k in SPLINE_VARS]
    Xtr, Xte = design(dtr, sts), design(dte, sts)
    m = sm.GLM(dtr.y.values, Xtr, family=sm.families.Binomial()).fit_regularized(alpha=1e-4, L1_wt=0.0)
    return np.clip(m.predict(Xte), 1e-7, 1 - 1e-7), m

def tab_features(d, terrain=False):
    """Flat feature matrix for the tree/NN baselines. `terrain=True` gives a baseline exactly the
    same SRTM-derived terrain block HERON sees, so the ablation can separate the contribution of
    the new features from the contribution of the new architecture."""
    cols = [d.Elevation_m.values, d.veg_f.values, d.veg_missing.values,
            d.Duration_min.values, d.Distance_km.values,
            d.per_0.values, d.per_1.values, d.per_2.values,
            d.doy_sin.values, d.doy_cos.values,
            d.Longitude.values, d.Latitude.values]
    if terrain and TR.FEATS[0] in d.columns:
        cols += [d[c].values for c in TR.FEATS]
    return np.nan_to_num(np.column_stack(cols)).astype(np.float32)

def _lgb(dtr, dte, seed=0, terrain=False):
    m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=40,
                           subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                           random_state=seed, n_jobs=8, verbose=-1)
    m.fit(tab_features(dtr, terrain), dtr.y.values)
    return m.predict_proba(tab_features(dte, terrain))[:, 1], m

def _xgb(dtr, dte, seed=0, terrain=False):
    m = xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6, min_child_weight=5,
                          subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, tree_method="hist",
                          random_state=seed, n_jobs=8, eval_metric="logloss")
    m.fit(tab_features(dtr, terrain), dtr.y.values)
    return m.predict_proba(tab_features(dte, terrain))[:, 1], m

def _rf(dtr, dte, seed=0, terrain=False):
    m = RandomForestClassifier(n_estimators=400, min_samples_leaf=5, max_features="sqrt",
                               n_jobs=8, random_state=seed)
    m.fit(tab_features(dtr, terrain), dtr.y.values)
    return m.predict_proba(tab_features(dte, terrain))[:, 1], m

def _mlp(dtr, dte, seed=0, terrain=False):
    sc = StandardScaler().fit(tab_features(dtr, terrain))
    m = MLPClassifier(hidden_layer_sizes=(128, 128, 128), alpha=1e-4, batch_size=512,
                      learning_rate_init=1e-3, max_iter=80, random_state=seed, early_stopping=False)
    m.fit(sc.transform(tab_features(dtr, terrain)), dtr.y.values)
    return m.predict_proba(sc.transform(tab_features(dte, terrain)))[:, 1], m

BASELINES = {
    "GLM-m4 (published)": _glm(PAPER_M4),
    "GLM-m3 (published)": _glm(PAPER_M3),
    "GLM-splines":        _glm_spline,
    "RandomForest":       _rf,
    "LightGBM":           _lgb,
    "XGBoost":            _xgb,
    "MLP":                _mlp,
    "GLM-m4 + terrain":   _glm_terrain,
    "LightGBM + terrain": lambda a, b, seed=0: _lgb(a, b, seed, terrain=True),
    "XGBoost + terrain":  lambda a, b, seed=0: _xgb(a, b, seed, terrain=True),
    "MLP + terrain":      lambda a, b, seed=0: _mlp(a, b, seed, terrain=True),
}
