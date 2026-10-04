"""Paired significance testing across cross-validation folds.

Repeated k-fold folds overlap in their training data, so the naive paired t-test is
anti-conservative. We use the Nadeau-Bengio corrected resampled t-test, which inflates the
variance by the train/test overlap ratio, alongside a distribution-free Wilcoxon signed-rank
test, and control the family-wise error rate across metrics with Holm's step-down procedure.
"""
import numpy as np, pandas as pd
from scipy import stats
from metrics import HIGHER_BETTER

def nadeau_bengio(diff, n_train, n_test):
    d = np.asarray(diff, float); n = len(d)
    if n < 2 or np.allclose(d.std(), 0): return np.nan, np.nan
    var = d.var(ddof=1) * (1.0 / n + n_test / max(1.0, n_train))
    t = d.mean() / np.sqrt(max(var, 1e-300))
    return t, 2 * stats.t.sf(abs(t), df=n - 1)

def holm(p):
    p = np.asarray(p, float); m = len(p); out = np.full(m, np.nan)
    ok = ~np.isnan(p); idx = np.where(ok)[0]
    o = idx[np.argsort(p[idx])]; run = 0.0
    for i, j in enumerate(o):
        run = max(run, (len(o) - i) * p[j]); out[j] = min(1.0, run)
    return out

def compare(df, ref, new, metrics=None, group_cols=("protocol", "index")):
    """Paired per-fold comparison of `new` against `ref` for every metric."""
    metrics = metrics or list(HIGHER_BETTER)
    rows = []
    for key, g in df.groupby(list(group_cols)):
        a = g[g.method == ref].set_index(["rep", "fold"])
        b = g[g.method == new].set_index(["rep", "fold"])
        common = a.index.intersection(b.index)
        if len(common) < 3: continue
        a, b = a.loc[common], b.loc[common]
        n_test = a.n_test.mean(); n_train = a.n_test.sum() - n_test
        for m in metrics:
            if m not in a: continue
            s = HIGHER_BETTER[m]
            d = s * (b[m].values - a[m].values)          # positive = new method better
            t, p_t = nadeau_bengio(d, n_train, n_test)
            try: p_w = stats.wilcoxon(d, zero_method="wilcox").pvalue
            except Exception: p_w = np.nan
            rows.append(dict(zip(group_cols, key if isinstance(key, tuple) else (key,))) | dict(
                metric=m, ref=ref, new=new, n_folds=len(common),
                ref_mean=a[m].mean(), ref_sd=a[m].std(ddof=1),
                new_mean=b[m].mean(), new_sd=b[m].std(ddof=1),
                delta=d.mean(), delta_sd=d.std(ddof=1),
                wins=int((d > 0).sum()), losses=int((d < 0).sum()),
                t_nb=t, p_nb=p_t, p_wilcoxon=p_w, better=bool(d.mean() > 0)))
    R = pd.DataFrame(rows)
    if len(R):
        for key, g in R.groupby(list(group_cols)):
            R.loc[g.index, "p_nb_holm"] = holm(g.p_nb.values)
            R.loc[g.index, "p_wil_holm"] = holm(g.p_wilcoxon.values)
        R["signif_win"]  = R.better & (R.p_nb_holm < 0.05)
        R["signif_loss"] = (~R.better) & (R.p_nb_holm < 0.05)
    return R

def summarise(df, metrics=None, group_cols=("protocol", "index")):
    metrics = metrics or list(HIGHER_BETTER)
    g = df.groupby(list(group_cols) + ["method"])
    out = g[metrics].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.join(g.size().rename("n_folds")).reset_index()
