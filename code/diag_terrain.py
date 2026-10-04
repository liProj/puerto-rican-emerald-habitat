"""Second diagnostic: do the SRTM terrain features and the auxiliary losses help, and does
training on the full missingness-included table beat the published complete-case fit?"""
import sys; sys.path.insert(0,'Birds/code')
import numpy as np, pandas as pd, data_prep as dp, heron, baselines as B, time
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
from sklearn.isotonic import IsotonicRegression

d   = dp.build('NDVI','complete'); d['Elevation_100m']=d.Elevation_m/100
dall= dp.build('NDVI','all');      dall['Elevation_100m']=dall.Elevation_m/100
y, yall = d.y.values, dall.y.values
grp     = pd.factorize(d.site.astype(str)+'|'+d.Period.astype(str))[0]
grpall  = pd.factorize(dall.site.astype(str)+'|'+dall.Period.astype(str))[0]
FOLDS=[0,2,5,9]
F     = {r: dp.spatial_folds(d,   10,0.08,r)[0] for r in (0,)}
Fall  = {r: dp.spatial_folds(dall,10,0.08,r)[0] for r in (0,)}
f, fall = F[0], Fall[0]

def ev(tag, fn):
    A,P,L=[],[],[]; t=time.time()
    for k in FOLDS:
        p=fn(k); te=f==k
        A.append(roc_auc_score(y[te],p)); P.append(average_precision_score(y[te],p))
        L.append(log_loss(y[te],np.clip(p,1e-7,1-1e-7),labels=[0,1]))
    print(f'{tag:52s} AUC={np.mean(A):.4f} AP={np.mean(P):.4f} LL={np.mean(L):.4f} {time.time()-t:4.0f}s', flush=True)

def bl(nm, terr):
    return lambda k: B.BASELINES[nm](d[f!=k], d[f==k], seed=0)[0]

for nm in ['GLM-m4 (published)','GLM-m4 + terrain','LightGBM + terrain','XGBoost + terrain']:
    ev(nm, bl(nm, True))

K=dict(epochs=350, batch=16384, lam_group=0.5, lam_vrex=1.0, gamma=0, pos_weight=1)
def heron_fn(terrain=True, use_all=False, n_ens=1, calib=True, **kw):
    Xp_e = dp.psi_features(d, None, terrain); Xd_e = dp.det_features(d); Xe_e = dp.effort_raw(d)
    Xp_a = dp.psi_features(dall, None, terrain); Xd_a = dp.det_features(dall); Xe_a = dp.effort_raw(dall)
    def fn(k):
        if use_all:
            m_ = fall != k
            args = (Xp_a[m_], Xd_a[m_], Xe_a[m_], yall[m_], grpall[m_], fall[m_])
        else:
            m_ = f != k
            args = (Xp_e[m_], Xd_e[m_], Xe_e[m_], y[m_], grp[m_], f[m_])
        ps=[]
        for s in range(n_ens):
            _,pred = heron.fit_heron(*args, seed=s, **{**K, **kw})
            ps.append(pred(Xp_e, Xd_e, Xe_e))
        p=np.mean(ps,0)
        if calib:
            idx=np.where(f!=k)[0]; rs=np.random.default_rng(0); rs.shuffle(idx); ci=idx[int(.85*len(idx)):]
            p=np.clip(IsotonicRegression(out_of_bounds='clip').fit(p[ci], y[ci]).predict(p),1e-7,1-1e-7)
        return p[f==k]
    return fn

ev('HERON  no terrain',                       heron_fn(terrain=False))
ev('HERON  +terrain',                         heron_fn(terrain=True))
ev('HERON  +terrain, no group/vrex',          heron_fn(terrain=True, lam_group=0, lam_vrex=0))
ev('HERON  +terrain, all-data training',      heron_fn(terrain=True, use_all=True))
ev('HERON  +terrain, all-data, 3x ensemble',  heron_fn(terrain=True, use_all=True, n_ens=3))
