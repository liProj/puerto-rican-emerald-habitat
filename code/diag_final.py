"""Decisive diagnostic: final HERON configuration vs the strongest baselines, four spatial folds."""
import sys; sys.path.insert(0,'Birds/code')
import numpy as np, pandas as pd, data_prep as dp, heron, baselines as B, time
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss
from sklearn.isotonic import IsotonicRegression
d   = dp.build('NDVI','complete'); d['Elevation_100m']=d.Elevation_m/100
dall= dp.build('NDVI','all')
y, ya = d.y.values, dall.y.values
g  = pd.factorize(d.site.astype(str)+'|'+d.Period.astype(str))[0]
ga = pd.factorize(dall.site.astype(str)+'|'+dall.Period.astype(str))[0]
f  = dp.spatial_folds(d,   10,0.08,0)[0]
fa = dp.spatial_folds(dall,10,0.08,0)[0]
FOLDS=[0,2,5,9]
Xp_e={t: dp.psi_features(d,None,t) for t in (True,False)}
Xp_a={t: dp.psi_features(dall,None,t) for t in (True,False)}
Xd_e,Xe_e = dp.det_features(d), dp.effort_raw(d)
Xd_a,Xe_a = dp.det_features(dall), dp.effort_raw(dall)
K=dict(epochs=350, batch=16384, gamma=0, pos_weight=1)
def ev(tag, fn):
    A,P,L,Br=[],[],[],[]; t=time.time()
    for k in FOLDS:
        p=fn(k); te=f==k
        A.append(roc_auc_score(y[te],p)); P.append(average_precision_score(y[te],p))
        L.append(log_loss(y[te],np.clip(p,1e-7,1-1e-7),labels=[0,1])); Br.append(brier_score_loss(y[te],p))
    print(f'{tag:46s} AUC={np.mean(A):.4f} AP={np.mean(P):.4f} LL={np.mean(L):.4f} Brier={np.mean(Br):.5f} {time.time()-t:4.0f}s', flush=True)
def hf(terrain=True, use_all=True, n_ens=1, calib=True, **kw):
    def fn(k):
        if use_all: m=fa!=k; args=(Xp_a[terrain][m],Xd_a[m],Xe_a[m],ya[m],ga[m],fa[m])
        else:       m=f!=k;  args=(Xp_e[terrain][m],Xd_e[m],Xe_e[m],y[m],g[m],f[m])
        ps=[heron.fit_heron(*args,seed=s,**{**K,**kw})[1](Xp_e[terrain],Xd_e,Xe_e) for s in range(n_ens)]
        p=np.mean(ps,0)
        if calib:
            i=np.where(f!=k)[0]; np.random.default_rng(0).shuffle(i); ci=i[int(.85*len(i)):]
            p=np.clip(IsotonicRegression(out_of_bounds='clip').fit(p[ci],y[ci]).predict(p),1e-7,1-1e-7)
        return p[f==k]
    return fn
for nm in ['GLM-m4 (published)','GLM-splines','XGBoost + terrain','LightGBM + terrain']:
    ev(nm, lambda k,nm=nm: B.BASELINES[nm](d[f!=k],d[f==k],seed=0)[0])
ev('HERON  no terrain, complete-case only', hf(terrain=False, use_all=False, lam_group=0.1, lam_vrex=1.0))
ev('HERON  +terrain, complete-case only',   hf(terrain=True,  use_all=False, lam_group=0.1, lam_vrex=1.0))
ev('HERON  +terrain, all-data',             hf(terrain=True,  use_all=True,  lam_group=0.1, lam_vrex=1.0))
ev('HERON  +terrain, all-data, no aux loss',hf(terrain=True,  use_all=True,  lam_group=0.0, lam_vrex=0.0))
ev('HERON  FULL (+terrain, all, 5x ens)',   hf(terrain=True,  use_all=True,  lam_group=0.1, lam_vrex=1.0, n_ens=5))
