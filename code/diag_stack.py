"""Does the stacked hybrid beat every single-family model on both AUC and AP?"""
import sys; sys.path.insert(0,'Birds/code')
import numpy as np, pandas as pd, time, data_prep as dp, baselines as B
from hybrid import HeronStack
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss
d  = dp.build('NDVI','complete'); d['Elevation_100m']=d.Elevation_m/100
da = dp.build('NDVI','all');      da['Elevation_100m']=da.Elevation_m/100
y=d.y.values
f  = dp.spatial_folds(d, 10,0.08,0)[0]
fa = dp.spatial_folds(da,10,0.08,0)[0]
FOLDS=[0,2,5,9]
def rep(tag, fn):
    A,P,L,Br=[],[],[],[]; t=time.time()
    for k in FOLDS:
        p=fn(k); te=f==k
        A.append(roc_auc_score(y[te],p)); P.append(average_precision_score(y[te],p))
        L.append(log_loss(y[te],np.clip(p,1e-7,1-1e-7),labels=[0,1])); Br.append(brier_score_loss(y[te],p))
    print(f'{tag:34s} AUC={np.mean(A):.4f} AP={np.mean(P):.4f} LL={np.mean(L):.4f} Brier={np.mean(Br):.5f} {time.time()-t:4.0f}s', flush=True)
for nm in ['GLM-m4 (published)','GLM-splines','LightGBM + terrain','XGBoost + terrain']:
    rep(nm, lambda k,nm=nm: B.BASELINES[nm](d[f!=k],d[f==k],seed=0)[0])
def stack(k, **kw):
    tr=f!=k; tra=fa!=k
    s=HeronStack(seed=0, **kw).fit(d[tr], f[tr], dtr_nn=da[tra], blk_nn=fa[tra])
    return s.predict(d[f==k])
rep('HERON stack (NN+trees+GLM)', lambda k: stack(k))
rep('HERON stack, no NN',         lambda k: stack(k, use_nn=False))
rep('HERON stack, no trees',      lambda k: stack(k, use_trees=False))
