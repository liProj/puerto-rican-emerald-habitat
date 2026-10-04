"""Component diagnostic for HERON on a fixed set of spatial folds, against the published GLM."""
import sys; sys.path.insert(0,'Birds/code')
import numpy as np, pandas as pd, data_prep as dp, heron, baselines as B, time
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
d=dp.build('NDVI','complete'); d['Elevation_100m']=d.Elevation_m/100
r=dp.RFF(seed=0); RF=r(d)
Xd=dp.det_features(d); Xe=dp.effort_raw(d); y=d.y.values
f,_=dp.spatial_folds(d,10,0.08,0); grp=pd.factorize(d.site.astype(str)+'|'+d.Period.astype(str))[0]
FOLDS=[0,2,5,9]     # spread of positive rates: 2.4%, 6.5%, 1.4%, 11.8%
def ev(tag, fn):
    A,P,L=[],[],[]; t=time.time()
    for k in FOLDS:
        tr,te=f!=k,f==k
        p=fn(tr,te,k)
        A.append(roc_auc_score(y[te],p)); P.append(average_precision_score(y[te],p))
        L.append(log_loss(y[te],np.clip(p,1e-7,1-1e-7),labels=[0,1]))
    print(f'{tag:46s} AUC={np.mean(A):.4f} AP={np.mean(P):.4f} LL={np.mean(L):.4f} {time.time()-t:4.0f}s', flush=True)
    return np.mean(A),np.mean(P)
for nm in ['GLM-m4 (published)','GLM-splines','LightGBM','XGBoost']:
    ev(nm, lambda tr,te,k,nm=nm: B.BASELINES[nm](d[tr],d[te],seed=0)[0])
K=dict(epochs=350, batch=16384)
def mk(use_rff, **kw):
    Xp=dp.psi_features(d, RF if use_rff else None)
    def fn(tr,te,k):
        _,pred=heron.fit_heron(Xp[tr],Xd[tr],Xe[tr],y[tr],grp[tr],f[tr],seed=0,**{**K,**kw})
        return pred(Xp[te],Xd[te],Xe[te])
    return fn
BASE=dict(lam_vrex=0, lam_group=0, gamma=0, pos_weight=1)
ev('joint MLP, no RFF, plain BCE',            mk(False, factorised=False, **BASE))
ev('joint MLP, +RFF',                         mk(True,  factorised=False, **BASE))
ev('factorised, no RFF',                      mk(False, **BASE))
ev('factorised, +RFF',                        mk(True,  **BASE))
ev('factorised, no RFF, +group',              mk(False, **{**BASE,'lam_group':0.5}))
ev('factorised, no RFF, +group +vrex',        mk(False, **{**BASE,'lam_group':0.5,'lam_vrex':1.0}))
ev('factorised, no RFF, +group +vrex +focal', mk(False, lam_group=0.5, lam_vrex=1.0, gamma=1, pos_weight=3))
ev('factorised, +RFF, +group +vrex',          mk(True,  **{**BASE,'lam_group':0.5,'lam_vrex':1.0}))
