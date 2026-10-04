"""Verify inner spatial early stopping and measure its effect on the neural branch."""
import sys; sys.path.insert(0,'Birds/code')
import numpy as np, pandas as pd, time, data_prep as dp, heron
from sklearn.metrics import roc_auc_score, average_precision_score
d=dp.build('NDVI','complete'); y=d.y.values
g=pd.factorize(d.site.astype(str)+'|'+d.Period.astype(str))[0]; f=dp.spatial_folds(d,10,0.08,0)[0]
Xd,Xe=dp.det_features(d),dp.effort_raw(d)
for terr in (False,True):
  Xp=dp.psi_features(d,None,terr)
  for vf in (0.0, 0.18):
    A=[];P=[]; t=time.time()
    for k in (0,2,5,9):
        tr,te=f!=k,f==k
        _,pr=heron.fit_heron(Xp[tr],Xd[tr],Xe[tr],y[tr],g[tr],f[tr],seed=0,epochs=300,
                             batch=32768,gamma=0,pos_weight=1,val_frac=vf)
        p=pr(Xp[te],Xd[te],Xe[te]); A.append(roc_auc_score(y[te],p)); P.append(average_precision_score(y[te],p))
    print(f'terrain={str(terr):5s} early_stop={"on " if vf else "off"}  AUC={np.mean(A):.4f} AP={np.mean(P):.4f}  {time.time()-t:.0f}s', flush=True)
