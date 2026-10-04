from pathlib import Path
import os,sys,json,time,hashlib
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];os.chdir(ROOT.parent);sys.path.insert(0,str(ROOT/'code'))
import data_prep as dp,baselines as B
O=ROOT/'results/environment_analysis';O.mkdir(exist_ok=True)
d=dp.build('NDVI','complete');f,block=dp.spatial_folds(d,10,.08,0)
old=pd.read_csv(ROOT/'results/oof_spatial_NDVI.csv')
assert len(d)==len(old)==68976 and d.SAMPLINGEVENTIDENTIFIER.is_unique
assert np.array_equal(d.y.values,old.y.values) and np.array_equal(f,old.fold.values)
# Reconstruct one conventional fold to validate the saved prediction order.
p,_=B.BASELINES['GLM-m4 (published)'](d[f!=0],d[f==0],seed=0)
err=float(np.max(np.abs(p-old.loc[f==0,'GLM-m4 (published)'].values)))
assert err<1e-8,err
cols=['SAMPLINGEVENTIDENTIFIER','site','Period','Elevation_m','veg','Duration_min','Distance_km','Longitude','Latitude','y']
a=d[cols].copy();a['fold']=f;a['block']=block
for name in ['GLM-m4 (published)','LightGBM + terrain','XGBoost + terrain']:a[name]=old[name].values
rf=np.full(len(d),np.nan);timings=[]
for k in range(10):
 tr,te=f!=k,f==k
 assert not(set(block[tr])&set(block[te]))
 assert not(set(d.loc[tr,'SAMPLINGEVENTIDENTIFIER'])&set(d.loc[te,'SAMPLINGEVENTIDENTIFIER']))
 start=time.time();rf[te],_=B.BASELINES['RandomForest'](d[tr],d[te],seed=0)
 timings.append({'fold':k,'seconds':time.time()-start,'n_train':int(tr.sum()),'n_test':int(te.sum())})
 print('RF fold',k,timings[-1],flush=True)
 a['RandomForest']=rf;a.to_csv(O/'environment_oof_predictions.csv',index=False)
assert np.isfinite(rf).all()
pd.DataFrame(timings).to_csv(O/'rf_training_times.csv',index=False)
(O/'oof_validation.json').write_text(json.dumps({'n':len(d),'unique_ids':True,'saved_y_and_fold_match':True,'recomputed_glm_fold0_max_error':err,'rf_disjoint_ids_and_blocks_all_folds':True,'data_sha256':hashlib.sha256(pd.util.hash_pandas_object(d,index=True).values.tobytes()).hexdigest()},indent=2))
print('RF completed',flush=True)
