from pathlib import Path
import json,numpy as np,pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
ROOT=Path(__file__).resolve().parents[2];O=ROOT/'results/environment_analysis'
settings=json.loads((O/'strata_definition.json').read_text());q=settings['checklist_NDVI_tertile_cutpoints'];order=['Low','Mid','High']
def strata(d,e,v):
 d['elev_band']=pd.cut(d[e],[-np.inf,300,600,np.inf],right=False,labels=order).astype(str)
 d['green_band']=pd.cut(d[v],[-np.inf,*q,np.inf],labels=order).astype(str)
 d['cell']=d.elev_band+'_'+d.green_band
 return d
# Joint coefficient simulations preserve covariance among occupancy process estimates.
s=strata(pd.read_csv(O/'occupancy_site_predictions.csv'),'elev_m','veg_raw');Y=pd.read_csv(ROOT/'results/baseline/occ_y_NDVI_author.csv').values
coef=pd.read_csv(O/'occupancy_coefficient_order.csv');V=pd.read_csv(O/'occupancy_covariance.csv',index_col=0)
assert list(V.index)==coef.name.tolist() and list(V.columns)==coef.name.tolist()
assert np.linalg.eigvalsh(V.values).min()>0
rng=np.random.default_rng(20261001);draw=rng.multivariate_normal(coef.estimate.values,V.values,size=2000)
x=np.column_stack([np.ones(len(s)),s.elev_z,s.veg_z]);xg=x[:,[0,2]]
psi=expit(x@draw[:,:3].T);persist=1-expit(x@draw[:,5:8].T);colon=expit(xg@draw[:,3:5].T)
y3=Y.reshape(len(s),3,3);reported=np.nansum(y3,axis=2)>0;complete=np.isfinite(y3).all(axis=2)
s['reported_periods']=reported.sum(axis=1);s['all_nine_visits']=complete.all(axis=1)
s['pattern']=[''.join('1' if x else '0' for x in r) for r in reported]
s.to_csv(O/'occupancy_environment_sites.csv',index=False)
rows=[];transition=[]
for key,sub in s.groupby('cell'):
 ix=sub.index.values;row={'cell':key,'n_sites':len(ix),'median_elev':sub.elev_m.median(),'median_ndvi':sub.veg_raw.median(),'all_nine_visits':int(complete[ix].all(axis=1).sum())}
 for name,mat,pt in [('initial',psi,s.psi_Predicted),('persistence',persist,1-s.ext_Predicted),('colonization',colon,s.col_Predicted)]:
  means=mat[ix].mean(axis=0);row[name]=pt.iloc[ix].mean();row[name+'_lo'],row[name+'_hi']=np.quantile(means,[.025,.975])
 # Equal-effort observed transitions: both periods have three selected visits.
 total_prev=repeat=total_zero=new=0;eligible_pairs=0
 for t in [0,1]:
  ok=ix[complete[ix,t]&complete[ix,t+1]];a=reported[ok,t];b=reported[ok,t+1]
  eligible_pairs+=len(ok);total_prev+=int(a.sum());repeat+=int((a&b).sum());total_zero+=int((~a).sum());new+=int(((~a)&b).sum())
 row.update(eligible_pairs=eligible_pairs,previous_report_pairs=total_prev,repeat_report_pairs=repeat,observed_repeat=repeat/total_prev if total_prev else np.nan,previous_nonreport_pairs=total_zero,new_report_pairs=new,observed_new=new/total_zero if total_zero else np.nan)
 rows.append(row)
pd.DataFrame(rows).to_csv(O/'occupancy_environment_summary.csv',index=False)
# Representative contrasts hold the other environmental covariate constant.
emean,esd=s.elev_m.mean(),s.elev_m.std(ddof=1);vmean,vsd=s.veg_raw.mean(),s.veg_raw.std(ddof=1)
assert np.max(np.abs((s.elev_m-emean)/esd-s.elev_z))<1e-8
assert np.max(np.abs((s.veg_raw-vmean)/vsd-s.veg_z))<1e-8
contr=[]
for name,left,right in [('greenness_at_100m',(100,.30),(100,.76)),('elevation_at_NDVI056',(100,.56),(750,.56))]:
 for comp,idx in [('persistence',[5,6,7]),('colonization',[3,4])]:
  probs=[];pts=[]
  for elev,veg in [left,right]:
   xx=np.array([1,(elev-emean)/esd,(veg-vmean)/vsd]) if comp=='persistence' else np.array([1,(veg-vmean)/vsd])
   pr=expit(draw[:,idx]@xx);pt=expit(coef.estimate.values[idx]@xx)
   if comp=='persistence':pr=1-pr;pt=1-pt
   probs.append(pr);pts.append(pt)
  lo,hi=np.quantile(probs[1]-probs[0],[.025,.975]);contr.append(dict(contrast=name,component=comp,left=pts[0],right=pts[1],delta=pts[1]-pts[0],lo=lo,hi=hi))
pd.DataFrame(contr).to_csv(O/'occupancy_standardized_contrasts.csv',index=False)
s.groupby(['elev_band','green_band','pattern']).size().rename('n').reset_index().to_csv(O/'observed_report_patterns.csv',index=False)
# One spatial repeat; all four models evaluated on exactly the same checklist identifiers.
d=strata(pd.read_csv(O/'environment_oof_predictions.csv'),'Elevation_m','veg');assert d.RandomForest.notna().all()
models=['GLM-m4 (published)','RandomForest','LightGBM + terrain','XGBoost + terrain']
metrics=[];policies=[]
for m in models:
 flags={p:np.zeros(len(d),bool) for p in ['global','elevation_stratified']}
 for fold,a in d.groupby('fold'):
  budget=int(np.floor(.1*len(a)));idx=a.sort_values([m,'SAMPLINGEVENTIDENTIFIER'],ascending=[False,True]).index[:budget]
  flags['global'][idx]=True
  counts=a.elev_band.value_counts().reindex(order,fill_value=0);alloc=np.floor(.1*counts).astype(int);extra=budget-alloc.sum()
  for band in (.1*counts-alloc).sort_values(ascending=False,kind='stable').index[:extra]:alloc[band]+=1
  for band in order:
   b=a[a.elev_band==band];idx=b.sort_values([m,'SAMPLINGEVENTIDENTIFIER'],ascending=[False,True]).index[:alloc[band]];flags['elevation_stratified'][idx]=True
  assert flags['global'][a.index].sum()==flags['elevation_stratified'][a.index].sum()==budget
 for policy,f in flags.items():d[m+'__'+policy]=f.astype(int)
 for groupcol in ['elev_band','cell']:
  for key,a in d.groupby(groupcol):
   y=a.y.values;p=a[m].values
   metrics.append(dict(model=m,group_type=groupcol,group=key,n=len(a),reports=int(y.sum()),prevalence=y.mean(),blocks=a.block.nunique(),AUC=roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan,AP=average_precision_score(y,p) if y.sum() else np.nan,Brier=brier_score_loss(y,p)))
 for policy,f in flags.items():
  for band in ['All',*order]:
   a=d if band=='All' else d[d.elev_band==band];sel=f[a.index];captured=int(a.y.values[sel].sum());npos=int(a.y.sum())
   policies.append(dict(model=m,policy=policy,elev_band=band,n=len(a),reports=npos,selected=int(sel.sum()),captured=captured,allocation=sel.mean(),recall=captured/npos,precision=captured/sel.sum() if sel.sum() else np.nan))
pd.DataFrame(metrics).to_csv(O/'environment_prediction_metrics.csv',index=False)
pd.DataFrame(policies).to_csv(O/'monitoring_policies.csv',index=False)
d.to_csv(O/'oof_predictions_with_policies.csv',index=False)
# Spatial block bootstrap: recall contrasts condition on the existing fitted predictions.
blocks=sorted(d.block.unique());W=np.random.default_rng(20261001).multinomial(len(blocks),np.ones(len(blocks))/len(blocks),size=1000)
contrasts=[]
for band in ['All',*order]:
 a=d if band=='All' else d[d.elev_band==band]
 den=a.groupby('block').y.sum().reindex(blocks,fill_value=0).values;denb=W@den
 def recallboot(model,policy):
  vals=a.y*a[model+'__'+policy];num=vals.groupby(a.block).sum().reindex(blocks,fill_value=0).values
  return (W@num)/denb, num.sum()/den.sum()
 for name,left,right in [('RF_minus_GLM_global',('RandomForest','global'),('GLM-m4 (published)','global')),('RF_stratified_minus_global',('RandomForest','elevation_stratified'),('RandomForest','global'))]:
  l,lp=recallboot(*left);r,rp=recallboot(*right);lo,hi=np.quantile(l-r,[.025,.975]);contrasts.append(dict(contrast=name,elev_band=band,delta=lp-rp,lo=lo,hi=hi))
pd.DataFrame(contrasts).to_csv(O/'monitoring_cluster_intervals.csv',index=False)
summary={'occupancy_sites':len(s),'complete_nine_visit_sites':int(complete.all(axis=1).sum()),'oof_n':len(d),'oof_reports':int(d.y.sum()),'selected_each_policy':int(d['RandomForest__global'].sum()),'policy_budgets_equal_all_folds':True,'covariance_draws':2000,'bootstrap_replicates':1000}
(O/'process_monitor_validation.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)
print(pd.DataFrame(rows).round(4).to_string(index=False),flush=True)
print(pd.DataFrame(policies).query("model=='RandomForest'").round(4).to_string(index=False),flush=True)
