from pathlib import Path
import os,sys,json
import numpy as np,pandas as pd,statsmodels.api as sm,statsmodels.formula.api as smf
from patsy import build_design_matrices,dmatrices
from scipy.special import expit
from scipy.optimize import minimize
ROOT=Path(__file__).resolve().parents[2];os.chdir(ROOT.parent);sys.path.insert(0,str(ROOT/'code'))
from common import load
O=ROOT/'results/environment_analysis';O.mkdir(exist_ok=True)
d=load('NDVI');d['period']=d.Period.astype(str).replace({'Inter-Huracanes':'Inter-Hurricanes'})
d['y']=d.Presence.astype(int);d['missing']=d.veg.isna().astype(int);d['logdur']=np.log(d.Duration_min)
d['season']=d.Date.dt.quarter.astype(str)
d['lon_c']=d.Longitude+66;d['lat_c']=(d.Latitude-18)/.3
d['elev_band']=pd.cut(d.Elevation_m,[-np.inf,300,600,np.inf],right=False,labels=['Low','Mid','High']).astype(str)
off=np.random.default_rng(0).uniform(0,.08,2)
d['block']=(np.floor((d.Longitude+off[0])/.08).astype(int).astype(str)+'_'+np.floor((d.Latitude+off[1])/.08).astype(int).astype(str))
q=d.veg.dropna().quantile([1/3,2/3]).values
bins=[-np.inf,*q,np.inf];d['green_band']=pd.cut(d.veg,bins,labels=['Low','Mid','High']).astype(str)
(O/'strata_definition.json').write_text(json.dumps({'elevation_edges_m':[300,600],'checklist_NDVI_tertile_cutpoints':q.tolist(),'block_deg':.08,'block_offset':off.tolist(),'bootstrap_replicates':1000,'seed':20261001,'exploratory_analysis':True},indent=2))
def aggregate(keys):
 return d.groupby(keys,observed=True).agg(n=('y','size'),reports=('y','sum'),report_rate=('y','mean'),missing_rate=('missing','mean'),median_elev=('Elevation_m','median'),median_duration=('Duration_min','median'),median_distance=('Distance_km','median'),sites=('site','nunique'),blocks=('block','nunique')).reset_index()
for name,keys in [('missing_by_elevation',['elev_band','missing']),('missing_by_period_elevation',['period','elev_band','missing']),('missing_overall',['missing']),('raw_environment_reporting',['period','elev_band','green_band'])]:aggregate(keys).to_csv(O/(name+'.csv'),index=False)
# Simple cluster bootstrap for period-level rates in the same resampled blocks.
c=d[d.missing==0].copy();c['cell']=c.elev_band+'_'+c.green_band
block_order=sorted(c.block.unique());cell_order=[e+'_'+v for e in ['Low','Mid','High'] for v in ['Low','Mid','High']]
periods=['Pre-Maria','Inter-Hurricanes','Post-Fiona'];rng=np.random.default_rng(20261001)
W=rng.multinomial(len(block_order),np.ones(len(block_order))/len(block_order),size=1000)
raw=[]
for cell in cell_order:
 rates={};counts={}
 for p in periods:
  s=c[(c.cell==cell)&(c.period==p)];g=s.groupby('block').y.agg(['sum','size']).reindex(block_order,fill_value=0)
  den=W@g['size'].values;num=W@g['sum'].values
  rates[p]=np.divide(num,den,out=np.full(1000,np.nan),where=den>0)
  counts[p]=[len(s),int(s.y.sum()),s.block.nunique()]
 delta=rates['Post-Fiona']-rates['Pre-Maria'];lo,hi=np.nanquantile(delta,[.025,.975])
 raw.append(dict(cell=cell,pre_n=counts[periods[0]][0],post_n=counts[periods[2]][0],pre_reports=counts[periods[0]][1],post_reports=counts[periods[2]][1],pre_blocks=counts[periods[0]][2],post_blocks=counts[periods[2]][2],raw_delta_lo=lo,raw_delta_hi=hi))
pd.DataFrame(raw).to_csv(O/'raw_rebound_cluster_intervals.csv',index=False)
# Sequential adjustment describes the composition of NDVI-available vs missing records.
forms={'unadjusted':'y ~ missing','environment':'y ~ missing + C(period) + bs(Elevation_m, df=5)','effort_space':'y ~ missing + C(period) + bs(Elevation_m, df=5) + logdur + np.log1p(Distance_km) + C(season) + lon_c + lat_c + I(lon_c**2) + I(lat_c**2) + lon_c:lat_c'}
def fit_cluster(form,data):
 y,x=dmatrices(form,data,return_type='dataframe');di=x.design_info
 start=np.zeros(x.shape[1]);start[0]=np.log(data.y.mean()/(1-data.y.mean()))
 xx=x.values;yy=y.values.ravel()
 def objective(b):
  eta=xx@b;return np.sum(np.logaddexp(0,eta)-yy*eta)
 def grad(b):return xx.T@(expit(xx@b)-yy)
 def hess(b):
  pr=expit(xx@b);return xx.T@(xx*(pr*(1-pr))[:,None])
 opt=minimize(objective,start,jac=grad,hess=hess,method='trust-exact',options={'maxiter':200,'gtol':1e-5})
 assert np.max(np.abs(grad(opt.x)))<.01,(form,opt.message,grad(opt.x))
 result=sm.GLM(y,x,family=sm.families.Binomial()).fit(start_params=opt.x,maxiter=20,cov_type='cluster',cov_kwds={'groups':data.block})
 assert result.converged and np.isfinite(result.params).all() and np.isfinite(result.bse).all() and abs(result.params).max()<100,(form,result.params)
 result._design_info=di
 return result
rows=[]
for name,form in forms.items():
 fit=fit_cluster(form,d)
 b,se=fit.params['missing'],fit.bse['missing'];rows.append(dict(model=name,beta=b,se=se,OR=np.exp(b),lo=np.exp(b-1.96*se),hi=np.exp(b+1.96*se),n=int(fit.nobs),clusters=d.block.nunique()))
 print('missingness',name,rows[-1],flush=True)
 if name=='effort_space':
  cr=[]
  for band,s in d.groupby('elev_band'):
   estimates=[];grads=[]
   for val in [0,1]:
    a=s.copy();a['missing']=val;x=np.asarray(build_design_matrices([fit._design_info],a)[0]);pr=expit(x@fit.params.values)
    estimates.append(pr.mean());grads.append((x*(pr*(1-pr))[:,None]).mean(axis=0))
   gradient=grads[1]-grads[0];se2=np.sqrt(gradient@fit.cov_params().values@gradient);de=estimates[1]-estimates[0]
   cr.append(dict(elev_band=band,available_adjusted=estimates[0],missing_adjusted=estimates[1],delta=de,lo=de-1.96*se2,hi=de+1.96*se2))
  pd.DataFrame(cr).to_csv(O/'missing_adjusted_by_elevation.csv',index=False)
pd.DataFrame(rows).to_csv(O/'missing_adjustment.csv',index=False)
# Standardize all three periods to the same observed covariate distribution within each cell.
c['Elevation_100m']=c.Elevation_m/100
fit=fit_cluster('y ~ C(period, Treatment(reference="Pre-Maria"))*Elevation_100m + veg + Duration_min + Distance_km',c)
assert abs(fit.aic-21373.31)<.1,fit.aic
rr=[]
for cell,s in c.groupby('cell'):
 preds=[];grads=[]
 for p in periods:
  a=s.copy();a['period']=p;x=np.asarray(build_design_matrices([fit._design_info],a)[0]);pr=expit(x@fit.params.values)
  preds.append(pr.mean());grads.append((x*(pr*(1-pr))[:,None]).mean(axis=0))
 g=grads[2]-grads[0];se=np.sqrt(g@fit.cov_params().values@g);delta=preds[2]-preds[0]
 rr.append(dict(cell=cell,n=len(s),reports=int(s.y.sum()),blocks=s.block.nunique(),median_elev=s.Elevation_m.median(),median_ndvi=s.veg.median(),pre=preds[0],inter=preds[1],post=preds[2],delta=delta,lo=delta-1.96*se,hi=delta+1.96*se))
pd.DataFrame(rr).to_csv(O/'standardized_rebound.csv',index=False)
# Observation continuity and NDVI coverage at the original occupancy grids.
site=d.groupby('site').agg(n=('y','size'),reports=('y','sum'),periods=('period','nunique'),valid_ndvi=('veg','count'),missing_rate=('missing','mean'),elevation=('Elevation_m','mean'))
site.to_csv(O/'grid_coverage.csv');summary={'checklists':len(d),'reports':int(d.y.sum()),'ndvi_available':len(c),'blocks':d.block.nunique(),'three_period_grids':int((site.periods==3).sum()),'three_period_grids_without_ndvi':int(((site.periods==3)&(site.valid_ndvi==0)).sum()),'reporting_glm_AIC':fit.aic}
(O/'stratification_validation.json').write_text(json.dumps(summary,indent=2));print('STRATIFICATION DONE',summary,flush=True)
