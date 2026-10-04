"""Regenerate narrative-rewrite summaries from saved outputs; never train models."""
from pathlib import Path
import ast, json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else Path(__file__).resolve().parents[1]
R=ROOT/'results'; P=ROOT/'paper'; F=ROOT/'figures'; A=R/'rewrite_audit'; A.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
def save(fig,name):
 for ext in ('pdf','png'): fig.savefig(F/(name+'.'+ext),dpi=220,bbox_inches='tight')
 plt.close(fig)
def table(name,en,zh,label,head,rows,spec):
 text='\\begin{table}[htbp]\\centering\\small\n\\caption{\\ifzh '+zh+'\\else '+en+'\\fi}\\label{'+label+'}\n\\begin{adjustbox}{max width=\\linewidth}\n\\begin{tabular}{'+spec+'}\\toprule\n'+head+' \\\\\n\\midrule\n'+'\n'.join(' & '.join(row)+' \\\\' for row in rows)+'\n\\bottomrule\\end{tabular}\n\\end{adjustbox}\n\\end{table}\n'
 (P/'tables'/name).write_text(text)
D=pd.read_csv(R/'author_channel/data/Riccordia_maugaeus_eBird_NDVI_corrected_nearest.csv')
D['Date']=pd.to_datetime(D.Date); D=D[D.Date.between('2013-01-01','2025-12-31')].reset_index(drop=True)
C=D[D.NDVI_corrected.notna()].reset_index(drop=True)
assert len(D)==89682 and len(C)==68976
# Reuse the exact archived partition function without importing model/training modules.
tree=ast.parse((ROOT/'code/data_prep.py').read_text()); fun=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='spatial_folds')
ns={'np':np,'pd':pd}; exec(compile(ast.Module(body=[fun],type_ignores=[]),'archived_spatial_folds','exec'),ns)
rows=[]
for rep in range(3):
 f=ns['spatial_folds'](C,10,.08,rep)[0]; fa=ns['spatial_folds'](D,10,.08,rep)[0]
 amap=pd.Series(fa,index=D.SAMPLINGEVENTIDENTIFIER); mapped=C.SAMPLINGEVENTIDENTIFIER.map(amap).to_numpy()
 for k in range(10):
  mask=f==k; rows.append({'rep':rep,'fold':k,'n_test':int(mask.sum()),'overlap':int((mapped[mask]!=k).sum())})
leak=pd.DataFrame(rows); leak['percent']=100*leak.overlap/leak.n_test; leak.to_csv(A/'training_source_overlap.csv',index=False)
lg=leak.groupby('rep')[['n_test','overlap']].sum(); lg['percent']=100*lg.overlap/lg.n_test
print('TRAINING_SOURCE_OVERLAP',lg.to_json(orient='index'))
sp=pd.read_csv(R/'cv_main_spatial_NDVI.csv'); rd=pd.read_csv(R/'cv_main_random_NDVI.csv')
methods=['GLM-m4 (published)','GLM-m3 (published)','GLM-splines','GLM-m4 + terrain','RandomForest','LightGBM','LightGBM + terrain','XGBoost','XGBoost + terrain','MLP','MLP + terrain']
valid=sp[sp.method.isin(methods)].copy(); assert not valid.duplicated(['method','rep','fold']).any()
keys=None
for m in methods:
 k=set(map(tuple,valid.loc[valid.method==m,['rep','fold']].to_numpy())); assert len(k)==30
 if keys is None: keys=k
 else: assert k==keys
metrics=['AUC','AP','Brier','Recall_at_10pct']; S=valid.groupby('method')[metrics].agg(['mean','std']).reindex(methods); S.to_csv(A/'conventional_spatial_summary.csv')
print('CONVENTIONAL',S.round(6).to_string())
short=lambda s:s.replace(' (published)','').replace('RandomForest','Random forest')
rows=[[short(m)]+[f'{S.loc[m,(k,"mean")]:.{4 if k=="Brier" else 3}f} $\\pm$ {S.loc[m,(k,"std")]:.{4 if k=="Brier" else 3}f}' for k in metrics] for m in methods]
table('spatial_valid.tex','Conventional models on the same 30 spatial test folds. Entries are unweighted means $\\pm$ between-fold SD; SD is not a confidence interval. Models use different feature sets.','常规模型在相同 30 个空间测试折上的表现。数值为不加权均值 $\\pm$ 折间标准差，标准差不是置信区间；模型输入信息并不完全相同。','tab:spatialvalid','Model & AUC & AP & Brier & Top-10\\% recall',rows,'lrrrr')
periods=['Pre-Maria','Inter-Huracanes','Post-Fiona']; en=['Pre-Maria','Inter-Hurricanes','Post-Fiona']; sample=[]
for per,lab in zip(periods,en):
 s=D[D.Period==per]; sample.append([lab,f'{len(s):,}',f'{int(s.Presence.sum()):,}',f'{100*s.Presence.mean():.2f}',f'{s.NDVI_corrected.notna().sum():,}',f'{100*s.NDVI_corrected.isna().mean():.2f}'])
table('periods_rewrite.tex','Checklist composition across hurricane-related periods. Reporting rates use all retained checklists; complete records enter the environmental GLM and conventional benchmark.','飓风相关时期的清单组成。报告率使用全部保留清单；完整记录用于环境 GLM 与常规模型比较。','tab:periods','Period & Checklists & Reports & Rate (\\%) & NDVI complete & Missing (\\%)',sample,'lrrrrr')
occ=pd.read_csv(R/'baseline/occupancy_ordering_test.csv'); oc=occ[(occ['mode']=='site_major')&(occ.parameter!='(Intercept)')].copy()
oc['lo']=oc.estimate-1.96*oc.SE; oc['hi']=oc.estimate+1.96*oc.SE; oc.to_csv(A/'corrected_occupancy_intervals.csv',index=False)
labels=['Initial occupancy: elevation','Initial occupancy: NDVI','Colonization: NDVI','Local extinction: elevation','Local extinction: NDVI','Detection: Pre-Maria','Detection: Post-Fiona']
order=[('psi','Elevation'),('psi','NDVI'),('col','NDVI'),('ext','Elevation'),('ext','NDVI'),('det','PeriodPre-Maria'),('det','PeriodPost-Fiona')]
rows=[]
for lab,(comp,par) in zip(labels,order):
 s=oc[(oc.component==comp)&(oc.parameter==par)].iloc[0]; pv='$<0.001$' if s.p<.001 else f'{s.p:.3f}'
 rows.append([lab,f'{s.estimate:.3f}',f'{s.SE:.3f}',f'[{s.lo:.3f}, {s.hi:.3f}]',pv])
table('occupancy_intervals.tex','Corrected dynamic occupancy coefficients. Environmental predictors are standardized across retained sites. Detection contrasts use Inter-Hurricanes as reference. Intervals are approximate 95\\% Wald intervals.','校正后的动态占域系数。环境变量在保留地点间标准化，探测时期以两次飓风之间为参照。区间为近似 95\\% Wald 区间。','tab:occintervals','Parameter & Estimate & SE & 95\\% interval & $p$',rows,'lrrrr')
# These are summaries/redraws of saved predictions, not new fitted experiments.
g=pd.read_csv(R/'baseline/glm_predictions_by_elevation.csv'); cen=pd.read_csv(R/'baseline/glm_elevational_centroids.csv')
colors=['#236b8e','#b0473c','#27836a']; fig,ax=plt.subplots(1,2,figsize=(9.3,3.3),layout='constrained')
for per,col in zip(en,colors):ax[0].plot(g.Elevation_m,100*g[per],color=col,label=per,lw=2)
ax[0].set(xlabel='Elevation (m)',ylabel='Predicted reporting probability (%)',title='(a) Reproduced conditional GLM curves'); ax[0].legend(fontsize=8)
ax[1].bar(range(3),cen.delta_vs_PreMaria,color=colors,width=.6); ax[1].set_xticks(range(3),['Pre-Maria','Inter-\nHurricanes','Post-Fiona']);ax[1].set(ylabel='Centroid shift from baseline (m)',title='(b) Probability-weighted centroid shifts')
for i,v in enumerate(cen.delta_vs_PreMaria): ax[1].text(i,v+.6,f'{v:+.1f}',ha='center')
ax[1].set_ylim(0,34);save(fig,'fig21_reporting_gradients_rewrite')
fig,axes=plt.subplots(1,3,figsize=(10,4.8),layout='constrained');y=np.arange(len(methods))
for ax,k in zip(axes,['AUC','AP','Brier']):
 ax.errorbar(S[(k,'mean')],y,xerr=S[(k,'std')],fmt='o',color='#236b8e',capsize=3,ms=4)
 ax.axvline(S.loc[methods[0],(k,'mean')],ls='--',color='gray',lw=1); ax.set_xlabel(k+(' (lower is better)' if k=='Brier' else ' (higher is better)')); ax.set_yticks(y);ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
axes[0].set_yticklabels([short(m) for m in methods],fontsize=9)
for ax in axes[1:]:ax.set_yticklabels([])
fig.suptitle('Conventional models: 30 shared spatial test folds; bars show fold SD',fontsize=11);save(fig,'fig22_conventional_spatial_rewrite')
selected=[methods[0],'RandomForest','LightGBM + terrain','XGBoost + terrain'];fig,axes=plt.subplots(1,2,figsize=(9,3.7),layout='constrained')
for ax,k in zip(axes,['AUC','AP']):
 for m,col in zip(selected,['#555555','#236b8e','#27836a','#b0473c']):ax.plot([0,1],[rd[rd.method==m][k].mean(),valid[valid.method==m][k].mean()],marker='o',color=col,label=short(m))
 ax.set_xticks([0,1],['Random\n15 evaluations','Spatial\n30 evaluations']);ax.set_ylabel(k);ax.margins(x=.2);ax.grid(axis='y',alpha=.15)
axes[0].legend(fontsize=8);save(fig,'fig23_protocols_rewrite')
fig,ax=plt.subplots(figsize=(8.2,3.2),layout='constrained')
for rep,col in zip(range(3),colors):
 t=leak[leak.rep==rep];ax.plot(t.fold+1,t.percent,'o-',color=col,label=f'Block origin {rep+1}')
ax.set(xlabel='Test fold',ylabel='Test records in neural training source (%)',ylim=(-2,102),xticks=range(1,11));ax.legend(fontsize=8);save(fig,'fig24_training_overlap_rewrite')
macros=[]
for tag,m in [('Glm',methods[0]),('Rf','RandomForest'),('LgbTerrain','LightGBM + terrain')]:
 for k in metrics:macros.append('\\newcommand{\\rw'+tag+k.replace('_','').replace('10','Ten')+'}{'+f'{S.loc[m,(k,"mean")]:.4f}'+'}')
for i,v in enumerate(lg.percent):macros.append('\\newcommand{\\rwOverlap'+['One','Two','Three'][i]+'}{'+f'{v:.2f}'+'}')
(P/'numbers_rewrite.tex').write_text('% Generated from saved results by code/rewrite_assets.py. No new training.\n'+'\n'.join(macros)+'\n')
print('ASSETS_READY')
