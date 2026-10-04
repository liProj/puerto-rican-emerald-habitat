from pathlib import Path
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
ROOT=Path(__file__).resolve().parents[2];O=ROOT/'results/environment_analysis';F=ROOT/'figures';T=ROOT/'paper/tables'
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220,'pdf.fonttype':42})
order=['Low','Mid','High'];labels=['<300 m','300–<600 m','≥600 m'];green=['Low NDVI','Mid NDVI','High NDVI'];colors=['#b35336','#216c9a','#338455']
def save(fig,name):
 fig.savefig(F/(name+'.pdf'),bbox_inches='tight');fig.savefig(F/(name+'.png'),bbox_inches='tight');plt.close(fig)
def tab(name,label,zcap,ecap,head,rows,notes=''):
 n=len(head.split('&'));text=r'\begin{table}[htbp]\centering\footnotesize'+'\n'+r'\caption{\ifzh '+zcap+r'\else '+ecap+r'\fi}\label{'+label+'}\n'+r'\begin{adjustbox}{max width=\linewidth}\begin{tabular}{'+('l'*2+'r'*(n-2))+r'}\toprule'+'\n'+head+r' \\ \midrule'+'\n'+'\n'.join(rows)+'\n'+r'\bottomrule\end{tabular}\end{adjustbox}'+'\n'+notes+'\n'+r'\end{table}'+'\n';(T/(name+'.tex')).write_text(text)
def row(vals):return ' & '.join(map(str,vals))+r' \\'
def pc(x):return f'{100*x:.1f}'
def ci(r,k):return f'{pc(r[k])} [{pc(r[k+"_lo"])}, {pc(r[k+"_hi"])}]'
a=pd.read_csv(O/'missing_by_elevation.csv');m=pd.read_csv(O/'missing_adjustment.csv')
fig,ax=plt.subplots(1,3,figsize=(11.5,3.2),layout='constrained')
missing=[]
for b in order:
 s=a[a.elev_band==b].set_index('missing');missing.append(s.loc[1,'n']/s.n.sum())
ax[0].bar(labels,np.array(missing)*100,color=colors)
for i,v in enumerate(missing):ax[0].text(i,100*v+1,f'{100*v:.1f}%',ha='center')
ax[0].set_ylim(0,60);ax[0].set_ylabel('NDVI missing (%)');ax[0].set_title('(a) Environmental coverage')
for j,(v,lab,c) in enumerate([(0,'NDVI available','#216c9a'),(1,'NDVI missing','#b35336')]):
 vals=[100*a[(a.elev_band==b)&(a.missing==v)].report_rate.iloc[0] for b in order];ax[1].plot(labels,vals,'o-',label=lab,color=c)
ax[1].set_ylabel('Positive reports (%)');ax[1].legend(frameon=False,fontsize=8);ax[1].set_title('(b) Reporting within elevation bands')
ax[2].errorbar(m.OR,np.arange(3),xerr=[m.OR-m.lo,m.hi-m.OR],fmt='o',capsize=4,color='#26475a');ax[2].axvline(1,color='.5',ls=':');ax[2].set_yticks(range(3),['Unadjusted','+ elevation, period','+ effort, season, location']);ax[2].invert_yaxis();ax[2].set_xlabel('Missing vs available: odds ratio');ax[2].set_title('(c) Composition adjustment')
save(fig,'fig25_environment_missingness')
rows=[]
for b in order:
 s=a[a.elev_band==b].set_index('missing');rows.append(row([b,int(s.loc[0,'n']),int(s.loc[1,'n']),pc(s.loc[1,'n']/s.n.sum()),pc(s.loc[0,'report_rate']),pc(s.loc[1,'report_rate'])]))
tab('environment_missingness','tab:envmissing','NDVI 可用性与报告率的海拔分层。报告率与缺失率均以百分比表示。','Elevation-stratified NDVI availability and reporting; rates are percentages.',r'Elevation & Available $n$ & Missing $n$ & Missing (\%) & Report: available (\%) & Report: missing (\%)',rows)
r=pd.read_csv(O/'standardized_rebound.csv').set_index('cell')
fig,ax=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
for j,(key,title) in enumerate([('pre','(a) Standardized Pre-Maria'),('post','(b) Standardized Post-Fiona'),('delta','(c) Post-Fiona minus Pre-Maria')]):
 mat=np.array([[100*r.loc[e+'_'+g,key] for g in order] for e in order]);im=ax[j].imshow(mat,cmap='YlGnBu' if j<2 else 'RdBu',vmin=0 if j<2 else -3,vmax=50 if j<2 else 3,aspect='auto')
 for i,e in enumerate(order):
  for k,g in enumerate(order):
   rr=r.loc[e+'_'+g];txt=(f'{mat[i,k]:+.2f} pp\n[{100*rr.lo:+.1f},\n {100*rr.hi:+.1f}]' if j==2 else f'{mat[i,k]:.2f}%')
   ax[j].text(k,i,txt,ha='center',va='center',fontsize=8,color='white' if (j<2 and mat[i,k]>30) or (j==2 and abs(mat[i,k])>2.1) else 'black')
 ax[j].set_xticks(range(3),['Low','Mid','High']);ax[j].set_yticks(range(3),labels);ax[j].set_xlabel('NDVI stratum');ax[j].set_title(title,fontsize=9)
 fig.colorbar(im,ax=ax[j],shrink=.8,label='Probability (%)' if j<2 else 'Percentage points')
save(fig,'fig26_environment_rebound')
rows=[]
for e in order:
 for g in order:
  rr=r.loc[e+'_'+g];rows.append(row([e+'/'+g,int(rr.n),int(rr.reports),pc(rr.pre),pc(rr.inter),pc(rr.post),f'{100*rr.delta:+.2f} [{100*rr.lo:+.2f}, {100*rr.hi:+.2f}]']))
tab('environment_rebound','tab:envrebound','海拔与绿度交叉分层的标准化报告概率。三个时期使用各层相同协变量分布；变化区间采用空间区块聚类协方差。','Standardized reporting by elevation/greenness stratum. Periods share the within-stratum covariate distribution; contrast intervals use spatial block-clustered covariance.',r'Elevation/NDVI & $n$ & Reports & Pre (\%) & Inter (\%) & Post (\%) & Change, pp [95\% CI]',rows)
s=pd.read_csv(O/'occupancy_environment_summary.csv').set_index('cell')
fig,ax=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
for j,(key,title) in enumerate([('persistence','(a) Occupancy persistence'),('colonization','(b) Colonization'),('observed_repeat','(c) Repeated reporting; equal visits')]):
 mat=np.array([[100*s.loc[e+'_'+g,key] for g in order] for e in order]);im=ax[j].imshow(mat,cmap='YlGnBu',vmin=0,vmax=100,aspect='auto')
 for i,e in enumerate(order):
  for k,g in enumerate(order):
   rr=s.loc[e+'_'+g];txt=f'{mat[i,k]:.1f}%\n'+(f'n={int(rr.n_sites)}' if j<2 else f'{int(rr.repeat_report_pairs)}/{int(rr.previous_report_pairs)}')
   if np.isnan(mat[i,k]):txt='No prior report\n0/0'
   ax[j].text(k,i,txt,ha='center',va='center',fontsize=8,color='white' if mat[i,k]>60 else 'black')
   if rr.n_sites<10:ax[j].add_patch(Rectangle((k-.5,i-.5),1,1,fill=False,hatch='///',edgecolor='.6',linewidth=0))
 ax[j].set_xticks(range(3),['Low','Mid','High']);ax[j].set_yticks(range(3),labels);ax[j].set_xlabel('Site mean NDVI stratum');ax[j].set_title(title,fontsize=8.5)
fig.colorbar(im,ax=ax,shrink=.8,label='Probability / observed proportion (%)')
save(fig,'fig27_environment_persistence')
rows=[]
for e in order:
 for g in order:
  rr=s.loc[e+'_'+g];rows.append(row([e+'/'+g+(r'$\dagger$' if rr.n_sites<10 else ''),int(rr.n_sites),ci(rr,'persistence'),ci(rr,'colonization'),f'{int(rr.repeat_report_pairs)}/{int(rr.previous_report_pairs)}']))
tab('environment_persistence','tab:envpersistence','环境分层的占域持续概率、定殖概率与再次记录。占域持续概率以网格在前一时期已被占据为条件；区间来自 2,000 次联合参数模拟。','Occupancy persistence, colonization, and repeated reports by environment. Occupancy persistence is conditional on the grid being occupied in the previous period; intervals use 2,000 joint parameter draws.',r'Elevation/NDVI & Grids & Persistence \% [95\% CI] & Colonization \% [95\% CI] & Repeated reports',rows,r'\par\smallskip\ifzh $\dagger$ 少于十个网格。再次记录计数仅使用相邻两个时期各有三次访问的网格--时期对，分母为前一时期记录到该物种的网格--时期对数。\else $\dagger$ Fewer than ten grids. Repeated reports use grid--period pairs with three visits in each adjacent period; the denominator counts pairs with a report in the previous period.\fi')
p=pd.read_csv(O/'monitoring_policies.csv');metrics=pd.read_csv(O/'environment_prediction_metrics.csv');z=metrics[metrics.group_type=='elev_band']
fig,ax=plt.subplots(1,3,figsize=(11,3.4),layout='constrained');x=np.arange(3)
for m,lab,c in [('GLM-m4 (published)','GLM m4','#b35336'),('RandomForest','Random forest','#216c9a')]:
 vals=z[z.model==m].set_index('group').loc[order,'AP'];ax[0].plot(x,vals,'o-',label=lab,color=c)
ax[0].set_title('(a) Positive-report ranking');ax[0].set_ylabel('Average precision');ax[0].legend(frameon=False)
for j,key in enumerate(['allocation','recall'],1):
 for k,(pol,lab,c) in enumerate([('global','Global ranking','#216c9a'),('elevation_stratified','Within-elevation ranking','#b35336')]):
  vals=p[(p.model=='RandomForest')&(p.policy==pol)].set_index('elev_band').loc[order,key]*100;ax[j].bar(x+(k-.5)*.32,vals,.32,label=lab,color=c)
 ax[j].set_title(['','(b) Random-forest budget allocation','(c) Random-forest report coverage'][j]);ax[j].set_ylabel('Selected checklists (%)' if j==1 else 'Positive reports captured (%)');ax[j].legend(frameon=False,fontsize=7)
for a in ax:a.set_xticks(x,labels);a.set_xlabel('Elevation stratum')
save(fig,'fig28_environment_monitoring')
rows=[]
for b in order:
 for m,lab in [('GLM-m4 (published)','GLM m4'),('RandomForest','RF')]:
  rr=z[(z.model==m)&(z.group==b)].iloc[0];gg=p[(p.model==m)&(p.elev_band==b)].set_index('policy');rows.append(row([b,lab,int(rr.reports),f'{rr.AP:.3f}',f'{rr.Brier:.4f}',pc(gg.loc['global','allocation']),pc(gg.loc['global','recall']),pc(gg.loc['elevation_stratified','recall'])]))
tab('environment_monitoring','tab:envmonitor',r'环境分层的预测与监测覆盖。两种排序各选择 6,890 条清单；Global 为每折总体前 10\%，Stratified 为每折各海拔层约 10\%。',r'Environment-stratified prediction and monitoring coverage. Both policies select 6,890 checklists: Global ranks the top 10\% within each fold, and Stratified allocates approximately 10\% within each elevation band and fold.',r'Elevation & Model & Reports & AP & Brier & Global selected \% & Global recall \% & Stratified recall \%',rows)
print('Four figures and four tables written')
