from pathlib import Path
import json,numpy as np,pandas as pd
R=Path(__file__).resolve().parents[2];O=R/'results/environment_analysis'
metrics=pd.read_csv(O/'environment_prediction_metrics.csv');p=pd.read_csv(O/'oof_predictions_with_policies.csv');site=pd.read_csv(O/'occupancy_environment_summary.csv')
assert len(p)==68976 and p.y.sum()==3370 and p.SAMPLINGEVENTIDENTIFIER.is_unique
assert not p[['RandomForest','GLM-m4 (published)','LightGBM + terrain','XGBoost + terrain']].isna().any().any()
assert metrics.query("group_type=='elev_band' and model=='RandomForest'").n.sum()==68976
assert site.n_sites.sum()==1044
assert ((site.persistence>=0)&(site.persistence<=1)).all()
for method in ['RandomForest','GLM-m4 (published)','LightGBM + terrain','XGBoost + terrain']:
 for fold,a in p.groupby('fold'):
  budget=int(.1*len(a));assert a[method+'__global'].sum()==a[method+'__elevation_stratified'].sum()==budget
b=pd.read_csv(O/'missing_by_elevation.csv');assert b.n.sum()==89682 and b.reports.sum()==5174
v=json.loads((O/'oof_validation.json').read_text());assert v['recomputed_glm_fold0_max_error']<1e-8 and v['rf_disjoint_ids_and_blocks_all_folds']
a=pd.read_csv(O/'occupancy_refit_coefficients.csv');old=pd.read_csv(R/'results/baseline/occupancy_ordering_test.csv').query("mode=='site_major'")
assert np.max(np.abs(a.estimate.values-old.estimate.values))<1e-4
assert (pd.read_csv(O/'missing_adjustment.csv')[['beta','se','OR','lo','hi']].apply(np.isfinite)).all().all()
result={'checklist_totals_verified':True,'all_oof_predictions_complete':True,'identifiers_unique':True,'spatial_training_separation_verified':True,'equal_budgets_every_fold_and_model':True,'occupancy_refit_reproduced':True,'new_figures':4,'new_tables':4}
(O/'EXPERIMENT_QA.json').write_text(json.dumps(result,indent=2));print(result)
