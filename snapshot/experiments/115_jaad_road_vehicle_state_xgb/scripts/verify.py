"""Independent saved-artifact verification; does not import the experiment runner."""
from pathlib import Path
import json,hashlib,itertools
import numpy as np,pandas as pd,joblib
from sklearn.metrics import average_precision_score,roc_auc_score,log_loss
OUT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
freeze=json.loads((OUT/'FROZEN.json').read_text());test=json.loads((OUT/'test_sources.json').read_text());evaluated=json.loads((OUT/'EVALUATED.json').read_text());trained=json.loads((OUT/'TRAINED.json').read_text())
for records in [freeze['sources'],test,evaluated['outputs']]:
 for p,h in records.items():assert sha(p)==h,p
assert trained['frozen_sha256']==sha(OUT/'FROZEN.json')
assert len(trained['models'])==9
for r in trained['models']:assert sha(r['path'])==r['sha256']
cols=json.loads((OUT/'feature_contract.json').read_text());assert len(cols['RV257'])==len(set(cols['RV257']))==257 and cols['RV257'][:242]==cols['R242']
expected={(c,s,k,a) for c in ['jaad_clean','jaad_beh','jaad_all'] for s in ['validation','test'] for k in [42,43,44] for a in (['R242','RV257','V197'] if c=='jaad_clean' else ['R242','RV257'])}
metrics=pd.read_csv(OUT/'metrics/per_seed_results.csv');actual=set(zip(metrics.cohort,metrics.split,metrics.seed,metrics.arm));assert len(metrics)==42 and actual==expected
checks=[];max_metric_error=0
for c in ['jaad_clean','jaad_beh','jaad_all']:
 splitrows=[pd.read_csv(OUT/f'cache/{c}_{s}_rows.csv') for s in ['train','validation','test']]
 for a,b in itertools.combinations(splitrows,2):
  for col in ['video_id','track_id','sequence_id']:assert not set(a[col])&set(b[col])
 for seed in [42,43,44]:
  model=joblib.load(OUT/f'models/{c}_seed{seed}.joblib');tr=np.load(OUT/f'cache/{c}_train.npz')
  np.testing.assert_allclose(model.named_steps['imputer'].statistics_,np.median(tr['X'],axis=0),rtol=0,atol=0)
  params=model.named_steps['model'].get_params();plan=next(p for p in freeze['models'] if p['cohort']==c and p['seed']==seed)
  assert {k:repr(v) for k,v in params.items()}==plan['params_repr']
  np.testing.assert_allclose(params['scale_pos_weight'],(tr['label']==0).sum()/(tr['label']==1).sum(),rtol=0,atol=1e-15)
for p in sorted((OUT/'predictions').glob('*.csv')):
 d=pd.read_csv(p);r=d.iloc[0];z=np.load(OUT/f'cache/{r.cohort}_{r.split}.npz')
 assert not d.sequence_id.duplicated().any();np.testing.assert_array_equal(d.sequence_id,z['sequence_id']);np.testing.assert_array_equal(d.label,z['label']);assert d.threshold.eq(.5).all()
 y=d.label.to_numpy(int);prob=d.probability.to_numpy(float);pred=prob>=.5;np.testing.assert_array_equal(d.prediction,pred)
 assert np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
 if r.arm=='RV257':
  m=joblib.load(OUT/f'models/{r.cohort}_seed{r.seed}.joblib');live=m.predict_proba(z['X'])[:,1].astype(float)
  np.testing.assert_allclose(live,prob,rtol=1e-12,atol=1e-15)
 tp=int(((y==1)&pred).sum());tn=int(((y==0)&~pred).sum());fp=int(((y==0)&pred).sum());fn=int(((y==1)&~pred).sum())
 precision=tp/(tp+fp) if tp+fp else 0;recall=tp/(tp+fn);fpr=fp/(tn+fp);fnr=fn/(tp+fn)
 manual=dict(accuracy=(tp+tn)/len(y),precision=precision,recall=recall,f1=2*tp/(2*tp+fp+fn),balanced_accuracy=(recall+tn/(tn+fp))/2,pr_auc=average_precision_score(y,prob),roc_auc=roc_auc_score(y,prob),brier=np.mean((y-prob)**2),log_loss=log_loss(y,prob),fpr=fpr,fnr=fnr,tn=tn,fp=fp,fn=fn,tp=tp)
 row=metrics[(metrics.cohort==r.cohort)&(metrics.split==r.split)&(metrics.seed==r.seed)&(metrics.arm==r.arm)].iloc[0]
 for name,value in manual.items():
  error=abs(row[name]-value);max_metric_error=max(max_metric_error,error);assert error<1e-12,(p,name,error)
 checks.append(dict(cohort=r.cohort,split=r.split,seed=int(r.seed),arm=r.arm,n=len(y),verified=True))
summary=pd.read_csv(OUT/'metrics/summary_results.csv')
for row in summary.itertuples(index=False):
 subset=metrics[(metrics.cohort==row.cohort)&(metrics.split==row.split)&(metrics.arm==row.arm)];assert len(subset)==3
 for name in manual:
  np.testing.assert_allclose([subset[name].mean(),subset[name].std(ddof=1)],[getattr(row,name+'_mean'),getattr(row,name+'_std')],rtol=1e-12,atol=1e-12)
for row in pd.read_csv(OUT/'metrics/paired_deltas.csv').itertuples(index=False):
 base=row.comparison.removeprefix('RV257_minus_');subset=metrics[(metrics.cohort==row.cohort)&(metrics.split==row.split)&(metrics.seed==row.seed)].set_index('arm')
 for name in manual:assert abs(getattr(row,name)-(subset.loc['RV257',name]-subset.loc[base,name]))<1e-12
out=dict(status='PASS',models_verified=9,prediction_groups_verified=42,metric_cells_verified=42*len(manual),max_metric_error=float(max_metric_error),source_and_output_hashes_verified=True,train_only_imputation_and_class_weights_verified=True,paired_deltas_verified=True,checks=checks)
(OUT/'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='checks'},indent=2))
