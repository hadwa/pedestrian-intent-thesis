"""Exact diagnostic branch attribution for all nine frozen Exp115 models."""
from pathlib import Path
import hashlib,json
import joblib,numpy as np,pandas as pd,xgboost as xgb
BASE=Path(__file__).resolve().parents[1];ROOT=BASE.parents[1];OUT=BASE/'attribution'
OUT.mkdir(exist_ok=True)
assert not (OUT/'summary.csv').exists(),'Refuse to overwrite completed audit'
sources={}
def source(path):
 p=Path(path);sources[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest();return p
verified=json.loads(source(BASE/'INDEPENDENT_VERIFICATION.json').read_text());assert verified['status']=='PASS'
trained=json.loads(source(BASE/'TRAINED.json').read_text())
for r in trained['models']:assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256']
names=json.loads(source(BASE/'feature_contract.json').read_text())['RV257']
bbox=set(json.loads(source(ROOT/'experiments/113_psi_native_biu_xgb/feature_contract.json').read_text())['A'])
def group(n):
 if n.startswith('road_compensated__'):return 'Road context'
 if n.startswith('vehicle_'):return 'Vehicle state'
 if n.startswith('temporal__'):return 'Explicit temporal'
 if n in bbox:return 'BBox'
 return 'Combined BIU package'
groups=['BBox','Combined BIU package','Explicit temporal','Road context','Vehicle state']
mapping=pd.DataFrame(dict(index=range(257),feature=names,group=[group(n) for n in names]));mapping.to_csv(OUT/'feature_groups.csv',index=False)
assert mapping.groupby('group').size().to_dict()==dict(zip(groups,[53,108,21,60,15]))
idx={g:np.flatnonzero(mapping.group.to_numpy()==g) for g in groups}
rows=[];feature_rows=[];checks=[]
for cohort in ['jaad_clean','jaad_beh','jaad_all']:
 exp='115_'+cohort.removeprefix('jaad_')
 for split in ['validation','test']:
  cache=np.load(source(BASE/f'cache/{cohort}_{split}.npz'),allow_pickle=False)
  labels=pd.DataFrame({'sequence_id':cache['sequence_id'],'label':cache['label']})
  matrix=cache['X'];assert np.isfinite(matrix).all()
  y=labels.label.to_numpy(int);assert matrix.shape==(len(y),257)
  for seed in [42,43,44]:
   pipe=joblib.load(source(BASE/f'models/{cohort}_seed{seed}.joblib'));booster=pipe.named_steps['model'].get_booster();booster.set_param({'nthread':4})
   data=pipe.named_steps['imputer'].transform(matrix)
   dm=xgb.DMatrix(data,feature_names=booster.feature_names)
   p=booster.predict(dm);margin=booster.predict(dm,output_margin=True).astype(float)
   saved=pd.read_csv(source(BASE/f'predictions/{cohort}_{split}_RV257_seed{seed}.csv'))
   assert np.array_equal(saved.sequence_id,labels.sequence_id);assert np.array_equal(saved.label,y)
   assert np.array_equal(p>=.5,saved.prediction)
   np.testing.assert_allclose(p,saved.probability,rtol=1e-12,atol=1e-15)
   phi=booster.predict(dm,pred_contribs=True,approx_contribs=False).astype(float)
   np.testing.assert_allclose(phi.sum(1),margin,rtol=1e-5,atol=2e-5)
   np.testing.assert_allclose(1/(1+np.exp(-margin)),p,rtol=1e-6,atol=1e-7)
   grouped=np.column_stack([phi[:,idx[g]].sum(1) for g in groups])
   np.testing.assert_allclose(grouped.sum(1)+phi[:,-1],margin,rtol=1e-5,atol=2e-5)
   np.savez_compressed(OUT/f'groups_{exp}_{split}_seed{seed}.npz',sequence_id=labels.sequence_id.to_numpy(str),label=y,probability=p,baseline=phi[:,-1],group_contribution=grouped,groups=np.array(groups))
   pred=p>=.5;masks={'ALL':np.ones(len(y),bool),'positive':y==1,'negative':y==0,'TP':(y==1)&pred,'FN':(y==1)&~pred,'TN':(y==0)&~pred,'FP':(y==0)&pred}
   for label_group,mask in masks.items():
    assert mask.any();magnitudes=np.abs(grouped[mask]).mean(0);mass=np.array([np.abs(phi[mask][:,idx[g]]).sum(1).mean() for g in groups]);assert magnitudes.sum()>0
    for j,g in enumerate(groups):
     aligned=(2*y[mask]-1)*grouped[mask,j]
     rows.append(dict(experiment=exp,split=split,seed=seed,label_group=label_group,group=g,n=int(mask.sum()),n_features=len(idx[g]),mean_abs_net= magnitudes[j],net_share_pct=100*magnitudes[j]/magnitudes.sum(),sum_abs_features=mass[j],mass_share_pct=100*mass[j]/mass.sum(),aligned_mean=aligned.mean(),toward_pct=100*(aligned>1e-10).mean(),away_pct=100*(aligned< -1e-10).mean(),neutral_pct=100*(np.abs(aligned)<=1e-10).mean()))
   for j,n in enumerate(names):feature_rows.append(dict(experiment=exp,split=split,seed=seed,feature=n,group=group(n),mean_abs=np.abs(phi[:,j]).mean()))
   checks.append(dict(experiment=exp,split=split,seed=seed,n=len(y),max_probability_error=float(np.abs(p-saved.probability).max()),max_additivity_error=float(np.abs(phi.sum(1)-margin).max())))
   print(exp,split,seed,'verified',flush=True)
d=pd.DataFrame(rows);d.to_csv(OUT/'per_seed.csv',index=False)
keys=['experiment','split','label_group','group'];metrics=['mean_abs_net','net_share_pct','sum_abs_features','mass_share_pct','aligned_mean','toward_pct','away_pct','neutral_pct']
s=d.groupby(keys,sort=False)[metrics].agg(['mean','std']);s.columns=['_'.join(c) for c in s.columns];s=s.reset_index();s.to_csv(OUT/'summary.csv',index=False)
f=pd.DataFrame(feature_rows);f.to_csv(OUT/'feature_per_seed.csv',index=False);f.groupby(['experiment','split','feature','group'],sort=False).mean_abs.agg(['mean','std']).reset_index().to_csv(OUT/'feature_summary.csv',index=False)
pd.DataFrame(checks).to_csv(OUT/'verification.csv',index=False)
for p,h in sources.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h
(OUT/'sources.json').write_text(json.dumps(sources,indent=2)+'\n')
print(s[(s.split=='test')&(s.label_group=='ALL')][['experiment','group','net_share_pct_mean','net_share_pct_std','toward_pct_mean']].to_string(index=False))
