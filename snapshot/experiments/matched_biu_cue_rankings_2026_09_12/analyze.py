"""Saved-model diagnostic; never fits or edits upstream artifacts."""
from pathlib import Path
import hashlib, importlib.util, json, re, sys
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
SOURCES = {}
SEEDS = (42, 43, 44)
SETTINGS = {
 'pie_096': ('PIE / Exp096 / temporal + speed', '096_pie_exp072_matched_biu_ablation'),
 'clean_098': ('Clean JAAD / Exp098 / BBox + BIU', '098_jaad_matched_biu_xgb'),
 'clean_vehicle_100': ('Clean JAAD / Exp100 / temporal + vehicle state', '100_jaad_exp099_biu_pathways'),
 'beh_102': ('JAAD_beh / Exp102 / BBox + BIU', '102_jaad_api_matched_biu'),
 'all_102': ('JAAD_all / Exp102 / BBox + BIU', '102_jaad_api_matched_biu'),
 'psi_113': ('PSI / Exp113 / BBox + BIU / native intent task', '113_psi_native_biu_xgb'),
}

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20), b''): h.update(b)
 return h.hexdigest()

def source(p):
 p=Path(p); p=p if p.is_absolute() else ROOT/p
 if str(p) not in SOURCES: SOURCES[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size}
 return p

def read(p): return pd.read_csv(source(p),float_precision='round_trip')
def js(p): return json.loads(source(p).read_text())
def arr(p): return np.load(source(p),allow_pickle=False)
def write(name, obj):
 p=OUT/name
 if p.exists(): raise FileExistsError(p)
 if isinstance(obj,pd.DataFrame): obj.to_csv(p,index=False)
 else: p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def module(p):
 spec=importlib.util.spec_from_file_location('rank_contract',source(p)); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

DEF='experiments/012_pie_biu_v2_30_60_benchmark/metrics/pie_biu_v2_feature_set_definitions.csv'

def bundles(setting,split):
 base=ROOT/'experiments'/SETTINGS[setting][1]
 native='validation' if split=='val' else 'test'
 if setting=='pie_096':
  contract=module(base/'scripts/feature_contract.py').build_matched_arm_columns(source(DEF),source('experiments/037_pie_joint_biu_temporal_ego/metrics/feature_contract.json'))
  rows=read(f'experiments/072_pie_joint_biu_temporal_ego_confirmation/metrics/prepared_{native}.csv')
  predictions=read(base/f'predictions/{native}_predictions.csv')
  for arm,version in [('v1','v1'),('v2','v2_core56'),('combined','combined108')]:
   names=list(contract[arm]);yield version,rows,rows[names].to_numpy(float),names,[(base/f'models/{arm}_seed{s}.joblib',predictions[(predictions.arm==arm)&(predictions.seed==s)],None) for s in SEEDS]
 elif setting=='clean_098':
  contract=js(base/'metrics/feature_contract.json');cache=ROOT/'experiments/089_pedcmt_standardized_jaad_biu_transfer/cache'/('test_source' if split=='test' else 'trainval_source')
  allrows=read(cache/'rows.csv');mask=allrows.split.eq(split).to_numpy();rows=allrows.loc[mask].reset_index(drop=True)
  meta=js(cache/'metadata.json'); assert contract['B'][53:]==['v1__'+n for n in meta['biu_v1_feature_names']]
  assert contract['C'][53:]==['v2__'+n for n in meta['biu_v2_feature_names']]
  parent=read(f'experiments/077_jaad_two_stage_early_late_xgb_transfer/metrics/prepared_jaad_{native}.csv')
  parent=parent.set_index('sequence_id').loc[rows.sequence_id];np.testing.assert_array_equal(parent.label,rows.label)
  bbox=parent[contract['A']].to_numpy(float);v1=arr(cache/'biu_v1.npy')[mask];v2=arr(cache/'biu_v2.npy')[mask]
  for arm,version,x in [('B','v1',np.column_stack([bbox,v1])),('C','v2_full63',np.column_stack([bbox,v2])),('D','union135',np.column_stack([bbox,v1,v2]))]:
   files=[(base/f'models/{arm}_seed{s}.joblib',read(base/f'predictions/{native}_{arm}_seed{s}.csv'),ROOT/f'experiments/jaad_biu_attribution_2026_09_06/contributions/clean_{split}_{s}.npz' if arm=='D' else None) for s in SEEDS]
   yield version,rows,x,contract[arm],files
 elif setting=='clean_vehicle_100':
  contract=js(base/'metrics/feature_contract.json');rows=read(base/f'cache/{native}/rows.csv')
  with arr(base/f'cache/{native}/features.npz') as z: matrices={a:z[a] for a in contract}
  for arm,version in [('v1','v1'),('v2','v2_core56'),('combined','combined108')]:
   modelbase=base if arm!='combined' else ROOT/'experiments/099_jaad_exp077_vehicle_action'
   modelarm=arm if arm!='combined' else 'B'
   yield version,rows,matrices[arm],contract[arm],[(modelbase/f'models/{modelarm}_seed{s}.joblib',read(base/f'predictions/{native}_{arm}_seed{s}.csv'),None) for s in SEEDS]
 elif setting in ('beh_102','all_102'):
  pop='jaad_beh' if setting=='beh_102' else 'jaad_all';contract=js(base/'metrics/feature_contract.json');rows=read(base/f'cache/{split}/{pop}_rows.csv')
  with arr(base/f'cache/{split}/{pop}.npz') as z: bbox,v1,v2=z['bbox'],z['v1'],z['v2']
  v2names=contract['D'][53:];indices=[v2names.index(n) for n in contract['C'][53:]]
  matrices={'B':np.column_stack([bbox,v1]),'C':np.column_stack([bbox,v2[:,indices]]),'D':np.column_stack([bbox,v2]),'E':np.column_stack([bbox,v1,v2])}
  for arm,version in [('B','v1'),('C','v2_core56'),('D','v2_full63'),('E','union135')]:
   files=[(base/f'models/{pop}_{arm}_{s}.joblib',read(base/f'predictions/{pop}_{split}_{arm}_{s}.csv'),ROOT/f'experiments/jaad_biu_attribution_2026_09_06/contributions/{pop}_{split}_{s}.npz' if arm=='E' else None) for s in SEEDS]
   yield version,rows,matrices[arm],contract[arm],files
 else:
  contract=js(base/'feature_contract.json');cache=base/('cache/val' if split=='val' else 'test_evaluation/cache');rows=read(cache/'rows.csv')
  for arm,version in [('B','v1'),('C','v2_full63')]:
   pred=base/('predictions' if split=='val' else 'test_evaluation/predictions')
   yield version,rows,arr(cache/f'{arm}.npy'),contract[arm],[(base/f'models/{arm}_seed{s}.joblib',read(pred/f'{native}_{arm}_seed{s}.csv'),None) for s in SEEDS]

def mapping(names,bbox):
 result=[]
 for n in names:
  if n in bbox or n.startswith(('obd_speed_','vehicle_','ego_','road__')): section,cue='context','context'
  elif n.startswith('temporal__'):
   c=re.sub(r'_(late_minus_early|early|late)$','',n[len('temporal__'):])
   section='context' if c.startswith('bbox_') else 'pose_temporal';cue=c
  else:
   lineage='v2' if n.startswith('v2__') else 'v1';c=re.sub(r'^v[12]__','',n)
   if c not in ('biu_valid_frame_count','biu_low_confidence_frame_count','direction_sign','was_flipped','direction_confidence'):
    c=re.sub(r'_(velocity_mean|velocity_std|mean|std|first|last|delta|min|max)$','',c)
   section='static_biu';cue=lineage+'__'+c
  result.append((section,cue))
 return result

def run():
 assert not (OUT/'verification.json').exists(),'Completed audit exists'
 source(__file__);source(OUT/'DESIGN.md')
 definitions=read(DEF).set_index('feature_set');bbox=set(json.loads(definitions.loc['bbox_only','feature_columns_json']))
 records=[];checks=[];membership=[];cohorts=[]
 for setting in SETTINGS:
  ids={}
  for split in ('val','test'):
   matched=None
   for version,rows,x,names,files in bundles(setting,split):
    assert x.shape==(len(rows),len(names)) and len(set(names))==len(names) and rows.sequence_id.is_unique
    identity=rows[['sequence_id','label']].reset_index(drop=True)
    if matched is None: matched=identity
    else: pd.testing.assert_frame_equal(matched,identity)
    ids[split]=set(rows.track_id)
    y=rows.label.to_numpy(int);assert set(y)=={0,1}
    cohorts.append(dict(setting=setting,split=split,version=version,windows=len(rows),tracks=rows.track_id.nunique(),positive=int(y.sum()),features=len(names)))
    pairs=mapping(names,bbox);static=np.array([s=='static_biu' for s,c in pairs]);assert static.sum() in (56,63,72,108,135)
    groups={p:np.array([i for i,q in enumerate(pairs) if p==q]) for p in dict.fromkeys(pairs) if p[0]!='context'}
    if split=='val':
     membership.extend(dict(setting=setting,version=version,index=i,feature=n,section=s,cue=c) for i,(n,(s,c)) in enumerate(zip(names,pairs)))
    for seed,(modelpath,saved,oldphi) in zip(SEEDS,files):
     assert saved.sequence_id.is_unique
     saved=saved.set_index('sequence_id').loc[rows.sequence_id];np.testing.assert_array_equal(saved.label,y)
     pipe=joblib.load(source(modelpath));booster=pipe.named_steps['model'].get_booster();booster.set_param({'nthread':4})
     imputer=pipe.named_steps['imputer']
     named=pd.DataFrame(x,columns=names) if hasattr(imputer,'feature_names_in_') else x
     transformed=imputer.transform(named);assert transformed.shape==x.shape
     dm=xgb.DMatrix(transformed,feature_names=booster.feature_names,nthread=4);prob=booster.predict(dm).astype(float);margin=booster.predict(dm,output_margin=True).astype(float)
     np.testing.assert_allclose(prob,saved.probability,rtol=0,atol=6e-8)
     if oldphi is not None:
      with arr(oldphi) as z:
       np.testing.assert_array_equal(z['sequence_id'],rows.sequence_id);np.testing.assert_array_equal(z['feature_names'],names);phi=z['shap'].astype(float)
     else: phi=booster.predict(dm,pred_contribs=True,approx_contribs=False).astype(float)
     assert phi.shape==(len(rows),len(names)+1)
     np.testing.assert_allclose(phi.sum(1),margin,rtol=1e-5,atol=2e-5)
     np.testing.assert_allclose(1/(1+np.exp(-margin)),prob,rtol=1e-6,atol=1e-7)
     checks.append(dict(setting=setting,split=split,version=version,seed=seed,n=len(rows),reused_shap=oldphi is not None,max_probability_error=float(np.abs(prob-saved.probability).max()),max_additivity_error=float(np.abs(phi.sum(1)-margin).max())))
     masks={'ALL':np.ones(len(y),bool),'positive':y==1,'negative':y==0,'TP':(y==1)&(prob>=.5),'FN':(y==1)&(prob<.5),'TN':(y==0)&(prob<.5),'FP':(y==0)&(prob>=.5)}
     masses=np.abs(phi[:,:-1]);net=np.column_stack([phi[:,idx].sum(1) for idx in groups.values()]);aligned=net*(2*y-1)[:,None]
     # Save compact cue-level values for future diagnostics; do not copy full feature arrays.
     np.savez_compressed(OUT/f'cue_values_{setting}_{split}_{version}_{seed}.npz',sequence_id=rows.sequence_id.to_numpy(str),track_id=rows.track_id.to_numpy(str),label=y,probability=prob,cue=np.array([s+'|'+c for s,c in groups]),contribution=net,baseline=phi[:,-1])
     for subset,mask in masks.items():
      if not mask.any():continue
      absfeatures=masses[mask].mean(0);total=absfeatures.sum();staticmass=absfeatures[static].sum()
      for j,((section,cue),idx) in enumerate(groups.items()):
       a=aligned[mask,j];track=pd.DataFrame({'t':rows.track_id.to_numpy()[mask],'a':a}).groupby('t').a.mean().mean()
       records.append(dict(setting=setting,split=split,version=version,seed=seed,subset=subset,section=section,cue=cue,n=int(mask.sum()),n_features=len(idx),importance_pct=100*absfeatures[idx].sum()/staticmass if section=='static_biu' else np.nan,all_input_pct=100*absfeatures[idx].sum()/total,toward_pct=100*np.mean(a>1e-10),away_pct=100*np.mean(a< -1e-10),neutral_pct=100*np.mean(np.abs(a)<=1e-10),support_mean=np.maximum(a,0).mean(),opposition_mean=np.maximum(-a,0).mean(),net_mean=a.mean(),track_equal_net=track))
     print(f'{setting}/{split}/{version}/{seed}: parity and attribution verified ({len(rows)} windows)',flush=True)
   if split=='test':assert not ids['val'] & ids['test']
 d=pd.DataFrame(records);write('per_seed.csv',d);write('checks.csv',pd.DataFrame(checks));write('membership.csv',pd.DataFrame(membership));write('cohorts.csv',pd.DataFrame(cohorts))
 keys=['setting','split','version','subset','section','cue'];metrics=['importance_pct','all_input_pct','toward_pct','away_pct','neutral_pct','support_mean','opposition_mean','net_mean','track_equal_net']
 summary=d.groupby(keys,sort=False)[metrics].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns];summary=summary.reset_index();write('summary.csv',summary)
 # Separate equal-class signed ranking from model reliance ranking.
 classkeys=['setting','split','version','section','cue','seed']
 balanced=d[d.subset.isin(['positive','negative'])].groupby(classkeys).net_mean.mean().reset_index(name='balanced_net')
 balanced=balanced.groupby(classkeys[:-1]).balanced_net.agg(['mean','std',lambda a:int((a>0).sum())]).reset_index().rename(columns={'mean':'balanced_net_mean','std':'balanced_net_std','<lambda_0>':'positive_seeds'})
 ranked=summary[summary.subset=='ALL'].merge(balanced,on=classkeys[:-1])
 rankkeys=['setting','split','version','section']
 ranked['importance_rank']=ranked.groupby(rankkeys).importance_pct_mean.rank(method='min',ascending=False)
 ranked['support_rank']=ranked.groupby(rankkeys).balanced_net_mean.rank(method='min',ascending=False)
 write('rankings.csv',ranked)
 np.testing.assert_allclose(d.support_mean-d.opposition_mean,d.net_mean,rtol=0,atol=1e-12)
 np.testing.assert_allclose(d.toward_pct+d.away_pct+d.neutral_pct,100,rtol=0,atol=1e-10)
 np.testing.assert_allclose(d[d.section=='static_biu'].groupby(['setting','split','version','seed','subset']).importance_pct.sum(),100,rtol=0,atol=1e-9)
 assert len(checks)==114
 for p,r in SOURCES.items():assert sha(p)==r['sha256'],p
 write('sources.json',SOURCES)
 write('verification.json',dict(model_split_seed_checks=len(checks),reused_shap=sum(c['reused_shap'] for c in checks),new_exact_shap=sum(not c['reused_shap'] for c in checks),source_files=len(SOURCES),sources_unchanged=True,no_fits=True,no_threshold_changes=True,xgboost=xgb.__version__,max_probability_error=max(c['max_probability_error'] for c in checks),max_additivity_error=max(c['max_additivity_error'] for c in checks)))

if __name__=='__main__':run()
