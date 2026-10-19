"""Frozen Exp115: authenticated cached feature reuse, nine XGBoost fits, fixed0.5."""
from pathlib import Path
import sys,json,hashlib,time,platform,xml.etree.ElementTree as ET
import numpy as np,pandas as pd,joblib,xgboost as xgb,sklearn
from sklearn.base import clone
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,average_precision_score,roc_auc_score,balanced_accuracy_score,brier_score_loss,log_loss,confusion_matrix
OUT=Path(__file__).resolve().parents[1];ROOT=OUT.parents[1]
P77=ROOT/'experiments/077_jaad_two_stage_early_late_xgb_transfer'
P81=ROOT/'experiments/081_jaad_ego_compensated_road_motion'
P82=ROOT/'experiments/082_jaad_compensated_road_followup_evaluation'
P83=ROOT/'experiments/083_jaad_pcpa_aligned_exp082'
P99=ROOT/'experiments/099_jaad_exp077_vehicle_action'
COHORTS=['jaad_clean','jaad_beh','jaad_all'];SEEDS=[42,43,44]
CATEGORIES=['stopped','moving_slow','moving_fast','decelerating','accelerating']
SOURCES={};AUDITS=[];PARITY=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def source(p):
 p=Path(p).resolve();SOURCES[str(p)]=sha(p);return p
def dump(p,obj):
 p=OUT/p;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2)+'\n')
def authenticate(p,marker,key='outputs'):
 manifest=json.loads(source(marker).read_text());records=manifest[key] if isinstance(manifest,dict) else manifest
 match=[r for r in records if Path(r['path']).resolve()==Path(p).resolve()];assert len(match)==1,str(p)
 assert sha(source(p))==match[0]['sha256'],str(p);return Path(p)
def verify_hashes(hashes):
 for p,h in hashes.items():assert sha(p)==h,p

def contracts():
 road=json.loads(source(P83/'metrics/feature_contract.json').read_text())['predictor_columns']
 vehicle=json.loads(source(P99/'metrics/feature_contract.json').read_text())
 assert road[:182]==vehicle['A'] and len(road)==242
 extra=[f'vehicle_{stat}__{name}' for stat in ['fraction','first','last'] for name in CATEGORIES]
 assert vehicle['B'][182:]==extra
 return road,extra

def road_model(cohort,seed):
 if cohort=='jaad_clean':return authenticate(P81/f'models/variant_D_seed{seed}.joblib',P81/'metrics/model_manifest_pre_gate.json','models')
 return authenticate(P83/f'models/{cohort}_seed{seed}.joblib',P83/'metrics/trained_model_manifest.json','models')

def load_road(cohort,split):
 road,extra=contracts()
 if cohort=='jaad_clean':
  p=authenticate(P77/f'metrics/prepared_jaad_{split}.csv',P77/'metrics'/('.evaluate.complete.json' if split=='test' else '.prepare.complete.json'))
  raw=pd.read_csv(p)
  frames=[list(range(int(a),int(b)+1)) for a,b in zip(raw.start_frame,raw.end_frame)]
  assert all(len(f)==16 for f in frames)
  rows=raw[['sequence_id','track_id','video_id','label','start_frame','end_frame','tte_frames']].copy()
  rows['observed_frames_json']=[json.dumps(f) for f in frames]
  base=P82 if split=='test' else P81
  man=json.loads(source(base/'metrics'/('test_extraction_manifest.json' if split=='test' else 'extraction_manifest.json')).read_text())
  parts=[r for r in man['parts'] if r['split']==split];assert parts
  zs=[]
  for r in parts:
   assert sha(source(r['path']))==r['sha256'];zs.append(np.load(r['path'],allow_pickle=False))
  ids=np.concatenate([z['sequence_id'] for z in zs]).astype(str)
  assert len(set(ids))==len(ids) and set(ids)==set(rows.sequence_id)
  order=pd.Series(np.arange(len(ids)),index=ids).loc[rows.sequence_id].to_numpy()
  ctx=np.concatenate([z['road_compensated_60d'] for z in zs])[order]
  X=np.column_stack([raw[road[:182]].to_numpy(float),ctx])
 else:
  short='val' if split=='validation' else split
  p=authenticate(P83/f'cache/{cohort}_{short}_features.npz',P83/'metrics'/('.prepare-test.complete.json' if split=='test' else '.prepare-development.complete.json'))
  z=np.load(p,allow_pickle=False);assert z['predictor_columns'].tolist()==road
  rows=pd.read_csv(source(P83/f'manifests/{cohort}_{short}.csv'))
  assert not rows.sequence_id.duplicated().any()
  rows=rows.set_index('sequence_id').loc[z['sequence_id'].astype(str)].reset_index()
  np.testing.assert_array_equal(rows.label,z['label']);np.testing.assert_array_equal(rows.track_id,z['track_id'])
  X=z['X']
  rows=rows[['sequence_id','track_id','video_id','label','start_frame','end_frame','tte_frames','observed_frames_json']].copy()
 assert not rows.sequence_id.duplicated().any() and X.shape==(len(rows),242) and np.isfinite(X).all()
 assert rows.tte_frames.between(30,60).all() and set(rows.label.unique())=={0,1}
 gaps=[]
 for r in rows.itertuples(index=False):
  fr=np.array(json.loads(r.observed_frames_json));assert len(fr)==16 and (np.diff(fr)>0).all()
  assert (fr[0],fr[-1])==(r.start_frame,r.end_frame)
  if np.diff(fr).max()!=1:gaps.append((r.track_id,int(np.diff(fr).max())))
 expected=[('video_0098:0_98_544',134)]*2 if cohort=='jaad_all' and split=='train' else []
 assert gaps==expected,(cohort,split,gaps)
 return rows,X

def encode_vehicle(codes):
 codes=np.asarray(codes);assert codes.ndim==2 and codes.shape[1]==16 and np.isin(codes,np.arange(5)).all()
 oh=np.eye(5)[codes.astype(int)]
 return np.concatenate([oh.mean(1),oh[:,0],oh[:,-1]],axis=1)

def vehicle(rows):
 lookup={}
 for video in rows.video_id.unique():
  path=source(Path('/home/abdelrahman/Hadwa/datasets/JAAD/annotations_vehicle')/f'{video}_vehicle.xml')
  entries=ET.parse(path).getroot().findall('./frame')
  values={int(e.get('id')):CATEGORIES.index(e.get('action')) for e in entries}
  assert len(values)==len(entries);lookup[video]=values
 codes=np.array([[lookup[r.video_id][int(f)] for f in json.loads(r.observed_frames_json)] for r in rows.itertuples(index=False)])
 result=encode_vehicle(codes)
 assert result.shape==(len(rows),15) and np.isfinite(result).all()
 np.testing.assert_allclose(result[:,:5].sum(1),1);np.testing.assert_array_equal(result[:,5:10].sum(1),1);np.testing.assert_array_equal(result[:,10:].sum(1),1)
 return result,codes

def save_cache(cohort,split):
 rows,road=load_road(cohort,split);state,codes=vehicle(rows)
 X=np.column_stack([road,state]);assert X.shape==(len(rows),257)
 path=OUT/f'cache/{cohort}_{split}.npz';path.parent.mkdir(exist_ok=True)
 assert not path.exists(),str(path)
 np.savez_compressed(path,X=X,label=rows.label.to_numpy(int),sequence_id=rows.sequence_id.to_numpy(str))
 rows.to_csv(OUT/f'cache/{cohort}_{split}_rows.csv',index=False)
 AUDITS.append(dict(cohort=cohort,split=split,n=len(rows),positive=int(rows.label.sum()),negative=int((rows.label==0).sum()),tracks=rows.track_id.nunique(),videos=rows.video_id.nunique(),vehicle_annotation_coverage=1.0,no_dropped_rows=True,category_observation_counts=dict(zip(CATEGORIES,np.bincount(codes.ravel(),minlength=5).tolist()))))
 return rows,X

def load_cache(cohort,split):
 z=np.load(OUT/f'cache/{cohort}_{split}.npz',allow_pickle=False);rows=pd.read_csv(OUT/f'cache/{cohort}_{split}_rows.csv')
 np.testing.assert_array_equal(z['sequence_id'],rows.sequence_id);np.testing.assert_array_equal(z['label'],rows.label)
 return rows,z['X']

def predict(pipe,X,names):
 data=pd.DataFrame(X,columns=names) if hasattr(pipe,'feature_names_in_') else X
 return pipe.predict_proba(data)[:,1].astype(float)

def baseline(cohort,split,seed,rows,X,arm='R242'):
 road,extra=contracts()
 if arm=='V197':
  assert cohort=='jaad_clean';path=authenticate(P99/f'models/B_seed{seed}.joblib',P99/'VALIDATE_COMPLETE.json')
  probpath=authenticate(P99/f'predictions/{split}_B_seed{seed}.csv',P99/('EVALUATE_COMPLETE.json' if split=='test' else 'VALIDATE_COMPLETE.json'))
  saved=pd.read_csv(probpath);data=np.column_stack([X[:,:182],X[:,242:]]);names=road[:182]+extra;predcol='prediction'
 else:
  path=road_model(cohort,seed);data=X[:,:242];names=road
  if cohort=='jaad_clean':
   saved=pd.read_csv(source(P82/'predictions/test_A_D_all_seeds.csv' if split=='test' else P81/'predictions/validation_all_seeds.csv'))
   saved=saved[(saved.variant=='D')&(saved.seed==seed)];predcol='prediction'
  else:
   saved=pd.read_csv(authenticate(P83/f'predictions/{split}_all_seeds.csv',P83/'metrics'/('.evaluate.complete.json' if split=='test' else '.train-validate.complete.json')))
   saved=saved[(saved.cohort==cohort)&(saved.seed==seed)];predcol='prediction_fixed_0p5'
 assert not saved.sequence_id.duplicated().any() and set(saved.sequence_id)==set(rows.sequence_id)
 saved=saved.set_index('sequence_id').loc[rows.sequence_id].reset_index();np.testing.assert_array_equal(saved.label,rows.label)
 pipe=joblib.load(path);p=predict(pipe,data,names)
 np.testing.assert_allclose(p,saved.probability,rtol=1e-7,atol=1e-8);np.testing.assert_array_equal(p>=.5,saved[predcol])
 PARITY.append(dict(cohort=cohort,split=split,seed=seed,arm=arm,max_probability_error=float(np.abs(p-saved.probability).max()),ids_labels_match=True))
 # Retain the originally recorded probabilities; live inference only authenticates reuse.
 return pipe,saved.probability.to_numpy(float)

def metric(y,p):
 pred=p>=.5;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
 return dict(accuracy=accuracy_score(y,pred),precision=precision_score(y,pred,zero_division=0),recall=recall_score(y,pred),f1=f1_score(y,pred),pr_auc=average_precision_score(y,p),roc_auc=roc_auc_score(y,p),balanced_accuracy=balanced_accuracy_score(y,pred),brier=brier_score_loss(y,p),log_loss=log_loss(y,p),fpr=float(fp/(fp+tn)),fnr=float(fn/(fn+tp)),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))

def save_predictions(cohort,split,seed,arm,rows,p):
 f=rows[['sequence_id','video_id','track_id','label']].copy();f['cohort']=cohort;f['split']=split;f['seed']=seed;f['arm']=arm;f['probability']=p;f['prediction']=(p>=.5).astype(int);f['threshold']=.5
 path=OUT/f'predictions/{cohort}_{split}_{arm}_seed{seed}.csv';path.parent.mkdir(exist_ok=True);f.to_csv(path,index=False)

def isolation(cohort,splits):
 frames=[load_cache(cohort,s)[0] for s in splits]
 for i in range(len(frames)):
  for j in range(i):
   for col in ['sequence_id','video_id','track_id']:assert not set(frames[i][col])&set(frames[j][col]),(cohort,col)

def prepare():
 assert not (OUT/'FROZEN.json').exists()
 road,extra=contracts();dump('feature_contract.json',dict(R242=road,RV257=road+extra,vehicle_categories=CATEGORIES))
 plans=[]
 for cohort in COHORTS:
  for split in ['train','validation']:save_cache(cohort,split)
  isolation(cohort,['train','validation'])
  train,xt=load_cache(cohort,'train');val,xv=load_cache(cohort,'validation')
  for seed in SEEDS:
   parent,p=baseline(cohort,'validation',seed,val,xv)
   params=parent.named_steps['model'].get_params();weight=(train.label==0).sum()/(train.label==1).sum()
   assert params['random_state']==seed and params['n_estimators']==250 and params['max_depth']==4 and params['learning_rate']==.05
   assert params['early_stopping_rounds'] is None;np.testing.assert_allclose(params['scale_pos_weight'],weight,rtol=0,atol=1e-15)
   plans.append(dict(cohort=cohort,seed=seed,parent=str(road_model(cohort,seed)),params_repr={k:repr(v) for k,v in params.items()},training_rows=len(train)))
   save_predictions(cohort,'validation',seed,'R242',val,p)
   if cohort=='jaad_clean':
    _,p=baseline(cohort,'validation',seed,val,xv,'V197');save_predictions(cohort,'validation',seed,'V197',val,p)
 for p in [OUT/'DESIGN.md',Path(__file__),OUT/'feature_contract.json',*sorted((OUT/'cache').glob('*'))]:source(p)
 verify_hashes(SOURCES)
 dump('metrics/development_audit.json',AUDITS);dump('metrics/development_parity.json',PARITY)
 dump('FROZEN.json',dict(sources=SOURCES,models=plans,seed=SEEDS,threshold=.5,test_features_loaded_in_this_run=False,previous_test_exposure=True,versions=dict(python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__,xgboost=xgb.__version__)))
 print('PREPARE PASS: all development annotations covered; 12 baseline validation parity checks; nine model plans frozen',flush=True)

def train():
 assert not (OUT/'TRAINED.json').exists();freeze=json.loads((OUT/'FROZEN.json').read_text());verify_hashes(freeze['sources']);records=[];timing=[]
 for plan in freeze['models']:
  cohort,seed=plan['cohort'],plan['seed'];rows,X=load_cache(cohort,'train');val,V=load_cache(cohort,'validation');pipe=clone(joblib.load(plan['parent']))
  assert {k:repr(v) for k,v in pipe.named_steps['model'].get_params().items()}==plan['params_repr']
  path=OUT/f'models/{cohort}_seed{seed}.joblib';path.parent.mkdir(exist_ok=True);assert not path.exists()
  start=time.perf_counter();pipe.fit(X,rows.label.to_numpy(int));fit_seconds=time.perf_counter()-start
  np.testing.assert_allclose(pipe.named_steps['imputer'].statistics_,np.median(X,axis=0),rtol=0,atol=0)
  joblib.dump(pipe,path);reloaded=joblib.load(path);p=pipe.predict_proba(V)[:,1].astype(float);np.testing.assert_array_equal(p,reloaded.predict_proba(V)[:,1].astype(float))
  save_predictions(cohort,'validation',seed,'RV257',val,p)
  records.append(dict(cohort=cohort,seed=seed,path=str(path),sha256=sha(path),bytes=path.stat().st_size))
  timing.append(dict(cohort=cohort,seed=seed,fit_seconds=fit_seconds,model_bytes=path.stat().st_size))
  print('TRAINED',cohort,seed,'validation F1',metric(val.label,p)['f1'],flush=True)
 verify_hashes(freeze['sources']);dump('TRAINED.json',dict(models=records,frozen_sha256=sha(OUT/'FROZEN.json'),test_loaded_before_freeze=False))
 pd.DataFrame(timing).to_csv(OUT/'metrics/training_timing.csv',index=False)
 print('TRAIN PASS: nine persisted/reloaded models frozen before test feature loading',flush=True)

def evaluate():
 assert not (OUT/'EVALUATED.json').exists();freeze=json.loads((OUT/'FROZEN.json').read_text());verify_hashes(freeze['sources']);trained=json.loads((OUT/'TRAINED.json').read_text());assert trained['frozen_sha256']==sha(OUT/'FROZEN.json')
 for r in trained['models']:assert sha(r['path'])==r['sha256']
 timing=[]
 for cohort in COHORTS:
  rows,X=save_cache(cohort,'test');isolation(cohort,['train','validation','test'])
  for seed in SEEDS:
   _,p=baseline(cohort,'test',seed,rows,X);save_predictions(cohort,'test',seed,'R242',rows,p)
   if cohort=='jaad_clean':
    _,p=baseline(cohort,'test',seed,rows,X,'V197');save_predictions(cohort,'test',seed,'V197',rows,p)
   pipe=joblib.load(OUT/f'models/{cohort}_seed{seed}.joblib')
   start=time.perf_counter();p=pipe.predict_proba(X)[:,1].astype(float);seconds=time.perf_counter()-start
   save_predictions(cohort,'test',seed,'RV257',rows,p)
   timing.append(dict(cohort=cohort,seed=seed,n=len(rows),cached_batch_prediction_seconds=seconds,ms_per_window=seconds*1000/len(rows)))
   print('TEST',cohort,seed,metric(rows.label,p),flush=True)
 verify_hashes(SOURCES);verify_hashes(freeze['sources'])
 dump('metrics/test_audit.json',AUDITS);dump('metrics/test_parity.json',PARITY);dump('test_sources.json',SOURCES)
 pd.DataFrame(timing).to_csv(OUT/'metrics/prediction_timing.csv',index=False)
 records=[]
 for path in sorted((OUT/'predictions').glob('*.csv')):
  f=pd.read_csv(path);one=f.iloc[0]
  records.append(dict(cohort=one.cohort,split=one.split,seed=int(one.seed),arm=one.arm,n=len(f),threshold=.5,**metric(f.label,f.probability)))
 d=pd.DataFrame(records);assert len(d)==42;d.to_csv(OUT/'metrics/per_seed_results.csv',index=False)
 metrics=list(metric(np.array([0,1]),np.array([.1,.9])))
 s=d.groupby(['cohort','split','arm'])[metrics].agg(['mean','std']);s.columns=['_'.join(c) for c in s.columns];s.reset_index().to_csv(OUT/'metrics/summary_results.csv',index=False)
 deltas=[]
 for (cohort,split,seed),frame in d.groupby(['cohort','split','seed']):
  f=frame.set_index('arm')
  for arm in ['R242','V197']:
   if arm in f.index:deltas.append(dict(cohort=cohort,split=split,seed=int(seed),comparison='RV257_minus_'+arm,**{m:float(f.loc['RV257',m]-f.loc[arm,m]) for m in metrics}))
 pd.DataFrame(deltas).to_csv(OUT/'metrics/paired_deltas.csv',index=False)
 outputs=[p for sub in ['models','predictions','metrics','cache'] for p in sorted((OUT/sub).glob('*')) if p.is_file()]
 dump('EVALUATED.json',dict(model_count=9,prediction_files=42,threshold=.5,outputs={str(p):sha(p) for p in outputs}))
 print('EVALUATE PASS: all cohorts; 42 prediction groups; no selection or calibration',flush=True)

if __name__=='__main__':
 {'prepare':prepare,'train':train,'evaluate':evaluate}[sys.argv[1]]()
