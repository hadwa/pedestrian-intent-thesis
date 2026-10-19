from pathlib import Path
import os,sys,json,csv,hashlib,importlib.util,time,platform,argparse,warnings
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits,threadpool_info
R=Path(__file__).resolve().parents[2];O=Path(__file__).resolve().parent
E96=R/'experiments/096_pie_exp072_matched_biu_ablation';E116=R/'experiments/116_xgboost_v2_core_completion';E75=R/'experiments/075_pie_twinlitenetplus_road_context'
SOURCES={}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def source(p):
 p=Path(p).resolve();SOURCES[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size};return p
def module(p,name):
 spec=importlib.util.spec_from_file_location(name,source(p));m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def writej(n,x):(O/n).write_text(json.dumps(x,indent=2,allow_nan=False))
def setup():
 sys.path.insert(0,str(R));sys.path.insert(0,str(R/'src'))
 raw=module(R/'experiments/037_pie_joint_biu_temporal_ego/scripts/run_experiment.py','runtime037')
 box=module('/home/abdelrahman/Hadwa/BIU-PID-Distillation/src/features/bbox_features.py','runtime_bbox')
 sys.path.insert(0,str(E75/'scripts'));ctx=module(E75/'scripts/context.py','runtime_context')
 for p in [R/'src/biu_v2_extractor.py',R/'src/biu_extractor.py',R/'experiments/037_pie_joint_biu_temporal_ego/scripts/feature_core.py',R/'experiments/063_pie_corrected_yolop_road_relative_context/scripts/representations.py']:source(p)
 return raw,box,ctx
RAW,BOX,CTX=setup()
def prepare():
 assert not (O/'raw_runs.csv').exists()
 ids=pd.read_csv(source(R/'experiments/aibthings_model_only_runtime/selected_sample_ids.csv'))
 base=pd.read_csv(source(E116/'cache/pie_road/test/rows.csv')).set_index('sequence_id').loc[ids.sequence_id].reset_index()
 assert len(base)==100 and base.sequence_id.is_unique
 for c in ids:
  if c in base:np.testing.assert_array_equal(base[c],ids[c])
 names=json.loads(source(E116/'cache/pie_road/test/columns.json').read_text())
 cached=np.load(source(E116/'cache/pie_road/test/features.npy'))
 allrows=pd.read_csv(E116/'cache/pie_road/test/rows.csv'); ix=allrows.set_index('sequence_id').assign(index=np.arange(len(allrows))).loc[base.sequence_id,'index'].to_numpy();expected=cached[ix]
 needed=RAW._needed_keys(base);print('Loading original keypoint observations...',flush=True)
 source(RAW.KEYPOINTS_PATH);kp=RAW._load_keypoints(needed,{'set03'})
 boxes=RAW._load_bboxes(base,needed);obd=RAW._load_obd(base)
 for se,vid in sorted(set(zip(base.set_id,base.video_id))):
  source(RAW.ANNOTATIONS_DIR/se/f'{vid}_annt.xml');source(RAW.OBD_DIR/se/f'{vid}_obd.xml')
 data={k:[] for k in ['coords','conf','boxes','speed','frames','road']}
 parts={}
 for row in base.itertuples(index=False):
  frames=np.arange(row.start_frame,row.end_frame+1);keys=[(row.set_id,row.video_id,row.ped_id,int(f)) for f in frames]
  for key,val in [('coords',[kp[k]['keypoints'] for k in keys]),('conf',[kp[k]['kp_conf'] for k in keys]),('boxes',[boxes[k] for k in keys]),('speed',[obd[(row.set_id,row.video_id)][int(f)] for f in frames]),('frames',frames)]:data[key].append(val)
  if row.video_id not in parts:
   p=source(E75/'cache/test_parts'/f'{row.set_id}_{row.video_id}.npz');z=np.load(p,allow_pickle=False);parts[row.video_id]={str(s):v for s,v in zip(z['sequence_id'],z['road_relative'])}
  data['road'].append(parts[row.video_id][row.sequence_id])
 data={k:np.asarray(v) for k,v in data.items()};data['expected']=expected
 np.savez_compressed(O/'observations.npz',**data);base.to_csv(O/'sample_ids.csv',index=False);writej('columns.json',names)
 writej('sources.json',SOURCES);print('Prepared 100 unchanged observation windows.',flush=True)
def load():
 global SOURCES
 SOURCES.update(json.loads((O/'sources.json').read_text()));z={k:v for k,v in np.load(O/'observations.npz').items()};names=json.loads((O/'columns.json').read_text());base=pd.read_csv(O/'sample_ids.csv')
 names_by={'core':names[:135],'road':names};assert len(names)==195
 models={}
 for kind in ['core','road']:
  for seed in [42,43,44]:
   p=source((E96/f'models/v2_seed{seed}.joblib') if kind=='core' else (E116/f'models/pie_road_seed{seed}.joblib'))
   m=joblib.load(p);assert list(m.feature_names_in_)==names_by[kind];models[(kind,seed)]=m
 return z,names_by,base,models
def parts(z,i):
 b=BOX.compute_bbox_motion_features(z['boxes'][i]);v=RAW.extract_biu_v2(z['coords'][i],z['conf'][i],bbox_sequence=z['boxes'][i]);t=RAW.feature_core.temporal_contrast_features(z['coords'][i],z['conf'][i],z['boxes'][i],extractor=RAW.extract_biu_v2);e=RAW.feature_core.obd_speed_features(dict(zip(z['frames'][i],z['speed'][i])),z['frames'][i]);rd=CTX.summarize_road(z['road'][i]);return b,v,t,e,rd
def pack(b,v,t,e,rd,names):
 d=dict(b);d.update({'v2__'+k:val for k,val in v.items()});d.update({'temporal__'+k:val for k,val in t.items()});d.update(e)
 if len(names)==195:d.update(zip(CTX.context_columns(),rd))
 return pd.DataFrame(np.asarray([[d[n] for n in names]],dtype=np.float64),columns=names)
def verify():
 z,names,base,models=load();X=np.concatenate([pack(*parts(z,i),names['road']).to_numpy() for i in range(100)])
 diff=np.abs(X-z['expected']);np.testing.assert_allclose(X,z['expected'],rtol=1e-6,atol=1e-6)
 pd.DataFrame({'feature':names['road'],'max_abs_error':diff.max(axis=0)}).to_csv(O/'feature_parity.csv',index=False)
 out=[]
 for (kind,seed),m in models.items():
  frame=pd.DataFrame(X[:,:len(names[kind])],columns=names[kind]);p=m.predict_proba(frame)[:,1];m.named_steps['model'].set_params(n_jobs=1);one=m.predict_proba(frame)[:,1];np.testing.assert_array_equal(p,one)
  path=source(E96/'predictions/test_predictions.csv' if kind=='core' else E116/f'predictions/pie_road_test_seed{seed}.csv');saved=pd.read_csv(path)
  if kind=='core':saved=saved[(saved.arm=='v2')&(saved.seed==seed)]
  saved=saved.set_index('sequence_id').loc[base.sequence_id];np.testing.assert_array_equal(saved.label,base.label);error=float(np.max(np.abs(p-saved.probability.to_numpy())));assert error<1e-6,(kind,seed,error)
  out.append(dict(model=kind,seed=seed,max_probability_error=error,worker_count_prediction_parity=True,rows=100))
 pd.DataFrame(out).to_csv(O/'prediction_parity.csv',index=False);writej('sources.json',SOURCES);writej('PARITY_PASS.json',dict(max_feature_error=float(diff.max()),models=6,windows=100));print('Feature and prediction parity PASS.',flush=True)
RAW_ROWS=[]
def record(stage,case,seed,batch,rep,elapsed):RAW_ROWS.append(dict(stage=stage,case=case,seed=seed,batch=batch,repeat=rep,elapsed_ms=elapsed))
def timecall(fn):
 start=time.perf_counter_ns();x=fn();return x,(time.perf_counter_ns()-start)/1e6
def timing():
 assert (O/'PARITY_PASS.json').exists();assert not (O/'raw_runs.csv').exists(),'outputs immutable; use a new folder for a repeat'
 z,names,base,models=load()
 for m in models.values():m.named_steps['model'].set_params(n_jobs=1)
 cpu=next((x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),'unknown')
 env=dict(cpu=cpu,platform=platform.platform(),python=sys.version,numpy=np.__version__,pandas=pd.__version__,xgboost=__import__('xgboost').__version__,sklearn=__import__('sklearn').__version__,logical_cpus=os.cpu_count(),affinity=sorted(os.sched_getaffinity(0)),threadpools=threadpool_info(),xgboost_threads=1,loadavg_before=os.getloadavg(),started=time.strftime('%Y-%m-%d %H:%M:%S %z'),input_dtype='float64 DataFrame; XGBoost internal precision unchanged')
 writej('environment.json',env)
 print('Timing feature components: 200 warm-up + 2000 windows...',flush=True)
 for rep in range(-200,2000):
  i=rep%100
  b,tb=timecall(lambda:BOX.compute_bbox_motion_features(z['boxes'][i]))
  v,tv=timecall(lambda:RAW.extract_biu_v2(z['coords'][i],z['conf'][i],bbox_sequence=z['boxes'][i]))
  t,tt=timecall(lambda:RAW.feature_core.temporal_contrast_features(z['coords'][i],z['conf'][i],z['boxes'][i],extractor=RAW.extract_biu_v2))
  e,te=timecall(lambda:RAW.feature_core.obd_speed_features(dict(zip(z['frames'][i],z['speed'][i])),z['frames'][i]))
  rd,tr=timecall(lambda:CTX.summarize_road(z['road'][i]))
  for kind in ['core','road']:
   _,tp=timecall(lambda:pack(b,v,t,e,rd,names[kind]))
   if rep>=0:record('fusion_name_packing',kind,-1,1,rep,tp)
  if rep>=0:
   for stage,el in [('bbox53',tb),('biu56',tv),('temporal21',tt),('ego5',te),('road_summary60_from_geometry',tr)]:record(stage,'shared',-1,1,rep,el)
  if rep in [499,999,1499,1999]:print('Feature windows:',rep+1,flush=True)
 for (kind,seed),m in models.items():
  print('Timing cached predictor',kind,seed,flush=True)
  full=pd.DataFrame(z['expected'][:,:len(names[kind])],columns=names[kind]);singles=[full.iloc[i:i+1].copy() for i in range(100)]
  for batch,warm,reps in [(1,200,2000),(100,20,200)]:
   for rep in range(-warm,reps):
    frame=singles[rep%100] if batch==1 else full
    _,el=timecall(lambda:m.predict_proba(frame))
    if rep>=0:record('cached_pipeline_predict',kind,seed,batch,rep,el)
 for kind in ['core','road']:
  m=models[(kind,42)];print('Timing actual sequential window workload',kind,flush=True)
  def sequential(i):
   b=BOX.compute_bbox_motion_features(z['boxes'][i]);v=RAW.extract_biu_v2(z['coords'][i],z['conf'][i],bbox_sequence=z['boxes'][i]);t=RAW.feature_core.temporal_contrast_features(z['coords'][i],z['conf'][i],z['boxes'][i],extractor=RAW.extract_biu_v2);e=RAW.feature_core.obd_speed_features(dict(zip(z['frames'][i],z['speed'][i])),z['frames'][i]);rd=CTX.summarize_road(z['road'][i]) if kind=='road' else None
   return m.predict_proba(pack(b,v,t,e,rd,names[kind]))
  for rep in range(-100,1000):
   _,el=timecall(lambda:sequential(rep%100))
   if rep>=0:record('sequential_from_ready_perception',kind,42,1,rep,el)
   if rep in [499,999]:print(kind,'sequential windows:',rep+1,flush=True)
 pd.DataFrame(RAW_ROWS).to_csv(O/'raw_runs.csv',index=False)
 env.update(loadavg_after=os.getloadavg(),ended=time.strftime('%Y-%m-%d %H:%M:%S %z'));writej('environment.json',env)
 for p,rec in SOURCES.items():
  if '/models/' in p:assert sha(p)==rec['sha256']
 writej('sources.json',SOURCES);writej('TIMING_COMPLETE.json',dict(rows=len(RAW_ROWS),models=6,no_refit=True,model_hashes_unchanged=True))
 print('Timing complete.',flush=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','verify','time']);args=parser.parse_args()
 with threadpool_limits(limits=1):{'prepare':prepare,'verify':verify,'time':timing}[args.phase]()
