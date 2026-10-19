"""Read-only retained-model audit. All generated artifacts stay beside this script in /tmp."""
import os,sys
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,csv,hashlib,importlib.util,io,zipfile,platform,collections
import numpy as np,torch,joblib,xgboost,sklearn
R=Path('/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean');O=Path(__file__).resolve().parents[1]
SOURCES={};COUNTS=[];TENSORS=[];TREES=[]
torch.set_num_threads(1);torch.set_num_interop_threads(1)
def exp(n):
 ps=list((R/'experiments').glob(f'{n:03d}_*'));assert len(ps)==1;return ps[0]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def source(p):
 p=Path(p).resolve();SOURCES[str(p)]={'bytes':p.stat().st_size,'sha256':sha(p)};return p
def mod(p,name):
 p=source(p);spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def writejson(p,obj):p.write_text(json.dumps(obj,indent=2,default=str)+'\n')
def writecsv(name,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with (O/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def tensor_bytes(m):
 seen=set();nb=0
 for t in list(m.parameters())+list(m.buffers()):
  key=(t.untyped_storage().data_ptr(),t.untyped_storage().nbytes())
  if key not in seen:nb+=key[1];seen.add(key)
 return nb
M53=mod(exp(53)/'scripts/exp053_core.py','audit_m53')
M95=mod(exp(95)/'scripts/exp095_core.py','audit_m95')
sys.path.insert(0,str(exp(84)/'scripts'))
PED=mod(exp(84)/'scripts/pedcmt_model.py','pedcmt_model')
BR=mod(exp(90)/'scripts/direct_fusion_model.py','audit_branch')

def make_neural(kind):
 if kind=='mslstm_base':return M53.MslstmBbox53Model('G')
 if kind=='mslstm_63':M53.STREAM_DIMS['biu_v2']=63;return M53.MslstmBbox53Model('I')
 if kind=='mslstm_56':M53.STREAM_DIMS['biu_v2']=56;return M53.MslstmBbox53Model('I')
 if kind=='mslstm_raw':return M95.MslstmBbox53ControlModel('J')
 if kind.startswith('ped_parent'):return PED.PedCMTTransfer(variant='A',ego_dim=1)
 if kind.startswith('ped_branch'):return BR.DirectFusionModel('C')
 raise ValueError(kind)

def specs():
 for seed in (42,43,44):
  for kind,n,stem in [('mslstm_base',53,'G'),('mslstm_63',53,'I'),('mslstm_raw',95,'J')]:yield kind,seed,exp(n)/f'checkpoints/{stem}_seed{seed}.best.pt'
  yield 'mslstm_56',seed,exp(117)/f'mslstm_pie/checkpoints/I_seed{seed}.best.pt'
  for ds,np_,nb,stem in [('pie',84,93,'R'),('jaad',89,91,'N')]:
   yield 'ped_parent_'+ds,seed,exp(np_)/f'checkpoints/A_seed{seed}.best.pt'
   yield 'ped_branch_'+ds,seed,exp(nb)/f'checkpoints/{stem}_seed{seed}.best.pt'

def nnrecord(kind,seed,m,paths,metadata,state=None):
 params=dict(m.named_parameters());state=m.state_dict() if state is None else state
 total=sum(p.numel() for p in params.values());trainable=sum(p.numel() for p in params.values() if p.requires_grad)
 b=io.BytesIO();torch.save(state,b);raw=b.getvalue();z=zipfile.ZipFile(io.BytesIO(raw))
 assert all(i.compress_type==zipfile.ZIP_STORED for i in z.infolist())
 if seed==42:(O/'inference_weights'/f'{kind}_seed42.state_dict.pt').write_bytes(raw)
 inferred_scalar=next(iter(params.values())).element_size()
 row=dict(kind=kind,seed=seed,source_paths=[str(p) for p in paths],total_parameters=total,trainable_parameters=trainable,frozen_parameters=total-trainable,
          branch_added_parameters=(total-6765321 if kind.startswith('mslstm_') and kind!='mslstm_base' else total if kind.startswith('ped_branch') else 100225 if kind.startswith('ped_complete') else 0),
          original_bytes=sum(p.stat().st_size for p in paths),inference_state_dict_bytes=len(raw),parameter_tensor_bytes=sum(p.numel()*p.element_size() for p in params.values()),
          registered_buffer_bytes=sum(t.numel()*t.element_size() for t in m.buffers()),tensor_storage_bytes=tensor_bytes(m),persistent_state_tensor_bytes=sum(t.numel()*t.element_size() for t in state.values()),
          precision=sorted(set(str(p.dtype) for p in params.values())),original_format='torch.save pickle + uncompressed ZIP_STORED tensor payloads',inference_format='torch.save state_dict; ZIP_STORED; lossless, uncompressed',optimizer_state_present=False,metadata=metadata)
 COUNTS.append(row)
 for name,p in params.items():TENSORS.append(dict(kind=kind,seed=seed,name=name,shape=list(p.shape),numel=p.numel(),bytes=p.numel()*p.element_size(),requires_grad=p.requires_grad,dtype=str(p.dtype)))
 return row

def audit_neural():
 loaded={}
 for kind,seed,p in specs():
  p=source(p);c=torch.load(p,map_location='cpu',weights_only=False)
  assert not any('optim' in k.lower() or 'scheduler' in k.lower() for k in c)
  assert all(i.compress_type==0 for i in zipfile.ZipFile(p).infolist())
  m=make_neural(kind);m.load_state_dict(c['model_state'],strict=True);m.eval()
  md={k:v for k,v in c.items() if k!='model_state'}
  if kind.startswith('ped_parent'):m.requires_grad_(False)
  row=nnrecord(kind,seed,m,[p],md)
  for k in ['total_parameters','parameter_count','trainable_parameters']:
   if k in c:assert c[k]==row['total_parameters']
  if kind.startswith('ped_'):loaded[kind,seed]=(m,p,c)
  print(kind,seed,row['total_parameters'],row['trainable_parameters'],row['inference_state_dict_bytes'],flush=True)
 for ds in ('pie','jaad'):
  for seed in (42,43,44):
   parent,pp,pc=loaded['ped_parent_'+ds,seed];branch,bp,bc=loaded['ped_branch_'+ds,seed]
   assert bc['parent_sha256']==sha(pp)
   composite=torch.nn.ModuleDict({'parent':parent,'correction':branch})
   nnrecord('ped_complete_'+ds,seed,composite,[pp,bp],{'cap_scale':.1,'correction_bound_before_scale':2.,'branch_parent_sha256_matches':True,'parent_original_training_all_requires_grad':True,'frozen_for_correction_training':True})

def audit_xgb():
 for kind,n,stem,dim in [('xgb_core',96,'v2',135),('xgb_road',116,'pie_road',195)]:
  for seed in (42,43,44):
   p=source(exp(n)/f'models/{stem}_seed{seed}.joblib');pipeline=joblib.load(p);m=pipeline.named_steps['model'];b=m.get_booster()
   cfg=json.loads(b.save_config());raw=json.loads(b.save_raw(raw_format='json'));ts=raw['learner']['gradient_booster']['model']['trees']
   dumps=[json.loads(t) for t in b.get_dump(dump_format='json',with_stats=True)];used=set();depths=[];splits=leaves=0
   for ti,t in enumerate(dumps):
    tc={'splits':0,'leaves':0,'max_depth':0}
    def visit(node,depth):
     tc['max_depth']=max(tc['max_depth'],depth)
     if 'leaf' in node:tc['leaves']+=1
     else:
      tc['splits']+=1;used.add(int(node['split'].removeprefix('f')))
      for child in node['children']:visit(child,depth+1)
    visit(t,0);assert tc['leaves']==tc['splits']+1
    splits+=tc['splits'];leaves+=tc['leaves'];depths.append(tc['max_depth']);TREES.append(dict(kind=kind,seed=seed,tree=ti,**tc))
   assert dim==len(pipeline.feature_names_in_)==int(raw['learner']['learner_model_param']['num_feature'])
   all_arrays=collections.Counter()
   for t in ts:
    for k,v in t.items():
     if isinstance(v,list):all_arrays[k]+=len(v)
   base=raw['learner']['learner_model_param']['base_score'];bv=np.asarray(json.loads(base) if base.startswith('[') else [float(base)],dtype=float)
   imputer=pipeline.named_steps['imputer']
   native=b.save_raw(raw_format='ubj')
   if seed==42:(O/'inference_weights'/f'{kind}_seed42.booster.ubj').write_bytes(native)
   row=dict(kind=kind,seed=seed,source_paths=[str(p)],input_features=dim,trees=len(ts),split_nodes=splits,leaves=leaves,total_nodes=splits+leaves,
       maximum_actual_depth=max(depths),mean_tree_maximum_depth=float(np.mean(depths)),unique_features_used=len(used),feature_indices_used=sorted(used),feature_names_used=[str(pipeline.feature_names_in_[i]) for i in sorted(used)],split_threshold_values=splits,leaf_output_values=leaves,
       numeric_tree_values=splits+leaves,stored_base_value_count=len(bv),numeric_tree_values_plus_stored_base=splits+leaves+len(bv),base_score_representation=base,base_score_probability_values=bv.tolist(),base_margin_logodds=np.log(bv/(1-bv)).tolist(),
       boost_from_average=cfg['learner']['learner_model_param'].get('boost_from_average'),original_bytes=p.stat().st_size,inference_state_dict_bytes=None,
       original_format='uncompressed joblib pickle: sklearn Pipeline(SimpleImputer + XGBClassifier)',original_header_hex=p.read_bytes()[:8].hex(),booster_only_ubj_bytes=len(native),optimizer_state_present=False,
       imputer_statistics_count=int(imputer.statistics_.size),imputer_statistics_bytes=int(imputer.statistics_.nbytes),imputer_strategy=imputer.strategy,
       native_serialized_array_lengths=dict(all_arrays),config=cfg,classifier_get_params=m.get_params(),native_model_metadata=raw['learner']['learner_model_param'])
   COUNTS.append(row);writejson(O/f'{kind}_seed{seed}_native_structure.json',raw);writejson(O/f'{kind}_seed{seed}_config.json',cfg)
   print(kind,seed,'trees',len(ts),'splits',splits,'leaves',leaves,'bytes',p.stat().st_size,flush=True)

def main():
 audit_neural();audit_xgb()
 for name,m in list(sys.modules.items()):
  p=getattr(m,'__file__',None)
  if p and str(p).startswith('/home/abdelrahman/Hadwa/') and Path(p).is_file():source(p)
 for p in [R/'thesis/14_RESULT_SOURCE_MAP.md',R/'thesis/manuscript/Tables/ch4_neural.tex',R/'thesis/manuscript/Tables/ch4_pedcmt_capped.tex',exp(93)/'scripts/run_experiment.py',exp(92)/'scripts/run_experiment.py',exp(117)/'scripts/run.py']:source(p)
 writejson(O/'counts.json',COUNTS);writecsv('counts.csv',[{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in COUNTS]);writecsv('neural_parameter_inventory.csv',TENSORS);writecsv('tree_inventory.csv',TREES)
 writejson(O/'sources.json',SOURCES)
 writejson(O/'environment.json',{'platform':platform.platform(),'python':sys.version,'pytorch':torch.__version__,'xgboost':xgboost.__version__,'sklearn':sklearn.__version__,'numpy':np.__version__,'joblib':joblib.__version__,'cpu':next(x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),'cpu_affinity':sorted(os.sched_getaffinity(0)),'loadavg':os.getloadavg(),'torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads()})
 for p,meta in SOURCES.items():assert sha(p)==meta['sha256'],p
 print('Audit count/size complete; source hashes unchanged.',flush=True)
if __name__=='__main__':main()
