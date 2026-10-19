"""CPU/one-thread/batch-one audit with cached real inputs, prediction replay and no fitting."""
import os,sys,time,gc,json
sys.dont_write_bytecode=True
import audit_models as A
from pathlib import Path
import numpy as np,pandas as pd,torch,joblib
from threadpoolctl import threadpool_limits,threadpool_info
O=A.O;S=[];RAW=[];PARITY=[]
# Pin this process only; no machine-wide affinity/configuration changes.
affinity=sorted(os.sched_getaffinity(0));cpu=affinity[min(2,len(affinity)-1)];os.sched_setaffinity(0,{cpu})
def arr(p):return np.load(A.source(p),mmap_mode='r')
def csv(p):return pd.read_csv(A.source(p))
def js(p):return json.loads(A.source(p).read_text())
N=32
piecache=A.exp(50)/'cache/set03_final';pierows=csv(piecache/'rows.csv');inds=np.linspace(0,len(pierows)-1,N,dtype=int);ids=pierows.iloc[inds].sequence_id.astype(str).tolist()
arrays={k:np.array(arr(piecache/f'{k}.npy')[inds],np.float32) for k in ['bbox','ego_speed','ego_acc','yolop','pose','biu_v1','biu_v2']}
arrays['bbox53']=np.array(arr(A.exp(53)/'cache/bbox53_test.npy')[inds],np.float32)
index56=js(A.exp(117)/'mslstm_pie/metrics/feature_contract.json')['original_indices']
# XGB core columns are exactly the retained prefix of this road cache; names asserted below.
xcache=A.exp(116)/'cache/pie_road/test';xr=csv(xcache/'rows.csv');xind=xr.set_index('sequence_id').assign(pos=np.arange(len(xr))).loc[ids,'pos'].to_numpy();xf=np.array(arr(xcache/'features.npy')[xind]);xn=js(xcache/'columns.json')
class Capped(torch.nn.Module):
 def __init__(self,parent,branch):
  super().__init__();self.parent=parent;self.branch=branch;self.anchor=None
  self.parent.endpoint.register_forward_pre_hook(self.capture)
 def capture(self,module,args):self.anchor=args[0]
 def forward(self,bbox,ego,biu):
  logits=self.parent(bbox,ego)[0];_,delta=self.branch(logits,self.anchor,biu)
  return torch.sigmoid(logits+.1*delta)
def sliced_dict(d):return [{k:torch.from_numpy(np.ascontiguousarray(v[i:i+1])) for k,v in d.items()} for i in range(N)]
def parity(kind,seed,got,saved,sids,prob='probability',filter_args=None):
 df=csv(saved)
 if filter_args:
  for k,v in filter_args.items():df=df[df[k]==v]
 assert not df.sequence_id.duplicated().any()
 expected=df.set_index('sequence_id').loc[sids,prob].to_numpy();error=float(np.max(np.abs(np.asarray(got).reshape(-1)-expected)))
 assert error<1e-4,(kind,seed,error)
 assert np.array_equal(np.asarray(got).reshape(-1)>=.5,expected>=.5),(kind,seed,'threshold decision mismatch')
 row=dict(kind=kind,seed=seed,windows=len(sids),max_probability_error=error,reference=str(saved),column=prob,tolerance=1e-4,threshold_decisions_match=True);PARITY.append(row);print('PARITY',kind,seed,error,flush=True)
def measure(kind,seed,fn,items):
 gc.collect();old=gc.isenabled();gc.disable()
 try:
  with torch.inference_mode():
   for block in range(3):
    for i in range(60):fn(items[i%N])
    for i in range(200):
     item=items[i%N];t=time.perf_counter_ns();out=fn(item);dt=(time.perf_counter_ns()-t)/1e6
     RAW.append(dict(kind=kind,seed=seed,block=block,repeat=i,window=i%N,ms=dt))
 finally:
  if old:gc.enable()
 vals=[r['ms'] for r in RAW if r['kind']==kind and r['seed']==seed]
 A.writecsv('latency_raw.csv',RAW);A.writecsv('prediction_replay.csv',PARITY)
 print('TIME',kind,seed,float(np.mean(vals)),float(np.median(vals)),float(np.percentile(vals,95)),flush=True)
def load(kind,seed):
 p=next(p for k,s,p in A.specs() if k==kind and s==seed);c=torch.load(A.source(p),map_location='cpu',weights_only=False);m=A.make_neural(kind);m.load_state_dict(c['model_state'],strict=True);m.eval();return m
with threadpool_limits(limits=1):
 for kind,stem in [('xgb_core','v2'),('xgb_road','pie_road')]:
  dim=135 if kind=='xgb_core' else 195
  for seed in (42,43,44):
   p=A.exp(96 if dim==135 else 116)/f'models/{stem}_seed{seed}.joblib';m=joblib.load(A.source(p));m.named_steps['model'].set_params(n_jobs=1)
   assert list(m.feature_names_in_)==xn[:dim]
   items=[pd.DataFrame(xf[i:i+1,:dim],columns=xn[:dim]) for i in range(N)]
   got=[m.predict_proba(x)[0,1] for x in items]
   saved=A.exp(96)/'predictions/test_predictions.csv' if dim==135 else A.exp(116)/f'predictions/pie_road_test_seed{seed}.csv'
   parity(kind,seed,got,saved,ids,filter_args={'arm':'v2','seed':seed} if dim==135 else None)
   measure(kind,seed,lambda x:m.predict_proba(x),items)
 for kind in ['mslstm_base','mslstm_63','mslstm_raw','mslstm_56']:
  for seed in (42,43,44):
   m=load(kind,seed);data={k:arrays[k] for k in m.streams}
   if kind=='mslstm_56':data['biu_v2']=data['biu_v2'][:,index56]
   items=sliced_dict(data)
   fn=lambda x:torch.softmax(m(x),dim=1)[:,1]
   with torch.inference_mode():got=[float(fn(x)[0]) for x in items]
   if kind=='mslstm_56':saved=A.exp(117)/f'predictions/mslstm_pie_test_seed{seed}.csv'
   else:
    n,v=(95,'J') if kind=='mslstm_raw' else (53,'G' if kind=='mslstm_base' else 'I');saved=A.exp(n)/f'predictions/set03_{v}_seed{seed}.csv'
   parity(kind,seed,got,saved,ids);measure(kind,seed,fn,items)
 for ds,nparent,nbranch,branchstem,result_exp,resultstem in [('pie',84,93,'R',93,'R10'),('jaad',89,91,'N',92,'N10')]:
  if ds=='pie':
   bx=arrays['bbox'];cx,cy,w,h=np.moveaxis(bx,-1,0);boxes=np.stack([cx-w/2,cy-h/2,cx+w/2,cy+h/2],axis=-1).astype(np.float32);ego=arrays['ego_speed'];union=np.concatenate([arrays['biu_v1'],arrays['biu_v2']],axis=1);sids=ids
  else:
   cache=A.exp(89)/'cache/test_source';r=csv(cache/'rows.csv');ix=np.linspace(0,len(r)-1,N,dtype=int);sids=r.iloc[ix].sequence_id.astype(str).tolist()
   boxes=np.array(arr(cache/'bbox.npy')[ix],np.float32);ego=np.array(arr(cache/'ego_speed.npy')[ix],np.float32);union=np.concatenate([arr(cache/'biu_v1.npy')[ix],arr(cache/'biu_v2.npy')[ix]],axis=1)
  stats=np.load(A.source(A.exp(nparent)/'cache/trainval/training_stats.npz'));union=((union-stats['union_mean'])/stats['union_std']).astype(np.float32)
  items=[tuple(torch.from_numpy(np.ascontiguousarray(v[i:i+1])) for v in (boxes,ego,union)) for i in range(N)]
  for seed in (42,43,44):
   parent=load('ped_parent_'+ds,seed);parent.requires_grad_(False);branch=load('ped_branch_'+ds,seed);model=Capped(parent,branch).eval()
   parent_fn=lambda x:torch.sigmoid(parent(x[0],x[1])[0]);full_fn=lambda x:model(*x)
   with torch.inference_mode():
    gotparent=[float(parent_fn(x)[0,0]) for x in items];got=[float(full_fn(x)[0,0]) for x in items]
   saved=A.exp(result_exp)/f'predictions/test_{resultstem}_seed{seed}.csv'
   parity('ped_parent_'+ds,seed,gotparent,saved,sids,prob='parent_probability');parity('ped_complete_'+ds,seed,got,saved,sids)
   measure('ped_parent_'+ds,seed,parent_fn,items);measure('ped_complete_'+ds,seed,full_fn,items)
   # Branch-only ready logits/anchor/BIU. Diagnostic; never replace full-predictor latency.
   cached=[]
   with torch.inference_mode():
    for x in items:logit=parent(x[0],x[1])[0];cached.append((logit,model.anchor.detach(),x[2]))
   measure('ped_branch_'+ds,seed,lambda x:torch.sigmoid(x[0]+.1*branch(*x)[1]),cached)
 A.writecsv('latency_raw.csv',RAW);A.writecsv('prediction_replay.csv',PARITY)
 df=pd.DataFrame(RAW)
 for keys,grp in df.groupby(['kind','seed']):
  S.append(dict(kind=keys[0],seed=keys[1],n=len(grp),mean_ms=grp.ms.mean(),median_ms=grp.ms.median(),p95_ms=grp.ms.quantile(.95),min_ms=grp.ms.min(),max_ms=grp.ms.max()))
 A.writecsv('latency_per_seed.csv',S)
 summary=[dict(kind=k,n=len(g),mean_ms=g.ms.mean(),median_ms=g.ms.median(),p95_ms=g.ms.quantile(.95),seed_count=g.seed.nunique()) for k,g in df.groupby('kind')]
 A.writecsv('latency_summary.csv',summary)
 A.writejson(O/'latency_environment.json',{'cpu_affinity':[cpu],'cpu':next(x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')),'threads':1,'device':'CPU','batch':1,'warmup_per_block':60,'repetitions_per_block':200,'blocks':3,'windows':32,'seeds':[42,43,44],'torch_inference_mode':True,'torch_compile':False,'torch_mkldnn':torch.backends.mkldnn.enabled,'threadpools':threadpool_info(),'loadavg_after':os.getloadavg(),'window_selection':'32 evenly spaced cached rows; PIE sequence IDs shared by all predictor families','upstream_excluded':True,'mslstm_input_shapes':{k:list(v.shape[1:]) for k,v in arrays.items()},'xgb_input_dtype':'float64 pandas DataFrame','neural_input_dtype':'float32 CPU tensors'})
 A.writejson(O/'latency_sources.json',A.SOURCES)
 for p,meta in A.SOURCES.items():assert A.sha(p)==meta['sha256'],p
 print('LATENCY COMPLETE',flush=True)
