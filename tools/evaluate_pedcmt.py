"""Full frozen-parent plus capped-branch saved-model replay; no training."""
from pathlib import Path
import sys,json,argparse,platform
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import torch
from evaluate_tabular import local,sha,require,metrics,ROOT
from pedcmt_model import PedCMTTransfer
from pedcmt_branch import DirectFusionModel

def sigmoid(x):return 1/(1+np.exp(-np.clip(np.asarray(x,dtype=np.float64),-80,80)))
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',default='data/pedcmt_evaluation/manifest.json');p.add_argument('--output',required=True);p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto');p.add_argument('--case',choices=['pie','jaad']);a=p.parse_args()
 torch.set_num_threads(1);torch.set_num_interop_threads(1);device='cuda' if a.device=='auto' and torch.cuda.is_available() else 'cpu' if a.device=='auto' else a.device
 require(device!='cuda' or torch.cuda.is_available(),'CUDA unavailable');manifest=local(a.manifest);m=json.loads(manifest.read_text());require(m['cap']==.1,'Cap changed');jobs=[j for j in m['jobs'] if not a.case or j['case']==a.case];needed=set()
 for j in jobs:
  needed.update([j['parent'],j['reference']]);d=m['datasets'][j['dataset']];needed.add(d['rows']);needed.update(d['arrays'].values())
  if j['branch']:needed.add(j['branch'])
 for rel in needed:require(sha(local(rel))==m['files'][rel],f'Checksum mismatch: {rel}')
 out=local(a.output);require(out.is_relative_to(ROOT/'runs'),'Output must be under runs');out.mkdir(parents=True,exist_ok=False);(out/'predictions').mkdir();data={};parents={};results=[]
 for j in jobs:
  key=j['dataset'];d=m['datasets'][key]
  if key not in data:
   rows=pd.read_csv(local(d['rows']));v={k:np.load(local(p),allow_pickle=False) for k,p in d['arrays'].items()};require(rows.sequence_id.is_unique and len(rows)==d['count'],'Invalid population')
   require(all(len(x)==len(rows) and np.isfinite(x).all() for x in v.values()),'Invalid arrays');data[key]=(rows,v)
  rows,v=data[key];pk=(key,j['seed'])
  if pk not in parents:
   parent=PedCMTTransfer(variant='A',ego_dim=v['ego'].shape[-1]).to(device);ck=torch.load(local(j['parent']),map_location='cpu',weights_only=False);parent.load_state_dict(ck['model_state'],strict=True);parent.eval();captured=[];logits=[];anchors=[]
   def capture(module,inputs):captured.append(inputs[0].detach())
   hook=parent.endpoint.register_forward_pre_hook(capture)
   try:
    with torch.inference_mode():
     for i in range(0,len(rows),256):
      captured.clear();logit=parent(torch.as_tensor(v['bbox'][i:i+256],device=device),torch.as_tensor(v['ego'][i:i+256],device=device))[0];require(len(captured)==1 and captured[0].shape[1]==256,'Invalid parent anchor');logits.append(logit.cpu().numpy());anchors.append(captured[0].cpu().numpy())
   finally:hook.remove()
   parents[pk]=(np.concatenate(logits),np.concatenate(anchors));del parent,ck
  logits,anchors=parents[pk]
  if j['arm']=='baseline':
   # Exp084/089 saved parent references use torch float32 sigmoid.
   # Preserve that arithmetic: promoting to float64 changes rank ties.
   prob=torch.sigmoid(torch.as_tensor(logits,device=device)).cpu().numpy().ravel()
  else:
   branch=DirectFusionModel('B' if j['arm']=='raw' else 'C',biu_dim={'core56':56,'full63':63}.get(j['arm'],135)).to(device);ck=torch.load(local(j['branch']),map_location='cpu',weights_only=False)
   if 'parent_sha256' in ck:require(ck['parent_sha256']==m['files'][j['parent']],'Branch/parent hash mismatch')
   branch.load_state_dict(ck['model_state'],strict=True);branch.eval();corrections=[]
   with torch.inference_mode():
    for i in range(0,len(rows),256):
     _,c=branch(torch.as_tensor(logits[i:i+256],device=device),torch.as_tensor(anchors[i:i+256],device=device),torch.as_tensor(v[j['arm']][i:i+256],device=device));corrections.append(c.cpu().numpy().ravel())
   prob=sigmoid(logits.ravel()+.1*np.concatenate(corrections));del branch,ck
  ref=pd.read_csv(local(j['reference']));require(rows.sequence_id.equals(ref.sequence_id) and rows.label.equals(ref.label),'Reference mismatch');err=float(np.max(np.abs(prob-ref.probability.to_numpy())));changed=int(np.sum((prob>=.5)!=(ref.probability.to_numpy()>=.5)));scores=metrics(rows.label,prob);old=metrics(ref.label,ref.probability.to_numpy());delta=max(abs(scores[k]-old[k]) for k in scores)
  rec={k:j[k] for k in ['case','split','arm','seed']}|{'max_probability_error':err,'changed_decisions':changed,'max_metric_error':delta}|scores;results.append(rec);pd.DataFrame(results).to_csv(out/'per_seed_results.csv',index=False)
  require(err<=1e-5 and changed==0 and delta<=1e-5,f'Replay mismatch: {rec}')
  frame=rows[['sequence_id','label']].copy();frame['probability']=prob;frame['prediction']=(prob>=.5).astype(int);frame['threshold']=.5;frame.to_csv(out/'predictions'/f"{j['case']}_{j['arm']}_{j['split']}_seed{j['seed']}.csv",index=False)
  print(f"PASS {j['case']} {j['arm']} {j['split']} {j['seed']} maxerr={err:.3g}",flush=True)
 data=pd.DataFrame(results);summary=data.groupby(['case','arm','split'])[list(scores)].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns];summary.reset_index().to_csv(out/'summary_results.csv',index=False)
 report={'classification':'complete predictor saved-model replay; no training or raw-input extraction','jobs':len(jobs),'unique_parent_checkpoints':len({j['parent'] for j in jobs}),'unique_branch_checkpoints':len({j['branch'] for j in jobs if j['branch']}),'max_probability_error':max(r['max_probability_error'] for r in results),'max_metric_error':max(r['max_metric_error'] for r in results),'changed_decisions':sum(r['changed_decisions'] for r in results),'device':device,'torch':torch.__version__,'python':platform.python_version(),'manifest_sha256':sha(manifest),'code_hashes':{f:sha(Path(__file__).with_name(f)) for f in ['evaluate_pedcmt.py','pedcmt_model.py','pedcmt_branch.py']},'models_trained':0}
 (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
