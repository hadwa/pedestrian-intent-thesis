"""Saved MSLSTM inference only, with strict checkpoint loading and reference replay."""
from pathlib import Path
import sys,argparse,json,hashlib,platform
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
import torch
from mslstm_model import Model
from evaluate_tabular import local,sha,require,metrics,ROOT

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',default='data/mslstm_evaluation/manifest.json');p.add_argument('--output',required=True);p.add_argument('--case',choices=['pie','jaad']);p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');p.add_argument('--batch-size',type=int,default=512);a=p.parse_args()
 require(a.batch_size>0,'Invalid batch size');torch.set_num_threads(1);torch.set_num_interop_threads(1)
 device='cuda' if a.device=='auto' and torch.cuda.is_available() else 'cpu' if a.device=='auto' else a.device
 require(device!='cuda' or torch.cuda.is_available(),'CUDA unavailable')
 manifest=local(a.manifest);m=json.loads(manifest.read_text());jobs=[j for j in m['jobs'] if not a.case or j['case']==a.case];require(bool(jobs),'No selected jobs')
 needed=set()
 for j in jobs:
  needed.update([j['checkpoint'],j['reference']]);d=m['datasets'][j['dataset']];needed.add(d['rows']);needed.update(d['arrays'][j['input_keys'][k]] for k in j['streams'])
 for rel in sorted(needed):require(sha(local(rel))==m['files'][rel],f'Checksum mismatch: {rel}')
 out=local(a.output);require(out.is_relative_to(ROOT/'runs'),'Output must be under runs');out.mkdir(parents=True,exist_ok=False);(out/'predictions').mkdir();cache={};records=[]
 for j in jobs:
  key=j['dataset'];d=m['datasets'][key]
  if key not in cache:
   rows=pd.read_csv(local(d['rows']));arrays={k:np.load(local(v),allow_pickle=False) for k,v in d['arrays'].items()};require(rows.sequence_id.is_unique and len(rows)==d['count'],'Invalid population');cache[key]=(rows,arrays)
  rows,arrays=cache[key];model=Model(j['streams'],j['dimensions'],j['vectors'])
  ck=torch.load(local(j['checkpoint']),map_location='cpu',weights_only=False);model.load_state_dict(ck['model_state'],strict=True)
  if j['case']=='jaad':require(ck['selection']=='ap' and ck['arm']==j['arm'] and ck['seed']==j['seed'],'Wrong selected checkpoint')
  model.to(device).eval();inputs={}
  for name in j['streams']:
   x=arrays[j['input_keys'][name]];require(x.dtype==np.float32 and len(x)==len(rows) and x.shape[-1]==j['dimensions'][name] and np.isfinite(x).all(),'Invalid input stream')
   require(x.ndim==(2 if name in j['vectors'] else 3),'Stream rank mismatch')
   if x.ndim==3:require(x.shape[1]==16,'Observation length mismatch')
   inputs[name]=torch.as_tensor(x,device=device)
  values=[]
  with torch.no_grad():
   for start in range(0,len(rows),a.batch_size):values.append(torch.softmax(model({k:v[start:start+a.batch_size] for k,v in inputs.items()}),dim=1)[:,1].cpu().numpy())
  prob=np.concatenate(values);ref=pd.read_csv(local(j['reference']));require(rows.sequence_id.equals(ref.sequence_id) and rows.label.equals(ref.label),'Reference population mismatch')
  err=float(np.max(np.abs(prob-ref.probability.to_numpy())));decisions=int(np.sum((prob>=.5)!=(ref.probability.to_numpy()>=.5)))
  result=metrics(rows.label,prob);old=metrics(ref.label,ref.probability.to_numpy());delta=max(abs(result[k]-old[k]) for k in result)
  record={k:j[k] for k in ['case','arm','seed','split']}|{'max_probability_error':err,'changed_decisions':decisions,'max_metric_error':delta}|result;records.append(record)
  pd.DataFrame(records).to_csv(out/'per_seed_results.csv',index=False)
  require(err<=1e-5 and decisions==0 and delta<=1e-5,f'Replay mismatch: {record}')
  frame=rows[['sequence_id','label']].copy();frame['probability']=prob;frame['prediction']=(prob>=.5).astype(int);frame['threshold']=.5;frame.to_csv(out/'predictions'/f"{j['case']}_{j['arm']}_{j['split']}_seed{j['seed']}.csv",index=False)
  print(f"PASS {j['case']} {j['arm']} {j['split']} {j['seed']} maxerr={err:.3g}",flush=True)
  del model,inputs,ck
 data=pd.DataFrame(records);names=list(result);summary=data.groupby(['case','arm','split'])[names].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns];summary.reset_index().to_csv(out/'summary_results.csv',index=False)
 report={'classification':'saved-model replay, not training or raw-input reproduction','models':len({j['checkpoint'] for j in jobs}),'jobs':len(jobs),'max_probability_error':max(r['max_probability_error'] for r in records),'max_metric_error':max(r['max_metric_error'] for r in records),'changed_decisions':sum(r['changed_decisions'] for r in records),'device':device,'batch_size':a.batch_size,'torch':torch.__version__,'python':platform.python_version(),'manifest_sha256':sha(manifest),'architecture_sha256':sha(Path(__file__).with_name('mslstm_model.py')),'code_sha256':sha(Path(__file__)),'models_trained':0}
 (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
