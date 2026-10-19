"""Saved tabular model replay on frozen inputs. No training command exists."""
from pathlib import Path
import argparse,json,hashlib,platform,importlib.metadata
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,average_precision_score,roc_auc_score,confusion_matrix
ROOT=Path(__file__).resolve().parents[1]
def require(ok,msg):
 if not ok:raise ValueError(msg)
def local(p):
 p=(ROOT/p).resolve() if not Path(p).is_absolute() else Path(p).resolve();require(p.is_relative_to(ROOT),'Path outside companion repository');return p
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def metrics(y,p):
 pred=p>=.5;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
 return {'accuracy':float(accuracy_score(y,pred)),'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),'f1':float(f1_score(y,pred,zero_division=0)),'pr_auc':float(average_precision_score(y,p)),'roc_auc':float(roc_auc_score(y,p)),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',default='data/tabular_evaluation_v2/manifest.json');p.add_argument('--list',action='store_true');p.add_argument('--case',action='append');p.add_argument('--output');a=p.parse_args()
 manifest=local(a.manifest);m=json.loads(manifest.read_text());cases=sorted({j['case'] for j in m['jobs']})
 if a.list:print('\n'.join(cases));return
 require(a.output is not None,'--output is required');require(not a.case or set(a.case)<=set(cases),'Unknown case')
 jobs=[j for j in m['jobs'] if not a.case or j['case'] in a.case];require(bool(jobs),'No jobs selected')
 # Verify required files before any deserialization; joblib inputs must be trusted release files.
 needed=set()
 for j in jobs:
  require(j['threshold']==.5,'Threshold changed');require(not (j['experiment']==118 and j['split']!='validation'),'Exp118 is validation-only')
  needed.update([j['model'],j['reference']]);d=m['datasets'][j['dataset']];needed.update(d[k] for k in ['features','columns','rows'])
 for rel in sorted(needed):require(sha(local(rel))==m['files'][rel],f'Checksum mismatch: {rel}')
 out=local(a.output);require(out.is_relative_to(ROOT/'runs'),'Outputs must be under runs/');out.mkdir(parents=True,exist_ok=False);(out/'predictions').mkdir()
 records=[];cache={};models={}
 for j in jobs:
  key=j['dataset'];d=m['datasets'][key]
  if key not in cache:
   rows=pd.read_csv(local(d['rows']));x=np.load(local(d['features']),allow_pickle=False);cols=json.loads(local(d['columns']).read_text())
   require(rows.sequence_id.is_unique and len(rows)==d['count'],'Invalid identities');require(x.shape==(len(rows),d['dimensions']) and len(cols)==x.shape[1] and len(set(cols))==len(cols),'Feature shape/order invalid');require(np.isfinite(x).all(),'Nonfinite inputs')
   require(set(rows.label)=={0,1},'Nonbinary targets');cache[key]=(rows,x,cols)
  rows,x,cols=cache[key]
  if j['model'] not in models:models[j['model']]=joblib.load(local(j['model']))
  model=models[j['model']]
  require(model.n_features_in_==len(cols),'Model feature count mismatch')
  if hasattr(model,'feature_names_in_'):
   require(list(model.feature_names_in_)==cols,'Model feature names mismatch');inputs=pd.DataFrame(x,columns=cols)
  else:inputs=x
  probability=model.predict_proba(inputs)[:,1];require(np.isfinite(probability).all(),'Nonfinite probabilities')
  ref=pd.read_csv(local(j['reference']));require(rows.sequence_id.equals(ref.sequence_id) and rows.label.equals(ref.label),'Reference identity mismatch')
  err=float(np.max(np.abs(probability-ref.probability.to_numpy())));require(err<=1e-7,f"Prediction mismatch {j['case']} {j['seed']} {j['split']}: {err}")
  require(np.array_equal(probability>=.5,ref.probability.to_numpy()>=.5),'Thresholded decisions differ')
  result=metrics(rows.label,probability);reference_metrics=metrics(ref.label,ref.probability.to_numpy());delta=max(abs(result[k]-reference_metrics[k]) for k in result)
  require(delta<=1e-7,f'Metric mismatch {j}: {delta}')
  prediction=rows.copy();prediction['probability']=probability;prediction['prediction']=(probability>=.5).astype(int);prediction['threshold']=.5
  prediction.to_csv(out/'predictions'/f"{j['case']}_{j['split']}_seed{j['seed']}.csv",index=False)
  records.append({k:j[k] for k in ['experiment','case','split','seed']}|{'rows':len(rows),'max_probability_error':err,'max_metric_error':delta}|result)
 data=pd.DataFrame(records);data.to_csv(out/'per_seed_results.csv',index=False)
 metric_names=list(result);summary=data.groupby(['experiment','case','split'])[metric_names].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns];summary.reset_index().to_csv(out/'summary_results.csv',index=False)
 report={'classification':'diagnostic saved-model replay; no training or raw-input extraction','models':len(models),'evaluation_jobs':len(records),'max_probability_error':max(r['max_probability_error'] for r in records),'max_metric_error':max(r['max_metric_error'] for r in records),'manifest_sha256':sha(manifest),'code_sha256':sha(Path(__file__)),'environment':{'python':platform.python_version(),**{k:importlib.metadata.version(k) for k in ['numpy','pandas','joblib','scikit-learn','xgboost']}},'models_trained':0}
 (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
