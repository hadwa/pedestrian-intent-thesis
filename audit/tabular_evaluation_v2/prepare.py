"""Curator export of audited saved inputs; does not load or train models."""
from pathlib import Path
import json,hashlib,sys,csv
import numpy as np
import pandas as pd
D=Path('/home/abdelrahman/Hadwa/pedestrian-intent-thesis');S=Path('/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean')
B=D/'data/tabular_evaluation_v2';B.mkdir(exist_ok=False);A=D/'audit/tabular_evaluation_v2';A.mkdir(exist_ok=False)
def exp(n):return next((S/'experiments').glob(f'{n:03d}_*'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
sources={}
def source(p):sources[str(p)]=sha(p);return p
def read(p):return pd.read_csv(source(p))
def js(p):return json.loads(source(p).read_text())
def arr(p):return np.load(source(p),allow_pickle=False)
manifest={'files':{},'datasets':{},'jobs':[],'policy':'saved-model evaluation only; no training'}
def register(p):manifest['files'][str(p.relative_to(D))]=sha(p);return str(p.relative_to(D))
def dataset(case,split,rows,x,cols):
 key=case+'/'+split;out=B/case/split;out.mkdir(parents=True)
 assert rows.sequence_id.is_unique and len(rows)==len(x) and x.shape[1]==len(cols) and len(set(cols))==len(cols) and np.isfinite(x).all()
 keep=[c for c in ['sequence_id','set_id','video_id','track_id','label'] if c in rows]
 rows[keep].to_csv(out/'rows.csv',index=False);np.save(out/'features.npy',x,allow_pickle=False);(out/'columns.json').write_text(json.dumps(cols)+'\n')
 manifest['datasets'][key]={'rows':register(out/'rows.csv'),'features':register(out/'features.npy'),'columns':register(out/'columns.json'),'count':len(rows),'dimensions':len(cols),'split':split}
 return key

def job(n,case,split,seed,model,reference,key):
 # Use the already packaged exact weights, not originals.
 model=D/'data/saved_weights'/model.relative_to(S);assert model.is_file()
 rows=read(B/case/split/'rows.csv');reference=reference.set_index('sequence_id',drop=False)
 assert reference.index.is_unique and set(reference.index)==set(rows.sequence_id)
 reference=reference.loc[rows.sequence_id].reset_index(drop=True);assert np.array_equal(reference.label,rows.label)
 path=B/case/split/f'reference_seed{seed}.csv';reference[['sequence_id','label','probability']].to_csv(path,index=False)
 manifest['jobs'].append({'case':case,'experiment':n,'split':split,'seed':seed,'dataset':key,'model':register(model),'reference':register(path),'threshold':.5})
cols96=js(exp(96)/'metrics/feature_arms.json')
for split in ['validation','test']:
 f=read(exp(72)/f'metrics/prepared_{split}.csv').sort_values('sequence_id',kind='stable').reset_index(drop=True)
 saved=read(exp(96)/f'predictions/{split}_predictions.csv')
 for arm in ['base','v1','v2']:
  case=f'exp096_{arm}';key=dataset(case,split,f,f[cols96[arm]].to_numpy(float),cols96[arm])
  for seed in (42,43,44):job(96,case,split,seed,exp(96)/f'models/{arm}_seed{seed}.joblib',saved[(saved.arm==arm)&(saved.seed==seed)],key)
# Exp098 A is byte-compatible BBox prefix of the frozen simple Exp116 cache.
for split in ['validation','test']:
 folder=exp(116)/'cache/jaad_clean_simple'/split;rows=read(folder/'rows.csv');x=arr(folder/'features.npy');cols=js(folder/'columns.json')
 case='exp098_A';key=dataset(case,split,rows,x[:,:53],cols[:53])
 for seed in (42,43,44):job(98,case,split,seed,exp(98)/f'models/A_seed{seed}.joblib',read(exp(98)/f'predictions/{split}_A_seed{seed}.csv'),key)
for split in ['validation','test']:
 folder=exp(100)/'cache'/split;rows=read(folder/'rows.csv');z=arr(folder/'features.npz');cols=js(exp(100)/'metrics/feature_contract.json')['v2'];case='exp100_v2';key=dataset(case,split,rows,z['v2'],cols);z.close()
 for seed in (42,43,44):job(100,case,split,seed,exp(100)/f'models/v2_seed{seed}.joblib',read(exp(100)/f'predictions/{split}_v2_seed{seed}.csv'),key)
names=js(exp(89)/'cache/trainval_source/metadata.json')['biu_v2_feature_names'];defs=read(exp(12)/'metrics/pie_biu_v2_feature_set_definitions.csv').set_index('feature_set');core=json.loads(defs.loc['biu_v2_core','feature_columns_json']);indices=[names.index(n) for n in core];contract=js(exp(102)/'metrics/feature_contract.json')
for split in ['validation','test']:
 for cohort,arm in [('jaad_beh','A'),('jaad_beh','C'),('jaad_all','C')]:
  split102='val' if split=='validation' else split;folder=exp(102)/'cache'/split102;rows=read(folder/f'{cohort}_rows.csv');z=arr(folder/f'{cohort}.npz');x=z['bbox'] if arm=='A' else np.column_stack((z['bbox'],z['v2'][:,indices]));z.close();case=f'exp102_{cohort}_{arm}';key=dataset(case,split,rows,x,contract[arm])
  for seed in (42,43,44):job(102,case,split,seed,exp(102)/f'models/{cohort}_{arm}_{seed}.joblib',read(exp(102)/f'predictions/{cohort}_{split102}_{arm}_{seed}.csv'),key)
cases=['pie_road','jaad_clean_simple','jaad_clean_road','jaad_clean_road_vehicle','jaad_beh_road','jaad_beh_road_vehicle','jaad_all_road','jaad_all_road_vehicle']
for case0 in cases:
 for split in ['validation','test']:
  folder=exp(116)/'cache'/case0/split;case='exp116_'+case0;rows=read(folder/'rows.csv');key=dataset(case,split,rows,arr(folder/'features.npy'),js(folder/'columns.json'))
  for seed in (42,43,44):job(116,case,split,seed,exp(116)/f'models/{case0}_seed{seed}.joblib',read(exp(116)/f'predictions/{case0}_{split}_seed{seed}.csv'),key)
# Ordered-history is validation-only. Never create/open a test arm for Exp118.
frozen=js(exp(118)/'metrics/FROZEN.json');rows=read(exp(72)/'metrics/prepared_validation.csv').sort_values('sequence_id',kind='stable').reset_index(drop=True)
meta=read(exp(40)/'metrics/prepared_metadata.csv');meta['cache_row']=np.arange(len(meta));matched=meta.set_index('sequence_id').loc[rows.sequence_id]
for key in ['label','video_id','track_id','set_id']:assert np.array_equal(rows[key],matched[key])
z=arr(exp(40)/'metrics/prepared_sequences.npz');seq=z['sequences'];inds=list(range(16))+list(range(19,24));x=np.column_stack((rows[cols96['v2']].to_numpy(float),seq[matched.cache_row.to_numpy()][:,:,inds].reshape(len(rows),-1).astype(float)));z.close()
case='exp118_ordered';key=dataset(case,'validation',rows,x,frozen['columns'])
for seed in (42,43,44):job(118,case,'validation',seed,exp(118)/f'models/ordered_seed{seed}.joblib',read(exp(118)/f'predictions/ordered_seed{seed}_validation.csv'),key)
(B/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(A/'source_hashes.json').write_text(json.dumps(sources,indent=2)+'\n');(A/'prepare.py').write_bytes(Path(__file__).read_bytes());(A/'bundle_manifest_sha256.txt').write_text(sha(B/'manifest.json')+'\n')
print(json.dumps({'datasets':len(manifest['datasets']),'evaluation_jobs':len(manifest['jobs']),'unique_models':len({j['model'] for j in manifest['jobs']}),'models_trained':0}))
