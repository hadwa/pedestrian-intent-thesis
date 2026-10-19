from pathlib import Path
import hashlib,json,shutil,csv
D=Path('/home/abdelrahman/Hadwa/pedestrian-intent-thesis');S=Path('/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean');B=Path('/tmp/psi_companion_build')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
A=D/'audit/psi_workflow';A.mkdir(exist_ok=False)
(D/'configs').mkdir(exist_ok=True)
for name in ['psi.py','replay_psi.py']:shutil.copyfile(B/name,D/'tools'/name)
e116=S/'experiments/116_xgboost_v2_core_completion';e124=S/'experiments/124_psi_matched_lightweight_ablation'
recorded=json.loads((e124/'metrics/source_hashes.json').read_text())
columns=json.loads((e116/'cache/psi_road_ego/train/columns.json').read_text())
inputs={}
for split in ['train','validation','test']:
    inputs[split]={}
    for file in ['rows.csv','features.npy','columns.json']:
        p=e116/'cache/psi_road_ego'/split/file;h=sha(p);assert h==recorded[str(p)],p;inputs[split][file]=h
cfg={'protocol':'PSI-native: 15 observations, next-frame mean intent vote >= 0.5','threshold':.5,'seeds':[42,43,44],'rows':{'train':26295,'validation':14922,'test':20547},'dimensions':{'A':53,'B':109,'C':135,'D':195},'columns':columns,'inputs':inputs,'model':{'n_estimators':250,'max_depth':4,'learning_rate':.05,'subsample':.8,'colsample_bytree':.8,'reg_lambda':1.0,'objective':'binary:logistic','eval_metric':'logloss','n_jobs':4,'scale_pos_weight':7697/18598}}
dump(D/'configs/psi_exp124.json',cfg)
(D/'requirements-psi.txt').write_text('numpy==2.2.6\npandas==2.3.3\nscikit-learn==1.7.2\nxgboost==3.2.0\njoblib==1.5.3\nscipy==1.15.3\n')
ref=D/'data/psi_reference';ref.mkdir(parents=True,exist_ok=False);manifest={'files':{},'metrics':{}};provenance=[]
with (e124/'metrics/per_seed_results.csv').open() as f:
    for r in csv.DictReader(f):manifest['metrics'][f"{r['split']}/{r['arm']}/{r['seed']}"]={k:float(v) for k,v in r.items() if k not in {'split','arm','seed','source'}}
def copy(p,q):
    h=sha(p);shutil.copyfile(p,q);assert sha(q)==h;manifest['files'][q.name]=h;provenance.append({'source':str(p),'destination':str(q.relative_to(D)),'sha256':h})
for arm in 'ABCD':
    for seed in [42,43,44]:
        case={'C':'psi_ego','D':'psi_road_ego'}.get(arm)
        p=e124/f'models/{arm}_seed{seed}.joblib' if arm in 'AB' else e116/f'models/{case}_seed{seed}.joblib';copy(p,ref/f'{arm}_seed{seed}.joblib')
        for split in ['validation','test']:
            p=e124/f'predictions/{split}_{arm}_seed{seed}.csv';copy(p,ref/f'{split}_{arm}_seed{seed}.csv')
dump(ref/'manifest.json',manifest);dump(A/'reference_copy_manifest.json',provenance)
shutil.copyfile(B/'install.py',A/'install.py')
print('Installed portable PSI workflow and local reference artifacts; no training.')
