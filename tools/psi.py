"""Frozen-feature PSI Exp124/116 reproduction; no research-repository imports."""
from pathlib import Path
import argparse, hashlib, itertools, json, platform, sys
import importlib.metadata
import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, roc_auc_score, confusion_matrix, balanced_accuracy_score
from xgboost import XGBClassifier
ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/psi_exp124.json'
SPLITS = ('train', 'validation', 'test')
DIMS = {'A':53, 'B':109, 'C':135, 'D':195}
SEEDS = (42,43,44)

def require(condition, message):
    if not condition: raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def dump(path, value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def config(): return json.loads(CONFIG.read_text())

def inside(path, root=ROOT):
    p=Path(path).resolve(); require(p.is_relative_to(root.resolve()),f'Path must be inside {root}: {p}');return p

def fresh(path):
    p=inside(path,ROOT/'runs');p.mkdir(parents=True,exist_ok=False);return p

def env():
    return {'python':platform.python_version(), **{n:importlib.metadata.version(n) for n in ('numpy','pandas','scikit-learn','xgboost','joblib','scipy')}}

def load(bundle, split):
    cfg=config(); base=inside(bundle)/split
    expected=cfg['inputs'][split]
    for name,digest in expected.items(): require(sha(base/name)==digest,f'Input checksum mismatch: {split}/{name}')
    rows=pd.read_csv(base/'rows.csv'); x=np.load(base/'features.npy',allow_pickle=False)
    cols=json.loads((base/'columns.json').read_text())
    require(cols==cfg['columns'],'Feature order changed')
    require(x.dtype==np.float32 and x.shape==(cfg['rows'][split],195),'Shape/dtype changed')
    require(np.isfinite(x).all(),'Non-finite features')
    require(rows.sequence_id.is_unique and len(rows)==len(x),'Invalid identities')
    require(set(rows.label)=={0,1},'Invalid labels')
    frames=np.asarray(rows.observed_frames_json.map(json.loads).tolist())
    require(frames.shape==(len(rows),15) and (np.diff(frames,axis=1)==1).all(),'Expected 15 consecutive observations')
    require(np.array_equal(frames[:,-1]+1,rows.target_frame),'Target is not next frame')
    require(np.array_equal(rows.label,(rows.annotation_vote>=.5).astype(int)),'Vote/label mismatch')
    if split=='train':
        require(int(rows.label.sum())==18598,'Training class balance changed')
        require(not set(rows.video_id)&{'video_0032','video_0055','video_0093'},'Excluded videos found')
    return rows,x,cols

def disjoint(populations):
    for (sa,a),(sb,b) in itertools.combinations(populations.items(),2):
        for key in ('sequence_id','video_id','track_id'): require(not set(a[key])&set(b[key]),f'{sa}/{sb} overlap: {key}')

def model(seed):
    return Pipeline([('imputer',SimpleImputer(strategy='median')),('model',XGBClassifier(**config()['model'],random_state=seed))])

def metrics(y,p):
    require(np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'Invalid probabilities')
    pred=p>=.5;tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return dict(accuracy=accuracy_score(y,pred),precision=precision_score(y,pred,zero_division=0),recall=recall_score(y,pred,zero_division=0),f1=f1_score(y,pred,zero_division=0),pr_auc=average_precision_score(y,p),roc_auc=roc_auc_score(y,p),macro_f1=f1_score(y,pred,average='macro'),balanced_accuracy=balanced_accuracy_score(y,pred),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),high_conf_fp=int(((np.asarray(y)==0)&(p>=.9)).sum()),high_conf_fn=int(((np.asarray(y)==1)&(p<=.1)).sum()))

def prediction(out,rows,p,split,arm,seed):
    frame=rows[['sequence_id','video_id','track_id','label']].copy()
    frame['probability']=p;frame['prediction']=(p>=.5).astype(int);frame['threshold']=.5
    path=out/f'{split}_{arm}_seed{seed}.csv';frame.to_csv(path,index=False)
    return dict(split=split,arm=arm,seed=seed,**metrics(rows.label,p))

def reports(out, records):
    data=pd.DataFrame(records);data.to_csv(out/'per_seed_results.csv',index=False)
    names=[c for c in data if c not in ('split','arm','seed')]
    summary=data.groupby(['split','arm'])[names].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(out/'summary_results.csv',index=False)

def prepare(args):
    # Explicit import from an upstream Exp116 cache; copies only the frozen D
    # matrix, whose first 135 columns exactly match C. No model is loaded.
    import shutil
    source=Path(args.source).resolve(); out=inside(args.output,ROOT/'data');require(not out.exists(),'Output already exists')
    cfg=config()
    for split in SPLITS:
        src=source/'psi_road_ego'/split
        for name,digest in cfg['inputs'][split].items(): require(sha(src/name)==digest,f'Unexpected source: {src/name}')
        c=source/'psi_ego'/split
        cx=np.load(c/'features.npy',allow_pickle=False);dx=np.load(src/'features.npy',allow_pickle=False)
        require(np.array_equal(cx,dx[:,:135]),'C/D prefix mismatch')
        require(pd.read_csv(c/'rows.csv').equals(pd.read_csv(src/'rows.csv')),'C/D population mismatch')
        require(json.loads((c/'columns.json').read_text())==cfg['columns'][:135],'C column order mismatch')
    out.mkdir(parents=True)
    for split in SPLITS:
        (out/split).mkdir()
        for name in cfg['inputs'][split]:shutil.copyfile(source/'psi_road_ego'/split/name,out/split/name)
    populations={s:load(out,s)[0] for s in SPLITS};disjoint(populations)
    dump(out/'BUNDLE.json',{'classification':'frozen-feature input import; not raw-video preprocessing','config_sha256':sha(CONFIG),'rows':{s:len(r) for s,r in populations.items()}})
    print(json.dumps({'bundle':str(out),'rows':{s:len(r) for s,r in populations.items()}}))

def preflight(args):
    # Development only: never open test files here.
    data={s:load(args.bundle,s) for s in ('train','validation')};disjoint({s:d[0] for s,d in data.items()})
    print(json.dumps({'status':'passed','test_opened':False,'models_fitted':0,'environment':env(),'rows':{s:len(d[0]) for s,d in data.items()}},indent=2))

def train(args):
    data={s:load(args.bundle,s) for s in ('train','validation')};disjoint({s:d[0] for s,d in data.items()})
    out=fresh(args.output);(out/'models').mkdir();(out/'predictions').mkdir()
    freeze={'config_sha256':sha(CONFIG),'code_sha256':sha(__file__),'environment':env(),'seeds':list(SEEDS),'arms':DIMS,'test_opened':False,'classification':'fresh fit of all 12 models; historical Exp124 reused C/D from Exp116'}
    dump(out/'FROZEN.json',freeze)
    tr,tx,cols=data['train'];vr,vx,_=data['validation'];records=[];hashes={}
    for arm,dim in DIMS.items():
        for seed in SEEDS:
            m=model(seed);m.fit(pd.DataFrame(tx[:,:dim],columns=cols[:dim]),tr.label.to_numpy())
            path=out/'models'/f'{arm}_seed{seed}.joblib';joblib.dump(m,path);hashes[str(path.relative_to(out))]=sha(path)
            p=m.predict_proba(pd.DataFrame(vx[:,:dim],columns=cols[:dim]))[:,1]
            records.append(prediction(out/'predictions',vr,p,'validation',arm,seed))
    reports(out,records);dump(out/'TRAINED.json',{'freeze_sha256':sha(out/'FROZEN.json'),'models':hashes})
    print(json.dumps({'trained':str(out),'models':12,'test_opened':False}))

def evaluate(args):
    trained=inside(args.trained,ROOT/'runs');done=json.loads((trained/'TRAINED.json').read_text());frozen=json.loads((trained/'FROZEN.json').read_text())
    require(sha(trained/'FROZEN.json')==done['freeze_sha256'],'Training freeze changed')
    require(frozen['config_sha256']==sha(CONFIG) and frozen['code_sha256']==sha(__file__),'Code/config changed after training')
    require(frozen['environment']==env(),'Environment changed after training')
    wanted={f'models/{a}_seed{s}.joblib' for a in DIMS for s in SEEDS}
    require(set(done['models'])==wanted,'Incomplete frozen model set')
    for p,h in done['models'].items():require(sha(trained/p)==h,f'Changed model: {p}')
    # All 12 fits authenticated before loading test inputs. No selection.
    data={s:load(args.bundle,s) for s in SPLITS};disjoint({s:d[0] for s,d in data.items()})
    out=fresh(args.output);(out/'predictions').mkdir();rows,x,cols=data['test'];records=[]
    for arm,dim in DIMS.items():
        for seed in SEEDS:
            m=joblib.load(trained/'models'/f'{arm}_seed{seed}.joblib')
            p=m.predict_proba(pd.DataFrame(x[:,:dim],columns=cols[:dim]))[:,1]
            records.append(prediction(out/'predictions',rows,p,'test',arm,seed))
    reports(out,records);dump(out/'EVALUATED.json',{'training_manifest_sha256':sha(trained/'TRAINED.json'),'config_sha256':sha(CONFIG),'models_evaluated':12,'threshold':.5,'environment':env()})
    print(json.dumps({'evaluated':str(out),'models':12}))

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--source',required=True,help='Exp116 cache directory containing psi_ego and psi_road_ego');q.add_argument('--output',required=True)
    for cmd in ('preflight','train','evaluate'):
        q=sub.add_parser(cmd);q.add_argument('--bundle',required=True)
        if cmd!='preflight':q.add_argument('--output',required=True)
        if cmd=='evaluate':q.add_argument('--trained',required=True)
    a=p.parse_args();globals()[a.command](a)
if __name__=='__main__':main()
