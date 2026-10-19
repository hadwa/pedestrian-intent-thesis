"""Complete two missing controls; reuse matched PSI extended results."""
from pathlib import Path
import importlib.util
import sys
import json
import hashlib
import time
import fcntl
import itertools
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent;OLD=ROOT.parent/'116_xgboost_v2_core_completion'
spec=importlib.util.spec_from_file_location('exp124_metrics',OLD/'scripts/run.py')
M=importlib.util.module_from_spec(spec);sys.modules[spec.name]=M;spec.loader.exec_module(M)
SOURCES={};TEST=False
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def source(p):
    p=Path(p)
    if not TEST: assert not any(x=='test' or '_test_' in x for x in p.parts),p
    SOURCES[str(p)]=sha(p);return p
def dump(p,x): p.write_text(json.dumps(x,indent=2)+'\n')
def log(s): print(time.strftime('%Y-%m-%d %H:%M:%S')+' '+s,flush=True)
def load(split):
    values={}
    for arm,case in [('C','psi_ego'),('D','psi_road_ego')]:
        p=OLD/f'cache/{case}/{split}'
        rows=pd.read_csv(source(p/'rows.csv'));x=np.load(source(p/'features.npy'))
        cols=json.loads(source(p/'columns.json').read_text())
        assert x.dtype==np.float32 and x.shape==(len(rows),len(cols))
        assert not np.isinf(x).any()
        values[arm]=(rows,x,cols)
    rows,x,cols=values['C'];r,y,n=values['D']
    assert rows.equals(r) and np.array_equal(x,y[:,:135],equal_nan=True) and cols==n[:135]
    assert len(cols)==135 and len(n)==195
    for arm,dim in [('A',53),('B',109)]: values[arm]=(rows,x[:,:dim],cols[:dim])
    assert len(rows)=={'train':26295,'validation':14922,'test':20547}[split]
    return values
def disjoint(*frames):
    for a,b in itertools.combinations(frames,2):
        for col in ('sequence_id','video_id','track_id'): assert not set(a[col])&set(b[col])
def reused(arm,seed,split,data):
    case={'C':'psi_ego','D':'psi_road_ego'}[arm]
    rows,x,cols=data
    model=joblib.load(source(OLD/f'models/{case}_seed{seed}.joblib'))
    saved=pd.read_csv(source(OLD/f'predictions/{case}_{split}_seed{seed}.csv'))
    assert saved.sequence_id.equals(rows.sequence_id) and saved.label.equals(rows.label)
    p=model.predict_proba(pd.DataFrame(x,columns=cols))[:,1]
    assert np.allclose(p,saved.probability,rtol=0,atol=1e-7)
    return saved.probability.to_numpy(),model
def save(rows,p,arm,seed,split):
    out=rows.copy();out['probability']=p;out['prediction']=(p>=.5).astype(int);out['threshold']=.5
    path=ROOT/f'predictions/{split}_{arm}_seed{seed}.csv';out.to_csv(path,index=False)
    v=pd.read_csv(path);a=M.metrics(rows.label,p);b=M.metrics(v.label,v.probability.to_numpy())
    assert all(abs(a[k]-b[k])<1e-12 for k in a)
    return dict(arm=arm,seed=seed,split=split,source='new' if arm in 'AB' else 'Exp116 reused',**a)
def main():
    global TEST
    assert not (ROOT/'EXPERIMENT_COMPLETE.json').exists()
    for d in ('metrics','models','predictions','logs'): (ROOT/d).mkdir(exist_ok=True)
    lock=(ROOT/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    source(OLD/'scripts/run.py')
    train=load('train');val=load('validation');disjoint(train['A'][0],val['A'][0])
    definitions=pd.read_csv(source(ROOT.parent/'012_pie_biu_v2_30_60_benchmark/metrics/pie_biu_v2_feature_set_definitions.csv')).set_index('feature_set')
    core=json.loads(definitions.loc['biu_v2_core','feature_columns_json'])
    cols=train['B'][2];assert len(cols[:53])==53 and [n.removeprefix('v2__') for n in cols[53:]]==core
    assert all(not n.startswith('v1__') for n in cols)
    for arm in 'ABCD': assert train[arm][2]==val[arm][2]
    parents={}
    for seed in (42,43,44):
        _,parents[seed]=reused('C',seed,'validation',val['C']);_,road=reused('D',seed,'validation',val['D'])
        q=parents[seed].steps[-1][1].get_params();q2=road.steps[-1][1].get_params()
        assert repr(q)==repr(q2)
        assert q['random_state']==seed and q['n_estimators']==250 and q['max_depth']==4 and q['n_jobs']==4
        assert abs(q['scale_pos_weight']-7697/18598)<1e-12
    frozen={str(p):sha(p) for p in [ROOT/'DESIGN.md',Path(__file__)]}
    dump(ROOT/'FROZEN.json',frozen);dump(ROOT/'metrics/feature_contract.json',{a:train[a][2] for a in 'ABCD'})
    dump(ROOT/'PRETRAINING_AUDIT_PASSED.json',dict(train=26295,validation=14922,shared_features_exact=True,recipe_parity=True,test_accessed=False))
    results=[];trained={};times=[]
    for seed in (42,43,44):
        for arm in 'AB':
            r,x,c=train[arm];model=clone(parents[seed]);t=time.perf_counter()
            model.fit(pd.DataFrame(x,columns=c),r.label.to_numpy())
            elapsed=time.perf_counter()-t
            path=ROOT/f'models/{arm}_seed{seed}.joblib';joblib.dump(model,path);trained[str(path)]=sha(path)
            vr,vx,vc=val[arm];p=model.predict_proba(pd.DataFrame(vx,columns=vc))[:,1]
            results.append(save(vr,p,arm,seed,'validation'));trained[str(ROOT/f'predictions/validation_{arm}_seed{seed}.csv')]=sha(ROOT/f'predictions/validation_{arm}_seed{seed}.csv')
            times.append(dict(arm=arm,seed=seed,seconds=elapsed));log(f'FIT {arm}{seed}: {elapsed:.2f}s')
    dump(ROOT/'TRAINED.json',trained);TEST=True
    log('Six new models frozen; opening test for all predeclared arms')
    test=load('test');disjoint(train['A'][0],val['A'][0],test['A'][0])
    for split,data in [('validation',val),('test',test)]:
        for seed in (42,43,44):
            for arm in 'ABCD':
                if split=='validation' and arm in 'AB': continue
                rows,x,cols=data[arm]
                if arm in 'CD': p,_=reused(arm,seed,split,data[arm])
                else:
                    path=ROOT/f'models/{arm}_seed{seed}.joblib';assert sha(path)==trained[str(path)]
                    p=joblib.load(path).predict_proba(pd.DataFrame(x,columns=cols))[:,1]
                results.append(save(rows,p,arm,seed,split))
    d=pd.DataFrame(results);d.to_csv(ROOT/'metrics/per_seed_results.csv',index=False)
    keys=['accuracy','macro_f1','f1','pr_auc','roc_auc','precision','recall','tn','fp','fn','tp']
    summary=d.groupby(['split','arm'])[keys].agg(['mean','std']);summary.columns=['_'.join(c) for c in summary.columns]
    summary.to_csv(ROOT/'metrics/summary_results.csv');pd.DataFrame(times).to_csv(ROOT/'metrics/fit_times.csv',index=False)
    paired=[]
    for split in ('validation','test'):
        for seed in (42,43,44):
            sub=d[(d.split==split)&(d.seed==seed)].set_index('arm')
            for ref,arm in [('A','B'),('B','C'),('C','D')]:
                paired.append(dict(split=split,seed=seed,reference=ref,arm=arm,**{k:float(sub.loc[arm,k]-sub.loc[ref,k]) for k in keys}))
    pd.DataFrame(paired).to_csv(ROOT/'metrics/paired_deltas.csv',index=False)
    for p,h in {**SOURCES,**frozen,**trained}.items(): assert sha(p)==h,p
    dump(ROOT/'metrics/source_hashes.json',SOURCES)
    out=['# Matched PSI lightweight architecture results','',
        'All four arms share26295train/14922validation/20547test windows, labels/order, seeds42/43/44, model recipe and threshold0.5. A/B are new; C/D reuse Exp116. PSI-native15 observed frames -> intent at frame16; not the16-frame/TTE30--60 task. Retrospective matched evidence on previously exposed cohorts.','']
    for split in ('test','validation'):
        out += [f'## {split.title()} means','', '| Input | Accuracy | Macro-F1 | Positive F1 | AP | ROC-AUC |','| --- | --- | --- | --- | --- | --- |']
        for arm,name in [('A','BBox53'),('B','BBox53+BIU56'),('C','+temporal+ego'),('D','+road')]:
            r=summary.loc[(split,arm)];out.append(f'| {name} | '+' | '.join(f'{r[k+"_mean"]:.6f}' for k in keys[:5])+' |')
    out+=['','## Controlled scope','',
        'A53/B109/C135/D195 inputs. Static BIU is exact named56-D core, with no7aux/v1/union. Temporal21 includes3pose-confidence summaries outside BIU56. All usefloat32, training-only median imputation,250trees/depth4/LR.05/subsample.8/colsample.8/L2=1, class weight7697/18598. No search, early stopping, calibration or resampling. The same585 missing-speed training windows are excluded from every arm.','',
        'The B->C contrast adds temporal and ego blocks together; it does not isolate their individual contributions. Adjacent paired seed deltas and mean/sampleSD are in metrics/. These are descriptive comparisons, not significance tests. Class imbalance makes positive-F1 alone insufficient.','',
        'Exact sources and hashes: metrics/source_hashes.json. Inputs: Exp116 cache/psi_ego and cache/psi_road_ego/{train,validation,test}; reused models/predictions: Exp116 models/psi_{ego,road_ego}_seed*.joblib and predictions/psi_{ego,road_ego}_{validation,test}_seed*.csv. Original Exp113/116 simple results remain on their original larger training population.','',
        'Command: `env PYTHONDONTWRITEBYTECODE=1 /home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python -u experiments/124_psi_matched_lightweight_ablation/run.py`. CPU fitting preserves the historical4-thread recipe; no GPU perception needed.','',
        'All six new models frozen before test; all24 saved prediction files checked; reused probabilities replayed and source hashes unchanged. No thesis changes, commits, pushes or further experiments.']
    (ROOT/'FINAL_REPORT.md').write_text('\n'.join(out)+'\n')
    dump(ROOT/'EXPERIMENT_COMPLETE.json',dict(new_fits=6,reused_models=6,prediction_files=24,time=time.time()))
    log('EXPERIMENT COMPLETE')
if __name__=='__main__':
    try: main()
    except Exception as e:
        dump(ROOT/'FAILURE.json',dict(error=repr(e),time=time.time()));raise
