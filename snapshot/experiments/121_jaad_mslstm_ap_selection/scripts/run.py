"""Replay Exp120 exactly; compare validation-only F1/AP checkpoint selection."""
import copy
import importlib.util
import json
import sys
from pathlib import Path
import time
import fcntl
import numpy as np
import pandas as pd
import torch

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]
OLD=REPO/'experiments/120_jaad_mslstm_v2_completion'
spec=importlib.util.spec_from_file_location('exp121_parent',OLD/'scripts/run.py')
P=importlib.util.module_from_spec(spec);sys.modules[spec.name]=P;spec.loader.exec_module(P)
P.ROOT=ROOT
P.TEST=False
original_src=P.src

def guarded_src(path):
    path=Path(path)
    assert not any('test' in s.lower() or 'set03' in s.lower() for s in path.parts), path
    return original_src(path)
P.src=guarded_src

def scene(rows):
    records={}
    for video in sorted(rows.video_id.unique()):
        path=P.src(OLD/'cache/yolop'/f'{video}.npz')
        with np.load(path,allow_pickle=False) as f:
            for ped,frame,value in zip(f['ped_id'],f['frame'],f['features']):
                records[(video,str(ped),int(frame))]=value
    return np.stack([np.stack([records[(r.video_id,r.ped_id,int(f))] for f in json.loads(r.observed_frames_json)]) for r in rows.itertuples()])
P.scene=scene

def diagnostic(rows,a,manifest,aug,seed):
    names=json.loads((ROOT/'metrics/feature_contract.json').read_text())['full63']
    mask=manifest.augmented.to_numpy(dtype=bool)
    source=manifest.loc[mask,'source_cache_index'].to_numpy()
    original=a['biu63'][source]
    changed=aug['biu63'][mask]
    std=a['biu63'][rows.split.eq('train')].astype(float).std(axis=0)
    delta=np.abs(changed.astype(float)-original)
    result=[]
    for j,name in enumerate(names):
        z=delta[:,j]/std[j] if std[j]>1e-12 else np.full(len(delta),np.nan)
        result.append(dict(seed=seed,feature=name,n=len(delta),training_std=std[j],
            median_abs_change=np.median(delta[:,j]),p95_abs_change=np.quantile(delta[:,j],.95),
            median_training_sd_change=np.median(z),p95_training_sd_change=np.quantile(z,.95),
            fraction_changed=float((delta[:,j]>1e-6).mean())))
    pd.DataFrame(result).to_csv(ROOT/f'metrics/augmentation_diagnostic_seed{seed}.csv',index=False)

def train(arm,seed,rows,a,m,aug):
    P.C.seed_everything(seed)
    model=P.C.MslstmPracticalModel(arm).to(P.DEVICE)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    x=P.tensors(aug,arm);y=torch.tensor(m.label.to_numpy(),device=P.DEVICE,dtype=torch.long)
    va=np.flatnonzero(rows.split.eq('val'));vx=P.tensors(a,arm,va)
    rng=np.random.RandomState(seed);history=[];best={'f1':-1.,'ap':-1.};states={};epochs={}
    for epoch in range(1,16):
        model.train();order=rng.permutation(len(y));loss_sum=0.
        for start in range(0,len(y),32):
            ids=torch.tensor(order[start:start+32],device=P.DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits=model({k:v[ids] for k,v in x.items()})
            loss=torch.nn.functional.cross_entropy(logits,y[ids])
            assert torch.isfinite(loss)
            loss.backward();optimizer.step();loss_sum+=float(loss)*len(ids)
        p=P.TRAIN.probabilities(model,vx,len(va));metrics=P.C.compute_metrics(rows.iloc[va].label,p)
        history.append(dict(epoch=epoch,train_loss=loss_sum/len(y),**metrics))
        for policy,key in [('f1','f1'),('ap','pr_auc')]:
            if metrics[key]>best[policy]:
                best[policy]=metrics[key];epochs[policy]=epoch
                states[policy]=copy.deepcopy({k:v.cpu() for k,v in model.state_dict().items()})
        pd.DataFrame(history).to_csv(ROOT/f'logs/{arm}_seed{seed}.history.csv',index=False)
        P.log(f'{arm}{seed} epoch={epoch}/15 val_AP={metrics["pr_auc"]:.6f} AP_best_epoch={epochs["ap"]}')
    old=pd.read_csv(P.src(OLD/f'logs/{arm}_seed{seed}.history.csv'))
    new=pd.DataFrame(history)
    err=float(np.abs(new.to_numpy()-old[new.columns].to_numpy()).max())
    assert err<=1e-5, ('history replay mismatch',arm,seed,err)
    output=[]
    for policy in ('f1','ap'):
        model.load_state_dict(states[policy]);p=P.TRAIN.probabilities(model,vx,len(va))
        if policy=='f1':
            oldp=pd.read_csv(P.src(OLD/f'predictions/validation_{arm}_seed{seed}.csv'))
            assert list(oldp.sequence_id)==list(rows.iloc[va].sequence_id)
            pred_err=float(np.abs(p-oldp.probability.to_numpy()).max())
            assert pred_err<=2e-6, ('prediction replay mismatch',pred_err)
        saved=rows.iloc[va].copy();saved['probability']=p;saved['prediction']=(p>=.5).astype(int)
        saved.to_csv(ROOT/f'predictions/validation_{arm}_seed{seed}_{policy}.csv',index=False)
        torch.save(dict(model_state=states[policy],arm=arm,seed=seed,selection=policy,best_epoch=epochs[policy]),ROOT/f'checkpoints/{arm}_seed{seed}_{policy}.pt')
        output.append(dict(arm=arm,seed=seed,selection=policy,epoch=epochs[policy],**P.C.compute_metrics(saved.label,p)))
    P.dump(ROOT/f'logs/{arm}_seed{seed}.complete.json',dict(history_max_error=err,f1_prediction_max_error=pred_err,epochs=epochs))
    return output

def main():
    assert not (ROOT/'VALIDATION_COMPLETE.json').exists()
    for d in ('logs','metrics','cache','predictions','checkpoints'): (ROOT/d).mkdir(exist_ok=True)
    lock=(ROOT/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert torch.cuda.is_available()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    # Fail closed instead of touching any test source, even through parent helpers.
    try: P.src(REPO/'test_source/forbidden.npy')
    except AssertionError: pass
    else: raise AssertionError('test access guard failed')
    inventory=sorted(str(p.relative_to(OLD)) for p in (OLD/'checkpoints').glob('*'))
    P.dump(ROOT/'metrics/checkpoint_inventory.json',dict(files=inventory,reason_for_retrain='Only best-F1 checkpoints retained; no epoch archive in Exp120'))
    for n in (49,50,51,53,89,120):
        for path in (P.exp(n)/'scripts').glob('*.py'): P.src(path)
    for path in (REPO/'src').glob('*biu*py'): P.src(path)
    P.src(P.C.ARCHIVE_BBOX_CORE_PATH)
    rows,a,scaler,ix=P.data('trainval')
    assert len(rows)==1826 and set(rows.split)=={'train','val'}
    params={}
    for arm in 'ABCD':
        model=P.C.MslstmPracticalModel(arm).to(P.DEVICE)
        loss=torch.nn.functional.cross_entropy(model(P.tensors(a,arm,np.arange(4))),torch.tensor(rows.label.to_numpy()[:4],device=P.DEVICE))
        loss.backward();assert torch.isfinite(loss)
        params[arm]=P.C.count_parameters(model)
    assert params==dict(A=4590087,B=6830857,C=4663432,D=4664328)
    specs={str(p):P.sha(p) for p in [ROOT/'DESIGN.md',Path(__file__)]}
    P.dump(ROOT/'FROZEN.json',specs)
    P.dump(ROOT/'PRETRAINING_AUDIT_PASSED.json',dict(test_accessed=False,gpu=torch.cuda.get_device_name(0),parameters=params))
    result=[]
    for seed in (42,43,44):
        manifest,aug=P.augment(rows,a,scaler,ix,seed)
        old=P.src(OLD/f'metrics/augmentation_seed{seed}.csv')
        assert pd.read_csv(old).equals(pd.read_csv(ROOT/f'metrics/augmentation_seed{seed}.csv'))
        diagnostic(rows,a,manifest,aug,seed)
        for arm in 'ABCD': result.extend(train(arm,seed,rows,a,manifest,aug))
        pd.DataFrame(result).to_csv(ROOT/'metrics/per_seed_validation.csv',index=False)
    df=pd.DataFrame(result)
    keys=['f1','pr_auc','roc_auc','accuracy','precision','recall','fp','fn']
    summary=df.groupby(['arm','selection'])[keys].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns]
    summary.to_csv(ROOT/'metrics/summary_validation.csv')
    paired=[]
    for (arm,seed),g in df.groupby(['arm','seed']):
        g=g.set_index('selection')
        paired.append(dict(arm=arm,seed=seed,**{k:float(g.loc['ap',k]-g.loc['f1',k]) for k in keys}))
    pd.DataFrame(paired).to_csv(ROOT/'metrics/paired_selection_deltas.csv',index=False)
    for p,h in P.SOURCES.items(): assert P.sha(p)==h
    for p,h in specs.items(): assert P.sha(p)==h
    P.dump(ROOT/'metrics/source_hashes.json',P.SOURCES)
    P.dump(ROOT/'VALIDATION_COMPLETE.json',dict(runs=12,test_accessed=False,time=time.time()))
    P.log('VALIDATION COMPLETE; TEST NOT ACCESSED')

if __name__=='__main__':
    try: main()
    except Exception as e:
        P.dump(ROOT/'FAILURE.json',dict(error=repr(e),time=time.time(),test_accessed=False))
        raise
