"""Controlled removal of measurement noise; balanced manifests unchanged."""
import copy
import importlib.util
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
OLD=REPO/'experiments/121_jaad_mslstm_ap_selection'
spec=importlib.util.spec_from_file_location('exp122_parent',OLD/'scripts/run.py')
Q=importlib.util.module_from_spec(spec);sys.modules[spec.name]=Q;spec.loader.exec_module(Q)
P=Q.P
P.ROOT=ROOT
P.TEST=False

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
    output=[]
    for policy in ('f1','ap'):
        model.load_state_dict(states[policy]);p=P.TRAIN.probabilities(model,vx,len(va))
        saved=rows.iloc[va].copy();saved['probability']=p;saved['prediction']=(p>=.5).astype(int)
        saved.to_csv(ROOT/f'predictions/validation_{arm}_seed{seed}_{policy}.csv',index=False)
        torch.save(dict(model_state=states[policy],arm=arm,seed=seed,selection=policy,best_epoch=epochs[policy]),ROOT/f'checkpoints/{arm}_seed{seed}_{policy}.pt')
        output.append(dict(arm=arm,seed=seed,selection=policy,epoch=epochs[policy],**P.C.compute_metrics(saved.label,p)))
    P.dump(ROOT/f'logs/{arm}_seed{seed}.complete.json',dict(epochs=epochs))
    return output

def main():
    assert not (ROOT/'VALIDATION_COMPLETE.json').exists()
    for d in ('logs','metrics','cache','predictions','checkpoints'): (ROOT/d).mkdir(exist_ok=True)
    lock=(ROOT/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert torch.cuda.is_available()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    try: P.src(REPO/'test_source/forbidden.npy')
    except AssertionError: pass
    else: raise AssertionError('test guard failed')
    for n in (49,50,51,53,89,120,121):
        for path in (P.exp(n)/'scripts').glob('*.py'): P.src(path)
    for path in (REPO/'src').glob('*biu*py'): P.src(path)
    P.src(P.C.ARCHIVE_BBOX_CORE_PATH)
    rows,a,scaler,ix=P.data('trainval')
    assert len(rows)==1826 and set(rows.split)=={'train','val'}
    assert all(np.isfinite(v).all() for v in a.values())
    params={}
    for arm in 'ABCD':
        model=P.C.MslstmPracticalModel(arm).to(P.DEVICE)
        loss=torch.nn.functional.cross_entropy(model(P.tensors(a,arm,np.arange(4))),torch.tensor(rows.label.to_numpy()[:4],device=P.DEVICE))
        loss.backward();assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        params[arm]=P.C.count_parameters(model)
    assert params==dict(A=4590087,B=6830857,C=4663432,D=4664328)
    manifests={}
    r=rows.copy();r['cache_index']=np.arange(len(rows))
    for seed in (42,43,44):
        m=P.CORE.build_oversampling_manifest(r.loc[r.split.eq('train')],seed)
        m.to_csv(ROOT/f'metrics/oversampling_seed{seed}.csv',index=False)
        old=pd.read_csv(P.src(OLD/f'metrics/augmentation_seed{seed}.csv'))
        assert pd.read_csv(ROOT/f'metrics/oversampling_seed{seed}.csv').equals(old)
        assert m.label.value_counts().to_dict()=={0:999,1:999}
        assert rows.iloc[m.source_cache_index].split.eq('train').all()
        manifests[seed]=m
    specs={str(p):P.sha(p) for p in [ROOT/'DESIGN.md',Path(__file__)]}
    P.dump(ROOT/'FROZEN.json',specs)
    P.dump(ROOT/'PRETRAINING_AUDIT_PASSED.json',dict(test_accessed=False,gpu=torch.cuda.get_device_name(0),parameters=params,manifest_parity=True))
    result=[]
    for seed,m in manifests.items():
        duplicated={k:v[m.source_cache_index].copy() for k,v in a.items()}
        assert all(np.array_equal(v,a[k][m.source_cache_index]) for k,v in duplicated.items())
        for arm in 'ABCD':
            result.extend(train(arm,seed,rows,a,m,duplicated))
            pd.DataFrame(result).to_csv(ROOT/'metrics/per_seed_validation.csv',index=False)
    df=pd.DataFrame(result)
    keys=['f1','pr_auc','roc_auc','accuracy','precision','recall','fp','fn']
    summary=df.groupby(['arm','selection'])[keys].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns]
    summary.to_csv(ROOT/'metrics/summary_validation.csv')
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
