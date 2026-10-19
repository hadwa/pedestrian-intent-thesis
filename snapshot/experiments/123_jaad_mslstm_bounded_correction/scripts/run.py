"""Validation-only frozen MSLSTM correction with matched pose/BIU branches."""
import copy
import importlib.util
import sys
from pathlib import Path
import time
import fcntl
import numpy as np
import pandas as pd
import torch
from torch import nn

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parents[1]
OLD=REPO/'experiments/122_jaad_mslstm_no_noise'
spec=importlib.util.spec_from_file_location('exp123_parent',OLD/'scripts/run.py')
Q=importlib.util.module_from_spec(spec);sys.modules[spec.name]=Q;spec.loader.exec_module(Q)
P=Q.P;P.ROOT=ROOT;P.TEST=False

class Correction(nn.Module):
    def __init__(self,arm):
        super().__init__();self.arm=arm
        if arm=='raw':
            self.encoder=nn.LSTM(34,256,2,batch_first=True,dropout=.3,bidirectional=True)
            self.attention=P.C.StreamAttention()
        else:
            self.encoder=nn.Sequential(nn.Linear(56 if arm=='core56' else 63,128),nn.ReLU(),nn.Dropout(.3),nn.Linear(128,512))
        self.anchor=nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.LayerNorm(128))
        self.adapter=nn.Sequential(nn.Linear(512,128),nn.GELU(),nn.LayerNorm(128))
        self.head=nn.Sequential(nn.Linear(512,64),nn.GELU(),nn.Dropout(.3),nn.Linear(64,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
    def forward(self,logit,anchor,extra):
        e=self.attention(self.encoder(extra)[0]) if self.arm=='raw' else self.encoder(extra)
        a=self.anchor(anchor);e=self.adapter(e)
        interaction=torch.cat([a,e,a*e,torch.abs(a-e)],dim=1)
        change=.2*torch.tanh(self.head(interaction).squeeze(1))
        return logit+change,change

def capture(model,a,rows):
    vectors=torch.empty((len(rows),256));logits=torch.empty(len(rows));captured=[]
    handle=model.fc2.register_forward_pre_hook(lambda m,x:captured.append(x[0].detach()))
    with torch.no_grad():
        for split in ('train','val'):
            indices=np.flatnonzero(rows.split.eq(split))
            for start in range(0,len(indices),512):
                captured.clear();ids=indices[start:start+512]
                output=model(P.tensors(a,'A',ids))
                assert len(captured)==1 and captured[0].shape[1]==256
                vectors[ids]=captured[0].cpu();logits[ids]=(output[:,1]-output[:,0]).cpu()
    handle.remove()
    return logits.to(P.DEVICE),vectors.to(P.DEVICE)

def evaluate(model,inputs,ids):
    model.eval();prob=[];changes=[]
    with torch.no_grad():
        for start in range(0,len(ids),32):
            ix=ids[start:start+32]
            logit,change=model(*[v[ix] for v in inputs])
            assert torch.isfinite(logit).all() and change.abs().max()<=.200001
            prob.append(logit.sigmoid().cpu().numpy());changes.append(change.cpu().numpy())
    return np.concatenate(prob),np.concatenate(changes)

def savepred(rows,p,change,arm,seed):
    out=rows.copy();out['probability']=p;out['prediction']=(p>=.5).astype(int);out['correction_logit']=change
    out.to_csv(ROOT/f'predictions/validation_{arm}_seed{seed}.csv',index=False)

def train(arm,seed,rows,a,m,logit,anchor):
    P.C.seed_everything(seed);model=Correction(arm).to(P.DEVICE)
    key={'raw':'pose','core56':'biu56','full63':'biu63'}[arm]
    inputs=(logit,anchor,torch.as_tensor(a[key],device=P.DEVICE))
    va=np.flatnonzero(rows.split.eq('val'));y=rows.label.to_numpy()
    p,change=evaluate(model,inputs,va);assert np.count_nonzero(change)==0
    best=P.C.compute_metrics(y[va],p)['pr_auc'];best_epoch=0
    state=copy.deepcopy({k:v.cpu() for k,v in model.state_dict().items()})
    history=[dict(epoch=0,train_loss=np.nan,**P.C.compute_metrics(y[va],p))]
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    rng=np.random.RandomState(seed);indices=m.source_cache_index.to_numpy()
    labels=torch.tensor(y,device=P.DEVICE,dtype=torch.float32)
    for epoch in range(1,16):
        model.train();order=rng.permutation(len(indices));loss_sum=0.
        for start in range(0,len(order),32):
            ix=indices[order[start:start+32]]
            optimizer.zero_grad(set_to_none=True)
            pred,change=model(*[v[ix] for v in inputs])
            loss=nn.functional.binary_cross_entropy_with_logits(pred,labels[ix])
            assert torch.isfinite(loss);loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            optimizer.step();loss_sum+=float(loss)*len(ix)
        p,change=evaluate(model,inputs,va);metrics=P.C.compute_metrics(y[va],p)
        history.append(dict(epoch=epoch,train_loss=loss_sum/len(indices),**metrics))
        if metrics['pr_auc']>best:
            best=metrics['pr_auc'];best_epoch=epoch
            state=copy.deepcopy({k:v.cpu() for k,v in model.state_dict().items()})
        pd.DataFrame(history).to_csv(ROOT/f'logs/{arm}_seed{seed}.history.csv',index=False)
        P.log(f'{arm}{seed} epoch={epoch}/15 val_AP={metrics["pr_auc"]:.6f} best_epoch={best_epoch}')
    model.load_state_dict(state);p,change=evaluate(model,inputs,va)
    savepred(rows.iloc[va],p,change,arm,seed)
    torch.save(dict(model_state=state,arm=arm,seed=seed,best_epoch=best_epoch,cap=.2),ROOT/f'checkpoints/{arm}_seed{seed}.pt')
    params=sum(p.numel() for p in model.parameters())
    P.dump(ROOT/f'logs/{arm}_seed{seed}.complete.json',dict(best_epoch=best_epoch,branch_parameters=params,max_absolute_correction=float(np.abs(change).max())))
    return dict(arm=arm,seed=seed,epoch=best_epoch,branch_parameters=params,**P.C.compute_metrics(y[va],p))

def main():
    assert not (ROOT/'VALIDATION_COMPLETE.json').exists()
    for d in ('logs','metrics','cache','predictions','checkpoints'): (ROOT/d).mkdir(exist_ok=True)
    lock=(ROOT/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert torch.cuda.is_available();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    try: P.src(REPO/'test_source/blocked.npy')
    except AssertionError: pass
    else: raise AssertionError('test guard failed')
    for n in (49,50,51,53,89,120,121,122):
        for path in (P.exp(n)/'scripts').glob('*.py'): P.src(path)
    for path in (REPO/'src').glob('*biu*py'): P.src(path)
    P.src(P.C.ARCHIVE_BBOX_CORE_PATH)
    rows,a,scaler,ix=P.data('trainval');assert len(rows)==1826 and set(rows.split)=={'train','val'}
    va=np.flatnonzero(rows.split.eq('val'));prepared={};audit=[]
    for seed in (42,43,44):
        ck=P.src(OLD/f'checkpoints/A_seed{seed}_ap.pt')
        model=P.C.MslstmPracticalModel('A').to(P.DEVICE)
        state=torch.load(ck,map_location=P.DEVICE,weights_only=False)['model_state'];model.load_state_dict(state)
        model.eval();model.requires_grad_(False)
        logit,anchor=capture(model,a,rows)
        ref=pd.read_csv(P.src(OLD/f'predictions/validation_A_seed{seed}_ap.csv'))
        assert ref.sequence_id.tolist()==rows.iloc[va].sequence_id.tolist()
        error=float(np.max(np.abs(logit[va].sigmoid().cpu().numpy()-ref.probability)))
        P.log(f'parent{seed} validation replay error={error:.9g}')
        assert error<2e-6, (seed,error)
        m=pd.read_csv(P.src(OLD/f'metrics/oversampling_seed{seed}.csv'))
        assert rows.iloc[m.source_cache_index].split.eq('train').all()
        assert np.array_equal(rows.iloc[m.source_cache_index].label,m.label)
        assert m.label.value_counts().to_dict()=={0:999,1:999}
        for arm in ('raw','core56','full63'):
            branch=Correction(arm).to(P.DEVICE)
            key={'raw':'pose','core56':'biu56','full63':'biu63'}[arm]
            z,c=branch(logit[:4],anchor[:4],torch.tensor(a[key][:4],device=P.DEVICE))
            assert torch.count_nonzero(c)==0
            nn.functional.binary_cross_entropy_with_logits(z,torch.tensor(rows.label.to_numpy()[:4],device=P.DEVICE,dtype=torch.float32)).backward()
            assert all(torch.isfinite(p.grad).all() for p in branch.parameters() if p.grad is not None)
        assert all(p.grad is None for p in model.parameters())
        audit.append(dict(seed=seed,parent_replay_max_error=error,parent_parameters=sum(p.numel() for p in model.parameters()),parent_hash=P.sha(ck)))
        prepared[seed]=(model,copy.deepcopy(state),logit,anchor,m)
    specs={str(p):P.sha(p) for p in [ROOT/'DESIGN.md',Path(__file__)]}
    P.dump(ROOT/'FROZEN.json',specs)
    P.dump(ROOT/'PRETRAINING_AUDIT_PASSED.json',dict(test_accessed=False,gpu=torch.cuda.get_device_name(0),parents=audit,zero_correction=True))
    result=[]
    for seed,(parent,state,logit,anchor,m) in prepared.items():
        p=logit[va].sigmoid().cpu().numpy();savepred(rows.iloc[va],p,np.zeros(len(va)),'baseline',seed)
        result.append(dict(arm='baseline',seed=seed,epoch=-1,branch_parameters=0,**P.C.compute_metrics(rows.iloc[va].label,p)))
        for arm in ('raw','core56','full63'):
            result.append(train(arm,seed,rows,a,m,logit,anchor))
            assert all(torch.equal(v,state[k]) for k,v in parent.state_dict().items())
            assert all(p.grad is None for p in parent.parameters())
            pd.DataFrame(result).to_csv(ROOT/'metrics/per_seed_validation.csv',index=False)
    for p,h in P.SOURCES.items(): assert P.sha(p)==h
    for p,h in specs.items(): assert P.sha(p)==h
    P.dump(ROOT/'metrics/source_hashes.json',P.SOURCES)
    P.dump(ROOT/'VALIDATION_COMPLETE.json',dict(runs=9,test_accessed=False,time=time.time()))
    P.log('VALIDATION COMPLETE; TEST NOT ACCESSED')

if __name__=='__main__':
    try: main()
    except Exception as e:
        P.dump(ROOT/'FAILURE.json',dict(error=repr(e),time=time.time(),test_accessed=False));raise
