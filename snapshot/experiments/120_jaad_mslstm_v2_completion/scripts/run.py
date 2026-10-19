"""Matched clean-JAAD MSLSTM controls with exact observed-frame scene extraction."""
import copy
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess
import time

sys.dont_write_bytecode = True
import cv2
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
SEEDS = (42,43,44)
TEST = False
SOURCES = {}
DEVICE = torch.device('cuda')


def exp(n):
    p = list((REPO/'experiments').glob(f'{n:03d}_*'))
    assert len(p)==1
    return p[0]


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def src(p):
    p=Path(p).resolve(strict=True)
    if not TEST and any('test_source' in x or 'set03' in x for x in p.parts):
        raise PermissionError(f'Final test locked: {p}')
    if str(p) not in SOURCES: SOURCES[str(p)]=sha(p)
    return p


def load(p,name):
    src(p)
    sys.path.insert(0,str(p.parent))
    spec=importlib.util.spec_from_file_location(name,p)
    m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m
    spec.loader.exec_module(m)
    return m


def dump(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp')
    t.write_text(json.dumps(x,indent=2)+'\n')
    os.replace(t,p)


def log(s):
    print(time.strftime('%Y-%m-%d %H:%M:%S')+' '+s,flush=True)


W=load(exp(53)/'scripts/run_experiment.py','exp120_parent')
C=W.core
TRAIN=W.base
Y=load(exp(49)/'scripts/yolop.py','exp120_yolop')
CORE=load(exp(50)/'scripts/exp050_core.py','exp120_preprocessing')
C.VARIANTS=('A','B','C','D')
base=('bbox53','vehicle_state','yolop')
C.VARIANT_STREAMS={'A':base,'B':base+('pose',),'C':base+('biu56',),'D':base+('biu63',)}
C.STREAM_DIMS.update(vehicle_state=1,biu56=56,biu63=63)
C.WINDOW_VECTOR_STREAMS=set(C.WINDOW_VECTOR_STREAMS)|{'biu56','biu63'}


def scene(rows):
    """Extract each image once, retaining exact tracked-pedestrian identities."""
    required={}
    for r in rows.itertuples():
        frames=json.loads(r.observed_frames_json)
        boxes=json.loads(r.bbox_sequence_json)
        assert len(frames)==len(boxes)==16 and frames==list(range(r.start_frame,r.end_frame+1))
        for f,b in zip(frames,boxes):
            key=(r.video_id,int(f),r.ped_id)
            if key in required: assert required[key]==b
            required[key]=b
    records={}
    jobs=[]
    for vi,video in enumerate(sorted({k[0] for k in required})):
        keys=sorted(k for k in required if k[0]==video)
        local=ROOT/'cache/yolop'/f'{video}.npz'
        old=exp(49)/'cache/yolop/yolop_parts'/f'{video}.npz'
        for p in (old,local):
            if p.exists():
                with np.load(src(p),allow_pickle=False) as data:
                    for ped,f,vector in zip(data['ped_id'],data['frame'],data['features']):
                        key=(video,int(f),str(ped))
                        if key in required:
                            assert vector.shape==(20,) and np.isfinite(vector).all()
                            records[key]=vector
        missing=[k for k in keys if k not in records]
        for key in missing:
            image=Y.JAAD_IMAGE_ROOT/video/f'{key[1]:05d}.png'
            src(image)
            jobs.append(dict(video=video,frame=key[1],ped=key[2],bbox=required[key],image=str(image)))
        log(f'YOLOP videos={vi+1} frame_records={len(records)}/{len(required)}')
    if jobs:
        src(Y.YOLOP_WEIGHTS);src(Y.YOLOP_NOTEBOOK)
        request=ROOT/'cache'/('scene_test_request.json' if TEST else 'scene_development_request.json')
        dump(request,jobs)
        worker=src(ROOT/'scripts/extract_scene.py')
        subprocess.run(['/home/abdelrahman/miniconda3/envs/mslstm_pid_env/bin/python','-u',str(worker),str(request)],check=True)
        for video in sorted({j['video'] for j in jobs}):
            with np.load(ROOT/'cache/yolop'/f'{video}.npz',allow_pickle=False) as payload:
                for ped,fr,v in zip(payload['ped_id'],payload['frame'],payload['features']):
                    records[(video,int(fr),str(ped))]=v
    return np.stack([np.stack([records[(r.video_id,int(f),r.ped_id)] for f in json.loads(r.observed_frames_json)]) for r in rows.itertuples()])


def data(phase,scaler=None):
    path=exp(89)/'cache'/f'{phase}_source'
    rows=pd.read_csv(src(path/'rows.csv'))
    expected={'train':1552,'val':274} if phase=='trainval' else {'test':1501}
    assert rows.split.value_counts().to_dict()==expected
    assert not rows.sequence_id.duplicated().any()
    assert rows.tte_frames.between(30,60).all() and (rows.end_frame-rows.start_frame).eq(15).all()
    metadata=json.loads(src(path/'metadata.json').read_text())
    names=metadata['biu_v2_feature_names']
    defs=pd.read_csv(src(exp(12)/'metrics/pie_biu_v2_feature_set_definitions.csv')).set_index('feature_set')
    names56=json.loads(defs.loc['biu_v2_core','feature_columns_json'])
    ix=[names.index(n) for n in names56]
    assert len(ix)==56 and len(names)==63
    xy=np.load(src(path/'bbox.npy')).astype(np.float32)
    bbox=np.stack([(xy[:,:,0]+xy[:,:,2])/2,(xy[:,:,1]+xy[:,:,3])/2,xy[:,:,2]-xy[:,:,0],xy[:,:,3]-xy[:,:,1]],axis=2)
    arrays={k:np.load(src(path/f'{v}.npy')).astype(np.float32) for k,v in
            [('pose','pose'),('biu63','biu_v2'),('vehicle_state','ego_speed')]}
    assert arrays['pose'].shape==(len(rows),16,34)
    assert np.isin(arrays['vehicle_state'],[0,.25,.5,.75,1]).all()
    arrays['bbox']=bbox
    arrays['biu56']=arrays['biu63'][:,ix]
    b53=np.stack([C.bbox53_from_normalized_bbox(bbox[i],width=int(r.frame_width),height=int(r.frame_height)) for i,r in enumerate(rows.itertuples())])
    if scaler is None:
        assert phase=='trainval'
        tr=rows.split.eq('train')
        assert not set(rows.loc[tr,'video_id']) & set(rows.loc[~tr,'video_id'])
        scaler=C.fit_bbox53_scaler(b53[tr])
        dump(ROOT/'metrics/bbox53_scaler.json',{k:v.tolist() for k,v in scaler.items()})
        dump(ROOT/'metrics/feature_contract.json',dict(core56=names56,full63=names,streams=C.VARIANT_STREAMS))
    arrays['bbox53']=C.transform_bbox53(b53,scaler)
    arrays['yolop']=scene(rows)
    assert all(len(v)==len(rows) and np.isfinite(v).all() for v in arrays.values())
    return rows,arrays,scaler,ix


def augment(rows,a,scaler,ix,seed):
    r=rows.copy();r['cache_index']=np.arange(len(r))
    manifest=CORE.build_oversampling_manifest(r.loc[r.split.eq('train')],seed)
    manifest.to_csv(ROOT/f'metrics/augmentation_seed{seed}.csv',index=False)
    out={k:v[manifest.source_cache_index].copy() for k,v in a.items()}
    for j,item in enumerate(manifest.itertuples()):
        if not item.augmented: continue
        i=item.source_cache_index
        measurements={k:a[k][i] for k in ('bbox','pose','yolop')}
        changed=CORE.augment_measurements(measurements,noise_seed=item.noise_seed)
        for k,v in changed.items(): out[k][j]=v
        width,height=int(rows.iloc[i].frame_width),int(rows.iloc[i].frame_height)
        v,names,_=CORE.extract_biu_v2_window(changed['pose'],changed['bbox'],width=width,height=height)
        assert list(names)==json.loads((ROOT/'metrics/feature_contract.json').read_text())['full63']
        out['biu63'][j]=v;out['biu56'][j]=v[ix]
        out['bbox53'][j]=C.transform_bbox53(C.bbox53_from_normalized_bbox(changed['bbox'],width=width,height=height)[None],scaler)[0]
    assert np.array_equal(out['vehicle_state'],a['vehicle_state'][manifest.source_cache_index])
    assert all(np.isfinite(v).all() for v in out.values())
    return manifest,out


def tensors(a,arm,idx=None):
    return {k:torch.as_tensor(a[k] if idx is None else a[k][idx],device=DEVICE,dtype=torch.float32) for k in C.VARIANT_STREAMS[arm]}


def savepred(rows,p,arm,seed,phase):
    out=rows.copy();out['probability']=p;out['prediction']=(p>=.5).astype(int)
    path=ROOT/f'predictions/{phase}_{arm}_seed{seed}.csv'
    path.parent.mkdir(exist_ok=True)
    out.to_csv(path,index=False)


def train(arm,seed,rows,a,m,aug):
    marker=ROOT/f'logs/{arm}_seed{seed}.complete.json'
    ck=ROOT/f'checkpoints/{arm}_seed{seed}.best.pt'
    if marker.exists():
        info=json.loads(marker.read_text());assert sha(ck)==info['sha256'];return
    C.seed_everything(seed)
    model=C.MslstmPracticalModel(arm).to(DEVICE)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
    x=tensors(aug,arm);y=torch.tensor(m.label.to_numpy(),device=DEVICE,dtype=torch.long)
    va=np.flatnonzero(rows.split.eq('val'));vx=tensors(a,arm,va)
    rng=np.random.RandomState(seed);best=-1.;history=[];started=time.time()
    for epoch in range(1,16):
        model.train();order=rng.permutation(len(y));loss_sum=0.
        for start in range(0,len(y),32):
            ids=torch.tensor(order[start:start+32],device=DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits=model({k:v[ids] for k,v in x.items()})
            loss=torch.nn.functional.cross_entropy(logits,y[ids])
            assert torch.isfinite(loss)
            loss.backward();optimizer.step();loss_sum+=float(loss)*len(ids)
        p=TRAIN.probabilities(model,vx,len(va));metrics=C.compute_metrics(rows.iloc[va].label,p)
        history.append(dict(epoch=epoch,train_loss=loss_sum/len(y),**metrics))
        if metrics['f1']>best:
            best=metrics['f1'];best_epoch=epoch
            best_state=copy.deepcopy({k:v.cpu() for k,v in model.state_dict().items()})
        pd.DataFrame(history).to_csv(ROOT/f'logs/{arm}_seed{seed}.history.csv',index=False)
        log(f'{arm}{seed} epoch={epoch}/15 val_f1={metrics["f1"]:.6f} best_epoch={best_epoch}')
    model.load_state_dict(best_state)
    p=TRAIN.probabilities(model,vx,len(va));savepred(rows.iloc[va],p,arm,seed,'validation')
    ck.parent.mkdir(exist_ok=True)
    torch.save(dict(model_state=best_state,arm=arm,seed=seed,best_epoch=best_epoch,
                    streams=C.VARIANT_STREAMS[arm],parameters=C.count_parameters(model)),ck)
    dump(marker,dict(sha256=sha(ck),best_epoch=best_epoch,seconds=time.time()-started,
         parameters=C.count_parameters(model),validation_sha256=sha(ROOT/f'predictions/validation_{arm}_seed{seed}.csv')))


def main():
    global TEST
    if (ROOT/'EXPERIMENT_COMPLETE.json').exists(): raise RuntimeError('Already complete')
    lock=(ROOT/'.execution.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for d in ('logs','metrics','cache','checkpoints','predictions'): (ROOT/d).mkdir(exist_ok=True)
    assert torch.cuda.is_available()
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    for n in (49,50,51,53,89):
        for p in (exp(n)/'scripts').glob('*.py'): src(p)
    for p in (REPO/'src').glob('*biu*py'): src(p)
    src(C.ARCHIVE_BBOX_CORE_PATH)
    specs={str(p):sha(p) for p in [ROOT/'DESIGN.md',Path(__file__),ROOT/'scripts/extract_scene.py']}
    frozen=ROOT/'FROZEN.json'
    if frozen.exists(): assert json.loads(frozen.read_text())==specs
    rows,a,scaler,ix=data('trainval')
    # Freeze source hashes only after deterministic development preparation passes.
    for arm in C.VARIANTS:
        model=C.MslstmPracticalModel(arm).to(DEVICE)
        loss=torch.nn.functional.cross_entropy(model(tensors(a,arm,np.arange(4))),torch.tensor(rows.label.to_numpy()[:4],device=DEVICE))
        loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    dump(ROOT/'metrics/pretraining_sources.json',SOURCES)
    dump(ROOT/'PRETRAINING_AUDIT_PASSED.json',dict(test_accessed=False,gpu=torch.cuda.get_device_name(0),smoke_backward=True))
    dump(frozen,specs)
    for seed in SEEDS:
        manifest,aug=augment(rows,a,scaler,ix,seed)
        for arm in C.VARIANTS: train(arm,seed,rows,a,manifest,aug)
    hashes={str(p):sha(p) for p in ROOT.glob('checkpoints/*.best.pt')}
    assert len(hashes)==12
    for p,h in SOURCES.items(): assert sha(p)==h
    dump(ROOT/'TRAINING_COMPLETE.json',dict(checkpoints=hashes,test_accessed=False))
    TEST=True
    testrows,test,_,_=data('test',scaler)
    assert not set(testrows.sequence_id)&set(rows.sequence_id)
    assert not set(testrows.video_id)&set(rows.video_id)
    result=[];transitions=[]
    for seed in SEEDS:
        probabilities={}
        for arm in C.VARIANTS:
            ck=ROOT/f'checkpoints/{arm}_seed{seed}.best.pt';assert sha(ck)==hashes[str(ck)]
            model=C.MslstmPracticalModel(arm).to(DEVICE)
            state=torch.load(ck,map_location=DEVICE,weights_only=False);model.load_state_dict(state['model_state'],strict=True)
            p=TRAIN.probabilities(model,tensors(test,arm),len(testrows));savepred(testrows,p,arm,seed,'test')
            probabilities[arm]=p
            result.append(dict(seed=seed,arm=arm,split='test',parameters=C.count_parameters(model),**C.compute_metrics(testrows.label,p)))
            val=pd.read_csv(ROOT/f'predictions/validation_{arm}_seed{seed}.csv')
            result.append(dict(seed=seed,arm=arm,split='validation',parameters=C.count_parameters(model),**C.compute_metrics(val.label,val.probability)))
        y=testrows.label.to_numpy()
        for ref in ('A','B'):
            for arm in ('C','D'):
                pa,pb=probabilities[ref]>=.5,probabilities[arm]>=.5
                transitions.append(dict(seed=seed,reference=ref,arm=arm,wrong_to_correct=int(((pa!=y)&(pb==y)).sum()),correct_to_wrong=int(((pa==y)&(pb!=y)).sum())))
    out=pd.DataFrame(result);out.to_csv(ROOT/'metrics/per_seed_results.csv',index=False)
    summary=out.groupby(['split','arm']).agg({k:['mean','std'] for k in out.select_dtypes('number').columns if k!='seed'})
    summary.columns=['_'.join(c) for c in summary.columns]
    summary.reset_index().to_csv(ROOT/'metrics/summary_results.csv',index=False)
    pd.DataFrame(transitions).to_csv(ROOT/'metrics/error_transitions.csv',index=False)
    paired=[]
    for seed in SEEDS:
        sub=out.query('seed==@seed and split=="test"').set_index('arm')
        for ref in ('A','B','C'):
            for arm in ('B','C','D'):
                if arm<=ref: continue
                paired.append(dict(seed=seed,reference=ref,arm=arm,**{k:float(sub.loc[arm,k]-sub.loc[ref,k]) for k in ('f1','pr_auc','roc_auc','accuracy')}))
    pd.DataFrame(paired).to_csv(ROOT/'metrics/paired_deltas.csv',index=False)
    for p,h in SOURCES.items(): assert sha(p)==h
    dump(ROOT/'metrics/all_sources.json',SOURCES)
    dump(ROOT/'EXPERIMENT_COMPLETE.json',dict(runs=12,seeds=SEEDS,time=time.time()))
    log('EXPERIMENT COMPLETE')


if __name__=='__main__':
    try: main()
    except Exception as e:
        dump(ROOT/'FAILURE.json',dict(error=repr(e),time=time.time(),test_accessed=TEST))
        raise
