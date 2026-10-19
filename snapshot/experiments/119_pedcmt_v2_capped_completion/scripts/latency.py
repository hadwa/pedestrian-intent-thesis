"""Complete capped predictor latency on ready validation inputs, without refitting."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]
assert (ROOT/'EXPERIMENT_COMPLETE.json').exists()
torch.set_num_threads(1);torch.set_num_interop_threads(1)
allowed=os.sched_getaffinity(0);cpu=2 if 2 in allowed else min(allowed);os.sched_setaffinity(0,{cpu})
spec=importlib.util.spec_from_file_location('latency_completion',ROOT/'scripts/run.py')
M=importlib.util.module_from_spec(spec);sys.modules[spec.name]=M;spec.loader.exec_module(M)
exp=M.exp
records=[];counts=[];parity=[]
for case in ('pie','jaad'):
    parentroot=exp(84 if case=='pie' else 89)
    folder=exp(50)/'cache/trainval_base' if case=='pie' else parentroot/'cache/trainval_source'
    rows=pd.read_csv(folder/'rows.csv');idx=np.flatnonzero(rows.split.eq('val'))
    idx=idx[np.linspace(0,len(idx)-1,32,dtype=int)]
    chosen=rows.iloc[idx]
    bbox=np.load(folder/'bbox.npy')[idx]
    if case=='pie': bbox=M.P93.EXP084_DATA._center_bbox_to_xyxy(bbox)
    ego=np.load(folder/'ego_speed.npy')[idx]
    full=np.load(parentroot/'cache/trainval/biu_v2_std.npy')[idx]
    raw=np.load(parentroot/'cache/trainval/pose_std.npy')[idx]
    names=json.loads((folder/'metadata.json').read_text())['biu_v2_feature_names']
    core=json.loads((ROOT/case/'metrics/feature_contract.json').read_text())['core56']
    extra={'raw':raw,'full63':full,'core56':full[:,[names.index(n) for n in core]]}
    samples=[(torch.tensor(bbox[i:i+1],dtype=torch.float32),torch.tensor(ego[i:i+1],dtype=torch.float32)) for i in range(32)]
    for seed in (42,43,44):
        parent=M.P93.load_parent(seed,1,torch.device('cpu')) if case=='pie' else M.BASE.load_parent_model(seed,torch.device('cpu'))
        captured=[]
        hook=parent.endpoint.register_forward_pre_hook(lambda module,args:captured.append(args[0]))
        for arm in ('baseline','raw','core56','full63'):
            branch=None
            if arm!='baseline':
                ck=exp(94)/f'checkpoints/P_seed{seed}.best.pt' if case=='jaad' and arm=='raw' else ROOT/case/f'checkpoints/{arm}_seed{seed}.best.pt'
                branch=M.model(arm).eval();branch.load_state_dict(torch.load(ck,map_location='cpu',weights_only=False)['model_state'],strict=True)
                extras=[torch.tensor(v[None],dtype=torch.float32) for v in extra[arm]]
            def forward(i):
                captured.clear();logit=parent(*samples[i])[0]
                if branch is not None:
                    assert len(captured)==1
                    correction=branch(logit,captured[0],extras[i])[1]
                    logit=logit+.1*correction
                return torch.sigmoid(logit)
            if arm=='baseline': pred=parentroot/f'predictions/validation_A_seed{seed}.csv'
            elif case=='jaad' and arm=='raw': pred=exp(94)/f'predictions/validation_P10_seed{seed}.csv'
            else: pred=ROOT/case/f'predictions/validation_{arm}_seed{seed}.csv'
            saved=pd.read_csv(pred).set_index('sequence_id').loc[chosen.sequence_id].probability.to_numpy()
            with torch.inference_mode():
                p=np.array([forward(i).item() for i in range(32)])
                assert np.max(np.abs(p-saved))<1e-4
                assert np.array_equal(p>=.5,saved>=.5)
                parity.append(dict(case=case,arm=arm,seed=seed,max_abs_error=float(np.max(np.abs(p-saved)))))
                for block in range(3):
                    for i in range(60): forward(i%32)
                    for i in range(200):
                        start=time.perf_counter_ns();forward(i%32)
                        records.append(dict(case=case,arm=arm,seed=seed,block=block,ms=(time.perf_counter_ns()-start)/1e6))
            n=sum(p.numel() for p in parent.parameters());b=0 if branch is None else sum(p.numel() for p in branch.parameters())
            counts.append(dict(case=case,arm=arm,seed=seed,frozen_parent_parameters=n,trainable_branch_parameters=b,total_parameters=n+b))
        hook.remove()
out=pd.DataFrame(records);out.to_csv(ROOT/'metrics/latency_raw.csv',index=False)
out.groupby(['case','arm']).ms.agg(mean_ms='mean',median_ms='median',p95_ms=lambda x:x.quantile(.95),n='count').reset_index().to_csv(ROOT/'metrics/efficiency.csv',index=False)
pd.DataFrame(counts).to_csv(ROOT/'metrics/parameter_counts.csv',index=False)
(ROOT/'metrics/latency_replay.json').write_text(json.dumps(parity,indent=2)+'\n')
(ROOT/'metrics/latency_scope.json').write_text(json.dumps(dict(threads=1,cpu=cpu,batch=1,windows=32,
    warmups_per_block=60,measurements_per_block=200,blocks=3,seeds=[42,43,44],
    scope='ready inputs through frozen parent and correction to probability; excludes feature preparation/perception; validation windows only',
    torch_version=torch.__version__),indent=2)+'\n')
print('CPU efficiency complete',flush=True)
