"""Wait for frozen results, verify probabilities and report all matched neural families."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import subprocess
import time

sys.dont_write_bytecode=True
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score,f1_score,roc_auc_score,average_precision_score

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[1]


def exp(n):
    paths=list((REPO/'experiments').glob(f'{n:03d}_*'));assert len(paths)==1
    return paths[0]


def load(path,name):
    sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def wait():
    started=time.time()
    while not (ROOT/'EXPERIMENT_COMPLETE.json').exists():
        if (ROOT/'FAILURE.json').exists():
            raise RuntimeError('Exp120 failed; inspect FAILURE.json, do not present partial completion')
        if time.time()-started>12*3600: raise TimeoutError('12-hour reporting wait expired')
        time.sleep(15)


def verify():
    per=pd.read_csv(ROOT/'metrics/per_seed_results.csv')
    verified=[]
    for r in per.itertuples():
        d=pd.read_csv(ROOT/f'predictions/{r.split}_{r.arm}_seed{r.seed}.csv')
        for key,fn in [('accuracy',accuracy_score),('f1',f1_score),('roc_auc',roc_auc_score),('pr_auc',average_precision_score)]:
            observed=fn(d.label,d.probability>=.5) if key in ('accuracy','f1') else fn(d.label,d.probability)
            assert abs(observed-getattr(r,key))<1e-12
        verified.append(dict(arm=r.arm,seed=r.seed,split=r.split,n=len(d)))
    (ROOT/'metrics/verified_predictions.json').write_text(json.dumps(verified,indent=2)+'\n')


def latency():
    """Ready-input CPU cost only, batch one; all representation preprocessing excluded."""
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    affinity=os.sched_getaffinity(0);cpu=2 if 2 in affinity else min(affinity)
    os.sched_setaffinity(0,{cpu})
    m=load(ROOT/'scripts/run.py','report_mslstm')
    folder=exp(89)/'cache/trainval_source'
    rows=pd.read_csv(folder/'rows.csv')
    indices=np.flatnonzero(rows.split.eq('val'))[np.linspace(0,273,32,dtype=int)]
    chosen=rows.iloc[indices].reset_index(drop=True)
    xy=np.load(folder/'bbox.npy')[indices]
    bbox=np.stack([(xy[:,:,0]+xy[:,:,2])/2,(xy[:,:,1]+xy[:,:,3])/2,xy[:,:,2]-xy[:,:,0],xy[:,:,3]-xy[:,:,1]],axis=2)
    scaler={k:np.array(v) for k,v in json.loads((ROOT/'metrics/bbox53_scaler.json').read_text()).items()}
    b53=np.stack([m.C.bbox53_from_normalized_bbox(bbox[i],width=int(r.frame_width),height=int(r.frame_height)) for i,r in enumerate(chosen.itertuples())])
    contract=json.loads((ROOT/'metrics/feature_contract.json').read_text())
    ix=[contract['full63'].index(n) for n in contract['core56']]
    arrays={'bbox53':m.C.transform_bbox53(b53,scaler),
            'pose':np.load(folder/'pose.npy')[indices],
            'vehicle_state':np.load(folder/'ego_speed.npy')[indices],
            'biu63':np.load(folder/'biu_v2.npy')[indices]}
    arrays['biu56']=arrays['biu63'][:,ix]
    parts={}
    for v in chosen.video_id.unique():
        with np.load(ROOT/'cache/yolop'/f'{v}.npz') as d:
            parts[v]={(str(p),int(fr)):f for p,fr,f in zip(d['ped_id'],d['frame'],d['features'])}
    arrays['yolop']=np.stack([np.stack([parts[r.video_id][(r.ped_id,int(f))] for f in json.loads(r.observed_frames_json)]) for r in chosen.itertuples()])
    samples=[{k:torch.tensor(v[i:i+1],dtype=torch.float32) for k,v in arrays.items()} for i in range(32)]
    raw=[];counts=[]
    for arm in ('A','B','C','D'):
        for seed in (42,43,44):
            path=ROOT/f'checkpoints/{arm}_seed{seed}.best.pt'
            model=m.C.MslstmPracticalModel(arm).eval()
            model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model_state'],strict=True)
            with torch.inference_mode():
                reproduced=np.array([torch.softmax(model(s),dim=1)[0,1].item() for s in samples])
                saved=pd.read_csv(ROOT/f'predictions/validation_{arm}_seed{seed}.csv').set_index('sequence_id').loc[chosen.sequence_id]
                assert np.max(np.abs(reproduced-saved.probability))<1e-4
                assert np.array_equal(reproduced>=.5,saved.probability.to_numpy()>=.5)
                for block in range(3):
                    for i in range(60): torch.softmax(model(samples[i%32]),dim=1)
                    for i in range(200):
                        s=samples[i%32];start=time.perf_counter_ns()
                        torch.softmax(model(s),dim=1)
                        raw.append(dict(arm=arm,seed=seed,block=block,ms=(time.perf_counter_ns()-start)/1e6))
            branch=sum(p.numel() for name,p in model.named_parameters() if any(f'.{x}.' in name for x in ('pose','biu56','biu63')))
            counts.append(dict(arm=arm,seed=seed,total_parameters=m.C.count_parameters(model),pose_pathway_parameters=branch,checkpoint_bytes=path.stat().st_size))
    frame=pd.DataFrame(raw);frame.to_csv(ROOT/'metrics/latency_raw.csv',index=False)
    summary=frame.groupby('arm').ms.agg(mean_ms='mean',median_ms='median',p95_ms=lambda x:x.quantile(.95),n='count').reset_index()
    summary.to_csv(ROOT/'metrics/efficiency.csv',index=False)
    pd.DataFrame(counts).to_csv(ROOT/'metrics/parameter_counts.csv',index=False)
    (ROOT/'metrics/latency_scope.json').write_text(json.dumps(dict(cpu_logical_id=cpu,threads=1,batch=1,
        warmups_per_block=60,measurements_per_block=200,blocks=3,seeds=[42,43,44],windows=32,
        scope='ready tensors to probabilities; excludes all perception/feature construction/normalization/input allocation',
        probability_replay_atol=1e-4,decisions_match=True,torch_version=torch.__version__),indent=2)+'\n')


def report():
    rows=[]
    old=pd.read_csv(exp(117)/'metrics/summary_results.csv')
    for case,mapping in [('mslstm_pie',{'G':'baseline','J':'raw','core56':'core56','I':'full63'}),
                         ('pedcmt_pie',{'A':'baseline','B':'raw','core56':'core56','D':'full63'}),
                         ('pedcmt_jaad',{'A':'baseline','B':'raw','core56':'core56','D':'full63'})]:
        for arm,name in mapping.items():
            r=old.query('case==@case and split=="test" and arm==@arm').iloc[0].to_dict()
            rows.append(dict(r,family=case+'_direct' if case.startswith('pedcmt') else case,representation=name,source='Exp117 (reused parent controls)'))
    for _,r in pd.read_csv(exp(119)/'metrics/summary_results.csv').query('split=="test"').iterrows():
        rows.append(dict(r,family='pedcmt_'+r['case']+'_capped',representation=r['arm'],source='Exp119; JAAD raw reused Exp094'))
    for _,r in pd.read_csv(ROOT/'metrics/summary_results.csv').query('split=="test"').iterrows():
        rows.append(dict(r,family='mslstm_jaad',representation={'A':'baseline','B':'raw','C':'core56','D':'full63'}[r['arm']],source='Exp120'))
    combined=pd.DataFrame(rows)
    combined.to_csv(ROOT/'metrics/all_neural_families.csv',index=False)
    text=['# Missing neural comparisons: consolidated completion','',
          'Matched comparisons are within each family only. PIE and clean JAAD have distinct datasets/non-pose inputs. Direct-token and capped families use different integration/training designs. No API_beh/API_all expansion. Existing PIE MSLSTM and direct PedCMT runs are reused, not retrained.','',
          '| Family | Representation | F1 mean +/- SD | AP mean +/- SD | ROC mean +/- SD | Accuracy |',
          '| --- | --- | --- | --- | --- | --- |']
    for r in rows:
        text.append(f'| {r["family"]} | {r["representation"]} | '+' | '.join(f'{r[k+"_mean"]:.4f} +/- {r[k+"_std"]:.4f}' for k in ('f1','pr_auc','roc_auc'))+f' | {r["accuracy_mean"]:.4f} |')
    text+=['','## Clean-JAAD MSLSTM paired evidence','',
           '| Comparison | Mean F1 delta | Mean AP delta | Mean ROC delta | Positive seeds F1/AP/ROC |',
           '| --- | --- | --- | --- | --- |']
    delta=pd.read_csv(ROOT/'metrics/paired_deltas.csv')
    for (ref,arm),g in delta.groupby(['reference','arm']):
        text.append(f'| {arm} minus {ref} | '+' | '.join(f'{g[k].mean():+.4f}' for k in ('f1','pr_auc','roc_auc'))+' | '+' / '.join(f'{int((g[k]>0).sum())}/3' for k in ('f1','pr_auc','roc_auc'))+' |')
    text+=['','A=baseline, B=raw pose, C=core56, D=full63. These descriptive seed comparisons do not establish statistical significance. No result was suppressed by a validation gate. Negative/neutral direct-integration evidence remains visible rather than being replaced by capped results.',
           '', '## Efficiency','',
           'Ready-input one-core CPU measurements and parameter counts: `metrics/efficiency.csv`, `metrics/parameter_counts.csv`, `metrics/latency_scope.json`. This excludes upstream perception and BIU/BBox construction and is not an end-to-end comparison. A raw-pose/BIU branch-cost difference is not proof of accuracy causality.',
           '', '## Provenance and limitations','',
           'Exp119 FINAL_REPORT.md details capped v2-only findings and reused JAAD raw controls. Exp120 DESIGN.md specifies the approved vehicle-state/no-acceleration adaptation. Exact old-family rows are from Exp117 summary artifacts; metrics/all_neural_families.csv retains their source labels. Existing and new tests are retrospective matched evidence on exposed cohorts, not pristine confirmations or exact published-paper reproduction. No thesis edits, additional model search, commit or push.']
    (ROOT/'FINAL_REPORT.md').write_text('\n'.join(text)+'\n')
    (ROOT/'REPORT_COMPLETE.json').write_text(json.dumps(dict(time=time.time(),verified_predictions=True,cpu_latency_complete=True))+'\n')


if __name__=='__main__':
    try:
        wait();verify();latency()
        subprocess.run([sys.executable,str(exp(119)/'scripts/latency.py')],check=True)
        report()
    except Exception as e:
        (ROOT/'REPORT_FAILURE.json').write_text(json.dumps(dict(time=time.time(),error=repr(e)))+'\n')
        raise
