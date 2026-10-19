"""One frozen test evaluation, AP checkpoints only; no fitting code invoked."""
import importlib.util
import sys
import json
from pathlib import Path
import time
import fcntl
import numpy as np
import pandas as pd
import torch

sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent;REPO=ROOT.parents[1]
source=REPO/'experiments/120_jaad_mslstm_v2_completion/scripts/run.py'
spec=importlib.util.spec_from_file_location('exp122_evaluation_core',source)
P=importlib.util.module_from_spec(spec);sys.modules[spec.name]=P;spec.loader.exec_module(P)
P.ROOT=OUT

def scene(rows):
    records={}
    for video in sorted(rows.video_id.unique()):
        path=P.src(P.exp(49)/'cache/yolop/yolop_parts'/f'{video}.npz')
        with np.load(path,allow_pickle=False) as f:
            for ped,frame,value in zip(f['ped_id'],f['frame'],f['features']):
                records[(video,str(ped),int(frame))]=value
    return np.stack([np.stack([records[(r.video_id,r.ped_id,int(f))] for f in json.loads(r.observed_frames_json)]) for r in rows.itertuples()])
P.scene=scene

def main():
    assert not (OUT/'TEST_COMPLETE.json').exists()
    lock=(OUT/'.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for d in ('metrics','predictions','logs'): (OUT/d).mkdir(exist_ok=True)
    assert (ROOT/'VALIDATION_COMPLETE.json').exists() and (ROOT/'REPORT_COMPLETE.json').exists()
    for file in ('FROZEN.json','metrics/source_hashes.json'):
        for path,h in json.loads((ROOT/file).read_text()).items(): assert P.sha(path)==h, path
    assert torch.cuda.is_available();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    frozen={};states={}
    for seed in (42,43,44):
        for arm in 'ABCD':
            path=ROOT/f'checkpoints/{arm}_seed{seed}_ap.pt';frozen[str(path)]=P.sha(path)
            ck=torch.load(path,map_location='cpu',weights_only=False)
            h=pd.read_csv(ROOT/f'logs/{arm}_seed{seed}.history.csv')
            expected=int(h.loc[h.pr_auc.idxmax(),'epoch'])
            assert ck['arm']==arm and ck['seed']==seed and ck['selection']=='ap' and ck['best_epoch']==expected
            states[(arm,seed)]=ck
    for path in [OUT/'DESIGN.md',Path(__file__),ROOT/'metrics/bbox53_scaler.json',ROOT/'metrics/feature_contract.json',source]:
        frozen[str(path)]=P.sha(path)
    P.dump(OUT/'FROZEN_EVALUATION.json',frozen)
    P.log('AP checkpoint/epoch audit passed; opening approved test cohort')
    P.TEST=True
    scaler={k:np.asarray(v) for k,v in json.loads((ROOT/'metrics/bbox53_scaler.json').read_text()).items()}
    rows,a,_,_=P.data('test',scaler)
    dev=pd.read_csv(P.src(P.exp(89)/'cache/trainval_source/rows.csv'))
    assert not set(rows.video_id)&set(dev.video_id)
    assert not set(rows.sequence_id)&set(dev.sequence_id)
    assert len(rows)==1501 and rows.label.value_counts().to_dict()=={0:781,1:720}
    result=[];transitions=[];paired=[]
    for seed in (42,43,44):
        probs={}
        for arm in 'ABCD':
            model=P.C.MslstmPracticalModel(arm).to(P.DEVICE)
            model.load_state_dict(states[(arm,seed)]['model_state'],strict=True)
            p=P.TRAIN.probabilities(model,P.tensors(a,arm),len(rows));assert np.isfinite(p).all()
            P.savepred(rows,p,arm,seed,'test');probs[arm]=p
            metrics=P.C.compute_metrics(rows.label,p)
            saved=pd.read_csv(OUT/f'predictions/test_{arm}_seed{seed}.csv')
            verified=P.C.compute_metrics(saved.label,saved.probability)
            assert all(abs(metrics[k]-verified[k])<1e-12 for k in metrics)
            result.append(dict(seed=seed,arm=arm,split='test',selected_epoch=states[(arm,seed)]['best_epoch'],**metrics))
            P.log(f'{arm}{seed} test F1={metrics["f1"]:.6f} AP={metrics["pr_auc"]:.6f}')
        sub=pd.DataFrame(result).query('seed==@seed').set_index('arm');y=rows.label.to_numpy()
        for ref,arm in [('A','B'),('A','C'),('A','D'),('B','C'),('B','D'),('C','D')]:
            b=probs[ref]>=.5;p=probs[arm]>=.5
            paired.append(dict(seed=seed,reference=ref,arm=arm,**{k:float(sub.loc[arm,k]-sub.loc[ref,k]) for k in ('f1','pr_auc','roc_auc','accuracy','fp','fn')}))
            transitions.append(dict(seed=seed,reference=ref,arm=arm,wrong_to_correct=int(((b!=y)&(p==y)).sum()),correct_to_wrong=int(((b==y)&(p!=y)).sum()),
                fp_corrected=int(((y==0)&b&~p).sum()),fn_corrected=int(((y==1)&~b&p).sum()),
                fp_introduced=int(((y==0)&~b&p).sum()),fn_introduced=int(((y==1)&b&~p).sum())))
    d=pd.DataFrame(result);d.to_csv(OUT/'metrics/per_seed_results.csv',index=False)
    keys=['f1','pr_auc','roc_auc','accuracy','precision','recall','tp','tn','fp','fn']
    summary=d.groupby('arm')[keys].agg(['mean','std']);summary.columns=['_'.join(k) for k in summary.columns]
    summary.to_csv(OUT/'metrics/summary_results.csv')
    pd.DataFrame(paired).to_csv(OUT/'metrics/paired_deltas.csv',index=False)
    pd.DataFrame(transitions).to_csv(OUT/'metrics/error_transitions.csv',index=False)
    for p,h in {**P.SOURCES,**frozen}.items(): assert P.sha(p)==h,p
    P.dump(OUT/'metrics/source_hashes.json',P.SOURCES)
    text=['# Exp122: approved clean-JAAD test results','',
          'All four arms, seeds42/43/44, existing validation-AP-selected checkpoints; fixed threshold0.5. No retraining, threshold search or test-based selection.1501 test windows (720 crossing,781 noncrossing),16 observed frames,TTE30--60. Retrospective controlled follow-up on an exposed cohort, not pristine confirmation.','',
          '| Arm | F1 mean +/- SD | AP mean +/- SD | ROC-AUC mean +/- SD | Accuracy | Precision | Recall | Mean FP | Mean FN |',
          '| --- | --- | --- | --- | --- | --- | --- | --- | --- |']
    for arm,name in [('A','Baseline'),('B','Raw pose'),('C','BIU56'),('D','BIU56 +7 auxiliaries63')]:
        r=summary.loc[arm]
        text.append(f'| {name} | '+' | '.join(f'{r[k+"_mean"]:.4f} +/- {r[k+"_std"]:.4f}' for k in ('f1','pr_auc','roc_auc'))+f' | {r.accuracy_mean:.4f} | {r.precision_mean:.4f} | {r.recall_mean:.4f} | {r.fp_mean:.1f} | {r.fn_mean:.1f} |')
    text+=['','## Paired comparisons','', '| Comparison | F1 delta | AP delta | ROC delta | Positive seeds F1/AP/ROC |','| --- | --- | --- | --- | --- |']
    for (ref,arm),g in pd.DataFrame(paired).groupby(['reference','arm']):
        r=g[['f1','pr_auc','roc_auc']].mean();n=(g[['f1','pr_auc','roc_auc']]>0).sum()
        text.append(f'| {arm} minus {ref} | {r.f1:+.4f} | {r.pr_auc:+.4f} | {r.roc_auc:+.4f} | {n.f1}/3, {n.pr_auc}/3, {n.roc_auc}/3 |')
    text+=['','## Scope and verification','',
        'A=BBox53+categorical vehicle state+YOLOP; B=A+raw pose; C=A+BIU56; D=A+BIU56+7aux. No acceleration. Training-only scaler reused. AP means average precision, not trapezoidal PR area. Sample SD across three seeds is not a confidence interval. No significance claim.',
        '','Frozen training/source hashes, all checkpoint selection epochs, disjoint development/test videos/IDs, fixed cohort counts, saved-probability metrics and post-evaluation source/checkpoint hashes passed. Exact input source paths/hashes are in `metrics/source_hashes.json`; checkpoint paths/hashes in `FROZEN_EVALUATION.json`. All predictions are in `predictions/`. Original Exp120 and Exp122 validation results remain untouched.',
        '','Command: `env LD_LIBRARY_PATH=/usr/lib/wsl/lib PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python -u experiments/122_jaad_mslstm_no_noise/test_evaluation/evaluate.py`.',
        '','No model changes, thesis edits, commits, pushes or further experiment followed evaluation.']
    (OUT/'FINAL_REPORT.md').write_text('\n'.join(text)+'\n')
    P.dump(OUT/'TEST_COMPLETE.json',dict(prediction_files=12,source_integrity=True,saved_predictions_verified=True,time=time.time()))
    P.log('TEST EVALUATION COMPLETE')

if __name__=='__main__':
    try: main()
    except Exception as e:
        P.dump(OUT/'FAILURE.json',dict(error=repr(e),time=time.time()));raise
