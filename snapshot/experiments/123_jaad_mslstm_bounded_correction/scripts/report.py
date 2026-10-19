"""Saved-probability verification and bounded-correction validation report."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,average_precision_score,roc_auc_score,accuracy_score

ROOT=Path(__file__).resolve().parents[1]
assert (ROOT/'VALIDATION_COMPLETE.json').exists()
d=pd.read_csv(ROOT/'metrics/per_seed_validation.csv');assert len(d)==12
keys=['f1','pr_auc','roc_auc','accuracy','precision','recall','fp','fn']
summary=d.groupby('arm')[keys+['branch_parameters']].agg(['mean','std'])
summary.columns=['_'.join(k) for k in summary.columns];summary.to_csv(ROOT/'metrics/summary_validation.csv')
transitions=[];reach=[];errors=[];paired=[]
for seed in (42,43,44):
    base=pd.read_csv(ROOT/f'predictions/validation_baseline_seed{seed}.csv')
    y=base.label.to_numpy();p0=base.probability.to_numpy();b=p0>=.5
    lo=1/(1+np.exp(.2));hi=1/(1+np.exp(-.2));reachable=(p0>=lo)&(p0<=hi)
    reach.append(dict(seed=seed,reachable_samples=int(reachable.sum()),baseline_errors=int((b!=y).sum()),
        reachable_errors=int((reachable&(b!=y)).sum()),reachable_fp=int((reachable&b&(y==0)).sum()),reachable_fn=int((reachable&~b&(y==1)).sum())))
    for arm in ('baseline','raw','core56','full63'):
        saved=pd.read_csv(ROOT/f'predictions/validation_{arm}_seed{seed}.csv')
        assert saved.sequence_id.equals(base.sequence_id) and saved.label.equals(base.label)
        p=saved.probability.to_numpy();c=saved.correction_logit.to_numpy();pred=p>=.5
        assert np.abs(c).max()<=.200001 and len(saved)==274 and set(saved.split)=={'val'}
        reconstructed=1/(1+np.exp(-(np.log(p0/(1-p0))+c)))
        assert np.abs(p-reconstructed).max()<2e-6
        assert not ((pred!=b)&~reachable).any()
        r=d[(d.seed==seed)&(d.arm==arm)].iloc[0]
        actual=dict(f1=f1_score(y,pred),pr_auc=average_precision_score(y,p),roc_auc=roc_auc_score(y,p),accuracy=accuracy_score(y,pred))
        errors.append(max(abs(actual[k]-r[k]) for k in actual))
        if arm!='baseline':
            transitions.append(dict(seed=seed,arm=arm,wrong_to_correct=int(((b!=y)&(pred==y)).sum()),
                correct_to_wrong=int(((b==y)&(pred!=y)).sum()),max_correction=float(np.abs(c).max())))
    sub=d[d.seed==seed].set_index('arm')
    for ref,arm in [('baseline','raw'),('baseline','core56'),('baseline','full63'),('raw','core56'),('raw','full63')]:
        paired.append(dict(seed=seed,reference=ref,arm=arm,**{k:float(sub.loc[arm,k]-sub.loc[ref,k]) for k in keys}))
assert max(errors)<1e-12
pd.DataFrame(reach).to_csv(ROOT/'metrics/cap_reachability.csv',index=False)
pd.DataFrame(transitions).to_csv(ROOT/'metrics/error_transitions.csv',index=False)
pd.DataFrame(paired).to_csv(ROOT/'metrics/paired_deltas.csv',index=False)
out=['# Exp123: frozen MSLSTM bounded correction','',
'Validation-only retrospective integration study. Parent=Exp122 A AP-selected checkpoint for each seed, unchanged. No test access. Three added branches use raw pose, BIU56, or BIU56+seven auxiliaries63. Cap0.10*2*tanh applies during training and inference. Epoch0 identity is eligible under AP selection.','',
'## Three-seed means, fixed threshold0.5','',
'| Arm | F1 | AP | ROC-AUC | Accuracy | FP | FN | Branch parameters |',
'| --- | --- | --- | --- | --- | --- | --- | --- |']
for arm in ('baseline','raw','core56','full63'):
    r=summary.loc[arm]
    out.append(f'| {arm} | {r.f1_mean:.4f} | {r.pr_auc_mean:.4f} | {r.roc_auc_mean:.4f} | {r.accuracy_mean:.4f} | {r.fp_mean:.1f} | {r.fn_mean:.1f} | {r.branch_parameters_mean:.0f} |')
out+=['','## Paired directions','',
'| Comparison | F1 delta | AP delta | ROC delta | Positive seeds F1/AP/ROC |','| --- | --- | --- | --- | --- |']
for (ref,arm),g in pd.DataFrame(paired).groupby(['reference','arm']):
    r=g[['f1','pr_auc','roc_auc']].mean();n=(g[['f1','pr_auc','roc_auc']]>0).sum()
    out.append(f'| {arm} minus {ref} | {r.f1:+.4f} | {r.pr_auc:+.4f} | {r.roc_auc:+.4f} | {n.f1}/3, {n.pr_auc}/3, {n.roc_auc}/3 |')
out+=['','## Correction reachability','',
'Only parent probabilities approximately0.450166--0.549834 can change class under this cap. The cap limits changes to confident baseline decisions, including confidently wrong decisions.','',
'| Seed | Baseline errors | Errors within reach | Reachable FP | Reachable FN |','| --- | --- | --- | --- | --- |']
for r in reach: out.append(f'| {r["seed"]} | {r["baseline_errors"]} | {r["reachable_errors"]} | {r["reachable_fp"]} | {r["reachable_fn"]} |')
out+=['','## Integrity and interpretation boundaries','',
f'All12 saved validation files reproduce metrics within {max(errors):.3g}; bounded-logit reconstruction and unreachable-decision preservation passed. Parent checkpoint/state immutability, replay and no-gradient checks passed. Nine branches trained; no source models were retrained or changed.',
'','Validation AP cannot be worse than the replayed baseline AP because epoch0 identity was eligible. That is a selection property, not independent generalization evidence. F1 can still deteriorate. This design differs from jointly trained Exp122 and from uncapped-training PedCMT; do not attribute every difference to one scalar fusion weight.',
'','No cap/threshold search, final test evaluation, new architecture, thesis modification, commit or push followed this result. Full per-seed metrics, sample SD and transitions are in `metrics/`.',
'','## Sources and commands','',
'Frozen sources: `metrics/source_hashes.json`; procedure: `DESIGN.md`; parent paths: `../122_jaad_mslstm_no_noise/checkpoints/A_seed{42,43,44}_ap.pt`; exact parent replay evidence: `PRETRAINING_AUDIT_PASSED.json`.',
'','Training: `env LD_LIBRARY_PATH=/usr/lib/wsl/lib PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python -u experiments/123_jaad_mslstm_bounded_correction/scripts/run.py`.',
'','Report: `env PYTHONDONTWRITEBYTECODE=1 /home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/123_jaad_mslstm_bounded_correction/scripts/report.py`.']
(ROOT/'FINAL_REPORT.md').write_text('\n'.join(out)+'\n')
(ROOT/'REPORT_COMPLETE.json').write_text(json.dumps(dict(prediction_files=12,max_metric_error=max(errors),test_accessed=False),indent=2)+'\n')
print(summary.to_string())
