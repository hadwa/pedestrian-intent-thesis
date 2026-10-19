"""Verify validation probabilities; report every matched no-noise comparison."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score,average_precision_score,roc_auc_score,accuracy_score,precision_score,recall_score

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'121_jaad_mslstm_ap_selection'
assert (ROOT/'VALIDATION_COMPLETE.json').exists()
d=pd.read_csv(ROOT/'metrics/per_seed_validation.csv')
old=pd.read_csv(OLD/'metrics/per_seed_validation.csv')
assert len(d)==24
errors=[]
for r in d.itertuples():
    p=pd.read_csv(ROOT/f'predictions/validation_{r.arm}_seed{r.seed}_{r.selection}.csv')
    prior=pd.read_csv(OLD/f'predictions/validation_{r.arm}_seed{r.seed}_{r.selection}.csv')
    assert p.sequence_id.equals(prior.sequence_id) and p.label.equals(prior.label)
    assert len(p)==274 and set(p.split)=={'val'}
    y=p.label.to_numpy();v=p.probability.to_numpy();yp=v>=.5
    actual=dict(f1=f1_score(y,yp),pr_auc=average_precision_score(y,v),roc_auc=roc_auc_score(y,v),
        accuracy=accuracy_score(y,yp),precision=precision_score(y,yp),recall=recall_score(y,yp),
        fp=int(((y==0)&yp).sum()),fn=int(((y==1)&~yp).sum()))
    errors.append(max(abs(actual[k]-getattr(r,k)) for k in actual))
assert max(errors)<1e-12
keys=['f1','pr_auc','roc_auc','accuracy','precision','recall','fp','fn']
paired=[]
for r in d.itertuples():
    ref=old[(old.arm==r.arm)&(old.seed==r.seed)&(old.selection==r.selection)].iloc[0]
    paired.append(dict(arm=r.arm,seed=r.seed,selection=r.selection,**{k:float(getattr(r,k)-ref[k]) for k in keys}))
delta=pd.DataFrame(paired);delta.to_csv(ROOT/'metrics/paired_noise_removal_deltas.csv',index=False)
means=d.groupby(['arm','selection'])[keys].mean()
oldmeans=old.groupby(['arm','selection'])[keys].mean()
out=['# Exp122: no measurement noise, validation-only results','',
'Same clean-JAAD population and balanced training manifests as Exp121. Only measurement-noise augmentation is removed. A=baseline, B=raw pose, C=canonical BIU56, D=BIU56+seven auxiliaries63. Means over42/43/44; threshold0.5. These are retrospective development/selection-set results, not test estimates.','',
'## Primary AP-selected comparison','',
'| Arm | Training | F1 | AP | ROC-AUC | Accuracy | FP | FN |',
'| --- | --- | --- | --- | --- | --- | --- | --- |']
for arm in 'ABCD':
    for name,frame in [('Exp121 noise',oldmeans),('Exp122 no noise',means)]:
        r=frame.loc[(arm,'ap')]
        out.append(f'| {arm} | {name} | {r.f1:.4f} | {r.pr_auc:.4f} | {r.roc_auc:.4f} | {r.accuracy:.4f} | {r.fp:.1f} | {r.fn:.1f} |')
out+=['','## Paired seed directions for noise removal','',
'| Arm | Mean F1 delta | Mean AP delta | Mean ROC delta | Positive seeds F1/AP/ROC |',
'| --- | --- | --- | --- | --- |']
for arm in 'ABCD':
    s=delta[(delta.arm==arm)&(delta.selection=='ap')][['f1','pr_auc','roc_auc']]
    r=s.mean();n=(s>0).sum()
    out.append(f'| {arm} | {r.f1:+.4f} | {r.pr_auc:+.4f} | {r.roc_auc:+.4f} | {n.f1}/3, {n.pr_auc}/3, {n.roc_auc}/3 |')
out+=['','## Within-run representation contrasts','',
'| Comparison (AP selected) | F1 delta | AP delta | ROC delta | Positive seeds F1/AP/ROC |',
'| --- | --- | --- | --- | --- |']
new=d[d.selection=='ap'].set_index(['arm','seed'])
for ref,arm in [('A','B'),('A','C'),('A','D'),('B','C'),('B','D'),('C','D')]:
    s=new.loc[arm][['f1','pr_auc','roc_auc']]-new.loc[ref][['f1','pr_auc','roc_auc']]
    r=s.mean();n=(s>0).sum()
    out.append(f'| {arm} minus {ref} | {r.f1:+.4f} | {r.pr_auc:+.4f} | {r.roc_auc:+.4f} | {n.f1}/3, {n.pr_auc}/3, {n.roc_auc}/3 |')
out+=['','## Secondary F1-selected diagnostic (not primary model selection)','',
'| Arm | F1 | AP | ROC-AUC |','| --- | --- | --- | --- |']
for arm in 'ABCD':
    r=means.loc[(arm,'f1')]
    out.append(f'| {arm} | {r.f1:.4f} | {r.pr_auc:.4f} | {r.roc_auc:.4f} |')
out+=['','## Verification and boundaries','',
f'All24 saved prediction files reproduce metrics within {max(errors):.3g}; validation IDs/labels match Exp121 exactly. Balanced row manifests match exactly. No test data was accessed, no threshold changed, no feature or architecture redesigned. Duplicate observations remain dependent observations, not new data.',
'','Noise removal comparisons include the resulting optimization/checkpoint changes. They do not isolate an individual cue or prove that all augmentation is harmful. AP-selected validation scores are selected on this same validation set; seed consistency is descriptive, not statistical significance. No automatic test evaluation or further experiment is launched.',
'','## Exact commands and sources','',
'Training: `env LD_LIBRARY_PATH=/usr/lib/wsl/lib PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python -u experiments/122_jaad_mslstm_no_noise/scripts/run.py`.',
'','Report: `env PYTHONDONTWRITEBYTECODE=1 /home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/122_jaad_mslstm_no_noise/scripts/report.py`.',
'','Original-development caches and source code: `metrics/source_hashes.json`. Old comparator: `../121_jaad_mslstm_ap_selection/metrics/per_seed_validation.csv` and its `predictions/validation_*.csv`. New results: `metrics/per_seed_validation.csv`, `metrics/summary_validation.csv`, `metrics/paired_noise_removal_deltas.csv`, `predictions/validation_*.csv`. No thesis edits, commits or pushes.']
(ROOT/'FINAL_REPORT.md').write_text('\n'.join(out)+'\n')
sources=[OLD/'metrics/per_seed_validation.csv',*sorted((OLD/'predictions').glob('validation_*.csv'))]
(ROOT/'metrics/comparator_hashes.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},indent=2)+'\n')
(ROOT/'REPORT_COMPLETE.json').write_text(json.dumps(dict(prediction_files=24,max_metric_error=max(errors),test_accessed=False),indent=2)+'\n')
print(means.to_string())
