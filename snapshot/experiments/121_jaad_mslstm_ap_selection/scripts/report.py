"""Verify saved development probabilities and summarize the frozen comparison."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score,f1_score,accuracy_score

ROOT=Path(__file__).resolve().parents[1]
assert (ROOT/'VALIDATION_COMPLETE.json').exists()
d=pd.read_csv(ROOT/'metrics/per_seed_validation.csv')
errors=[]
for r in d.itertuples():
    p=pd.read_csv(ROOT/f'predictions/validation_{r.arm}_seed{r.seed}_{r.selection}.csv')
    assert set(p.split)=={'val'} and len(p)==274
    y=p.label.to_numpy();v=p.probability.to_numpy()
    actual=dict(f1=f1_score(y,v>=.5),pr_auc=average_precision_score(y,v),roc_auc=roc_auc_score(y,v),accuracy=accuracy_score(y,v>=.5))
    errors.append(max(abs(actual[k]-getattr(r,k)) for k in actual))
assert max(errors)<1e-12
replay=[json.loads(p.read_text()) for p in sorted((ROOT/'logs').glob('*.complete.json'))]
assert len(replay)==12
summary=d.groupby(['arm','selection'])[['f1','pr_auc','roc_auc','accuracy','fp','fn']].mean()
out=['# Exp121: validation-only checkpoint-selection comparison','',
'Retrospective development evidence only. No test source was accessed or evaluated. Same twelve training trajectories, inputs and seeds; only checkpoint selection differs. A=baseline, B=raw pose, C=BIU56, D=BIU56+seven auxiliaries63.','',
'## Validation means (three seeds, fixed threshold 0.5)','',
'| Arm | Selection | F1 | AP | ROC-AUC | Accuracy | FP | FN |','| --- | --- | --- | --- | --- | --- | --- | --- |']
for (arm,selection),r in summary.iterrows():
    out.append(f'| {arm} | {selection} | {r.f1:.4f} | {r.pr_auc:.4f} | {r.roc_auc:.4f} | {r.accuracy:.4f} | {r.fp:.1f} | {r.fn:.1f} |')
out+=['','## Paired interpretation','',
'AP-selected validation AP cannot be lower than F1-selected validation AP on the same trajectory; the reverse holds for F1. This is a consequence of selecting on those metrics, NOT independent evidence that the selection change improves generalization. Other metrics and failure modes are descriptive. No threshold was tuned.','',
'| AP-selected comparison | Mean F1 delta | Mean AP delta | Mean ROC delta | Positive seeds F1/AP/ROC |',
'| --- | --- | --- | --- | --- |']
ap=d[d.selection.eq('ap')].set_index(['arm','seed'])
for ref,arm in [('A','B'),('A','C'),('A','D'),('B','C'),('B','D'),('C','D')]:
    delta=ap.loc[arm][['f1','pr_auc','roc_auc']]-ap.loc[ref][['f1','pr_auc','roc_auc']]
    mean=delta.mean();counts=(delta>0).sum()
    out.append(f'| {arm} minus {ref} | {mean.f1:+.4f} | {mean.pr_auc:+.4f} | {mean.roc_auc:+.4f} | {counts.f1}/3, {counts.pr_auc}/3, {counts.roc_auc}/3 |')
out+=['','## Integrity','',
f'All 24 saved validation prediction files reproduce metrics within {max(errors):.3g}. Twelve replay histories have maximum absolute difference {max(r["history_max_error"] for r in replay):.3g}; F1-selected probabilities differ from Exp120 by at most {max(r["f1_prediction_max_error"] for r in replay):.3g}.',
'','## Augmentation measurements','',
'All duplicated rows retain the inherited Gaussian augmentation; nothing was changed based on this diagnostic. Ratios below compare absolute perturbation with original-training feature standard deviation; they are not calibrated uncertainty or proof that measurements are physically invalid.','',
'| Feature | Mean across seeds: median change / train SD | Mean across seeds: p95 change / train SD | Mean fraction changed |',
'| --- | --- | --- | --- |']
aug=pd.concat([pd.read_csv(p) for p in sorted((ROOT/'metrics').glob('augmentation_diagnostic_seed*.csv'))])
means=aug.groupby('feature')[['median_training_sd_change','p95_training_sd_change','fraction_changed']].mean()
for name in ['direction_sign','was_flipped','body_yaw_angle_delta','stride_length_std','hip_center_x_velocity_std','lower_body_motion_magnitude_mean','torso_motion_magnitude_mean']:
    r=means.loc[name]
    out.append(f'| {name} | {r.median_training_sd_change:.3f} | {r.p95_training_sd_change:.3f} | {r.fraction_changed:.3f} |')
out+=['','## Scope and reproduction','',
'No new test claims, source changes, model redesign, thesis edits or commits. The original Exp120 negative result remains intact. This completes the authorized validation comparison; final test evaluation or a new intervention requires a separate decision.',
'','Training: `env LD_LIBRARY_PATH=/usr/lib/wsl/lib PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python -u experiments/121_jaad_mslstm_ap_selection/scripts/run.py`.',
'','Verification/report: `env PYTHONDONTWRITEBYTECODE=1 /home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/121_jaad_mslstm_ap_selection/scripts/report.py`.',
'','Exact sources: `metrics/source_hashes.json`. Checkpoints: `checkpoints/`. Predictions: `predictions/validation_*.csv`. Epoch histories and replay checks: `logs/`. Protocol: `DESIGN.md`.']
(ROOT/'FINAL_REPORT.md').write_text('\n'.join(out)+'\n')
(ROOT/'REPORT_COMPLETE.json').write_text(json.dumps(dict(prediction_files=24,max_metric_error=max(errors),test_accessed=False),indent=2)+'\n')
print(summary.to_string())
