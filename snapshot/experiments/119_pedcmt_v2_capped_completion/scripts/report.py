"""Verify saved new predictions and summarize the frozen completion without refitting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

ROOT=Path(__file__).resolve().parents[1]
assert (ROOT/'EXPERIMENT_COMPLETE.json').exists()
summary=pd.read_csv(ROOT/'metrics/summary_results.csv')
per=pd.read_csv(ROOT/'metrics/per_seed_results.csv')
checks=[]
for case,arms in {'pie':['raw','core56','full63'],'jaad':['core56','full63']}.items():
    for arm in arms:
        for seed in (42,43,44):
            for split in ('validation','test'):
                p=ROOT/case/f'predictions/{split}_{arm}_seed{seed}.csv'
                d=pd.read_csv(p)
                row=per.query('case==@case and arm==@arm and seed==@seed and split==@split').iloc[0]
                for metric,fn in [('f1',f1_score),('accuracy',accuracy_score),('roc_auc',roc_auc_score),('pr_auc',average_precision_score)]:
                    actual=fn(d.label,d.probability>=.5) if metric in ('f1','accuracy') else fn(d.label,d.probability)
                    assert abs(actual-row[metric])<1e-12,(case,arm,seed,split,metric)
                assert np.array_equal(d.prediction,(d.probability>=.5).astype(int))
                assert d.full_correction_logit.abs().max()<=2.000001
                checks.append(dict(case=case,arm=arm,seed=seed,split=split,rows=len(d),verified=True))
(ROOT/'metrics/saved_prediction_verification.json').write_text(json.dumps(checks,indent=2)+'\n')
delta=pd.read_csv(ROOT/'metrics/paired_deltas.csv')
text=['# Exp119: v2-only capped PedCMT completion','',
      'All 15 new trainings and both final evaluations completed. Means and sample SD are over seeds 42/43/44; fixed threshold 0.5, AP reported as PR-AUC. Retrospective matched controls, not pristine confirmations. No test-based model/cap selection.','',
      '## Capped results','',
      '| Dataset | Arm | F1 mean +/- SD | AP mean +/- SD | ROC-AUC mean +/- SD | Accuracy | Mean FP | Mean FN |',
      '| --- | --- | --- | --- | --- | --- | --- | --- |']
for case in ('pie','jaad'):
    for arm in ('baseline','raw','core56','full63'):
        r=summary.query('case==@case and split=="test" and arm==@arm').iloc[0]
        text.append(f'| {case} | {arm} | '+ ' | '.join(f'{r[k+"_mean"]:.4f} +/- {r[k+"_std"]:.4f}' for k in ('f1','pr_auc','roc_auc'))+f' | {r.accuracy_mean:.4f} | {r.fp_mean:.1f} | {r.fn_mean:.1f} |')
text+=['','## Paired deltas and seed direction','',
       '| Dataset | Comparison | Mean F1 delta | Mean AP delta | Mean ROC delta | Favorable seeds F1/AP/ROC |',
       '| --- | --- | --- | --- | --- | --- |']
for (case,ref,arm),g in delta.groupby(['case','reference','arm']):
    text.append(f'| {case} | {arm} minus {ref} | '+ ' | '.join(f'{g[k].mean():+.4f}' for k in ('f1','pr_auc','roc_auc'))+' | '+' / '.join(str(int((g[k]>0).sum()))+'/3' for k in ('f1','pr_auc','roc_auc'))+' |')
text+=['','## Interpretation','',
       '- On PIE, both v2 forms improve mean capped F1 over the parent and matched raw pose. Full63 improves F1/AP/ROC over both in every seed. Core56 ranking improvements are not uniformly positive for AP.',
       '- On clean JAAD, both v2 forms improve mean F1/AP/ROC over the parent and capped raw pose. F1 improves in two seeds and ties in the third; both ranking metrics improve in every seed. Full63 exceeds core56 in mean metrics, with F1 tied in one seed.',
       '- These are small PIE refinements and model-specific JAAD improvements, not a claim of universal v2 superiority or statistical significance. No clustered confidence intervals were computed in this completion.',
       '- The predeclared uncapped probabilities are a diagnostic from the same trained branches; they do not replace the capped primary comparison or the earlier direct-token family.',
       '- JAAD raw predictions are reused exactly from Exp094. Parents are authenticated Exp084/089 checkpoints. Training uses original cached parent outputs. GPU replay differs slightly in floating point; this can alter tied probability rankings at tiny scales, so parent AP/ROC may differ slightly from historical CSVs while fixed-threshold decisions agree.',
       '- This fills the capped v2-only gap without relying on union135. It does not fill the separately running clean-JAAD MSLSTM family (Exp120).',
       '', '## Sources','',
       '`DESIGN.md`, `metrics/pretraining_sources.json`, `metrics/all_sources.json`, `metrics/per_seed_results.csv`, `metrics/summary_results.csv`, `metrics/paired_deltas.csv`, `metrics/error_transitions.csv`, per-case predictions and completion/checkpoint markers. Exact reused/fresh experiment mappings: `MISSING_RESULTS_INVENTORY.md`. All new saved probabilities recompute their reported metrics within 1e-12; see `metrics/saved_prediction_verification.json`. No thesis edits, commit or push.']
(ROOT/'FINAL_REPORT.md').write_text('\n'.join(text)+'\n')
print('\n'.join(text[:17]))
