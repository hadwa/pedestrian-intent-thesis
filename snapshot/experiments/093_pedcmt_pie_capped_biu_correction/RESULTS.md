# Exp093 Results

## Status and scope

Exp093 is a completed exploratory cross-dataset transfer test on the standard
PIE 16-observation-frame, TTE 30--60 benchmark. It applies the fixed correction
scale selected previously on JAAD without tuning that scale on PIE.

The experiment reuses the exact Exp084 Variant A BBox+ego checkpoints and adds
one freshly trained 100,225-parameter BIU correction branch. The parent model,
pose cache, BIU definition, splits and fixed 0.5 decision threshold are
unchanged. PIE set03 contains 3,760 test windows.

## Fixed-0.5 test results

| Variant | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FP | FN | HC FP | HC FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Exp084 A parent | 0.8887 | 0.7921 | 0.7947 | 0.7926 | 0.9192 | 0.8584 | 212.0 | 206.3 | 57.3 | 95.0 |
| R10: parent + 10% BIU correction | 0.8898 | 0.7939 | 0.7960 | 0.7944 | 0.9230 | 0.8674 | 209.3 | 205.0 | 53.7 | 92.7 |
| Mean delta, R10 - parent | +0.0011 | +0.0018 | +0.0013 | +0.0018 | +0.0038 | +0.0090 | -2.7 | -1.3 | -3.7 | -2.3 |

High-confidence FP means `p >= 0.9` for a negative example and
high-confidence FN means `p <= 0.1` for a positive example. Values above are
means across seeds 42, 43 and 44.

### Per-seed ranking and F1 deltas

| Seed | F1 delta | ROC-AUC delta | PR-AUC delta |
|---:|---:|---:|---:|
| 42 | +0.00305 | +0.00632 | +0.01689 |
| 43 | +0.00330 | +0.00277 | +0.00534 |
| 44 | -0.00103 | +0.00243 | +0.00488 |

PR-AUC and ROC-AUC improved in all three seeds. Fixed-0.5 F1 improved in two
seeds and decreased slightly in seed 44, so the F1 improvement is not
directionally consistent.

## Decision transitions

Across the three seeds, the correction changed 46 fixed-0.5 decisions:

- 29 parent errors became correct: 23 FP-to-TN and 6 FN-to-TP.
- 17 parent-correct decisions became wrong: 15 TN-to-FP and 2 TP-to-FN.
- Net effect: 12 fewer errors across the three seed evaluations.

## Interpretation

The fixed JAAD-derived 10% cap transferred positively to PIE as a ranking
correction. It produced a consistent PR-AUC gain and fewer high-confidence
errors without materially moving the fixed-threshold classification metrics.
It did not beat the established tree benchmarks. Exp012 BIU-v1+BBox random
forest has mean F1 `0.8678`, PR-AUC `0.9115` and ROC-AUC `0.9651`; Exp012
BIU-v2+BBox XGBoost has mean F1 `0.8596`, PR-AUC `0.9156` and ROC-AUC `0.9666`
under fixed 0.5. Exp093 therefore supports the limited conclusion that a small
BIU residual can refine this PedCMT parent's probabilities; it does not
establish a new best PIE predictor.

This is exploratory evidence rather than a pristine model-selection result.
The scale came from JAAD, but the parent and BIU-union choice were already known
from earlier PIE experiments. No Exp093 decision was changed after its PIE
validation or set03 results were observed.

## Provenance

- Parent checkpoints and data: `experiments/084_pedcmt_biu_transfer/`
- Correction architecture/training lineage:
  `experiments/090_pedcmt_direct_biu_fusion/`
- Parent authentication and train/validation audit:
  `metrics/source_and_parent_audit.json`
- Test predictions: `predictions/test_R10_seed{42,43,44}.csv`
- Exact per-seed metrics: `metrics/per_seed_results.csv`
- Mean and standard deviation: `metrics/summary_results.csv`
- Paired deltas: `metrics/paired_deltas.csv`
- Fixed-0.5 transitions: `metrics/error_transitions_fixed_0p5.csv`

## Commands

```bash
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python \
  experiments/093_pedcmt_pie_capped_biu_correction/scripts/run_experiment.py prepare --device cpu
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python \
  experiments/093_pedcmt_pie_capped_biu_correction/scripts/run_experiment.py smoke --device cpu
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python \
  experiments/093_pedcmt_pie_capped_biu_correction/scripts/run_experiment.py train-all --device cpu
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python \
  experiments/093_pedcmt_pie_capped_biu_correction/scripts/run_experiment.py validate
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python \
  experiments/093_pedcmt_pie_capped_biu_correction/scripts/run_experiment.py evaluate --device cpu
```
