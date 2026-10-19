# Exp122: no measurement noise, validation-only results

Same clean-JAAD population and balanced training manifests as Exp121. Only measurement-noise augmentation is removed. A=baseline, B=raw pose, C=canonical BIU56, D=BIU56+seven auxiliaries63. Means over42/43/44; threshold0.5. These are retrospective development/selection-set results, not test estimates.

## Primary AP-selected comparison

| Arm | Training | F1 | AP | ROC-AUC | Accuracy | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | Exp121 noise | 0.8148 | 0.8293 | 0.7559 | 0.7506 | 48.7 | 19.7 |
| A | Exp122 no noise | 0.6741 | 0.8216 | 0.7653 | 0.6545 | 30.7 | 64.0 |
| B | Exp121 noise | 0.7818 | 0.8279 | 0.7551 | 0.7214 | 44.0 | 32.3 |
| B | Exp122 no noise | 0.7389 | 0.8255 | 0.7689 | 0.6934 | 34.7 | 49.3 |
| C | Exp121 noise | 0.7804 | 0.8668 | 0.7862 | 0.7141 | 47.7 | 30.7 |
| C | Exp122 no noise | 0.7676 | 0.8651 | 0.7991 | 0.7324 | 25.0 | 48.3 |
| D | Exp121 noise | 0.7644 | 0.8537 | 0.7796 | 0.7080 | 40.7 | 39.3 |
| D | Exp122 no noise | 0.7813 | 0.8439 | 0.7917 | 0.7360 | 33.3 | 39.0 |

## Paired seed directions for noise removal

| Arm | Mean F1 delta | Mean AP delta | Mean ROC delta | Positive seeds F1/AP/ROC |
| --- | --- | --- | --- | --- |
| A | -0.1408 | -0.0078 | +0.0094 | 0/3, 0/3, 2/3 |
| B | -0.0429 | -0.0023 | +0.0138 | 0/3, 1/3, 3/3 |
| C | -0.0128 | -0.0017 | +0.0129 | 1/3, 1/3, 2/3 |
| D | +0.0168 | -0.0098 | +0.0121 | 1/3, 0/3, 2/3 |

## Within-run representation contrasts

| Comparison (AP selected) | F1 delta | AP delta | ROC delta | Positive seeds F1/AP/ROC |
| --- | --- | --- | --- | --- |
| B minus A | +0.0648 | +0.0040 | +0.0036 | 1/3, 2/3, 3/3 |
| C minus A | +0.0935 | +0.0436 | +0.0338 | 2/3, 3/3, 3/3 |
| D minus A | +0.1072 | +0.0224 | +0.0264 | 1/3, 3/3, 3/3 |
| C minus B | +0.0287 | +0.0396 | +0.0302 | 1/3, 3/3, 3/3 |
| D minus B | +0.0424 | +0.0184 | +0.0228 | 2/3, 3/3, 3/3 |
| D minus C | +0.0137 | -0.0212 | -0.0074 | 2/3, 0/3, 0/3 |

## Secondary F1-selected diagnostic (not primary model selection)

| Arm | F1 | AP | ROC-AUC |
| --- | --- | --- | --- |
| A | 0.8050 | 0.8169 | 0.7594 |
| B | 0.7957 | 0.8224 | 0.7634 |
| C | 0.7954 | 0.8571 | 0.7869 |
| D | 0.8094 | 0.8392 | 0.7847 |

## Verification and boundaries

All24 saved prediction files reproduce metrics within 0; validation IDs/labels match Exp121 exactly. Balanced row manifests match exactly. No test data was accessed, no threshold changed, no feature or architecture redesigned. Duplicate observations remain dependent observations, not new data.

Noise removal comparisons include the resulting optimization/checkpoint changes. They do not isolate an individual cue or prove that all augmentation is harmful. AP-selected validation scores are selected on this same validation set; seed consistency is descriptive, not statistical significance. No automatic test evaluation or further experiment is launched.

## Exact commands and sources

Training: `env LD_LIBRARY_PATH=/usr/lib/wsl/lib PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python -u experiments/122_jaad_mslstm_no_noise/scripts/run.py`.

Report: `env PYTHONDONTWRITEBYTECODE=1 /home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/122_jaad_mslstm_no_noise/scripts/report.py`.

Original-development caches and source code: `metrics/source_hashes.json`. Old comparator: `../121_jaad_mslstm_ap_selection/metrics/per_seed_validation.csv` and its `predictions/validation_*.csv`. New results: `metrics/per_seed_validation.csv`, `metrics/summary_validation.csv`, `metrics/paired_noise_removal_deltas.csv`, `predictions/validation_*.csv`. No thesis edits, commits or pushes.
