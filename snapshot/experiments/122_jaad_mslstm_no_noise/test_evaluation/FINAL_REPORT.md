# Exp122: approved clean-JAAD test results

All four arms, seeds42/43/44, existing validation-AP-selected checkpoints; fixed threshold0.5. No retraining, threshold search or test-based selection.1501 test windows (720 crossing,781 noncrossing),16 observed frames,TTE30--60. Retrospective controlled follow-up on an exposed cohort, not pristine confirmation.

| Arm | F1 mean +/- SD | AP mean +/- SD | ROC-AUC mean +/- SD | Accuracy | Precision | Recall | Mean FP | Mean FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline | 0.7224 +/- 0.0302 | 0.7957 +/- 0.0011 | 0.8114 +/- 0.0022 | 0.7131 | 0.6801 | 0.7870 | 277.3 | 153.3 |
| Raw pose | 0.7460 +/- 0.0105 | 0.7879 +/- 0.0045 | 0.8090 +/- 0.0021 | 0.7240 | 0.6693 | 0.8444 | 302.3 | 112.0 |
| BIU56 | 0.7257 +/- 0.0055 | 0.8000 +/- 0.0036 | 0.8130 +/- 0.0046 | 0.7124 | 0.6702 | 0.7926 | 282.3 | 149.3 |
| BIU56 +7 auxiliaries63 | 0.7505 +/- 0.0084 | 0.7904 +/- 0.0058 | 0.8154 +/- 0.0020 | 0.7268 | 0.6682 | 0.8574 | 307.3 | 102.7 |

## Paired comparisons

| Comparison | F1 delta | AP delta | ROC delta | Positive seeds F1/AP/ROC |
| --- | --- | --- | --- | --- |
| B minus A | +0.0235 | -0.0078 | -0.0024 | 2/3, 0/3, 1/3 |
| C minus A | +0.0033 | +0.0042 | +0.0015 | 1/3, 3/3, 2/3 |
| D minus A | +0.0281 | -0.0053 | +0.0040 | 2/3, 1/3, 3/3 |
| C minus B | -0.0202 | +0.0120 | +0.0040 | 0/3, 3/3, 2/3 |
| D minus B | +0.0045 | +0.0024 | +0.0064 | 2/3, 1/3, 3/3 |
| D minus C | +0.0248 | -0.0096 | +0.0025 | 3/3, 0/3, 2/3 |

## Scope and verification

A=BBox53+categorical vehicle state+YOLOP; B=A+raw pose; C=A+BIU56; D=A+BIU56+7aux. No acceleration. Training-only scaler reused. AP means average precision, not trapezoidal PR area. Sample SD across three seeds is not a confidence interval. No significance claim.

Frozen training/source hashes, all checkpoint selection epochs, disjoint development/test videos/IDs, fixed cohort counts, saved-probability metrics and post-evaluation source/checkpoint hashes passed. Exact input source paths/hashes are in `metrics/source_hashes.json`; checkpoint paths/hashes in `FROZEN_EVALUATION.json`. All predictions are in `predictions/`. Original Exp120 and Exp122 validation results remain untouched.

Command: `env LD_LIBRARY_PATH=/usr/lib/wsl/lib PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python -u experiments/122_jaad_mslstm_no_noise/test_evaluation/evaluate.py`.

No model changes, thesis edits, commits, pushes or further experiment followed evaluation.
