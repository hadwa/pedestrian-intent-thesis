# Exp121: validation-only checkpoint-selection comparison

Retrospective development evidence only. No test source was accessed or evaluated. Same twelve training trajectories, inputs and seeds; only checkpoint selection differs. A=baseline, B=raw pose, C=BIU56, D=BIU56+seven auxiliaries63.

## Validation means (three seeds, fixed threshold 0.5)

| Arm | Selection | F1 | AP | ROC-AUC | Accuracy | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | ap | 0.8148 | 0.8293 | 0.7559 | 0.7506 | 48.7 | 19.7 |
| A | f1 | 0.8234 | 0.8123 | 0.7417 | 0.7591 | 50.3 | 15.7 |
| B | ap | 0.7818 | 0.8279 | 0.7551 | 0.7214 | 44.0 | 32.3 |
| B | f1 | 0.8163 | 0.8178 | 0.7436 | 0.7433 | 56.3 | 14.0 |
| C | ap | 0.7804 | 0.8668 | 0.7862 | 0.7141 | 47.7 | 30.7 |
| C | f1 | 0.8064 | 0.8089 | 0.7091 | 0.7165 | 68.3 | 9.3 |
| D | ap | 0.7644 | 0.8537 | 0.7796 | 0.7080 | 40.7 | 39.3 |
| D | f1 | 0.8095 | 0.7974 | 0.7069 | 0.7178 | 71.0 | 6.3 |

## Paired interpretation

AP-selected validation AP cannot be lower than F1-selected validation AP on the same trajectory; the reverse holds for F1. This is a consequence of selecting on those metrics, NOT independent evidence that the selection change improves generalization. Other metrics and failure modes are descriptive. No threshold was tuned.

| AP-selected comparison | Mean F1 delta | Mean AP delta | Mean ROC delta | Positive seeds F1/AP/ROC |
| --- | --- | --- | --- | --- |
| B minus A | -0.0330 | -0.0015 | -0.0008 | 0/3, 1/3, 2/3 |
| C minus A | -0.0344 | +0.0375 | +0.0303 | 0/3, 3/3, 3/3 |
| D minus A | -0.0504 | +0.0244 | +0.0237 | 0/3, 3/3, 2/3 |
| C minus B | -0.0014 | +0.0389 | +0.0311 | 2/3, 3/3, 3/3 |
| D minus B | -0.0174 | +0.0259 | +0.0245 | 2/3, 3/3, 2/3 |
| D minus C | -0.0160 | -0.0131 | -0.0066 | 1/3, 1/3, 1/3 |

## Integrity

All 24 saved validation prediction files reproduce metrics within 0. Twelve replay histories have maximum absolute difference 1.11e-16; F1-selected probabilities differ from Exp120 by at most 2.98e-08.

## Augmentation measurements

All duplicated rows retain the inherited Gaussian augmentation; nothing was changed based on this diagnostic. Ratios below compare absolute perturbation with original-training feature standard deviation; they are not calibrated uncertainty or proof that measurements are physically invalid.

| Feature | Mean across seeds: median change / train SD | Mean across seeds: p95 change / train SD | Mean fraction changed |
| --- | --- | --- | --- |
| direction_sign | 0.000 | 2.001 | 0.108 |
| was_flipped | 0.000 | 2.001 | 0.108 |
| body_yaw_angle_delta | 0.811 | 2.418 | 1.000 |
| stride_length_std | 0.117 | 0.475 | 1.000 |
| hip_center_x_velocity_std | 12.812 | 18.664 | 1.000 |
| lower_body_motion_magnitude_mean | 3.972 | 5.556 | 1.000 |
| torso_motion_magnitude_mean | 4.170 | 5.683 | 1.000 |

## Scope and reproduction

No new test claims, source changes, model redesign, thesis edits or commits. The original Exp120 negative result remains intact. This completes the authorized validation comparison; final test evaluation or a new intervention requires a separate decision.

Training: `env LD_LIBRARY_PATH=/usr/lib/wsl/lib PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python -u experiments/121_jaad_mslstm_ap_selection/scripts/run.py`.

Verification/report: `env PYTHONDONTWRITEBYTECODE=1 /home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/121_jaad_mslstm_ap_selection/scripts/report.py`.

Exact sources: `metrics/source_hashes.json`. Checkpoints: `checkpoints/`. Predictions: `predictions/validation_*.csv`. Epoch histories and replay checks: `logs/`. Protocol: `DESIGN.md`.
