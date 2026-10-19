# Exp119: v2-only capped PedCMT completion

All 15 new trainings and both final evaluations completed. Means and sample SD are over seeds 42/43/44; fixed threshold 0.5, AP reported as PR-AUC. Retrospective matched controls, not pristine confirmations. No test-based model/cap selection.

## Capped results

| Dataset | Arm | F1 mean +/- SD | AP mean +/- SD | ROC-AUC mean +/- SD | Accuracy | Mean FP | Mean FN |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pie | baseline | 0.7926 +/- 0.0094 | 0.8584 +/- 0.0306 | 0.9192 +/- 0.0127 | 0.8887 | 212.0 | 206.3 |
| pie | raw | 0.7917 +/- 0.0101 | 0.8580 +/- 0.0350 | 0.9204 +/- 0.0110 | 0.8884 | 211.7 | 208.0 |
| pie | core56 | 0.7946 +/- 0.0090 | 0.8606 +/- 0.0332 | 0.9217 +/- 0.0120 | 0.8898 | 210.0 | 204.3 |
| pie | full63 | 0.7955 +/- 0.0108 | 0.8665 +/- 0.0251 | 0.9228 +/- 0.0112 | 0.8905 | 206.7 | 205.0 |
| jaad | baseline | 0.6734 +/- 0.0566 | 0.6256 +/- 0.0207 | 0.6891 +/- 0.0246 | 0.6329 | 406.3 | 144.7 |
| jaad | raw | 0.6830 +/- 0.0474 | 0.6022 +/- 0.0027 | 0.6856 +/- 0.0215 | 0.6394 | 410.0 | 131.3 |
| jaad | core56 | 0.6913 +/- 0.0446 | 0.6574 +/- 0.0344 | 0.7132 +/- 0.0325 | 0.6480 | 405.3 | 123.0 |
| jaad | full63 | 0.6981 +/- 0.0360 | 0.6700 +/- 0.0368 | 0.7259 +/- 0.0365 | 0.6542 | 403.3 | 115.7 |

## Paired deltas and seed direction

| Dataset | Comparison | Mean F1 delta | Mean AP delta | Mean ROC delta | Favorable seeds F1/AP/ROC |
| --- | --- | --- | --- | --- | --- |
| jaad | core56 minus baseline | +0.0179 | +0.0318 | +0.0241 | 2/3 / 3/3 / 3/3 |
| jaad | full63 minus baseline | +0.0247 | +0.0444 | +0.0369 | 2/3 / 3/3 / 3/3 |
| jaad | full63 minus core56 | +0.0068 | +0.0126 | +0.0128 | 2/3 / 3/3 / 3/3 |
| jaad | core56 minus raw | +0.0083 | +0.0552 | +0.0275 | 2/3 / 3/3 / 3/3 |
| jaad | full63 minus raw | +0.0151 | +0.0678 | +0.0403 | 2/3 / 3/3 / 3/3 |
| pie | core56 minus baseline | +0.0020 | +0.0022 | +0.0025 | 3/3 / 2/3 / 3/3 |
| pie | full63 minus baseline | +0.0029 | +0.0081 | +0.0036 | 3/3 / 3/3 / 3/3 |
| pie | full63 minus core56 | +0.0010 | +0.0059 | +0.0011 | 2/3 / 3/3 / 3/3 |
| pie | core56 minus raw | +0.0029 | +0.0027 | +0.0013 | 3/3 / 2/3 / 3/3 |
| pie | full63 minus raw | +0.0038 | +0.0085 | +0.0024 | 3/3 / 3/3 / 3/3 |

## Interpretation

- On PIE, both v2 forms improve mean capped F1 over the parent and matched raw pose. Full63 improves F1/AP/ROC over both in every seed. Core56 ranking improvements are not uniformly positive for AP.
- On clean JAAD, both v2 forms improve mean F1/AP/ROC over the parent and capped raw pose. F1 improves in two seeds and ties in the third; both ranking metrics improve in every seed. Full63 exceeds core56 in mean metrics, with F1 tied in one seed.
- These are small PIE refinements and model-specific JAAD improvements, not a claim of universal v2 superiority or statistical significance. No clustered confidence intervals were computed in this completion.
- The predeclared uncapped probabilities are a diagnostic from the same trained branches; they do not replace the capped primary comparison or the earlier direct-token family.
- JAAD raw predictions are reused exactly from Exp094. Parents are authenticated Exp084/089 checkpoints. Training uses original cached parent outputs. GPU replay differs slightly in floating point; this can alter tied probability rankings at tiny scales, so parent AP/ROC may differ slightly from historical CSVs while fixed-threshold decisions agree.
- This fills the capped v2-only gap without relying on union135. The separate clean-JAAD MSLSTM family (Exp120) is also complete; its negative BIU findings are retained in its consolidated report.

## Measured predictor cost

| Arm | Total parameters | Trainable correction parameters | PIE mean ms | Clean-JAAD mean ms |
| --- | --- | --- | --- | --- |
| Baseline | 1,834,416 | 0 | 3.246 | 3.212 |
| Capped raw pose | 2,434,993 | 600,577 | 4.459 | 4.451 |
| Capped core56 | 1,924,529 | 90,113 | 3.424 | 3.441 |
| Capped full63 | 1,925,425 | 91,009 | 3.430 | 3.434 |

Measurements include the frozen parent forward pass and the correction branch, using prepared inputs, batch size one and one CPU thread. They exclude feature construction and upstream perception, so they are not camera-to-prediction latency. Each row pools 1,800 timed calls across three seeds. Median/p95, replay verification and exact scope are in `metrics/efficiency.csv`, `metrics/latency_replay.json` and `metrics/latency_scope.json`. Command: `env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/119_pedcmt_v2_capped_completion/scripts/latency.py`.

## Sources

`DESIGN.md`, `metrics/pretraining_sources.json`, `metrics/all_sources.json`, `metrics/per_seed_results.csv`, `metrics/summary_results.csv`, `metrics/paired_deltas.csv`, `metrics/error_transitions.csv`, per-case predictions and completion/checkpoint markers. Exact reused/fresh experiment mappings: `MISSING_RESULTS_INVENTORY.md`. All new saved probabilities recompute their reported metrics within 1e-12; see `metrics/saved_prediction_verification.json`. No thesis edits, commit or push.
