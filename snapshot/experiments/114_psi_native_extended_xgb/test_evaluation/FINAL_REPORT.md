# Exp114 Authorized Test Evaluation

## Status

Complete and independently verified. This is the one authorized evaluation of
the six frozen Exp114 models; no fitting, tuning, calibration, or test-selected
cohort change occurred. The pinned PSI2_Test cohort contains 20,547 windows,
49 videos, 96 tracks, 15,462 positives, and 5,085 negatives. `video_0192`
remains excluded by the previously frozen PSI2 release choice.

All 13,144 unique observed `(video, frame)` identities had numeric official
`speed(km/hr)` values. All 49 exact RGB archive members passed size and CRC
validation. S187 and R247 share bit-identical ordered rows and a bit-identical
187-column prefix. Inputs are native 15-frame observations, FIRST7/LAST8
summaries, and next-frame16 labels; target frame16 is not a feature input.

## Fixed-0.5 Results

| Arm | Accuracy | Precision | Recall | Positive F1 | Macro F1 | AP | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| S187 mean | 0.699989 | 0.798578 | 0.804165 | 0.801355 | 0.594330 | 0.824576 | 0.629648 |
| R247 mean | 0.719683 | 0.801994 | 0.833226 | 0.817304 | 0.607625 | 0.849057 | 0.666407 |
| R247 - S187 | +0.019695 | +0.003416 | +0.029060 | +0.015948 | +0.013295 | +0.024481 | +0.036759 |

Accuracy, positive F1, macro F1, AP, and ROC-AUC deltas are positive for each
of seeds 42, 43, and 44. Mean FN decreases from 3,028.0 to 2,578.7, while mean
FP increases from 3,136.3 to 3,181.0. High-confidence FN decreases from 538.0
to 275.3 and high-confidence FP decreases from 1,401.0 to 1,283.0.

The always-cross baseline has accuracy 0.752519 and positive F1 0.858785 due
to the 75.25% positive prevalence, but macro F1 0.429393, AP 0.752519, and
ROC-AUC 0.5. Therefore accuracy and positive F1 alone are misleading here.

This result supports only the paired matched-population Exp114 R247-minus-S187
comparison. It must not be presented as a causal comparison with Exp113,
whose training population differs.

## Evidence

- Per-seed metrics: `metrics/per_seed_results.csv`
- Paired deltas: `metrics/paired_deltas.csv`
- Error transitions: `metrics/error_transitions.csv`
- Cohort and trivial baseline: `metrics/cohort.json`, `metrics/always_cross.json`
- Saved probabilities: `predictions/test_{S187,R247}_seed{42,43,44}.csv`
- Protocol freeze: `freeze.json`, `DESIGN.md`, `archive_gate.json`
- Internal and independent checks: `VERIFIED.json`, `INDEPENDENT_VERIFICATION.json`

Verification command:

```bash
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python \
  experiments/114_psi_native_extended_xgb/test_evaluation/verify_results.py
```
