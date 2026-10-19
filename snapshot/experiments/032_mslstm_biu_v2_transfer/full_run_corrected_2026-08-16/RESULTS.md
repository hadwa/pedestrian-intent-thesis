# Exp032 Corrected Result

## Status and scope

The corrected controlled run is complete. It compares a fresh recovered-notebook
MSLSTM control (Variant A) with the same model plus a late 128-D BIU-v2 branch
(Variant B), using three paired seeds. It is a **diagnostic transfer study**, not the
16-frame/TTE 30--60 thesis benchmark. PIE set03 had already been visually inspected,
so its metrics are not an independent confirmation.

The run uses 30 consecutive observed raw frames and predicts `behavior.cross` one raw
frame later. Training uses the immutable balanced notebook index (109,934 rows,
54,967 per class); validation uses all 77,288 set05/set06 windows; diagnostic test uses
all 271,078 set03 windows. Checkpoints are selected only by validation F1 at fixed 0.5.

## Primary fixed-0.5 results

Means and sample standard deviations are over seeds 42, 43, and 44.

| Cohort | Variant | F1 | PR-AUC | ROC-AUC | Precision | Recall | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| Validation | A: MSLSTM control | 0.81798 +/- 0.00298 | 0.89140 +/- 0.00306 | 0.97777 +/- 0.00201 | 0.75418 +/- 0.02244 | 0.89508 +/- 0.03470 | 0.93913 +/- 0.00223 |
| Validation | B: MSLSTM + BIU-v2 | 0.80646 +/- 0.00043 | 0.89178 +/- 0.00307 | 0.97811 +/- 0.00064 | 0.74569 +/- 0.02159 | 0.87928 +/- 0.02992 | 0.93551 +/- 0.00203 |
| Set03 diagnostic | A: MSLSTM control | 0.72041 +/- 0.00319 | 0.80396 +/- 0.00970 | 0.94277 +/- 0.00448 | 0.65231 +/- 0.01570 | 0.80521 +/- 0.02358 | 0.88480 +/- 0.00378 |
| Set03 diagnostic | B: MSLSTM + BIU-v2 | 0.73169 +/- 0.00700 | 0.80420 +/- 0.00880 | 0.94794 +/- 0.00413 | 0.67308 +/- 0.02183 | 0.80305 +/- 0.03497 | 0.89148 +/- 0.00434 |

Paired B-minus-A deltas at fixed 0.5:

| Cohort | F1 delta | PR-AUC delta | ROC-AUC delta | Interpretation |
|---|---:|---:|---:|---|
| Validation | -0.01152 (`-,-,-`) | +0.00038 (`+,-,+`) | +0.00034 (`+,-,-`) | F1 is consistently worse; ranking changes are trivial/inconsistent. |
| Set03 diagnostic | +0.01128 (`+,+,+`) | +0.00024 (`+,+,-`) | +0.00516 (`+,+,-`) | F1 improves through mean precision +0.02077 with mean recall -0.00216; ranking gains are not seed-consistent. |

The secondary validation-selected-threshold result is also mixed: set03 F1 delta is
+0.00938 with signs `+,+,-`, while the best validation F1 remains lower for Variant B
in all three seeds.

## Controlled verdict

Exp032 does **not** demonstrate strong or ranking/complementarity transfer. Its honest
classification is mixed/inconclusive: BIU-v2 consistently hurts the validation
fixed-0.5 operating point, while the already-inspected set03 diagnostic shows a small,
consistent fixed-0.5 F1 gain but inconsistent PR-AUC and ROC-AUC changes. The set03
pattern cannot supersede the contradictory validation evidence or establish an
independent improvement claim.

The BIU branch adds 40,960 parameters, a 0.2649% increase over the 15,464,976-parameter
control. Capacity growth is therefore small and does not trigger the planned generic
parameter-control branch.

## Data and parity audit

The same YOLOv8x-pose-p6 cache supplies both the raw pose stream and BIU-v2. Cache reuse
was checked against live extraction on 16 uniformly selected rows and 100 random rows
(seed 32042); both checks had maximum absolute error 0.0.

Pose keys are present for nearly every frame, but many stored vectors are detector
zero-fallbacks rather than measured keypoints. Effective-cohort zero-pose rates are:

| Cohort | Zero/absent pose frames | All-zero pose windows |
|---|---:|---:|
| Balanced train | 1,378,882 / 3,298,020 (41.81%) | 37,945 / 109,934 (34.52%) |
| Validation | 1,730,059 / 2,318,640 (74.62%) | 52,025 / 77,288 (67.31%) |
| Set03 diagnostic | 5,016,429 / 8,132,340 (61.68%) | 146,030 / 271,078 (53.87%) |

This severe and split-dependent pose sparsity is a grounded limitation of the transfer
test. It plausibly limits the useful information available to BIU-v2, but Exp032 does
not isolate it causally.

Variant A fixed-0.5 set03 F1 (0.72041) is 0.17859 below the paper reference used by the
runner, triggering the planned parity investigation. The local paper and runnable
notebook are not numerically equivalent: the paper describes one-second-ahead prediction,
oversampling plus Gaussian noise, YOLOP scene context, and the final augmented model;
the recovered notebook uses a one-raw-frame target, a saved undersampled index, no noise,
and the seven active streams used here. No historical pose-capable paper-run bundle or
paper threshold artifact exists. The paper value is therefore descriptive only; the
fresh paired A/B comparison is the valid Exp032 control.

## Artifacts

- Per-seed metrics: `metrics/per_seed_results.csv`
- Fixed-0.5 metrics: `metrics/fixed_0p5_results.csv`
- Summary: `metrics/summary_results.csv`
- Paired diagnostic test deltas: `metrics/paired_deltas.csv`
- Per-sample probabilities: `predictions/test_predictions.csv`
- Error transitions: `metrics/error_transitions_fixed_0p5.csv`
- Pose coverage: `metrics/pose_availability_by_cohort.csv`
- Cache validation: `metrics/cache_live_validation.csv`
- Training history: `logs/training_history.csv`
- Run and cache lineage: `RUN_PROVENANCE.md`, `CACHE_REUSE_PROVENANCE.md`
