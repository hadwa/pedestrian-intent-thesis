# Exp053 Frozen Design

## Question

Does replacing MSLSTM's weak raw `16x4` BBox LSTM branch with the exact Exp012 53-D engineered BBox representation improve practical-core, BIU-v1 and BIU-v2 predictors?

## Controlled Pairs

| Condition | Raw control | BBox53 replacement |
|---|---|---|
| Practical core | Exp051 C: raw BBox + ego + YOLOP | G: BBox53 + ego + YOLOP |
| BIU-v1 | Exp051 F: raw BBox + ego + YOLOP + BIU-v1 | H: BBox53 + ego + YOLOP + BIU-v1 |
| BIU-v2 | Exp051 E: raw BBox + ego + YOLOP + BIU-v2 | I: BBox53 + ego + YOLOP + BIU-v2 |

The three raw controls are reused only because task, cohorts, caches, normalized fusion, non-BBox streams, projectors, training, seeds, checkpoint selection and evaluation are identical. G/H/I are fresh runs.

## BBox53 Pathway

The exact archived Exp012 feature order and equations are frozen. Original rows reuse archived values byte-for-byte. Gaussian-augmented training BBoxes are passed through the same equations. Median imputation and mean/std scaling are fit on original training rows only.

`BBox53 -> Linear(53,128) -> ReLU -> Dropout(0.3) -> Linear(128,512) -> non-affine LayerNorm -> unchanged weighted fusion`.

BIU-v1 remains 72-D and BIU-v2 remains 63-D, each with its already frozen `input->128->512` projector. No BIU definition changes.

## Frozen Protocol

- PIE 16 consecutive observed frames, TTE30--60, overlap 0.6.
- Train/validation/test: 2,366 / 3,619 / 3,760; test is set03.
- Seeds 42/43/44, 15 epochs, Adam `1e-4`, batch size 32.
- Same Gaussian minority augmentation, active non-BBox streams, classifier, normalized fusion and validation-F1 checkpoint rule as Exp051.
- Fixed threshold 0.5 plus ROC-AUC and PR-AUC.
- No set03 access until all nine new checkpoints authenticate.

## Success Rule

BBox53 succeeds for a condition if mean PR-AUC or mean fixed-0.5 F1 improves over its paired raw control while the other declines by no more than `0.005`. Same favorable delta sign on all three seeds is reported separately.

## Exposure

This is a controlled post-hoc experiment on previously exposed set03. Design and success rules are frozen before evaluation; no changes follow test results.
