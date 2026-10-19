# Exp091 Design

## Scope

Validation-only causal follow-up to Exp090. The final 1,501-window JAAD test
cohort is sealed and must not be loaded by this experiment.

## Frozen Inputs

- Parent: exact SHA-authenticated Exp089 A checkpoints reused by Exp090.
- Representation: Exp089 standardized 135-D BIU-v1/v2 union.
- Architecture: Exp090 C direct signed residual, unchanged.
- Optimizer, batch size, epoch limit, patience, initialization and minimum
  validation-BCE checkpoint selection: unchanged from Exp090.
- Seeds: 42, 43 and 44.
- Primary threshold: fixed 0.5.

## Sole Training Change

Exp090 duplicated the 553 NOT-CROSS training windows to match the 999 CROSS
windows. Exp091 uses every one of the natural 1,552 training windows exactly
once per epoch. No class weighting, resampling or focal loss is added.

## Calibration Diagnostic

For each seed, a single additive logit intercept is fitted with five-fold
group cross-fitting by validation track. The slope remains fixed at one. This
is a secondary validation diagnostic, not independent generalization evidence.
No temperature scaling or flexible isotonic calibration is used.

## Predeclared Validation Gate

Against the frozen A parent, the raw natural-sampling variant must:

- improve mean PR-AUC by at least 0.005;
- keep mean fixed-0.5 F1 within 0.005;
- keep mean fixed-0.5 accuracy within 0.005;
- not increase mean high-confidence FP+FN count.

Regardless of outcome, Exp091 stops after validation analysis.
