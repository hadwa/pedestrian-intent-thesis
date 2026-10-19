# Exp093 Design

## Question

Does the ten-percent signed BIU correction that succeeded on standardized JAAD
transfer unchanged to the standardized PIE 16-frame/TTE30--60 PedCMT setup?

## Frozen Design

- Cohorts, BBox, ego speed and BIU union: exact Exp084 artifacts.
- Parent: exact per-seed Exp084 A BBox+ego checkpoints.
- BIU: train-standardized 135-D BIU-v1/v2 union.
- Correction architecture: exact Exp090 full-256-D-anchor signed direct branch.
- Branch training: exact Exp084 train-only positive oversampling and Exp090
  optimizer, BCE, batch size, patience and minimum-validation-BCE selection.
- Seeds: 42, 43 and 44.
- Final prediction: `parent_logit + 0.10 * learned_correction`.
- Classification threshold: fixed 0.5.

The scale is transferred from JAAD and is not selected or modified using PIE
validation or set03. Only the small correction branch is trained. Exp084 parent
and all pose/BIU artifacts remain frozen.

Validation is completed and recorded before one unchanged set03 evaluation.
No result-dependent gate or percentage search is performed because this is an
explicit exploratory cross-dataset transfer test.
