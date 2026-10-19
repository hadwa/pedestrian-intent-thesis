# Exp072 Frozen Design

The authoritative design is
`docs/superpowers/specs/2026-08-26-pie-joint-biu-temporal-ego-exp072-design.md`.

## Candidate

One fixed XGBoost classifier consumes exactly 187 columns:

- 161-column deduplicated union of BBox-53, BIU-v1-72, and BIU-v2-core-56;
- 21 fixed early/late observation-half temporal contrasts;
- 5 exact observed-frame OBD speed summaries.

Seeds are 42, 43, and 44. The primary threshold is fixed at 0.5. Each model is
fit on sets01/02/06 only, saved and hashed before validation, then reused
without refitting if set03 evaluation is authorized.

## Validation Gate

Relative to corresponding saved Exp012 BIU-v1+BBox RF predictions:

- mean PR-AUC delta must be at least +0.005;
- mean fixed-0.5 F1 delta must be at least +0.005;
- mean ROC-AUC delta must be at least -0.002;
- no seed PR-AUC delta may be below -0.005;
- no seed F1 delta may be below -0.005.

Set03 is inaccessible before all checks pass. High-confidence error tails and
calibration metrics are diagnostic rather than promotion gates.

## Final Success

The three-seed mean must simultaneously exceed F1 0.8685, PR-AUC 0.9119, and
ROC-AUC 0.9652. Set03 has prior exposure elsewhere, so the result is not called
pristine untouched confirmation. No Exp072 choice may change after set03 is
opened.
