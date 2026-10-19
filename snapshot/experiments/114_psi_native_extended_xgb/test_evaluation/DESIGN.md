# Frozen authorized Exp114 test evaluation

Evaluate the six saved Exp114 S187/R247 models (seeds42/43/44) on the pinned
PSI2_Test release cohort only. Use the test release's 49-video split, excluding
`video_0192` exactly as frozen by Exp113 test evaluation. Reuse its 20,547
native windows, labels, BBoxes, pose cache assignments, and A/B/C source arrays.
No refit, tuning, calibration, threshold selection, population filtering, or
test-driven feature change is permitted.

Inputs remain 15 observed consecutive frames predicting the next raw frame16.
Every temporal and road summary uses FIRST7/LAST8. S187 is the identical
historical deduplicated 161-D BBox/BIU union plus 21 temporal and five exact
official-speed features. R247 is the exact S187 prefix plus the same 60
TwinLiteNet+ road features used in amended development. Target frame16 is never
an input. Missing pose behavior stays inherited from the existing Exp113 test
cache; no pose extraction or pose-dependent sample filtering is allowed.

Before RGB recovery, require a numeric official `speed(km/hr)` value for every
observed frame in all 20,547 windows. Fail closed on any absent member/frame,
malformed unit/value, ambiguous identity, or cohort mismatch. Recover only the
49 necessary PSI2 test videos. Deduplicate exact `(video_id, frame_index)`
identities and use the SHA-pinned Exp075 TwinLiteNet+ Large model with batch1
CUDA FP16, its original-coordinate mask ABI, exact Exp063 road equations, and
60-D channel-major summaries.

Freeze the six model hashes, code/design/source contracts, and validate all six
saved development predictions before building test features. Both arms must
share identical ordered rows and a bit-identical 187-D prefix. Evaluate every
saved model at fixed threshold .5. Report accuracy, precision, recall, positive
F1, macro F1, AP, ROC-AUC, FP/FN, high-confidence FP/FN, paired deltas, error
transitions, cohort/pose coverage, and always-cross imbalance baseline.

Interpret only the matched Exp114 R247-minus-S187 comparison. Exp114 used a
different training population from Exp113, so cross-experiment differences are
not causal representation comparisons. Stop after this evaluation; do not
launch another model or architecture.

## Execution Note

The first continuation completed all 49 atomic per-video road feature parts,
then stopped before R247 assembly or model prediction because of a
tuple-parenthesization error in the final concatenation. The concatenation was
split into explicit 60- and 247-column matrices, and the immutable freeze was
refreshed before any test prediction. Cached road parts are reused unchanged.
The first scoring attempt also stopped before prediction because byte-identical
source records containing `../` were compared against canonical path strings;
the freeze recorder now resolves paths before recording them.
