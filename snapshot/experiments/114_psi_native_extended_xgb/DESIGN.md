# Frozen PSI-native extended XGBoost design

Status: blocked at the source preflight before data recovery, perception
extraction, feature preparation, or fitting. This design remains the approved
design if an exact official speed source for every cohort window is supplied.

## Question and matched arms

Does frozen TwinLiteNet+ Large road-relative context add validation information
beyond the PSI-native adaptation of the complete Exp075 source representation?

- S187: 161-D historical deduplicated union of BBox53, BIU-v1 72 and BIU-v2
  core56, retaining v1 semantics for the 20 overlapping names; 21 observation
  temporal contrasts; and five official per-frame speed summaries.
- R247: S187 followed by the exact 60 channel-major TwinLiteNet+ road-relative
  summaries used by Exp075.

This is exactly two arms and six fresh fits (two arms times seeds 42/43/44).
Exp113 A/B/C are not valid saved baselines for this comparison because none has
the S187 composition. Exp113 rows, labels, BBox53, BIU-v1, and compatible BIU-v2
source values may be reused only after their hashes and feature names pass.

## PSI protocol

Reuse Exp113's official PSI2 train/validation split, extension, database
behavior, labels, and complete cohorts: 26,880 training windows and 14,922
validation windows. Each sample observes 15 consecutive frames and predicts the
next raw frame. Every early/late calculation uses FIRST7/LAST8; target frame16
is never included or used as padding. Preserve zero pose for absent cached
frames and never drop windows for pose or speed availability.

Test annotations, test features, test predictions, and test metrics are closed.
There is no automatic test phase.

## Features

The static 161-D union and names come from the actual frozen Exp037/072 feature
contract, not from BIU-v2 alone. The seven 3-stat temporal bases are unchanged:
lower-body motion magnitude mean, lead-ankle forward delta, torso motion
magnitude mean, torso-center x delta, normalized BBox-center x displacement,
BBox log-area change, and mean pose confidence. For each base, calculate early,
late, and late-minus-early on frames 1--7 and 8--15 respectively.

The five ego features are mean, population standard deviation, first, last, and
last-minus-first over exactly the 15 observed official `speed(km/hr)` values.
No acceleration stream, driver-decision category, imputation, cross-release
substitution, or invented zero is permitted.

Road context uses the SHA-pinned Exp075 official TwinLiteNet+ Large checkpoint
and adapter, original-coordinate masks, and exact Exp063 20-D road-relative
equations. Each road channel is summarized as FIRST7 mean, LAST8 mean, and
LAST8-minus-FIRST7, yielding 60 columns. Required RGB frames are deduplicated by
exact `(video_id, frame_index)` before inference. No full-video frame dump is
permitted.

## Predictor and reporting

Use the Exp113 recipe without search: median imputation fit on training only;
XGBoost 250 trees, depth4, learning rate .05, subsample .8, colsample .8,
L2=1, binary logistic/log-loss, four CPU threads, training-derived class weight,
and seeds42/43/44. Fit each arm/seed once. Report fixed-.5 accuracy, precision,
recall, binary F1, macro F1, AP, ROC-AUC, FP, FN, and paired R247-minus-S187
deltas. Validation is descriptive and cannot authorize automatic test access.

## Mandatory preflight

Before any video download or feature computation, require an exact official
numeric speed value for every observed frame of every Exp113 train/validation
window. Fail closed on an absent video, frame, malformed value, or ambiguous
frame-index mapping. Freeze source/code hashes before fitting.

The 2026-09-11 preflight fails: PSI2 provides no CV annotation member for
`video_0032`, `video_0055`, or `video_0093`, affecting 585 training windows.
PSI1 provides CV members for the first two only, not `video_0093`, and is not an
automatically interchangeable PSI2 annotation source. No expensive assets may
be downloaded and no fit may run while this gate fails.
