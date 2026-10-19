# Exp118: Ordered History XGBoost

Frozen before fitting. Validation-only exploratory comparison on PIE, 16 observed
frames, TTE 30--60. No set03 data, predictions, metrics or models selected on test
are accessed. Prior validation reuse means this is development evidence.

## Matched Arms

- A: exact Exp096 v2 baseline: 135 columns, including BBox53, BIU-v2-core56
  and the existing temporal/ego blocks. Reuse saved seeds 42/43/44, subject to
  exact validation probability replay and estimator-parameter parity.
- B: A plus 336 ordered values (16 frames x 21 channels) from the authenticated
  Exp040 sequence cache. Include its first 16 behavioral channels and five BBox
  channels. Exclude all three confidence channels and both sequential OBD channels;
  existing ego inputs remain unchanged. Flatten time-major, frame1 through frame16.
  Total 471 columns. No new BIU version; the window-level BIU56 remains unchanged.

These are ordered engineered measurements, not every raw pose coordinate. They
include anatomical angles, stride/foot/arm measurements, body-center and motion
measurements, and relative BBox geometry/velocity. The source extractor is
Exp040/scripts/sequence_core.py. Observation-window canonicalization/normalization
uses only observed frames, not future frames. This tests incremental ordered
measurements, not a pure causal test of ordering versus an information-identical
unordered control. Do not claim failure rules out all useful temporal detail.

Train sets01/02/06: 2366 rows; validation sets04/05: 3619 rows. Exact Exp072/096
rows, labels, track/video IDs, order and frame windows. No sample dropping.
Median imputation is fit on training only. Reuse Exp072 model factory unchanged:
250 trees, depth4, learning rate .05, subsample .8, colsample .8, L2=1,
training-only negative/positive class weight, four CPU threads. CPU preserves the
parent recipe and avoids GPU setup overhead for three small fits. No search,
early stopping, calibration, threshold tuning or additional architecture.

## Checks and Reporting

Authenticate Exp040 cache against its preparation marker and Exp072 train/val
against Exp096 pinned hashes. Exact sequence ID joins, labels/track/video parity,
disjoint train/validation tracks, frame length and TTE checks must pass. Verify
flatten/unflatten identity and deterministic extraction from cache. Replay A
against saved validation probabilities within 1e-7 and identical decisions.

Report fixed-.5 metrics, AP (labelled PR-AUC consistently with parent), ROC-AUC,
FP/FN and high-confidence errors (FP p>=.9; FN p<=.1), three-seed means/stds,
paired deltas and fit time. Promising is predeclared as positive mean F1 and AP
deltas, neither metric declining more than .005 in any seed. Report sign
consistency separately. No automatic test authorization follows this criterion.

## Prior Work Audit

Exp072 already used half-window contrasts. Exp076/077 used an early prediction
followed by a full-window XGBoost, not all frame values. Exp040 used this ordered
cache in a TCN. Repository README/DESIGN and script searches for flattened,
per-frame and ordered representations found no matching summaries-plus-ordered
XGBoost experiment. This is not a rerun of those experiments.
