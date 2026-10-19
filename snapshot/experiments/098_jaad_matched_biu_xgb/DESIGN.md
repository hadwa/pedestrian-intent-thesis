# Exp098: Matched JAAD BIU-version XGBoost comparison

Approved scope: no vehicle action, physical ego speed, visual ego proxy, road
context, extra temporal contrasts, raw pose, or new confidence features.
Vehicle action is reserved for a separately approved Exp077 follow-up.

## Frozen protocol

- Clean behavior-pedestrian JAAD, official default video split.
- 16 consecutive observed frames; event/negative-endpoint TTE 30--60 frames.
- Train 1,552 windows, validation 274, test 1,501. No population exclusions.
- Train only on training rows; seeds 42, 43, 44; threshold fixed at 0.5.
- All four arms complete validation before any test feature source is opened.
  All advance together, irrespective of validation scores. No model selection,
  threshold selection, early stopping, tuning, or train+validation refit.
- This is a retrospective matched ablation on a previously exposed test cohort,
  not a new untouched confirmation or a basis for post-test model redesign.

## Frozen arms

| Arm | Input | Dimension |
|---|---|---:|
| A | BBox-53 | 53 |
| B | BBox-53 + frozen BIU-v1 | 125 |
| C | BBox-53 + frozen BIU-v2 | 116 |
| D | BBox-53 + source-namespaced v1 and v2 concatenation | 188 |

V1 is the full 72-D Exp089 representation, including its existing two validity
counters. V2 is the full 63-D Exp089 representation, including direction_sign,
was_flipped, direction_confidence and geometric scale summaries. Direction
confidence is an existing direction-estimation statistic, not added keypoint
confidence. The existing BBox direction fallback remains unchanged. V2 is not
Exp009's 56-column core. D retains both complete vectors without deduplication:
matching names need not imply identical equations or measurements.

## Sources and estimator

Reuse unstandardized arrays in Exp089 cache/{trainval_source,test_source} and
their row manifests. Reuse Exp077 metrics/prepared_jaad_{train,validation,test}.csv
for BBox-53 only. Join by exact sequence ID and verify labels, frame bounds,
video/track identities and exact row coverage. Compare BBox features to the
frozen Exp076 extractor on observation boxes. No source cache is modified.

The estimator is Exp072 build_candidate, the same recipe as Exp077's one-stage
builder: 250 trees, depth 4, learning rate 0.05, subsample 0.8, colsample_bytree
0.8, L2 1, binary logistic objective, logloss, four CPU threads, training-only
negative/positive scale_pos_weight and median imputation. No scaling, no data
augmentation or oversampling. CPU execution preserves the existing recipe;
these small tabular fits do not require a new GPU implementation.

No completed equivalent for all four arms was found: Exp009 used a different
v2 dimension/population; Exp077 combined features and temporal contrasts;
Exp089 used PedCMT; Exp096 is PIE with different feature arms. Twelve new fits
are authorized. Cached features, not historical fitted predictors, are reused.

## Analyses frozen before test

Report each seed and mean/sample SD for accuracy, precision, recall, F1,
average precision (labelled PR-AUC consistently with the existing experiments),
ROC-AUC, confusion counts, FPR/FNR, Brier score and high-confidence FP >=0.9 /
FN <=0.1. Save all probabilities and exact frame identities.

Primary comparison: C minus B. Also report B/A, C/A, D/A, D/B and D/C paired
deltas. A favorable sign on all three seeds is directional consistency, not a
significance test. Perform 5,000 paired pedestrian-track bootstrap replicates
(seed 42) on seed-ensemble probabilities; report 95% percentile intervals for
F1, PR-AUC and ROC-AUC. These are descriptive, without multiplicity correction.
Report fixed-0.5 B-to-C error transitions. Do not collapse conflicting metrics
into a universal version winner; distinguish ranking from decision performance.

Pin the design, runner, imported code and development artifacts before fitting.
Authenticate all twelve model/prediction files before opening test. Authenticate
test sources at final evaluation; reuse saved models, never refit. Create
EXPERIMENT_COMPLETE.json only after result, prediction and analysis verification.
