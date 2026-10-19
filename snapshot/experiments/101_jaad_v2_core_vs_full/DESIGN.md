# Exp101: Matched v2-core-56 versus v2-full-63 pathway

This is the final approved comparison. Stop after reporting; do not tune or
launch any additional variant. Exp100 and all existing outputs remain read-only.

## Frozen arms

| Arm | Inputs | Total dimension | Training |
|---|---|---:|---|
| core56 | Exact Exp100 v2 pathway | 145 | reuse Exp100 seeds42/43/44 |
| full63 | Same 145 columns, unchanged order, plus seven fixed values | 152 | three fresh fits |

Append only, in this order: direction_sign, was_flipped, direction_confidence,
shoulder_width_mean, shoulder_width_delta, torso_length_mean, torso_length_delta.
The first three are direction metadata; direction_confidence is the existing
displacement/shoulder-scale statistic, not confidence from a pose estimator.
The last four are existing scale/viewpoint summaries. Do not add any other
numeric extractor outputs, confidence fields, features or temporal summaries.

All BBox, BBox-temporal, v2-temporal, existing pose-quality, and annotated
categorical vehicle-action inputs remain fixed. The experiment isolates these
seven added values, not architectural or fusion changes. Vehicle action remains
an annotation-derived proxy, not measured physical ego speed.

## Exact feature provenance

Use Exp100's authenticated 145-D matrices. Recompute seven values from the
same SHA-pinned primary/supplemental pose JSONLs and observation BBoxes with the
exact Exp100 extractor call: float64 coordinates/confidences, observed BBoxes,
historical image_size=(1920,1080) default and include_optional=False. No pose
estimator is rerun. Do not import the differently configured Exp089/098 63-D
cache. Verify all 56 recomputed core values against Exp100 within rtol1e-9,
atol1e-8 before appending the seven. Preserve the direction fallback.

## Protocol and baseline parity

- Clean JAAD behavior-pedestrian, official default split; train1552/val274/
  test1501; 16 consecutive observations; TTE30--60; overlap0.8; fixed0.5.
- Seeds42/43/44, exact Exp100 250-tree XGBoost recipe: depth4, LR0.05,
  subsample0.8, colsample_bytree0.8, reg_lambda1, binary logistic/logloss,
  training negative/positive class weighting, median imputation, n_jobs4 CPU.
- Reuse baseline only with authenticated parent completion, model/input/schema
  parity, identical imputer medians/estimator parameters and exact reproduction
  of saved validation probabilities. Reuse parent test probabilities directly.
- Train only the full63 arm on training data. No refit, early stopping,
  hyperparameter/feature search, calibration, threshold selection or augmentation.
- All three new validation runs finish before any test source opens. Both arms
  advance regardless of scores. Freeze code/design/development sources first.
- Preserve all rows, ordering, labels, observation frames, video/track splits;
  fail closed on missing/conflicting input. No future frame enters features.

## Predeclared analysis

Report per-seed and mean/sample SD accuracy, precision, recall, F1, average
precision (PR-AUC), ROC-AUC, confusion counts, FPR/FNR, Brier, high-confidence
FP>=0.9 and FN<=0.1 counts; paired full63-core56 deltas and error transitions.
Report favorable-seed counts separately from significance. Bootstrap paired
pedestrian tracks 5000 times, seed42, with seed-ensemble probabilities and
95% percentile intervals for F1/PR-AUC/ROC-AUC (descriptive, not multiplicity
adjusted). Do not conflate ensemble intervals with mean-per-seed intervals.

This is a retrospective matched ablation on previously exposed JAAD test data,
not untouched confirmation. A result applies to these seven features within
this pathway, not universal BIU-version superiority. Save every prediction,
source/reuse audit, model and result. Verify before EXPERIMENT_COMPLETE.json.
