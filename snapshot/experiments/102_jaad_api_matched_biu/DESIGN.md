# Exp102: API-style matched BBox/BIU ablation

Approved on 2026-09-06. One controlled experiment; do not alter existing runs.
Reuse exact Exp083 JAAD_beh and JAAD_all manifests and pose sources. Train each
population separately, seeds42/43/44. All arms advance to evaluation together
after all development runs, irrespective of scores. Fixed0.5, no tuning.

## Frozen arms

| Arm | Inputs | Dimension |
|---|---|---:|
| A | BBox53 | 53 |
| B | BBox53 + v1-72 | 125 |
| C | BBox53 + v2-core56 | 109 |
| D | BBox53 + full-v2-63 | 116 |
| E | BBox53 + complete namespaced v1-72 + full-v2-63 | 188 |

The union is concatenation, not Exp083's historical name-deduplicated union.
C selects the historical56 named columns from the SAME full63 values as D.
No road, temporal contrasts, ego, vehicle action, raw-pose stream, calibration,
new confidence, or architecture changes. Existing v1 validity counters and
v2 direction metadata remain. Do not interpret confidence metadata as an
independent new experiment.

## Extraction and training

Match Exp098/089 preprocessing: float32 input coordinates, confidences and
BIU BBoxes; Exp089 frozen v1/v2 helpers; per-video original image dimensions;
v2 kp_conf=None, include_optional=False, existing BBox direction fallback.
Use the exact Exp083 cached pose measurements, without re-extraction.
BBox53 uses the original float64 annotation BBoxes and historical53 order.
Recompute BIUs only from observation frames, once per unique source window;
reuse JAAD_all features for identical JAAD_beh windows, not duplicate extraction.

Frozen Exp098 estimator: median imputation fitted on training only; XGBoost
250 trees, depth4, LR0.05, subsample0.8, colsample_bytree0.8, reg_lambda1,
binary logistic/logloss, n_jobs4; scale_pos_weight=train negatives/positives.
No validation-driven threshold, checkpoint, classifier or feature selection.

## Cohort and protocol

Exp083 exact populations/order/labels/BBoxes/frames, official video memberships:
JAAD_beh train2134/val242/test1881; JAAD_all train8613/val1265/test6732.
16 consecutive TRACK observations, nominal endpoint TTE30--60 track-index
steps, overlap0.8. TTE is api_track_length-1-api_window_end_index, not raw
event_frame-end_frame when missing track annotations lie before the endpoint.
The pretraining audit found33 such behavior training windows (42 all training
windows); preserve them and report every split's mismatch count. Preserve
Exp083's two JAAD_all training windows spanning a134-frame raw gap. Do not
silently reinterpret those as16 consecutive raw frames or fix them here.
Observed-crossing/unverified-onset windows and non-behavior negatives remain
as prescribed by this API-style benchmark; they are not input label features.
This is published-protocol-aligned, not proof of row-identity to another paper,
nor a clean event-aligned benchmark. Test cohorts have historical exposure.

## Reuse gate

Check Exp031 for A/JAAD_all equivalence before fitting. Training/validation
population, order, labels, frame indices, BBoxes,53 feature order/equations,
seeds and estimator parameters must match. Its chosen XGBoost candidates are
the same fixed estimator; historical validation-selected reporting thresholds
do not change their probabilities. Recompute fixed0.5 metrics from saved
probabilities; do not reuse its selected thresholds. No Exp031 checkpoint was
saved, so live model reproduction is unavailable: report this limitation.
If material parity fails, record it and fit A freshly; never claim exact reuse.
Other candidates (009,077,079,080,083,089,098--101, AIBThings tables) must be
recorded in REUSE_AUDIT.md. A feature superset model is not an equivalent.
Expected27 new fits and3 saved-prediction baselines, subject to audit.

## Analysis and completion

Save all validation/test per-seed probabilities, metrics, mean/sampleSD,
fixed0.5 confusion and high-confidence FP>=0.9/FN<=0.1 counts. Report all10
paired feature-arm deltas and transitions separately within each population.
For B-A,C-B,D-C,E-B, use paired5000-replicate pedestrian-track bootstrap,
seed42, of three-seed ensemble probabilities for F1/AP/ROC; descriptive95%
intervals, no multiplicity correction. Do not confuse those with mean-seed CIs.
Compare clean Exp098 rows as historical different-population references,
not matched-sample causal deltas. Do not call every API-style result easier.
Source/schema/code/development artifacts frozen before fitting. Test features
and probabilities consumed only after validation completion. Identity-only
reuse audits may inspect historical manifest metadata beforehand.
Persistent logs and verified completion marker required. Stop after reporting.
