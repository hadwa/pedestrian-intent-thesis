# Exp099: Exp077 one-stage XGBoost plus categorical vehicle action

## Frozen comparison

| Arm | Inputs | Dimension |
|---|---|---:|
| A | Exact Exp077 portable one-stage feature vector | 182 |
| B | A plus categorical vehicle-action summaries | 197 |

The shared vector retains BBox-53, the historical combined BIU representation
and the existing temporal contrasts. Do not replace it with Exp098's v1-only
vector or its final 63-D v2. No second-stage model, road context, physical ego
speed, visual ego proxy, confidence addition or BIU change is allowed.

## Vehicle representation

JAAD annotations_vehicle/*_vehicle.xml supplies five nominal categories:
stopped, moving_slow, moving_fast, decelerating, accelerating (codes 0--4).
Use only the exact 16 observation-frame annotations. For each category encode
its observation occupancy fraction (5 numbers), first-frame one-hot indicator
(5) and last-frame one-hot indicator (5). Concatenate in that order. No numeric
speed interpretation, average of category codes, inferred physical acceleration,
future annotation, fitted encoding, unknown-category substitution or feature search.

Verify against Exp089's existing observation-aligned ego_speed.npy cache,
whose historical name actually denotes category_code/4. Convert exactly back
to integer codes only after checking finite values and the five allowed levels.
Verify all used codes against the raw vehicle XML. Missing frame/category fails
closed; no sample may be dropped. The encoding is frozen before validation.

These are annotated vehicle states, not automatically inferred physical ego
measurements. A positive result would establish conditional predictive value
of this annotation, not an end-to-end automatic deployment claim.

## Protocol and reuse gate

- Clean JAAD behavior-pedestrian, default video split; train 1552, validation
  274, test 1501; 16 consecutive observations, TTE30--60, overlap0.8.
- Seeds 42/43/44; train-only fitting; fixed threshold 0.5; no threshold search,
  early stopping, calibration, augmentation, oversampling or train+val refit.
- Exact Exp077 one-stage recipe: 250-tree XGBoost, depth4, LR0.05,
  subsample0.8, colsample_bytree0.8, reg_lambda1, binary logistic/logloss,
  training negative/positive class weight, median imputation, n_jobs4 CPU.
- Reuse A only after authenticating Exp077 completion-manifest hashes for
  prepared development matrices, models and saved validation probabilities;
  verify estimator parameters, imputer medians, feature count/order and
  exact float32 validation-probability reproduction under the current runtime.
  If parity fails, stop before fitting and document the material difference.
- Train three B models freshly; A models remain read-only in Exp077.
- All three B runs and all A validation rows must finish before opening test.
  Both arms advance together regardless of validation score. No post-test edits.
- Test baseline predictions are reused from the authenticated Exp077 artifact,
  not regenerated. Evaluate B once from its frozen validation models.

## Frozen reporting

Report per-seed and mean/sample-SD fixed-0.5 accuracy, precision, recall, F1,
FPR/FNR and confusion counts, PR-AUC (average precision), ROC-AUC, Brier,
high-confidence FP(p>=0.9)/FN(p<=0.1), and paired B-minus-A deltas. Report
corrections and newly introduced errors separately at fixed0.5.

Primary evidence: paired changes in F1 and PR-AUC, with seed consistency and
5000-replicate pedestrian-track bootstrap (seed42) of seed-ensemble probabilities.
Report 95% percentile intervals for F1, PR-AUC, ROC-AUC; do not confuse these
with intervals for mean per-seed metrics. No multiplicity correction. No single
universal winner if metrics conflict. This is a retrospective matched ablation
on previously exposed JAAD test data, not untouched generalization confirmation.

Save design/code/source hashes before fitting, persistent logs, new B checkpoints,
both arms' per-seed predictions, provenance and reuse audits. Verify saved
probability round trips and all metric values before EXPERIMENT_COMPLETE.json.
Do not modify Exp077, Exp089, Exp098, or start another experiment afterward.
