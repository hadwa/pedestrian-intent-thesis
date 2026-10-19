# Frozen XGBoost v2-core completion

Scope approved: complete missing standalone 56-D BIU-v2 XGBoost comparisons;
finish these before deciding on neural reruns. No neural or thesis files change.
Existing benchmark outputs are immutable. Experiment number 116 was free.

## Cases and interpretation

| Case | Parent reference | New dimension |
|---|---|---:|
| jaad_clean_simple | Exp098 C, BBox + full63 | 109 |
| psi_simple | Exp113 C, BBox + full63 | 109 |
| pie_road | Exp075 candidate, combined + temporal + ego + roads | 195 |
| jaad_clean_road | Exp082 D / Exp081 D, combined + temporal + roads | 190 |
| jaad_beh_road | Exp083 JAAD_beh, same configuration | 190 |
| jaad_all_road | Exp083 JAAD_all, same configuration | 190 |
| jaad_clean_road_vehicle | Exp115 clean, preceding road model + vehicle state | 205 |
| jaad_beh_road_vehicle | Exp115 behavior, same configuration | 205 |
| jaad_all_road_vehicle | Exp115 all, same configuration | 205 |
| psi_ego | Exp114 S187, combined + temporal + ego | 135 |
| psi_road_ego | Exp114 R247, preceding model + roads | 195 |

These are 11 configurations x seeds 42/43/44 = 33 fits. Reuse rather than rerun
the existing simple PIE core56 (Exp012), extended PIE core56 (Exp096), clean
JAAD vehicle core56 (Exp100), and API JAAD simple core56 (Exp102).
The simple cases directly compare core56 with full63. Extended cases replace
the historical combined static BIU block with core56; they do NOT isolate the
seven extra full63 features. All existing 21 temporal features remain unchanged,
including 12 v2-based, six BBox-based and three pose-quality summaries. No newly
introduced confidence input. The frozen BBox direction fallback remains.

## Fixed recipe

Clone each corresponding saved parent pipeline, retaining its exact imputer,
250-tree depth-4 XGBoost, learning rate 0.05, row/column subsampling 0.8,
regularization, training-only negative/positive class weight and four CPU threads.
Refit the cloned model and imputer on the identical training population only.
No hyperparameter search, threshold selection, early stopping or calibration.
All probabilities are evaluated at fixed 0.5; PR-AUC means average precision.
Report accuracy, precision, recall, F1, AP, ROC-AUC, macro F1, balanced accuracy,
FP/FN and high-confidence errors (FP >=0.9, FN <=0.1), per seed and mean/sample SD.
Report paired new-minus-parent deltas without selecting a winner for deployment.

## Frozen features and populations

Use the ordered 56 names in Exp012's feature-set definitions, never the first
56 columns of an unspecified cache. Retain original parent precision and
normalization. Reuse cached pose/features, not a new pose extraction. For API
JAAD extended cases recover absent overlapping v2 measurements using the exact
Exp083 cached coordinates, confidence, boxes and extractor defaults; verify all
36 already-present v2 measurements against the parent matrix. Never substitute
v1 measurements simply because a cue name overlaps.

Retain parent BBox/ego/road/vehicle/temporal values bit-for-bit and identical
sample IDs, ordering, labels, frames and split membership. PSI uses 15 observed
frames and a human-intent vote target at the next (16th) frame, NOT the PIE/JAAD
crossing-event benchmark. Its simple and extended parent training populations
differ; compare each new arm only to its own parent. Retain documented missing
pose behavior and exclusions. PIE uses 16 consecutive observations/TTE30-60.
Clean JAAD uses that construction; API JAAD retains its parent's sampling
semantics and any documented frame gaps rather than silently repairing it.

## Gates and reporting

Freeze this design and code before any new validation result. Prepare all train/
validation features, check row and feature invariants, replay every parent
validation prediction, and verify split disjointness before training. Complete
and hash all 33 models and validation predictions before opening test caches or
test prediction files. Then evaluate every predeclared case once, regardless of
validation outcome. No redesign after test. Reconstruct metrics from saved
probabilities and replay parent test predictions before marking complete.

These are retrospective controlled completion ablations on previously exposed
cohorts, not pristine new generalization confirmations. Tiny differences are
not evidence of universal superiority. Keep dataset/task/cohort distinctions.
Publish source hashes, complete ordered feature contracts, parity audits,
per-seed predictions, fit times, summaries and paired deltas. Completion marker
is written only after the entire grid and saved-prediction checks pass.

## Execution

From /home/abdelrahman/Hadwa/BIU-PID-Distillation-clean:

```bash
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python -u experiments/116_xgboost_v2_core_completion/scripts/run.py
```

The runner persists progress to logs/run.log and writes only under Exp116.
No tmux, perception rerun, archive edit, git commit, or neural training.
