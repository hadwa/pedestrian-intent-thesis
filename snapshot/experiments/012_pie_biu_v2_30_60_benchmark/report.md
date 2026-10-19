# PIE BIU v2 vs BIU v1 benchmark report (30-60, overlap 0.6, test = set03)

## Headline: does BIU v2 beat/match BIU v1 on PIE 30-60?

**Pose-only: BIU v2 core decisively beats BIU v1.** Mean test deltas over seeds 42/43/44
(`biu_v2_core` minus `biu_v1_only`, from `metrics/pie_biu_v2_vs_v1_delta_table.csv`):

| threshold rule | model | F1 v1 | F1 v2 | delta F1 | PR-AUC v1 | PR-AUC v2 | delta PR-AUC |
|---|---|---|---|---|---|---|---|
| fixed_0_5 | logreg | 0.455 | 0.788 | **+0.333** | 0.388 | 0.793 | **+0.405** |
| fixed_0_5 | rf | 0.356 | 0.816 | **+0.460** | 0.533 | 0.869 | **+0.337** |
| fixed_0_5 | xgb | 0.475 | 0.812 | **+0.336** | 0.515 | 0.869 | **+0.354** |
| val_f1 | logreg | 0.464 | 0.790 | **+0.326** | 0.388 | 0.793 | **+0.405** |
| val_f1 | rf | 0.515 | 0.820 | **+0.306** | 0.533 | 0.869 | **+0.337** |
| val_f1 | xgb | 0.511 | 0.810 | **+0.299** | 0.515 | 0.869 | **+0.354** |

**With bbox present: BIU v2 matches but does not beat BIU v1.** (`biu_v2_core_plus_bbox` minus `biu_v1_plus_bbox`):

| threshold rule | model | F1 v1+bbox | F1 v2+bbox | delta F1 | PR-AUC v1+bbox | PR-AUC v2+bbox | delta PR-AUC |
|---|---|---|---|---|---|---|---|
| fixed_0_5 | logreg | 0.813 | 0.816 | +0.003 | 0.805 | 0.818 | +0.013 |
| fixed_0_5 | rf | 0.868 | 0.847 | -0.021 | 0.911 | 0.900 | -0.012 |
| fixed_0_5 | xgb | 0.858 | 0.860 | +0.001 | 0.904 | 0.916 | +0.012 |
| val_f1 | logreg | 0.826 | 0.816 | -0.010 | 0.805 | 0.818 | +0.013 |
| val_f1 | rf | 0.866 | 0.854 | -0.012 | 0.911 | 0.900 | -0.012 |
| val_f1 | xgb | 0.857 | 0.859 | +0.001 | 0.904 | 0.916 | +0.012 |

Summary: on PIE 30-60 the direction-aware BIU v2 core transforms the pose-only signal from
near-useless (BIU v1 F1 approximately 0.36-0.51) into a competitive standalone feature set
(F1 approximately 0.79-0.82, PR-AUC approximately 0.79-0.87), consistent with the JAAD 009
finding. Once bbox motion features are present, v2 is at parity: xgb gains slightly
(+0.001 F1, +0.012 PR-AUC), logreg is mixed (-0.010 F1, +0.013 PR-AUC at val_f1), and rf loses
slightly (-0.012 F1, -0.012 PR-AUC at val_f1). The best single configuration on PIE remains
`biu_v1_plus_bbox` rf at fixed 0.5 (F1 0.868) with `biu_v2_core_plus_bbox` xgb close behind
(F1 0.860, and the best PR-AUC of all sets at 0.916).

## Sanity anchor vs published archive numbers

From `metrics/pie_biu_v2_sanity_anchor_comparison.csv`; archive anchor is
`/home/abdelrahman/Hadwa/BIU-PID-Distillation/results/wacv_pcpa_30_60_overlap06/lightweight_baseline_metrics.csv`
(random_forest, seed 42, validation-F1 threshold):

| feature set (this run) | archive set | archive test F1 | this run seed42 | this run mean +/- std | abs dev (seed42) |
|---|---|---|---|---|---|
| bbox_only rf @ val_f1 | bbox | 0.857564 | 0.861553 | 0.8613 +/- 0.0003 | 0.0040 |
| biu_v1_plus_bbox rf @ val_f1 | biu_bbox | 0.866359 | 0.862069 | 0.8664 +/- 0.0044 | 0.0043 |

Both anchors reproduce within 0.005 (< 0.01 tolerance). Residual deviation is expected from
pipeline differences: the archive suite used its own RF hyperparameters and threshold grid,
whereas this run reuses the JAAD-009 model builders (RF n_estimators=300, min_samples_leaf=2,
class_weight=balanced_subsample, median imputation) and the JAAD-009 candidate-threshold rule.

## Data coverage

- Missing-keypoint windows: **0 of 9745 (0.0%)** — `metrics/pie_biu_v2_missing_keypoint_rows.csv` is empty
  (consistent with the archive feature-table report which also found 0 missing frames).
- Per-frame PIE bboxes (direction-estimation fallback only) were available for all 16 frames of
  all 9745 windows (`metrics/pie_biu_v2_extraction_coverage.csv`).
- Row counts: train 2366 (set01+set02+set06), val 3619 (set04+set05), test 3760 (set03);
  all feature sets cover all split rows with zero dropped sequences
  (`metrics/pie_biu_v2_feature_set_definitions.csv`).
- Feature counts: bbox_only 53, biu_v1_only 72, biu_v1_plus_bbox 125 (matching the archived
  trainer's `feature_count`), biu_v2_core 133, biu_v2_core_plus_bbox 186.

## Protocol notes

- Threshold selection and model selection use validation data only; test = set03 is evaluation-only.
- `fixed_0_5` and `val_f1` are the reported rules; `val_balanced_accuracy`, `val_fpr_le_0p1`,
  and `val_precision_ge_0p8` rows are also logged in `metrics/pie_biu_v2_benchmark_results.csv`
  for completeness but are separate analyses per AGENTS.md.
- The JAAD-009 ambiguous-negative subgroup breakdown was NOT ported: under this PIE protocol,
  negatives are pure never-crossing tracks by construction (crossing-irrelevant tracks excluded),
  so the JAAD ambiguity rule does not transfer directly. PIE ambiguity analysis lives in
  `experiments/015_pie_ambiguous_negative_intention_audit`.
- Bbox sequences enter BIU v2 only as the low-motion direction fallback inside
  `extract_biu_v2`; bbox feature values in `*_plus_bbox` sets come from the frozen archived tables.

## Interpretation

- The PIE result mirrors JAAD: BIU v2's direction-aware lower-body/torso commitment features are
  the difference between a broken pose baseline and a usable one, but bbox motion already carries
  most of the discriminative signal at TTE 30-60, so v2 adds little on top of bbox at F1
  (parity within about +/-0.02 depending on model) while xgb shows a small consistent PR-AUC gain (+0.012).
- If the paper needs a "pose features help" claim on PIE, it should be phrased as
  "BIU v2 makes pose-only prediction viable and matches v1+bbox pipelines", not as a strict
  improvement over the published `biu_v1_plus_bbox` benchmark number.

## Traceability

- Script: `scripts/experiments/pie_biu_v2_30_60_benchmark.py`
- Command: `python3 scripts/experiments/pie_biu_v2_30_60_benchmark.py` (run 2026-07-12)
- All quoted numbers: `metrics/pie_biu_v2_vs_v1_delta_table.csv`,
  `metrics/pie_biu_v2_benchmark_results_mean_std.csv`,
  `metrics/pie_biu_v2_sanity_anchor_comparison.csv`.
- Per-sample probabilities: `predictions/{feature_set}__{model}__seed{seed}.csv` (45 files, val+test).
