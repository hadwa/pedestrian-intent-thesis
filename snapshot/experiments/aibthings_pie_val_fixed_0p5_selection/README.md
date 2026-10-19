# AIBThings PIE validation-based fixed-0.5 selection, significance tests, and paper figures

Created 2026-07-10 for the AIBThings 2026 paper revision. All computations reuse saved
probabilities and the verified reproduced estimator; no model was retrained.

## Why this exists

The earlier paper draft's PIE Table rows for KP+BBox and BIU+KP+BBox were taken from
`docs/AIBTHINGS_FIXED_THRESHOLD_RESULTS.md` ("best fixed_0.5 model per feature set"),
where "best" was effectively chosen by **test** F1 for those two feature sets
(LightGBM does not win on any validation metric there). That contradicts the paper's
stated selection rule and AGENTS.md ("Do not use test performance to choose models").
This experiment recomputes validation metrics at fixed threshold 0.5 from the saved
**validation** probability files and selects each feature set's model by validation
fixed-0.5 F1.

## Selection outcome (PIE, 16 obs frames, TTE 30-60, overlap 0.6, fixed 0.5)

| feature set | selected model (val fixed-0.5 F1) | test Acc | test ROC-AUC | test PR-AUC | test F1 | test P | test R |
|---|---|---|---|---|---|---|---|
| bbox | random_forest (0.7152) | 0.9242 | 0.9613 | 0.8863 | 0.8651 | 0.8249 | 0.9095 |
| biu_bbox | random_forest (0.6904) | 0.9279 | 0.9652 | 0.9119 | 0.8685 | 0.8475 | 0.8905 |
| kp_bbox | random_forest (0.6948) | 0.8995 | 0.9521 | 0.8560 | 0.8118 | 0.8126 | 0.8109 |
| biu_kp_bbox | random_forest (0.6897) | 0.9008 | 0.9565 | 0.8732 | 0.8111 | 0.8258 | 0.7970 |
| biu | xgboost (0.3778) | 0.7426 | 0.7213 | 0.5130 | 0.4710 | 0.5224 | 0.4289 |
| kp | xgboost (0.5299) | 0.7957 | 0.8366 | 0.6250 | 0.5710 | 0.6510 | 0.5085 |
| biu_kp | lightgbm (0.5366) | 0.7973 | 0.8403 | 0.6554 | 0.5354 | 0.6913 | 0.4368 |

BBox and BIU+BBox selections are unchanged from the draft. KP+BBox and BIU+KP+BBox
change from LightGBM (test-selected, invalid) to Random Forest (validation-selected).

## Significance tests (fixed 0.5, exact McNemar + 10k paired bootstrap, seed 12345)

- PIE BBox/RF vs BIU+BBox/RF: b=66, c=80, p=0.282; dF1 95% CI [-0.008, +0.015]; dAcc CI [-0.003, +0.010]. Not significant.
- JAAD BBox/RF vs BIU+BBox/LR: b=141, c=201, p=0.0014; dAcc 95% CI [+0.016, +0.063]. Significant accuracy gain.
- JAAD BBox/RF vs BIU+KP+BBox/LGBM: b=48, c=107, p=1.9e-6; dF1 95% CI [+0.013, +0.036]. Significant F1 gain.

## Naive baselines (test)

- PIE all-positive: Acc 0.267, F1 0.422. PIE majority (negative): Acc 0.733.
- JAAD all-positive: Acc 0.480, F1 0.648. JAAD majority (negative): Acc 0.520.

## Feature importances (paper interpretability figure)

Estimator: `experiments/aibthings_gpu_runtime_minimal/pie_biu_bbox_random_forest_reproduced.joblib`
(verified against saved val/test probabilities to <1e-12; see
`experiments/aibthings_model_only_runtime/model_reproduction_check.csv`). Feature order from
`/home/abdelrahman/Hadwa/BIU-PID-Distillation/data/processed/features_wacv_pcpa_30_60_overlap06/biu_bbox_test.csv`.
Loaded under scikit-learn 1.8.0 (fitted under 1.6.1) only to read `feature_importances_`
(a stored attribute); an `InconsistentVersionWarning` was emitted but no predictions from
this load are used anywhere.

BIU cues carry 20.9% of total Gini importance in the fused BIU+BBox RF; top BIU cues:
lead_foot_elevation 0.045, stride_length 0.028, norm_shoulder_width 0.022,
head_yaw_angle 0.022, face_yaw_angle 0.020, gaze_to_car_angle 0.019, body_yaw_angle 0.018,
arm_swing_asymmetry 0.016, torso_lean_angle 0.015, lead_foot_lifted 0.003.

## Files

- `run_val_fixed_0p5_selection.py` — selection + significance + baselines
- `run_feature_importance.py` — RF importances
- `make_paper_figures.py` — `fig_biu_bbox_importance.pdf/.png`, `fig_pipeline.pdf/.png`
- `pie_val_fixed_0p5_all_models.csv`, `pie_selected_by_val_fixed_0p5.csv`,
  `significance_tests.csv`, `naive_baselines.csv`,
  `rf_biu_bbox_feature_importances.csv`, `rf_biu_bbox_group_importances.csv`

## Inputs (read-only)

- PIE val/test probabilities: `/home/abdelrahman/Hadwa/BIU-PID-Distillation/results/wacv_pcpa_30_60_overlap06/student_*_{val,test}_predictions_*.csv`
- JAAD probabilities: `experiments/aibthings_jaad_all_models_fixed_0p5_30_60/probabilities/`

## Commands run

```
python experiments/aibthings_pie_val_fixed_0p5_selection/run_val_fixed_0p5_selection.py
python experiments/aibthings_pie_val_fixed_0p5_selection/run_feature_importance.py
python experiments/aibthings_pie_val_fixed_0p5_selection/make_paper_figures.py
```

## Track-level (cluster) bootstrap — supersedes window-level tests for the paper

Overlapping windows within a track are correlated, so the window-level McNemar/bootstrap
above are anti-conservative. `run_cluster_bootstrap.py` resamples test TRACKS with
replacement (10k iterations, seed 12345); results in `cluster_bootstrap_tests.csv`:

- JAAD BBox/RF vs BIU+KP+BBox/LGBM (151 tracks): dF1 CI [+0.002,+0.050] p=0.028; dAcc CI [+0.007,+0.075] p=0.017. SIGNIFICANT.
- JAAD BBox/RF vs BIU+BBox/LR: dAcc CI [-0.018,+0.096] p=0.17. NOT significant at track level
  (the window-level McNemar p=0.0014 was anti-conservative, as suspected).
- PIE BBox/RF vs BIU+BBox/RF (654 tracks): dF1 CI [-0.014,+0.021] p=0.70. Not significant.

The paper (docs/aibthings_paper/main.tex) quotes ONLY these cluster-level numbers.

Command: `python experiments/aibthings_pie_val_fixed_0p5_selection/run_cluster_bootstrap.py`
