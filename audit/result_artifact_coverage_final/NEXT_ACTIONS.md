# Exact remaining packaging work

This is an audit handoff, not permission to retrain. All 69 weight files below exist; no refit is needed. Their paths, checksums and mapped result groups are in [weights_to_package.csv](weights_to_package.csv).

| Experiment | Existing weight files not yet packaged |
|---|---|
| 053_mslstm_bbox53_replacement | 9 |
| 095_mslstm_bbox53_final_controls | 6 |
| 096_pie_exp072_matched_biu_ablation | 9 |
| 098_jaad_matched_biu_xgb | 3 |
| 100_jaad_exp099_biu_pathways | 3 |
| 102_jaad_api_matched_biu | 9 |
| 116_xgboost_v2_core_completion | 24 |
| 117_neural_v2_core_completion | 3 |
| 118_pie_ordered_history_xgb | 3 |

Other gaps are listed in [artifact_gaps.json](artifact_gaps.json): saved prediction inputs, current exhibit sources, bibliography and plotting code. Checkpoint counts do not include the explicitly unavailable original JAAD-all/A baseline, nor unlocated original Exp012/earlier PIE classifier weights. Preserve their prediction-supported status.

The 985 recorded dependency references all exist locally, but their source manifests may cover excluded arms. Curate their necessity before copying; do not package every cached file automatically. Presence was checked, not the integrity of every upstream payload.

The 25 distinct primary result files are present and already byte-identically packaged. The check resolves 66 primary table rows by exact keys; it does not independently verify every printed number or third-party citation.
