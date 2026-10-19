# Original experiment recipes

Packaging policy: no retraining. Training details are documentation for readers; commands are not being executed. Use saved weights and evidence where available.

## Verified recipe excerpts

| Experiment | Original settings | Authoritative source |
|---|---|---|
| Exp053 PIE MSLSTM | Seeds 42/43/44; Adam 1e-4; batch 32; 15 epochs; validation-F1 checkpoint selection; original Gaussian minority augmentation; fixed threshold .5 | [Design](snapshot/experiments/053_mslstm_bbox53_replacement/DESIGN.md) |
| Exp119 capped PedCMT | Adam 5e-4, betas (.9,.98), epsilon 1e-6; batch 64; up to 200 epochs; patience 25; minimum validation uncapped BCE, earliest tie; frozen parent; inference cap .10; fixed threshold .5 | [Design](snapshot/experiments/119_pedcmt_v2_capped_completion/DESIGN.md) |
| Exp122 JAAD MSLSTM | Seeds 42/43/44; Adam 1e-4; batch 32; 15 epochs; balanced oversampling without Gaussian noise; maximum validation AP, earliest tie; fixed threshold .5 | [Design](snapshot/experiments/122_jaad_mslstm_no_noise/DESIGN.md), [test evaluation](snapshot/experiments/122_jaad_mslstm_no_noise/test_evaluation/evaluate.py) |
| Exp124 / Exp116 PSI | Median imputation; XGBoost 250 trees, depth 4, LR .05, subsample/column sample .8, L2 1; 4 CPU threads; weight 7697/18598; seeds 42/43/44; no early stopping/selection; threshold .5 | [Design](snapshot/experiments/124_psi_matched_lightweight_ablation/DESIGN.md), [frozen config](configs/psi_exp124.json) |

Exp119 uses PIE balanced indices and JAAD natural indices. Preserve this distinction. Exp122 development documentation predates its separately approved test evaluation; use its test-evaluation folder for the thesis evidence. Do not apply the PSI recipe to every XGBoost experiment.

## Mapped experiment source index

These links preserve the full recipes and source code; historical runners are not portable commands. Missing documents are not replaced by guessed settings.

### 012_pie_biu_v2_30_60_benchmark

Sources: [README.md](snapshot/experiments/012_pie_biu_v2_30_60_benchmark/README.md).

### 053_mslstm_bbox53_replacement

Sources: [DESIGN.md](snapshot/experiments/053_mslstm_bbox53_replacement/DESIGN.md); [README.md](snapshot/experiments/053_mslstm_bbox53_replacement/README.md); [FINAL_REPORT.md](snapshot/experiments/053_mslstm_bbox53_replacement/FINAL_REPORT.md).

Recorded implementations: [scripts/run_experiment.py](snapshot/experiments/053_mslstm_bbox53_replacement/scripts/run_experiment.py).

### 092_pedcmt_capped_biu_correction

Sources: [DESIGN.md](snapshot/experiments/092_pedcmt_capped_biu_correction/DESIGN.md); [README.md](snapshot/experiments/092_pedcmt_capped_biu_correction/README.md).

Recorded implementations: [scripts/run_experiment.py](snapshot/experiments/092_pedcmt_capped_biu_correction/scripts/run_experiment.py).

### 093_pedcmt_pie_capped_biu_correction

Sources: [DESIGN.md](snapshot/experiments/093_pedcmt_pie_capped_biu_correction/DESIGN.md); [README.md](snapshot/experiments/093_pedcmt_pie_capped_biu_correction/README.md).

Recorded implementations: [scripts/run_experiment.py](snapshot/experiments/093_pedcmt_pie_capped_biu_correction/scripts/run_experiment.py).

### 094_pedcmt_jaad_capped_raw_pose_control

Sources: [DESIGN.md](snapshot/experiments/094_pedcmt_jaad_capped_raw_pose_control/DESIGN.md); [README.md](snapshot/experiments/094_pedcmt_jaad_capped_raw_pose_control/README.md).

Recorded implementations: [scripts/run_experiment.py](snapshot/experiments/094_pedcmt_jaad_capped_raw_pose_control/scripts/run_experiment.py).

### 095_mslstm_bbox53_final_controls

Sources: [DESIGN.md](snapshot/experiments/095_mslstm_bbox53_final_controls/DESIGN.md); [README.md](snapshot/experiments/095_mslstm_bbox53_final_controls/README.md); [FINAL_REPORT.md](snapshot/experiments/095_mslstm_bbox53_final_controls/FINAL_REPORT.md).

Recorded implementations: [scripts/run_experiment.py](snapshot/experiments/095_mslstm_bbox53_final_controls/scripts/run_experiment.py).

### 096_pie_exp072_matched_biu_ablation

Sources: [DESIGN.md](snapshot/experiments/096_pie_exp072_matched_biu_ablation/DESIGN.md); [README.md](snapshot/experiments/096_pie_exp072_matched_biu_ablation/README.md); [FINAL_REPORT.md](snapshot/experiments/096_pie_exp072_matched_biu_ablation/FINAL_REPORT.md).

Recorded implementations: [scripts/run_experiment.py](snapshot/experiments/096_pie_exp072_matched_biu_ablation/scripts/run_experiment.py).

### 098_jaad_matched_biu_xgb

Sources: [DESIGN.md](snapshot/experiments/098_jaad_matched_biu_xgb/DESIGN.md); [README.md](snapshot/experiments/098_jaad_matched_biu_xgb/README.md); [FINAL_REPORT.md](snapshot/experiments/098_jaad_matched_biu_xgb/FINAL_REPORT.md).

Recorded implementations: [scripts/run_experiment.py](snapshot/experiments/098_jaad_matched_biu_xgb/scripts/run_experiment.py).

### 100_jaad_exp099_biu_pathways

Sources: [DESIGN.md](snapshot/experiments/100_jaad_exp099_biu_pathways/DESIGN.md); [README.md](snapshot/experiments/100_jaad_exp099_biu_pathways/README.md); [FINAL_REPORT.md](snapshot/experiments/100_jaad_exp099_biu_pathways/FINAL_REPORT.md).

Recorded implementations: [scripts/run_experiment.py](snapshot/experiments/100_jaad_exp099_biu_pathways/scripts/run_experiment.py).

### 101_jaad_v2_core_vs_full

Sources: [DESIGN.md](snapshot/experiments/101_jaad_v2_core_vs_full/DESIGN.md); [README.md](snapshot/experiments/101_jaad_v2_core_vs_full/README.md); [FINAL_REPORT.md](snapshot/experiments/101_jaad_v2_core_vs_full/FINAL_REPORT.md).

Recorded implementations: [scripts/run_experiment.py](snapshot/experiments/101_jaad_v2_core_vs_full/scripts/run_experiment.py).

### 102_jaad_api_matched_biu

Sources: [DESIGN.md](snapshot/experiments/102_jaad_api_matched_biu/DESIGN.md); [README.md](snapshot/experiments/102_jaad_api_matched_biu/README.md); [FINAL_REPORT.md](snapshot/experiments/102_jaad_api_matched_biu/FINAL_REPORT.md).

Recorded implementations: [scripts/run_experiment.py](snapshot/experiments/102_jaad_api_matched_biu/scripts/run_experiment.py).

### 113_psi_native_biu_xgb

Sources: [DESIGN.md](snapshot/experiments/113_psi_native_biu_xgb/DESIGN.md); [README.md](snapshot/experiments/113_psi_native_biu_xgb/README.md).

Recorded implementations: [run.py](snapshot/experiments/113_psi_native_biu_xgb/run.py); [test_evaluation/evaluate.py](snapshot/experiments/113_psi_native_biu_xgb/test_evaluation/evaluate.py).

### 116_xgboost_v2_core_completion

Sources: [DESIGN.md](snapshot/experiments/116_xgboost_v2_core_completion/DESIGN.md); [README.md](snapshot/experiments/116_xgboost_v2_core_completion/README.md); [FINAL_REPORT.md](snapshot/experiments/116_xgboost_v2_core_completion/FINAL_REPORT.md).

Recorded implementations: [scripts/run.py](snapshot/experiments/116_xgboost_v2_core_completion/scripts/run.py).

### 117_neural_v2_core_completion

Sources: [DESIGN.md](snapshot/experiments/117_neural_v2_core_completion/DESIGN.md); [README.md](snapshot/experiments/117_neural_v2_core_completion/README.md); [FINAL_REPORT.md](snapshot/experiments/117_neural_v2_core_completion/FINAL_REPORT.md).

Recorded implementations: [scripts/run.py](snapshot/experiments/117_neural_v2_core_completion/scripts/run.py).

### 118_pie_ordered_history_xgb

Sources: [DESIGN.md](snapshot/experiments/118_pie_ordered_history_xgb/DESIGN.md); [README.md](snapshot/experiments/118_pie_ordered_history_xgb/README.md).

Recorded implementations: [scripts/run.py](snapshot/experiments/118_pie_ordered_history_xgb/scripts/run.py).

### 119_pedcmt_v2_capped_completion

Sources: [DESIGN.md](snapshot/experiments/119_pedcmt_v2_capped_completion/DESIGN.md); [README.md](snapshot/experiments/119_pedcmt_v2_capped_completion/README.md); [FINAL_REPORT.md](snapshot/experiments/119_pedcmt_v2_capped_completion/FINAL_REPORT.md).

Recorded implementations: [scripts/run.py](snapshot/experiments/119_pedcmt_v2_capped_completion/scripts/run.py).

### 122_jaad_mslstm_no_noise

Sources: [DESIGN.md](snapshot/experiments/122_jaad_mslstm_no_noise/DESIGN.md); [README.md](snapshot/experiments/122_jaad_mslstm_no_noise/README.md); [FINAL_REPORT.md](snapshot/experiments/122_jaad_mslstm_no_noise/FINAL_REPORT.md).

Recorded implementations: [scripts/run.py](snapshot/experiments/122_jaad_mslstm_no_noise/scripts/run.py); [test_evaluation/evaluate.py](snapshot/experiments/122_jaad_mslstm_no_noise/test_evaluation/evaluate.py).

### 124_psi_matched_lightweight_ablation

Sources: [DESIGN.md](snapshot/experiments/124_psi_matched_lightweight_ablation/DESIGN.md); [README.md](snapshot/experiments/124_psi_matched_lightweight_ablation/README.md); [FINAL_REPORT.md](snapshot/experiments/124_psi_matched_lightweight_ablation/FINAL_REPORT.md).

Recorded implementations: [run.py](snapshot/experiments/124_psi_matched_lightweight_ablation/run.py).

