# Exp074 Final Report

Validation-only controlled comparison of Exp072 against Exp072 plus 60 frozen road-relative YOLOP summaries.

## Validation

| scope | model | seed_count | accuracy_mean | accuracy_std | precision_mean | precision_std | recall_mean | recall_std | f1_mean | f1_std | roc_auc_mean | roc_auc_std | pr_auc_mean | pr_auc_std | fpr_mean | fpr_std | fnr_mean | fnr_std | brier_mean | brier_std | log_loss_mean | log_loss_std | ece_10bin_mean | ece_10bin_std | fp_ge_0p9_count_mean | fp_ge_0p9_count_std | fn_le_0p1_count_mean | fn_le_0p1_count_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation | candidate | 3 | 0.874367 | 0.001798 | 0.679342 | 0.005577 | 0.747448 | 0.002772 | 0.711757 | 0.002886 | 0.919512 | 0.001623 | 0.772815 | 0.004168 | 0.092399 | 0.002514 | 0.252552 | 0.002772 | 0.098781 | 0.001215 | 0.358177 | 0.004470 | 0.075202 | 0.002107 | 103.000000 | 5.291503 | 111.333333 | 2.081666 |
| validation | reference | 3 | 0.873906 | 0.002092 | 0.672245 | 0.003894 | 0.765646 | 0.006919 | 0.715910 | 0.005208 | 0.918170 | 0.001012 | 0.775749 | 0.004666 | 0.097745 | 0.000877 | 0.234354 | 0.006919 | 0.098937 | 0.000726 | 0.353450 | 0.003858 | 0.070205 | 0.000399 | 109.666667 | 3.214550 | 90.666667 | 1.154701 |

| scope | seed | delta_accuracy | delta_precision | delta_recall | delta_f1 | delta_roc_auc | delta_pr_auc | delta_fpr | delta_fnr | delta_brier | delta_log_loss | delta_ece_10bin | delta_fp_ge_0p9_count | delta_fn_le_0p1_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation | 42 | -0.002211 | -0.000215 | -0.019973 | -0.008944 | 0.000795 | -0.003217 | -0.002441 | 0.019973 | 0.000791 | 0.005300 | 0.006515 | -3.000000 | 17.000000 |
| validation | 43 | -0.001105 | 0.004815 | -0.025300 | -0.008609 | -0.000333 | -0.005339 | -0.005230 | 0.025300 | 0.000797 | 0.008959 | 0.005334 | -5.000000 | 23.000000 |
| validation | 44 | 0.004697 | 0.016691 | -0.009321 | 0.005094 | 0.003561 | -0.000245 | -0.008368 | 0.009321 | -0.002057 | -0.000080 | 0.003141 | -12.000000 | 22.000000 |

```json
{
  "confidence_path": false,
  "mean_deltas": {
    "f1": -0.0041530237633650895,
    "pr_auc": -0.00293386509873117,
    "roc_auc": 0.0013410756833752702
  },
  "mean_high_confidence": {
    "candidate_combined": 214.33333333333331,
    "candidate_fn": 111.33333333333333,
    "candidate_fp": 103.0,
    "combined_ratio": 1.0698835274542429,
    "reference_combined": 200.33333333333334,
    "reference_fn": 90.66666666666667,
    "reference_fp": 109.66666666666667
  },
  "mean_transitions": {
    "corrections": 62.666666666666664,
    "harms": 61.0
  },
  "model_manifest_pre_gate": {
    "sha256": "cdfb40535bee0d2d770df773b8a59d9d85357d648643f65525147acc341caecc",
    "size_bytes": 636
  },
  "per_seed_safety": true,
  "performance_path": false,
  "reference": "Exp072_XGBoost",
  "set03_materialized": false,
  "status": "no_advance",
  "threshold": 0.5
}
```

## Inference Latency

| model | seed | batch_size | warmups | repetitions | mean_ms | median_ms | p95_ms | per_sample_mean_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Exp072_XGBoost | 42 | 1 | 100 | 1000 | 1.271967 | 1.231620 | 1.548977 | 1.271967 |
| Exp072_XGBoost | 42 | 256 | 20 | 200 | 1.599019 | 1.578857 | 1.886996 | 0.006246 |
| Exp074_Road_XGBoost | 42 | 1 | 100 | 1000 | 1.434920 | 1.338966 | 1.705254 | 1.434920 |
| Exp074_Road_XGBoost | 42 | 256 | 20 | 200 | 1.893122 | 1.794098 | 2.270938 | 0.007395 |
| Exp072_XGBoost | 43 | 1 | 100 | 1000 | 1.259057 | 1.207294 | 1.492661 | 1.259057 |
| Exp072_XGBoost | 43 | 256 | 20 | 200 | 1.711216 | 1.656643 | 1.957847 | 0.006684 |
| Exp074_Road_XGBoost | 43 | 1 | 100 | 1000 | 1.384656 | 1.339781 | 1.635880 | 1.384656 |
| Exp074_Road_XGBoost | 43 | 256 | 20 | 200 | 1.771771 | 1.738030 | 2.050843 | 0.006921 |
| Exp072_XGBoost | 44 | 1 | 100 | 1000 | 1.224745 | 1.200410 | 1.405984 | 1.224745 |
| Exp072_XGBoost | 44 | 256 | 20 | 200 | 1.713799 | 1.656117 | 2.011319 | 0.006695 |
| Exp074_Road_XGBoost | 44 | 1 | 100 | 1000 | 1.399942 | 1.343065 | 1.647310 | 1.399942 |
| Exp074_Road_XGBoost | 44 | 256 | 20 | 200 | 1.774543 | 1.755858 | 1.929690 | 0.006932 |

## Outcome

`NO ADVANCE`. Set03 road extraction and evaluation were not run.
