# Exp072 Final Report

PIE 16 observed frames, TTE 30--60, overlap 0.6, fixed threshold 0.5.
The candidate uses 187 frozen BBox/BIU/temporal/ego features and three seeds.

## Validation

| scope | model | seed_count | accuracy_mean | accuracy_std | precision_mean | precision_std | recall_mean | recall_std | f1_mean | f1_std | roc_auc_mean | roc_auc_std | pr_auc_mean | pr_auc_std | fpr_mean | fpr_std | fnr_mean | fnr_std | brier_mean | brier_std | log_loss_mean | log_loss_std | ece_10bin_mean | ece_10bin_std | fp_ge_0p9_count_mean | fp_ge_0p9_count_std | fn_le_0p1_count_mean | fn_le_0p1_count_std | fp_ge_0p9_rate_mean | fp_ge_0p9_rate_std | fn_le_0p1_rate_mean | fn_le_0p1_rate_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation | anchor | 3 | 0.877038 | 0.002086 | 0.712950 | 0.007563 | 0.682202 | 0.001538 | 0.697222 | 0.003555 | 0.891349 | 0.002162 | 0.701152 | 0.007792 | 0.071943 | 0.002708 | 0.317798 | 0.001538 | 0.102957 | 0.001580 | 0.347507 | 0.003758 | 0.078247 | 0.001609 | 14.333333 | 2.516611 | 36.000000 | 1.000000 | 0.004998 | 0.000877 | 0.047936 | 0.001332 |
| validation | candidate | 3 | 0.873906 | 0.002092 | 0.672245 | 0.003894 | 0.765646 | 0.006919 | 0.715910 | 0.005208 | 0.918170 | 0.001012 | 0.775749 | 0.004666 | 0.097745 | 0.000877 | 0.234354 | 0.006919 | 0.098937 | 0.000726 | 0.353450 | 0.003858 | 0.070205 | 0.000399 | 109.666667 | 3.214550 | 90.666667 | 1.154701 | 0.038238 | 0.001121 | 0.120728 | 0.001538 |

### Paired Candidate Minus Anchor Deltas

| scope | seed | delta_accuracy | delta_precision | delta_recall | delta_f1 | delta_roc_auc | delta_pr_auc | delta_fpr | delta_fnr | delta_brier | delta_log_loss | delta_ece_10bin | delta_fp_ge_0p9_count | delta_fn_le_0p1_count | delta_fp_ge_0p9_rate | delta_fn_le_0p1_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation | 42 | -0.002487 | -0.041026 | 0.089214 | 0.021324 | 0.026961 | 0.073676 | 0.026499 | -0.089214 | -0.003405 | 0.009904 | -0.006009 | 95.000000 | 55.000000 | 0.033124 | 0.073236 |
| validation | 43 | -0.003592 | -0.044261 | 0.086551 | 0.018533 | 0.025508 | 0.070447 | 0.027197 | -0.086551 | -0.003501 | 0.005636 | -0.008093 | 94.000000 | 55.000000 | 0.032775 | 0.073236 |
| validation | 44 | -0.003316 | -0.036830 | 0.074567 | 0.016207 | 0.027996 | 0.079668 | 0.023710 | -0.074567 | -0.005155 | 0.002289 | -0.010024 | 97.000000 | 54.000000 | 0.033821 | 0.071904 |

### Frozen Gate

```json
{
  "checks": {
    "all_seed_f1_delta_ge_minus_0p005": true,
    "all_seed_pr_auc_delta_ge_minus_0p005": true,
    "mean_f1_delta_ge_0p005": true,
    "mean_pr_auc_delta_ge_0p005": true,
    "mean_roc_auc_delta_ge_minus_0p002": true
  },
  "mean_deltas": {
    "f1": 0.018687900778151596,
    "pr_auc": 0.07459671212180403,
    "roc_auc": 0.026821668427839285
  },
  "minimum_seed_deltas": {
    "f1": 0.016206934513001148,
    "pr_auc": 0.07044656958862916
  },
  "model_manifest_pre_gate": {
    "sha256": "293f81befe982484ae660101e5d391ab3bc145ce0c9a03baed6c299d540d5f6f",
    "size_bytes": 963
  },
  "set03_materialized": false,
  "status": "advance",
  "threshold": 0.5
}
```

## Set03 Evaluation

| scope | seed | model | n | positive_count | negative_count | threshold | tn | fp | fn | tp | accuracy | precision | recall | f1 | roc_auc | pr_auc | fpr | fnr | brier | log_loss | ece_10bin | fp_ge_0p9_count | fn_le_0p1_count | fp_ge_0p9_rate | fn_le_0p1_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 42 | candidate | 3760 | 1005 | 2755 | 0.500000 | 2581 | 174 | 89 | 916 | 0.930053 | 0.840367 | 0.911443 | 0.874463 | 0.972894 | 0.934858 | 0.063158 | 0.088557 | 0.052491 | 0.189429 | 0.027233 | 61 | 39 | 0.022142 | 0.038806 |
| test | 42 | anchor | 3760 | 1005 | 2755 | 0.500000 | 2594 | 161 | 113 | 892 | 0.927128 | 0.847104 | 0.887562 | 0.866861 | 0.964523 | 0.909571 | 0.058439 | 0.112438 | 0.066912 | 0.247379 | 0.078098 | 1 | 12 | 0.000363 | 0.011940 |
| test | 43 | candidate | 3760 | 1005 | 2755 | 0.500000 | 2588 | 167 | 84 | 921 | 0.933245 | 0.846507 | 0.916418 | 0.880076 | 0.973478 | 0.938915 | 0.060617 | 0.083582 | 0.051807 | 0.185665 | 0.027101 | 56 | 32 | 0.020327 | 0.031841 |
| test | 43 | anchor | 3760 | 1005 | 2755 | 0.500000 | 2591 | 164 | 119 | 886 | 0.924734 | 0.843810 | 0.881592 | 0.862287 | 0.965478 | 0.913449 | 0.059528 | 0.118408 | 0.066064 | 0.237390 | 0.077782 | 0 | 11 | 0.000000 | 0.010945 |
| test | 44 | candidate | 3760 | 1005 | 2755 | 0.500000 | 2580 | 175 | 91 | 914 | 0.929255 | 0.839302 | 0.909453 | 0.872970 | 0.972843 | 0.934738 | 0.063521 | 0.090547 | 0.053607 | 0.190448 | 0.027022 | 57 | 34 | 0.020690 | 0.033831 |
| test | 44 | anchor | 3760 | 1005 | 2755 | 0.500000 | 2597 | 158 | 102 | 903 | 0.930851 | 0.851084 | 0.898507 | 0.874153 | 0.965442 | 0.911364 | 0.057350 | 0.101493 | 0.066458 | 0.239289 | 0.083690 | 1 | 13 | 0.000363 | 0.012935 |

### Mean And Standard Deviation

| scope | model | seed_count | accuracy_mean | accuracy_std | precision_mean | precision_std | recall_mean | recall_std | f1_mean | f1_std | roc_auc_mean | roc_auc_std | pr_auc_mean | pr_auc_std | fpr_mean | fpr_std | fnr_mean | fnr_std | brier_mean | brier_std | log_loss_mean | log_loss_std | ece_10bin_mean | ece_10bin_std | fp_ge_0p9_count_mean | fp_ge_0p9_count_std | fn_le_0p1_count_mean | fn_le_0p1_count_std | fp_ge_0p9_rate_mean | fp_ge_0p9_rate_std | fn_le_0p1_rate_mean | fn_le_0p1_rate_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | anchor | 3 | 0.927571 | 0.003083 | 0.847332 | 0.003643 | 0.889221 | 0.008579 | 0.867767 | 0.005985 | 0.965148 | 0.000541 | 0.911461 | 0.001941 | 0.058439 | 0.001089 | 0.110779 | 0.008579 | 0.066478 | 0.000424 | 0.241352 | 0.005305 | 0.079857 | 0.003324 | 0.666667 | 0.577350 | 12.000000 | 1.000000 | 0.000242 | 0.000210 | 0.011940 | 0.000995 |
| test | candidate | 3 | 0.930851 | 0.002111 | 0.842059 | 0.003889 | 0.912438 | 0.003588 | 0.875837 | 0.003747 | 0.973072 | 0.000353 | 0.936171 | 0.002378 | 0.062432 | 0.001582 | 0.087562 | 0.003588 | 0.052635 | 0.000909 | 0.188514 | 0.002520 | 0.027119 | 0.000107 | 58.000000 | 2.645751 | 35.000000 | 3.605551 | 0.021053 | 0.000960 | 0.034826 | 0.003588 |

### Candidate Minus Anchor Deltas

| scope | seed | delta_accuracy | delta_precision | delta_recall | delta_f1 | delta_roc_auc | delta_pr_auc | delta_fpr | delta_fnr | delta_brier | delta_log_loss | delta_ece_10bin | delta_fp_ge_0p9_count | delta_fn_le_0p1_count | delta_fp_ge_0p9_rate | delta_fn_le_0p1_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 42 | 0.002926 | -0.006737 | 0.023881 | 0.007602 | 0.008371 | 0.025287 | 0.004719 | -0.023881 | -0.014421 | -0.057950 | -0.050865 | 60.000000 | 27.000000 | 0.021779 | 0.026866 |
| test | 43 | 0.008511 | 0.002698 | 0.034826 | 0.017789 | 0.008000 | 0.025467 | 0.001089 | -0.034826 | -0.014257 | -0.051725 | -0.050681 | 56.000000 | 21.000000 | 0.020327 | 0.020896 |
| test | 44 | -0.001596 | -0.011782 | 0.010945 | -0.001183 | 0.007401 | 0.023374 | 0.006171 | -0.010945 | -0.012851 | -0.048840 | -0.056668 | 56.000000 | 21.000000 | 0.020327 | 0.020896 |

### Track-Cluster Bootstrap

| seed | metric | replicates | bootstrap_seed | mean_delta | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- |
| 42 | f1 | 10000 | 42 | 0.007649 | -0.011473 | 0.026965 |
| 42 | pr_auc | 10000 | 42 | 0.025310 | 0.006232 | 0.047058 |
| 42 | roc_auc | 10000 | 42 | 0.008380 | 0.000671 | 0.016375 |
| 43 | f1 | 10000 | 42 | 0.017800 | -0.002897 | 0.038972 |
| 43 | pr_auc | 10000 | 42 | 0.025458 | 0.006593 | 0.048449 |
| 43 | roc_auc | 10000 | 42 | 0.007999 | 0.000698 | 0.015892 |
| 44 | f1 | 10000 | 42 | -0.001205 | -0.020417 | 0.017811 |
| 44 | pr_auc | 10000 | 42 | 0.023326 | 0.004859 | 0.045468 |
| 44 | roc_auc | 10000 | 42 | 0.007398 | 0.000483 | 0.014679 |

### Strict Target Assessment

```json
{
  "all_targets_beaten": true,
  "basis": "candidate_three_seed_metric_mean",
  "checks": {
    "f1": true,
    "pr_auc": true,
    "roc_auc": true
  },
  "means": {
    "f1": 0.8758366146829247,
    "pr_auc": 0.936170656474752,
    "roc_auc": 0.9730715448769462
  },
  "targets": {
    "f1": 0.8685,
    "pr_auc": 0.9119,
    "roc_auc": 0.9652
  }
}
```

## Limitations

Set03 is the official Exp072 test cohort but had prior exposure in separate experiments and qualitative work.
All Exp072 choices were frozen before this evaluation, and the pre-gate model hashes remained unchanged.
The historical pose cache is byte-pinned but its original ViTPose-S checkpoint identity is not cryptographically verifiable.
