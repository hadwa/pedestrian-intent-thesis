# Exp039 Results

## Scientific Status

- Set06 confirmation: `no_advance`.
- Threshold remained fixed at `0.5`.
- Temperatures were frozen before set06 from Exp038 development folds.
- One artifact-free stalled confirmation invocation was terminated and the identical frozen command rerun; see `EXECUTION_LOG.md`.

## Set06 Raw

| scope | set_id | arm | n | positive_count | threshold | tn | fp | fn | tp | accuracy | precision | recall | f1_fixed_0p5 | fpr | fnr | pr_auc | roc_auc | brier | log_loss | ece | fp_ge_0p9_count | fn_le_0p1_count | fp_ge_0p9_rate_all_negatives | fn_le_0p1_rate_all_positives |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| set06_confirmation_raw | set06 | joint_temporal_ego_xgb | 1204 | 207 | 0.5 | 929 | 68 | 24 | 183 | 0.9235880398671096 | 0.7290836653386454 | 0.8840579710144928 | 0.7991266375545851 | 0.0682046138415245 | 0.1159420289855072 | 0.8264869740364016 | 0.9663095566893918 | 0.0572129252415582 | 0.2031384806666916 | 0.0442059311284042 | 24 | 6 | 0.0240722166499498 | 0.0289855072463768 |
| set06_confirmation_raw | set06 | v1_bbox_rf | 1204 | 207 | 0.5 | 940 | 57 | 54 | 153 | 0.9078073089700996 | 0.7285714285714285 | 0.7391304347826086 | 0.7338129496402878 | 0.0571715145436308 | 0.2608695652173913 | 0.7347554625676174 | 0.938685622083642 | 0.0725264988179407 | 0.2493307445972928 | 0.0634597893471624 | 9 | 4 | 0.0090270812437311 | 0.0193236714975845 |
| set06_confirmation_raw | set06 | v2_bbox_xgb | 1204 | 207 | 0.5 | 926 | 71 | 35 | 172 | 0.9119601328903656 | 0.7078189300411523 | 0.8309178743961353 | 0.7644444444444445 | 0.0712136409227683 | 0.1690821256038647 | 0.7297405672708196 | 0.937459722161654 | 0.0728971648871085 | 0.2748749662985101 | 0.0608137819541431 | 36 | 12 | 0.0361083249749247 | 0.0579710144927536 |

## Set06 Calibrated

| scope | set_id | arm | n | positive_count | threshold | tn | fp | fn | tp | accuracy | precision | recall | f1_fixed_0p5 | fpr | fnr | pr_auc | roc_auc | brier | log_loss | ece | fp_ge_0p9_count | fn_le_0p1_count | fp_ge_0p9_rate_all_negatives | fn_le_0p1_rate_all_positives |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| set06_confirmation_calibrated | set06 | soft_temp_joint_temporal_ego_xgb | 1204 | 207 | 0.5 | 929 | 68 | 24 | 183 | 0.9235880398671096 | 0.7290836653386454 | 0.8840579710144928 | 0.7991266375545851 | 0.0682046138415245 | 0.1159420289855072 | 0.8264869740364016 | 0.9663095566893918 | 0.05943976653261 | 0.2182727436803811 | 0.0733450149845712 | 8 | 2 | 0.0080240722166499 | 0.0096618357487922 |
| set06_confirmation_calibrated | set06 | soft_temp_v1_bbox_rf | 1204 | 207 | 0.5 | 940 | 57 | 54 | 153 | 0.9078073089700996 | 0.7285714285714285 | 0.7391304347826086 | 0.7338129496402878 | 0.0571715145436308 | 0.2608695652173913 | 0.7347554625676174 | 0.938685622083642 | 0.0725264988179407 | 0.2493307445972928 | 0.063459789347164 | 9 | 4 | 0.0090270812437311 | 0.0193236714975845 |
| set06_confirmation_calibrated | set06 | soft_temp_v2_bbox_xgb | 1204 | 207 | 0.5 | 926 | 71 | 35 | 172 | 0.9119601328903656 | 0.7078189300411523 | 0.8309178743961353 | 0.7644444444444445 | 0.0712136409227683 | 0.1690821256038647 | 0.7297405672708196 | 0.937459722161654 | 0.0829194822856508 | 0.3019707698742526 | 0.1293167490425887 | 12 | 1 | 0.0120361083249749 | 0.0048309178743961 |

## Paired Track Bootstrap

| metric | replicates | mean_delta | ci_low | ci_high | reference_arm |
| --- | --- | --- | --- | --- | --- |
| pr_auc | 10000 | 0.0962986844012252 | 0.0260475721759186 | 0.1807440200471978 | soft_temp_v2_bbox_xgb |
| f1_fixed_0p5 | 10000 | 0.0347650644572351 | -0.008042318657368 | 0.0788108503806582 | soft_temp_v2_bbox_xgb |

## Gate

```json
{
  "checks": {
    "bootstrap_f1_positive": false,
    "bootstrap_pr_positive": true,
    "brier_improved": false,
    "combined_tail_reduction": true,
    "confusion_invariant": true,
    "ece_nonincrease": false,
    "f1_gain": true,
    "fn_tail_nonincrease": true,
    "fp_tail_nonincrease": true,
    "log_loss_improved": false,
    "pr_auc_gain": true
  },
  "observed": {
    "best_baseline_f1": 0.7644444444444445,
    "best_f1_reference_arm": "soft_temp_v2_bbox_xgb",
    "calibrated_combined_tail": 10,
    "candidate_f1": 0.7991266375545851,
    "candidate_pr_auc": 0.8264869740364016,
    "combined_tail_limit": 22,
    "raw_combined_tail": 30,
    "v2_pr_auc": 0.7297405672708196
  },
  "status": "no_advance"
}
```

Set03 was not materialized because set06 did not advance.
