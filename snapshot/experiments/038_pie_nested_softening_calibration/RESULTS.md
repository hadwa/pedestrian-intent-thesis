# Exp038 Results

## Scientific Status

- Development promotion: `no_advance`.
- Set06 confirmation: `no_advance`.
- Threshold remained fixed at `0.5`.
- Softening temperature cannot change fixed-0.5 decisions within a cohort.

## Raw Development OOF

| scope | set_id | arm | n | positive_count | threshold | tn | fp | fn | tp | accuracy | precision | recall | f1_fixed_0p5 | fpr | fnr | pr_auc | roc_auc | brier | log_loss | ece | fp_ge_0p9_count | fn_le_0p1_count | fp_ge_0p9_rate_all_negatives | fn_le_0p1_rate_all_positives |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_development_oof | pooled | joint_temporal_ego_xgb | 4781 | 1223 | 0.5 | 3264 | 294 | 260 | 963 | 0.8841246601129471 | 0.766109785202864 | 0.7874080130825838 | 0.7766129032258065 | 0.0826306913996627 | 0.2125919869174161 | 0.812923096297851 | 0.928224121059862 | 0.0919138620833041 | 0.3369995628408223 | 0.0526169443720458 | 132 | 110 | 0.0370994940978077 | 0.0899427636958299 |
| raw_development_oof | pooled | v1_bbox_rf | 4781 | 1223 | 0.5 | 3275 | 283 | 384 | 839 | 0.8604894373562016 | 0.7477718360071302 | 0.6860179885527392 | 0.7155650319829424 | 0.079539066891512 | 0.3139820114472608 | 0.7381160511278673 | 0.902685873208694 | 0.1067418541887078 | 0.3501787124810901 | 0.0569280777753362 | 21 | 43 | 0.005902192242833 | 0.035159443990188 |
| raw_development_oof | pooled | v2_bbox_xgb | 4781 | 1223 | 0.5 | 3217 | 341 | 362 | 861 | 0.8529596318761765 | 0.71630615640599 | 0.7040065412919051 | 0.7101030927835051 | 0.09584035975267 | 0.2959934587080948 | 0.7280485552627252 | 0.8932501331744891 | 0.1163629533945121 | 0.4299322591311942 | 0.0768802945783498 | 155 | 159 | 0.0435637998875772 | 0.1300081766148814 |

## Calibrated Development OOF

| scope | set_id | arm | n | positive_count | threshold | tn | fp | fn | tp | accuracy | precision | recall | f1_fixed_0p5 | fpr | fnr | pr_auc | roc_auc | brier | log_loss | ece | fp_ge_0p9_count | fn_le_0p1_count | fp_ge_0p9_rate_all_negatives | fn_le_0p1_rate_all_positives |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| calibrated_development_oof | pooled | soft_temp_joint_temporal_ego_xgb | 4781 | 1223 | 0.5 | 3264 | 294 | 260 | 963 | 0.8841246601129471 | 0.766109785202864 | 0.7874080130825838 | 0.7766129032258065 | 0.0826306913996627 | 0.2125919869174161 | 0.8079271795031812 | 0.9268932954056064 | 0.0885721715044502 | 0.2956365032318011 | 0.0375959063116842 | 39 | 59 | 0.0109612141652613 | 0.0482420278004906 |
| calibrated_development_oof | pooled | soft_temp_v1_bbox_rf | 4781 | 1223 | 0.5 | 3275 | 283 | 384 | 839 | 0.8604894373562016 | 0.7477718360071302 | 0.6860179885527392 | 0.7155650319829424 | 0.079539066891512 | 0.3139820114472608 | 0.7381160511278673 | 0.902685873208694 | 0.1067418541887078 | 0.3501787124810901 | 0.056928077775337 | 21 | 43 | 0.005902192242833 | 0.035159443990188 |
| calibrated_development_oof | pooled | soft_temp_v2_bbox_xgb | 4781 | 1223 | 0.5 | 3217 | 341 | 362 | 861 | 0.8529596318761765 | 0.71630615640599 | 0.7040065412919051 | 0.7101030927835051 | 0.09584035975267 | 0.2959934587080948 | 0.7201073978117076 | 0.8912085073564255 | 0.1094289203702407 | 0.3569641520682171 | 0.0400653703284595 | 47 | 73 | 0.0132096683530073 | 0.0596892886345053 |

## Outer-Fold Temperatures

| outer_set | raw_arm | calibrated_arm | temperature | identity_log_loss | fitted_log_loss | optimizer_temperature | optimizer_success | inner_rows | outer_rows | remaining_sets_json |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| set01 | joint_temporal_ego_xgb | soft_temp_joint_temporal_ego_xgb | 1.9324965493214652 | 0.4308790005061639 | 0.3491161626570127 | 1.9324965493214652 | True | 4218 | 563 | ["set02","set04","set05"] |
| set01 | v2_bbox_xgb | soft_temp_v2_bbox_xgb | 2.0767356473949308 | 0.4805611576675747 | 0.3743053814288959 | 2.0767356473949308 | True | 4218 | 563 | ["set02","set04","set05"] |
| set01 | v1_bbox_rf | soft_temp_v1_bbox_rf | 1.0 | 0.3741814142093204 | 0.3741814142093204 | 1.0000000241103628 | True | 4218 | 563 | ["set02","set04","set05"] |
| set02 | joint_temporal_ego_xgb | soft_temp_joint_temporal_ego_xgb | 1.9087482934748 | 0.443298943313994 | 0.3691548261046949 | 1.9087482934748 | True | 4182 | 599 | ["set01","set04","set05"] |
| set02 | v2_bbox_xgb | soft_temp_v2_bbox_xgb | 2.21744701983194 | 0.5305711731647763 | 0.4115848793183551 | 2.21744701983194 | True | 4182 | 599 | ["set01","set04","set05"] |
| set02 | v1_bbox_rf | soft_temp_v1_bbox_rf | 1.0 | 0.4070153037161765 | 0.4070153037161765 | 1.0000000241103628 | True | 4182 | 599 | ["set01","set04","set05"] |
| set04 | joint_temporal_ego_xgb | soft_temp_joint_temporal_ego_xgb | 1.7703830850278055 | 0.3808064699953969 | 0.3237029956421553 | 1.7703830850278055 | True | 1248 | 3533 | ["set01","set02","set05"] |
| set04 | v2_bbox_xgb | soft_temp_v2_bbox_xgb | 1.9531901103814129 | 0.4451970266177652 | 0.3629091258527201 | 1.9531901103814129 | True | 1248 | 3533 | ["set01","set02","set05"] |
| set04 | v1_bbox_rf | soft_temp_v1_bbox_rf | 1.0 | 0.3930472349976714 | 0.3930472349976714 | 1.0000000241103628 | True | 1248 | 3533 | ["set01","set02","set05"] |
| set05 | joint_temporal_ego_xgb | soft_temp_joint_temporal_ego_xgb | 1.6534494338776404 | 0.3569378816418134 | 0.3137594560696908 | 1.6534494338776404 | True | 4695 | 86 | ["set01","set02","set04"] |
| set05 | v2_bbox_xgb | soft_temp_v2_bbox_xgb | 1.8191415721079849 | 0.4529224631239564 | 0.386067364217628 | 1.8191415721079849 | True | 4695 | 86 | ["set01","set02","set04"] |
| set05 | v1_bbox_rf | soft_temp_v1_bbox_rf | 1.0 | 0.435448208130974 | 0.435448208130974 | 1.0000000241103628 | True | 4695 | 86 | ["set01","set02","set04"] |

## Gate

```json
{
  "checks": {
    "bootstrap_f1_positive": true,
    "bootstrap_pr_positive": true,
    "brier_improved": true,
    "combined_tail_reduction": true,
    "confusion_invariant": true,
    "ece_nonincrease": true,
    "f1_gain": true,
    "f1_invariant": true,
    "fn_tail_nonincrease": true,
    "fp_tail_nonincrease": true,
    "log_loss_improved": true,
    "per_set_f1": true,
    "per_set_pr": true,
    "pr_auc_gain": true,
    "pr_auc_noninferiority": false
  },
  "observed": {
    "candidate_f1": 0.7766129032258065,
    "candidate_pr_auc": 0.8079271795031812,
    "combined_tail_count": 98,
    "combined_tail_limit": 181
  },
  "status": "no_advance"
}
```

Set06 and set03 were not materialized because development did not advance.
