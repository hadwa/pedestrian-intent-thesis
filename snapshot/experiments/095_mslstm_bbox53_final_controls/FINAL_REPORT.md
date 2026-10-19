# Exp095 Final Report

Matched MSLSTM BBox53 controls on PIE: 16 observed frames, TTE30--60, fixed threshold 0.5.

| variant | condition | accuracy_mean | accuracy_std | precision_mean | precision_std | recall_mean | recall_std | f1_mean | f1_std | roc_auc_mean | roc_auc_std | pr_auc_mean | pr_auc_std | total_parameters_mean | total_parameters_std | training_seconds_mean | training_seconds_std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G | core | 0.901773 | 0.005187 | 0.840111 | 0.031453 | 0.783085 | 0.028701 | 0.809927 | 0.008528 | 0.935398 | 0.007298 | 0.85365 | 0.019578 | 6765321 | 0 | 34.398743 | 5.381713 |
| H | biu_v1 | 0.906117 | 0.003685 | 0.833765 | 0.024455 | 0.811609 | 0.0204 | 0.822135 | 0.003165 | 0.93483 | 0.003169 | 0.846777 | 0.009543 | 6840714 | 0 | 33.740131 | 4.046522 |
| I | biu_v2 | 0.909309 | 0.005754 | 0.838543 | 0.04081 | 0.821559 | 0.032155 | 0.828921 | 0.00531 | 0.938407 | 0.006973 | 0.859921 | 0.012702 | 6839562 | 0 | 31.252657 | 2.200633 |
| J | raw_pose | 0.907801 | 0.005146 | 0.847217 | 0.028179 | 0.800663 | 0.026445 | 0.822746 | 0.008873 | 0.937325 | 0.005278 | 0.854871 | 0.015619 | 9006091 | 0 | 40.440736 | 0.699946 |
| K | gt_action_look | 0.90789 | 0.008216 | 0.828825 | 0.017096 | 0.826202 | 0.022697 | 0.827391 | 0.01606 | 0.945671 | 0.005506 | 0.862852 | 0.010801 | 11111693 | 0 | 47.867083 | 0.835783 |
| L | biu_combined | 0.907004 | 0.007783 | 0.839022 | 0.032878 | 0.808624 | 0.021256 | 0.823058 | 0.011955 | 0.936324 | 0.008323 | 0.86043 | 0.015667 | 6848778 | 0 | 31.625465 | 1.83327 |

G/H/I are authenticated Exp053 references. J/K/L are fresh matched controls.
This is a retrospective control study on the previously exposed set03 cohort, not pristine confirmation.
Comparisons concern predictive utility under this architecture; they do not establish semantic equivalence or universal replacement.
