# Exp037 Results

## Scientific Status

- Development selection: `none`.
- Set06 promotion: `no_advance`.
- Threshold remained fixed at `0.5`.
- Development used only set01, set02, set04, and set05.
- Set06 was the locked internal confirmation cohort.
- The combined keypoint file was streamed under the frozen exception; no unauthorized record was retained, and the hardened-loader reconstruction matched the saved matrix.
- Set03 was not accessed or materialized because the frozen gate did not advance.

## Development OOF

| scope | set_id | arm | n_features | n | positive_count | threshold | tn | fp | fn | tp | accuracy | precision | recall | f1_fixed_0p5 | fpr | fnr | pr_auc | roc_auc | brier | log_loss | ece | fp_ge_0p9_count | fn_le_0p1_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| development_oof | pooled | joint_rf | 161 | 4781 | 1223 | 0.5 | 3218 | 340 | 376 | 847 | 0.8502405354528342 | 0.7135636057287279 | 0.6925592804578904 | 0.7029045643153526 | 0.0955593029792018 | 0.3074407195421095 | 0.7420281198670472 | 0.8939857527426591 | 0.1098117728592385 | 0.3559806124550599 | 0.0331850572782236 | 42 | 58 |
| development_oof | pooled | joint_temporal_ego_xgb | 187 | 4781 | 1223 | 0.5 | 3264 | 294 | 260 | 963 | 0.8841246601129471 | 0.766109785202864 | 0.7874080130825838 | 0.7766129032258065 | 0.0826306913996627 | 0.2125919869174161 | 0.812923096297851 | 0.928224121059862 | 0.0919138620833041 | 0.3369995628408221 | 0.0526169443720457 | 132 | 110 |
| development_oof | pooled | joint_temporal_xgb | 182 | 4781 | 1223 | 0.5 | 3207 | 351 | 319 | 904 | 0.8598619535661995 | 0.7203187250996016 | 0.7391659852820932 | 0.7296206618240516 | 0.0986509274873524 | 0.2608340147179068 | 0.7473476288820046 | 0.9061998872095958 | 0.1082737206473255 | 0.3981846014622325 | 0.0674826356585137 | 166 | 134 |
| development_oof | pooled | joint_xgb | 161 | 4781 | 1223 | 0.5 | 3214 | 344 | 343 | 880 | 0.8563062120895211 | 0.7189542483660131 | 0.7195421095666394 | 0.7192480588475685 | 0.0966835300730747 | 0.2804578904333606 | 0.7226245571305497 | 0.8919438971152958 | 0.1144628923480894 | 0.4325971434531124 | 0.0732296658359463 | 169 | 154 |
| development_oof | pooled | v1_bbox_rf | 125 | 4781 | 1223 | 0.5 | 3275 | 283 | 384 | 839 | 0.8604894373562016 | 0.7477718360071302 | 0.6860179885527392 | 0.7155650319829424 | 0.079539066891512 | 0.3139820114472608 | 0.7381160511278673 | 0.902685873208694 | 0.1067418541887078 | 0.3501787124810893 | 0.0569280777753362 | 21 | 43 |
| development_oof | pooled | v2_bbox_xgb | 109 | 4781 | 1223 | 0.5 | 3217 | 341 | 362 | 861 | 0.8529596318761765 | 0.71630615640599 | 0.7040065412919051 | 0.7101030927835051 | 0.09584035975267 | 0.2959934587080948 | 0.7280485552627252 | 0.8932501331744891 | 0.1163629533945121 | 0.429932259131194 | 0.0768802945783498 | 155 | 159 |

The strongest candidate, `joint_temporal_ego_xgb`, improved PR-AUC by `0.084875` and fixed-0.5 F1 by `0.061048`. It was rejected because its high-confidence FP/FN counts were `132`/`110` versus the dual-baseline minima `21`/`43`.

## Paired Track Bootstrap

| metric | replicates | mean_delta | ci_low | ci_high | arm | reference_arm |
| --- | --- | --- | --- | --- | --- | --- |
| pr_auc | 10000 | -0.0048386490885337 | -0.0230097101344387 | 0.0137792519278397 | joint_xgb | v2_bbox_xgb |
| f1_fixed_0p5 | 10000 | 0.0035887444967778 | -0.0177399885556982 | 0.0255259300502133 | joint_xgb | v1_bbox_rf |
| pr_auc | 10000 | 0.0136027033125812 | -0.010649876825097 | 0.0387906571366723 | joint_rf | v2_bbox_xgb |
| f1_fixed_0p5 | 10000 | -0.012767785488476 | -0.0308394883924983 | 0.0051985210199273 | joint_rf | v1_bbox_rf |
| pr_auc | 10000 | 0.0194061371723646 | -0.0005623588569615 | 0.0401135151518989 | joint_temporal_xgb | v2_bbox_xgb |
| f1_fixed_0p5 | 10000 | 0.0139994518492915 | -0.0080752320866785 | 0.0374408852797603 | joint_temporal_xgb | v1_bbox_rf |
| pr_auc | 10000 | 0.0843867074387418 | 0.0597526913321226 | 0.1113236325940325 | joint_temporal_ego_xgb | v2_bbox_xgb |
| f1_fixed_0p5 | 10000 | 0.061147880503725 | 0.0362063958079306 | 0.0874570720014946 | joint_temporal_ego_xgb | v1_bbox_rf |

## Per-Set Deltas

| arm | set_id | pr_auc_delta_vs_v2 | f1_delta_vs_best |
| --- | --- | --- | --- |
| joint_rf | set01 | -0.0335748602198368 | -0.0790439992269536 |
| joint_rf | set02 | -0.0093288001479717 | -0.046721072125655 |
| joint_rf | set04 | 0.0373252898908836 | -0.0200148027717036 |
| joint_rf | set05 | -0.0752868557094049 | -0.0701902748414375 |
| joint_temporal_ego_xgb | set01 | 0.0016709433070116 | 0.0588767277414209 |
| joint_temporal_ego_xgb | set02 | 0.0272771998924048 | 0.034673549512869 |
| joint_temporal_ego_xgb | set04 | 0.1046903990257535 | 0.029090022296544 |
| joint_temporal_ego_xgb | set05 | 0.3302450283748133 | 0.1381818181818181 |
| joint_temporal_xgb | set01 | -0.027096309270324 | 0.0198707294552169 |
| joint_temporal_xgb | set02 | 0.0121566314345371 | 0.0299516111097162 |
| joint_temporal_xgb | set04 | 0.0335430713336654 | -0.032711424120287 |
| joint_temporal_xgb | set05 | -0.003695877394957 | 0.0363636363636363 |
| joint_xgb | set01 | -0.0219532442407502 | 0.0228177195720183 |
| joint_xgb | set02 | 0.0002630108395023 | 0.0081336143522442 |
| joint_xgb | set04 | 0.0058907498019779 | -0.041290407844046 |
| joint_xgb | set05 | -0.0990364767991238 | -0.0157804459691252 |

## Reproduction

```bash
python experiments/037_pie_joint_biu_temporal_ego/scripts/run_experiment.py prepare
python experiments/037_pie_joint_biu_temporal_ego/scripts/run_experiment.py develop
python experiments/037_pie_joint_biu_temporal_ego/scripts/run_experiment.py confirm
python experiments/037_pie_joint_biu_temporal_ego/scripts/run_experiment.py evaluate
python experiments/037_pie_joint_biu_temporal_ego/scripts/run_experiment.py report
python experiments/037_pie_joint_biu_temporal_ego/scripts/run_experiment.py verify
```
