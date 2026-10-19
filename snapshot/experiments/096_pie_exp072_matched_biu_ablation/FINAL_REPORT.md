# Exp096 Final Report

Retrospective matched control for the Exp072 XGBoost architecture on PIE set03.
All arms use 16 observed frames, TTE30--60, overlap0.6, fixed threshold 0.5,
the same train/validation/test rows, the same seeds, and the same estimator recipe.

## Fixed-0.5 test summary

| scope | arm | feature_count | accuracy_mean | accuracy_std | precision_mean | precision_std | recall_mean | recall_std | f1_mean | f1_std | roc_auc_mean | roc_auc_std | pr_auc_mean | pr_auc_std | brier_mean | brier_std | log_loss_mean | log_loss_std | ece_10bin_mean | ece_10bin_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | base | 64 | 0.932181 | 0.001407 | 0.850466 | 0.002026 | 0.905473 | 0.003447 | 0.877107 | 0.002642 | 0.970104 | 0.000987 | 0.919403 | 0.002403 | 0.052195 | 0.000482 | 0.191682 | 0.002431 | 0.021239 | 0.001561 |
| test | v1 | 139 | 0.928901 | 0.001465 | 0.837659 | 0.002478 | 0.910448 | 0.004337 | 0.872535 | 0.002752 | 0.970063 | 0.000577 | 0.929760 | 0.002924 | 0.055467 | 0.000540 | 0.199344 | 0.003203 | 0.029333 | 0.000838 |
| test | v2 | 135 | 0.931649 | 0.001219 | 0.844809 | 0.002374 | 0.911774 | 0.003040 | 0.877013 | 0.002219 | 0.974041 | 0.000226 | 0.939475 | 0.000818 | 0.051580 | 0.000822 | 0.184095 | 0.001212 | 0.026693 | 0.001106 |
| test | combined | 187 | 0.930851 | 0.002111 | 0.842059 | 0.003889 | 0.912438 | 0.003588 | 0.875837 | 0.003747 | 0.973072 | 0.000353 | 0.936171 | 0.002378 | 0.052635 | 0.000909 | 0.188514 | 0.002520 | 0.027119 | 0.000107 |

## Mean paired deltas

| comparison | f1_delta_mean | roc_auc_delta_mean | pr_auc_delta_mean | f1_favorable_seeds |
| --- | --- | --- | --- | --- |
| v1_minus_base | -0.004573 | -0.000041 | 0.010357 | 0 |
| v2_minus_base | -0.000094 | 0.003936 | 0.020073 | 1 |
| combined_minus_base | -0.001271 | 0.002967 | 0.016768 | 1 |
| v2_minus_v1 | 0.004479 | 0.003977 | 0.009716 | 3 |
| combined_minus_v1 | 0.003302 | 0.003008 | 0.006411 | 2 |
| combined_minus_v2 | -0.001177 | -0.000969 | -0.003305 | 1 |

## Seed-ensemble track bootstrap

| comparison | seed | metric | replicates | bootstrap_seed | mean_delta | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v2_minus_v1 | 0 | f1 | 10000 | 42 | 0.005068 | -0.004250 | 0.014654 |
| v2_minus_v1 | 0 | pr_auc | 10000 | 42 | 0.009666 | -0.000258 | 0.020191 |
| v2_minus_v1 | 0 | roc_auc | 10000 | 42 | 0.004121 | 0.000527 | 0.008379 |
| combined_minus_v1 | 0 | f1 | 10000 | 42 | 0.003977 | -0.003186 | 0.011358 |
| combined_minus_v1 | 0 | pr_auc | 10000 | 42 | 0.006241 | 0.000219 | 0.012991 |
| combined_minus_v1 | 0 | roc_auc | 10000 | 42 | 0.003086 | 0.001007 | 0.005546 |
| combined_minus_v2 | 0 | f1 | 10000 | 42 | -0.001091 | -0.009225 | 0.007011 |
| combined_minus_v2 | 0 | pr_auc | 10000 | 42 | -0.003425 | -0.011074 | 0.004314 |
| combined_minus_v2 | 0 | roc_auc | 10000 | 42 | -0.001035 | -0.003834 | 0.001584 |

## BIU arms versus the shared base: seed-ensemble track bootstrap

| comparison | seed | metric | replicates | bootstrap_seed | mean_delta | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v1_minus_base | 0 | f1 | 10000 | 42 | -0.001610 | -0.016852 | 0.013158 |
| v1_minus_base | 0 | pr_auc | 10000 | 42 | 0.009895 | -0.006292 | 0.029793 |
| v1_minus_base | 0 | roc_auc | 10000 | 42 | -0.000078 | -0.004661 | 0.004379 |
| v2_minus_base | 0 | f1 | 10000 | 42 | 0.003458 | -0.011517 | 0.018370 |
| v2_minus_base | 0 | pr_auc | 10000 | 42 | 0.019561 | 0.005254 | 0.036897 |
| v2_minus_base | 0 | roc_auc | 10000 | 42 | 0.004044 | 0.000772 | 0.007611 |
| combined_minus_base | 0 | f1 | 10000 | 42 | 0.002367 | -0.013180 | 0.017748 |
| combined_minus_base | 0 | pr_auc | 10000 | 42 | 0.016136 | -0.000674 | 0.036742 |
| combined_minus_base | 0 | roc_auc | 10000 | 42 | 0.003009 | -0.000957 | 0.007189 |

## Interpretation boundary

This experiment was conceived after prior set03 exposure and is therefore a retrospective
matched control, not pristine confirmation. BIU-Combined reproduces Exp072's exact 187-feature
candidate. The v1 and v2 arms isolate their respective pose representations while retaining
the shared BBox, BBox-temporal, ego-speed, and pose-quality inputs. V2-specific temporal
contrasts remain part of the v2 design. The result supports an architecture-specific version
verdict and does not establish universal dominance across models or datasets.
