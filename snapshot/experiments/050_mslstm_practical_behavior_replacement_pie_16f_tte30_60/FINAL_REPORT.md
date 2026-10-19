# Exp050 Final Report

Fresh controlled A--F results on PIE 16 observed frames / TTE30--60. Fixed threshold: 0.5.

| variant | accuracy_mean | accuracy_std | precision_mean | precision_std | recall_mean | recall_std | f1_mean | f1_std | roc_auc_mean | roc_auc_std | pr_auc_mean | pr_auc_std | total_parameters_mean | total_parameters_std | training_seconds_mean | training_seconds_std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.877837 | 0.006793 | 0.766436 | 0.014452 | 0.781095 | 0.010763 | 0.773675 | 0.01184 | 0.898946 | 0.00807 | 0.852879 | 0.009777 | 15436304 | 0 | 65.779778 | 7.533038 |
| B | 0.899113 | 0.004942 | 0.817135 | 0.014704 | 0.802322 | 0.008104 | 0.809601 | 0.008174 | 0.927095 | 0.003422 | 0.872941 | 0.011003 | 13218062 | 0 | 55.896989 | 4.931725 |
| C | 0.878901 | 0.002583 | 0.784229 | 0.017952 | 0.755556 | 0.02239 | 0.769283 | 0.005086 | 0.892493 | 0.003901 | 0.835329 | 0.005812 | 8871690 | 0 | 37.184943 | 4.743145 |
| D | 0.878546 | 0.003532 | 0.772731 | 0.021609 | 0.774461 | 0.022697 | 0.773174 | 0.001851 | 0.891563 | 0.002228 | 0.836475 | 0.004695 | 11112460 | 0 | 43.495858 | 1.744671 |
| E | 0.864716 | 0.023444 | 0.73434 | 0.078459 | 0.792371 | 0.040349 | 0.759152 | 0.023243 | 0.884696 | 0.009668 | 0.803132 | 0.013402 | 8945931 | 0 | 36.895529 | 3.926227 |
| F | 0.868528 | 0.023527 | 0.744297 | 0.085379 | 0.8 | 0.06916 | 0.765947 | 0.01755 | 0.903571 | 0.005349 | 0.837309 | 0.012197 | 8947083 | 0 | 37.349732 | 2.281326 |

The set03 result is the designated Exp050 final evaluation. set03 had prior independent post-hoc exposure, but Exp050 was frozen before this evaluation.

Pose extraction and YOLOP extraction runtimes are excluded because pinned historical caches were reused.
