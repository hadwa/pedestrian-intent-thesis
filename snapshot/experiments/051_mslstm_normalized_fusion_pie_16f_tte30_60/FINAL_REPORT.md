# Exp051 Final Report

Fresh normalized-fusion A--F results on PIE 16 observed frames / TTE30--60. Fixed threshold: 0.5.

| variant | accuracy_mean | accuracy_std | precision_mean | precision_std | recall_mean | recall_std | f1_mean | f1_std | roc_auc_mean | roc_auc_std | pr_auc_mean | pr_auc_std | total_parameters_mean | total_parameters_std | training_seconds_mean | training_seconds_std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.887589 | 0.008533 | 0.801318 | 0.005598 | 0.770813 | 0.049355 | 0.785136 | 0.023491 | 0.91357 | 0.008204 | 0.869104 | 0.010335 | 15436304 | 0 | 73.136889 | 2.420427 |
| B | 0.903191 | 0.006536 | 0.861876 | 0.051188 | 0.764179 | 0.043269 | 0.80834 | 0.010244 | 0.933009 | 0.002767 | 0.87701 | 0.006888 | 13218062 | 0 | 69.784508 | 5.498084 |
| C | 0.873759 | 0.009249 | 0.752782 | 0.016238 | 0.785738 | 0.019909 | 0.768884 | 0.017279 | 0.897375 | 0.007203 | 0.851135 | 0.012365 | 8871690 | 0 | 39.382755 | 2.55196 |
| D | 0.885904 | 0.013868 | 0.794813 | 0.068414 | 0.782421 | 0.038206 | 0.78621 | 0.013007 | 0.903666 | 0.002804 | 0.858936 | 0.003934 | 11112460 | 0 | 45.731183 | 0.25263 |
| E | 0.885372 | 0.007037 | 0.804948 | 0.036921 | 0.756882 | 0.025297 | 0.77936 | 0.005069 | 0.894618 | 0.000796 | 0.818596 | 0.004047 | 8945931 | 0 | 38.319146 | 0.754237 |
| F | 0.887145 | 0.005034 | 0.79625 | 0.023839 | 0.777778 | 0.015068 | 0.786586 | 0.004184 | 0.900579 | 0.002958 | 0.831884 | 0.015042 | 8947083 | 0 | 38.754644 | 0.750999 |

The set03 result is the designated Exp051 final evaluation. set03 had prior independent post-hoc exposure, but Exp051 was frozen before this evaluation.

Pose extraction and YOLOP extraction runtimes are excluded because pinned historical caches were reused.

The sole model change from Exp050 is parameter-free LayerNorm on each 512-D stream context immediately before the unchanged learned weighted sum.

## Exp050 Paired Comparison

| variant | accuracy_delta_mean | accuracy_delta_std | precision_delta_mean | precision_delta_std | recall_delta_mean | recall_delta_std | f1_delta_mean | f1_delta_std | roc_auc_delta_mean | roc_auc_delta_std | pr_auc_delta_mean | pr_auc_delta_std |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.009752 | 0.014034 | 0.034882 | 0.008991 | -0.010282 | 0.053955 | 0.011461 | 0.032266 | 0.014624 | 0.014908 | 0.016225 | 0.019129 |
| B | 0.004078 | 0.011474 | 0.044741 | 0.060173 | -0.038143 | 0.035395 | -0.001261 | 0.015514 | 0.005914 | 0.001152 | 0.004069 | 0.015284 |
| C | -0.005142 | 0.010553 | -0.031447 | 0.032604 | 0.030182 | 0.007334 | -0.000399 | 0.014249 | 0.004882 | 0.004965 | 0.015806 | 0.006554 |
| D | 0.007358 | 0.011612 | 0.022082 | 0.062048 | 0.00796 | 0.040832 | 0.013035 | 0.011767 | 0.012103 | 0.00485 | 0.022461 | 0.008067 |
| E | 0.020656 | 0.0174 | 0.070608 | 0.051914 | -0.035489 | 0.022123 | 0.020208 | 0.018443 | 0.009922 | 0.010184 | 0.015463 | 0.014583 |
| F | 0.018617 | 0.027443 | 0.051953 | 0.10687 | -0.022222 | 0.083347 | 0.020639 | 0.020022 | -0.002992 | 0.008254 | -0.005425 | 0.027149 |
