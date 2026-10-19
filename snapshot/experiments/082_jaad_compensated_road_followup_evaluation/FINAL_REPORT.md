# Exp082 Final Report

This is a validation-selected follow-up evaluation. Exp081 remains `no_advance`; this report does not reinterpret its gate.

## Protocol

Clean JAAD; 1,501 test windows; 16 observed consecutive frames; TTE 30--60; fixed threshold 0.5; frozen Exp081 seeds 42/43/44; no retraining or retuning.

## Test Results

| variant | seeds | accuracy_mean | accuracy_std | precision_mean | precision_std | recall_mean | recall_std | f1_mean | f1_std | roc_auc_mean | roc_auc_std | pr_auc_mean | pr_auc_std | brier_mean | brier_std | log_loss_mean | log_loss_std | tn_mean | tn_std | fp_mean | fp_std | fn_mean | fn_std | tp_mean | tp_std | high_confidence_fp_mean | high_confidence_fp_std | high_confidence_fn_mean | high_confidence_fn_std | high_confidence_errors_mean | high_confidence_errors_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 3 | 0.659560293137908 | 0.0017626591013088 | 0.6018946233326882 | 0.0020189499131879 | 0.8574074074074073 | 0.0034952937200328 | 0.7072757098888548 | 0.0004812651208241 | 0.7528912955849577 | 0.0003998753700781 | 0.7132659550506388 | 0.0051423422118421 | 0.2742023642936655 | 0.0019187689380233 | 1.0224181066446514 | 0.0117073502869948 | 372.6666666666667 | 5.033222956847167 | 408.3333333333333 | 5.033222956847167 | 102.66666666666669 | 2.516611478423583 | 617.3333333333334 | 2.516611478423583 | 247.33333333333331 | 5.131601439446884 | 33.666666666666664 | 4.725815626252609 | 281.0 | 5.291502622129181 |
| D | 3 | 0.7041972018654231 | 0.0037093699952231 | 0.6441481115281841 | 0.0026651575055781 | 0.8564814814814815 | 0.0057824055540726 | 0.7352904786173572 | 0.0036775823874202 | 0.7502830535400958 | 0.0048863325031672 | 0.6593775301871627 | 0.0043398706689799 | 0.2391213776334517 | 0.0024437997349885 | 0.9370799387495764 | 0.0159128966062291 | 440.3333333333333 | 2.516611478423583 | 340.6666666666667 | 2.516611478423583 | 103.33333333333331 | 4.163331998932265 | 616.6666666666666 | 4.163331998932265 | 202.33333333333331 | 3.2145502536643185 | 30.33333333333333 | 3.5118845842842465 | 232.66666666666663 | 6.6583281184793925 |

Mean D-A deltas: F1 `+0.028015`, PR-AUC `-0.053888`, ROC-AUC `-0.002608`.

## Per-seed D-A Deltas

| seed | comparison | delta_accuracy | delta_precision | delta_recall | delta_f1 | delta_roc_auc | delta_pr_auc | delta_brier | delta_log_loss | delta_tn | delta_fp | delta_fn | delta_tp | delta_high_confidence_fp | delta_high_confidence_fn | delta_high_confidence_errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | D_minus_A | 0.0419720186542305 | 0.0389543881312238 | 0.0041666666666666 | 0.0275666095514136 | -0.0036207141840945 | -0.0540143334015421 | -0.0329433056875922 | -0.0846601388884685 | 60.0 | -60.0 | -3.0 | 3.0 | -42.0 | -2.0 | -44.0 |
| 43 | D_minus_A | 0.0499666888740839 | 0.0469880175502772 | 0.0 | 0.0316161567399866 | 0.0024114383269311 | -0.0469963364499762 | -0.0395520925250584 | -0.1109955134777614 | 75.0 | -75.0 | 0.0 | 0.0 | -53.0 | -3.0 | -56.0 |
| 44 | D_minus_A | 0.0419720186542305 | 0.0408180589049865 | -0.0069444444444444 | 0.0248615398941067 | -0.0066154502774222 | -0.0606546047389098 | -0.0327475617679904 | -0.0603588513189944 | 68.0 | -68.0 | 5.0 | -5.0 | -40.0 | -5.0 | -45.0 |

## Fixed-0.5 Error Transitions

| seed | transition | count |
| --- | --- | --- |
| 42 | A_correct_to_D_wrong | 83 |
| 42 | A_wrong_to_D_correct | 146 |
| 42 | unchanged_correctness | 1272 |
| 43 | A_correct_to_D_wrong | 79 |
| 43 | A_wrong_to_D_correct | 154 |
| 43 | unchanged_correctness | 1268 |
| 44 | A_correct_to_D_wrong | 86 |
| 44 | A_wrong_to_D_correct | 149 |
| 44 | unchanged_correctness | 1266 |

## Limitations

The A/D comparison was selected after reviewing validation metrics, and JAAD test was historically exposed in earlier repository experiments. Treat this as transparent follow-up test evidence, not pristine external confirmation.
