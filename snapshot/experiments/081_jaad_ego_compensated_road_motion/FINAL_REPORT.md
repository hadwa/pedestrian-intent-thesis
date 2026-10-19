# Exp081 Final Report

Exp081 evaluates an image-derived visual ego-motion proxy. It is not physical ego speed and cannot be interpreted as measured vehicle telemetry.

## Protocol

Clean JAAD behavior-pedestrian benchmark; 16 observed frames; TTE 30--60; fixed threshold 0.5; seeds 42/43/44.

## Validation

| variant | seeds | accuracy_mean | accuracy_std | precision_mean | precision_std | recall_mean | recall_std | f1_mean | f1_std | roc_auc_mean | roc_auc_std | pr_auc_mean | pr_auc_std | brier_mean | brier_std | log_loss_mean | log_loss_std | high_confidence_fp_mean | high_confidence_fp_std | high_confidence_fn_mean | high_confidence_fn_std | high_confidence_errors_mean | high_confidence_errors_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 3 | 0.7043795620437957 | 0.0096560266827174 | 0.7665956938580166 | 0.0114558926441343 | 0.7529411764705882 | 0.0101885341621698 | 0.7596488004301408 | 0.0071862897743361 | 0.7278657616892911 | 0.0070145628895568 | 0.8194671086528814 | 0.0059345335318698 | 0.2371409890321697 | 0.0055416693924357 | 0.9140698085504192 | 0.0275827676630231 | 23.33333333333333 | 0.5773502691896258 | 15.333333333333334 | 2.516611478423584 | 38.66666666666666 | 3.055050463303893 |
| B | 3 | 0.6605839416058394 | 0.0072992700729926 | 0.7142678777663326 | 0.0034588335146152 | 0.7549019607843137 | 0.0122450941145066 | 0.7340027532150492 | 0.0073136463327253 | 0.7071266968325792 | 0.0030295841042629 | 0.807434011841603 | 0.0045623351426 | 0.2574304405163406 | 0.0039095928366881 | 0.9395822358683708 | 0.0116899358665992 | 24.0 | 1.0 | 18.33333333333333 | 1.1547005383792517 | 42.333333333333336 | 2.0816659994661326 |
| C | 3 | 0.7968369829683698 | 0.0042142355415301 | 0.8521971655436177 | 0.0034650767491601 | 0.8137254901960785 | 0.0122450941145066 | 0.8324678169073215 | 0.0049782233037711 | 0.7854638009049774 | 0.0041475130153435 | 0.7967245605017871 | 0.0063475818907836 | 0.1662185021658122 | 0.0006615978775018 | 0.7431927941460622 | 0.0175079810455645 | 22.0 | 0.0 | 7.666666666666667 | 0.5773502691896258 | 29.666666666666668 | 0.5773502691896258 |
| D | 3 | 0.8114355231143552 | 0.0042142355415301 | 0.8585858585858586 | 0.0034990925405432 | 0.8333333333333334 | 0.0033961780540566 | 0.8457711442786069 | 0.003446867278744 | 0.7967760180995476 | 0.0039333345363606 | 0.8073805768234811 | 0.0067772490968846 | 0.1624361743609355 | 0.0020127381829682 | 0.7066063395620744 | 0.0124123013540106 | 22.0 | 0.0 | 7.333333333333333 | 0.5773502691896258 | 29.33333333333333 | 0.5773502691896258 |

Frozen validation decision: `no_advance`.

## Efficiency

| component | warmups | repetitions | scope | mean_ms | std_ms | median_ms | p95_ms | min_ms | max_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| twinlite_masks | 3 | 30 | 16 predecoded JAAD frames; CUDA TwinLiteNet+ Large | 157.5426571333333 | 4.617223900643589 | 156.4000545 | 168.03712149999998 | 151.684229 | 172.252839 |
| sparse_motion | 3 | 30 | 15 transitions; all-pedestrian masks; CPU OpenCV | 410.26579910000015 | 13.700984267460369 | 411.8232075 | 429.1537686 | 388.557446 | 441.110989 |
| raw_road | 10 | 100 | cached TwinLite masks to uncompensated 60-D summary | 84.55415275000001 | 5.045371408429934 | 83.08738 | 91.26811349999998 | 78.970591 | 112.432129 |
| compensated_road | 10 | 100 | cached masks and motion to compensated 60-D summary | 83.1900665 | 2.4017619057703885 | 82.683589 | 88.08634975 | 79.461716 | 90.143549 |
| cached_xgboost | 20 | 500 | cached 242-D Variant D row; one predictor | 1.811942856 | 0.2041105607890764 | 1.738155 | 2.20653265 | 1.543288 | 2.771734 |
| incremental_pipeline | 3 | 30 | TwinLite + motion + compensated road + XGBoost; frame decode and ViTPose/anchor extraction excluded | 666.7446881666668 | 9.977802644626802 | 664.668703 | 686.4578315 | 648.339331 | 697.710426 |

The incremental timing excludes frame decode and ViTPose/anchor extraction; it is not a complete end-to-end claim.

## Limitations

JAAD test was historically exposed in separate repository work. Exp081 froze its design and validation gate without accessing that cohort during development.
