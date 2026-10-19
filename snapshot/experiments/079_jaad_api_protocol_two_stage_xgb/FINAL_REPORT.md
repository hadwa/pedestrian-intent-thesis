# Exp079 Final Report

Diagnostic reconstruction of the flawed JAAD API-style protocol: 16 observed frames, nominal TTE 30--60, fixed threshold 0.5.
The one/two-stage model recipe is unchanged from Exp077; only the JAAD cohort/protocol differs.

## Native Validation

| scope | model | seed_count | accuracy_mean | accuracy_std | precision_mean | precision_std | recall_mean | recall_std | f1_mean | f1_std | roc_auc_mean | roc_auc_std | pr_auc_mean | pr_auc_std | fpr_mean | fpr_std | fnr_mean | fnr_std | brier_mean | brier_std | log_loss_mean | log_loss_std | fp_ge_0p9_count_mean | fp_ge_0p9_count_std | fn_le_0p1_count_mean | fn_le_0p1_count_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| jaad_api_protocol_native_validation | one_stage | 3 | 0.9167325428194992 | 0.0009128067497069 | 0.6438104205900816 | 0.0075824722279825 | 0.899621212121212 | 0.0215110164613646 | 0.7503609910174331 | 0.0024836329810501 | 0.9645229846676128 | 0.0004453628321863 | 0.7385134070777012 | 0.0090040015388995 | 0.0805019895928986 | 0.0045297363290323 | 0.1003787878787878 | 0.0215110164613646 | 0.0640900439694453 | 0.0019046850772709 | 0.2197319439191846 | 0.0046973085737964 | 22.33333333333333 | 0.5773502691896258 | 3.0 | 0.0 |
| jaad_api_protocol_native_validation | two_stage | 3 | 0.9003952569169961 | 0.0036225894821785 | 0.5872769298401302 | 0.0088830289866766 | 0.956439393939394 | 0.0118276477242393 | 0.7276869613435476 | 0.0085392110020729 | 0.9629125274786432 | 0.0017294955007538 | 0.7494668527295184 | 0.0328140190642615 | 0.1086623813896541 | 0.0038230780522794 | 0.043560606060606 | 0.0118276477242393 | 0.0777578045250223 | 0.0027160747329263 | 0.2610367378270337 | 0.0088382066371174 | 36.66666666666666 | 2.8867513459481287 | 0.6666666666666666 | 0.5773502691896258 |

## Flawed-Protocol Test

| scope | model | seed_count | accuracy_mean | accuracy_std | precision_mean | precision_std | recall_mean | recall_std | f1_mean | f1_std | roc_auc_mean | roc_auc_std | pr_auc_mean | pr_auc_std | fpr_mean | fpr_std | fnr_mean | fnr_std | brier_mean | brier_std | log_loss_mean | log_loss_std | fp_ge_0p9_count_mean | fp_ge_0p9_count_std | fn_le_0p1_count_mean | fn_le_0p1_count_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| jaad_api_protocol_native_test | one_stage | 3 | 0.8740344622697563 | 0.0007427213309565 | 0.6079664146963166 | 0.0016864282647157 | 0.7870291702067403 | 0.006376850891644 | 0.6859971930259291 | 0.0026456063378597 | 0.9304197539550048 | 0.0008863981523536 | 0.7751474476484135 | 0.0025588421661094 | 0.1075307530753075 | 0.0011573538267018 | 0.2129708297932596 | 0.0063768508916441 | 0.0914688840628415 | 0.0007494626323002 | 0.2960951172588169 | 0.001779694768116 | 155.0 | 6.082762530298219 | 56.333333333333336 | 2.0816659994661326 |
| jaad_api_protocol_native_test | two_stage | 3 | 0.8470984353337295 | 0.0028079712574287 | 0.5408240675541881 | 0.0053436306056104 | 0.8312092891532142 | 0.0060276965884431 | 0.655286360589042 | 0.0057793765281816 | 0.9270393309509372 | 0.0015097971857387 | 0.759702134730428 | 0.0077465605829422 | 0.1495349534953495 | 0.0021476783880195 | 0.1687907108467856 | 0.0060276965884431 | 0.1054763552590011 | 0.001400102771102 | 0.3353145881412174 | 0.0036005992087696 | 194.33333333333331 | 1.5275252316519468 | 33.333333333333336 | 2.516611478423583 |
| pie_to_jaad_api_zero_shot_test | one_stage | 3 | 0.4611309170132699 | 0.0052734957946402 | 0.1613123973632166 | 0.0010845553205113 | 0.4958935145851034 | 0.0114724605497191 | 0.2434271515186504 | 0.0025667257544941 | 0.4699130320848567 | 0.0015376391799617 | 0.1864694137082413 | 0.005395679953708 | 0.5462346234623462 | 0.0087625199334037 | 0.5041064854148967 | 0.0114724605497191 | 0.4294516098064693 | 0.0054394008469064 | 1.466440982515946 | 0.03470626083907 | 1697.6666666666667 | 61.068267810159256 | 268.3333333333333 | 2.516611478423583 |
| pie_to_jaad_api_zero_shot_test | two_stage | 3 | 0.4104773222420281 | 0.0091933581118744 | 0.1543138030549373 | 0.0025613402739192 | 0.5293118096856415 | 0.0081048360358278 | 0.2389535919056141 | 0.0035168827300676 | 0.4480440567421228 | 0.012014644865689 | 0.1876931763540896 | 0.0071288083419367 | 0.6147014701470147 | 0.011647503230721 | 0.4706881903143585 | 0.0081048360358279 | 0.4391540693008725 | 0.0056673227955675 | 1.3235261364276645 | 0.020267720638853 | 1391.0 | 36.05551275463989 | 205.0 | 15.716233645501712 |

## Two-Stage Minus One-Stage Deltas

| scope | seed | delta_accuracy | delta_precision | delta_recall | delta_f1 | delta_roc_auc | delta_pr_auc | delta_fpr | delta_fnr | delta_brier | delta_log_loss | delta_fp_ge_0p9_count | delta_fn_le_0p1_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pie_to_jaad_api_zero_shot_test | 42 | -0.0447118241235888 | -0.0045676780271543 | 0.038232795242141 | -0.0010472564166249 | -0.0071922315426104 | 0.0153768614298594 | 0.0622862286228622 | -0.038232795242141 | 0.0010868642412275 | -0.1920317478072135 | -406.0 | -52.0 |
| pie_to_jaad_api_zero_shot_test | 43 | -0.0521390374331551 | -0.0069142436454989 | 0.0356839422259983 | -0.0039946675749338 | -0.0254502170692855 | -0.0032821476629082 | 0.0707470747074707 | -0.0356839422259983 | 0.01574156506305 | -0.1105393871422657 | -214.0 | -79.0 |
| pie_to_jaad_api_zero_shot_test | 44 | -0.0551099227569815 | -0.0095138612521846 | 0.0263381478334748 | -0.0083787548475499 | -0.0329644774163058 | -0.0084234258294062 | 0.0723672367236724 | -0.0263381478334749 | 0.0122789491789317 | -0.1261734033153667 | -300.0 | -59.0 |
| jaad_api_protocol_native_test | 42 | -0.0230243612596554 | -0.0591546085204125 | 0.0509770603228546 | -0.0229130742224813 | -0.0027284121785161 | -0.0136690486851973 | 0.0387038703870387 | -0.0509770603228547 | 0.0123514240478977 | 0.0363526953912308 | 48.0 | -21.0 |
| jaad_api_protocol_native_test | 43 | -0.0286690433749257 | -0.0699798859605154 | 0.0365335598980459 | -0.0355329272911425 | -0.0030344580762239 | -0.0113270308079869 | 0.0424842484248424 | -0.0365335598980458 | 0.0147254572415937 | 0.0400853945899164 | 34.0 | -21.0 |
| jaad_api_protocol_native_test | 44 | -0.0291146761734996 | -0.0722925469454577 | 0.0450297366185216 | -0.0336864957970377 | -0.0043783987574628 | -0.0213398592607719 | 0.0448244824482448 | -0.0450297366185216 | 0.0149455322989873 | 0.0412203226660544 | 36.0 | -27.0 |

## Protocol References

| protocol | source | model | f1 | pr_auc | roc_auc | accuracy | fp_ge_0p9_count | fn_le_0p1_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| clean_event_aligned | Exp077 | one_stage | 0.7072757098888548 | 0.7132659550506388 | 0.7528912955849577 | 0.6595602931379081 | 247.33333333333331 | 33.666666666666664 |
| clean_event_aligned | Exp077 | two_stage | 0.7039101400047966 | 0.7207940339306989 | 0.7449074074074074 | 0.6573395514101711 | 226.33333333333331 | 23.33333333333333 |
| flawed_api | Exp031 | bbox53_xgb_exp031 | 0.6603180465373707 | 0.7495223698962556 | 0.9277799487680288 | 0.8501683501683502 | 190.33333333333331 | 20.666666666666668 |
| flawed_api | Exp079 | one_stage | 0.6859971930259293 | 0.7751474476484135 | 0.9304197539550046 | 0.8740344622697563 | 155.0 | 56.333333333333336 |
| flawed_api | Exp079 | two_stage | 0.655286360589042 | 0.759702134730428 | 0.927039330950937 | 0.8470984353337294 | 194.33333333333331 | 33.333333333333336 |

The clean and flawed rows are not directly comparable benchmarks because the populations and label/temporal semantics differ.

## Primary Flawed-Protocol Deltas

| comparison | delta_f1 | delta_pr_auc | delta_roc_auc | delta_accuracy | delta_fp_ge_0p9_count | delta_fn_le_0p1_count |
| --- | --- | --- | --- | --- | --- | --- |
| one_stage_minus_bbox53 | 0.0256791464885586 | 0.0256250777521578 | 0.0026398051869758 | 0.0238661121014061 | -35.33333333333334 | 35.66666666666667 |
| two_stage_minus_bbox53 | -0.0050316859483287 | 0.0101797648341723 | -0.0007406178170917 | -0.0030699148346208 | 4.0 | 12.666666666666668 |
| two_stage_minus_one_stage | -0.0307108324368873 | -0.0154453129179854 | -0.0033804230040676 | -0.0269360269360269 | 39.33333333333334 | -23.0 |

## Population Diagnostic

| track_type | windows | positives | unique_pedestrians | api_flaw_affected |
| --- | --- | --- | --- | --- |
| behavior_b | 1881 | 1177 | 171 | 737 |
| non_behavior | 4851 | 0 | 441 | 0 |

## Pose Runtime

- CUDA device: `cuda:0`.
- Median synchronized pose inference per unique image: `19.419 ms`.
- Sequential 16-frame estimate from the median: `310.710 ms`.

## Integrity Findings

- Exact-overlap feature parity: `1441` windows, maximum absolute feature difference `0.0`.
- Exact-overlap zero-shot prediction parity: `663` test windows across three seeds, maximum probability difference `0.0`.
- All 4,851 non-behavior test windows are negative, whereas the 1,881 behavior-track windows contain all 1,177 positives.

## Conclusions

- Against Exp031 BBox53 at the same flawed protocol and fixed 0.5, one-stage gains F1 `0.0257`, PR-AUC `0.0256`, and ROC-AUC `0.0026`.
- Two-stage minus one-stage changes F1 by `-0.0307` and PR-AUC by `-0.0154`; the sequential design is not supported as the better model here.
- PIE zero-shot failure is driven by the added non-behavior negative population, not an implementation mismatch: exact-overlap features and probabilities are identical to Exp077.

This is a protocol-sensitivity diagnostic, not the clean JAAD benchmark. The flawed protocol changes both population and temporal/label semantics.
