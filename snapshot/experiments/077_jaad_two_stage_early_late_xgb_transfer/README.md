# Exp077: JAAD Two-Stage Transfer and Native Training

Status: complete.

This experiment evaluates both the one-stage and 8+8 two-stage XGBoost in two
separate settings: frozen PIE-to-JAAD zero-shot transfer and fresh JAAD-native
training. It uses the shared no-ego 182-D portable representation and never
tunes a threshold on JAAD test.

The exact-frame extended-plus-supplemental pose lineage differs from the older
archived BIU-v1 table's nearest-frame repair in 30 documented windows. This is
audited explicitly in `DESIGN.md`; samples are not dropped or substituted.

## Final Mean Results

| Setting | Model | F1 | PR-AUC | ROC-AUC | High-conf FP | High-conf FN |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| PIE-to-JAAD zero-shot | One-stage | 0.6166 | 0.6930 | 0.7169 | 81.7 | 155.3 |
| PIE-to-JAAD zero-shot | Two-stage | 0.6304 | 0.6869 | 0.7149 | 78.0 | 82.3 |
| JAAD-native | One-stage | 0.7073 | 0.7133 | 0.7529 | 247.3 | 33.7 |
| JAAD-native | Two-stage | 0.7039 | 0.7208 | 0.7449 | 226.3 | 23.3 |

The two-stage design improved zero-shot fixed-0.5 F1 and native PR-AUC while
substantially reducing high-confidence errors, but it was not an across-metric
replacement for the one-stage model. See `FINAL_REPORT.md`.
