# Exp100 Final Report

Clean JAAD; 16 observations, TTE30--60, fixed0.5, seeds42/43/44.
All arms retain the same BBox and categorical vehicle-action inputs.
v1/v2 use six new fits; combined reuses Exp099 models and probabilities.

| Pathway | Features | F1 mean +/- SD | PR-AUC mean +/- SD | ROC-AUC mean +/- SD |
|---|---:|---:|---:|---:|
| v1 | 149 | 0.720232 +/- 0.002121 | 0.751196 +/- 0.005508 | 0.772816 +/- 0.001762 |
| v2 | 145 | 0.725254 +/- 0.001950 | 0.730421 +/- 0.006032 | 0.764966 +/- 0.002559 |
| combined | 197 | 0.719125 +/- 0.003319 | 0.755670 +/- 0.005050 | 0.777743 +/- 0.002805 |

## v2_minus_v1

- f1: +0.005022; favorable seeds 3/3.
- pr_auc: -0.020776; favorable seeds 0/3.
- roc_auc: -0.007849; favorable seeds 0/3.

## combined_minus_v1

- f1: -0.001107; favorable seeds 1/3.
- pr_auc: +0.004474; favorable seeds 2/3.
- roc_auc: +0.004928; favorable seeds 3/3.

## combined_minus_v2

- f1: -0.006129; favorable seeds 0/3.
- pr_auc: +0.025249; favorable seeds 3/3.
- roc_auc: +0.012777; favorable seeds 3/3.

## Interpretation limits

V2 is the complete historical core-56, not full-63. Its 20 overlapping
feature names were reconstructed from the original cached pose measurements,
not filled with v1 values from the deduplicated parent. All 36 unique v2
and 72 v1 cached columns passed numeric parity checks. No pose estimator ran.

This is an Exp096-style pathway comparison: v2 carries twelve additional
historical temporal features; both arms retain three existing pose-quality
summaries. Vehicle input is annotation-derived categorical state, not physical
speed. The existing normalization and direction fallback were unchanged.

Previously exposed test cohort: retrospective matched ablation, not pristine
confirmation. Track-bootstrap intervals are descriptive, based on seed-ensemble
probabilities and not multiplicity adjusted. Do not claim universal version
dominance or silently equate this pathway comparison with Exp098.
