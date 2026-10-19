# Exp094 Results

## Status

Exp094 is a retrospective matched control conceived after the Exp092 JAAD test result was already known. It is not pristine confirmation. The frozen setup was evaluated without a performance gate, and this closes the planned neural-control line regardless of outcome.

## Mean fixed-threshold results

| Arm | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.6329 | 0.5850 | 0.7991 | 0.6734 | 0.6891 | 0.6256 |
| P10 | 0.6394 | 0.5888 | 0.8176 | 0.6830 | 0.6856 | 0.6022 |
| U10 | 0.6529 | 0.5990 | 0.8366 | 0.6967 | 0.7225 | 0.6633 |

## Primary interpretation

U10 satisfies the prespecified mean-F1 criterion: it exceeds both A and P10.

U10 improves F1 over A in 2/3 seeds and over P10 in 2/3 seeds.

The three-seed ensemble track-cluster bootstrap gives:

| Comparison | Metric | Observed delta | 95% interval | Excludes zero |
|---|---|---:|---:|:---:|
| P10_minus_A | f1 | 0.0028 | [-0.0110, 0.0201] | no |
| P10_minus_A | roc_auc | 0.0016 | [-0.0087, 0.0118] | no |
| P10_minus_A | pr_auc | -0.0017 | [-0.0103, 0.0073] | no |
| U10_minus_A | f1 | 0.0074 | [-0.0023, 0.0224] | no |
| U10_minus_A | roc_auc | 0.0320 | [0.0069, 0.0596] | yes |
| U10_minus_A | pr_auc | 0.0640 | [0.0236, 0.1007] | yes |
| U10_minus_P10 | f1 | 0.0046 | [-0.0033, 0.0152] | no |
| U10_minus_P10 | roc_auc | 0.0304 | [0.0060, 0.0571] | yes |
| U10_minus_P10 | pr_auc | 0.0657 | [0.0241, 0.1033] | yes |

## Fixed-0.5 error transitions

| Comparison | Wrong to correct | Correct to wrong | FP to TN | FN to TP | TN to FP | TP to FN |
|---|---:|---:|---:|---:|---:|---:|
| U10_minus_A | 115 | 25 | 19 | 96 | 10 | 15 |
| P10_minus_A | 45 | 16 | 4 | 41 | 15 | 1 |
| U10_minus_P10 | 90 | 29 | 25 | 65 | 5 | 24 |

## Limits

Raw pose and BIUs originate from the same pose measurements; this comparison concerns representation and fusion behavior, not access to different source information. No result here establishes that BIUs universally replace raw pose or that a 0.10 cap is universally optimal. The result is specific to this frozen parent, JAAD cohort, observation/TTE protocol, and fixed-0.5 operating point.
