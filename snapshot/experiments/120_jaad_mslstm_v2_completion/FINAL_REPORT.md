# Missing neural comparisons: consolidated completion

Matched comparisons are within each family only. PIE and clean JAAD have distinct datasets/non-pose inputs. Direct-token and capped families use different integration/training designs. No API_beh/API_all expansion. Existing PIE MSLSTM and direct PedCMT runs are reused, not retrained.

| Family | Representation | F1 mean +/- SD | AP mean +/- SD | ROC mean +/- SD | Accuracy |
| --- | --- | --- | --- | --- | --- |
| mslstm_pie | baseline | 0.8099 +/- 0.0085 | 0.8537 +/- 0.0196 | 0.9354 +/- 0.0073 | 0.9018 |
| mslstm_pie | raw | 0.8227 +/- 0.0089 | 0.8549 +/- 0.0156 | 0.9373 +/- 0.0053 | 0.9078 |
| mslstm_pie | core56 | 0.8122 +/- 0.0063 | 0.8514 +/- 0.0109 | 0.9321 +/- 0.0046 | 0.9024 |
| mslstm_pie | full63 | 0.8289 +/- 0.0053 | 0.8599 +/- 0.0127 | 0.9384 +/- 0.0070 | 0.9093 |
| pedcmt_pie_direct | baseline | 0.7926 +/- 0.0094 | 0.8584 +/- 0.0306 | 0.9192 +/- 0.0127 | 0.8887 |
| pedcmt_pie_direct | raw | 0.7746 +/- 0.0270 | 0.8217 +/- 0.0367 | 0.9140 +/- 0.0109 | 0.8757 |
| pedcmt_pie_direct | core56 | 0.8211 +/- 0.0036 | 0.8492 +/- 0.0119 | 0.9303 +/- 0.0044 | 0.8971 |
| pedcmt_pie_direct | full63 | 0.8097 +/- 0.0403 | 0.7895 +/- 0.1057 | 0.9260 +/- 0.0301 | 0.8850 |
| pedcmt_jaad_direct | baseline | 0.6734 +/- 0.0566 | 0.6256 +/- 0.0207 | 0.6891 +/- 0.0246 | 0.6329 |
| pedcmt_jaad_direct | raw | 0.6767 +/- 0.0588 | 0.6061 +/- 0.0815 | 0.6639 +/- 0.0488 | 0.6376 |
| pedcmt_jaad_direct | core56 | 0.5911 +/- 0.0496 | 0.6327 +/- 0.0652 | 0.6699 +/- 0.0665 | 0.5683 |
| pedcmt_jaad_direct | full63 | 0.6373 +/- 0.0270 | 0.6610 +/- 0.0630 | 0.6879 +/- 0.0346 | 0.5734 |
| pedcmt_jaad_capped | baseline | 0.6734 +/- 0.0566 | 0.6256 +/- 0.0207 | 0.6891 +/- 0.0246 | 0.6329 |
| pedcmt_jaad_capped | core56 | 0.6913 +/- 0.0446 | 0.6574 +/- 0.0344 | 0.7132 +/- 0.0325 | 0.6480 |
| pedcmt_jaad_capped | full63 | 0.6981 +/- 0.0360 | 0.6700 +/- 0.0368 | 0.7259 +/- 0.0365 | 0.6542 |
| pedcmt_jaad_capped | raw | 0.6830 +/- 0.0474 | 0.6022 +/- 0.0027 | 0.6856 +/- 0.0215 | 0.6394 |
| pedcmt_pie_capped | baseline | 0.7926 +/- 0.0094 | 0.8584 +/- 0.0306 | 0.9192 +/- 0.0127 | 0.8887 |
| pedcmt_pie_capped | core56 | 0.7946 +/- 0.0090 | 0.8606 +/- 0.0332 | 0.9217 +/- 0.0120 | 0.8898 |
| pedcmt_pie_capped | full63 | 0.7955 +/- 0.0108 | 0.8665 +/- 0.0251 | 0.9228 +/- 0.0112 | 0.8905 |
| pedcmt_pie_capped | raw | 0.7917 +/- 0.0101 | 0.8580 +/- 0.0350 | 0.9204 +/- 0.0110 | 0.8884 |
| mslstm_jaad | baseline | 0.7557 +/- 0.0068 | 0.8093 +/- 0.0100 | 0.8248 +/- 0.0084 | 0.7104 |
| mslstm_jaad | raw | 0.7585 +/- 0.0119 | 0.7985 +/- 0.0139 | 0.8191 +/- 0.0048 | 0.7171 |
| mslstm_jaad | core56 | 0.7301 +/- 0.0327 | 0.7767 +/- 0.0566 | 0.7981 +/- 0.0392 | 0.6600 |
| mslstm_jaad | full63 | 0.7222 +/- 0.0295 | 0.7757 +/- 0.0490 | 0.8008 +/- 0.0423 | 0.6433 |

## Clean-JAAD MSLSTM paired evidence

| Comparison | Mean F1 delta | Mean AP delta | Mean ROC delta | Positive seeds F1/AP/ROC |
| --- | --- | --- | --- | --- |
| B minus A | +0.0028 | -0.0108 | -0.0057 | 2/3 / 0/3 / 0/3 |
| C minus A | -0.0256 | -0.0326 | -0.0267 | 0/3 / 0/3 / 0/3 |
| D minus A | -0.0335 | -0.0335 | -0.0240 | 0/3 / 0/3 / 0/3 |
| C minus B | -0.0284 | -0.0218 | -0.0210 | 0/3 / 1/3 / 1/3 |
| D minus B | -0.0363 | -0.0228 | -0.0183 | 0/3 / 1/3 / 2/3 |
| D minus C | -0.0079 | -0.0010 | +0.0027 | 0/3 / 2/3 / 2/3 |

A=baseline, B=raw pose, C=core56, D=full63. These descriptive seed comparisons do not establish statistical significance. No result was suppressed by a validation gate. Negative/neutral direct-integration evidence remains visible rather than being replaced by capped results.

## Efficiency

The clean-JAAD MSLSTM result is negative for the tested BIU integrations: both forms reduce F1/AP/ROC relative to the baseline in every seed. Mean FN falls from 48.0 (baseline) to 35.0 (core56) and 28.0 (full63), but mean FP rises from 386.7 to 475.3 and 507.3, respectively. This is not an overall error reduction.

| Arm | Total parameters | Mean CPU predictor ms |
| --- | --- | --- |
| A: baseline | 4,590,087 | 5.311 |
| B: raw pose | 6,830,857 | 8.282 |
| C: core56 | 4,663,432 | 5.449 |
| D: full63 | 4,664,328 | 5.622 |

The BIU arms are smaller and faster than raw pose under this measurement, but the predictive loss prevents describing them as performance-equivalent replacements.

Ready-input one-core CPU measurements and parameter counts: `metrics/efficiency.csv`, `metrics/parameter_counts.csv`, `metrics/latency_scope.json`. This excludes upstream perception and BIU/BBox construction and is not an end-to-end comparison. A raw-pose/BIU branch-cost difference is not proof of accuracy causality.

## Provenance and limitations

Exp119 FINAL_REPORT.md details capped v2-only findings and reused JAAD raw controls. Exp120 DESIGN.md specifies the approved vehicle-state/no-acceleration adaptation. Exact old-family rows are from Exp117 summary artifacts; metrics/all_neural_families.csv retains their source labels. Existing and new tests are retrospective matched evidence on exposed cohorts, not pristine confirmations or exact published-paper reproduction. No thesis edits, additional model search, commit or push.
