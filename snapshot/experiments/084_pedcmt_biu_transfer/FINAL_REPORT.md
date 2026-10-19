# Exp084 Final Report

## Status and protocol

Exp084 completed all 15 frozen GPU runs and one gated set03 evaluation. It used
PIE, 16 consecutive observed frames, TTE30--60, train sets 01/02/06, validation
sets 04/05, final test set03, seeds 42/43/44, and fixed threshold 0.5.

The experiment is a controlled PedCMT architecture transfer, not a reproduction
of PedCMT's reported random-split numbers. Set03 had prior exposure in independent
project work; all Exp084 choices were frozen before this evaluation.

Independent output checks verified:

- all 15 prediction files contain the same 3,760 ordered sequence IDs and labels;
- saved F1, ROC-AUC and PR-AUC recompute exactly from saved probabilities;
- every checkpoint is the first epoch attaining minimum validation classification
  loss;
- final evaluation started only after all 15 checkpoints existed.

## Final set03 results

Mean +/- sample standard deviation over three seeds:

| Variant | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| A BBox + ego | 0.8887 +/- 0.0084 | 0.7921 +/- 0.0380 | 0.7947 +/- 0.0250 | 0.7926 +/- 0.0094 | 0.9192 +/- 0.0127 | 0.8584 +/- 0.0306 |
| B A + raw pose | 0.8757 +/- 0.0174 | 0.7545 +/- 0.0459 | 0.7977 +/- 0.0370 | 0.7746 +/- 0.0270 | 0.9140 +/- 0.0109 | 0.8217 +/- 0.0367 |
| C A + BIU-v1 | 0.8605 +/- 0.0241 | 0.7074 +/- 0.0624 | 0.8325 +/- 0.0413 | 0.7625 +/- 0.0229 | 0.9079 +/- 0.0026 | 0.7917 +/- 0.0260 |
| D A + BIU-v2 | 0.8850 +/- 0.0327 | 0.7358 +/- 0.0727 | 0.9045 +/- 0.0141 | 0.8097 +/- 0.0403 | 0.9260 +/- 0.0301 | 0.7895 +/- 0.1057 |
| E A + v1/v2 union | **0.8928 +/- 0.0053** | 0.7598 +/- 0.0254 | 0.8783 +/- 0.0256 | **0.8142 +/- 0.0031** | **0.9342 +/- 0.0082** | 0.8518 +/- 0.0122 |

Variant A retained the best mean precision and PR-AUC. Variant D retained the
best recall. Variant E had the best accuracy, F1 and ROC-AUC and the lowest F1
seed variability.

Mean deltas:

| Comparison | Delta F1 | Delta ROC-AUC | Delta PR-AUC |
|---|---:|---:|---:|
| A -> B | -0.0180 | -0.0052 | -0.0367 |
| A -> C | -0.0301 | -0.0113 | -0.0667 |
| A -> D | +0.0171 | +0.0068 | -0.0689 |
| A -> E | **+0.0216** | **+0.0151** | -0.0066 |
| B -> E | **+0.0396** | **+0.0203** | **+0.0301** |
| C -> E | +0.0517 | +0.0263 | +0.0601 |
| D -> E | +0.0045 | +0.0083 | +0.0623 |

E improved F1 and ROC-AUC over A and B in every seed. E improved PR-AUC over B
in two of three seeds, but improved PR-AUC over A in only one of three seeds.
Therefore the PR-AUC result is not directionally consistent.

## Errors and efficiency

Mean high-confidence errors per seed (`FP p>=0.9`, `FN p<=0.1`):

| Variant | High-conf FP | High-conf FN | Combined |
|---|---:|---:|---:|
| A | 57.3 | 95.0 | 152.3 |
| B | 46.0 | 94.3 | 140.3 |
| C | 55.7 | 86.0 | 141.7 |
| D | **32.0** | **67.3** | **99.3** |
| E | 88.3 | 69.7 | 158.0 |

D produced the fewest mean high-confidence errors, but its PR-AUC was unstable
across seeds. E's improved F1 did not reduce combined high-confidence errors; its
FP confidence behavior requires calibration/error analysis before deployment,
not a post-hoc change to Exp084.

| Variant | Parameters | Mean predictor latency, batch 1 |
|---|---:|---:|
| A | 1,834,416 | 5.82 ms |
| B | 2,368,944 | 7.52 ms |
| C | 1,860,528 | 5.42 ms |
| D | 1,859,376 | 5.49 ms |
| E | 1,868,592 | 5.60 ms |

Relative to B, E has 21.12% fewer total parameters and 93.61% less incremental
pathway parameter overhead (34,176 versus 534,528 above A). These latencies cover
the predictor from already-computed inputs; ViTPose extraction is excluded.

All training runs together took 1,181.8 seconds of measured GPU run time. The
models stopped after 27--69 epochs under the frozen patience rule.

## Conclusion

This is partial BIU architecture-transfer evidence. The lossless BIU-v1/v2 union
was substantially better and smaller than the raw-pose extension, and improved
PedCMT's fixed-0.5 F1 and ROC-AUC consistently. It did not improve baseline
PR-AUC: mean PR-AUC fell from 0.8584 to 0.8518, outside the frozen 0.005 tolerance
by 0.0016. BIU-v1 alone was negative; BIU-v2 raised F1/recall but was unstable in
ranking performance.

Exp084 therefore does not establish a new overall best model. In particular, its
best PR-AUC remains below the established Exp012/Exp072 tree-based BIU+BBox
results. The defensible conclusion is that explicit BIU abstraction transfers
more effectively and compactly than raw pose into this PedCMT reconstruction for
F1/ROC ranking, while preserving strong PR-AUC and controlling confidence errors
remain unresolved.

Exact per-seed values and probabilities are in `metrics/per_seed_results.csv` and
`predictions/test_<variant>_seed<seed>.csv`.
