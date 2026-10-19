# Exp092 Results

## Outcome

**Successful capped-BIU correction.** The predeclared validation gate passed,
after which the exact frozen 10% rule was applied once to the 1,501-window
final JAAD cohort. No model was retrained.

## Validation Gate

Means across seeds 42, 43 and 44 at fixed threshold 0.5:

| Variant | Accuracy | F1 | ROC-AUC | PR-AUC | High-confidence errors |
|---|---:|---:|---:|---:|---:|
| A parent | 0.7725 | 0.8250 | 0.7846 | 0.8120 | 0.0 |
| N10 | 0.7713 | 0.8237 | 0.8218 | 0.8765 | 0.0 |

All four frozen criteria passed before test access.

## Final JAAD Results

Means +/- sample standard deviation across seeds:

| Variant | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FP | FN | High-confidence errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A parent | 0.6329 +/- 0.0374 | 0.5850 +/- 0.0196 | 0.7991 +/- 0.1256 | 0.6734 +/- 0.0566 | 0.6891 +/- 0.0246 | 0.6256 +/- 0.0207 | 406.3 | 144.7 | 1.33 |
| N10 | **0.6529 +/- 0.0193** | **0.5990 +/- 0.0098** | **0.8366 +/- 0.0910** | **0.6967 +/- 0.0338** | **0.7225 +/- 0.0387** | **0.6633 +/- 0.0314** | **403.3** | **117.7** | **1.00** |

Mean deltas N10 minus A:

- Accuracy: `+0.0200`
- Precision: `+0.0140`
- Recall: `+0.0375`
- F1: `+0.0233`
- ROC-AUC: `+0.0335`
- PR-AUC: `+0.0377`
- False positives: `-3.0`
- False negatives: `-27.0`

ROC-AUC and PR-AUC improved in all three seeds. Fixed-0.5 F1 and accuracy
improved in seeds 42 and 43 and were exactly unchanged in seed 44.

## Error Transitions

Across all three seed predictions, N10 changed 115 parent mistakes to correct
decisions and changed 25 parent correct decisions to mistakes. This comprised:

- 19 FP to TN and 96 FN to TP corrections;
- 10 TN to FP and 15 TP to FN harms.

Seed 44 had no threshold-level transitions, but its ROC-AUC and PR-AUC still
improved by `0.0451` and `0.0481`, respectively.

## Interpretation

The BIU union was useful when its influence was constrained. Full-strength
Exp091 correction degraded fixed-threshold behavior, whereas the frozen 10%
scale preserved the parent and improved both ranking and classification on the
final cohort. This supports a conditional thesis claim: task-specific BIU
features can improve a PedCMT predictor, but the fusion must prevent the
auxiliary behavioral pathway from overpowering the stronger anchor.

The 10% value was chosen using validation-only analysis and is not claimed as a
universal modality weight. The final cohort had been used by earlier separate
experiments, but neither its labels nor Exp092 test results informed the 10%
selection or gate.

## Traceability

- `metrics/lineage_audit.json`
- `metrics/validation_gate_report.json`
- `metrics/validation_per_seed_results.csv`
- `metrics/per_seed_results.csv`
- `metrics/summary_results.csv`
- `metrics/paired_deltas.csv`
- `metrics/error_transitions_fixed_0p5.csv`
- `predictions/validation_N10_seed*.csv`
- `predictions/test_N10_seed*.csv`
- `EXPERIMENT_COMPLETE.json`
