# Exp083 Final Report

## Result

Exp083 freshly trained the frozen 242-D Exp082-D XGBoost design on separate
official PCPA-style JAAD_beh and JAAD_all cohorts. All results below are the
mean and sample standard deviation across seeds 42/43/44 at fixed threshold
0.5.

| Cohort | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| JAAD_beh | 0.678185 ± 0.005325 | 0.682485 ± 0.003091 | 0.908241 ± 0.005168 | 0.779342 ± 0.003767 | 0.672254 ± 0.007168 | 0.746922 ± 0.005827 |
| JAAD_all | 0.891414 ± 0.001417 | 0.648088 ± 0.003907 | 0.829227 ± 0.002549 | 0.727548 ± 0.002898 | 0.945511 ± 0.000240 | 0.811555 ± 0.002694 |

Per-seed values, calibration, confusion counts, and validation-selected
operating points are in `metrics/per_seed_results.csv`,
`metrics/summary_results.csv`, and the two threshold-specific tables.

## PCAP-SL Comparison

PCAP-SL reports JAAD_beh F1 0.801 and JAAD_all F1 0.841. Exp083 is lower by
0.0217 and 0.1135 F1 respectively. On JAAD_all, Exp083 reports higher accuracy
(0.891 versus 0.852) and ROC-AUC (0.946 versus 0.852), but lower precision and
recall at fixed 0.5. PCAP-SL does not report PR-AUC or its classification
threshold, so its F1 is not a threshold-matched comparison.

This is published-protocol-aligned evidence, not proven row-identical evidence:
PCAP-SL does not release sample IDs, exact window counts, or its threshold.
The exact literature table is `metrics/literature_comparison.csv`.

## Error Confidence

JAAD_beh averaged 497.3 FP and 108.0 FN per seed, including 300.0 FP at
probability at least 0.9 and 20.7 FN at probability at most 0.1. JAAD_all
averaged 530.0 FP and 201.0 FN, including 138.3 high-confidence FP and 47.3
high-confidence FN. The broader all-cohort reduces high-confidence FP relative
to its much larger negative population, but substantial confident errors
remain; this result does not solve calibration.

## Efficiency And Limits

The six model files total about 2.2 MB. Cached-feature XGBoost prediction took
roughly 0.0014--0.0016 ms per window in batched local measurement. This excludes
ViTPose, TwinLiteNet+, optical flow, and feature construction and must not be
presented as end-to-end latency. Fresh TwinLite/flow extraction for 6,072 test
windows took 39:09 on the RTX 3070 Laptop GPU plus CPU optical flow.

JAAD_all adds official non-behavior tracks as negatives and has much lower
positive prevalence than JAAD_beh. The official API-style endpoint population
also includes observed-crossing and unverified-onset windows. Therefore these
absolute values must not be mixed with the clean event-aligned JAAD benchmark.
See `LABEL_AND_COHORT_AUDIT.md`.

## Conclusion

The frozen compact geometric/road model is highly ranking-effective on the
broader JAAD_all population (PR-AUC 0.812, ROC-AUC 0.946) but its fixed-0.5 F1
does not match PCAP-SL. On JAAD_beh it approaches PCAP-SL's reported F1 while
remaining below it. The comparison supports the model as a compact independent
baseline, not as a superior reproduction of PCAP-SL.
