# Thesis result-to-artifact coverage

Use the [current thesis index](thesis/14_RESULT_SOURCE_MAP.md) for the final mapping, including the illustrative PIE scene added after the coverage audit.

The [coverage audit](audit/result_artifact_coverage_final/REPORT.md) checked the then-current 13 result groups: 25 primary result files had byte-identical copies, and 66 table rows resolved uniquely to CSV records. This is evidence coverage, not a complete numerical audit of the final manuscript.

All 117 audited existing weight files are now locally packaged and exercised by saved-model replay. See [evaluation commands and limitations](SAVED_MODEL_EVALUATION.md) and the [completion report](audit/saved_evaluation_completion/REPORT.md). No models were retrained. Early prediction-only results remain explicitly qualified.

Code and reports are published in this repository. Weights and frozen input bundles remain local; public downloads, raw-data reconstruction and fresh-install validation remain outstanding. Earlier audit reports preserve historical packaging states and may contain superseded missing-file counts.
