# Exp124: matched PSI lightweight ablation

Complete. Six new fits supply BBox-only and BBox+BIU56 on the extended PSI
training cohort. Temporal+ego and road arms reuse authenticated Exp116 models
and predictions. All four rows now share populations,recipe,seeds and threshold.

Results: `FINAL_REPORT.md`; per-seed/meanSD: `metrics/`; source hashes:
`metrics/source_hashes.json`; named inputs: `metrics/feature_contract.json`.
`TRAINED.json` freezes all six new models before test; `EXPERIMENT_COMPLETE.json`
confirms completed evaluation and source/probability checks. Exact command and
protocol are in the report and `DESIGN.md`. No original results replaced.
