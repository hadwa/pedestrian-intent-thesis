# Thesis result-to-artifact coverage audit

Classification: diagnostic release-readiness audit. No training, inference, threshold selection, or benchmark recomputation. All outputs are in the independent companion repository.

Scope: all 13 Appendix A mapping entries plus primary classifier/architecture/neural table rows. The current manuscript/source-map files are copied under `scope/` with hashes in `artifact_catalog.json`. Published comparison rows are citation evidence and were not externally verified.

## Coverage matrix

“Packaged” means a byte-identical copy was found through the recorded copy manifests. It does not mean runnable or publicly downloadable. Weight lists are explicit required models where resolved, not every checkpoint in an experiment folder.

| Group | Primary results present / packaged | Required weights present / packaged | Note |
|---|---|---|---|
| 01 Classifier/version controls and initial confusion figure | 2/2 / 2/2 | see limitation / see limitation | Exp012 predictions available; runner has no model serialization. No weights found in this experiment. Wider original-model availability needs verification. |
| 02 Earlier raw-pose and BIU fusion | 1/1 / 1/1 | see limitation / see limitation | Individual runs, not three-seed averages. Source probabilities reside in archive. Original checkpoint identity for each row needs verification; a later reproduced RF is not automatically the original model. |
| 03 PIE XGBoost versions and architecture | 2/2 / 2/2 | 12/12 / 0/12 | Use Exp096 base/v1/v2 and Exp116 pie_road only; union is not required for these displayed rows. |
| 04 Capped PedCMT and confusion figure | 1/1 / 1/1 | 24/24 / 24/24 | Need frozen parent plus correction branch, statistics, and inference cap .10. JAAD raw uses Exp094 P; do not replace it with an uncapped diagnostic. |
| 05 PIE MSLSTM and confusion figure | 1/1 / 1/1 | 18/18 / 0/18 | G/H/I from Exp053; J raw and K GT Action/Look from Exp095; core56 from Exp117. GT row is an annotation reference; L neural union excluded. |
| 06 Clean-JAAD MSLSTM | 2/2 / 2/2 | 12/12 / 12/12 | A/B/C/D use AP-selected weights. Exp120/121 are development history, not alternate main test evidence. |
| 07 JAAD native XGBoost | 4/4 / 4/4 | 36/36 / 0/36 | JAAD_all/A intentionally reuses Exp031 saved probabilities; original checkpoint unavailable per reuse audit. Keep clean/API-beh/API-all cohorts separate. |
| 08 PSI XGBoost and confusion figure | 1/1 / 1/1 | 12/12 / 12/12 | A/B Exp124; C/D Exp116. Frozen-feature evaluation verified; raw-video pipeline and public inputs not ready. |
| 09 Our rows in published-literature comparisons | 4/4 / 4/4 | 60/60 / 12/60 | Our rows reuse groups 03/07/08. Published third-party rows are citation-based, not local reruns; external papers were not verified in this audit. |
| 10 BIU-v2 attribution and cue table | 3/3 / 3/3 | see limitation / see limitation | Select PIE_096 and clean_vehicle_100 core56 test ALL rows; PSI 56-feature v2 block. Older v1/union attribution files are not automatically main exhibits. |
| 11 Runtime table and figure | 2/2 / 2/2 | 12/12 / 0/12 | Historical timing evidence; no new timing performed. Feature-to-prediction excludes upstream perception; road begins with ready geometry. |
| 12 Predictor complexity and latency | 5/5 / 5/5 | 54/54 / 24/54 | Inference-only serialized PedCMT sizes are explicitly unmeasured, not missing reported values. PedCMT latency uses validation; earlier audit uses test windows. |
| 13 Ordered-history trial | 2/2 / 2/2 | 3/3 / 0/3 | Exploratory/negative trial retained in thesis; not the main proposed benchmark. |

## Actionable findings

- Exp102 JAAD_all/A is intentionally prediction-only: `metrics/reuse_audit.json` states `checkpoints_available=false`. Preserve the Exp031-derived probabilities and reconstruction recipe; do not promise checkpoint replay or retrain to fill the gap.
- Exp012: the inspected runner writes predictions/metrics but has no model export. Original weights were not found in the experiment folder. Its saved predictions are an available evidence route, not proof that an original checkpoint is available.
- Earlier PIE pose-fusion rows depend on archived saved probabilities. The exact files are listed and checked in `artifact_catalog.json`. A later reproduced random-forest weight is not evidence that all original classifier weights exist.
- Existing PIE MSLSTM and many PIE/JAAD XGBoost weights are local but not yet copied into the companion package. The explicit missing-copy list is `artifact_gaps.json`; no refit is required for available weights.
- Neural and tabular model inputs remain partially unpackaged. `recorded_dependencies.csv` lists source-manifest references and availability; some manifests contain excluded arms. This is not a claim that every candidate input belongs in the final release.
- Current exhibit sources, plotting scripts and supporting confusion-matrix records are separate from experiment summaries. Missing package copies are visible in `chapter_sources.csv` and `artifact_gaps.json`. Historical table-generation scripts can describe earlier exhibits; do not run them to overwrite current thesis tables.
- PSI frozen-feature evaluation is the only independently replay-verified model route in this companion so far. Neural copies are integrity-checked only. All weights/data are still local; no public download or tagged remote exists.
- PedCMT inference-only serialized size is explicitly unmeasured in the thesis, so it is not treated as a missing claimed result.

## Exact traceability and limits

`table_row_selectors.json` stores source CSV line numbers, exact row filters and full matching records. This verifies evidence coverage, not every rounded typeset number. Figure cells, attribution values, runtime numbers and model counts are covered by source files rather than independently recomputed. Files absent from the selected package are not automatically missing from the research workspace.

## Recommended completion order (no training)

1. Preserve prediction-only evidence for Exp012, earlier PIE classifiers and JAAD_all/A, with explicit weight limitations.
2. Copy only resolved missing weights and their preprocessing contracts from the gap manifest; distinguish model parents, branches and selected seeds.
3. Package the exact current plotting/table evidence and complete portable saved-model evaluation adapters.
4. Resolve and document dataset/feature acquisition and distribution, then publish reviewed artifacts with hashes and update thesis links.

## Commands

`python /tmp/audit_thesis_coverage.py` generated this audit. The exact script is preserved as `audit.py`; it uses Python stdlib file reads, CSV/JSON parsing, SHA-256, and file copies of the scope into this new audit directory only. No historical runner is imported. Inspection commands are recorded in `inspection_commands.txt`.

## Summary

```json
{
  "appendix_groups": 13,
  "table_rows_traced": 66,
  "nonunique_or_missing_selectors": 8,
  "primary_files": 25,
  "missing_primary_files": 0,
  "audited_artifact_paths": 289,
  "missing_paths": 7,
  "not_packaged_paths": 152,
  "scope_changed_since_snapshot": false,
  "models_trained": 0
}
```
