# Start here

Completed saved-model diagnostic: 19 setting/version combinations, each evaluated on validation and test across three seeds (114 checks). Six settings cover PIE, clean JAAD with and without vehicle state, JAAD_beh, JAAD_all and PSI. No fitting, cue removal, threshold changes or manuscript edits.

- Printable full tables: `Matched_BIU_Cue_Rankings.pdf` (21 pages; two explanatory pages plus 19 model/version pages).
- Sortable complete comparison: `review_table.csv`. Importance rank and support rank use validation. Signed test columns are diagnostic follow-up only.
- Complete numeric summaries: `rankings.csv`, `summary.csv`, `per_seed.csv`. These include per-class and error-group rows, seed variation, support/opposition magnitude and equal-track means.
- Methods, boundaries, selected high/low validation-support cues, exact commands: `REPORT.md`. Detailed source paths and hashes: `sources.json`.

## Findings to review

These examples use equal-class mean truth-aligned SHAP contributions, in log-odds. They do not measure accuracy gains or causal value of cue removal.

| Setting/version | Cue | Validation balanced net | Test balanced net | What it supports |
|---|---|---:|---:|---|
| PIE Exp096, historical Combined | v2 shoulder-center horizontal motion | +0.2084 | +0.2870 | Strong supportive model contribution in both splits; test toward-label fraction 85.97%, not 85.97% classification accuracy. |
| Clean JAAD Exp098, full union | v1 normalized shoulder width | +0.1922 | +0.1519 | Evidence against calling this cue generally misleading in this model. |
| Clean JAAD Exp100, historical Combined | v1 normalized shoulder width | +0.1664 | +0.1357 | Supportive averages also persist with vehicle-state context. |
| JAAD_all Exp102, full union | v2 knee-center horizontal motion | -0.0107 | -0.0041 | Adverse class-balanced net in both splits and all three seeds; a scrutiny candidate, not an established deletion benefit. |
| JAAD_all Exp102, full union | v2 torso-motion magnitude | -0.0133 | -0.0087 | Same adverse sign pattern across splits/seeds. |
| PSI Exp113, v2 full | Nose-shoulder orientation | +0.2821 | +0.2452 | Strong supportive contribution on PSI's different native intent task. |

Normalized v1 shoulder width differs from full-v2 shoulder-width metadata. Historical Combined108 differs from namespaced union135. The same cue can have different contributions when other predictors or inputs change; the report does not infer culture or pose accuracy as the cause.

Percentages of windows pushed toward/away and mean signed magnitude answer different questions. A cue can push slightly the wrong way on many windows but strongly toward the label on fewer windows. Neither statistic alone is a keep/remove verdict. Validation-versus-test, positive-versus-negative and seed patterns should be reviewed together.

## Verification

Reused 18 existing full-union SHAP arrays; recomputed exact SHAP for 96 saved model/split/seed combinations where the needed per-window cue values were absent. All saved probabilities matched within 2.99e-8, and maximum SHAP additivity error was 8.61e-6 log-odds. Independent checks reconstructed 15,664 cue/subset rows, verified 748 balanced-ranking rows, and compared corresponding net values against the older corrected standalone audit. Sources remained unchanged.

PDF verification command (system Python provides PyMuPDF):

```bash
python3 experiments/matched_biu_cue_rankings_2026_09_12/verify_pdf.py
```

Incomplete attempts are explicitly isolated and excluded: `attempt_001_source_path_error/` (incorrect input path) and `report_attempt_001_missing_tabulate/` (optional Markdown dependency, removed in final reporting). They did not alter any benchmark results.
