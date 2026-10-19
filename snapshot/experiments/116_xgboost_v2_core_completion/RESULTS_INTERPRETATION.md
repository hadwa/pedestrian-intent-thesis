# What the completed 56-D tests establish

All 11 predeclared configurations were trained with seeds 42/43/44 and evaluated
at 0.5 after every model was frozen. Independent verification passed. No existing
benchmark results or neural models were changed.

## Direct core56 versus full63

| BBox + BIU task | Version | Mean F1 | Mean AP | Mean ROC-AUC |
|---|---|---:|---:|---:|
| Clean JAAD | New core56 | 0.705147 | 0.674278 | 0.707973 |
| Clean JAAD | Saved full63, Exp098 C | 0.710638 | 0.680854 | 0.708192 |
| PSI native intent | New core56 | 0.763880 | 0.862163 | 0.660432 |
| PSI native intent | Saved full63, Exp113 C | 0.759333 | 0.863645 | 0.660694 |

Clean JAAD does not favor removing the seven features in this simple model:
F1 declines in all three seeds, with a mean decline of 0.005491; AP declines by
0.006576 on average. This is descriptive paired evidence, not a significance
claim. PSI is close: core56 gains 0.004547 F1 but loses 0.001482 AP and 0.000262
ROC-AUC. It would be incorrect to call either version universally superior.

## Extended models: core56 versus historical Combined

These compare different static BIU representations, NOT full63 versus core56.
All parent temporal and contextual features remain, and total input size falls
by 52 columns. The retained temporal features include v2-derived and pose-quality
summaries; the complete extended predictor is not literally a 56-input model.

- PIE with roads: mean F1 rises from 0.879739 to 0.882893, AP from 0.927432 to
  0.930437, ROC-AUC from 0.972513 to 0.973090. These gains are small; F1 improves
  in all seeds, AP and ROC-AUC in two of three.
- PSI with ego: F1/AP/ROC-AUC improve in all three seeds. Mean AP increases by
  0.022833 and ROC-AUC by 0.031124; F1 gains a smaller 0.002643.
- PSI with ego and roads: all three metrics improve in all three seeds. Mean
  F1/AP/ROC-AUC change from 0.817304/0.849057/0.666407 to
  0.824709/0.862200/0.690414. This is evidence in favor of the smaller static
  representation within this particular matched extended PSI predictor.
- Clean JAAD road models: AP improves in every seed, but F1 and ROC-AUC decline
  in every seed, both with and without vehicle state. This is a metric trade-off.
- JAAD_beh road models: AP and ROC-AUC improve in every seed, while F1 declines
  in every seed. With vehicle state, AP gains 0.033255 and ROC-AUC gains 0.025985,
  while F1 declines 0.005623. Ranking improvement is not a fixed-0.5 F1 improvement.
- JAAD_all: mean differences remain small. With road and vehicle state, F1
  changes by -0.000579, AP by +0.002040 and ROC-AUC by -0.000289.

These observations do not establish a causal explanation for why individual
errors change. No new calibration, threshold adjustment, feature redesign or
selection followed test evaluation.

## What was not repeated

Existing standalone core56 results remain available in their original folders:

| Already completed scope | Existing source | Reuse status |
|---|---|---|
| PIE BBox + core56 | ../012_pie_biu_v2_30_60_benchmark/ | Historical results retained; no duplicate fit |
| PIE extended core56, without road | ../096_pie_exp072_matched_biu_ablation/ | Existing matched core56 pathway; no duplicate fit |
| Clean JAAD temporal + vehicle-state core56 | ../100_jaad_exp099_biu_pathways/ | Existing matched pathway; static cache reused here |
| JAAD_beh/all BBox + core56 | ../102_jaad_api_matched_biu/ | Core/full comparison already exists; no duplicate fit |

This completion does not supply missing neural core56 controls. Those remain a
separate user decision. It also does not supply an unmeasured standalone full63
extended arm or a new v1+core56 concatenation.

## Scope and artifacts

PSI uses 15 observed frames and an intent-vote label at the next, 16th frame,
not the PIE crossing-event task. Its positive test prevalence is high; use the
parent baseline context rather than interpreting F1 alone. Simple and extended
PSI retain different parent training populations and must not be treated as an
otherwise-identical context-only comparison.

JAAD_all retains the parent's two nonconsecutive training windows. All API test
windows are consecutive. Clean/API cohorts are not pooled. Previously exposed
test cohorts make these retrospective completion ablations, not untouched
external confirmations.

- Complete new and parent per-seed results: `metrics/per_seed_results.csv`.
- Means and sample SD: `metrics/summary_results.csv`.
- Paired deltas and error transitions: `metrics/paired_deltas.csv`,
  `metrics/error_transitions.csv`.
- Exact source paths and SHA-256: `metrics/all_sources.json`.
- Every input name: `cache/<case>/<split>/columns.json`.
- Fit times: `metrics/fit_times.csv` (25.97 seconds summed over the 33 fits;
  excludes preparation, audits, cached-pose BIU recovery and final verification).
- Verification: `INDEPENDENT_VERIFICATION.json` and `EXPERIMENT_COMPLETE.json`.

The strongest new positive finding is the consistent extended-PSI improvement
with the smaller core56 static pathway. The results do not justify relabeling
all historical v2 experiments as 56-D or declaring 56-D the best version in
every JAAD configuration.
