# Exp098 Final Report

Retrospective matched clean-JAAD ablation; not untouched confirmation.
16 observed frames, TTE 30--60, fixed 0.5; identical cohorts and XGBoost recipe.
No vehicle-action, ego, road or extra temporal inputs. No pose re-extraction.

| Arm | Features | F1 mean +/- SD | PR-AUC mean +/- SD | ROC-AUC mean +/- SD |
|---|---:|---:|---:|---:|
| A | 53 | 0.703798 +/- 0.005861 | 0.677426 +/- 0.003333 | 0.720507 +/- 0.003165 |
| B | 125 | 0.712575 +/- 0.002124 | 0.705516 +/- 0.003804 | 0.740235 +/- 0.002727 |
| C | 116 | 0.710638 +/- 0.003814 | 0.680854 +/- 0.006016 | 0.708192 +/- 0.001661 |
| D | 188 | 0.712179 +/- 0.003357 | 0.690850 +/- 0.009392 | 0.730667 +/- 0.004984 |

## C (v2) minus B (v1)

- f1: mean delta -0.001936; favorable seeds 1/3.
- pr_auc: mean delta -0.024662; favorable seeds 0/3.
- roc_auc: mean delta -0.032043; favorable seeds 0/3.

Track-bootstrap uncertainty is in metrics/ensemble_track_bootstrap.csv; it uses
seed-ensemble probabilities, not mean per-seed metrics. Intervals are descriptive
and are not multiplicity-adjusted. Per-seed evidence and B-to-C error transitions
are reported separately. Do not infer universal version superiority.

Source arrays and historical code were reused read-only. Twelve new train-only
fits used cached features. All twelve validation models were frozen before test
access, and saved models were evaluated without refitting. Existing 72-D v1
validity counters and 63-D v2 direction metadata remain intact. See DESIGN.md
and metrics/freeze.json for the exact experimental and provenance boundaries.
