# Exp102 Final Report

API-style Exp083 JAAD cohorts, fixed0.5, seeds42/43/44.
A=BBox; B=+v1-72; C=+v2-core56; D=+v2-full63; E=+complete v1 and full63.
27 new250-tree XGBoost fits; three Exp031 JAAD_all/A prediction sets reused.

| Cohort | Arm | Features | Accuracy | F1 mean +/- SD | PR-AUC mean +/- SD | ROC-AUC mean +/- SD |
|---|---|---:|---:|---:|---:|---:|
| jaad_all | A | 53 | 0.850168 | 0.660318 +/- 0.004000 | 0.749522 +/- 0.005863 | 0.927780 +/- 0.000691 |
| jaad_all | B | 125 | 0.875817 | 0.690144 +/- 0.005163 | 0.772970 +/- 0.004925 | 0.932490 +/- 0.001635 |
| jaad_all | C | 109 | 0.866805 | 0.677206 +/- 0.004968 | 0.768659 +/- 0.004816 | 0.928386 +/- 0.002726 |
| jaad_all | D | 116 | 0.868439 | 0.679608 +/- 0.000807 | 0.772136 +/- 0.005430 | 0.928459 +/- 0.000735 |
| jaad_all | E | 188 | 0.875520 | 0.689324 +/- 0.004198 | 0.774357 +/- 0.001987 | 0.932026 +/- 0.000681 |
| jaad_beh | A | 53 | 0.663831 | 0.776370 +/- 0.002648 | 0.722481 +/- 0.005975 | 0.651162 +/- 0.001709 |
| jaad_beh | B | 125 | 0.675704 | 0.781669 +/- 0.003731 | 0.745902 +/- 0.004116 | 0.684650 +/- 0.003194 |
| jaad_beh | C | 109 | 0.674996 | 0.778232 +/- 0.004409 | 0.745954 +/- 0.004484 | 0.688548 +/- 0.005409 |
| jaad_beh | D | 116 | 0.676236 | 0.779577 +/- 0.003506 | 0.753045 +/- 0.010509 | 0.691779 +/- 0.005598 |
| jaad_beh | E | 188 | 0.682970 | 0.782335 +/- 0.001499 | 0.772707 +/- 0.008176 | 0.722024 +/- 0.004950 |

## Paired directions

- jaad_all B_minus_A: f1 +0.029826 (3/3 favorable); pr_auc +0.023447 (3/3 favorable); roc_auc +0.004710 (3/3 favorable).
- jaad_all C_minus_A: f1 +0.016888 (3/3 favorable); pr_auc +0.019137 (3/3 favorable); roc_auc +0.000606 (1/3 favorable).
- jaad_all C_minus_B: f1 -0.012938 (0/3 favorable); pr_auc -0.004310 (1/3 favorable); roc_auc -0.004105 (0/3 favorable).
- jaad_all D_minus_A: f1 +0.019290 (3/3 favorable); pr_auc +0.022614 (3/3 favorable); roc_auc +0.000679 (2/3 favorable).
- jaad_all D_minus_B: f1 -0.010537 (0/3 favorable); pr_auc -0.000833 (1/3 favorable); roc_auc -0.004031 (0/3 favorable).
- jaad_all D_minus_C: f1 +0.002401 (2/3 favorable); pr_auc +0.003477 (2/3 favorable); roc_auc +0.000073 (2/3 favorable).
- jaad_all E_minus_A: f1 +0.029006 (3/3 favorable); pr_auc +0.024835 (3/3 favorable); roc_auc +0.004246 (3/3 favorable).
- jaad_all E_minus_B: f1 -0.000820 (2/3 favorable); pr_auc +0.001388 (2/3 favorable); roc_auc -0.000465 (2/3 favorable).
- jaad_all E_minus_C: f1 +0.012118 (3/3 favorable); pr_auc +0.005698 (2/3 favorable); roc_auc +0.003640 (3/3 favorable).
- jaad_all E_minus_D: f1 +0.009717 (3/3 favorable); pr_auc +0.002221 (1/3 favorable); roc_auc +0.003567 (3/3 favorable).
- jaad_beh B_minus_A: f1 +0.005299 (2/3 favorable); pr_auc +0.023421 (3/3 favorable); roc_auc +0.033488 (3/3 favorable).
- jaad_beh C_minus_A: f1 +0.001863 (1/3 favorable); pr_auc +0.023473 (3/3 favorable); roc_auc +0.037386 (3/3 favorable).
- jaad_beh C_minus_B: f1 -0.003436 (1/3 favorable); pr_auc +0.000053 (1/3 favorable); roc_auc +0.003898 (2/3 favorable).
- jaad_beh D_minus_A: f1 +0.003207 (2/3 favorable); pr_auc +0.030564 (3/3 favorable); roc_auc +0.040618 (3/3 favorable).
- jaad_beh D_minus_B: f1 -0.002092 (0/3 favorable); pr_auc +0.007144 (2/3 favorable); roc_auc +0.007130 (2/3 favorable).
- jaad_beh D_minus_C: f1 +0.001345 (2/3 favorable); pr_auc +0.007091 (2/3 favorable); roc_auc +0.003232 (2/3 favorable).
- jaad_beh E_minus_A: f1 +0.005965 (3/3 favorable); pr_auc +0.050225 (3/3 favorable); roc_auc +0.070862 (3/3 favorable).
- jaad_beh E_minus_B: f1 +0.000666 (2/3 favorable); pr_auc +0.026805 (3/3 favorable); roc_auc +0.037374 (3/3 favorable).
- jaad_beh E_minus_C: f1 +0.004103 (2/3 favorable); pr_auc +0.026752 (3/3 favorable); roc_auc +0.033476 (3/3 favorable).
- jaad_beh E_minus_D: f1 +0.002758 (2/3 favorable); pr_auc +0.019661 (3/3 favorable); roc_auc +0.030244 (3/3 favorable).

## Historical clean-cohort reference

metrics/clean_exp098_historical_reference.csv records the completed Exp098 scores.
Exp098 A/B/C/D map to Exp102 A/B/D/E respectively; Exp098 has no core56 arm.
Exact shared development rows passed extraction/schema parity. These are separate
training/test populations, not a matched-sample causal estimate of sampling effects.

## Boundaries

This tests frozen representation bundles, including existing v1 validity counters
and v2 metadata. Core56 is an exact subset of full63 from the same extraction.
No new ego, road, temporal or confidence stream was added. No raw pose estimator ran.
The original train/validation/test populations remain unchanged, including observed
crossing, unverified endpoints, non-behavior negatives and two nonconsecutive raw-frame
training windows. API TTE counts track observations to an endpoint, not necessarily
raw frames or time to a verified crossing onset; see per-split source audits.
These cohorts have historical exposure. Published-protocol alignment is not proof
of row or threshold identity to another paper. Do not select a universal BIU winner
or compare clean/API absolute metrics as if they shared the same population.

JAAD_all/A probabilities are reused from Exp031 after reconstructed source, cohort,
ordering and recipe checks. Its original model files and runtime lock are unavailable;
this is not live checkpoint reproduction. See REUSE_AUDIT.md and metrics/reuse_audit.json.
Bootstrap intervals are descriptive, not multiplicity-adjusted, and operate on
seed-ensemble probabilities rather than mean-per-seed metrics.

Exact sources/hashes: metrics/freeze.json and *_source_audit.json.
Execution commands: logs/commands.log; persistent status: logs/progress.log.
