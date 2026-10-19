# Exp115: JAAD road context plus vehicle state

**Cue-level extension:** [Full nine-page attribution report](../biu_cue_attribution_report_2026_09_12/Input_Attribution_with_BIU_Cues.pdf) | [Only cue pages 7–9](../biu_cue_attribution_report_2026_09_12/BIU_Cue_Contribution_Extra_Pages.pdf) | [Cue methods and exact feature membership](../biu_cue_attribution_report_2026_09_12/REPORT.md). This later diagnostic extension preserves all results below.

Completed diagnostic/retrospective extension, 2026-09-12. Nine new 257-D XGBoost models (three seeds per cohort) combine both context branches. All prior models, predictions, cached upstream features and manuscript edits are preserved.

[New two-page printable supplement](report/Exp115_Results_and_Attribution.pdf) | [Full extended attribution PDF](report/Input_Attribution_with_Exp115.pdf)

## Fixed-0.5 test performance

| Cohort | Context | Accuracy | F1 | AP (PR-AUC) | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Clean JAAD | Road only | 70.42 | 73.53 | 65.94 | 75.03 |
| Clean JAAD | Vehicle state only | 67.73 | 71.91 | 75.57 | 77.77 |
| Clean JAAD | Road + vehicle state | 71.13 | 73.99 | 66.89 | 75.66 |
| JAAD_beh | Road only | 67.82 | 77.93 | 74.69 | 67.23 |
| JAAD_beh | Road + vehicle state | 67.98 | 78.12 | 74.86 | 67.10 |
| JAAD_all | Road only | 89.14 | 72.75 | 81.16 | 94.55 |
| JAAD_all | Road + vehicle state | 89.47 | 73.33 | 82.06 | 94.90 |

All displayed metrics are percentages, means across seeds42/43/44; not ensembles. Tables are generated from metrics/summary_results.csv, with full mean/sampleSD there and per-seed data in metrics/per_seed_results.csv. No model or threshold is selected from these test results. FP/FN counts and rates are retained in CSVs, omitted from compact main tables.

## Matched test changes

| Cohort | Comparison | F1 change (percentage points) | AP change | ROC-AUC change | Favorable F1 / AP / ROC seeds |
|---|---|---:|---:|---:|---|
| jaad_all | RV257_minus_R242 | +0.573 | +0.908 | +0.351 | 3/3 / 3/3 / 3/3 |
| jaad_beh | RV257_minus_R242 | +0.181 | +0.168 | -0.128 | 2/3 / 2/3 / 1/3 |
| jaad_clean | RV257_minus_R242 | +0.461 | +0.950 | +0.635 | 3/3 / 3/3 / 2/3 |
| jaad_clean | RV257_minus_V197 | +2.077 | -8.679 | -2.111 | 3/3 / 0/3 / 0/3 |

Against road-only R242, RV257 improves mean F1/AP in all three cohorts; JAAD_beh has a small mean ROC-AUC decrease and inconsistent seed changes. Clean JAAD and JAAD_all show F1/AP gains in all three seeds. These small descriptive differences are not statistical significance claims. Against clean V197, adding road increases fixed-threshold F1 in every seed but lowers AP and ROC-AUC in every seed. No universal context winner is established.

## Inputs and cohort boundaries

- R242: exact saved Exp081-D/Exp082 road parent for clean JAAD, or saved Exp083 model for JAAD_beh/all. RV257 adds only Exp099's15 categorical vehicle summaries with the same estimator recipe and cohort. Clean V197 is saved Exp099-B. No vehicle-only reruns were introduced for JAAD_beh/all.
- Portable182 = BBox53 + historical Combined BIU108 + explicit temporal21. Compensated road60 + vehicle state15 gives257. Both branch sets use Combined BIUs; this is not a BIU-version comparison. Two pose-quality counters are included in static BIU108.
- Vehicle state is stopped / moving_slow / moving_fast / decelerating / accelerating. Five observation occupancy fractions, five first-frame one-hot indicators and five last-frame indicators. Raw XML at exactly the recorded observation frames supplies labels; no numeric physical speed or future annotation. Missing annotations would stop the run; all windows had coverage, none were dropped.
- Clean JAAD and API-style JAAD_beh/all remain separate native-training cohorts. API cohorts include observed-crossing/unverified-onset examples and JAAD_all non-behavior negatives according to the parent definitions. This is not PIE-to-JAAD zero-shot transfer. The clean road reference retains its Exp082 validation-selected follow-up status.
- Carry forward Exp083's documented two JAAD_all training windows with a134-frame annotation gap. Actual observed frame IDs were used for vehicle summaries; no silent consecutive-frame substitution. Every validation/test window remains raw-frame consecutive.

| Cohort | Split | Windows | Positive | Tracks | Videos |
|---|---|---:|---:|---:|---:|
| jaad_clean | train | 1552 | 999 | 154 | 98 |
| jaad_clean | validation | 274 | 170 | 28 | 15 |
| jaad_beh | train | 2134 | 1760 | 194 | 106 |
| jaad_beh | validation | 242 | 176 | 22 | 14 |
| jaad_all | train | 8613 | 1760 | 783 | 155 |
| jaad_all | validation | 1265 | 176 | 115 | 26 |
| jaad_clean | test | 1501 | 720 | 151 | 73 |
| jaad_beh | test | 1881 | 1177 | 171 | 81 |
| jaad_all | test | 6732 | 1177 | 612 | 112 |

## Training and verification

Exact parent XGBoost recipe:250 trees, depth4, LR0.05, subsample0.8, colsample_bytree0.8, L2=1, train negative/positive class weight, median imputation, four CPU threads, seeds42/43/44. No early stopping, validation-selected threshold, calibration, feature/model search or train+validation refit. All nine models were frozen before this run loaded test features; all advanced regardless of validation. Historical test exposure is disclosed, so this is retrospective evidence.

All9 checkpoints were saved/reloaded and checked against their frozen parameters and train medians. Source/feature contracts, cohort IDs and labels, split video/track disjointness, raw-frame gaps and vehicle coverage were audited. Twenty-four parent model/split reuse checks reproduced saved probabilities. Independent verification recomputed42 prediction groups and630 metric cells, with max absolute difference1.11e-16, and verified all paired deltas and seed summaries. INDEPENDENT_VERIFICATION.json records the checks.

Training timing and model bytes: metrics/training_timing.csv. Cached batch prediction timing: metrics/prediction_timing.csv. These single-pass CPU measurements exclude feature construction and upstream pose/road perception; they are not end-to-end latency claims.

## Exact input attribution

| Branch | Clean JAAD | JAAD_beh | JAAD_all |
|---|---:|---:|---:|
| BBox | 22.20 ± 0.36 | 17.04 ± 0.69 | 51.85 ± 0.36 |
| Combined BIU package | 19.24 ± 0.40 | 34.91 ± 0.21 | 18.66 ± 0.26 |
| Explicit temporal | 21.52 ± 0.11 | 11.40 ± 0.40 | 4.19 ± 0.41 |
| Road context | 34.75 ± 0.55 | 35.90 ± 0.96 | 22.32 ± 0.40 |
| Vehicle state | 2.28 ± 0.13 | 0.75 ± 0.34 | 2.99 ± 0.11 |

Test-window attribution percentages, mean ± sample seed SD. Exact training-path-statistics feature TreeSHAP uses raw log-odds: sum signed feature contributions within each branch per window; average the absolute net; normalize included branches to100 within each seed; average across seeds. Baseline excluded. This is aggregation of feature Shapley values, not an independent coalition/group-SHAP computation.

Vehicle-state share is smaller with road present than in the clean V197 model; this is conditional attribution under changed feature competition and a refitted model. It does not demonstrate that the vehicle annotation lacks value, nor that road causally replaces it. Correlation, group width and cancellation affect allocation. Temporal and road features also use pose/BBox; static BIU importance is not total pose importance. These are not percentages of accuracy, probability, causal benefit or culture-specific behavior.

All18 new attribution model/split checks passed saved probability and SHAP additivity checks. The report independently reconstructs attribution shares and direction from all54 saved group arrays across prior reports and this extension. report/combined_attribution_summary.csv covers all nine experiment/cohort contexts; validation, label/error strata, direction toward labels and alternative feature-mass shares remain available.

Final JAAD interpretation and any calibration decision remain open. The original four PDF pages are preserved and the Exp115 pages are appended; their earlier experiment percentages remain valid for those earlier models.

## Exact commands and source records

Working directory: /home/abdelrahman/Hadwa/BIU-PID-Distillation-clean

```bash
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python -u experiments/115_jaad_road_vehicle_state_xgb/scripts/run.py prepare > experiments/115_jaad_road_vehicle_state_xgb/logs/prepare.log 2>&1
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python -u experiments/115_jaad_road_vehicle_state_xgb/scripts/run.py train > experiments/115_jaad_road_vehicle_state_xgb/logs/train.log 2>&1
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python -u experiments/115_jaad_road_vehicle_state_xgb/scripts/run.py evaluate > experiments/115_jaad_road_vehicle_state_xgb/logs/evaluate.log 2>&1
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/115_jaad_road_vehicle_state_xgb/scripts/verify.py > experiments/115_jaad_road_vehicle_state_xgb/logs/verify.log 2>&1
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python -u experiments/115_jaad_road_vehicle_state_xgb/scripts/attribute.py > experiments/115_jaad_road_vehicle_state_xgb/logs/attribution.log 2>&1
MPLCONFIGDIR=/tmp/thesis-input-attribution-mpl /home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/115_jaad_road_vehicle_state_xgb/scripts/report.py
python3 experiments/115_jaad_road_vehicle_state_xgb/scripts/merge_pdf.py
```

FROZEN.json pins exact development source paths/hashes, training plans and environment versions; test_sources.json pins test inputs. TRAINED.json freezes all9 checkpoint hashes before test loading. EVALUATED.json pins caches, predictions and metric outputs. attribution/sources.json pins explanation inputs. report/sources.json pins reused and new report inputs. models/ contains the new checkpoints; cache/ contains257-D matrices and row manifests; predictions/ contains42 new/reused prediction groups. No old benchmark/model artifacts were modified, no perception rerun, no manuscript changes and no git commit/push.
