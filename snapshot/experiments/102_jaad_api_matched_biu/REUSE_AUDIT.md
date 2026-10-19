# Exp102 Existing-Run Reuse Audit

Audited before training. This is not an invitation to rerun every JAAD model.

| Source | Relationship | Decision |
|---|---|---|
| Exp031 API-style BBox XGBoost42/43/44 | JAAD_all train8613,val1265,test6732; same53 columns and250-tree recipe; saved probabilities | Reuse A/all after numerical cohort/frame/order checks |
| Exp009 | Clean population, incomplete historical v2 coverage; multiple horizons | Not an API-style equivalent |
| Exp077 | Clean/zero-shot182-D union plus temporal or two-stage model | Different task/population or inputs |
| Exp079 | API-style182-D or sequential model, not53/125/109/116/188-D arms | Do not retrain it; not equivalent |
| Exp080 | RF182-D, not frozen XGBoost feature arms | Different model/inputs |
| Exp083 | Correct cohorts, but242-D union plus temporal and road | Reuse manifests/pose measurements; do not reuse its classifier |
| Exp089 | PedCMT on clean population | Reuse frozen BIU adapter, not model |
| Exp098 | Desired basic XGBoost arms on clean1552/274/1501 population | New population requires fresh BIU-arm models |
| Exp099--101 | Clean population plus temporal/vehicle features | Different inputs/populations |
| AIBThings clean/all-vs-beh support tables | Clean1501 test or separate bystander augmentation/height filtering | Not exact Exp083 cohorts |
| Archive experiments001--006 | Historical paper/KD lineage, no matching API cohort five-arm experiment | Read-only reference only |

## Exp031 reuse proof and limitations

Direct inspection: its API-style training population is exactly Exp083 all's
8613 samples in the same ordering. BBox53 column ordering and extractor values
match the frozen Exp098 builder. Its XGBoost configuration is250 trees, depth4,
LR0.05, subsample0.8,colsample0.8,L2=1,n_jobs4 and train neg/pos weighting.
The runner automatically verifies training/validation labels, frame windows,
ordering, BBoxes, all53 recomputed values and parameter literals before fitting.
It verifies test identities again only after all validation runs finish.

Historical candidate selection chose XGBoost for all three seeds. Selection
does not alter that fitted candidate. Its val_f1 threshold is not reused;
saved probabilities are evaluated at0.5 for this experiment. Files:

- experiments/031_jaad_api_label_protocol_sensitivity/metrics/protocol_a_window_audit.csv
- experiments/031_jaad_api_label_protocol_sensitivity/metrics/selected_model_predictions.csv
- scripts/experiments/jaad_api_label_protocol_sensitivity.py
- scripts/experiments/jaad_biu_v2_benchmark_comparison.py
- /home/abdelrahman/Hadwa/BIU-PID-Distillation/src/features/bbox_features.py

No historical checkpoint or environment lock exists for these runs. Thus this
is source/prediction-lineage reuse, not cryptographically verified historical
runtime identity or live checkpoint parity. This limitation must remain in
the final report. New-feature arms share one fresh runner; avoid claiming that
the reused baseline has stronger provenance than the artifacts support.

Expected outcome:27 new fits,3 prediction baselines reused, zero pose reruns.
Machine-readable proof and hashes: metrics/reuse_audit.json and metrics/freeze.json.
