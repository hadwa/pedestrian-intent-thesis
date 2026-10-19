# Exp114 amended complete-input validation report

## Scope

This is the user-approved amendment to the originally blocked Exp114 design.
Both matched arms use 26,295 training windows after excluding all 585 windows
from the three training videos without PSI2 speed files. Validation remains the
unchanged 14,922-window Exp113 validation cohort. This is not the same training
population as Exp113 and is not a full-cohort result.

S187 is the PSI15 FIRST7/LAST8 adaptation of Exp075's complete 187-D source
block: the historical deduplicated BBox/BIU-v1/unique-BIU-v2-core union, temporal
contrasts, and exact official speed summaries. R247 appends the exact 60-D
TwinLiteNet+ Large road-relative summaries. Both arms share identical source
features and rows. No test data were accessed.

## Validation results

Mean +/- sample standard deviation over seeds42/43/44 at fixed threshold .5:

| Arm | Accuracy | Crossing F1 | Macro F1 | AP | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| S187 | .730510 +/- .006168 | .826844 +/- .005093 | .609567 +/- .003320 | .895862 +/- .002024 | .728142 +/- .005791 |
| R247 | .750279 +/- .005846 | .840139 +/- .004203 | .634889 +/- .006362 | .902144 +/- .004604 | .749740 +/- .007226 |

Mean paired R247-minus-S187 deltas are +.019769 accuracy, +.013295 crossing
F1, +.025322 macro F1, +.006282 AP, and +.021598 ROC-AUC. Every seed has a
positive delta for all five metrics. Mean FP decreases from 2,384.00 to
2,278.67 and mean FN decreases from 1,637.33 to 1,447.67.

Under this amended PSI-native validation setup, the frozen road representation
adds useful validation information beyond its matched complete source block.
This conclusion is descriptive: windows overlap, validation guided no search or
threshold selection, and there is no held-out test evidence.

## Provenance and execution

- Reused: exact Exp113 train/validation rows, labels, BBox/BIU source arrays, and
  cached pose file SHA `31d8f45...a3fc7`; no pose extraction.
- Recovered: official numeric `speed(km/hr)` with complete retained-frame
  coverage and 140 original TrainVal MP4 members, each ZIP CRC/size verified.
- Recomputed: S187 FIRST7/LAST8 temporal/ego adaptation, TwinLite batch1 FP16
  road features for 29,694 unique video/frame identities, six XGBoost fits, and
  validation predictions/metrics.
- TwinLite checkpoint SHA:
  `5605d1a8ce762c1b6f7d23f4c2e7ec9384bfd6e6887b040b43e1f8902a817e22`.
- GPU: NVIDIA GeForce RTX 3070 Laptop GPU. Exp075's failed batch1/batch8 parity
  result was preserved, so extraction used batch1.

Commands used from the active repository included:

```bash
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/114_psi_native_extended_xgb/scripts/amended_preflight.py
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/114_psi_native_extended_xgb/scripts/recover_sources.py speed
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/114_psi_native_extended_xgb/scripts/recover_sources.py videos --shard-index <0..3> --shard-count 4
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/114_psi_native_extended_xgb/run.py prepare-source
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/114_psi_native_extended_xgb/run.py extract-road
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/114_psi_native_extended_xgb/run.py train
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/114_psi_native_extended_xgb/run.py verify
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/114_psi_native_extended_xgb/verify_results.py
```

Primary artifacts are `metrics/source_feature_audit.json`,
`metrics/road_extraction.json`, `metrics/per_seed_results.csv`,
`metrics/summary_results.csv`, `metrics/paired_deltas.csv`,
`EXPERIMENT_COMPLETE.json`, `VERIFIED.json`, and
`INDEPENDENT_VERIFICATION.json`.
