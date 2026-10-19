> Evaluation update: portable tabular, MSLSTM and full capped-PedCMT saved-model replay now passes. See [SAVED_MODEL_EVALUATION.md](SAVED_MODEL_EVALUATION.md). Earlier references below to pending neural/tabular inference are historical; raw-data and publication gaps remain.

> Packaging update: all 69 audited outstanding weights and 60 prediction-only evidence files now have verified local copies. See [artifact package completion](audit/artifact_package_completion/REPORT.md). Earlier missing-copy counts below describe the prior audit snapshot.

# Saved weights and evaluation

No retraining is needed to use existing weights. This package uses preserved benchmark outputs; it does not replace them with newly trained models.

## Locally packaged artifacts

| Result family | Weights | Other required components | Status |
|---|---|---|---|
| PSI Exp124 A/B; Exp116 C/D | `data/psi_reference/{A,B,C,D}_seed{42,43,44}.joblib` | Frozen feature bundle, exact columns, pinned Python dependencies | Saved-model replay verified |
| Exp119 capped PedCMT | `data/saved_weights/experiments/119_pedcmt_v2_capped_completion/{pie,jaad}/checkpoints/` | Exp084 PIE or Exp089 JAAD parent, training normalization and feature contract, model code; fixed 0.10 cap | Saved-model replay verified |
| Exp094 JAAD raw control | `data/saved_weights/experiments/094_pedcmt_jaad_capped_raw_pose_control/checkpoints/` | Exp089 frozen parent and original raw-pose branch implementation | Saved-model replay verified |
| Exp122 clean-JAAD MSLSTM | `data/saved_weights/experiments/122_jaad_mslstm_no_noise/checkpoints/*_ap.pt` | Original model/stream code, BBox53 scaler and feature contract | Primary AP-selected weights replay verified |

Exp084/089 parent checkpoints and `training_stats.npz` are included in their original relative experiment layout under `data/saved_weights/`. The neural preprocessing files are also included there. Parent and branch weights must be matched by seed. Exp122's F1-selected checkpoints are diagnostic and were not copied as the primary results.

The exact paths, sizes and SHA-256 hashes of newly copied neural artifacts are in `audit/saved_models/copy_manifest.json`. PSI artifact hashes are in `data/psi_reference/manifest.json`, with original paths recorded in `audit/psi_workflow/reference_copy_manifest.json`.

## Inventory scope

`audit/saved_models/weight_inventory.csv` lists local candidate weights in the previously selected experiment/dependency trees. This is an availability inventory, not a declaration that every file supports a thesis row. Smoke/debug/F1 files remain identifiable. Uncopied candidates do not yet have verified hashes in this inventory. Exp012 has no checkpoint files in its active experiment directory; availability elsewhere still needs verification. No refit is planned to fill missing weights.

## Commands

Verify copied neural artifact bytes without importing models:

```sh
python tools/verify_saved_weights.py
```

For the verified PSI evaluation route, see `DATA_PREPARATION.md` and `PSI_REPRODUCTION.md`. Portable neural evaluation commands are documented in `SAVED_MODEL_EVALUATION.md`. Historical checkpoint objects also depend on the correct architecture and preprocessing; copying a `.pt` file is not sufficient on its own.

## Publication

These are local packages only. No weights have been committed, uploaded, or published. Keep model artifacts separate from the code history and provide explicit release download locations and checksums when the release is ready. The thesis should link to the reviewed companion release once it exists.
