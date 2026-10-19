# Artifact package completion

All direct gaps from the result-to-artifact audit now have verified local copies. Original files, benchmark outputs and thesis sources were read only. No models were trained or loaded.

## Added artifacts

- 69 previously identified weights, preserving experiment and seed names.
- 60 saved-prediction files for Exp012, earlier PIE classifiers and the Exp031-derived JAAD-all baseline.
- 50 schema/normalization files, including selected Exp116 column orders and neural scalers/feature contracts.
- Current exhibit sources, plotting code, bibliography and confusion-matrix evidence were copied into `evidence/release_additions/`; the original snapshot was not overwritten.

## Locations

Weights: `data/saved_weights/experiments/…`
Prediction-only evidence: `data/reference_predictions/{research,archive}/…`
Preprocessing metadata: `data/preprocessing/experiments/…`
Exact mappings: `copy_manifest.json` and `coverage_after_packaging.json`.

## Verification

Run `python tools/verify_artifact_package.py` from the companion repository. It checks destination files only and needs neither the original repository nor ML libraries. Hashes validate copied bytes, not model accuracy or completeness of preprocessing.

## Remaining limitations

- Full feature arrays, pose/road extraction assets and dataset acquisition paths are not all bundled. The small preprocessing files describe the inputs; they do not replace those inputs.
- Portable saved-model inference for non-PSI models remains unfinished. Existing historical code must not be run directly.
- Prediction-only results remain prediction-only. Packaging does not recover nonexistent original checkpoints.
- Some current TeX exhibit files reference additional assets; this is an evidence package, not a standalone build of the thesis manuscript.
- No public artifact download, GitHub remote, or tagged release exists. Data/weights remain outside Git history.

## Summary

```json
{
  "copied_files": 203,
  "copied_bytes": 627547474,
  "roles": {
    "audited_scope": 1,
    "predictions": 60,
    "weights": 69,
    "contracts": 1,
    "code": 1,
    "chapter_source_comment": 2,
    "current_exhibit": 19,
    "preprocessing_metadata": 50
  },
  "audited_paths_with_verified_copy": 287,
  "audited_weight_paths_with_verified_copy": 117,
  "missing_metadata": [],
  "models_trained": 0,
  "scope": "Direct audit gaps and selected metadata only; full feature caches and portable inference are not completed by copying."
}
```
