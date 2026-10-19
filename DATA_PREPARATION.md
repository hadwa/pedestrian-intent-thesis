> Evaluation update: portable tabular, MSLSTM and full capped-PedCMT saved-model replay now passes. See [SAVED_MODEL_EVALUATION.md](SAVED_MODEL_EVALUATION.md). Earlier references below to pending neural/tabular inference are historical; raw-data and publication gaps remain.

# Data preparation and input contracts

The release is being prepared without retraining. Existing outputs are the reference evidence. The historical preprocessing code is included for traceability; only commands explicitly marked portable have been checked outside the original repositories.

## Start with a result and its experiment

Use `thesis/14_RESULT_SOURCE_MAP.md` and `EXPERIMENT_RECIPES.md`. Choose the experiment and dataset cohort first. Preserve sequence IDs, row order, feature names, split membership, and label policy. Do not generate a new random split or assume that all experiments use identical cohorts. Fit imputation and normalization only on the original training rows. Saved weights require their original feature order and preprocessing statistics.

## PIE

For Exp012 and the documented Exp053/119 cohorts:

1. Obtain the required PIE annotations/video and the pose inputs used by the source experiment. Data acquisition and exact pose generation are not yet packaged as a portable command.
2. Use the frozen 16-consecutive-frame, TTE30–60 manifests. Exp012 records train set01/set02/set06, validation set04/set05, test set03, with 2,366/3,619/3,760 windows. Do not use the legacy root README's different split list. Verify the manifest against the selected experiment before processing.
3. Align per-frame boxes and keypoints to each observed window by dataset/video/pedestrian/frame identity. Keep frames after the observation window out of predictor inputs.
4. Produce BBox53 and BIU features using the named feature definitions. Exp012 reuses archived BBox/BIU-v1 tables; its historical runner is `snapshot/scripts/experiments/pie_biu_v2_30_60_benchmark.py`, with exact inputs documented in `snapshot/experiments/012_pie_biu_v2_30_60_benchmark/README.md`.
5. For temporal/ego/road models, retain the experiment-specific block construction. Exp096 uses Exp072 inputs; Exp116 adds canonical BIU56 completion. Do not substitute a simple static feature table for the extended model.
6. For neural models, preserve the stored training-only normalization, model-specific BBox convention, and stream order. Exp119 reconstructs these checks in its `load_data` function and selects BIU56 by name.

These are source-grounded preparation instructions, not a verified raw-PIE command chain. Portable pose generation, dataset acquisition instructions, and the remaining archive-dependent input builders still need completion.

## JAAD

1. Select the exact cohort. Clean standardized JAAD used by Exp119/122 is distinct from the API behavior/all cohorts explored in Exp102/116; do not silently exchange them.
2. For Exp119/122 use 16 consecutive observations, TTE30–60, and the frozen 1,552/274/1,501 train/validation/test windows. Preserve identities and check disjoint videos and tracks across splits.
3. Exp089's source adapter documents the standardized rows, cached pose inputs and normalization. Exp120 prepares the MSLSTM-specific data using Exp053/050 utilities and historical YOLOP features.
4. Keep categorical vehicle state distinct from measured ego speed. Exp122 uses BBox53, categorical vehicle state and YOLOP; its A/B/C/D arms add raw pose, BIU56, or BIU63 as specified by its design. It does not use measured speed or acceleration.
5. For saved Exp122 weights use `metrics/bbox53_scaler.json` and `metrics/feature_contract.json` packaged with the checkpoints. Preserve the model's original stream transformations; the scaler alone is not the full preprocessing pipeline.
6. For Exp119/094 preserve the frozen parent's normalization and combine parent and correction-branch outputs with the fixed 0.10 cap. A branch checkpoint by itself is not the complete predictor.

Portable neural inference on frozen inputs is verified; raw-JAAD reconstruction is not yet validated. Historical scripts under `snapshot/` still contain original filesystem assumptions and must not be executed directly.

## PSI: portable frozen-feature route

The checked route starts from the frozen feature bundle, not raw videos. See `PSI_REPRODUCTION.md` for exact hashes, dimensions and commands.

If `data/psi_frozen` is already supplied:

```sh
python tools/psi.py preflight --bundle data/psi_frozen
python tools/replay_psi.py --bundle data/psi_frozen --reference data/psi_reference --output runs/psi_saved_evaluation_01
```

These commands do not train. The second requires the packaged trusted historical checkpoints/predictions and evaluates them on validation/test data. Use a new output folder.

If curating the bundle from Exp116 caches, the portable import command is:

```sh
python tools/psi.py prepare --source /path/to/exp116/cache --output data/psi_frozen
```

The raw-input lineage is: official PSI annotation extension/database/window creation (Exp113) → cached ViTPose-S tracks plus cognitive boxes → static BBox/BIU features → temporal first7/last8, speed and TwinLite road features (Exp114) → row-aligned canonical BIU56 replacement (Exp116). PSI uses 15 observed frames and next-frame intent, not the PIE/JAAD TTE protocol. The complete raw-video route still needs a portable pose extraction command and the perception model assets.

## Distribution status

The local `data/` folder is Git-ignored. Weights and feature bundles have not been uploaded, and no public artifact URL exists. Before release, provide explicit download locations/checksums for distributable artifacts and acquisition/regeneration instructions for inputs that cannot be distributed. Do not claim that cloning the code alone is sufficient yet.
