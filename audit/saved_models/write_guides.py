from pathlib import Path
import json,re
D=Path('/home/abdelrahman/Hadwa/pedestrian-intent-thesis');A=D/'audit/saved_models'
(D/'DATA_PREPARATION.md').write_text('''# Data preparation and input contracts

The release is being prepared without retraining. Existing outputs are the reference evidence. The historical preprocessing code is included for traceability; only commands explicitly marked portable have been checked outside the original repositories.

## Start with a result and its experiment

Use `snapshot/thesis/14_RESULT_SOURCE_MAP.md` and `EXPERIMENT_RECIPES.md`. Choose the experiment and dataset cohort first. Preserve sequence IDs, row order, feature names, split membership, and label policy. Do not generate a new random split or assume that all experiments use identical cohorts. Fit imputation and normalization only on the original training rows. Saved weights require their original feature order and preprocessing statistics.

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

Raw-JAAD reconstruction and portable neural inference are not yet validated. Historical scripts under `snapshot/` still contain original filesystem assumptions and must not be executed directly.

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
''')
(D/'SAVED_WEIGHTS.md').write_text('''# Saved weights and evaluation

No retraining is needed to use existing weights. This package uses preserved benchmark outputs; it does not replace them with newly trained models.

## Locally packaged artifacts

| Result family | Weights | Other required components | Status |
|---|---|---|---|
| PSI Exp124 A/B; Exp116 C/D | `data/psi_reference/{A,B,C,D}_seed{42,43,44}.joblib` | Frozen feature bundle, exact columns, pinned Python dependencies | Saved-model replay verified |
| Exp119 capped PedCMT | `data/saved_weights/experiments/119_pedcmt_v2_capped_completion/{pie,jaad}/checkpoints/` | Exp084 PIE or Exp089 JAAD parent, training normalization and feature contract, model code; fixed 0.10 cap | Files copied and hashed; portable inference pending |
| Exp094 JAAD raw control | `data/saved_weights/experiments/094_pedcmt_jaad_capped_raw_pose_control/checkpoints/` | Exp089 frozen parent and original raw-pose branch implementation | Files copied and hashed; portable inference pending |
| Exp122 clean-JAAD MSLSTM | `data/saved_weights/experiments/122_jaad_mslstm_no_noise/checkpoints/*_ap.pt` | Original model/stream code, BBox53 scaler and feature contract | Primary AP-selected weights copied and hashed; portable inference pending |

Exp084/089 parent checkpoints and `training_stats.npz` are included in their original relative experiment layout under `data/saved_weights/`. The neural preprocessing files are also included there. Parent and branch weights must be matched by seed. Exp122's F1-selected checkpoints are diagnostic and were not copied as the primary results.

The exact paths, sizes and SHA-256 hashes of newly copied neural artifacts are in `audit/saved_models/copy_manifest.json`. PSI artifact hashes are in `data/psi_reference/manifest.json`, with original paths recorded in `audit/psi_workflow/reference_copy_manifest.json`.

## Inventory scope

`audit/saved_models/weight_inventory.csv` lists local candidate weights in the previously selected experiment/dependency trees. This is an availability inventory, not a declaration that every file supports a thesis row. Smoke/debug/F1 files remain identifiable. Uncopied candidates do not yet have verified hashes in this inventory. Exp012 has no checkpoint files in its active experiment directory; availability elsewhere still needs verification. No refit is planned to fill missing weights.

## Commands

Verify copied neural artifact bytes without importing models:

```sh
python tools/verify_saved_weights.py
```

For the verified PSI evaluation route, see `DATA_PREPARATION.md` and `PSI_REPRODUCTION.md`. No portable neural evaluation command is claimed yet. Historical checkpoint objects also depend on the correct architecture and preprocessing; copying a `.pt` file is not sufficient on its own.

## Publication

These are local packages only. No weights have been committed, uploaded, or published. Keep model artifacts separate from the code history and provide explicit release download locations and checksums when the release is ready. The thesis should link to the reviewed companion release once it exists.
''')
# Per-experiment source index. Do not infer a training recipe from a neighboring experiment.
maptext=(D/'snapshot/thesis/14_RESULT_SOURCE_MAP.md').read_text();section=maptext.split('## Numbered experiments',1)[1].split('## Named reports',1)[0]
folders=sorted(set(re.findall(r'\.\./(experiments/[^/)]+)',section)))
lines=['# Original experiment recipes','','Packaging policy: no retraining. Training details are documentation for readers; commands are not being executed. Use saved weights and evidence where available.','','## Verified recipe excerpts','','| Experiment | Original settings | Authoritative source |','|---|---|---|',
'| Exp053 PIE MSLSTM | Seeds 42/43/44; Adam 1e-4; batch 32; 15 epochs; validation-F1 checkpoint selection; original Gaussian minority augmentation; fixed threshold .5 | [Design](snapshot/experiments/053_mslstm_bbox53_replacement/DESIGN.md) |',
'| Exp119 capped PedCMT | Adam 5e-4, betas (.9,.98), epsilon 1e-6; batch 64; up to 200 epochs; patience 25; minimum validation uncapped BCE, earliest tie; frozen parent; inference cap .10; fixed threshold .5 | [Design](snapshot/experiments/119_pedcmt_v2_capped_completion/DESIGN.md) |',
'| Exp122 JAAD MSLSTM | Seeds 42/43/44; Adam 1e-4; batch 32; 15 epochs; balanced oversampling without Gaussian noise; maximum validation AP, earliest tie; fixed threshold .5 | [Design](snapshot/experiments/122_jaad_mslstm_no_noise/DESIGN.md), [test evaluation](snapshot/experiments/122_jaad_mslstm_no_noise/test_evaluation/evaluate.py) |',
'| Exp124 / Exp116 PSI | Median imputation; XGBoost 250 trees, depth 4, LR .05, subsample/column sample .8, L2 1; 4 CPU threads; weight 7697/18598; seeds 42/43/44; no early stopping/selection; threshold .5 | [Design](snapshot/experiments/124_psi_matched_lightweight_ablation/DESIGN.md), [frozen config](configs/psi_exp124.json) |',
'','Exp119 uses PIE balanced indices and JAAD natural indices. Preserve this distinction. Exp122 development documentation predates its separately approved test evaluation; use its test-evaluation folder for the thesis evidence. Do not apply the PSI recipe to every XGBoost experiment.','','## Mapped experiment source index','','These links preserve the full recipes and source code; historical runners are not portable commands. Missing documents are not replaced by guessed settings.','']
for rel in folders:
 root=D/'snapshot'/rel; docs=[p for name in ['DESIGN.md','README.md','FINAL_REPORT.md','AMENDED_FINAL_REPORT.md'] if (p:=root/name).exists()]
 code=[p for p in root.rglob('*.py') if p.name in {'run.py','run_experiment.py','evaluate.py'} and not any(x in p.parts for x in ['official_code','before','build'])]
 lines.append('### '+Path(rel).name+'\n')
 lines.append('Sources: '+('; '.join(f'[{p.name}]({p.relative_to(D)})' for p in docs) or 'needs verification')+'.\n')
 if code:lines.append('Recorded implementations: '+ '; '.join(f'[{p.relative_to(root)}]({p.relative_to(D)})' for p in sorted(code)) + '.\n')
(D/'EXPERIMENT_RECIPES.md').write_text('\n'.join(lines)+'\n')
(D/'tools/verify_saved_weights.py').write_text('''"""Verify copied neural weights/metadata; does not import or train models."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
records=json.loads((ROOT/'audit/saved_models/copy_manifest.json').read_text())
for record in records:
    path=(ROOT/record['destination']).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('Artifact outside companion repository')
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    if h.hexdigest()!=record['sha256']:raise ValueError(f'Checksum mismatch: {path}')
print(json.dumps({'verified_artifacts':len(records),'models_trained':0}))
''')
p=D/'README.md';s=p.read_text();s+='\n## Current release approach: no retraining\n\nPackage original training details, preprocessing guidance and existing weights. Start with [DATA_PREPARATION.md](DATA_PREPARATION.md), [EXPERIMENT_RECIPES.md](EXPERIMENT_RECIPES.md) and [SAVED_WEIGHTS.md](SAVED_WEIGHTS.md). Fresh training validation is not planned. Earlier lists mentioning possible retraining are superseded by the author’s instruction.\n';p.write_text(s)
p=D/'PSI_REPRODUCTION.md';s=p.read_text();s=s.replace('optionally authorize full retraining,','');s='> Current policy: no retraining during packaging. Training commands below are optional instructions for future readers, not required steps for using the saved models.\n\n'+s;p.write_text(s)
(A/'write_guides.py').write_bytes(Path(__file__).read_bytes());(A/'commands.txt').write_text('python /tmp/package_saved_models.py\npython /tmp/write_release_guides.py\npython /home/abdelrahman/Hadwa/pedestrian-intent-thesis/tools/verify_saved_weights.py\nOriginal source paths and hashes: copy_manifest.json\nCandidate source paths and sizes: weight_inventory.csv\n')
print('Saved-model, data-preparation and original-training guides written; no training.')
