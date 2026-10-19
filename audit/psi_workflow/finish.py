from pathlib import Path
import json,shutil,subprocess,hashlib
D=Path('/home/abdelrahman/Hadwa/pedestrian-intent-thesis');A=D/'audit/psi_workflow'
(D/'PSI_REPRODUCTION.md').write_text('''# PSI Exp124 / Exp116 reproduction

## What is runnable

`tools/psi.py` provides input preparation, development preflight, fresh training of all four arms, and gated test evaluation. It imports no research/archive experiment modules. It starts from the exact frozen feature bundle, **not raw video**. Input preparation imports the Exp116 feature caches and validates their hashes and common feature prefixes; it does not run perception or extract features.

The locally staged bundle is under `data/psi_frozen/`. It is ignored by Git and has **no public download yet**. Publishing code alone is therefore insufficient for outside readers. A distributable feature-bundle route or complete raw-input reconstruction route must be provided before release.

## Setup

Python 3.10.20 and the pinned dependencies in `requirements-psi.txt` were used for saved-model replay. A fresh-environment installation and fresh training have not been validated.

```sh
python3.10 -m venv .venv-psi
. .venv-psi/bin/activate
python -m pip install -r requirements-psi.txt
```

Run the following commands from the companion repository root. Input and output arguments are explicit. All outputs must remain within this repository. Use a new run name each time; existing output directories are refused.

For a curator possessing the original Exp116 caches:

```sh
python tools/psi.py prepare --source /path/to/exp116/cache --output data/psi_frozen
```

This expects `psi_ego/{train,validation,test}` and `psi_road_ego/{train,validation,test}`, each containing `rows.csv`, `columns.json`, and `features.npy`. The exact expected hashes are in `configs/psi_exp124.json`. The command retains the 195-column matrix; A/B/C are exact prefixes. If the frozen bundle is already installed, skip this import.

```sh
python tools/psi.py preflight --bundle data/psi_frozen
python tools/psi.py train --bundle data/psi_frozen --output runs/psi_training_01
python tools/psi.py evaluate --bundle data/psi_frozen --trained runs/psi_training_01 --output runs/psi_test_01
```

**Training has not been run during packaging.** The train command performs 12 actual fits. It reads only train/validation features, fits each imputer on training rows, and uses validation only to report metrics. It has no early stopping, threshold search, or model selection. All model hashes are frozen before test evaluation is permitted. Predictions and per-seed/mean/sample-standard-deviation tables are written into the new run folders.

The historical Exp124 fitted A/B and reused C/D from Exp116. This standalone workflow deliberately fits all four arms with the same frozen recipe, so it needs no parent checkpoints. Treat its outputs as new reproduction runs, not replacements for historical results. Equality after fresh training is not yet verified.

## Fixed protocol

- PSI-native next-frame intent task: 15 consecutive observed frames; target at frame 16. This is not the PIE/JAAD 16-observation TTE30–60 protocol.
- Labels are `annotation_vote >= 0.5`; prediction threshold is fixed at 0.5.
- Frozen populations: 26,295 training, 14,922 validation, and 20,547 test windows. Training videos 0032, 0055, and 0093 were excluded for unavailable speed, including in arms that do not consume speed.
- A: BBox53; B: BBox53 + BIU56 (109); C: B + temporal21 + ego5 (135); D: C + road60 (195). Input order is frozen in the configuration.
- Seeds 42/43/44. Median imputation; XGBoost 250 trees, depth 4, learning rate .05, subsample .8, column subsample .8, L2 1, four CPU threads, training negative/positive weight 7697/18598. AP means average precision.

## Validation performed

The diagnostic replay command uses trusted locally copied historical model files and reference predictions:

```sh
python tools/replay_psi.py --bundle data/psi_frozen --reference data/psi_reference --output runs/psi_replay_01
```

All 12 saved models passed on validation and test (24 arm/seed/split combinations), including row/label alignment, explicit model settings, imputer settings, probability comparison at absolute tolerance 1e-7, and 14 metric comparisons per combination at absolute tolerance 1e-10. Maximum probability difference was 2.980194091506405e-08. No models were fitted. This is saved-model replay, not evidence that fresh training or raw-data extraction reproduces the benchmark.

Checks ran from `/tmp` under isolated Python mode, with an audit hook blocking access to both original research repositories. This validates absence of those runtime dependencies for these exercised paths; it is not a clean machine/environment installation test. Checks also verified rejection of existing output folders, external output destinations, and test evaluation without a completed training manifest.

Exact commands and stdout/stderr: `audit/psi_workflow/checks_v2.json`. Results: `runs/psi_saved_model_replay_v2/replay.json`. Earlier failed check artifacts remain preserved; that check compared NaN imputer settings by ordinary equality and was corrected without changing model settings or features.

## Upstream feature lineage and remaining release work

1. Exp113 `run.py` uses the recovered official PSI annotation extension, database builder, and sequence processor to freeze 15-frame windows and next-frame mean-vote labels. Provenance and official code commits are recorded in `snapshot/experiments/113_psi_native_biu_xgb/recovery.json`.
2. Exp113 extracts BBox53 and BIU features from cognitive bounding boxes and the pre-existing `psi_vitpose_s_all_tracks.jsonl`. Static BIU-v2 is called with `kp_conf=None` through Exp089; changing that would alter the feature contract. The exact pose file is identified by SHA-256 `31d8f45a7f8b61803b2dde8f981ce012086ac9905455d7aaac743bc1ec2a3fc7`. A portable command reproducing that pose file still needs verification.
3. Exp114 `run.py` constructs first-7/last-8 temporal features and official speed summaries. Its road branch uses the Exp075 TwinLiteNet+ Large adapter and geometry routines. The three missing-speed training videos are excluded. Full raw-input extraction also needs the model weights, source videos, coordinate conventions, and licensing/acquisition instructions.
4. Exp116 `build()` aligns Exp113 static features to the extended Exp114 cohort by sequence ID, replaces combined BIU inputs with canonical BIU56, preserves all remaining feature columns, and saves float32 135/195-column matrices.
5. Exp124 uses prefixes of those exact matrices and reuses Exp116 C/D results. The new runner reproduces that model recipe from a frozen input bundle, without importing any of those historical runners.

Historical files under `snapshot/` are lineage evidence, not portable entry points. No raw-video preprocessing command is claimed here. Before publication: finalize data distribution or raw reconstruction, validate setup in a fresh environment, optionally authorize full retraining, and supply the final release URL to the thesis session.
''')
for name in ['.venv-psi/']:
    p=D/'.gitignore';s=p.read_text();p.write_text(s+(name+'\n' if name not in s else ''))
p=D/'README.md';s=p.read_text();s+='\n## PSI model workflow\n\nA portable frozen-feature preparation/training/evaluation runner is now available: [PSI_REPRODUCTION.md](PSI_REPRODUCTION.md). Saved-model replay passed with original repositories blocked. Fresh training, a clean environment installation, and raw-video preprocessing remain unverified. Input bundles are local and unpublished.\n';p.write_text(s)
p=D/'REPRODUCING.md';s=p.read_text();s+='\n## PSI implementation update\n\nSee [PSI_REPRODUCTION.md](PSI_REPRODUCTION.md) for the new standalone cached-feature workflow. It supersedes the earlier training-dependency blockers for that route only; the raw-data chain remains unfinished.\n';p.write_text(s)
(A/'commands.txt').write_text('''Preparation and installation:
python /tmp/psi_companion_build/install.py
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python -B /home/abdelrahman/Hadwa/pedestrian-intent-thesis/tools/psi.py prepare --source /home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/116_xgboost_v2_core_completion/cache --output /home/abdelrahman/Hadwa/pedestrian-intent-thesis/data/psi_frozen
python /tmp/psi_companion_build/check.py
cp /tmp/psi_companion_build/replay_psi.py /home/abdelrahman/Hadwa/pedestrian-intent-thesis/tools/replay_psi.py
python /tmp/psi_companion_build/check.py
python /tmp/psi_companion_build/finish.py

Exact isolated subprocess commands and outputs: checks.json (initial failed check), checks_v2.json (corrected checks).
Source paths/checksums for copied reference artifacts: reference_copy_manifest.json.
Frozen feature file checksums: configs/psi_exp124.json.
No training command was executed successfully; train command tests intentionally failed before fit or output creation.
''')
(A/'finish.py').write_bytes(Path(__file__).read_bytes())
# Preserve package status and verify no remote was accidentally configured.
r=subprocess.check_output(['git','-C',str(D),'remote','-v'],text=True);assert not r
print('Usage/provenance documentation complete; companion repository still has no remote.')
