# Exp117: standardized neural BIU-v2-core completion

Nine new core56 models: three seeds each for the standardized PIE MSLSTM,
standardized PIE PedCMT, and clean-JAAD PedCMT comparisons. Existing full63,
v1, raw-pose and other matched controls are reused from Exp053/095/084/089.
No native 30-frame experiment or historical architecture search is rerun.

The only model shape change is the BIU projector input, from 63 to 56, reducing
each model by 896 parameters. Every parent's fusion, preprocessing, augmentation,
training and checkpoint selection remains unchanged. In particular MSLSTM
selects best validation F1; PedCMT selects lowest validation classification BCE.

See [DESIGN.md](DESIGN.md) for the frozen specification and
[FINAL_REPORT.md](FINAL_REPORT.md) for results once complete.

## Execution

From `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean`:

```bash
env LD_LIBRARY_PATH=/usr/lib/wsl/lib /home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python -u experiments/117_neural_v2_core_completion/scripts/run.py
tail -n 20 experiments/117_neural_v2_core_completion/logs/progress.log
```

GPU execution requires the existing unsandboxed WSL permission. Verified on the
NVIDIA GeForce RTX 3070 Laptop GPU; no driver/environment files were modified.
No tmux. Per-epoch histories are under each case's `logs/` directory.

## Audit and artifacts

- `FROZEN.json`: script/design hashes frozen before new training.
- `metrics/pretraining_audit.json`: nine parent validation replays, model shape
  parity and parameter counts. Each case also passed a CUDA backward smoke test.
- `metrics/trainval_sources.json`: exact source paths and SHA-256 before training.
- `<case>/metrics/feature_contract.json`: all 56 feature names and source indices.
- `TRAINED.json`: hashes of all nine new checkpoints and validation predictions,
  frozen before test caches or test predictions are opened.
- `predictions/`: fresh core56 validation/test probabilities for every seed.
- `metrics/per_seed_results.csv`: new models and reused saved parent controls.
- `metrics/summary_results.csv`: means and sample standard deviations.
- `metrics/paired_deltas.csv`: core56 minus each corresponding saved control.
- `metrics/error_transitions_vs_full63.csv`: corrected and harmed sample counts.
- `metrics/all_sources.json`, `metrics/all_audits.json`: complete final provenance.
- `EXPERIMENT_COMPLETE.json`: present only after all models, test evaluations,
  saved-metric checks and source-immutability checks pass.

These are retrospective matched comparisons on previously exposed test cohorts.
Every new arm proceeds to the frozen final evaluation; no selection or threshold
tuning follows test results. No thesis edits, commits, or additional experiments.
