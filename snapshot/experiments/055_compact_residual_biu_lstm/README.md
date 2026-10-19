# Exp055: Compact Residual BIU-LSTM

This experiment implements the frozen design in `DESIGN.md`. It targets the project's
PIE 16-frame/TTE 30--60 benchmark with a compact neural BBox+BIU anchor, meaningful
temporal BBox/ego/YOLOP encoders, and sample-dependent protected residual context.

No training or final evaluation result exists until the corresponding authenticated
completion marker is present. Exp012, Exp050, Exp053 and Exp054 are reused read-only.

Primary source artifacts:

- Exp050 cached cohorts and automatically extracted inputs;
- Exp053 train-only-standardized BBox53 features;
- Exp012 feature-set definitions for the exact 56-D BIU-v2 core;
- SHA-pinned historical Exp012 pose lineage inherited through Exp050.

The historical pose-cache limitation remains unchanged: the cache is byte-identical to
the main BIU artifact and documented as ViTPose-S-derived, but the original checkpoint,
configuration and execution log are unavailable, so checkpoint identity is not
cryptographically verifiable.

## Execution

Use the `biu-pid-distill` environment:

```bash
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python \
  experiments/055_compact_residual_biu_lstm/scripts/run_experiment.py prepare
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python \
  experiments/055_compact_residual_biu_lstm/scripts/run_experiment.py stage1
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python \
  experiments/055_compact_residual_biu_lstm/scripts/run_experiment.py stage2
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python \
  experiments/055_compact_residual_biu_lstm/scripts/run_experiment.py evaluate
```

`stage1` trains and validates 27 runs, then freezes the selected anchor. `stage2`
reuses that context-free anchor run, trains nine context candidates and freezes one
final candidate. `evaluate` is the only phase allowed to construct/read the set03
cache.

## Pre-Launch Verification

- Tests: `27 passed` using `pytest -q experiments/055_compact_residual_biu_lstm/tests`.
- Compilation: `python -m compileall` passed for scripts and tests.
- Prepared population: 2,366 train and 3,619 validation windows.
- Source/cohort invariants: authenticated and recorded with `test_accessed=false`.
- Test cache: absent before launch.
- Model size: 17,617 to 38,646 trainable parameters depending on candidate.
- GPU smoke: forward/backward passed on NVIDIA GeForce RTX 3070 Laptop GPU.
