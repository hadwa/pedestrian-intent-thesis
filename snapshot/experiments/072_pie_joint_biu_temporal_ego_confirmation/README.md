# Exp072: PIE Joint BIU Temporal-Ego Confirmation

## Status

`PRE_REGISTERED - NOT RUN`

Exp072 tests the frozen Exp037 `joint_temporal_ego_xgb` representation on the
official PIE 16-frame, TTE 30--60, overlap-0.6 benchmark. It uses 187 automatic
features: BBox-53, BIU-v1, unique BIU-v2 cues, fixed observation-half temporal
contrasts, and exact-frame ego-speed summaries.

The model is trained once on sets01/02/06. Those same saved models predict
validation and, only after a passing validation gate, set03. There is no
train+validation refit, threshold tuning, YOLOP, calibration, or new feature
search.

## Commands

```bash
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python scripts/run_experiment.py prepare
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python scripts/run_experiment.py validate
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python scripts/run_experiment.py evaluate
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python scripts/run_experiment.py report
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python scripts/run_experiment.py verify
```

Run commands from this experiment folder or use the repository-relative script
path. The `evaluate` phase fails closed unless validation produced an
authenticated `advance` decision. No set03 feature, prediction, or metric may
exist before that decision.

The approved design is
`docs/superpowers/specs/2026-08-26-pie-joint-biu-temporal-ego-exp072-design.md`.
