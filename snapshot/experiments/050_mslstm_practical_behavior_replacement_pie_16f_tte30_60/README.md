# Exp050: Practical Behavior Replacement On PIE 16f/TTE30--60

Controlled A--F repetition of Exp048 under the project's standardized PIE anticipatory
benchmark. See `DESIGN.md` for the frozen protocol.

The runner is phase-separated and resumable:

```bash
python3 experiments/050_mslstm_practical_behavior_replacement_pie_16f_tte30_60/scripts/run_experiment.py prepare
python3 experiments/050_mslstm_practical_behavior_replacement_pie_16f_tte30_60/scripts/run_experiment.py train-all
python3 experiments/050_mslstm_practical_behavior_replacement_pie_16f_tte30_60/scripts/run_experiment.py evaluate
python3 experiments/050_mslstm_practical_behavior_replacement_pie_16f_tte30_60/scripts/run_experiment.py verify
```

`prepare` and `train-all` reject set03 access. `evaluate` requires authenticated
completion markers for all 18 A--F/42--44 checkpoints. `EXPERIMENT_COMPLETE` is written
only after the final package verifies.
