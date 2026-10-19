# Exp099: JAAD categorical vehicle-action follow-up

Two matched arms: the exact Exp077 182-D one-stage XGBoost baseline, and the
same inputs plus 15 frozen categorical vehicle-action summaries. No physical
ego-speed substitution or BIU-version change. See DESIGN.md.

From the active repository:

```bash
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python -u experiments/099_jaad_exp077_vehicle_action/scripts/run_experiment.py all
```

The runner authenticates baseline reuse, then trains only three new augmented
models. All validation completes before test opens. Artifacts: metrics/,
predictions/, models/, logs/, FINAL_REPORT.md and EXPERIMENT_COMPLETE.json.
