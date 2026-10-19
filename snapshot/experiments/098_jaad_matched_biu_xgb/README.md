# Exp098: JAAD matched BIU XGBoost ablation

Controlled retrospective comparison: BBox-53 alone, plus BIU-v1 (72-D), plus
BIU-v2 (63-D), and plus both. No ego inputs. See DESIGN.md for the frozen scope.

Run from the active repository:

```bash
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python -u experiments/098_jaad_matched_biu_xgb/scripts/run_experiment.py all
```

The frozen `all` command's last verification has a documented CSV float32
round-trip limitation. After it reaches that check, run
`scripts/verify_saved_outputs.py` using the same Python environment. See
EXECUTION_NOTE.md. No training or evaluation result needs to be repeated or
changed. Final completion is indicated by EXPERIMENT_COMPLETE.json.

Phases: prepare, validate, evaluate, report, verify. Persistent progress and
commands are recorded in logs/. Source hashes, feature contracts, population
audits, metrics and paired comparisons are in metrics/. Models and probabilities
are saved under models/ and predictions/. Existing experiments are read-only.
