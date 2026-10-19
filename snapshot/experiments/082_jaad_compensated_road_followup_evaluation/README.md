# Exp082: JAAD Compensated-Road Follow-up Evaluation

This experiment evaluates the already frozen Exp081 A and D models on the
designated clean JAAD test cohort. It performs no training or tuning.

Run phases separately:

```bash
python scripts/run_experiment.py audit
python scripts/run_experiment.py extract-test
python scripts/run_experiment.py evaluate
python scripts/run_experiment.py report
python scripts/run_experiment.py verify
```

See `DESIGN.md` and `pre_registration.json` for the frozen interpretation.
