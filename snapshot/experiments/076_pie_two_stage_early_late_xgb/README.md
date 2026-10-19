# Exp076: PIE Two-Stage Early/Late XGBoost

Status: complete.

This experiment compares the exact one-stage Exp072 representation against a
capacity-matched two-stage XGBoost. Stage 1 observes frames 1--8. Stage 2 uses
the full 16-frame Exp072 representation plus the leakage-safe provisional
probability. See `DESIGN.md` and `pre_registration.json`.

Expected commands:

```bash
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python scripts/run_experiment.py prepare
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python scripts/run_experiment.py validate
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python scripts/run_experiment.py evaluate
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python scripts/run_experiment.py report
```

## Final Mean Results

| Model | F1 | PR-AUC | ROC-AUC | High-conf FP | High-conf FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| One-stage 16-frame | 0.8758 | 0.9362 | 0.9731 | 58.0 | 35.0 |
| Two-stage 8+8 | 0.8735 | 0.9320 | 0.9718 | 53.7 | 29.0 |

The sequential model reduced high-confidence errors and increased recall, but
did not improve overall F1, PR-AUC, or ROC-AUC. See `FINAL_REPORT.md`.
