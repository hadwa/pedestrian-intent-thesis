# Exp096: PIE Exp072 Matched BIU Ablation

This experiment closes the missing BIU-version control inside the Exp072 XGBoost architecture.
It reuses Exp072's authenticated prepared matrices and recomputes only twelve lightweight XGBoost
fits: four feature arms across seeds 42, 43, and 44.

Run with:

```bash
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python \
  experiments/096_pie_exp072_matched_biu_ablation/scripts/run_experiment.py all
```

The final result is a retrospective matched control because PIE set03 has prior exposure elsewhere
in the project. Exact sources, feature definitions, and interpretation boundaries are recorded in
`DESIGN.md`, `metrics/source_audit.json`, and `FINAL_REPORT.md`.

`all` also computes the supplemental paired track-bootstrap comparisons of every BIU arm against
the shared 64-feature base. Existing trained models and predictions are authenticated and reused
when completed phases are rerun.
