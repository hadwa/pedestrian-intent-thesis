# Exp116: missing 56-D BIU-v2 XGBoost comparisons

User-approved completion of the missing standalone core56 tree pathways, before
any decision about neural retraining. See [DESIGN.md](DESIGN.md) for the frozen
11 configurations and [FINAL_REPORT.md](FINAL_REPORT.md) when complete.

- New: 33 tree fits, three seeds for each of 11 missing configurations.
- Reused: parent feature caches, pose estimates, road/ego/vehicle features,
  model recipes, sample definitions and saved parent predictions.
- Not rerun: existing core56 PIE simple/extended, clean JAAD vehicle-only, and
  JAAD_beh/all simple comparisons. No pose extraction or neural training.
- Interpretation: simple arms compare core56 against full63; extended arms
  compare a core56 static pathway against historical combined BIUs, with the
  existing temporal and contextual inputs retained. These are different contrasts.
- Test discipline: all new fits and validation predictions freeze before final
  test access. All predeclared arms are evaluated, with no validation/test-based
  selection. Previously exposed cohorts remain retrospective comparisons.

## Run and inspect

Working directory: `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean`.

```bash
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python -u experiments/116_xgboost_v2_core_completion/scripts/run.py
tail -n 20 experiments/116_xgboost_v2_core_completion/logs/run.log
```

This retains the parents' four-thread CPU XGBoost recipe. No tmux is used.
Small cached-feature tree fitting does not require perception on GPU.

## Evidence

- `FROZEN.json`: code/design hashes before training; pretraining-only repair
  snapshots are retained in `logs/`.
- `PREPARED.json`, `metrics/pretraining_invariants.json`: cohort, feature,
  deterministic recovery and parent validation-prediction parity checks.
- `TRAINED.json`: hashes for every new model and validation prediction before
  the test gate opens.
- `metrics/all_sources.json`: exact source paths, SHA-256 and file sizes.
- `cache/<case>/<split>/columns.json`: complete ordered feature names.
- `metrics/per_seed_results.csv`, `metrics/summary_results.csv`: fixed-0.5
  metrics, with average precision used for PR-AUC and sample SD across seeds.
- `metrics/paired_deltas.csv`: new core56 minus the corresponding saved parent.
- `metrics/error_transitions.csv`: parent mistakes corrected and correct
  decisions harmed by the replacement.
- `metrics/fit_times.csv`: measured tree fitting durations, not end-to-end time.
- `EXPERIMENT_COMPLETE.json`: written only after all 33 fits, final evaluations,
  source immutability and saved-prediction metric checks pass.

No changes to existing experiments, thesis, seminar, or benchmark results.
