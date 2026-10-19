# Exp122: clean-JAAD MSLSTM without measurement noise

Validation-only controlled successor to Exp121. Preserve balanced oversampling,
all four arms/seeds and AP checkpoint selection; remove Gaussian augmentation
only. Duplicated measurements are now exact copies. No test evaluation.

Protocol: `DESIGN.md`. Progress: `logs/pipeline.log`, `logs/*.complete.json`.
Pretraining checks: `PRETRAINING_AUDIT_PASSED.json`, `FROZEN.json`.
Completion: `VALIDATION_COMPLETE.json` after12 runs, `REPORT_COMPLETE.json`
after24 saved-probability verifications. Results: `FINAL_REPORT.md` and
`metrics/`. Secondary F1 checkpoints are diagnostic, not a change of primary
selection. Prior experiments and thesis untouched; no commit or push.

GPU execution uses the existing environment and detached `subprocess.Popen`
with `start_new_session=True`, stdout/stderr in `logs/pipeline.log`, no tmux.
Exact reproducible command lines and source paths are in the final report.
