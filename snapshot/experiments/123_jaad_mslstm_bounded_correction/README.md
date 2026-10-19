# Exp123: bounded correction of a frozen MSLSTM baseline

Validation only. Exp122 A parent reused per seed; train matched raw-pose,
BIU56 and BIU63 correction branches. No test access or parent updates.
Fixed logit cap0.2 applied during fitting and prediction. This is not a10%
importance weight. Baseline epoch0 is retained as an AP-selection candidate.

Design: `DESIGN.md`. Progress: `logs/pipeline.log`, `logs/*.complete.json`.
Pretraining gate: `PRETRAINING_AUDIT_PASSED.json`, `FROZEN.json`.
Completion: `VALIDATION_COMPLETE.json` after9 trainings; `REPORT_COMPLETE.json`
after saved-prediction and correction-bound verification. Results and exact
commands: `FINAL_REPORT.md`. Error reachability/transitions and paired deltas:
`metrics/`. No automatic test or follow-up. Existing artifacts untouched.

GPU execution is detached with `subprocess.Popen(start_new_session=True)` and
persistent stdout/stderr in `logs/pipeline.log`, using the existing environment.
