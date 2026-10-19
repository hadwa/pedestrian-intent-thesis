# Exp121: clean-JAAD MSLSTM validation-AP selection

Controlled, retrospective validation-only successor to Exp120. The only model
procedure change is checkpoint selection: maximum validation AP instead of F1,
with earliest tie. Both selections are saved from identical replay trajectories.
All four arms and three seeds are retained; no selective reporting.

- A: BBox53 + categorical vehicle state + historical YOLOP.
- B: A + raw pose.
- C: A + canonical BIU-v2 56-D.
- D: A + BIU-v2 + seven auxiliaries,63-D.

No measured JAAD speed or acceleration is fabricated. No test access,
calibration, fusion change, pose extraction, or augmentation modification.
The augmentation measurement audit is descriptive only. No thesis modifications.

`DESIGN.md` fixes the procedure. `metrics/checkpoint_inventory.json` explains
why retraining was necessary. `FROZEN.json` and
`PRETRAINING_AUDIT_PASSED.json` record the pretraining gate. Persistent progress:
`logs/pipeline.log` and `logs/*complete.json`. `VALIDATION_COMPLETE.json` is
created only after all12 replay/parity checks succeed; `REPORT_COMPLETE.json`
requires verification of all24 saved validation prediction files.

Final results: `FINAL_REPORT.md`, `metrics/per_seed_validation.csv`,
`metrics/summary_validation.csv`, `metrics/paired_selection_deltas.csv`.
Perturbation audit: `metrics/augmentation_diagnostic_seed*.csv`.

The process was launched with `subprocess.Popen(..., start_new_session=True)`
on the existing CUDA environment, with stdout/stderr appended to
`logs/pipeline.log`; no tmux or dependency installations. Exact training and
report commands are recorded in `FINAL_REPORT.md` by the report script.

AP selection mechanically maximizes validation AP. An increase over F1-selected
validation AP is not independent proof of better generalization. No automatic
test evaluation or follow-up architecture experiment is included.
