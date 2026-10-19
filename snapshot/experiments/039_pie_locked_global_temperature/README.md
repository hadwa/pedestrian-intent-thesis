# Exp039: PIE Locked Global-Temperature Confirmation

## Status

`COMPLETED - NO ADVANCE`.

Exp039 is a locked confirmation of Exp037's selected
`joint_temporal_ego_xgb` model with global post-hoc softening derived only from
Exp038 development folds. It does not search models, features, temperatures,
or thresholds. Set06 is the untouched confirmation cohort. Set03 is opened
only if every set06 gate passes and is then labeled test-informed diagnostic.

## Frozen Setup

- PIE standard anticipatory benchmark, 16 observations, TTE 30--60, overlap 0.6.
- Fixed threshold 0.5 and Exp037 seeds 42, 43, 44.
- Joint temperature: `1.8395656892513026`.
- BIU-v2+BBox temperature: `2.0149628788881717`.
- BIU-v1+BBox temperature: `1.0`.
- Models and feature representations are unchanged from Exp037.

## Research Basis

Guo et al., “On Calibration of Modern Neural Networks,” ICML 2017, motivates
temperature scaling. It does not establish the method for PIE or tree
ensembles; transfer is tested here rather than assumed. The official local
copy is `papers/Guo_2017_Calibration.pdf`, with provenance in
`papers/README.md`.

The complete frozen design is in
`docs/superpowers/specs/2026-08-15-pie-locked-global-temperature-exp039-design.md`.

## Outcome

The raw Exp037 joint model transferred strongly to set06: PR-AUC `0.826487`
and fixed-0.5 F1 `0.799127`, versus `0.729741` and `0.764444` for the v2+BBox
XGB baseline. The PR-AUC gain and F1 point-estimate gain gates passed. The
paired track bootstrap supported the PR-AUC gain (95% interval
`[0.026048, 0.180744]`) but not the F1 gain (95% interval
`[-0.008042, 0.078811]`).

Global softening preserved ranking, fixed-0.5 decisions, and the confusion
matrix exactly. It reduced the joint model's combined high-confidence error
tail from `30` to `10`, but worsened Brier score (`0.057213` to `0.059440`),
log loss (`0.203138` to `0.218273`), and 10-bin ECE (`0.044206` to
`0.073345`). Exp039 therefore did not advance and set03 remained closed.

The result supports the raw joint model's discrimination, not the frozen
global temperature. High-confidence-tail reduction alone is insufficient when
proper scoring rules and calibration error worsen.

## Execution Note

The first locked `confirm` invocation produced no files or metrics and left a
stalled tool shell after its worker disappeared; it was terminated. The same
frozen command was rerun without changing or inspecting any set06 result. This
is an operational repeat access to set06 and is recorded conservatively in
`EXECUTION_LOG.md`; it did not introduce adaptive selection or optional
stopping.

## Commands

```bash
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python scripts/run_experiment.py prepare
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python scripts/run_experiment.py confirm
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python scripts/run_experiment.py evaluate
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python scripts/run_experiment.py report
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python scripts/run_experiment.py verify
```
