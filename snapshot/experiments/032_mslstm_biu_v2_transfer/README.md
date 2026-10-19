# 032 — MSLSTM + BIU-v2 Transfer Under the Recovered Published-Setting MSLSTM Lineage

Status: corrected controlled run complete; result is mixed/inconclusive and
diagnostic-only.

The root-level `metrics/` and `predictions/` files are a 12-row smoke run and
must not be quoted as experiment results. The dated `full_run_2026-08-14/`
attempt was interrupted during the first epoch and produced no model metrics.
It also incorrectly used all 333,454 reconstructed training windows for a
40-epoch schedule. The recovered notebook lineage instead uses the immutable
balanced `MSLSTM-PID/checkpoints/train_index.pkl` (109,934 rows, exactly 54,967
per class) and five epochs. These constraints are now enforced by the runner.

The non-overwriting corrected execution is under
[`full_run_corrected_2026-08-16/`](./full_run_corrected_2026-08-16/). Its full
result and limitations are reported in
[`RESULTS.md`](./full_run_corrected_2026-08-16/RESULTS.md). The run reuses only
the deterministic BIU cache from the interrupted preparation; cache hashes,
live-recomputation checks, and provenance are recorded in that directory.

This experiment is a controlled transfer test, not a new benchmark sweep. It asks:

> Can BIU-v2 improve an MSLSTM-style temporal pedestrian-intention model when BIU-v2
> and the raw pose stream are derived from the same underlying pose measurements used
> by the recovered MSLSTM pipeline lineage?

## What Exp032 is

- A PIE-only transfer/generalization experiment.
- A fresh `Variant A` control and `Variant B` BIU-augmented model trained under the
  recovered local MSLSTM protocol lineage rather than the standardized Exp018 benchmark.
- A test of representation/inductive bias, not of new physical information.

## What Exp032 is not

- Not a rerun of Exp018.
- Not a new SOTA benchmark claim.
- Not a confidence/reliability experiment.
- Not a JAAD experiment.
- Not an architecture-search project.

## Key distinction from Exp018

Exp018 answered: how does MSLSTM behave under this repository's standardized
anticipatory PIE benchmark?

Exp032 is designed to answer a different question:

> Does BIU-v2 transfer into an independently developed temporal MSLSTM pipeline under
> that pipeline's own recovered task construction?

## Current deliverables

- [PUBLISHED_PROTOCOL_AUDIT.md](./PUBLISHED_PROTOCOL_AUDIT.md)
- [DESIGN.md](./DESIGN.md)
- [Corrected run result](./full_run_corrected_2026-08-16/RESULTS.md)

## Evidence roots used for the audit

- Local MSLSTM repo: `/home/abdelrahman/Hadwa/MSLSTM-PID`
- Standardized retrain reference: [../018_mslstm_retrain_benchmark/README.md](../018_mslstm_retrain_benchmark/README.md)
- Archive audits:
  - `/home/abdelrahman/Hadwa/BIU-PID-Distillation/reports/MSLSTM_PRED_LEN_VERIFICATION_REPORT.md`
  - `/home/abdelrahman/Hadwa/BIU-PID-Distillation/reports/MSLSTM_PID_REPO_AUDIT_REPORT.md`
  - `/home/abdelrahman/Hadwa/BIU-PID-Distillation/reports/MSLSTM_PID_STREAM_AVAILABILITY_REPORT.md`
  - `/home/abdelrahman/Hadwa/BIU-PID-Distillation/reports/MSLSTM_PID_TEACHER_INPUT_DRY_RUN_REPORT.md`
  - `/home/abdelrahman/Hadwa/BIU-PID-Distillation/reports/MSLSTM_INPUT_SEMANTICS_AUDIT_REPORT.md`

## Current execution rule

- Published paper metrics are descriptive reference values only.
- Fresh `Variant A` is the true controlled baseline.
- Scientific interpretation must come from the paired fresh
  `Variant A -> Variant B` comparison under identical runner/cohort/training conditions.

## Result in one sentence

Variant B reduced validation fixed-0.5 F1 in every seed (mean delta `-0.01152`),
while increasing inspected-set03 diagnostic F1 in every seed (mean delta
`+0.01128`); PR-AUC changes were trivial and inconsistent, so Exp032 does not
establish a general BIU-v2 transfer improvement.
