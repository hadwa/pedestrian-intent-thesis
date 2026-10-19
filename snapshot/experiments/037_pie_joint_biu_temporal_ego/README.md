# Exp037: PIE Joint BIU Temporal-Ego

## Status

`COMPLETED - NO ADVANCE`.

This experiment tests a raw joint BIU-v1 + BIU-v2 + BBox representation,
fixed observation-only temporal contrasts, and exact-frame ego speed. It uses
PIE's 16-frame, TTE `30_60`, overlap `0.6`, fixed-`0.5` protocol.

Development is leave-one-set-out on set01/set02/set04/set05. Set06 is locked
internal confirmation. Set03 is conditional diagnostic evaluation only and is
not independent confirmation because Exp033 and Exp036 already exposed its
errors and aggregate behavior.

No threshold tuning, post-hoc probability blending, future frames, or new pose
estimation was used. The development package covered exactly 4,781 windows and
all six frozen arms. The strongest arm, `joint_temporal_ego_xgb`, improved OOF
PR-AUC from `0.728049` to `0.812923` versus `v2_bbox_xgb`, and fixed-0.5 F1 from
the best baseline's `0.715565` to `0.776613`. Its paired 10,000-track-bootstrap
lower bounds were positive for both deltas, and it did not regress on any PIE
development set.

The arm was not selected because the preregistered tail gate required no more
high-confidence errors than the minimum across unlike RF/XGB baselines. It had
132 false positives at probability at least 0.9 and 110 false negatives at
probability at most 0.1, versus limits of 21 and 43 from `v1_bbox_rf`.
Consequently set06 was not opened, and the conditional set03 diagnostic loader
was not called. The physically combined keypoint JSONL was necessarily streamed
under the preregistered shared-file exception; no set03 record was retained and
no set03 split row, feature, prediction, or metric was materialized. The source
does contain a `label` field, so calling it "unlabeled" in the frozen contract
was imprecise. The hardened loader now rejects unauthorized records before JSON
decoding, and a read-only reconstruction verified the saved development matrix.
See `RESULTS.md`, `metrics/development_results.csv`,
`metrics/development_bootstrap.csv`, and `metrics/verification_report.json`.

Exact source paths, sizes, and hashes are in
`metrics/source_provenance_prepare.json`. Exact commands are in `RESULTS.md`.

## Research Basis

The PIE and PCPA papers motivate combining pedestrian motion/pose with BBox and
ego-vehicle information. The group-robustness paper motivates explicit
cross-group stress testing and regularization. These papers motivate the design;
they do not establish that Exp037 will improve this repository's PIE benchmark.

Paper copies and download provenance are recorded in `papers/README.md`.
