# Exp062 Provenance And Invariant Audit

## Status

Status: **CONTRACT FOUNDATION ONLY - TRAINING NOT AUTHORIZED BY THIS FILE.**
This document freezes the evidence and invariants that later phase-specific
audits must authenticate. It contains no benchmark result.

## Authenticated Reused Inputs

The primary experiment reuses only the following authenticated train/validation
caches after their hashes, row order, labels, cohort membership, and shapes are
verified:

- `experiments/055_compact_residual_biu_lstm/cache/trainval/rows.csv`:
  `a01ad1f3244157c1b649aada0470ad54d877bbd1e8075c275ee313c35dcc3bf1`.
- `experiments/055_compact_residual_biu_lstm/cache/trainval/ego.npy`:
  `410c4757e4012e223721f5686bffe5f8e4c4f53dd9fd9030e763fcfbe58dd302`.
- `experiments/055_compact_residual_biu_lstm/cache/trainval/yolop.npy`:
  `b353bf9d8fb7d5eedacf726fec643f67138994f5e9a864806f46e382075b65f5`.

The authoritative split lineage is
`data/splits_pie_wacv_pcpa_30_60_overlap06` in the read-only archive. Context
statistics must be fit only on the 2,366 original training rows; validation
and set03 rows must never influence those statistics.

## Feature Construction Parity

The authenticated loader calls the authoritative Exp012 feature-set builder
with only the authenticated Exp062 train/validation rows. Its historical
feature-definition source is
`experiments/012_pie_biu_v2_30_60_benchmark/metrics/pie_biu_v2_feature_set_definitions.csv`:
`6708d290d9263ad29000786cf6d15c5ec052dc8712c1b5a16741944295d0145c`.

The runtime anchor reconstruction calls the Exp012 builder using its actual
train/validation BIU and BBox input artifacts, then aligns the outputs to the
authenticated Exp055 row mirror by sequence ID, split order, and label. The
Exp055 `summary_v1.npy` and `summary_v2_core.npy` hashes remain authenticated
lineage evidence, but their values are not used as Exp062 anchors because they
failed exact parity against the authoritative Exp012 builder. The recomputed
`125`-D and `109`-D arrays are not new feature definitions. The source-value
check is `np.testing.assert_allclose(..., rtol=0, atol=0)`.

| source rows hash | Exp012 definition hash | max_abs_difference |
| --- | --- | --- |
| `a01ad1f3244157c1b649aada0470ad54d877bbd1e8075c275ee313c35dcc3bf1` | `6708d290d9263ad29000786cf6d15c5ec052dc8712c1b5a16741944295d0145c` | `0.0` |

## Refit, Do Not Reuse

Exp012 saved historical validation/test probabilities and metrics, but did not
serialize tree objects. Exp062 therefore recomputes fresh Random Forest and
XGBoost tree objects using the exact approved configurations. They are not
reused authenticated caches and they must pass fresh validation-probability
parity before any context-network training. Later phases must record each
recomputed tree object's SHA-256, size, tree count, node count, and leaf count.

## Invariants

- The row cohorts, row order, labels, sequence IDs, observation windows, fixed
  threshold, dimensions, seeds, and source hashes are immutable.
- The anchor parity tolerance is `1e-7`; metrics recomputed from compared
  probabilities must agree within `1e-10`.
- OOF anchor predictions must cover every original training row exactly once
  with no `track_id` crossing a fold boundary.
- Every phase marker is atomically written JSON and hashes its frozen config,
  source manifest, producing command, and all declared input artifacts.
- `FROZEN_FINAL_MODEL.json` is an authenticated final-freeze marker, not a
  nominal filename: creation requires the contract to recompute all ten gates
  as eligible and to verify a concrete candidate-manifest artifact. The manifest
  binds the selected anchor, validation/anchor gate summaries, config/source
  manifest, validation artifacts, and exactly three frozen-seed checkpoints;
  altered metric or checkpoint inputs are rejected before final-freeze creation.
- No set03 source may be constructed, opened, statted, or hashed between Exp062
  preparation and an immutable final-candidate freeze.

## Prior Exposure

Set03 was accessed in Exp012 and visually inspected in later post-hoc work.
The retention of the two anchor families is informed by historical set03
metrics. Exp062 is therefore not a pristine unseen-test discovery. This prior
set03 exposure is a disclosed interpretation limit, not permission to use set03
for Exp062 design, threshold, checkpoint, or iteration decisions.
