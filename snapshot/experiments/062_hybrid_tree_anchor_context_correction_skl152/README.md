# Exp062: Hybrid Tree-Anchor Context Correction on scikit-learn 1.5.2

## Question

Can automatic ego-motion and pedestrian-centred YOLOP context correct errors
from a frozen BBox+BIU tree anchor on the PIE 16-frame, TTE 30--60 benchmark
without giving weaker modalities an equal vote?

## Frozen Protocol

- Cohorts: train sets01/02/06 (2,366 rows), validation sets04/05 (3,619 rows),
  and final set03 (3,760 rows).
- Seeds: 42, 43, and 44. The only decision threshold is fixed `0.5`.
- Anchors: fresh `rf_v1_bbox`, a Random Forest over the exact Exp012
  BIU-v1+BBox representation (53 archived enriched BBox features followed by
  72 BIU-v1 values including its two historical quality counters, 125-D), and
  fresh `xgb_v2_bbox`, an XGBoost classifier over the exact Exp012
  BIU-v2-core+BBox representation (the same 53 enriched BBox features plus
  the frozen 56-D BIU-v2 core, 109-D). The authoritative feature-set names,
  ordering, and definition hash are frozen in `ExperimentConfig`; dimension
  equality alone is not sufficient. Historical Exp012 tree objects were not
  serialized, so Exp062 must refit them and must never claim to reuse tree
  weights.
- Context: train-only standardized 16x3 ego and 16x20 YOLOP sequences, used
  by a 5,932-parameter bounded neural correction. The tree remains frozen and
  non-neural.
- Exclusions: raw pose, RGB, GT semantics/context, future frames, TTE input,
  calibration, threshold search, alternative road features, and new anchors.

The authoritative scientific specification is
`docs/superpowers/specs/2026-08-25-exp061-hybrid-anchor-context-correction-design.md`;
the authorized runtime-only remediation is
`.superpowers/sdd/2026-08-25-exp061-hybrid-tree-context-correction/exp062-remediation-brief.md`.
Implementation cannot override either approved document.

## Development Boundary

All model and checkpoint decisions are validation-only. Set03 must not be
constructed, opened, statted, or hashed before an authenticated final candidate
freeze is written. `FROZEN_FINAL_MODEL.json` may be created only after all ten
predeclared validation gates pass. Its contract recomputes the gate, binds the
winning anchor family, and requires an authenticated candidate-manifest artifact
before freeze. That manifest canonically binds the config and source manifest,
the exact seeds 42/43/44, checkpoint SHA-256 values, candidate and anchor
validation summaries, and validation-artifact hashes. The final-freeze contract
rebuilds and verifies that manifest against its arguments before hashing it into
`FROZEN_FINAL_MODEL.json`. If neither corrected candidate passes every gate,
Exp062 closes validation-only and does not access set03.

Prior set03 exposure in Exp012 and later post-hoc work is disclosed in the
audit; it does not authorize iterative Exp062 test use.

## Contract Checks

```bash
experiments/062_hybrid_tree_anchor_context_correction_skl152/.venv/bin/python -m pytest experiments/062_hybrid_tree_anchor_context_correction_skl152/tests -q
experiments/062_hybrid_tree_anchor_context_correction_skl152/.venv/bin/python -m compileall experiments/062_hybrid_tree_anchor_context_correction_skl152/scripts
```

Pinned runtime packages are recorded in `requirements.lock`; wheel provenance
and the Exp061 failure boundary are recorded in `RUNTIME_PROVENANCE.md`. Every
public phase and test must use the experiment-local `.venv/bin/python`.
Generated caches,
models, predictions, metrics, logs, and phase markers remain uncommitted unless
explicitly requested.

## Guarded Runner

All executable phases use `scripts.runner`. Each command re-hashes the complete
authenticated predecessor chain before continuing. `stage-sources` is the only
command allowed to read mixed historical containers. It writes immutable,
train/validation-only Exp012 anchor matrices and six validation-only parity
references under `staged_sources/`, with original and derived hashes recorded
in `SOURCE_STAGING.json`. Every later pre-freeze phase authenticates and reads
only these isolated files; restaging or mutation is refused. `prepare` creates its own
`logs/` directory and persists `logs/prepare.log`; callers must not depend on an
external `tee` creating that directory.

```bash
experiments/062_hybrid_tree_anchor_context_correction_skl152/.venv/bin/python -m experiments.062_hybrid_tree_anchor_context_correction_skl152.scripts.runner stage-sources
experiments/062_hybrid_tree_anchor_context_correction_skl152/.venv/bin/python -m experiments.062_hybrid_tree_anchor_context_correction_skl152.scripts.runner prepare
experiments/062_hybrid_tree_anchor_context_correction_skl152/.venv/bin/python -m experiments.062_hybrid_tree_anchor_context_correction_skl152.scripts.runner fit-anchors
experiments/062_hybrid_tree_anchor_context_correction_skl152/.venv/bin/python -m experiments.062_hybrid_tree_anchor_context_correction_skl152.scripts.runner train --anchor rf_v1_bbox --seed 42
experiments/062_hybrid_tree_anchor_context_correction_skl152/.venv/bin/python -m experiments.062_hybrid_tree_anchor_context_correction_skl152.scripts.runner validate
experiments/062_hybrid_tree_anchor_context_correction_skl152/.venv/bin/python -m experiments.062_hybrid_tree_anchor_context_correction_skl152.scripts.runner freeze
experiments/062_hybrid_tree_anchor_context_correction_skl152/.venv/bin/python -m experiments.062_hybrid_tree_anchor_context_correction_skl152.scripts.runner final --threshold 0.5
experiments/062_hybrid_tree_anchor_context_correction_skl152/.venv/bin/python -m experiments.062_hybrid_tree_anchor_context_correction_skl152.scripts.runner verify
```

Run `train` once for each frozen anchor and seed combination. The aggregate
`TRAINING_COMPLETE.json` appears only after all six authenticated run markers
exist. Long phases may run in a detached `tmux` session, but the runner-owned
markers and logs, not the terminal session, define resumability.

The training boundary accepts only a persisted Task 3 `GroupedOOFResult` with
exactly one held-out assignment per training row, five zero-overlap track
audits, exact fold-model hashes, frozen anchor/seed/feature identity, and the
saved OOF vector hash. Validation scores must be the saved parity-verified
full-training anchor vector. Bare, in-sample, mutated, or cross-seed vectors
are rejected before `train_one_seed` is called.

Set03 authorization is impossible before a hash-valid
`FROZEN_FINAL_MODEL.json` proves an eligible ten-gate decision and binds all
three checkpoints, all three full anchors, the train-only scaler, validation
summary, every validation prediction/evidence file, the recursively verified
prepare-to-validation marker chain, config, source staging manifest, and final
arguments. The test builder import
is inside the authorized `final` command body. A validation-only decision and
a final freeze are mutually exclusive, and only a passing `verify` invocation
may create `EXPERIMENT_COMPLETE`. Repeating a successful verification is
idempotent; a failed verification removes only an invalid stale completion
marker.

Final authorization does not trust the declared winner. It recomputes the
anchor and corrected metrics for both candidate families from the authenticated
per-seed validation predictions, applies all ten gates independently, and
replays the frozen harmonic, PR-AUC, F1, then high-confidence-error selection
order. Both families' prediction hashes and summary digests are frozen. The
required marker topology is direct and path-exact:
`VALIDATION_COMPLETE.json -> TRAINING_COMPLETE.json -> ANCHORS_READY.json ->
PREPARED.json -> SOURCE_STAGING.json`; recursive discovery cannot replace a
missing direct link.

Final verification recomputes every metric, confusion count, and
high-confidence count from each saved Set03 prediction file at fixed `0.5`.
It then recomputes and checks the three-seed mean and population standard
deviation for every persisted summary field.
