# Exp119: missing v2-only capped PedCMT controls

Frozen before new training or test evaluation, 2026-09-18. Approved scope: fill missing
comparisons for the seminar, not search for favorable outcomes. Retrospective matched
controls on previously exposed PIE and clean-JAAD cohorts. Not API_beh or API_all.

## Reuse and missing arms

- Direct-token PIE baseline/raw/63: Exp084; core56: Exp117.
- Direct-token clean-JAAD baseline/raw/63: Exp089; core56: Exp117.
- PIE MSLSTM baseline/63: Exp053; raw: Exp095; core56: Exp117.
- Capped clean-JAAD raw: reuse Exp094 saved predictions; baseline Exp089.
- New capped PIE: raw, core56 and full63; new capped clean-JAAD: core56 and full63.
- Five new arms times seeds 42/43/44 = 15 trainings. Do not reuse union135 as v2.
- JAAD MSLSTM is a separate missing family requiring an input/coverage audit before
  training; it is not silently equated to Exp049 zero-shot 30-frame results.

## Fixed protocol and representation

16 consecutive observation frames, TTE30--60. Exact source cohort/order/labels:
PIE Exp084 via Exp050 (2366 train, 3619 validation, 3760 test); clean JAAD Exp089
(1552 train, 274 validation, 1501 test). Fixed threshold 0.5. No changes to pose
extraction, cohorts, labels, non-pose inputs, frozen parent or stored scaling.
56 columns are selected by Exp012 core feature names, not assumed positions.
Full63 is the existing BIU-v2 core plus seven auxiliaries. Both inherit the frozen
BBox direction fallback. Raw pose is standardized XY [16,34], no confidence input.
All scaling is reconstructed from training rows and checked against source statistics.

Reuse Exp090's exact residual architecture and fitting routine. BIU encoder input
changes from 135 to 56 or 63 only: Linear(d,128), GELU, Dropout(.3), Linear(128,128),
LayerNorm. Parent anchor 256->128; concatenate anchor, extra, product and absolute
difference; existing 512->64->1 head; correction 2*tanh. Raw uses Exp090 temporal
encoder unchanged. Parent remains frozen. Training uses the full correction, as in
Exp093/094; inference is sigmoid(parent_logit + 0.10*correction). No cap search.

Adam 5e-4, betas(.9,.98), eps1e-6; batch64; up to200 epochs, patience25; minimum
validation uncapped BCE, earliest tie. PIE uses Exp093/084 balanced indices; JAAD
uses Exp091/094 natural training indices. These dataset-specific recipes are kept
constant across representation arms within each dataset. No augmentation or search.

## Gates, reporting and interpretation

Code/design/source hashes frozen; all new checkpoints and validation predictions
must exist and be hashed before any final-test data are accessed. Report every arm,
without a performance gate, regardless of outcome. Save per-seed and mean/sample-SD
fixed-.5 metrics, AP, ROC, confusion counts, paired deltas and error transitions.
Also report uncapped saved-branch probabilities as a predeclared diagnostic, not a
new selectable candidate; never select cap or model using final results.
Compare capped56/63 with matched capped raw and frozen parent. Existing direct-token
results are separate families. Differences across fusion families are not a pure
representation-only effect. No claim that v2 must improve on raw pose or that any
added computation proves end-to-end speed. Count parent plus branch, not branch alone.

No original artifact writes, no thesis edits, no automatic follow-up search, no tmux.
Use GPU with persistent logs and atomic markers. EXPERIMENT_COMPLETE is only written
after all 15 trainings and both final evaluations succeed.
