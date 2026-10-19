# Experiment 031: JAAD API Label/Protocol Sensitivity

## Purpose

This planned experiment is a **benchmark/protocol validity audit**, not a new model-development experiment.

The question is:

> If a researcher relied on the problematic JAAD API-style labeling behavior instead of the corrected event-aligned anticipatory protocol, how much would reported pedestrian-intention performance change under the same modeling framework?

The clean JAAD benchmark remains the thesis benchmark regardless of outcome.

## Current Evidence Baseline

Two facts are already verified:

1. The raw JAAD hazard is real.
   - Tracks with `attributes['crossing']=1` and `crossing_point=-1` exist.
   - Treating them naively as positive throughout the whole track can create `2791` wrongly positive frames.
   - Evidence: [experiments/005_jaad_protocol_integrity_audit/report.md](/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/005_jaad_protocol_integrity_audit/report.md)

2. The current thesis benchmark artifacts are clean on this issue.
   - No observed crossing frames in the kept windows.
   - No positive samples using negative-`crossing_point` fallback.
   - No post-crossing windows.
   - Evidence: [experiments/005_jaad_protocol_integrity_audit/report.md](/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/005_jaad_protocol_integrity_audit/report.md)

## Proposed Comparison

### Protocol A — documented reconstruction of the upstream JAAD API-style flawed code path used by Noor's pipeline lineage

Protocol A is **not** described as an exact historical replay.

What is now recovered:

- Noor repository commit inspected: `65ecd1249797ca82806c3d373fec73f5f47b106c`
- Noor loader path: [data.py](/tmp/pose-to-intent-audit/data.py:77)
- Noor calls `JAAD.generate_data_trajectory_sequence(split, **data_opts)`
- Noor labels each observation window from the API-returned activity at the final observed frame:
  - `crossing = activity_seqs[i][w + obs_length - 1][0]`
  - [data.py](/tmp/pose-to-intent-audit/data.py:152)

Recovered Noor configuration:

- `fstride = 1`
- `sample_type = 'all'`
- `subset = 'default'`
- `data_split_type = 'default'`
- `seq_type = 'crossing'`
- `min_track_size = 76`
- observation length = `16`
- `TTE = [30, 60]`
- `overlap = 0.8`

What is still missing:

- Noor's repository does **not** contain a local `jaad_data.py`
- Noor's repository does **not** pin a JAAD upstream commit
- Noor's repository does **not** include a JAAD submodule
- the exact JAAD annotation/API snapshot that produced the thesis totals `39 tracks / 2791 frames` remains unrecovered

Therefore Protocol A is defined as:

- the recovered upstream JAAD `_get_crossing()` rule;
- Noor's recovered sequence-generation and windowing configuration;
- the current auditable JAAD API/annotation snapshot available for reconstruction.

### Protocol B — corrected / event-aligned

Use the existing clean benchmark lineage already used in the thesis:

- `16` observed frames
- JAAD `30_60`
- overlap `0.8`
- frozen train/val/test lineage
- pre-crossing event alignment
- no post-crossing positive windows

Primary clean split lineage:

- `/home/abdelrahman/Hadwa/BIU-PID-Distillation/data/jaad_splits_event_16_30_60_overlap08_default`

## Primary Modeling Recipe

Planned primary representation order:

1. `BBox`
2. if straightforward, `BBox + BIU-v2`

Planned benchmark backbone:

- models: `logreg`, `rf`, `xgb`
- seeds: `42`, `43`, `44`
- model selection: best validation `F1`
- same preprocessing and feature lineage where compatible

This is intentionally conservative. The audit is about protocol sensitivity, not about finding the strongest possible model.

## Temporal-Anchoring Caveat

Protocol A can differ from the clean benchmark in **two separate ways**:

1. label semantics
   - already-crossing or post-crossing observations can remain positive;
2. temporal anchoring
   - Noor's nominal `30_60` windowing measures `TTE` relative to the end of the API-generated crossing sequence, not necessarily to a verified physical frame-level crossing onset when `crossing_point == -1`.

These two effects must be reported separately in the experiment outputs.

## Controlled-Comparison Rule

This should be reported as:

1. same-model / same-training-recipe protocol comparison

and only additionally as:

2. matched-overlap sensitivity

if a legitimate overlapping subset can be identified.

If the flawed protocol changes the available windows, this is **not** a matched-cohort causal comparison and must not be described that way.

## Required Population Audit

For each protocol, the planned report should include:

- total samples
- positive samples
- negative samples
- positive rate
- unique tracks
- train / val / test counts
- number of `crossing=1, crossing_point=-1` tracks
- number of post-crossing or already-crossing windows kept
- number of windows specifically affected by the API flaw
- number of windows containing frame-level `cross=1`
- number of windows entirely after true crossing onset where recoverable
- number of windows partly overlapping actual crossing
- number of positive windows whose nominal TTE is not anchored to a verified crossing onset
- nominal API-style TTE distribution
- physical TTE distribution where recoverable
- observation / event relationship

The population audit is a primary result, not only setup bookkeeping.

## Count Reconciliation To Preserve

The report must keep these four quantities separate:

1. Full raw hazard from Exp005
   - `167` tracks with `crossing=1` and `crossing_point=-1`

2. Noor default-split hazard under recovered upstream replay
   - `161` such tracks in Noor's default train/val/test split
   - split counts: `95 / 6 / 60`

3. Reconstructed current-upstream affected population
   - `41` affected tracks
   - `2637` wrong frames

4. Historical thesis report
   - `39` affected tracks
   - `2791` wrong frames

The discrepancy between `(41, 2637)` and `(39, 2791)` should be described as a provenance limitation consistent with an unrecovered historical JAAD annotation/API snapshot.

## Most Important Diagnostic

Beyond metrics, the central audit question is whether any apparent performance increase is driven by easier positives, especially windows where:

- the pedestrian is already crossing;
- gross BBox motion is already obvious;
- the observation is effectively post-event rather than anticipatory;
- track-level crossing metadata keeps positive labels alive after the crossing has already happened.

## Methodological Concern

The main unresolved issue is now narrower:

- the **mechanism** is recovered from upstream JAAD and Noor's pipeline lineage;
- the exact **historical JAAD snapshot** used during Noor's thesis run is still unknown.

So Exp031 can no longer be described as a generic or memory-based reconstruction.

But it still cannot be described as an exact historical replay.

## Self-Review

### What this design does well

- Keeps the clean benchmark untouched.
- Treats the flawed protocol only as an audit target.
- Uses the same modeling backbone to isolate protocol sensitivity rather than model novelty.
- Grounds Protocol A in recovered upstream JAAD code plus Noor's own loader/windowing configuration.
- Forces a population-difference analysis instead of over-interpreting a single F1 delta.

### What remains risky

- Protocol A still depends on an unrecovered historical JAAD API/annotation snapshot if the goal were to match Noor's thesis totals exactly.
- Apparent performance differences may come from both label semantics and temporal-anchor drift; they must not be collapsed into a single explanation.
- The primary result will likely remain a protocol-level sensitivity comparison rather than a matched-cohort causal comparison.
