# JAAD API Flaw Definition

## Status

This note freezes the repository-supported definition of the previously discussed JAAD API labeling flaw **before** any protocol-sensitivity experiment is executed.

It does **not** treat the flawed protocol as a valid benchmark.

## Repository Evidence Used

- [experiments/005_jaad_protocol_integrity_audit/report.md](/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/005_jaad_protocol_integrity_audit/report.md)
- [papers/bachelor-4.pdf](/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/papers/bachelor-4.pdf)
- [build_jaad_event_aligned_candidates.py](/home/abdelrahman/Hadwa/BIU-PID-Distillation/src/benchmark/build_jaad_event_aligned_candidates.py:287)
- Noor repository commit `65ecd1249797ca82806c3d373fec73f5f47b106c`
- Upstream JAAD commit `d2feeac998366e47ecc54e5533c8a3fa61141dda`
- Upstream JAAD historical blame showing the same mechanism already present at `a04cb8511b85413bbddafd049911da1b8c3c214e`

## Noor Pipeline Provenance

Recovered Noor code path:

- Noor repository commit inspected: `65ecd1249797ca82806c3d373fec73f5f47b106c`
- Loader path: [data.py](/tmp/pose-to-intent-audit/data.py:77)
- Sequence generator call:
  - `JAAD.generate_data_trajectory_sequence(split, **data_opts)`
- Window label rule:
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

Noor repository limitations:

- no local `jaad_data.py`
- no JAAD git submodule
- no pinned JAAD upstream commit

Therefore the exact historical JAAD API / annotation snapshot used in Noor's thesis remains unknown.

## Verified Upstream Mechanism

The relevant upstream JAAD code path is `_get_crossing()` in:

- [jaad_data.py](/tmp/jaad-upstream-audit/jaad_data.py:1125)

Verified upstream behavior:

- for behavior pedestrians, `event_frame` is taken from `crossing_point`;
- if `crossing_point == -1`, the generator uses `end_idx = -3`;
- the retained activity sequence is created from the track-level attribute:
  - `int(attributes['crossing'] > 0)`
  - repeated across the retained sequence;
- the generator does **not** use the true frame-level `behavior['cross']` sequence for those retained frames.

This exact mechanism is already present by upstream commit:

- `a04cb8511b85413bbddafd049911da1b8c3c214e`

and remains present in later inspected commits including:

- `2c8803ca0d727a19500bfcb5b8489c0c7dec67a1`
- `3a0a4a4e6e6884e3144eee268095c0380a2b5943`
- `d2feeac998366e47ecc54e5533c8a3fa61141dda`

## What The Flaw Is

The flaw is **not** a problem in the current clean event-aligned JAAD benchmark artifacts.

The flaw is a **different JAAD API-style labeling path**, documented in Noor's bachelor thesis and now traced to upstream JAAD `_get_crossing()`, where crossing labels are assigned from track-level metadata rather than from the per-frame crossing behavior sequence.

For tracks with:

- `attributes['crossing'] = 1`
- `crossing_point = -1`

the verified upstream behavior is:

- treat the track as a crossing track;
- keep nearly the whole track (`end_idx = -3`, effectively dropping only the final two frames);
- assign the retained activity sequence from track-level `crossing > 0`;
- ignore whether `behavior['cross']` later returns to `0` after the physical crossing has already happened.

Under that behavior, post-crossing frames can remain effectively positive.

## Population Reconciliation

Keep these four populations distinct:

1. Full raw hazard from Exp005
   - `167` tracks with `crossing=1` and `crossing_point=-1`

2. Noor default-split hazard under recovered upstream replay
   - `161` such tracks in Noor's default train / val / test split
   - split counts: `95 / 6 / 60`

3. Reconstructed current-upstream affected population
   - `41` affected tracks
   - `2637` wrong frames

4. Historical thesis report
   - `39` affected tracks
   - `2791` wrong frames

The difference between `(41, 2637)` and `(39, 2791)` is consistent with an unrecovered historical JAAD annotation/API snapshot. The mechanism is reproduced; the exact historical dataset snapshot is not.

## What Samples It Affects

The clean audit and the student thesis agree that the raw JAAD annotations contain a real hazard population:

- raw behavior tracks with `attributes['crossing']=1`: `495`
- raw behavior tracks with `crossing_point=-1`: `269`
- raw behavior tracks with both `attributes['crossing']=1` and `crossing_point=-1`: `167`
- wrongly positive frames if such tracks were treated as positive for the whole track: `2791`

Evidence path:

- [experiments/005_jaad_protocol_integrity_audit/report.md](/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/005_jaad_protocol_integrity_audit/report.md)

The bachelor thesis gives the narrower historical student-benchmark count for actually wrong labels:

- affected tracks: `39`
- incorrectly labeled post-crossing frames: `2791`

and provides one concrete example:

- pedestrian `0_124_737b` in `video_0124`
- 432-frame track
- `attributes['crossing']=1`
- `crossing_point=-1`
- true per-frame crossing from frames `0..303`
- post-crossing standing on the far side from `304..431`
- API keeps `430` surviving frames positive, including `127` post-crossing frames

Evidence path:

- [papers/bachelor-4.pdf](/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/papers/bachelor-4.pdf)

## Temporal-Anchoring Issue

There are two distinct protocol effects:

1. label semantics
   - post-crossing or already-crossed observations can remain positive;
2. temporal anchoring
   - Noor's nominal `TTE` is measured relative to the end of the API-generated crossing sequence, not necessarily to a verified frame-level physical crossing onset when `crossing_point == -1`.

These effects must not be collapsed into a single claim.

## What The Current Corrected Protocol Does Differently

The current thesis benchmark uses event-aligned JAAD candidates built from frame-level transition logic first, with `crossing_point` used only as a constrained fallback:

- positive samples prefer the unique frame-level crossing transition as onset;
- fallback to `crossing_point` is allowed only when it is present and non-negative;
- tracks with `crossing_point < 0` and no clean frame-level onset are rejected as positives;
- observation windows containing already-crossing frames are rejected.

Evidence path:

- [build_jaad_event_aligned_candidates.py](/home/abdelrahman/Hadwa/BIU-PID-Distillation/src/benchmark/build_jaad_event_aligned_candidates.py:287)

The current clean-artifact audit found:

- positive windows using `crossing_point_fallback`: `0`
- positive windows that would require fallback with `crossing_point < 0`: `0`
- observation windows containing frame-level `cross=1`: `0`
- windows occurring in post-crossing regions: `0`

Evidence path:

- [experiments/005_jaad_protocol_integrity_audit/report.md](/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/005_jaad_protocol_integrity_audit/report.md)

## Important Clarification

This repository now provides:

- strong documentary evidence for the flaw;
- recovered Noor loader / windowing lineage;
- recovered upstream JAAD mechanism;
- strong evidence that the current clean benchmark is not contaminated by this issue.

But it still does **not** provide the exact historical JAAD API / annotation snapshot used during Noor's thesis run.

Therefore Protocol A should be described as:

- **documented reconstruction of the upstream JAAD API-style flawed code path used by Noor's pipeline lineage**

and **not** as:

- exact historical replay.
