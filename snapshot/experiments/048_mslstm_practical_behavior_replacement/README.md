# Exp048: MSLSTM Practical Behavior Replacement

## Status

`DESIGN FROZEN; BIU-v1 AMENDMENT APPROVED; TRAINING PAUSED; SET03 SEALED`

Exp048 tests whether automatically inferred raw Exp012 pose or frozen BIU-v2
can restore behavior information after privileged PIE Action, Look, and static
semantic annotations are removed, while retaining BBox, ego dynamics, and
YOLOP in every practical variant.

The experiment uses the recovered MSLSTM-native 30-frame/next-raw-frame task.
It is not the standardized 16-frame/TTE 30-60 benchmark.

## Frozen Variants

- A: BBox + GT Action + GT Look + Ego Speed + Ego Acc + GT Static + YOLOP.
- B: A without GT Static.
- C: BBox + Ego Speed + Ego Acc + YOLOP.
- D: C + raw Exp012 pose through the recurrent pose encoder.
- E: C + frozen 63-D BIU-v2 through `63 -> 128 -> 512` projection.
- F: C + exact historical 72-D BIU-v1 through `72 -> 128 -> 512` projection.

Variant F was approved as a transparent pre-test protocol amendment on
2026-08-19. Its exact schema, timing, and authorization rules are recorded in
`PROTOCOL_AMENDMENT_BIU_V1.md`. The original A-E core and runner remain
byte-identical.

## Frozen Pose Source

- Path: `/home/abdelrahman/Hadwa/Pedestrian-keypoints-crossing-intent/classification/data/data_keypoints_all.jsonl`
- SHA-256: `1f0907ebb33bb04c3a3434d9c89e7bd597e2b9e3d47fe30f488ffa9410054d87`

D and E use the same measurements. The artifact is byte-identical to the
historical main-BIU cache and documented as ViTPose-S-derived, but its original
checkpoint/configuration and execution log are unavailable; historical
checkpoint identity is not cryptographically verifiable.

Set03 remains sealed until all 15 original A-E and three amendment F training
records authenticate.

The preparation gate passed for 333,454 native training windows and 77,288
validation windows. The pinned Exp012 pose is valid in every window. Frozen
BIU-v2 used its BBox direction fallback in 19,423 training windows (5.8248%)
and 2,161 validation windows (2.7960%). Historical YOLOP is complete for all
but one training window and four validation windows; the recovered zero-fill
policy applies to five missing frames in total.

## Execution

The frozen runners use normal terminal execution, never tmux. Build and
authenticate the BIU-v1 train/validation cache first:

```bash
python experiments/048_mslstm_practical_behavior_replacement/scripts/run_biu_v1_amendment.py --resume audit
python experiments/048_mslstm_practical_behavior_replacement/scripts/run_experiment.py --resume train-all
python experiments/048_mslstm_practical_behavior_replacement/scripts/run_biu_v1_amendment.py --resume train-all
python experiments/048_mslstm_practical_behavior_replacement/scripts/run_biu_v1_amendment.py evaluate
python experiments/048_mslstm_practical_behavior_replacement/scripts/run_biu_v1_amendment.py finalize
```

After approval of Variant F, the original runner's `evaluate` and `finalize`
commands are prohibited. Only the amendment runner may perform the single
final A-F set03 evaluation.

Progress is available without touching set03:

```bash
python experiments/048_mslstm_practical_behavior_replacement/scripts/run_biu_v1_amendment.py status
tail -f experiments/048_mslstm_practical_behavior_replacement/logs/biu_v1_amendment_audit.log
```
