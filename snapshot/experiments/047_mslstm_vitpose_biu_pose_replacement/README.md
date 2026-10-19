# Exp047: MSLSTM ViTPose-BIU Pose Replacement

## Status

`PRETRAINING AUDIT COMPLETE; NINE FROZEN RUNS IN PROGRESS WITHOUT TMUX`

This is a fresh, controlled three-variant experiment under the recovered local
MSLSTM 30-frame/next-raw-frame task. It does not modify or reinterpret Exp032.

## Frozen Comparison

- A: original YOLOv8x-pose raw XY sequence and original pose BiLSTM/attention.
- B: historical Exp012/main-BIU pose cache in the same raw XY BiLSTM/attention.
- C: the exact same Exp012 pose measurements through frozen 63-D BIU-v2 and
  frozen `63 -> 128 -> 512` projection in the original pose-context slot.

All non-pose streams, fusion, classifier, loss, optimizer, data order, five
epochs, seeds, fixed-0.5 reporting, and validation-F1-at-0.5 checkpoint rule are
unchanged. Set03 remains the final Exp047 test cohort and may be evaluated only
after all nine training runs finish under the frozen design.

## Frozen Sources

- Exp012 pose JSONL: `/home/abdelrahman/Hadwa/Pedestrian-keypoints-crossing-intent/classification/data/data_keypoints_all.jsonl`
- Exp012 pose SHA-256: `1f0907ebb33bb04c3a3434d9c89e7bd597e2b9e3d47fe30f488ffa9410054d87`
- Original MSLSTM pose: `/home/abdelrahman/Hadwa/MSLSTM-PID/Extracted Poses/`
- Protocol rows: `experiments/032_mslstm_biu_v2_transfer/full_run_corrected_2026-08-16/metrics/protocol_rows.csv`
- Frozen design: [DESIGN.md](DESIGN.md)
- Pose audit: [POSE_PROVENANCE_AUDIT.md](POSE_PROVENANCE_AUDIT.md)

The historical cache is byte-identical to the main-BIU artifact and strongly
documented as ViTPose-S-derived. Its original checkpoint/configuration artifact
and execution log are unavailable, so historical checkpoint identity is not
cryptographically verifiable. This does not confound B to C because both use
the same SHA-pinned measurements.

## Execution Policy

Exp047 does not run in tmux. The phase runner uses persistent logs and
authenticated resumable state. The pose provenance, alignment, coverage,
BBox-fallback, architecture, checkpoint-selection, cache, transaction, and
set03-isolation tests pass. The real train/validation-only stream cache was
built from 187,222 rows and `.audit.complete.json` was issued only after all
cohort, source, array, and hash checks authenticated.

The pretraining runner reconstructs only sets01/02/04/05/06 from the original
set-specific PIE XML files. It does not deserialize the monolithic PIE pickle,
and its Exp012 JSONL loader skips set03 lines before keypoint decoding. The
reconstructed training scalers match the historical `scalers.pkl` values
exactly. Set03 XML, pose arrays, labels, and predictions are loaded only after
all nine training records authenticate in the final `evaluate` command.

## Commands

```bash
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python \
  experiments/047_mslstm_vitpose_biu_pose_replacement/scripts/run_experiment.py audit

PYTHONUNBUFFERED=1 /home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python \
  experiments/047_mslstm_vitpose_biu_pose_replacement/scripts/run_experiment.py train-all
```

The managed terminal session, `logs/train_all.log`, per-run logs, resumable
epoch state, best checkpoints, and authenticated completion records provide
progress visibility without tmux.

The execution environment exposes CPU only. A 50-batch real-cache benchmark
measured approximately 0.360 seconds/batch for A/B and 0.294 seconds/batch for
C, implying a long managed run; this changes no training setting.
