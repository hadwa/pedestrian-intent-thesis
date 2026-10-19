# Exp047 Pose Provenance Audit

## Gate Status

`POSE DATA GATE COMPLETE; OVERALL TRAINING GATE INCOMPLETE`

The full effective-cohort coverage, BBox direction-fallback, and BIU cache
authentication gates pass. Training remains unauthorized until the separate
train/validation-only, checkpoint, and final-set03 isolation gates pass.

## Original MSLSTM Pose (A)

- Cache: `/home/abdelrahman/Hadwa/MSLSTM-PID/Extracted Poses/{set}/{set}_{video}_poses.pkl`
- Extraction lineage: `MSLSTM-PID/Offline Extraction/yolo_pose_extraction.ipynb`
- Model family recorded by the recovered lineage: YOLOv8x-pose-p6.
- Skeleton: COCO 17 joints.
- Stored representation: flattened 34-D `(x,y)` coordinates normalized by full
  1920x1080 frame width/height.
- Confidence: not stored in the raw MSLSTM pose vector.

## Historical Exp012/Main-BIU Pose (B/C)

- Primary path: `/home/abdelrahman/Hadwa/Pedestrian-keypoints-crossing-intent/classification/data/data_keypoints_all.jsonl`
- Main-BIU mirror: `/home/abdelrahman/Hadwa/pedestrian-intent-biu/data/data_keypoints_all.jsonl`
- SHA-256 of both files: `1f0907ebb33bb04c3a3434d9c89e7bd597e2b9e3d47fe30f488ffa9410054d87`
- Size: 794,892,224 bytes.
- Rows: 740,901 unique frame records; no duplicate identity keys observed.
- Identity fields: `set`, `video`, `pid`, `frame_idx`.
- Skeleton: COCO 17-joint order:
  `nose,left_eye,right_eye,left_ear,right_ear,left_shoulder,right_shoulder,left_elbow,right_elbow,left_wrist,right_wrist,left_hip,right_hip,left_knee,right_knee,left_ankle,right_ankle`.
- Coordinate convention: full-image pixel `(x,y)` coordinates. B divides x by
  frame width and y by frame height and flattens to the original MSLSTM 34-D
  order. C supplies pixel coordinates to frozen BIU-v2.
- Serialized quality fields: 17 `kp_conf` values and `skeleton_conf`; Exp047
  does not use pose-estimator confidence as model input.

## ViTPose-S Evidence And Limitation

`classification/scripts/make_kp.py` identifies the intended small ViTPose COCO
top-down configuration/checkpoint and serializes predictions from PIE pedestrian
BBoxes. The historical BIU paper and repository describe this exact 740,901-row
cache as ViTPose-S skeleton tracks. The primary and main-BIU copies are
byte-identical.

The original historical checkpoint/configuration artifact and execution log are
no longer available. Historical checkpoint identity therefore is not
cryptographically verifiable. Exp047 does not re-extract pose. This limitation
does not confound B to C because both variants consume the exact same SHA-pinned
coordinates.

## Formal Exact-Frame And Coverage Audit

The frozen runner processed all 458,300 effective windows and wrote the
authenticated table at `metrics/pose_coverage.csv`.

| Source | Cohort | Windows | All-zero | Valid | Mean valid-joint fraction | Median |
|---|---|---:|---:|---:|---:|---:|
| YOLO | train | 109,934 | 37,945 (34.5162%) | 71,989 (65.4838%) | 0.444888 | 0.635294 |
| Exp012 | train | 109,934 | 0 | 109,934 (100%) | 1.000000 | 1.000000 |
| YOLO | validation | 77,288 | 52,025 (67.3132%) | 25,263 (32.6868%) | 0.195271 | 0.000000 |
| Exp012 | validation | 77,288 | 0 | 77,288 (100%) | 1.000000 | 1.000000 |
| YOLO | set03 | 271,078 | 146,030 (53.8701%) | 125,048 (46.1299%) | 0.289301 | 0.000000 |
| Exp012 | set03 | 271,078 | 0 | 271,078 (100%) | 1.000000 | 1.000000 |

Availability uses finite XY coordinates that are not both zero and does not
invent a keypoint-confidence threshold. No sample was dropped.

## BBox Direction Fallback

| Cohort | Windows | Fallback count | Fallback rate |
|---|---:|---:|---:|
| train | 109,934 | 3,911 | 3.5576% |
| validation | 77,288 | 2,161 | 2.7960% |
| set03 | 271,078 | 13,459 | 4.9650% |

C is not a mathematically pure pose-only transform for those windows; it uses
the already-existing BBox measurement through frozen BIU-v2's direction
fallback.

## Authenticated BIU Cache

- Path: `cache/biu_v2_exp012_30frame.npy`
- Shape/dtype: `(458300, 63)`, float32
- Finite values: 100%
- SHA-256: `05e09ca55fc4cd962950b62a24010cb7ff8552a04baf8cdf1e50092193078f5c`
- Completion record: `.pose_audit.complete.json`
