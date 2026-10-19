# Exp048 Pose Provenance Audit

## Frozen Source For D And E

- Path: `/home/abdelrahman/Hadwa/Pedestrian-keypoints-crossing-intent/classification/data/data_keypoints_all.jsonl`
- SHA-256: `1f0907ebb33bb04c3a3434d9c89e7bd597e2b9e3d47fe30f488ffa9410054d87`
- Size: 794,892,224 bytes.
- Historical frame records: 740,901 unique `(set, video, pid, frame_idx)` identities.
- Stored coordinates: full-image pixel `(x,y)` coordinates.
- D conversion: x divided by image width and y by image height, then flattened
  in frame order to MSLSTM's 34-D temporal pose stream.
- E conversion: the same pixel coordinates enter the unchanged frozen 63-D
  BIU-v2 extractor over the 30 observed frames. No future frame enters BIU.
- Confidence fields exist in the JSONL but are not model inputs in D or E.

The joint order is COCO-17:

`nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle`.

## ViTPose-S Lineage And Limitation

The artifact is byte-identical to the historical main-BIU benchmark cache.
Repository extraction code (`classification/scripts/make_kp.py`), the BIU
paper, and historical repository documentation identify this cache as derived
from the ViTPose-S COCO top-down pathway.

The original checkpoint/configuration artifact and execution log are no longer
available. Historical checkpoint identity is therefore not cryptographically
verifiable. Exp048 does not re-extract pose. This limitation does not confound
the primary D-versus-E comparison because both variants consume the exact same
SHA-pinned measurements.

## Coverage And BIU Fallback Gate

The preparation runner checks every native training and validation window,
does not drop missing-pose samples, and writes:

- `metrics/pose_coverage_trainval.csv`
- `metrics/bbox_direction_fallback_trainval.csv`

Authenticated preparation results:

| Cohort | Windows | Valid Exp012 pose | Mean/median valid-joint fraction | BBox fallback |
|---|---:|---:|---:|---:|
| Train | 333,454 | 333,454 (100%) | 1.0 / 1.0 | 19,423 (5.8248%) |
| Validation | 77,288 | 77,288 (100%) | 1.0 / 1.0 | 2,161 (2.7960%) |

Set03 remains unopened by Exp048 until all 15 training records authenticate.
Its corresponding coverage and fallback rows will be published during the one
final evaluation. Frozen BIU-v2 may use its established BBox-center direction
fallback, so E is not described as a mathematically pure pose-only transform
when that fallback is invoked.
