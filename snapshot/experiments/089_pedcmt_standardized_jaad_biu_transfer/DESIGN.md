# Exp089 Design: Standardized JAAD PedCMT BIU Transfer

## Question

Does an explicit BIU representation add useful predictive information to the
same PedCMT architecture beyond (A) BBox plus automatically available JAAD
vehicle action and (B) those streams plus raw pose?

## Frozen Protocol

- Dataset: JAAD behavior pedestrians, official default video split.
- Observation: 16 consecutive frames.
- Time to event: 30--60 frames.
- Cohorts: 1,552 train, 274 validation, 1,501 final test windows.
- Threshold: fixed 0.5.
- Ranking metrics: probability ROC-AUC and PR-AUC.
- Seeds: 42, 43, 44.
- Checkpoint: minimum validation classification loss, earlier epoch on ties.
- Training: Exp084 policy, including training-only class oversampling, Adam
  `5e-4`, batch size 64, 200 epochs maximum and patience 25.
- Test remains inaccessible until all 15 checkpoints exist.

## Frozen Variants

| Variant | Inputs |
|---|---|
| A | normalized 16-frame XYXY BBox + normalized JAAD vehicle action |
| B | A + normalized raw 17-joint XY pose sequence |
| C | A + frozen 72-D BIU-v1 |
| D | A + frozen 63-D BIU-v2 |
| E | A + 135-D source-namespaced BIU-v1/v2 union |

All variants use the exact Exp084 PedCMT architecture and fusion location. Raw
pose and BIU features are standardized using training-only statistics. Pose
confidence is used only by the frozen BIU-v1 validity counters and is never a
raw model input.

## JAAD Context

JAAD does not provide metric ego speed. Following the recovered PedCMT JAAD
lineage, the one-channel context is categorical ego-vehicle action (`0` stopped,
`1` slow, `2` fast, `3` decelerating, `4` accelerating), divided by four.

## Pose Provenance And Repair

Primary cache:
`data/jaad_keypoints_vitpose_s_extended_short_horizon.jsonl`, SHA-256
`4cf675a62f158dcedf79b8037ab39b4fdc94cefe1dd8783eb6e4523d792b2b64`.

Supplement:
`experiments/030_final_confidence_benchmark/jaad30_repair/jaad30_missing_supplemental_keypoints.jsonl`,
SHA-256 `660eb16b1d758f971ff34b8e41f2453ba1c43c5d1410753e77c23481a660c57e`.

The primary cache covers 14,230 of 14,310 unique required frames. The supplement
contains exactly the missing 80. The union provides complete pose coverage and
all 243,270 required joint observations are valid. Variants B--E are recomputed
from this same union. This removes the last-pose forward filling present in the
older feature tables for 30 windows; no sample is dropped.

## Frozen Comparisons

- A -> C/D/E: BIU beyond BBox plus vehicle action.
- B -> C/D/E: engineered BIU versus raw pose from the same measurements.
- Primary evidence requires reporting per-seed and mean fixed-0.5 F1, ROC-AUC,
  and PR-AUC. No post-test model or threshold changes are permitted.
