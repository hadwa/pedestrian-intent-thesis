# Exp083 Input Coverage Audit

**Gate status:** `blocked_new_extraction_required`

This is an identity/availability audit only. It did not decode test RGB, assemble test features, train a model, or compute test metrics.

## Pose Sources

- `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/049_exp048_to_jaad_zero_shot/cache/jaad_keypoints_vitpose_s_exp049_extended.jsonl`: SHA-256 `9eae258cd7fed3b1278aa86f6ab5c508fb759a24c7912d5c57677506ae86e4db`, 65,637 COCO-17 rows, 0 invalid rows.
- `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/079_jaad_api_protocol_two_stage_xgb/cache/jaad_api_protocol_missing_pose.jsonl`: SHA-256 `d0c0fd771eff1dadd7c499a680cef704409e6a07a72ca31f3f3b39d6a86fe038`, 56,463 COCO-17 rows, 0 invalid rows.

Pose identity is exact `(video_id, ped_id, frame_id)`; no suffix aliasing or nearby-frame substitution is permitted.

## Coverage

| cohort | split | input | unit | required | present | missing | coverage_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| jaad_beh | train | vitpose_s_coco17 | pedestrian_frame | 8924 | 8924 | 0 | 1.000000 |
| jaad_beh | train | rgb_frame | frame | 6294 | 6294 | 0 | 1.000000 |
| jaad_beh | train | motion_proxy_15d | sequence | 2134 | 649 | 1485 | 0.304124 |
| jaad_beh | train | road_context_60d | sequence | 2134 | 649 | 1485 | 0.304124 |
| jaad_beh | val | vitpose_s_coco17 | pedestrian_frame | 1012 | 1012 | 0 | 1.000000 |
| jaad_beh | val | rgb_frame | frame | 753 | 753 | 0 | 1.000000 |
| jaad_beh | val | motion_proxy_15d | sequence | 242 | 121 | 121 | 0.500000 |
| jaad_beh | val | road_context_60d | sequence | 242 | 121 | 121 | 0.500000 |
| jaad_beh | test | vitpose_s_coco17 | pedestrian_frame | 7866 | 7866 | 0 | 1.000000 |
| jaad_beh | test | rgb_frame | frame | 5295 | 5295 | 0 | 1.000000 |
| jaad_beh | test | motion_proxy_15d | sequence | 1881 | 660 | 1221 | 0.350877 |
| jaad_beh | test | road_context_60d | sequence | 1881 | 660 | 1221 | 0.350877 |
| jaad_all | train | vitpose_s_coco17 | pedestrian_frame | 36018 | 36018 | 0 | 1.000000 |
| jaad_all | train | rgb_frame | frame | 15185 | 15185 | 0 | 1.000000 |
| jaad_all | train | motion_proxy_15d | sequence | 8613 | 649 | 7964 | 0.075351 |
| jaad_all | train | road_context_60d | sequence | 8613 | 649 | 7964 | 0.075351 |
| jaad_all | val | vitpose_s_coco17 | pedestrian_frame | 5290 | 5290 | 0 | 1.000000 |
| jaad_all | val | rgb_frame | frame | 2368 | 2368 | 0 | 1.000000 |
| jaad_all | val | motion_proxy_15d | sequence | 1265 | 121 | 1144 | 0.095652 |
| jaad_all | val | road_context_60d | sequence | 1265 | 121 | 1144 | 0.095652 |
| jaad_all | test | vitpose_s_coco17 | pedestrian_frame | 28152 | 28152 | 0 | 1.000000 |
| jaad_all | test | rgb_frame | frame | 11109 | 11109 | 0 | 1.000000 |
| jaad_all | test | motion_proxy_15d | sequence | 6732 | 660 | 6072 | 0.098039 |
| jaad_all | test | road_context_60d | sequence | 6732 | 660 | 6072 | 0.098039 |

## Stop Decision

Existing compact Exp081/082 caches leave 15,180 unique official JAAD_all sequences requiring fresh motion/TwinLiteNet+ context extraction. Their observation windows contain 25,819 unique RGB frames.
The runner therefore stops before extraction, feature assembly, or training, as frozen in the Exp083 design.

The road identity inventory contains 186 per-video NPZ artifacts. Only their `sequence_id` arrays were read during this coverage gate; cached test feature tensors were not accessed.
