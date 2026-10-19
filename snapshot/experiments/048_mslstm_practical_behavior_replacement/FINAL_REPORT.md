# Exp048 Final Report

Final outcome category: **BIU-v1 efficiency success**.

This controlled experiment uses the recovered 30-frame, next-raw-frame MSLSTM task. The paper F1=0.8992 remains a historical reference only. Exp048 architecture and selection decisions were frozen before this one final set03 evaluation; set03 had been inspected previously in separate post-hoc work, which is disclosed as an exposure limitation.

## Fresh Mean Results

| Variant | F1 mean | PR-AUC mean | ROC-AUC mean | Parameters |
|---|---:|---:|---:|---:|
| A | 0.711750 | 0.776083 | 0.920513 | 15,436,304 |
| B | 0.722533 | 0.805712 | 0.949701 | 13,218,062 |
| C | 0.683132 | 0.760159 | 0.901240 | 8,871,690 |
| D | 0.688236 | 0.775416 | 0.904604 | 11,112,460 |
| E | 0.710144 | 0.778625 | 0.924176 | 8,945,931 |
| F | 0.709888 | 0.784938 | 0.927899 | 8,947,083 |

## Frozen Comparisons

E→F favorable for BIU-v1 on both F1 and PR-AUC in all seeds: `false`.
E→F favorable for BIU-v2 on both metrics in all seeds: `false`.
BIU-v2 within the frozen 0.005 F1/PR-AUC margin of raw pose: `true`.
BIU-v1 within the frozen 0.005 F1/PR-AUC margin of raw pose: `true`.
BIU-v2 recovery fraction: F1 `0.747856`, PR-AUC `0.306304`.
BIU-v1 recovery fraction: F1 `0.724882`, PR-AUC `0.474939`.

## Thesis-Safe Interpretation

D versus E/F compares raw temporal pose against two frozen task-specific behavioral representations from the same Exp012 measurements. E versus F compares complete frozen representations, not a single-cue ablation: BIU-v1 includes its historical low-confidence count, while BIU-v2 includes its frozen direction/scale metadata. Neither BIU adds body-pose information absent from the shared pose source. A versus E/F is an overall privileged-to-automatic pipeline comparison and must not be attributed solely to BIU.

## Per-Seed Fixed-0.5 Results

| Variant | Seed | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 42 | 0.891740 | 0.699154 | 0.724327 | 0.711518 | 0.920017 | 0.772908 |
| A | 43 | 0.883163 | 0.668658 | 0.725748 | 0.696034 | 0.900891 | 0.749264 |
| A | 44 | 0.888925 | 0.663790 | 0.805224 | 0.727699 | 0.940630 | 0.806077 |
| B | 42 | 0.873335 | 0.613219 | 0.847093 | 0.711428 | 0.947702 | 0.799272 |
| B | 43 | 0.893717 | 0.676902 | 0.810007 | 0.737497 | 0.949369 | 0.811345 |
| B | 44 | 0.874549 | 0.612513 | 0.869349 | 0.718674 | 0.952032 | 0.806519 |
| C | 42 | 0.893514 | 0.754419 | 0.626078 | 0.684283 | 0.893237 | 0.760509 |
| C | 43 | 0.896015 | 0.765593 | 0.628180 | 0.690112 | 0.914691 | 0.781294 |
| C | 44 | 0.882609 | 0.689191 | 0.661383 | 0.675001 | 0.895793 | 0.738675 |
| D | 42 | 0.896856 | 0.758463 | 0.646192 | 0.697841 | 0.894777 | 0.770277 |
| D | 43 | 0.867201 | 0.611173 | 0.768318 | 0.680795 | 0.909753 | 0.770237 |
| D | 44 | 0.900663 | 0.821629 | 0.588912 | 0.686073 | 0.909281 | 0.785733 |
| E | 42 | 0.902150 | 0.772065 | 0.665646 | 0.714917 | 0.925136 | 0.779376 |
| E | 43 | 0.899357 | 0.753617 | 0.674492 | 0.711863 | 0.918171 | 0.775563 |
| E | 44 | 0.899206 | 0.768054 | 0.649214 | 0.703652 | 0.929222 | 0.780936 |
| F | 42 | 0.899756 | 0.757811 | 0.670389 | 0.711425 | 0.928839 | 0.781278 |
| F | 43 | 0.903187 | 0.785225 | 0.653497 | 0.713331 | 0.930754 | 0.786661 |
| F | 44 | 0.897273 | 0.749065 | 0.665666 | 0.704907 | 0.924103 | 0.786874 |

## Provenance And Efficiency

Variants D, E, and F use the same SHA-pinned historical Exp012 pose artifact: `/home/abdelrahman/Hadwa/Pedestrian-keypoints-crossing-intent/classification/data/data_keypoints_all.jsonl` (`1f0907ebb33bb04c3a3434d9c89e7bd597e2b9e3d47fe30f488ffa9410054d87`). The artifact is byte-identical to the historical main-BIU cache and documented as ViTPose-S-derived, but the original checkpoint/configuration and execution log are unavailable, so historical checkpoint identity is not cryptographically verifiable.

F uses the exact historical 72-D BIU-v1 schema and a fixed `72 -> 128 -> 512` projector. E uses the frozen 63-D BIU-v2 schema and `63 -> 128 -> 512`. Pose-estimator runtime is excluded from predictor/pathway latency because both use the historical cache.
