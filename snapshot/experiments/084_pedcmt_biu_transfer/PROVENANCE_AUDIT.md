# Exp084 Provenance Audit

## PedCMT lineage

The architecture was reconstructed from the authors' public PedCMT repository at
commit `d9be3f87d4fa0d8d935c1fbc55188cab2c079ca3`. The reconstruction preserves the
released 128-D BBox/velocity transformer streams, global cross-modal token,
seven-layer three-unit bottleneck, endpoint auxiliary head, temporal reducer and
intent head. The exact baseline parameter-count test reproduces the public
two-channel-velocity configuration at 1,834,544 parameters.

Exp084 has one ego-speed channel because that is the authenticated input available
in the frozen thesis benchmark cache. Its baseline therefore has 1,834,416
trainable parameters. Exp084 does not claim numerical parity with the paper's
random-split experiment or its default synthetic-data runner.

## PIE population and inputs

The runner reuses aligned Exp050 caches without recomputing inputs:

- train/validation: `experiments/050_mslstm_practical_behavior_replacement_pie_16f_tte30_60/cache/trainval_base/`
- final set03: `experiments/050_mslstm_practical_behavior_replacement_pie_16f_tte30_60/cache/set03_final/`

The train/validation population contains 5,985 unique windows: 2,366 train and
3,619 validation. Every row has 16 consecutive observation frames and TTE in
30--60. The final set03 population contains 3,760 windows and is inaccessible to
the preparation/training commands.

`metrics/source_audit.json` records SHA-256 hashes for the exact train/validation
rows and numerical arrays, plus a hash of their ordered sequence IDs. The final
evaluation writes predictions in the unchanged cache order.

The auxiliary endpoint target is the normalized pedestrian BBox at the row's
stored `event_frame`, read from the exact PIE XML and pedestrian ID. Missing exact
matches fail closed; no nearby frame is substituted. This future BBox is used only
as a training target and never enters an input stream.

## Pose and BIU lineage

The raw pose and BIU arrays descend from the SHA-pinned historical Exp012/main-BIU
pose artifact:

`/home/abdelrahman/Hadwa/Pedestrian-keypoints-crossing-intent/classification/data/data_keypoints_all.jsonl`

SHA-256:
`1f0907ebb33bb04c3a3434d9c89e7bd597e2b9e3d47fe30f488ffa9410054d87`

The cache is byte-identical to the historical main-BIU cache and repository/paper
lineage documents it as ViTPose-S-derived. The original checkpoint/configuration
artifact and extraction execution log are unavailable, so its historical
checkpoint identity is not cryptographically verifiable. This does not confound
raw-pose versus BIU comparisons because all variants use measurements descended
from the exact same pinned cache.

Pose is COCO-17 XY in the cached normalized coordinate convention. Variant B uses
the 16 x 34 sequence without confidence. Variant C uses the authenticated 72-D
BIU-v1 representation; D uses frozen 63-D BIU-v2. Variant E concatenates all v1
and v2 values into a lossless 135-D source-namespaced union. Training-set means
and standard deviations alone are used for every standardization transform.

## Leakage and evaluation guards

- Preparation calls `load_population("trainval", allow_test=False)` only.
- Training repeats the same guarded train/validation load.
- Oversampling is deterministic and applies only to training indices.
- Checkpoint selection minimizes validation classification loss.
- `evaluate` checks for all 15 frozen checkpoints before loading set03.
- Accuracy, precision, recall and F1 use fixed 0.5; ROC-AUC and PR-AUC use raw
  probabilities.
- Automated invariant tests are in `tests/test_exp084_pedcmt_biu_transfer.py`.
