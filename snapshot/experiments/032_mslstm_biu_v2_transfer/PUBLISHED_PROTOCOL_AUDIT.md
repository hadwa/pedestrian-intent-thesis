# Exp032 Published-Protocol Audit

This document reconstructs the task-defining MSLSTM protocol from repository evidence
before any Exp032 implementation. It is intentionally conservative.

## Audit conclusion

The repository now provides enough evidence to recover the **task-defining published/local
MSLSTM setting**, but it still does **not** contain a complete historical pose-capable
experiment bundle for the original published result. What is recoverable is the local
**published/reproduced MSLSTM pipeline lineage**:

- dataset/task family: PIE pedestrian crossing prediction
- observation construction: `30` consecutive raw frames
- target construction: `pred_len = 1` raw frame after the last observed frame
- label source: `pdb["behavior"]["cross"]` at the target frame
- split lineage: train `set01,set02,set04`; val `set05,set06`; test `set03`
- pose lineage: extracted pose pickles produced by `Offline Extraction/yolo_pose_extraction.ipynb`

This is strong enough to recover the Exp032 hard-stop provenance, but not strong enough
to claim an exact paper-faithful replay or a locally pinned pose-capable parity artifact.

## Sources inspected

- `/home/abdelrahman/Hadwa/MSLSTM-PID/README.md`
- `/home/abdelrahman/Hadwa/MSLSTM-PID/Final Version.ipynb`
- `/home/abdelrahman/Hadwa/MSLSTM-PID/checkpoints/train_index.pkl`
- `/home/abdelrahman/Hadwa/MSLSTM-PID/Extracted Poses/set01/set01_video_0001_poses.pkl`
- [ICVES_samalouty.pdf](/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/papers/ICVES_samalouty.pdf)
- [../018_mslstm_retrain_benchmark/README.md](../018_mslstm_retrain_benchmark/README.md)
- `/home/abdelrahman/Hadwa/BIU-PID-Distillation/reports/MSLSTM_PRED_LEN_VERIFICATION_REPORT.md`
- `/home/abdelrahman/Hadwa/BIU-PID-Distillation/reports/MSLSTM_PID_REPO_AUDIT_REPORT.md`
- `/home/abdelrahman/Hadwa/BIU-PID-Distillation/reports/MSLSTM_PID_STREAM_AVAILABILITY_REPORT.md`
- `/home/abdelrahman/Hadwa/BIU-PID-Distillation/reports/MSLSTM_PID_TEACHER_INPUT_DRY_RUN_REPORT.md`
- `/home/abdelrahman/Hadwa/BIU-PID-Distillation/reports/MSLSTM_INPUT_SEMANTICS_AUDIT_REPORT.md`
- `/home/abdelrahman/miniconda3/envs/mslstm_pid_env/lib/python3.9/site-packages/ultralytics/cfg/datasets/coco-pose.yaml`
- `/home/abdelrahman/miniconda3/envs/mslstm_pid_env/lib/python3.9/site-packages/ultralytics/models/yolo/pose/predict.py`
- `/home/abdelrahman/miniconda3/envs/mslstm_pid_env/lib/python3.9/site-packages/ultralytics/engine/results.py`

## Task-defining provenance recovered

### Dataset and labels

- Dataset: PIE.
- Local MSLSTM notebook setting: `SEQ_LEN = 30`, `PRED_LEN = 1`.
- Sequence construction uses consecutive raw frames with no stride-based temporal jump.
- The target label is read from pedestrian `behavior.cross` at:

  `target_frame = start + seq_len + pred_len - 1`

- `pred_len = 1` therefore means one raw frame after the observed window, not a 30-frame
  or 1-second anticipatory horizon.

### Splits

Recovered local split lineage:

- train: `set01`, `set02`, `set04`
- val: `set05`, `set06`
- test: `set03`

The active notebook does not train on every reconstructed training window. It
loads `checkpoints/train_index.pkl`, a deterministic seed-42 majority-class
undersample containing `109,934` unique windows with exactly `54,967` examples
per class. The saved index records `seq_len=30` and `pred_len=1`. Validation
continues to enumerate all windows from set05 and set06.

This differs materially from Exp018's standardized PIE benchmark.

### Pose source

Recovered local pose lineage:

- extraction notebook: `/home/abdelrahman/Hadwa/MSLSTM-PID/Offline Extraction/yolo_pose_extraction.ipynb`
- model path inside notebook: `yolov8x-pose-p6.pt`
- saved pose artifact format:

  `Extracted Poses/{set}/{set}_{video}_poses.pkl`

- stored per-frame/per-pedestrian pose vectors are `34`-D flattened `x,y` values.
- direct sample inspection shows these values are already normalized to `[0,1]`.
- the extraction notebook normalizes by `x / vid_width`, `y / vid_height` before saving.

The full Exp032 preparation audit found that the recovered pose cache is sparse
for the fully enumerated cohort: `12,649,031/20,454,600` observation-frame pose
vectors were zero or absent (`61.8395%`), and `368,329/681,820` windows were
entirely pose-empty. A follow-up key-presence audit found only six absent keys;
almost all of this count is therefore stored all-zero detector fallback, not a
loader key mismatch. The source notebook itself zero-fills missing detections.

For the effective model cohorts, zero/absent pose-frame rates are `41.81%` in
balanced train, `74.62%` in validation, and `61.68%` in diagnostic set03. This is
a source-coverage limitation and must not be described as measured keypoints.

### Available streams in the local lineage

Recovered stream inventory from the local notebook/repo lineage:

- `bbox`
- `pose`
- `ego_speed`
- `ego_acc`
- `ego_gyro`
- `ped_action`
- `ped_look`
- `ped_occlusion`
- `traffic_light`
- `static_context`
- local YOLOP features also exist in the repo

However, the available saved checkpoints in `Teacher_PTH/` do **not** include pose or
YOLOP. They cover only:

- `bbox`
- `ego_acc`
- `ego_speed`
- `ped_action`
- `ped_look`
- `static_context`

Therefore Exp032 requires a fresh training path; it cannot rely on a saved pose-capable
checkpoint.

## Recovered evaluation behavior

The local notebook currently:

- saves the best checkpoint by validation `F1` at threshold `0.5`
- also computes a validation-tuned threshold by maximizing validation F1 after training

So the recovered published-setting lineage has two distinct operating points:

- checkpoint selection: validation `F1@0.5`
- secondary reporting path: validation-selected threshold

Exp032 will preserve that convention as a published-setting analysis, while also
reporting fixed `0.5` for thesis consistency.

## Standard benchmark vs recovered MSLSTM setting

| Property | Exp018 standardized benchmark | Recovered local MSLSTM lineage for Exp032 |
|---|---|---|
| Purpose | Honest anticipatory benchmark | Transfer into the original local MSLSTM task lineage |
| Dataset | PIE | PIE |
| Observed frames | `16` | `30` |
| Target horizon | crossing `30-60` frames before event | `pred_len = 1` raw frame after observation |
| Label construction | event-aligned benchmark label | `behavior.cross` at target frame |
| Split lineage | train `set01,set02,set06`; val `set04,set05`; test `set03` | train `set01,set02,set04`; val `set05,set06`; test `set03` |
| Pose source | project keypoints JSONL | local `Extracted Poses/*.pkl` from YOLOv8x-pose-p6 extraction |
| Pose format | pixel-keypoint cache normalized on read | stored normalized `34`-D flattened pose vectors |
| Available frames | frozen benchmark rows only | locally enumerated consecutive windows |
| Local stream family | deployment-clean subset | pose-capable notebook lineage using `bbox`, `pose`, `ped_action`, `ped_look`, `ego_speed`, `ego_acc`, `static_context` |
| Checkpoint selection | validation `F1@0.5` | validation `F1@0.5` |
| Evaluation threshold / decision rule | fixed `0.5` primary; val-F1 secondary | fixed `0.5` checkpoint tracking in notebook; validation-selected threshold also computed post-training |
| Reported metrics | fixed `0.5`, val-F1, ROC-AUC, PR-AUC | paper reports Acc/AUC/F1/Precision/Recall; Exp032 will additionally preserve fixed `0.5` and PR-AUC |
| Reference result | Exp018 preserved local benchmark outputs | paper-level descriptive reference only: Acc `0.916`, AUC `0.970`, F1 `0.899`, Precision `0.929`, Recall `0.871` |

## Numeric reference status

### Joint-order audit: extraction lineage to stored pose vectors

Recovered source chain:

- extraction notebook:
  `/home/abdelrahman/Hadwa/MSLSTM-PID/Offline Extraction/yolo_pose_extraction.ipynb`
- detector call inside notebook:
  `results = pose_model(frame_rgb, ...)`
- stored keypoints pulled from detector output:
  `results[0].keypoints.data.cpu().numpy()`
- saved pose vector construction:
  iterate `kp_idx` from `0..16`, append
  `x / vid_width`, `y / vid_height`, then save a flattened `34`-D vector
- saved artifact:
  `Extracted Poses/{set}/{set}_{video}_poses.pkl`

Verified storage behavior from notebook code and direct `.pkl` inspection:

- stored format is flattened `[j0_x, j0_y, j1_x, j1_y, ..., j16_x, j16_y]`
- only `x,y` are stored; confidence is discarded
- coordinates are normalized by full-frame width/height before saving
- there is **no explicit joint reordering** before saving
- there is **no explicit left/right swapping** before saving

Verified detector-side semantic ordering:

- local environment package inspected:
  `/home/abdelrahman/miniconda3/envs/mslstm_pid_env/lib/python3.9/site-packages/ultralytics/`
- `ultralytics/cfg/models/v8/yolov8-pose-p6.yaml` defines `kpt_shape: [17, 3]`
- `ultralytics/models/yolo/pose/predict.py` reshapes pose output with:
  `pred[:, 6:].view(pred.shape[0], *self.model.kpt_shape)` and updates
  `result.keypoints` without semantic reordering
- `ultralytics/engine/results.py` documents keypoints as `17 keypoints (COCO format)`
- `ultralytics/cfg/datasets/coco-pose.yaml` provides the semantic 17-joint COCO order and
  `flip_idx`, confirming preserved left/right identities

#### Stored 34-D pose layout vs BIU-v2 expected COCO joints

| Stored joint index | Stored x/y positions | Semantic joint name | BIU-v2 expected COCO joint | Match |
|---|---|---|---|---|
| 0 | 0, 1 | nose | nose | yes |
| 1 | 2, 3 | left_eye | left_eye | yes |
| 2 | 4, 5 | right_eye | right_eye | yes |
| 3 | 6, 7 | left_ear | left_ear | yes |
| 4 | 8, 9 | right_ear | right_ear | yes |
| 5 | 10, 11 | left_shoulder | left_shoulder | yes |
| 6 | 12, 13 | right_shoulder | right_shoulder | yes |
| 7 | 14, 15 | left_elbow | left_elbow | yes |
| 8 | 16, 17 | right_elbow | right_elbow | yes |
| 9 | 18, 19 | left_wrist | left_wrist | yes |
| 10 | 20, 21 | right_wrist | right_wrist | yes |
| 11 | 22, 23 | left_hip | left_hip | yes |
| 12 | 24, 25 | right_hip | right_hip | yes |
| 13 | 26, 27 | left_knee | left_knee | yes |
| 14 | 28, 29 | right_knee | right_knee | yes |
| 15 | 30, 31 | left_ankle | left_ankle | yes |
| 16 | 32, 33 | right_ankle | right_ankle | yes |

#### Joint-order audit decision

- hard-stop question answered: **joint order is confidently established**
- left/right identities are preserved semantically
- BIU-v2 can be computed from this skeleton **without semantic joint approximation**
- remaining compatibility caveat is coordinate scale:
  the stored MSLSTM pose vectors are normalized to `[0,1]`, while the existing BIU-v2
  extractor in this repository is defined over pixel-coordinate geometry
- therefore any Exp032 implementation must derive BIU-v2 from the same stored pose
  measurements after deterministic de-normalization back to frame coordinates using the
  original video width/height metadata; this is a coordinate conversion issue, not a joint
  identity issue

### Published/local baseline parity audit

#### Exact task-defining published reference recovered

Local paper artifact found:

- [ICVES_samalouty.pdf](/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/papers/ICVES_samalouty.pdf)

Published-paper statements matching the recovered Exp032 task family:

- experimental setup states:
  `For an observation window of T = 30 frames (1 second)`
- dataset setup states:
  `The model predicts the pedestrian's crossing status 1 second into the future from a 1-second historical observation.`
- the paper says it adheres to the `official data splits`
- the model family explicitly includes pose as a core stream:
  `Pose (34-D): normalized (x, y) coordinates of 17 skeletal keypoints`

Published final multimodal result on PIE (Table II):

- `Accuracy = 0.916`
- `AUC = 0.970`
- `F1 = 0.899`
- `Precision / Recall = 0.93 / 0.87`

Published ablation trajectory (Table I):

- `+ BiLSTM`: `F1 0.7564`
- `+ Balancing & Context`: `F1 0.7605`
- `+ Adaptive Fusion`: `F1 0.8215`
- `+ YOLOP Context`: `F1 0.8290`
- `Final (w/ Augment.)`: `F1 0.8992`, `Precision 0.9291`, `Recall 0.8711`

Paper-level model characterization:

- architecture family: multi-stream `BiLSTM + attention + adaptive weighted fusion`
- modalities described in paper:
  `bbox`, `pose`, explicit behavior labels, ego dynamics, YOLOP panoptic context, static
  environmental features
- run type: **single published run / point estimate**
- threshold provenance in paper: **not explicitly stated**

#### Local lineage evidence supporting task equivalence

Local repository evidence in `MSLSTM-PID`:

- `Final Version.ipynb` source defines:
  - `SEQ_LEN, PRED_LEN = 30, 1`
  - `VAL_SETS_STR = ["set05", "set06"]`
  - active pose-capable stream set:
    `["bbox", "pose", "ped_action", "ped_look", "ego_speed", "ego_acc", "static_context"]`
- prior repo audits already established:
  - train split: `set01,set02,set04`
  - val split: `set05,set06`
  - test split: `set03`
  - target label: `behavior.cross` at `start + seq_len + pred_len - 1`

This is consistent with the published paper's next-second PIE setting, but not identical in
preserved artifact form:

- current local notebook source uses `static_context` rather than the paper's full final
  `YOLOP + static context` description
- the notebook output preserved in the repository does **not** include a completed final
  held-out result for the pose-capable run
- the two saved `Teacher_PTH/*.pth` checkpoints are **not pose-capable**

#### Parity-reference classification

##### 1. Exact parity reference

None recovered as a locally pinned, fully reproducible artifact bundle.

Reason:

- no saved pose-capable checkpoint for the recovered published setting
- no saved pose-capable test-prediction file
- no completed notebook output capturing the final pose-capable held-out metrics
- no explicit threshold/decision rule reported with the paper headline

##### 2. Best available paper-level reference

The **best available parity reference** is the published PIE result from
`papers/ICVES_samalouty.pdf`, because it matches the recovered Exp032 task family:

- PIE
- 30-frame observation
- next-second crossing-status target
- pose-capable multimodal MSLSTM family
- official split protocol

However, it remains a **paper-level reference**, not a replayable local artifact.

##### 3. Approximate / reference-only local evidence

The local `Final Version.ipynb` source and repo audits strongly confirm the task and model
family, but do **not** preserve the final pose-capable held-out score.

##### 4. Incompatible historical references

The following are not acceptable as exact Exp032 parity targets:

- Exp018 standardized benchmark numbers
  - different observation length and different anticipatory task
- saved `Teacher_PTH/*.pth` checkpoints
  - not pose-capable
- earlier notebook outputs such as `Previous Versions/ Different Augmentation`
  - preserved metrics are for non-pose stream combinations and/or earlier ablation settings

#### Threshold/decision-rule provenance of the parity reference

- published paper headline (`0.916 / 0.970 / 0.899`, `0.93 / 0.87`):
  **threshold rule not explicitly documented**
- current local notebook lineage:
  - checkpoint selection at validation `F1@0.5`
  - also includes a secondary validation-threshold tuning path after training

Therefore the published parity number is scientifically useful as a paper reference, but it
cannot currently anchor an exact numerical parity tolerance with fully known threshold
provenance.

#### Published-baseline parity gate status

Because there is no locally pinned pose-capable result artifact, the future Exp032 parity gate
must distinguish two questions:

1. task-level parity:
   can fresh Variant A match the recovered published/local task family?
2. score-level parity:
   can fresh Variant A fall in a reasonable neighborhood of the published paper result?

Task-level parity is sufficiently recovered.

Score-level parity is only partially recovered because the best reference is a single
published point estimate with incomplete threshold provenance and no preserved local
pose-capable checkpoint/output.

### What is recoverable

- Exp018 explicitly documents that the external paper-level MSLSTM number often cited in
  this project is a next-second PIE task, not the standardized anticipatory benchmark.
- The local notebook, repo audits, and paper are sufficient to recover the **task-defining
  provenance** required by the Exp032 hard stop:
  observation length, target horizon, label semantics, split lineage, pose source, and
  feature family.
- The pose joint order is now explicitly verified and compatible with BIU-v2 semantics.

### What remains unrecovered

- no local pose-capable final checkpoint/output for the published-setting run
- no explicit threshold/decision rule attached to the paper headline
- no multi-run published variance from which to derive a statistically grounded parity band

Implication:

- Exp032 is **not blocked by pose-joint identity**
- Exp032 is approved for implementation/training under the revised parity standard:
  the paper result is an approximate published reference only, while fresh `Variant A`
  is the true controlled baseline
- the published result remains a plausibility check:
  if fresh `Variant A` is wildly inconsistent with the paper-level reference, investigate
  before interpreting `Variant B`
- Exp032 still must not be described as an exact replay or exact numerical reproduction of
  the published paper result

## Pose compatibility with BIU-v2

### What is encouraging

- Local pose vectors are `34`-D flattened `x,y` coordinates, which is structurally
  compatible with a `17 x 2` skeletal layout.
- The extraction notebook uses YOLOv8x-pose-p6 detections and preserves detector keypoint
  order when saving normalized `x,y` pairs.
- BIU-v2 is defined over 17-joint left/right articulated geometry and can in principle be
  computed if this detector order matches the frozen COCO ordering expected by
  `src/biu_v2_extractor.py`.

### Hard-stop caveat

The extraction notebook does not label saved joint names explicitly inside the pickle.
Therefore Exp032 must **verify before training** that:

1. the recovered local pose order matches the frozen BIU-v2 COCO joint order;
2. left/right identities are preserved;
3. all BIU-v2-required joints are available;
4. any normalized coordinates can be converted back to the coordinate system expected by
   the BIU-v2 extractor without semantic distortion.

If that check fails, Exp032 must stop for review rather than silently approximating BIUs.

## Stream realism note

The local MSLSTM lineage includes annotation-rich streams such as `ped_action` and
`ped_look`. Their use is historically faithful to the local MSLSTM pipeline, but they are
not deployment-clean. Exp032 is therefore a **transfer test into an existing multimodal
architecture**, not a deployable-system claim.

## Hard-stop unresolved issues before implementation

1. **Numeric published/local reference not pinned**
   - A credible pose-capable published-setting baseline number is not yet preserved in a
     local artifact bundle.
   - Exp032 must establish the parity target before interpreting Variant B.

2. **Joint-order confirmation still required**
   - The saved `34`-D pose vectors are structurally compatible with `17 x 2`, but joint
     order is not yet explicitly proven from a named-keypoint artifact.

3. **Training recipe is recoverable only as local notebook lineage**
   - Current notebook values such as learning rate, batch size, and epoch budget are
     recoverable, but historical paper-time hyperparameters are not independently pinned
     by a paper artifact.
   - This is acceptable for Exp032 only because training details are secondary to the
     task-defining provenance above, and because Variant A parity is frozen as a gate.

## Exp032 audit decision

Proceedable as:

> MSLSTM + BIU-v2 transfer under the recovered local published/reproduced MSLSTM lineage

Not supportable as:

> exact historical replay of the published MSLSTM paper experiment
