# Exp049 Feature Mapping Audit

Status: complete. Label-sealed non-YOLOP and YOLOP audits passed before final
evaluation.

## Frozen Sources

- BBox: exact JAAD track BBox for each of the 30 raw observation frames,
  converted from XYXY pixels to normalized center-X, center-Y, width, height.
- Action/Look: JAAD frame-level behavior values; numeric 1 means
  walking/looking and 0 means standing/not-looking. Missing observation values
  are zero-filled and counted.
- Ego speed: JAAD vehicle action divided by 4 and clipped to [0,1], following
  the recovered MSLSTM JAAD adapter. Ego acceleration is exactly two zeros per
  frame. No JAAD statistics are fitted.
- Static context: the exact Exp047/Exp048-lineage 23-D builder, including its
  defaults, one-hot handling, and lane grouping.
- Pose: historical JAAD ViTPose-S-lineage JSONL at
  `/home/abdelrahman/Hadwa/BIU-PID-Distillation/data/jaad_keypoints_vitpose_s_extended_short_horizon.jsonl`.
  SHA-256: `4cf675a62f158dcedf79b8037ab39b4fdc94cefe1dd8783eb6e4523d792b2b64`.
  Records contain COCO-17 joints in pixel XY plus confidence; model raw-pose
  input uses normalized XY only and excludes confidence.
- BIU-v2: exact Exp048 frozen 63-D extractor, including its established BBox
  direction fallback and excluding confidence.
- BIU-v1: exact Exp048-F frozen 72-D extractor (70 summaries plus valid-frame
  and low-confidence-frame counts).

All pose and BIU adapters request only the 30 observation frame identities and
never the next-frame target.

## Mandatory Test-Cohort Coverage Result

The label-free official-test manifest contains 44,646 windows over 273
behavior tracks. Exact identity matching against the SHA-pinned cache found:

| Quantity | Count |
|---|---:|
| Required unique `(video,ped,frame)` pose records | 52,621 |
| Required records present in cache | 10,699 |
| Required records missing from cache | 41,922 |
| Fully covered 30-frame windows | 5,331 |
| Windows missing at least one pose frame | 39,315 |

Every one of the 273 target tracks lacks at least one required pose record.
Consequently, the full official cohort cannot satisfy the frozen A--F parity
contract. Dropping missing-pose windows would reduce the cohort to 11.94% and
select it by pose availability, so this audit does not do that.

No pre-existing JAAD pose JSONL provided full coverage. The existing `cv_env`
contained the official ViTPose-S config and compatible MMPose runtime, while
the local checkpoint was pinned at SHA-256
`62d7a7128f13a63ec63d7319ea705c9822883ebf75130832cccdcb54313ddb13`.
A one-record GPU smoke extraction matched all historical coordinates and
confidence values exactly. The historical cache was then copied into the
Exp049 cache and extended without modifying the archive.

The completed Exp049 cache is
`cache/jaad_keypoints_vitpose_s_exp049_extended.jsonl`, SHA-256
`9eae258cd7fed3b1278aa86f6ab5c508fb759a24c7912d5c57677506ae86e4db`.
It contains 65,637 unique, finite, well-formed records: all 52,621 required
identities plus 13,016 historical extras. Independent reconstruction confirms
44,646 of 44,646 frozen windows have all 30 exact pose frames.

## Full Feature Audit

The complete 44,646-window pass finished in 346.26 seconds with target labels
removed from the in-memory database and without computing probabilities.

- Pose-valid windows: 44,646; invalid/all-zero windows: 0.
- Mean and median valid-joint fraction: 1.0.
- Nonfinite feature windows: 0.
- Missing Action observation values: 0.
- Missing Look observation values: 0.
- BIU-v2 direction source: hip center 44,373; ankle center 230; BBox center
  fallback 43 (0.096%).
- JAAD ego adapter range: 0.0 to 1.0; mean 0.68662.

Machine-readable evidence is stored in `metrics/feature_audit_test.json`.

## YOLOP Provenance And Frozen Mapping

YOLOP extraction is reconstructed from the historical MSLSTM notebook at
`/home/abdelrahman/Hadwa/MSLSTM-PID/Offline Extraction/yolop-extraction.ipynb`
(SHA-256 `2310285c26a88e9acc5aa019e1d301e9435b58bd8b69e7c2d9292951575f8b2f`),
the local YOLOP repository at commit
`8d8f68df318c71f01d6f813c024df646c7d1978f`, and checkpoint
`/home/abdelrahman/Hadwa/MSLSTM-PID/YOLOP/weights/End-to-end.pth`
(SHA-256 `620e7495328cfc8e04807c14cbb8f8df9916953359d190cbf494b2d049f03b0c`).
The checkpoint loads strictly into 542 state entries with zero missing or
unexpected keys and has 7,940,846 parameters.

The frozen per-frame/per-pedestrian vector is ordered as:

1. Nine 3-by-3 drivable-area mask means.
2. Nine 3-by-3 lane-line mask means.
3. Nearby vehicle count and nearby pedestrian count, each capped at 10 and
   divided by 10.

Both grids use the tracked pedestrian BBox bottom-center and a cell width and
height equal to the integer pedestrian BBox height. Nearby objects use a
450-pixel radius, COCO pedestrian class 0, and vehicle classes 2, 3, 5, and 7.
YOLOP input size is 640; detector confidence and IoU thresholds are 0.25 and
0.45. The implementation preserves the notebook's RGB conversion, division by
255, sigmoid-channel-zero masks, bilinear resizing, and uint8 conversion.

Three predeclared historical PIE records reproduced their cached 20-D vectors
exactly on the RTX 3070 (`max_abs_error = 0.0` for every case) after disabling
TF32 and enabling deterministic cuDNN. With TF32 left at the platform default,
two lane-mask averages differed by at most `1.62e-5`; that mode is therefore
not used. The formal evidence is
`metrics/yolop_historical_parity.json` with `atol=1e-6`, `rtol=0`.

The JAAD extraction uses only exact observation images and exact observation
BBoxes from the label-sealed manifest. Frame-level network outputs are reused
for all required pedestrians in the same image. No target frame, crossing
label, Exp048 checkpoint, or prediction probability is accessed during this
stage.

The completed label-sealed YOLOP cache contains 52,621 exact
pedestrian-frame vectors derived from 23,670 unique observation images across
111 videos. Extraction took 1,302.06 seconds on the RTX 3070. All 44,646
windows have 30/30 YOLOP vectors. The shared immutable feature cache contains
all ten required arrays for all windows with no missing or nonfinite values.
