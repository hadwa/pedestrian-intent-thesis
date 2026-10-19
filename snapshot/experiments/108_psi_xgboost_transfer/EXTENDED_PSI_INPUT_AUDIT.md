# PSI extended-XGBoost input feasibility

Read-only input feasibility check following Exp113 test. No new fit or test
evaluation. Exp075 lineage:187 BBox/BIU/temporal/ego columns plus60 TwinLiteNet+
road-relative summaries,247 total. Exp081 is a different242-D camera-motion
alternative; do not silently substitute it for the PIE extended model.

Verified inputs:

- BBoxes and cached COCO17 pose: used and aligned in Exp113.
- Ego speed: official CV frame annotation has `speed(km/hr)`. All10848 frame
  records across the24 previously recovered training video CV files contain
  float-convertible values (per-video extrema span0--62 km/hr overall).
  This is evidence of a labeled numeric speed field, not a complete telemetry
  calibration, sampling-rate or full-cohort coverage audit. Do not call driver
  decision labels speed. Reconstruct only the existing extended model's ego
  summaries; no newly invented acceleration feature.
- RGB availability: the original ZIP directory contains204 MP4 files with total
  compressed bytes5400594548. This was a metadata-only remote directory read,
  not a video download. Example member:
  PSI/PSI2.0_videos_Train-Val/videos/video_0001.mp4.
- Road/lane masks: not precomputed by this work for PSI. Generate with the same
  authenticated TwinLiteNet+ Large checkpoint/adapter as Exp075. This needs RGB
  downloads, GPU extraction, exact frame mapping and mask geometry validation.
- Prior bounded local search found the PSI JSONLs/annotation folders but no
  recovered full PSI RGB directory; not proof that no external host copy exists.

Protocol adaptation requiring an explicit frozen specification: Exp075 used
16 observations and8+8 summaries; Exp113 uses15 observations, next-frame labels.
Proposed adaptation: first7 versus last8 observed frames for every temporal/road
summary. Never pad using frame16, which is the target. Retrain on PSI; do not
apply PIE-trained weights and describe that as PSI-native training.

Next experiment should reuse Exp113 cohorts, labels, XGBoost settings and seeds;
start training/validation only. Source materials are available in principle, but
full speed/frame coverage and the road masks are not yet prepared or audited.
No claim that context improves performance is justified before this comparison.

References: Exp075 DESIGN.md and its linked frozen spec;
Exp081 DESIGN.md; Exp113 README.md and frozen feature contract;
psi_reference_recovery_2026_09_11/source_annotations/video_*.json;
official baseline README numeric-speed schema;
PSI_Dataset_1_2.zip (user-provided S3 URL, RemoteZip helper in the recovery folder).
