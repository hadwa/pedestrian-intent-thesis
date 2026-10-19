# Frozen PSI-native development design

One matched XGBoost experiment: A=BBox53; B=BBox53+BIU-v1(72);
C=BBox53+full BIU-v2(63). Dimensions 53/125/116. Seeds42/43/44, fixed250trees,
depth4, LR.05, subsample.8, colsample.8, reg_lambda1, four CPU threads,
training-only median imputation and negative/positive class weight. No search,
augmentation, resampling, calibration, validation checkpoint selection or ensemble.
This is a representation ablation, not reproduction of PSI's neural predictor.

Official PSI2 mean-vote binary target; threshold for annotation aggregation .5,
separate from fixed predictor threshold .5. cross/not_sure/not_cross vote1/.5/0.
All uncertain judgments retained under the official rule, including all-uncertain
ties becoming positive. Initial missing annotation is not_sure under the official
extension, followed by forward fill; preserve original already-crossed truncation.
Annotation preprocessing uses full tracks as the official label/cohort definition,
but predictor features never access beyond the 15 observed frames.

Run the pinned official extension and database functions. Use 16-frame samples
from surviving contiguous tracks >16 frames; first15 are observed, index15 is
target (next raw frame), stride1. Preserve historical split_frame_lists behavior,
including its gap handling, rather than silently fixing native cohort selection.
Document any discrepancy; exact comparison to published numbers is not claimed.
Ground-truth BBoxes from the native annotation, no detected-box deployment claim.

Train/val use the TrainVal release split108/35, identical in both split copies.
Test remains closed, including no test annotation downloads. Source-file IDs and
raw-frame indices must match exactly. No sample dropping for absent pose: zero XY
and confidence for missing cached joints/frames, report source coverage by split.
Use all native eligible train/val windows, not only cached-pose tracks; count
missing pose explicitly so incremental results cannot be attributed to cohort
filtering. Invalid native BBoxes or unknown label categories fail closed.

Reuse uncorrected PSI cache SHA31d8f45a7f8b61803b2dde8f981ce012086ac9905455d7aaac743bc1ec2a3fc7.
Its historical ViTPose-S naming/lineage is not new checkpoint authentication.
Confidence fields are reused as historical v1 quality summaries, not verified
probabilistic reliability. v2 uses the frozen no-confidence63-D extractor and
its existing BBox direction fallback; PSI image size1280x720 (official opts.py).
No engineered cue equations change. Both extractors support arbitrary window
lengths; use15 without padding to16. BBox53 uses the same available-window summaries.

Freeze design/code/source hashes before fitting. Check deterministic extraction,
dimensions, IDs, labels, split disjointness, no future access, complete feature
matrices and matching BBox columns. Report original missing-pose counts and
complete-pose-window secondary validation metrics from saved predictions only.
Report accuracy, precision, recall, binary F1, macro F1, AP, ROC-AUC and FP/FN.
Descriptive evidence for BIUs: paired positive mean macro-F1 and AP deltas vs A;
consistency requires nonnegative paired deltas in both on all3 seeds. v1 versus
v2 reported without selecting a winner from test. No test run automatically.
