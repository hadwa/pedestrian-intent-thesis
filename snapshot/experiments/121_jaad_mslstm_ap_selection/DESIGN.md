# Exp121: validation-only MSLSTM checkpoint-selection control

Approved 2026-09-18. Retrospective development following Exp120, not a fresh
held-out confirmation. No test access or evaluation is authorized here.

Reuse Exp120 clean-JAAD development cohort (1552 training,274 validation),
16 observed frames,TTE30--60. A=BBox53+categorical vehicle state+historical
YOLOP20; B=A+raw pose; C=A+canonical BIU56; D=A+BIU56+seven auxiliaries63.
No acceleration. Preserve all feature definitions, caches, normalization,
architecture, fusion, initialization, seeds42/43/44, augmentation, balanced
sampling, Adam1e-4,batch32 and15 epochs exactly. No learning-rate schedule,
calibration, cap or other model change. Fixed classification threshold0.5.

Only intervention: select maximum validation average precision (AP), earliest
epoch on exact tie, instead of maximum validation F1. AP is sklearn average
precision, not trapezoidal PR area. Save the F1-selected checkpoint from the same
trajectory as an explicit paired control. Validation selection metrics are
optimistically selected development evidence, not generalization estimates.

Exp120 retains only one best-F1 checkpoint per arm/seed. No epoch archive exists
there. Therefore replay all12 trajectories; verify histories and F1-selected
validation probabilities against Exp120. Stop on material replay differences
(history absolute tolerance1e-5, predictions2e-6). Do not claim exact parity if
this fails. Compare AP/F1/ROC/accuracy/precision/recall/FP/FN at validation only,
and record seed direction without a significance claim. No automatic test gate
or automatic subsequent experiment.

Augmentation diagnostic: reuse the exact training-only augmented rows. For each
seed and each feature compare perturbed-minus-original measurements against
original-training standard deviations (zero-variance columns flagged, not
divided). Report median/p95 standardized changes and categorical direction/flip
change rates. These are perturbation magnitudes, not proof of physical
implausibility or predictive harm. This diagnostic does not alter augmentation.

All writes confined to this new folder. Cache/source hashes and implementation
are frozen before training. Test paths denied at source access. No perception
extraction. No thesis edits, commits or pushes. Finish after validation report.
