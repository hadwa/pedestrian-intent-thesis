# Authorized frozen PSI test evaluation

Evaluate all A/B/C models, seeds42/43/44, from completed Exp113 validation. No
fitting, threshold changes, calibration or variant selection. Fixed probability
threshold.5; retain native binary mean-vote labels, 15 inputs/next raw frame target.

Use PSI2.0_Test/splits/PSI2_split.json, SHA
c9e1bce7ed3c2f0ca04be35cae7f954942c2235c69ec0e2a97eb23f213699a10.
Its49-video test set excludes video_0192, unlike the50-video TrainVal copy.
Choose the test release's own split before annotation recovery or predictions,
not based on model performance or cached-pose coverage. Train/val membership is
identical between split copies. Report all exclusions from native processing.

Recover only those49 official pedestrian-intent JSONs, not videos or driver data.
Run unmodified pinned official extension/database functions. A separate local
staging directory is named PSI2.0_TrainVal solely because official scripts hardcode
that folder; each source is authenticated as a PSI2.0_Test archive member.
Never overwrite the actual development dataset. Mean voting includes uncertainty;
already-crossed trimming, strict track length>16 and native gap splitting unchanged.

Reuse the exact Exp113 extract function (AST-loaded without its development-only
file-access guard) and the same 53/125/116-D contracts. No future feature inputs.
Missing poses zero-filled identically; no pose-dependent population filtering.
Record pose coverage, fallback counts, label distribution and .5 vote ties.

Verify frozen source/model hashes and saved validation predictions before test.
Verify rebuilt original validation features on deterministic sampled windows before
test scoring. Report per-seed and mean/std accuracy, positive/macro F1, AP, ROC-AUC,
FP/FN, confidence-error counts, paired deltas and error transitions. Complete-pose
subset uses the same saved predictions, no new fitting. Include trivial always-cross
baseline for imbalance context. Test is authorized after validation review, no
subsequent model changes. Preserve development artifacts and stop after reporting.
