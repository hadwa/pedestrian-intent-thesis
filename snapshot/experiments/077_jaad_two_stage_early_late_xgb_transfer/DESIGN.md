# Exp077 Design

Exp077 carries both frozen architectures into JAAD rather than transferring
only the PIE winner. This separates cross-dataset transfer from the ability to
learn the same architecture natively on JAAD.

## Portable Inputs

JAAD has no equivalent ego-speed annotation, so Exp077 uses the exact shared
portable subset in every arm: 161 joint BBox/BIU features plus 21 fixed
early/late contrasts, for 182 full-window features. The early model uses the
161 joint features recomputed from frames 1--8. The one-stage arm has 250
trees; the sequential arm has 125 trees at each stage.

## Evaluations

The zero-shot comparison trains both portable arms on PIE training data and
applies them to JAAD test without fine-tuning, calibration, JAAD threshold
selection, or feature adaptation.

The native comparison trains both arms on JAAD train, reports JAAD validation,
then evaluates the frozen models once on JAAD test. Stage-1 training scores are
always track-grouped out-of-fold probabilities.

Both use fixed threshold 0.5 and report F1, PR-AUC, ROC-AUC, confusion counts,
and high-confidence errors. Zero-shot and native metrics are never pooled.

## JAAD Pose Lineage

The frozen Exp077 source is the extended JAAD ViTPose-S cache plus Exp030's
exact supplemental extraction for its 80 absent frames. This gives exact
coverage without substituting a nearby frame. The older archived JAAD BIU-v1
tables used `jaad_keypoints_vitpose_s.jsonl` and repaired 90 missing frames
with nearby observations. BBox53 must reproduce the archived table exactly.
BIU-v1 differences are accepted only when their sequence IDs equal Exp030's
documented 30-window repair population; any other mismatch fails closed.
