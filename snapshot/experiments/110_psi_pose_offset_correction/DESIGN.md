# Exp110: PSI pose offset correction

Frozen before fitting, 2026-09-11. Training-only measurement study, not a crossing
benchmark. Number110 was free. No PSI validation/test or JAAD evaluation.

## Data and split

Reuse the authenticated29-track/24-video reference export (4,842 records),
original recovered visual annotations, and cached ViTPose points. All24 videos
are official PSI training videos. No new pose extraction. Match exact
(set,video,track,frame); assert no duplicate identity, all original hashes, COCO17
coordinates and missingness. Labels and confidence are not used for fitting.

Three video-disjoint folds: sort videos by SHA256('42:'+video), assign round-robin
fold0/1/2. Every track in a video follows that video. No fold retries. This is
training-only OOF, not an untouched confirmation: Exp017 inspected this population.

## One correction, no search

For each joint independently, fit normalized XY offset from GT minus estimated
position divided by current BBox height. Use only finite coordinates on source
joint keyframes=1 in the fitting videos. First median within each fitting video,
then median across fitting videos (equal video weight). At least3 contributing
training videos are required per joint; otherwise keep offset zero and report it.
Apply estimated_xy + offset_xy * current_bbox_height. This is34 constants, no
confidence gating, smoothing, learned orientation, network, or new BIU equations.
No future frames or GT joints at application. Original BBox is a benchmark input,
not a detector deployment claim. Invalid estimates remain invalid.

## Evaluation and advance gate

Compare unchanged versus corrected pose using identical held-out rows/masks.
Primary joint error: Euclidean position error/BBox height on keyframe joints,
macro-average over joint families (head0:5, arms5:11, lower11:17) within video,
then videos. Also report all-annotated points separately and PCK@0.1.

Reuse exact per-frame BIU-v1 equations from src/biu_extractor.py as in Exp017;
ten fixed cues and required-joint masks, no confidence input. This is not yet
the final72-D window-level or63-D v2 predictor comparison. Primary cue fidelity
requires all cue-dependent GT joints be finite and keyframe=1. Secondary uses
all annotated dependencies. Angles use wrapped absolute error, other cues use
absolute error. Scale each continuous cue by its fitting-video GT IQR (minimum1e-6),
with fixed scale1 for the binary lead_foot_lifted cue, then
macro-average cue/video normalized errors. Report each cue and its support,
never pool raw errors of different units. Normalization changes evaluation only.

Advance to JAAD planning only if BOTH primary macro joint error and primary
macro cue error improve at least2% overall, paired changes favor correction in
all3 folds for both, and primary cue comparison has >=3 held-out videos in each
fold. Otherwise stop with negative/inconclusive outcome; no alternative correction
search. No promise of better crossing F1 from better pose accuracy.

## Scope

No original caches/results altered. No source videos, crossing labels, PSI test
poses, or JAAD test predictions used. Cached file is streamed but only reference
identities retained, no out-of-cohort evaluation. Record per-fold constants,
hashes, support, paired metrics and completion. CPU-only small-array statistics;
GPU training is neither required nor beneficial for this implementation.
