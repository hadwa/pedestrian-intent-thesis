# Approved final Exp122 test evaluation

User approved evaluation of Exp122 on2026-09-18. All four arms A/B/C/D and
seeds42/43/44, using existing maximum-validation-AP checkpoints only, earliest
tie. Threshold0.5. No training, calibration, threshold selection, alternative
checkpoint evaluation, test-based variant selection or subsequent redesign.

Same clean-JAAD Exp089 official video-held-out test cohort as Exp120:1501
windows,16 observation frames,TTE30--60. Source pose/BIU caches and exact
historical YOLOP frame records reused; no perception re-extraction. Exp122
training-only BBox53 scaler retained. Baseline=BBox53+categorical vehicle state
+YOLOP; raw pose / BIU56 / BIU56+7aux63 arms are matched additions. No acceleration.

Verify training frozen hashes/source hashes and all12 AP checkpoint identities
and selected epochs before opening test arrays. Freeze checkpoint hashes and
this evaluation code/design. Check train/validation/test video/ID disjointness,
same labels/order/frames across arms, finite inputs, saved-probability metrics,
and source/checkpoint immutability after evaluation. Output all per-seed metrics,
sample mean/SD, paired deltas, FP/FN transitions, and saved probabilities.

Retrospective controlled follow-up on an exposed test cohort; not a pristine
confirmation. Original Exp120 and Exp122 validation evidence remain intact.
Do not select among Exp120/121/122/123 using these scores. Write only here;
no thesis updates, commits, pushes or further experiments.
