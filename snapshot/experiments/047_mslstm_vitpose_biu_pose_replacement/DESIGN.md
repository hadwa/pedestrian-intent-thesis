# Exp047 Frozen Design

The authoritative approved design is:

`docs/superpowers/specs/2026-08-16-mslstm-vitpose-biu-pose-replacement-design.md`

## Immutable Protocol

- PIE, 30 consecutive observed raw frames, target at the next raw frame.
- Effective train: immutable balanced 109,934 rows from sets01,02,04.
- Validation: all 77,288 set05,set06 rows.
- Final test: all 271,078 set03 rows.
- Seeds: 42,43,44; epochs: 5; batch: 32; Adam: `1e-4`.
- Checkpoint: maximum validation F1 at fixed 0.5 after all five epochs.
- Primary reporting: fixed 0.5 plus ROC-AUC and PR-AUC.
- Non-inferiority tolerance: exactly 0.005 for PR-AUC and fixed-0.5 F1.

## Immutable Variants

1. A: YOLO raw `30x34` pose -> original two-layer BiLSTM/attention -> 512-D.
2. B: Exp012 raw `30x34` pose -> the exact same BiLSTM/attention -> 512-D.
3. C: same Exp012 pose -> frozen 63-D BIU-v2 -> Linear(63,128) -> ReLU ->
   Dropout(0.3) -> Linear(128,512).

C has no raw-pose recurrent branch and occupies the existing seventh pose slot;
BIU is not added as another stream. Exp012's frozen BBox direction fallback is
preserved and audited. Pose-estimator confidence features are excluded; the
frozen deterministic `direction_confidence` BIU remains part of the approved
63-D representation.

Set03 was inspected in separate earlier post-hoc work, which must be disclosed,
but Exp047 decisions were frozen independently. Once the pretraining audit
passes, training/validation completes before a single final set03 evaluation.
Nothing may be changed after that evaluation.
