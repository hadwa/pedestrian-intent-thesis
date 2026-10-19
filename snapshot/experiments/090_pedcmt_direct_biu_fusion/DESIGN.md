# Exp090 Frozen Design

## Status

This design is frozen before any Exp090 training or final-test access.

## Question

Can a direct, signed correction of a frozen PedCMT baseline use the same
automatically estimated pose measurements more effectively when they are
represented as the 135-D BIU-v1/v2 union rather than as a raw 16x34 pose
sequence?

## Protocol

- Dataset: JAAD behavior-pedestrian cohort used by Exp089.
- Observation: 16 consecutive frames.
- TTE: 30--60 frames.
- Development split: official video-disjoint train/validation memberships.
- Populations: 1,552 train windows and 274 validation windows.
- Final test: 1,501 windows; locked until the validation gate passes.
- Labels, sequence IDs, ordering, BBox, ego input, pose source, and
  training-only standardization are reused exactly from Exp089.
- Fixed decision threshold: 0.5.
- Seeds: 42, 43, 44.

The final test cohort was evaluated in Exp089 and is therefore not historically
untouched. Exp090 nevertheless keeps it sealed during model development and
uses it only if the frozen validation gate passes.

## Frozen Parents

Variant A is not retrained. Each seed reuses the exact Exp089 A checkpoint:

| Seed | SHA-256 |
|---:|---|
| 42 | `b5338a27d66686afad84a811d5d4f55adf55ff9432c036a41a59aed54db37d7b` |
| 43 | `d3b1642136c7deb6354da535f11b152198ec4c036ec7d875223fc70f7ef9043c` |
| 44 | `c84b8ba274a413f08e338d024ada6644b14f689b572439c76bdc336fe431a77b` |

The parent parameters stay frozen. Its original BBox+ego logit is reproduced
and protected exactly. The full anchor representation is the concatenation of
the final 128-D BBox and 128-D ego class-token states before the endpoint and
bottleneck heads. A learned `Linear(256,128) -> GELU -> LayerNorm(128)` maps it
to the interaction space.

## Variants

- A: exact frozen Exp089 BBox+ego parent; saved reference only.
- B: parent plus raw standardized pose. Raw pose uses the Exp089/PedCMT
  `34 -> 128` stream embedding and four 128-D temporal self-attention/feed-forward
  blocks; its class token is the 128-D extra representation.
- C: parent plus training-standardized 135-D BIU-v1/v2 union. The encoder is
  `Linear(135,128) -> GELU -> Dropout(0.3) -> Linear(128,128) -> LayerNorm(128)`.

B and C then use the exact same fusion:

1. Form `[anchor, extra, anchor * extra, abs(anchor - extra)]` (512-D).
2. Apply `Linear(512,64) -> GELU -> Dropout(0.3) -> Linear(64,1)`.
3. Bound the correction as `2 * tanh(value)`.
4. Return `final_logit = frozen_parent_logit + correction`.

The last correction layer is initialized to zero, so both branches reproduce
the parent exactly at initialization. Corrections are signed; there is no
positive-only gate. No endpoint loss, test-informed rule, calibration,
confidence input, threshold tuning, or alternative architecture is added.

## Training

- Only the new B/C branch parameters are trainable.
- Exact Exp089 deterministic class balancing is reused on training only.
- Adam, learning rate `5e-4`, betas `(0.9, 0.98)`, epsilon `1e-6`.
- Batch size 64, at most 200 epochs, patience 25.
- Checkpoint selection: minimum validation binary cross-entropy, with earlier
  epoch as the deterministic tie-break. This matches Exp089.
- B and C receive identical sample order and optimization policy.

The frozen parent is stronger baseline protection than an auxiliary parent
loss: no branch gradient can change any parent representation or parent logit.

## Validation Gate

The final test is opened only if every condition below is met after all six
B/C checkpoints are frozen:

1. C mean validation PR-AUC exceeds both A and B by at least `0.005`.
2. C mean fixed-0.5 F1 is no worse than A or B by more than `0.005`.
3. C mean fixed-0.5 accuracy is no worse than A or B by more than `0.005`.
4. The C-minus-A and C-minus-B PR-AUC deltas are positive for all three seeds.
5. Each C seed produces both positive and negative corrections on at least 1%
   of validation windows.
6. Mean C high-confidence errors (`FP >= 0.9` plus `FN <= 0.1`) do not exceed
   the lower of A and B by more than 5 cases.

If any condition fails, Exp090 ends as a validation-only negative result and
does not access or evaluate the final test cohort.

## Success Interpretation

If the gate passes and C improves over B on the final cohort, the defensible
claim is that, from identical pose measurements, a task-specific BIU union was
more useful to this protected PedCMT correction architecture than the raw pose
sequence. It must not be described as BIU adding information absent from pose.

If the gate fails or the final paired result is negative, stop this neural line;
do not tune the architecture after observing the result.
