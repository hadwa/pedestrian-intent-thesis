# Exp051 Frozen Design

## Question

Does removing cross-stream context-scale imbalance improve the Exp050 MSLSTM variants on PIE's 16-observed-frame, TTE30--60 benchmark?

## Sole Model Change

For every active stream and every sample, apply `LayerNorm(512, elementwise_affine=False)` to its 512-D context immediately before the existing learned softmax-weighted sum.

The normalization has no learned scale or bias and adds zero parameters. It standardizes each context across its 512 dimensions. The original global fusion logits remain learned and are not forced equal.

## Frozen Invariants

- Variants A--F, active streams, inputs, cached measurements and labels are identical to Exp050.
- Observation length is 16 consecutive frames; TTE is 30--60 frames.
- Train/validation/test cohorts and ordering are identical to Exp050.
- Seeds are 42, 43 and 44; 15 epochs; Adam at `1e-4`; batch size 32.
- Oversampling, Gaussian augmentation, checkpoint selection by validation F1 at fixed 0.5, classifier and threshold remain unchanged.
- No set03 access is permitted until all 18 training runs are complete.
- Exp050 authenticated train/validation caches are reused. Pose and YOLOP extraction are not rerun.

## Comparisons

Each Exp051 variant/seed is paired against the same variant/seed in Exp050. Primary metrics are fixed-0.5 F1 and PR-AUC. ROC-AUC and the remaining fixed-0.5 metrics are secondary.

## Frozen Success Rule

Normalization succeeds for a variant if its mean PR-AUC or mean fixed-0.5 F1 improves over Exp050 and the other metric declines by no more than `0.005`. Seed consistency is reported separately and means that the paired delta has the same favorable sign for all three seeds.

## Exposure Limitation

set03 was previously evaluated in Exp050 and those results motivated this experiment. Exp051 is therefore a controlled post-hoc experiment, not a pristine untouched confirmation. All design choices above are frozen before Exp051 evaluation, and validation alone controls checkpoint selection.
