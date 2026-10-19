# Exp123: frozen MSLSTM with bounded representation correction

Validation-only retrospective development; no test access. Parent is each seed's
Exp122 A AP-selected checkpoint, chosen by immediate lineage, not by comparing
test outcomes or picking a stronger earlier baseline. No parent retraining.
Clean JAAD1552 train/274 validation,16 observed frames,TTE30--60. Parent inputs
are BBox53+categorical vehicle state+historical YOLOP. No acceleration.

Three added branches: raw image-normalized XY16x34, canonical BIU56, and BIU56
plus7aux63. Reuse identical Exp122 development arrays, no new standardization,
exact balanced1998-row manifests and no Gaussian augmentation. Seeds42/43/44.
Parent is eval-mode with gradients disabled; cache its logit difference
(cross minus noncross) and256-D pre-fc2 context from observations only. Replay
validation probabilities against saved parent predictions before training.

Branch: parent256->Linear128->GELU->LayerNorm. BIU d->Linear128->ReLU->
Dropout0.3->Linear512 (same MSLSTM vector projection), then Linear512->128->
GELU->LayerNorm. Raw branch uses the original2-layer bidirectional LSTM256
per direction plus original temporal attention, then the same512->128 adapter.
Concatenate parent embedding, extra embedding, product and absolute difference;
Linear512->64->GELU->Dropout0.3->Linear1, final layer zero initialized.
Output logit = frozen parent logit +0.10*2*tanh(branch output). The added logit
is bounded to[-0.2,0.2]. Cap is applied during training AND inference; this is
not an exact reproduction of PedCMT's uncapped-training/capped-inference recipe.
The0.10 factor is a transferred fixed correction scale, not feature importance.
No cap, width or fusion search. Only baseline probabilities in approximately
[0.450166,0.549834] can change class at0.5; report reachable-error counts.

Adam1e-4,unweighted binary cross-entropy on corrected logits,batch32,15epochs,
same seed-specific row order. BCE on a logit difference is equivalent to binary
two-class cross-entropy. Select maximum validation AP, earliest tie, INCLUDING
epoch0 zero-correction identity as safety candidate. Preserve/report epoch0
selection if it wins; do not suppress neutral/negative results. Fixed threshold
0.5; no calibration. This is a new integration family, not an isolated fusion
coefficient ablation of jointly trained Exp122 models.

Freeze before training. Require parent hashes/states unchanged, cache replay,
no future/test access, sample/label/manifest parity, zero initial correction,
bounded outputs, finite gradients and no parent gradients. Nine branch trainings,
all reported with three baseline seeds, per-seed/meanSD AP/F1/ROC and confusion
counts, paired deltas/transitions, counts of class changes/error reachability,
parameter counts and persistent logs. Validation results are selected development
evidence, not proof of generalization. No automatic test evaluation or next trial.
No thesis edits, original experiment writes, commits, pushes or re-extraction.
