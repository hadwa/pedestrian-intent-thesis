# Exp122: remove Gaussian measurement noise, validation only

Approved successor to Exp121, frozen before training. Retrospective development
on clean JAAD; no test evaluation or test-source access. Four arms A/B/C/D and
seeds42/43/44, identical to Exp121. A=BBox53+categorical vehicle state+historical
YOLOP20; B=A+raw XY pose; C=A+canonical BIU56; D=A+BIU56+seven auxiliaries63.
No measured speed or acceleration. 1552 original train,274 validation windows,
16 consecutive frames,TTE30--60. Cached pose/scene data are reused.

Only intervention: remove Gaussian measurement augmentation from oversampled
rows. Keep the exact same balanced row manifests and order as Exp121 (999
positive,999 negative,including446 duplicated negative rows per seed). Duplicate
rows now copy original cached measurements/features exactly, without noise or
noise-related clipping/recomputation. This is random oversampling without
measurement augmentation, not natural/unbalanced sampling.

Unchanged: architecture, initialization, learned scalar normalized fusion,
feature definitions/scalers, raw unstandardized BIUs, Adam1e-4,batch32,15epochs,
training data shuffling, dropout, seeds and fixed threshold0.5. Primary checkpoint
selection: maximum validation AP, earliest tie. Retain best validation-F1
checkpoints as secondary diagnostic controls exactly as Exp121; no choice
between selection rules after viewing outcomes. AP is average precision.

Compare all arms/seeds to authenticated Exp121 AP-selected validation predictions,
plus within-Exp122 arm deltas. Report F1/AP/ROC/accuracy/precision/recall/FP/FN,
mean/sampleSD, paired seed deltas, histories and saved probabilities. No automatic
success gate, test evaluation, architecture search or further experiment.
Selection-set metrics remain development evidence, not independent proof of
generalization or a causal claim that every noise-induced cue change is harmful.

Pretraining guards: development-only paths, exact sample/label/order/manifests,
all duplicated inputs equal original rows, finite features/backward, expected
parameter counts, frozen source/code hashes. No source artifacts/thesis edits,
commit/push, pose extraction, dependency installs or tmux. Detached GPU execution
with persistent logs. Completion only after12 runs and saved-prediction checks.
