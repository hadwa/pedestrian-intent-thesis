# Frozen neural core56 completion

User approved proceeding after Exp116, with expensive duplicate runs avoided.
Next free number verified: 117. All outputs stay here; parent experiments,
archive, thesis and seminar remain unchanged. No commit or push.

## Bounded scope

| Case | Exact parent | New run | Reused controls |
|---|---|---|---|
| mslstm_pie | Exp053 I / Exp095 standardized matched table | BBox53 + ego speed + acceleration + YOLOP + core56 | Exp053 G/H/I and Exp095 J/K/L |
| pedcmt_pie | Exp084 D | BBox + ego speed + core56 context token | Exp084 A/B/C/D/E |
| pedcmt_jaad | Exp089 D | BBox + vehicle action + core56 context token | Exp089 A/B/C/D/E |

Three seeds (42/43/44) per new case = nine fresh trainings. Only the seven
full63-only measurements are removed. The ordered core56 names are taken from
Exp012's frozen feature definitions. Same original pose observations and frozen
BIU equations, with existing BBox direction fallback. No pose extraction.

This completes the standardized neural representation tables relevant to the
current thesis scope, not every historical failed neural trial. Native
30-frame/next-frame MSLSTM (Exp047/048/049), native random-split PedCMT controls,
v1-only capped corrections, and historical architecture searches are not rerun.
The recorded author scope decision is in
`../thesis_mslstm_benchmark_scope_2026_09_10/README.md`.

## Frozen protocols

PIE: 16 observed consecutive frames, TTE30-60; train set01/02/06 (2366), validation
set04/05 (3619), test set03 (3760). JAAD: the Exp089 clean behavior-pedestrian
cohort, 16 consecutive frames/TTE30-60, 1552/274/1501 train/validation/test windows.
No rows, labels, observation windows, or split memberships change.

MSLSTM: exact Exp053 runner/model, BBox53 training-only scaling, existing Exp050
per-seed oversampling and Gaussian-augmented caches, unchanged raw BIU scaling
policy, 15 epochs, batch32, Adam1e-4, cross-entropy. Highest validation F1 at
fixed0.5 selects the checkpoint; earlier epoch wins ties. Non-affine LayerNorm
per stream and softmax-normalized fusion remain unchanged. Projector becomes
56 -> 128 -> ReLU -> Dropout0.3 -> 512, in the same BIU stream slot.

PedCMT: exact Exp084 training routine, also used by Exp089, with 128-D embedding,
256-D feedforward, 8 heads, 4 transformer layers, original bottleneck fusion and
auxiliary event-frame BBox regression target. Future BBox is a training target,
never an input. Projector becomes 56 -> 128 -> ReLU -> Dropout0.3 -> 128 ->
LayerNorm128 and enters the same cross-modal context-token location. Reuse
training-only standardization, verified independently against original training
rows. Same positive/minority oversampling algorithm and seeded DataLoader order,
batch64, Adam5e-4/betas(0.9,0.98)/eps1e-6, maximum200 epochs, patience25,
minimum validation classification BCE with earliest tie. No new F1-based
checkpoint rule is imposed on PedCMT.

Only the BIU projector's first weight changes shape: 128x63 -> 128x56, a reduction
of 896 parameters. All other state-dict keys and tensor shapes must match the
parent. Seed streams and training procedures are retained; a changed input
dimension need not produce identical random initializations in later layers.

## Gates, reuse and evaluation

Before training: pin code/design/source hashes, prove feature-name selection,
frame/split alignment and observation-only inputs, replay all nine parent63
validation predictions on CUDA, verify train-only standardization, check model
shape/count parity, and perform a disposable training-batch gradient smoke test.
No saved parent checkpoints are retrained. Save fresh core56 checkpoints only.

Freeze all nine completed models and validation predictions before opening any
test cache or test prediction file in this run. Evaluate all three cases, with
no validation or test gate based on achieving a desired metric. Fixed threshold
0.5 throughout; PR-AUC means average precision. Report per-seed metrics, mean
and sample SD, confusion and high-confidence errors (FP>=0.9/FN<=0.1), paired
core56-minus-full63 deltas, comparisons with saved v1/raw/base controls, and
parameter counts. Replay parent63 test probabilities and verify saved metrics.
No calibration, new fusion, projector-size search, union redesign or post-test
selection. Previously exposed test cohorts make this a retrospective matched
completion, not pristine confirmation. No universal BIU winner is pre-assumed.

## Execution

GPU-only neural training, one run at a time. Persistent histories and per-seed
completion markers. Parent MSLSTM epoch-level resume is retained; interrupted
PedCMT seeds restart from their original seed (same parent behavior), while
completed seeds are reused. The final completion marker requires all nine runs
and verification; no tmux and no session-dependent external service required.

Working directory `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean`:

```bash
env LD_LIBRARY_PATH=/usr/lib/wsl/lib /home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python -u experiments/117_neural_v2_core_completion/scripts/run.py
```

CUDA requires the normal unsandboxed execution permission on this WSL host.
No changes to driver installation, credentials, or environment configuration.
