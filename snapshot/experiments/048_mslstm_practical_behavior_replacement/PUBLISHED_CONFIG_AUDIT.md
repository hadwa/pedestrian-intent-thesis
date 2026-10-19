# Published Configuration Audit

## Evidence Hierarchy

1. The detailed thesis final-model/input-stream description is authoritative
   for the target Exp048 reference configuration.
2. `papers/ICVES_samalouty.pdf` is authoritative for the published historical
   result and paper-level architectural claims.
3. The public GitHub repository, local notebooks, cached features, and Kaggle
   checkpoints are implementation lineage. They do not prove that every final
   thesis experiment or stream list was committed.

Accordingly, Exp048 Variant A preserves:

`BBox + GT Action + GT Look + Ego Speed + Ego Acceleration + GT Static Context + YOLOP`

## Paper Evidence

The ICVES paper describes 30 observed frames, normalized BBox, 34-D raw pose,
Action/Look, ego speed/acceleration/yaw, 20-D YOLOP, and 23-D static context.
It describes per-stream two-layer BiLSTMs with attention, adaptive weighted
fusion, Adam `1e-4`, batch size 32, minority oversampling with Gaussian noise
`sigma=0.02`, and reports final PIE `F1=0.8992`, precision `0.9291`, recall
`0.8711`, accuracy `0.916`, and AUC `0.970`.

The paper's broad final feature list includes raw pose and yaw, whereas the
detailed thesis final-model description supplied for Exp048 specifies no raw
pose and no yaw in the reference. Exp048 follows the thesis stream list and
uses the paper metrics only as historical references.

## Repository And Notebook Discrepancy

The closest executable final YOLOP notebook is:

`/home/abdelrahman/Hadwa/MSLSTM-PID/Previous Versions/oversampling + yolop + weighted fusion`

Its executed augmented configuration uses BBox, GT Action, GT Look, ego speed,
ego acceleration, and YOLOP, but omits static context. It implements 30
consecutive observed frames, target at the next raw frame, train sets01/02/04,
validation sets05/06, 15 epochs, Adam `1e-4`, batch size 32, adaptive weighted
fusion, validation-F1-at-0.5 checkpointing, positive-class oversampling, and
Gaussian noise `sigma=0.02` on continuous streams.

The repository README/ACTIVE_STREAMS omission of static context is recorded as
a discrepancy, not used to redefine the target. Variant A adds the thesis-
specified static stream to this closest coherent implementation lineage.

The paper says prediction one second into the future, but the recovered code
constructs `target = start + 30 + 1 - 1`, exactly one raw frame after the
observation. Exp048 reports this implemented horizon honestly.

The executed notebook uses two DataLoader worker processes. The current Codex
sandbox rejects PyTorch worker IPC sockets with `PermissionError`, verified by
an actual batch smoke test. Exp048 therefore uses `num_workers=0` identically
for A-E. This changes input-pipeline throughput only: sampler generation,
sample identities/order, deterministic augmentation, optimizer, and model
updates are unchanged.

## Kaggle Recovery

The author's public Kaggle account contains two version-1 PyTorch checkpoints.
The adaptive checkpoint has six streams: BBox, Action, Look, ego speed, ego
acceleration, and static context; it has no YOLOP or pose and 13,224,206
parameters. The older concatenation checkpoint has the same streams and
17,815,560 parameters. Neither checkpoint is the paper's final YOLOP/pose model
or an exact Variant A, so neither may replace a fresh controlled run.

## Sources Frozen For Exp048

- Paper: `papers/ICVES_samalouty.pdf`.
- Full row population:
  `experiments/032_mslstm_biu_v2_transfer/full_run_corrected_2026-08-16/metrics/protocol_rows.csv`.
- PIE XML/parser and recovered scalar lineage: the same authenticated sources
  used by Exp047.
- YOLOP: `/home/abdelrahman/Hadwa/MSLSTM-PID/Extracted YOLOP/`, exact 20-D
  frame/pedestrian caches.
- Exp012 pose and SHA-256: recorded in `README.md`.
- Frozen BIU-v2: `src/biu_v2_extractor.py`, observation frames only.

The executed notebook initializes every requested YOLOP frame to a 20-D zero
vector and replaces it only when an exact finite per-frame/per-pedestrian
artifact exists. Exp048 preserves that missing-data behavior. It separately
records valid-frame fractions and complete-window rates; malformed present
entries remain fatal. This distinction was confirmed when a legitimate frame-0
PIE window preceded the corresponding YOLOP video's first cached frame.

## Interpretation Limit

Fresh Variant A need not reproduce `F1=0.8992`. The historical runner and exact
final checkpoint are unavailable. Valid Exp048 conclusions are the paired
fresh A-E deltas under one reconstructed task, cohort, augmentation recipe,
fusion implementation, checkpoint rule, and evaluation policy.
