# Exp120: clean-JAAD MSLSTM representation controls

Approved JAAD-specific adaptation, 2026-09-18. Four arms, seeds42/43/44:
A = BBox53 + categorical vehicle-state sequence + historical YOLOP20 sequence;
B = A + raw XY pose [16,34]; C = A + BIU-v2 core56; D = A + core56 plus seven auxiliaries63.
No measured ego speed is claimed and no acceleration stream is fabricated.
No GT Action/Look/static context. No union, new BIUs, cap, architecture search or API cohorts.

Exact clean-JAAD Exp089 samples: 1552 train,274 validation,1501 test; official
video split;16 consecutive observed frames,TTE30--60. This is a retrospective
matched adaptation, not a reproduction of published MSLSTM scores. PIE results
are the already-completed Exp053/095/117 family and will not be retrained.

Architecture: reuse Exp053 normalized BBox53 model. Two-layer bidirectional
LSTM256 per direction plus original temporal attention for vehicle state,
YOLOP and raw pose. BBox53 and BIU vectors use d->128->512 projectors with
ReLU and dropout. Non-affine LayerNorm on each context, learned softmax scalar
stream fusion, unchanged512->256->2 classifier. Dropout.3. Full model trained.

BBox53 retains Exp053 definitions and training-only median/mean/std scaling.
BIU features are raw, as in the retained PIE family, not newly standardized.
Raw XY pose uses image-size-normalized Exp089 source coordinates, not the
PedCMT-specific training standardization. Vehicle state retains Exp089 action/4
encoding, explicitly categorical, not km/h. Existing pose/BIU cache reused.
56 columns selected by Exp012 named schema. No confidence input added.

YOLOP: exact Exp049 historical extractor and pinned checkpoint, 20-D per-frame
grid/count definition, not corrected-grid or road-relative substitution. Existing
records reused only when exact video/ped/frame matches; otherwise extract on GPU
from exact observation PNG and BBox. Never use the target/event frame as input.
Development coverage audit found all6180 required source images present and no
matching train/validation records in Exp049. Do not rerun pose estimation.

Recipe: Exp050/053 style,15epochs,Adam1e-4,batch32,unweighted cross-entropy,
best validation F1 at .5, earliest tie. Minority oversampling using existing
manifest function; Gaussian noise std.02 only on duplicated training examples'
continuous BBox/YOLOP/pose measurements. Categorical vehicle state is unchanged.
BIUs and BBox53 are recomputed once per augmented window and seed, not each
epoch. Same augmentation rows and measurements across all arms. This categorical
handling and omitted acceleration are declared JAAD adaptations, not exact PIE
stream parity. No tuning or outcome-based stopping rule.

Freeze source/code/design hashes before training. Require finite arrays, exact
IDs/frames/labels, train-val video disjointness, observation-only feature building,
training-only scaler and augmentation, parameter counts and finite GPU backward.
Train all12 models and freeze checkpoints/validation before opening test arrays.
Then one fixed final evaluation of all arms regardless of direction. No test
labels select checkpoints, thresholds or model settings. Save per-seed/mean-SD
accuracy,precision,recall,F1,AP,ROC,confusion counts and paired deltas/transitions.
Efficiency: separately report parameter counts and ready-input predictor CPU
latency; exclude pose/YOLOP/BIU feature construction from that timing. No end-to-end
speed claim. Compare within this adaptation, not against published paper numbers.

No tmux. Detached execution queues behind Exp119 to avoid simultaneous GPU jobs.
Persistent logs, per-seed completion markers, final completion only after all runs
and final reporting succeed. No old experiment/source/thesis modifications.
