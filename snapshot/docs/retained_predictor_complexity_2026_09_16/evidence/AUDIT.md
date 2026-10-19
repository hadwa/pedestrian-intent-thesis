# Retained-predictor complexity audit

Date: 2026-09-16. Read-only diagnostic audit; no training, feature selection, test-set model selection, checkpoint replacement, thesis change, commit or push. All outputs are under `/tmp/biu_complexity_audit_2026_09_16`. Repository snapshot: `aac84f4018cc2bfd516ffc45f3349c628815e44a`.

## Executive summary

The retained XGBoost predictors each have **250 trees**, with **5,674–5,756 nodes** (behavioral) and **5,872–5,890 nodes** (road). Their original uncompressed joblib pipelines occupy **367,836–370,624 bytes** and **377,656–378,268 bytes**, respectively. These are compact saved predictors; the node and numeric-value counts are not neural-style parameter counts.

The exact adapted MSLSTM baseline contains **6,765,321 parameters**, the retained 63-feature BIU arm **6,839,562**, and the raw-pose arm **9,006,091**. The BIU arm adds **74,241** parameters relative to that baseline; the raw-pose arm adds **2,240,770**. All are trainable during their original fitting. This is a particularly useful within-family representation-cost comparison.

Capped PedCMT contains **1,834,416 frozen parent parameters plus 100,225 trainable correction parameters**, or **1,934,641 in the complete predictor**. These counts apply separately to the retained PIE and clean-JAAD parents/branches; their weights differ. Counting only the correction would omit most of the deployed model.

All retained neural `.best.pt` checkpoints examined contain model state and selection metadata, **not optimizer state**. Inference-only state-dict files were nevertheless calculated for all seeds and exported for seed 42. Removing metadata barely changes their size; serializer alignment can even make a clean state dict a few bytes larger. Original models are unchanged.

## Seminar-ready comparison

| Predictor | Model structure / neural scalar counts | Serialized inference size | Mean / median / p95 (ms) |
|---|---|---|---|
| XGBoost behavioral (135-D) | 250 trees / 5,674–5,756 nodes | 0.351–0.353 MiB (pipeline) | 1.06 / 1.05 / 1.17 |
| XGBoost + road context (195-D) | 250 trees / 5,872–5,890 nodes | 0.360–0.361 MiB (pipeline) | 1.17 / 1.16 / 1.26 |
| MSLSTM practical baseline | 6.765M trainable scalars | 25.829 MiB (weights) | 7.75 / 7.70 / 8.30 |
| MSLSTM + BIU-v2 + 7 auxiliaries (63-D) | 6.840M trainable scalars | 26.114 MiB (weights) | 7.96 / 7.87 / 8.65 |
| MSLSTM + raw pose | 9.006M trainable scalars | 34.383 MiB (weights) | 10.44 / 10.36 / 11.09 |
| PedCMT PIE frozen parent | 1.834M frozen scalars | 7.059 MiB (weights) | 3.27 / 3.23 / 3.59 |
| PedCMT PIE complete capped predictor | 1.935M total; 0.100M trainable | 7.449 MiB (weights) | 3.48 / 3.44 / 3.78 |

**Exact slide footnote:** Tree structure/value counts are not neural scalar-weight counts; trainability refers to fitting, and uncompressed artifact sizes and one-core CPU timings exclude upstream perception.

Timings above use the same CPU, single thread and batch-one boundary for every model. The input information is not identical: XGBoost receives engineered vectors, MSLSTM combines vectors and sequence streams (including ready YOLOP descriptors), and PedCMT receives BBox/ego sequences and optionally a BIU vector. This is a matched execution-scope comparison of retained implementations, not an architecture-only ablation or equal end-to-end perception workload. MSLSTM and PedCMT are thesis adaptations, not the papers’ full published image pipelines.

## A. Model identity and configuration

All seeds 42, 43 and 44 are audited; none is chosen because it had the highest test score. Source mapping is `thesis/14_RESULT_SOURCE_MAP.md`, with retained rows from `Tables/ch4_neural.tex` and `Tables/ch4_pedcmt_capped.tex`. Exact absolute paths and SHA-256 hashes are in `sources.json`; per-instance paths are also in `counts.json` and `serialized_sizes.csv`.

| Predictor | Result/source experiment | Checkpoint relative to experiment | Exact inputs / role |
|---|---|---|---|
| XGBoost behavioral | Exp096 | `models/v2_seed{42,43,44}.joblib` | 135 = BBox53 + BIU-v2 56 + temporal/quality 21 + ego5 |
| XGBoost road | Exp116 | `models/pie_road_seed{42,43,44}.joblib` | 195 = behavioral135 + road60 |
| MSLSTM baseline | Exp053 | `checkpoints/G_seed{42,43,44}.best.pt` | BBox53 + ego speed (16×1) + ego acceleration (16×2) + YOLOP (16×20) |
| MSLSTM BIU+aux | Exp053 | `checkpoints/I_seed{42,43,44}.best.pt` | Baseline + 63-D vector: 56-D BIU-v2 + seven auxiliaries |
| MSLSTM raw pose | Exp095 | `checkpoints/J_seed{42,43,44}.best.pt` | Baseline + normalized raw pose sequence (16×34) |
| MSLSTM core56 | Exp117 | `mslstm_pie/checkpoints/I_seed{42,43,44}.best.pt` | Baseline + explicitly indexed 56-D BIU-v2 |
| PedCMT PIE parent | Exp084, retained by Exp093 | `checkpoints/A_seed{42,43,44}.best.pt` | 16×4 BBox + 16×1 ego; frozen in capped adaptation |
| PedCMT PIE correction | Exp093 | `checkpoints/R_seed{42,43,44}.best.pt` | 135-D union (72+56+7), 256-D parent anchor and parent logit |
| PedCMT clean-JAAD parent | Exp089, retained by Exp092 | `checkpoints/A_seed{42,43,44}.best.pt` | 16×4 BBox + 16×1 vehicle-state input; frozen |
| PedCMT clean-JAAD correction | Exp091, retained by Exp092 | `checkpoints/N_seed{42,43,44}.best.pt` | 135-D union and frozen-parent outputs; Exp092 applies scale 0.10 |

The audit deliberately excludes Exp093 `smoke_R_seed42.best.pt`, Exp096 `attempt_001_manifest_path_bug`, historical Combined XGBoost models, and uncapped BIU-token PedCMT models (including Exp117 PedCMT), because those are not the requested retained capped configuration. Exp084/089 parent checkpoints are included because the capped results explicitly retain and authenticate them—not merely because they are available. Each correction checkpoint’s `parent_sha256` was checked against the exact parent file.

**MSLSTM architecture.** Each sequential stream uses a two-layer bidirectional LSTM, hidden width 256 per direction, then 128-wide stream attention. Window vectors use `Linear(input,128) → ReLU → dropout → Linear(128,512)`. Stream contexts undergo non-affine LayerNorm, learned softmax fusion, and a 512→256→2 classification head. Dropout is 0.3 and disabled for inference. The 63→56 change removes exactly 7×128 = 896 projection weights.

**PedCMT parent.** Exact `PedCMTTransfer(variant="A", ego_dim=1)` from Exp084: embedding width 128, eight attention heads, four layers per attention/FFN stack, FFN width 256, bottleneck width 3 with seven projection layers. State includes intent and endpoint heads, two training loss-weight scalars and an unused legacy dense layer; none was pruned.

**Capped branch.** Exact Exp090 `DirectFusionModel("C")`, with trained weights retained in Exp093/091: parent anchor 256→128, BIU 135→128→128, four-way interaction 512→64→1, bounded correction `2*tanh(.)`; final retained probability is `sigmoid(parent_logit + 0.10*correction)`. The 0.10 scale and 2.0 bound are fixed constants, not learned branch parameters.

## B. Neural parameter counts

Models were instantiated from their actual architecture code and every checkpoint loaded with `load_state_dict(..., strict=True)`. Total = `sum(p.numel() for p in model.parameters())`; trainable uses the same sum where `p.requires_grad`; frozen = total − trainable. Parameter objects are counted once by PyTorch. Buffers are counted separately. Metadata counts were cross-checked against instantiated counts. State dictionaries do not preserve `requires_grad`: the original fitting/adaptation code establishes the freeze policy. During execution, inference mode disables autograd for all models; “trainable” here always means their fitting/adaptation role.

| Model (same counts for all three seeds) | Total | Trainable during fitting/adaptation | Frozen | Added vs baseline/parent |
|---|---|---|---|---|
| MSLSTM practical baseline | 6,765,321 | 6,765,321 | 0 | 0 |
| MSLSTM + BIU-v2 (56-D) | 6,838,666 | 6,838,666 | 0 | 73,345 |
| MSLSTM + BIU-v2 + 7 auxiliaries (63-D) | 6,839,562 | 6,839,562 | 0 | 74,241 |
| MSLSTM + raw pose | 9,006,091 | 9,006,091 | 0 | 2,240,770 |
| PedCMT PIE frozen parent | 1,834,416 | 0 | 1,834,416 | 0 |
| PedCMT PIE BIU correction only | 100,225 | 100,225 | 0 | 100,225 |
| PedCMT PIE complete capped predictor | 1,934,641 | 100,225 | 1,834,416 | 100,225 |
| PedCMT clean-JAAD frozen parent | 1,834,416 | 0 | 1,834,416 | 0 |
| PedCMT clean-JAAD BIU correction only | 100,225 | 100,225 | 0 | 100,225 |
| PedCMT clean-JAAD complete capped predictor | 1,934,641 | 100,225 | 1,834,416 | 100,225 |

MSLSTM integration deltas include the new branch plus one additional scalar stream-fusion weight: core56 = 73,344 + 1; 63-D = 74,240 + 1; raw pose = 2,174,976 LSTM + 65,793 attention + 1. The baseline itself is trained end to end; its shared parameters are not frozen in the BIU/raw-pose arms. PedCMT parents originally had all 1,834,416 parameters registered with gradients enabled, but the capped training loaders explicitly freeze all of them. The table uses that retained frozen role.

The parent total includes 1,028 registered scalars in `legacy_dense`, unused by forward, and two training loss-weight scalars; the 1,028-scalar endpoint head is also retained. Thus these are exact complete registered-model counts, not a claim that every scalar contributes to the crossing logit. Inference-only export removes training metadata, not architectural modules. See `neural_parameter_inventory.csv` for every parameter name, shape, dtype and freeze flag.

## C. XGBoost structural complexity

Every saved sklearn pipeline was loaded without fitting. We parsed the actual booster JSON and traversed every tree from root depth 0. Internal nodes have a split; leaves have an output. Tree depth is the maximum root-to-leaf edge count, and mean depth is the mean of those 250 maximum depths. Unique feature usage is the union of split feature indices, mapped back to the pipeline’s ordered input names. Each binary tree satisfies leaves = internal nodes + 1. Both configurations use 250 histogram trees, configured maximum depth 4, learning rate 0.05, row/column subsampling 0.8, binary logistic objective, and training-set class weighting. Complete loaded configurations and native arrays are saved per seed.

| Model | Seed | Trees | Internal nodes / thresholds | Leaves / outputs | Total nodes | Max / mean depth | Unique inputs used |
|---|---|---|---|---|---|---|---|
| XGBoost behavioral (135-D) | 42 | 250 | 2,753 | 3,003 | 5,756 | 4 / 4.00 | 129 |
| XGBoost behavioral (135-D) | 43 | 250 | 2,725 | 2,975 | 5,700 | 4 / 4.00 | 129 |
| XGBoost behavioral (135-D) | 44 | 250 | 2,712 | 2,962 | 5,674 | 4 / 4.00 | 128 |
| XGBoost + road context (195-D) | 42 | 250 | 2,814 | 3,064 | 5,878 | 4 / 4.00 | 182 |
| XGBoost + road context (195-D) | 43 | 250 | 2,820 | 3,070 | 5,890 | 4 / 4.00 | 181 |
| XGBoost + road context (195-D) | 44 | 250 | 2,811 | 3,061 | 5,872 | 4 / 4.00 | 182 |

“**Learned numeric tree values**” denotes split thresholds + leaf outputs: 5,674–5,756 for core and 5,872–5,890 for road. Each model additionally stores one base probability as `[5E-1]` (0.5), equivalent to log-odds intercept 0. Loaded configuration records `boost_from_average=1`; the stored baseline is 0.5. We report it separately as a stored base scalar, not an extra neural-style trainable weight. Including it gives **5,675–5,757** and **5,873–5,891 stored numeric decision values**, respectively. These quantities intentionally exclude structural arrays and ancillary training statistics.

Each internal node also records its selected feature index, child routing and default direction for missing values. Native serialization has `left_children`, `right_children`, `parents`, `split_indices`, `split_type` and `default_left` arrays (with placeholders at leaves), alongside `split_conditions`. It also retains `base_weights`, `loss_changes` and `sum_hessian`; these are stored node statistics, not independent neural-equivalent parameters. All categorical arrays are empty. The tree arrays, tree IDs and ensemble metadata contribute to file size. The saved pipeline additionally has 135/195 learned median-imputation values (1,080/1,560 float64 bytes), feature names and estimator configuration. These are reported separately rather than hidden in the threshold/leaf count.

## D. Serialized size

`Path.stat().st_size` measures the actual original artifact. KiB = bytes/1024; MiB = bytes/1,048,576. All original `.pt` files were inspected as ZIP archives and use ZIP_STORED (no compression). They contain model tensors plus selection/provenance metadata and no optimizer/scheduler state. For each neural model we serialized the complete CPU state dict to `BytesIO` with the installed `torch.save` (also uncompressed ZIP_STORED) and measured its exact length. Seed 42 inference exports are in `inference_weights/`; sizes for all seeds are in `serialized_sizes.csv`. Complete PedCMT originals are the sum of two files; a combined prefixed state dictionary is an audit export, not a previously retained monolithic checkpoint.

| Model | Original bytes (range over seeds) | Original KiB | Original MiB | Inference-only bytes | Inference-only MiB |
|---|---|---|---|---|---|
| XGBoost behavioral (135-D) | 367,836–370,624 | 359.21–361.94 | 0.351–0.353 | 367,836–370,624 | 0.351–0.353 |
| XGBoost + road context (195-D) | 377,656–378,268 | 368.80–369.40 | 0.360–0.361 | 377,656–378,268 | 0.360–0.361 |
| MSLSTM practical baseline | 27,083,767 | 26,448.99 | 25.829 | 27,083,604 | 25.829 |
| MSLSTM + BIU-v2 (56-D) | 27,378,383 | 26,736.70 | 26.110 | 27,378,440 | 26.110 |
| MSLSTM + BIU-v2 + 7 auxiliaries (63-D) | 27,381,967 | 26,740.20 | 26.113 | 27,382,024 | 26.114 |
| MSLSTM + raw pose | 36,053,039 | 35,208.05 | 34.383 | 36,052,952 | 34.383 |
| PedCMT PIE frozen parent | 7,407,136 | 7,233.53 | 7.064 | 7,402,053 | 7.059 |
| PedCMT PIE BIU correction only | 407,341 | 397.79 | 0.388 | 406,073 | 0.387 |
| PedCMT PIE complete capped predictor | 7,814,477 | 7,631.33 | 7.452 | 7,810,857 | 7.449 |
| PedCMT clean-JAAD frozen parent | 7,407,136 | 7,233.53 | 7.064 | 7,402,053 | 7.059 |
| PedCMT clean-JAAD BIU correction only | 407,341 | 397.79 | 0.388 | 406,073 | 0.387 |
| PedCMT clean-JAAD complete capped predictor | 7,814,477 | 7,631.33 | 7.452 | 7,810,857 | 7.449 |

**XGBoost format:** actual retained uncompressed joblib/pickle containing `Pipeline(SimpleImputer(strategy="median"), XGBClassifier)`. Header bytes and loaded types are recorded in `counts.json`. No training data or optimizer state is contained in these retained pipelines. A native uncompressed UBJSON booster-only export was also measured: core **358,745–361,533 bytes**, road **365,477–366,089 bytes**. It excludes the sklearn wrapper/imputer and must not be called the complete saved pipeline.

File size is a practical storage comparison, not a universal model-complexity unit. Framework versions, precision, container metadata and compression affect it. All measurements here are uncompressed and lossless. Neural rows are complete registered model weights, excluding architecture code, scaler files, framework libraries and upstream neural perception. XGBoost rows conservatively include the imputer and wrapper. Inputs for timing are already prepared; preprocessing dependencies remain necessary in a full deployment. A cleaned state dict may be slightly larger than the original because ZIP names, pickle layout and padding differ; no optimizer savings are claimed.

## E. Memory accounting

Reliable full native resident-memory and peak activation isolation was not attempted: Python, PyTorch/oneDNN and XGBoost native allocators, shared libraries and cached workspaces would confound a simple RSS subtraction. No serialized-size number is presented as RAM usage. Instead, the table below gives reproducible **resident tensor storage only**, counted from distinct parameter/buffer storages after instantiation. It is a lower bound on process memory, not peak inference memory; XGBoost has no comparable PyTorch storage measure.

| Model | Parameter bytes (float32) | Registered buffer bytes | Total tensor storage bytes | Tensor MiB |
|---|---|---|---|---|
| MSLSTM practical baseline | 27,061,284 | 0 | 27,061,284 | 25.808 |
| MSLSTM + BIU-v2 (56-D) | 27,354,664 | 0 | 27,354,664 | 26.087 |
| MSLSTM + BIU-v2 + 7 auxiliaries (63-D) | 27,358,248 | 0 | 27,358,248 | 26.091 |
| MSLSTM + raw pose | 36,024,364 | 0 | 36,024,364 | 34.356 |
| PedCMT PIE frozen parent | 7,337,664 | 5,120,000 | 12,457,664 | 11.881 |
| PedCMT PIE BIU correction only | 400,900 | 0 | 400,900 | 0.382 |
| PedCMT PIE complete capped predictor | 7,738,564 | 5,120,000 | 12,858,564 | 12.263 |

PedCMT has two non-persistent positional-encoding buffers, each `[1,5000,128]` float32: together **5,120,000 bytes**. They are recreated by the model constructor, omitted from the checkpoint state dict and are not trainable parameters. This is a concrete reason its on-disk weights and post-load tensor memory differ. PIE and clean-JAAD architectures have the same storage counts. No peak CPU/GPU memory result is claimed.

## F. Matched predictor-only CPU latency

Boundary: **ready model input → predicted probability**. Excluded: detection, pose estimation, road segmentation, BIU/BBox feature construction, normalization, file loading, tensor/DataFrame construction and device transfer. XGBoost executes its retained pipeline including imputer; MSLSTM executes its full forward and softmax; PedCMT complete executes parent forward, extracts the actual 256-D parent anchor through the original hook mechanism, runs the branch, applies 0.10 and sigmoid. Endpoint-head computation is retained, matching the source implementation. Branch-only rows start from cached parent outputs and are explicitly diagnostic.

Machine: 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz; Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.35. CPU only; process pinned to logical CPU 2, one PyTorch intra/inter-op thread, XGBoost n_jobs=1 and BLAS/OpenMP limits 1. Software: Python 3.10.20, PyTorch 2.11.0+cu128, XGBoost 3.2.0, sklearn 1.7.2. Batch 1. Each model/seed has 3 blocks × (60 warm-ups + 200 timed calls), giving **600 measured calls/seed and 1,800/model**. A fixed set of 32 evenly spaced real cached windows is reused. PIE sequence IDs are shared across XGBoost/MSLSTM/PedCMT; clean-JAAD is separately labelled. Models run sequentially, using inference mode, enabled oneDNN and no compilation or quantization. Inputs are preallocated. Garbage collection is disabled only inside timing blocks. Frequency/turbo and WSL scheduling are not locked, so these are reproducible-method local diagnostic timings, not hard real-time bounds. Order effects are not eliminated by a randomized crossover study.

| Predictor | Mean ms | Median ms | p95 ms | Timed calls |
|---|---|---|---|---|
| XGBoost behavioral (135-D) | 1.065 | 1.051 | 1.170 | 1800 |
| XGBoost + road context (195-D) | 1.174 | 1.165 | 1.258 | 1800 |
| MSLSTM practical baseline | 7.749 | 7.697 | 8.302 | 1800 |
| MSLSTM + BIU-v2 (56-D) | 7.780 | 7.714 | 8.322 | 1800 |
| MSLSTM + BIU-v2 + 7 auxiliaries (63-D) | 7.965 | 7.874 | 8.648 | 1800 |
| MSLSTM + raw pose | 10.443 | 10.362 | 11.094 | 1800 |
| PedCMT PIE frozen parent | 3.272 | 3.225 | 3.590 | 1800 |
| PedCMT PIE BIU correction only | 0.170 | 0.154 | 0.292 | 1800 |
| PedCMT PIE complete capped predictor | 3.480 | 3.443 | 3.782 | 1800 |
| PedCMT clean-JAAD frozen parent | 3.289 | 3.250 | 3.580 | 1800 |
| PedCMT clean-JAAD BIU correction only | 0.157 | 0.151 | 0.184 | 1800 |
| PedCMT clean-JAAD complete capped predictor | 3.504 | 3.469 | 3.780 | 1800 |

Replay validation checks all 32 selected window IDs against saved probabilities for every seed before timing. Maximum absolute error across all checked models was **2.6250874e-05**; all sampled classification decisions matched. XGBoost differences are near floating-point serialization precision; neural replay crosses from original GPU/batched execution to CPU/batch1. The initial exploratory tolerance 2e-5 rejected MSLSTM seed 43 at 2.6251e-5; this run was stopped and its log retained. Final replay uses the explicitly recorded 1e-4 cross-device tolerance plus exact sampled decision agreement. This does not assert bitwise GPU parity or full-test replay. Initial partial timings are excluded from the final statistics. Raw timings, per-seed summaries, prediction checks and environment are provided.

The thesis’s earlier **9.64 ms** value measures feature construction plus prediction from available perception outputs/road geometry. Its earlier ready-feature pipeline measurement was about 1.17 ms. Neither number includes upstream perception. The newly measured ready-input timings above are a separate run, with no overwriting of the earlier runtime artifacts. They must not be mixed with GPU neural timings or advertised as full video-to-decision latency.

## G. Fair interpretation and slide recommendation

A neural scalar weight or bias is optimized within a specified computational graph. A learned tree split jointly determines **which feature is tested, a threshold and routing/missing-value behavior**, while a leaf holds an output score. Counting only thresholds and leaves discards learned/discrete structure and ancillary stored fields. Parameter count also does not determine operations, feature preparation, memory traffic or kernel efficiency. Therefore a “5.9k versus 58M parameters” headline conflates definitions and model scope.

**Recommended XGBoost headline:** “250 trees; approximately 0.36 MiB saved predictor; about 1.17 ms ready-input CPU inference.” Use storage and matched latency as the main visual quantities, with tree/node count as the structural description. Keep numeric threshold/leaf values in a backup slide, labelled exactly. No storage-ratio claim should omit the formats or exclude required neural branches.

**Fairest neural comparison:** show both retained adaptations. MSLSTM demonstrates the cost of a BIU-vector branch versus raw-pose sequence processing within the same family; PedCMT demonstrates the small trainable correction relative to a frozen complete parent. For a crowded slide, use XGBoost core/road, MSLSTM63 and complete capped PedCMT, and move baseline/raw-pose/branch breakdowns to backup. The baseline-to-BIU MSLSTM increase is about 1.10% of its registered parameter count; this is a legitimate same-definition comparison, not a claim about accuracy causality.

**Do not show:** thresholds/leaves as neural-equivalent parameters; the 100,225-parameter PedCMT branch as the whole predictor; “zero parameters” for its frozen parent; downloaded paper counts as if they came from these checkpoints; CPU versus GPU bars without separate hardware labels; 9.64 ms feature+prediction beside pure neural forward latency; pickle size as RAM; or upstream-free predictor sizes as complete autonomous-driving system costs. Do not label the 135-D neural union as standalone BIU-v2.

**20–30 second speaker script:**

> We compare the retained predictors using saved model size and the same single-core CPU inference boundary. Our XGBoost model uses 250 trees and occupies about 0.36 MiB, versus 26 MiB for adapted MSLSTM and 7.45 MiB for complete capped PedCMT. It takes about 1.2 milliseconds from ready input to prediction. Upstream perception is excluded. Tree counts and neural weights are reported separately because they are different quantities.

## H. Traceability, commands and deliverables

Primary calculation code: `scripts/audit_models.py` (actual structures, strict checkpoint loading, freeze policy, tensor inventories, file sizes and inference exports); `scripts/latency.py` (ready-input reconstruction, saved-prediction replay and CPU timing); `scripts/build_report.py` (unit conversions and presentation outputs). No training entry point or experiment writer was invoked. Imported architecture modules run with bytecode writing disabled.

Commands from the active repository directory:
```bash
PYTHONDONTWRITEBYTECODE=1 /home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python /tmp/biu_complexity_audit_2026_09_16/scripts/audit_models.py
PYTHONDONTWRITEBYTECODE=1 /home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python /tmp/biu_complexity_audit_2026_09_16/scripts/latency.py
python /tmp/biu_complexity_audit_2026_09_16/scripts/build_report.py
PYTHONDONTWRITEBYTECODE=1 python /tmp/biu_complexity_audit_2026_09_16/scripts/verify_integrity.py
PYTHONDONTWRITEBYTECODE=1 /home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python /tmp/biu_complexity_audit_2026_09_16/scripts/verify_exports.py
```

Measured: actual tensor counts/storage, original serialized byte lengths, reserialized inference lengths, tree topology and latency. Read as metadata and verified: arm/seed/checkpoint selection fields, stored XGBoost configuration, parent SHA lineage and saved reference probabilities. Derived: binary units, added-branch deltas, grouped summaries, mean/median/p95. The model selection rules are retained from the source experiments; this audit does not choose models based on these timings or test scores.

Files: `counts.json`/`.csv`, `neural_parameter_inventory.csv`, `tree_inventory.csv`, `serialized_sizes.csv`, per-seed XGBoost JSON/config exports, `latency_raw.csv`, `latency_per_seed.csv`, `latency_summary.csv`, `prediction_replay.csv`, `sources.json`, `latency_sources.json`, environment JSONs, `SEMINAR.md`, `seminar_table.csv`, `seminar_table.html`, and `inference_weights/`. Source paths are absolute; all audit-generated exports are local to this folder. `repository_before.json` and `immutability_check.json` document repository and source preservation. The final check found all 6,286 tracked-file hashes, all 85 recorded input-file hashes, HEAD and the pre-existing git status unchanged. `export_checks.json` verifies inference exports reproduce the original state tensors/tree dumps without modification.

### Exact checkpoint/file inventory

| Model | Seed | Exact original file(s) |
|---|---|---|
| MSLSTM practical baseline | 42 | `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/053_mslstm_bbox53_replacement/checkpoints/G_seed42.best.pt` |
| MSLSTM + BIU-v2 + 7 auxiliaries (63-D) | 42 | `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/053_mslstm_bbox53_replacement/checkpoints/I_seed42.best.pt` |
| MSLSTM + raw pose | 42 | `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/095_mslstm_bbox53_final_controls/checkpoints/J_seed42.best.pt` |
| MSLSTM + BIU-v2 (56-D) | 42 | `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/117_neural_v2_core_completion/mslstm_pie/checkpoints/I_seed42.best.pt` |
| PedCMT PIE frozen parent | 42 | `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/084_pedcmt_biu_transfer/checkpoints/A_seed42.best.pt` |
| PedCMT PIE BIU correction only | 42 | `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/093_pedcmt_pie_capped_biu_correction/checkpoints/R_seed42.best.pt` |
| PedCMT clean-JAAD frozen parent | 42 | `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/089_pedcmt_standardized_jaad_biu_transfer/checkpoints/A_seed42.best.pt` |
| PedCMT clean-JAAD BIU correction only | 42 | `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/091_pedcmt_natural_sampling_calibration/checkpoints/N_seed42.best.pt` |
| MSLSTM practical baseline | 43 | `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/053_mslstm_bbox53_replacement/checkpoints/G_seed43.best.pt` |
| MSLSTM + BIU-v2 + 7 auxiliaries (63-D) | 43 | `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/053_mslstm_bbox53_replacement/checkpoints/I_seed43.best.pt` |
| MSLSTM + raw pose | 43 | `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/095_mslstm_bbox53_final_controls/checkpoints/J_seed43.best.pt` |
| MSLSTM + BIU-v2 (56-D) | 43 | `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/117_neural_v2_core_completion/mslstm_pie/checkpoints/I_seed43.best.pt` |
| PedCMT PIE frozen parent | 43 | `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/084_pedcmt_biu_transfer/checkpoints/A_seed43.best.pt` |
| PedCMT PIE BIU correction only | 43 | `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/093_pedcmt_pie_capped_biu_correction/checkpoints/R_seed43.best.pt` |
| PedCMT clean-JAAD frozen parent | 43 | `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/089_pedcmt_standardized_jaad_biu_transfer/checkpoints/A_seed43.best.pt` |
| PedCMT clean-JAAD BIU correction only | 43 | `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/091_pedcmt_natural_sampling_calibration/checkpoints/N_seed43.best.pt` |
| MSLSTM practical baseline | 44 | `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/053_mslstm_bbox53_replacement/checkpoints/G_seed44.best.pt` |
| MSLSTM + BIU-v2 + 7 auxiliaries (63-D) | 44 | `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/053_mslstm_bbox53_replacement/checkpoints/I_seed44.best.pt` |
| MSLSTM + raw pose | 44 | `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/095_mslstm_bbox53_final_controls/checkpoints/J_seed44.best.pt` |
| MSLSTM + BIU-v2 (56-D) | 44 | `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/117_neural_v2_core_completion/mslstm_pie/checkpoints/I_seed44.best.pt` |
| PedCMT PIE frozen parent | 44 | `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/084_pedcmt_biu_transfer/checkpoints/A_seed44.best.pt` |
| PedCMT PIE BIU correction only | 44 | `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/093_pedcmt_pie_capped_biu_correction/checkpoints/R_seed44.best.pt` |
| PedCMT clean-JAAD frozen parent | 44 | `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/089_pedcmt_standardized_jaad_biu_transfer/checkpoints/A_seed44.best.pt` |
| PedCMT clean-JAAD BIU correction only | 44 | `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/091_pedcmt_natural_sampling_calibration/checkpoints/N_seed44.best.pt` |
| PedCMT PIE complete capped predictor | 42 | `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/084_pedcmt_biu_transfer/checkpoints/A_seed42.best.pt`<br>`/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/093_pedcmt_pie_capped_biu_correction/checkpoints/R_seed42.best.pt` |
| PedCMT PIE complete capped predictor | 43 | `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/084_pedcmt_biu_transfer/checkpoints/A_seed43.best.pt`<br>`/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/093_pedcmt_pie_capped_biu_correction/checkpoints/R_seed43.best.pt` |
| PedCMT PIE complete capped predictor | 44 | `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/084_pedcmt_biu_transfer/checkpoints/A_seed44.best.pt`<br>`/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/093_pedcmt_pie_capped_biu_correction/checkpoints/R_seed44.best.pt` |
| PedCMT clean-JAAD complete capped predictor | 42 | `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/089_pedcmt_standardized_jaad_biu_transfer/checkpoints/A_seed42.best.pt`<br>`/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/091_pedcmt_natural_sampling_calibration/checkpoints/N_seed42.best.pt` |
| PedCMT clean-JAAD complete capped predictor | 43 | `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/089_pedcmt_standardized_jaad_biu_transfer/checkpoints/A_seed43.best.pt`<br>`/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/091_pedcmt_natural_sampling_calibration/checkpoints/N_seed43.best.pt` |
| PedCMT clean-JAAD complete capped predictor | 44 | `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/089_pedcmt_standardized_jaad_biu_transfer/checkpoints/A_seed44.best.pt`<br>`/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/091_pedcmt_natural_sampling_calibration/checkpoints/N_seed44.best.pt` |
| XGBoost behavioral (135-D) | 42 | `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/096_pie_exp072_matched_biu_ablation/models/v2_seed42.joblib` |
| XGBoost behavioral (135-D) | 43 | `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/096_pie_exp072_matched_biu_ablation/models/v2_seed43.joblib` |
| XGBoost behavioral (135-D) | 44 | `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/096_pie_exp072_matched_biu_ablation/models/v2_seed44.joblib` |
| XGBoost + road context (195-D) | 42 | `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/116_xgboost_v2_core_completion/models/pie_road_seed42.joblib` |
| XGBoost + road context (195-D) | 43 | `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/116_xgboost_v2_core_completion/models/pie_road_seed43.joblib` |
| XGBoost + road context (195-D) | 44 | `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean/experiments/116_xgboost_v2_core_completion/models/pie_road_seed44.joblib` |
