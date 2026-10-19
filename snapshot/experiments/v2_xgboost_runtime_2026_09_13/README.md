# BIU-v2 / XGBoost runtime — 2026-09-13

[Updated thesis PDF](build/Thesis.pdf) · [Headline values](headline.csv) · [Stage statistics](stage_summary.csv)

Completed measurement of existing models, without retraining. This is an implementation/runtime audit of the retained PIE models, not a new predictive benchmark. All earlier benchmark results and archive files remain unchanged. No commit or push was made.

## Main results

Intel Core i7-11800H CPU, one XGBoost worker and one BLAS/OpenMP thread; 16-frame observations.

| Workload, ms per observation window | Core: 135 inputs | Road: 195 inputs |
|---|---:|---:|
| Ready-feature imputation + prediction, individual windows; mean across three model seeds | 1.0574 | 1.1739 |
| Ready-feature prediction, batch 100; effective per-window time | 0.01276 | 0.01431 |
| AIBThings-style effective cost, ready BBox53 + new feature construction + packing + batched prediction | 6.8222 | 6.8410 |
| Actual sequential feature construction + packing + prediction, including BBox53; seed 42 | 9.5549 | 9.6420 |
| 95th percentile of that sequential workload | 10.5865 | 10.3339 |

**Road boundary:** the road measurements start with the saved 16×20 road-relative geometry. They include construction of its 60-feature summary, but exclude deriving geometry from segmentation masks. They must not be described as complete mask-to-decision or camera-to-decision time.

All measurements exclude detection, pose estimation, road segmentation, video decoding, disk I/O, model loading and training. Arrays are materialized in RAM before entering timers. The core sequential workload includes BBox53 construction; the AIBThings-style effective estimate assumes those BBox features already exist. Batched effective cost is not single-window response latency.

The old 1.09 ms AIBThings result is a different workload: v1 + BBox / Random Forest, with ready BBox53 and batched inference. The new core includes explicit temporal and ego branches, and software versions differ. The historical number is not a controlled speed comparison of v1 against v2 or RF against XGBoost.

## Where time goes

| Feature stage | Mean ms/window | 95th percentile |
|---|---:|---:|
| BBox53 from observed boxes | 1.4743 | 1.6918 |
| BIU-v2 extractor, retaining 56 input features | 2.6890 | 2.9090 |
| Temporal21, including extractor reruns on both 8-frame halves | 3.9766 | 4.3191 |
| Ego5 | 0.0395 | 0.0519 |
| Road60 summary from existing geometry | 0.0249 | 0.0324 |

These are the existing feature implementations, not a newly optimized extractor. Component sums and separately measured sequential times can differ because they are collected in different loops. The small core/road total difference is descriptive; it is not an isolated estimate of the full road branch's cost.

## Reuse and correctness

- Models: `experiments/096_pie_exp072_matched_biu_ablation/models/v2_seed{42,43,44}.joblib` and `experiments/116_xgboost_v2_core_completion/models/pie_road_seed{42,43,44}.joblib`.
- Timing windows: the same 100 ordered IDs in `experiments/aibthings_model_only_runtime/selected_sample_ids.csv`, without choosing windows by prediction quality or runtime. The audit copy is `sample_ids.csv`.
- Core feature inputs: original pose/keypoint confidence from `/home/abdelrahman/Hadwa/pedestrian-intent-biu/data/data_keypoints_all.jsonl`; exact per-frame boxes and speed from the matching PIE XML annotations under `/home/abdelrahman/Hadwa/MSLSTM-PID/PIE/annotations/`.
- Road inputs: matching `experiments/075_pie_twinlitenetplus_road_context/cache/test_parts/set03_video_*.npz` geometry sequences.
- Frozen comparison vectors: `experiments/116_xgboost_v2_core_completion/cache/pie_road/test/{features.npy,columns.json,rows.csv}`.
- Feature functions: `src/biu_v2_extractor.py`; `experiments/037_pie_joint_biu_temporal_ego/scripts/feature_core.py`; `experiments/075_pie_twinlitenetplus_road_context/scripts/context.py`; the archive's read-only `src/features/bbox_features.py`.
- Saved probabilities: `experiments/096_pie_exp072_matched_biu_ablation/predictions/test_predictions.csv` and `experiments/116_xgboost_v2_core_completion/predictions/pie_road_test_seed{42,43,44}.csv`.

All 195 rebuilt columns match the saved vectors within a maximum absolute error of **4.88×10⁻⁸**. All six models reproduce recorded probabilities within **2.98×10⁻⁸** on the selected windows. Switching only the in-memory worker count from four to one yields identical probabilities. Original model-file hashes remain unchanged. See `feature_parity.csv`, `prediction_parity.csv`, `PARITY_PASS.json` and `sources.json` for exact source paths and hashes.

## Protocol and outputs

`DESIGN.md` was written before timing. Every model seed is reported; no fastest-model or fastest-run selection was made.

- Feature components/packing: 200 warm-up iterations and 2,000 measured window iterations.
- Each saved predictor: 200 warm-ups + 2,000 single-window calls; 20 warm-ups + 200 batch-100 calls.
- Sequential feature-to-prediction workload: seed 42, 100 warm-ups + 1,000 window calls for each configuration.
- Every single-window loop cycles through the same 100 observations.
- `raw_runs.csv`: all 29,200 recorded measurements; `stage_summary.csv`: means, medians, SD and 95th percentiles; `effective_cost.csv`: explicit component arithmetic; `headline.csv`: complete core/road summaries.
- `environment.json`: hardware, library versions, thread pools, CPU affinity, timestamps and load averages. Timing ran from 01:51:15 to 01:52:12 local time. Python 3.10.20, NumPy 2.2.6, pandas 2.3.3, XGBoost 3.2.0, scikit-learn 1.7.2. This is a CPU software-stack measurement, not an embedded-device or real-time scheduling guarantee.
- `observations.npz`: the small reused observation cache prepared for this audit; input loading is outside timing.
- `before/`, `session.patch`, `changed_files.json`: this session's thesis changes, preserving preexisting edits. The new figure replaces the old numeric runtime display with the measured v2 core workload. The compact table reports both models. The single numerical decision-rule statement and neural-only union scope are retained.

## Exact commands

From `/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean`:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/v2_xgboost_runtime_2026_09_13/run.py prepare
PYTHONDONTWRITEBYTECODE=1 /home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/v2_xgboost_runtime_2026_09_13/run.py verify
PYTHONDONTWRITEBYTECODE=1 /home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/v2_xgboost_runtime_2026_09_13/run.py time
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/v2_xgboost_runtime_2026_09_13/summarize.py
python3 experiments/v2_xgboost_runtime_2026_09_13/update_thesis.py
TEXMFVAR=/tmp/guc-thesis-texmf-var VARTEXFONTS=/tmp/guc-thesis-fonts latexmk -cd -pdf -bibtex- -interaction=nonstopmode -halt-on-error -outdir=../../experiments/v2_xgboost_runtime_2026_09_13/build thesis/manuscript/Thesis.tex
```

From `thesis/manuscript`:

```bash
biber --input-directory ../../experiments/v2_xgboost_runtime_2026_09_13/build --output-directory ../../experiments/v2_xgboost_runtime_2026_09_13/build Thesis
```

Then from the repository root:

```bash
TEXMFVAR=/tmp/guc-thesis-texmf-var VARTEXFONTS=/tmp/guc-thesis-fonts latexmk -g -cd -pdf -bibtex- -interaction=nonstopmode -halt-on-error -outdir=../../experiments/v2_xgboost_runtime_2026_09_13/build thesis/manuscript/Thesis.tex
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/v2_xgboost_runtime_2026_09_13/verify_results.py
python3 experiments/v2_xgboost_runtime_2026_09_13/verify_pdf.py
git diff --check -- thesis
```

Logs are `prepare.log`, `parity.log`, `timing.log`, `build_initial.log`, `biber.log` and `build_final.log`. Timing refuses to overwrite its recorded raw measurements. The thesis-update script is a one-time migration, not a general-purpose rerunnable generator.
