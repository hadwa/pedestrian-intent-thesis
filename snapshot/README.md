# BIU-PID-Distillation

This repository is the controlled experiment hub for pedestrian crossing intention prediction on the PIE dataset. Its first job is reproducibility: freeze one benchmark protocol, generate fixed train/validation/test sequence files, and force every later student, teacher, ablation, and knowledge distillation experiment to consume those same files.

No model training or KD result is comparable unless it reads all three frozen files:

- `data/splits/train_sequences.csv`
- `data/splits/val_sequences.csv`
- `data/splits/test_sequences.csv`

## Repository Goal

This repo will adapt prior BIU student feature/model code and the previous MSLSTM-PID multimodal teacher repo into one controlled experiment layout. The benchmark split and sequence IDs are the source of truth. Features, predictions, logits, and analysis outputs must be joined by `sequence_id`.

Raw PIE files, processed feature dumps, checkpoints, and large model artifacts are not committed.

## Frozen PIE Benchmark Protocol

Main dataset: PIE

Official benchmark-style split:

- train: `set01`, `set02`, `set04`
- validation: `set05`, `set06`
- test: `set03`

Main observation protocol:

- `observation_length`: 16
- `observation_unit`: raw consecutive frames
- `frame_sampling`: consecutive
- `frame_step`: 1
- no frame skipping
- `observation_span_frames`: 16
- approximate observation duration: `16 / 30 ~= 0.53` seconds, assuming 30 FPS
- `time_to_event_frames`: `[30, 60]`
- approximate time to event: 1-2 seconds, assuming 30 FPS
- `overlap`: 0.6 for PIE
- label: binary crossing / not crossing
- metrics: f1, precision, recall, accuracy, roc_auc, pr_auc
- threshold selection: choose threshold on validation set only, then apply once to test
- all scalers/statistics: fit on training data only

Important: the main benchmark does not use 16 frames with skip=2 and does not sample frames spanning about 1 second. It uses 16 consecutive raw frames with `frame_step=1` and no skipping.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Put PIE under `data/raw/` or another local path. Raw data is ignored by git.

## Generate Benchmark Sequences

The benchmark builder defines the stable sequence CSV interface, but the local PIE annotation parser boundary is intentionally explicit. Inspect your local PIE annotation files, then implement `load_pie_tracks()` in `src/benchmark/build_pie_benchmark_sequences.py`.

```bash
python -m src.benchmark.build_pie_benchmark_sequences \
  --pie_root /path/to/PIE \
  --config configs/benchmark_pie.yaml \
  --output_dir data/splits
```

Expected outputs:

- `data/splits/train_sequences.csv`
- `data/splits/val_sequences.csv`
- `data/splits/test_sequences.csv`
- `data/splits/label_distribution.json`
- `data/splits/benchmark_config.yaml`

Every later experiment must join features and predictions by `sequence_id`.

## Validate Splits

```bash
python -m src.benchmark.validate_splits --splits_dir data/splits --config configs/benchmark_pie.yaml
```

This checks required columns, duplicate `sequence_id`s, observation length, frame step, consecutive sampling, window length, and positive-sample time-to-event constraints. It writes `data/splits/split_validation_report.txt`.

## Add BIU Code

Integrate your existing BIU feature extraction in `src/features/biu_features.py`. The current module defines the expected interface and output shape, then raises clear TODO errors where project-specific keypoint loading and BIU feature logic should be pasted or adapted.

BIU feature tables must include `sequence_id` and should be joinable to the frozen sequence CSVs without changing sample membership.

## Add MSLSTM-PID Teacher Repo

Do not modify the external teacher repo directly. Keep integration code in `src/teachers/mslstm_adapter.py`.

Option A, as a git submodule:

```bash
git submodule add https://github.com/samalouty/MSLSTM-PID.git external/MSLSTM-PID
```

Option B, as a normal clone:

```bash
git clone https://github.com/samalouty/MSLSTM-PID.git external/MSLSTM-PID
```

The adapter maps the frozen sequence CSVs to teacher inputs, defines active streams from config files, and exports logits/predictions aligned by `sequence_id`.

## Student Ablations

Configs:

- `configs/student_biu.yaml`
- `configs/student_biu_bbox.yaml`
- `configs/student_biu_kp_bbox.yaml`

Dry-run planned experiments:

```bash
python -m src.students.run_student_ablation --dry_run
```

Run after feature tables exist:

```bash
python -m src.students.run_student_ablation
```

Each prediction output must include `sequence_id`, `split`, `label`, `student_logit`, `student_prob`, and `student_pred`.

## Teacher Ablations

Configs:

- `configs/teacher_ablation_bbox_pose.yaml`
- `configs/teacher_ablation_bbox_pose_yolop.yaml`
- `configs/teacher_ablation_bbox_pose_ego.yaml`
- `configs/teacher_ablation_full.yaml`

Dry-run planned experiments:

```bash
python -m src.teachers.run_teacher_ablation --dry_run
```

Run after the MSLSTM-PID integration boundary has been implemented:

```bash
python -m src.teachers.run_teacher_ablation
```

Teacher logits must be exported with `sequence_id` alignment.

## Later KD Plan

KD scaffolding lives in `src/distillation/`. Later KD work should:

1. Load frozen train/validation/test sequence CSVs.
2. Load student features by `sequence_id`.
3. Load teacher logits by `sequence_id`.
4. Train with hard labels plus KD loss.
5. Compare against the supervised student baseline using validation-selected thresholds.

KD is intentionally not the main task yet.

## Remote-Safe Long Jobs

VS Code Remote is convenient for editing, but it does not guarantee a process continues if it was launched in an interactive terminal and that session is killed. For long jobs, launch the process on the remote server with `tmux`, `screen`, or `nohup`.

The laptop can sleep or disconnect as long as the job is running inside `tmux` or under `nohup` on the remote server.

Run in tmux:

```bash
bash scripts/run_in_tmux.sh build_splits "python -m src.benchmark.build_pie_benchmark_sequences --pie_root /path/to/PIE --config configs/benchmark_pie.yaml --output_dir data/splits"
```

Reattach:

```bash
tmux attach -t build_splits
```

List sessions:

```bash
tmux ls
```

Run with nohup:

```bash
bash scripts/run_with_nohup.sh "python -m src.benchmark.validate_splits --splits_dir data/splits --config configs/benchmark_pie.yaml"
```

Tail logs:

```bash
tail -f logs/LOG_FILE.log
```

`screen` is also acceptable if that is the installed tool on the server.

## Results

Starter result files live under `results/`. They are placeholders until real experiments run. Do not create fake result rows. Do not compare experiments that did not use the same frozen split CSV files.
