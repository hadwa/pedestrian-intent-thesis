"""Retrain the MSLSTM-PID architecture under the frozen PIE 30-60 benchmark (experiment 018).

Retrains the multi-stream BiLSTM + temporal attention + adaptive weighted fusion
architecture from /home/abdelrahman/Hadwa/MSLSTM-PID/Final Version.ipynb, but under the
frozen PIE benchmark protocol used everywhere in this repo:
  16 observed frames, TTE 30-60, overlap 0.6, event-aligned labels,
  train=set01/02/06, val=set04/05, test=set03 (2366/3619/3760 windows).

The published MSLSTM paper number (PIE F1 0.899) is a NEXT-FRAME task (pred_len = 1 raw
frame, see archive reports/MSLSTM_PRED_LEN_VERIFICATION_REPORT.md) and must NEVER be
compared to these results. This retrain creates the honest deep baseline.

Deployment-clean input streams only (archive reports/MSLSTM_INPUT_SEMANTICS_AUDIT_REPORT.md):
  bbox  (4-D)  normalized cx,cy,w,h from the PIE database cache (image 1920x1080),
               exactly the MSLSTM notebook normalization.
  pose  (34-D) 17 COCO keypoints x/1920, y/1080 from the project keypoints JSONL,
               replicating the MSLSTM offline pose extraction normalization
               (Offline Extraction/yolo_pose_extraction.ipynb: x/vid_width, y/vid_height).
  ego   (4-D)  OBD_speed (GPS_speed fallback, as in the notebook), accX, accY, gyroZ from
               PIE vehicle_annotations; standardized with scalers recomputed on the frozen
               TRAIN split only (never the original MSLSTM scalers.pkl).
  yolop (20-D) cached per-(video,frame,ped) scene-context vectors from the MSLSTM clone
               (Extracted YOLOP/*.pkl). Read-only; never regenerated.
  biu   (11-D) 10 per-frame BIU v1 cues from src/biu_extractor.py:extract_biu_vector
               (pixel-coordinate joints), NaN->0 plus a validity flag; the 10 cue dims are
               standardized with frozen-train stats and clipped to [-10, 10].
EXCLUDED by design: ped_action, ped_look, ped_occlusion, age/gender, static_context,
traffic_light (manual annotations / leakage-adjacent, see input-semantics audit).

Variants (each x seeds 42/43/44):
  A mslstm_pose      : bbox + pose + ego + yolop
  B mslstm_biu       : bbox + biu  + ego + yolop  (BIU stream-swap ablation)
  C mslstm_bbox_only : bbox + ego                  (control)
  D mslstm_pose_no_ego: bbox + pose                (isolates ego+scene contribution)

Training: class-weighted CE (w_c = total / (2 * n_c)), Adam lr 1e-4, batch 32, up to 40
epochs; best checkpoint selected by VALIDATION F1 at threshold 0.5. Test is evaluated once
per (variant, seed) with the selected checkpoint. Thresholds: fixed 0.5 and
validation-F1-selected (via the shared _select_thresholds helper). No test data is ever
used for any selection.

Inputs (all read-only):
- Splits: /home/abdelrahman/Hadwa/BIU-PID-Distillation/data/splits_pie_wacv_pcpa_30_60_overlap06/{train,val,test}_sequences.csv
- PIE DB cache: /home/abdelrahman/Hadwa/MSLSTM-PID/PIE/annotations/data_cache/pie_database.pkl
- Keypoints: /home/abdelrahman/Hadwa/pedestrian-intent-biu/data/data_keypoints_all.jsonl
- YOLOP cache: /home/abdelrahman/Hadwa/MSLSTM-PID/Extracted YOLOP/*.pkl
- Tabular anchors: /home/abdelrahman/Hadwa/BIU-PID-Distillation/results/wacv_pcpa_30_60_overlap06/lightweight_baseline_metrics.csv

Outputs: experiments/018_mslstm_retrain_benchmark/{metrics,predictions}/
"""

from __future__ import annotations

import copy
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

CLEAN_ROOT = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation-clean")
ARCHIVE_ROOT = Path("/home/abdelrahman/Hadwa/BIU-PID-Distillation")
MSLSTM_ROOT = Path("/home/abdelrahman/Hadwa/MSLSTM-PID")
SCRIPTS_DIR = CLEAN_ROOT / "scripts/experiments"
SRC_ROOT = CLEAN_ROOT / "src"
for p in (str(SCRIPTS_DIR), str(SRC_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import jaad_biu_v2_benchmark_comparison as jaad  # noqa: E402  (reused threshold/metric helpers)
from biu_extractor import extract_biu_vector  # noqa: E402
from biu_v2_extractor import COCO_KEYPOINTS  # noqa: E402

SPLIT_DIR = ARCHIVE_ROOT / "data/splits_pie_wacv_pcpa_30_60_overlap06"
PIE_DB_CACHE = MSLSTM_ROOT / "PIE/annotations/data_cache/pie_database.pkl"
KEYPOINTS_JSONL = Path("/home/abdelrahman/Hadwa/pedestrian-intent-biu/data/data_keypoints_all.jsonl")
YOLOP_DIR = MSLSTM_ROOT / "Extracted YOLOP"
ANCHOR_SOURCE = ARCHIVE_ROOT / "results/wacv_pcpa_30_60_overlap06/lightweight_baseline_metrics.csv"

OUT_ROOT = CLEAN_ROOT / "experiments/018_mslstm_retrain_benchmark"
METRICS_DIR = OUT_ROOT / "metrics"
PREDICTIONS_DIR = OUT_ROOT / "predictions"

SPLITS = ["train", "val", "test"]
SEEDS = [42, 43, 44]

# Architecture hyperparameters (from MSLSTM Final Version.ipynb, weighted-fusion cell).
LSTM_HIDDEN_SIZE = 256
NUM_LSTM_LAYERS = 2
LSTM_DROPOUT = 0.3           # between-LSTM-layer dropout (notebook DROPOUT_RATE=0.3)
HEAD_DROPOUT_1 = 0.5         # after fusion (task spec; notebook used 0.3)
HEAD_DROPOUT_2 = 0.3         # after fc1
ATTENTION_DIM = 128
NUM_CLASSES = 2
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
NUM_EPOCHS = 40
IMG_W, IMG_H = 1920, 1080

STREAM_DIMS = {"bbox": 4, "pose": 34, "ego": 4, "yolop": 20, "biu": 11}

VARIANTS = {
    "mslstm_pose": ["bbox", "pose", "ego", "yolop"],
    "mslstm_biu": ["bbox", "biu", "ego", "yolop"],
    "mslstm_bbox_only": ["bbox", "ego"],
    "mslstm_pose_no_ego": ["bbox", "pose"],
}

BIU_CUE_NAMES = [
    "torso_lean_angle", "stride_length", "head_yaw_angle", "lead_foot_elevation",
    "lead_foot_lifted", "arm_swing_asymmetry", "gaze_to_car_angle", "body_yaw_angle",
    "face_yaw_angle", "norm_shoulder_width",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------

def load_split_rows() -> pd.DataFrame:
    dfs = []
    for split in SPLITS:
        df = pd.read_csv(SPLIT_DIR / f"{split}_sequences.csv")
        assert (df["split"] == split).all()
        dfs.append(df)
    out = pd.concat(dfs, ignore_index=True)
    assert (out["frame_step"] == 1).all()
    assert (out["frame_sampling"] == "consecutive").all()
    assert (out["observation_length"] == 16).all()
    assert ((out["end_frame"] - out["start_frame"]) == 15).all()
    assert out["sequence_id"].is_unique
    expected_sets = {"train": {"set01", "set02", "set06"}, "val": {"set04", "set05"}, "test": {"set03"}}
    for split, sets in expected_sets.items():
        assert set(out.loc[out["split"] == split, "set_id"].unique()) == sets, f"unexpected sets for {split}"
    return out


def observed_frames(row) -> list[int]:
    return list(range(int(row.start_frame), int(row.end_frame) + 1))


def load_keypoint_index(split_df: pd.DataFrame) -> dict[tuple, np.ndarray]:
    needed = set()
    for row in split_df.itertuples():
        for f in observed_frames(row):
            needed.add((row.set_id, row.video_id, row.ped_id, f))
    out = {}
    with KEYPOINTS_JSONL.open("r", encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            key = (str(rec["set"]), str(rec["video"]), str(rec["pid"]), int(rec["frame_idx"]))
            if key in needed:
                out[key] = np.asarray(rec["keypoints"], dtype=np.float64)  # (17, 2) pixel coords
    return out


class YolopCache:
    def __init__(self, directory: Path):
        self.directory = directory
        self._cache: dict[tuple[str, str], dict] = {}

    def get(self, set_id: str, video_id: str) -> dict:
        key = (set_id, video_id)
        if key not in self._cache:
            path = self.directory / f"{set_id}_{video_id}_yolop_features.pkl"
            if path.exists():
                with path.open("rb") as fp:
                    self._cache[key] = pickle.load(fp).get(f"{set_id}_{video_id}", {})
            else:
                self._cache[key] = {}
        return self._cache[key]


def biu_frame_vector(kp: np.ndarray) -> tuple[np.ndarray, int]:
    """10 BIU v1 cues + validity flag from one frame of (17,2) pixel keypoints."""
    joints = {name: kp[i] for i, name in enumerate(COCO_KEYPOINTS)}
    try:
        vec = np.asarray(extract_biu_vector(joints), dtype=np.float64)
        valid = int(np.all(np.isfinite(vec)))
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception:
        vec = np.zeros(len(BIU_CUE_NAMES), dtype=np.float64)
        valid = 0
    return vec, valid


def assemble_streams(split_df: pd.DataFrame):
    """Build raw per-window stream arrays for every frozen window; returns dict of arrays,
    aligned to split_df row order, plus missing-data stats."""
    print("Loading PIE database cache ...")
    t0 = time.time()
    with PIE_DB_CACHE.open("rb") as fp:
        pie_db = pickle.load(fp)
    print(f"  loaded in {time.time() - t0:.1f}s")

    print("Indexing keypoints JSONL ...")
    t0 = time.time()
    kp_index = load_keypoint_index(split_df)
    print(f"  {len(kp_index)} frame records in {time.time() - t0:.1f}s")

    yolop = YolopCache(YOLOP_DIR)

    n = len(split_df)
    arrays = {
        "bbox": np.zeros((n, 16, 4), dtype=np.float32),
        "pose": np.zeros((n, 16, 34), dtype=np.float32),
        "ego": np.zeros((n, 16, 4), dtype=np.float32),
        "yolop": np.zeros((n, 16, 20), dtype=np.float32),
        "biu": np.zeros((n, 16, 11), dtype=np.float32),
    }
    miss = {
        "bbox_missing_frames": 0, "pose_missing_frames": 0, "ego_missing_frames": 0,
        "yolop_missing_frames": 0, "biu_invalid_frames": 0, "total_frames": n * 16,
        "windows": n,
    }

    for i, row in enumerate(split_df.itertuples()):
        vdb = pie_db[row.set_id][row.video_id]
        pdb = vdb["ped_annotations"][row.ped_id]
        ego_db = vdb.get("vehicle_annotations", {})
        w_img = vdb.get("width", IMG_W)
        h_img = vdb.get("height", IMG_H)
        assert (w_img, h_img) == (IMG_W, IMG_H), f"unexpected PIE resolution {w_img}x{h_img}"
        frame_list = pdb["frames"]
        ymap = yolop.get(row.set_id, row.video_id)

        for t, fn in enumerate(observed_frames(row)):
            # bbox (MSLSTM notebook normalization: cx,cy,w,h over image size)
            if fn in frame_list:
                j = frame_list.index(fn)
                x1, y1, x2, y2 = pdb["bbox"][j]
                arrays["bbox"][i, t] = [((x1 + x2) / 2) / w_img, ((y1 + y2) / 2) / h_img,
                                        (x2 - x1) / w_img, (y2 - y1) / h_img]
            else:
                miss["bbox_missing_frames"] += 1

            # ego (raw; standardized later with frozen-train scalers)
            ef = ego_db.get(fn)
            if ef is not None:
                speed = ef.get("OBD_speed", 0.0) or ef.get("GPS_speed", 0.0)
                arrays["ego"][i, t] = [speed, ef.get("accX", 0.0), ef.get("accY", 0.0), ef.get("gyroZ", 0.0)]
            else:
                miss["ego_missing_frames"] += 1

            # pose + biu from keypoints JSONL
            kp = kp_index.get((row.set_id, row.video_id, row.ped_id, fn))
            if kp is not None:
                arrays["pose"][i, t] = (kp / np.array([IMG_W, IMG_H])).reshape(-1)
                cues, valid = biu_frame_vector(kp)
                arrays["biu"][i, t, :10] = cues
                arrays["biu"][i, t, 10] = valid
                if not valid:
                    miss["biu_invalid_frames"] += 1
            else:
                miss["pose_missing_frames"] += 1
                miss["biu_invalid_frames"] += 1  # biu validity flag stays 0

            # yolop
            yvec = ymap.get(fn, {}).get(row.ped_id)
            if yvec is not None:
                arrays["yolop"][i, t] = np.asarray(yvec, dtype=np.float32)
            else:
                miss["yolop_missing_frames"] += 1

    return arrays, miss


def standardize_streams(arrays: dict, split_mask_train: np.ndarray) -> pd.DataFrame:
    """Standardize ego (all 4 dims) and BIU cue dims (first 10) using TRAIN-split stats only.
    BIU standardized values are clipped to [-10, 10]. Returns the scaler table."""
    rows = []
    ego_train = arrays["ego"][split_mask_train].reshape(-1, 4)
    ego_names = ["ego_speed", "ego_accX", "ego_accY", "ego_gyroZ"]
    for d in range(4):
        mu, sd = float(ego_train[:, d].mean()), float(ego_train[:, d].std())
        sd = sd if sd > 1e-8 else 1.0
        arrays["ego"][..., d] = (arrays["ego"][..., d] - mu) / sd
        rows.append({"stream": "ego", "dim": ego_names[d], "train_mean": mu, "train_std": sd, "clip": ""})

    biu_train = arrays["biu"][split_mask_train].reshape(-1, 11)
    for d in range(10):
        mu, sd = float(biu_train[:, d].mean()), float(biu_train[:, d].std())
        sd = sd if sd > 1e-8 else 1.0
        arrays["biu"][..., d] = np.clip((arrays["biu"][..., d] - mu) / sd, -10.0, 10.0)
        rows.append({"stream": "biu", "dim": BIU_CUE_NAMES[d], "train_mean": mu, "train_std": sd, "clip": "[-10,10]"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Model (clean PyTorch reimplementation of MultiStreamWeightedAvgLSTM)
# ---------------------------------------------------------------------------

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int, attention_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim, attention_dim), nn.Tanh(), nn.Linear(attention_dim, 1))

    def forward(self, lstm_output):
        scores = self.net(lstm_output).squeeze(2)
        weights = torch.softmax(scores, dim=1)
        return torch.sum(lstm_output * weights.unsqueeze(2), dim=1)


class MultiStreamWeightedAvgLSTM(nn.Module):
    """Per-stream 2-layer BiLSTM(256) -> temporal attention -> adaptive softmax-weighted
    fusion of stream context vectors -> MLP head 512->256->2."""

    def __init__(self, stream_names: list[str], input_sizes: dict[str, int]):
        super().__init__()
        self.stream_names = list(stream_names)
        out_dim = LSTM_HIDDEN_SIZE * 2
        self.lstms = nn.ModuleDict()
        self.attentions = nn.ModuleDict()
        for name in self.stream_names:
            self.lstms[name] = nn.LSTM(
                input_sizes[name], LSTM_HIDDEN_SIZE, NUM_LSTM_LAYERS, batch_first=True,
                dropout=LSTM_DROPOUT if NUM_LSTM_LAYERS > 1 else 0.0, bidirectional=True,
            )
            self.attentions[name] = TemporalAttention(out_dim, ATTENTION_DIM)
        self.fusion_weights = nn.Parameter(torch.ones(len(self.stream_names)))
        inter_dim = out_dim // 2  # 256
        self.dropout1 = nn.Dropout(HEAD_DROPOUT_1)
        self.fc1 = nn.Linear(out_dim, inter_dim)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(HEAD_DROPOUT_2)
        self.fc2 = nn.Linear(inter_dim, NUM_CLASSES)

    def forward(self, x: dict[str, torch.Tensor]):
        ctx = []
        for name in self.stream_names:
            lstm_out, _ = self.lstms[name](x[name])
            ctx.append(self.attentions[name](lstm_out))
        stacked = torch.stack(ctx, dim=1)
        weights = torch.softmax(self.fusion_weights, dim=0).view(1, -1, 1)
        fused = torch.sum(stacked * weights, dim=1)
        out = self.dropout1(fused)
        out = self.relu(self.fc1(out))
        out = self.dropout2(out)
        return self.fc2(out)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def batched_probs(model, tensors: dict[str, torch.Tensor], n: int, batch: int = 512) -> np.ndarray:
    model.eval()
    probs = []
    with torch.no_grad():
        for s in range(0, n, batch):
            x = {k: v[s:s + batch] for k, v in tensors.items()}
            probs.append(torch.softmax(model(x), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def train_one(variant: str, streams: list[str], seed: int, data, labels, split_idx):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    tr, va, te = split_idx["train"], split_idx["val"], split_idx["test"]
    tensors = {s: {k: data[s][k] for k in streams} for s in SPLITS}
    y = {s: labels[s] for s in SPLITS}

    n1 = int(y["train"].sum())
    n0 = len(y["train"]) - n1
    total = len(y["train"])
    w = torch.tensor([total / (2.0 * n0), total / (2.0 * n1)], dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=w)

    model = MultiStreamWeightedAvgLSTM(streams, STREAM_DIMS).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    y_train_t = torch.tensor(y["train"], dtype=torch.long, device=DEVICE)

    rng = np.random.RandomState(seed)
    best_val_f1, best_state, best_epoch = -1.0, None, -1
    curves = []
    n_train = len(y["train"])
    t_start = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        order = rng.permutation(n_train)
        losses = []
        for s in range(0, n_train, BATCH_SIZE):
            idx = order[s:s + BATCH_SIZE]
            x = {k: tensors["train"][k][idx] for k in streams}
            optimizer.zero_grad()
            loss = criterion(model(x), y_train_t[idx])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        p_val = batched_probs(model, tensors["val"], len(y["val"]))
        val_m = jaad._compute_metrics(y["val"], p_val, 0.5)
        curves.append({
            "variant": variant, "seed": seed, "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_f1_at_0_5": val_m["f1"], "val_precision_at_0_5": val_m["precision"],
            "val_recall_at_0_5": val_m["recall"], "val_roc_auc": val_m["roc_auc"],
            "val_pr_auc": val_m["pr_auc"],
        })
        if val_m["f1"] > best_val_f1:
            best_val_f1 = val_m["f1"]
            best_epoch = epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})

    model.load_state_dict(best_state)
    model.to(DEVICE)
    p_val = batched_probs(model, tensors["val"], len(y["val"]))
    p_test = batched_probs(model, tensors["test"], len(y["test"]))

    pred_rows = pd.concat([
        pd.DataFrame({"sequence_id": va["sequence_id"].to_numpy(), "split": "val", "label": y["val"], "prob": p_val}),
        pd.DataFrame({"sequence_id": te["sequence_id"].to_numpy(), "split": "test", "label": y["test"], "prob": p_test}),
    ], ignore_index=True)
    pred_rows.insert(0, "variant", variant)
    pred_rows.insert(1, "seed", seed)
    pred_rows.to_csv(PREDICTIONS_DIR / f"{variant}__seed{seed}.csv", index=False)

    results = []
    for sel in jaad._select_thresholds(y["val"], p_val):
        row = {
            "variant": variant, "streams": "+".join(streams), "seed": seed,
            "threshold_rule": sel.rule, "threshold_status": sel.status,
            "selected_threshold": sel.threshold, "best_epoch": best_epoch,
            "n_params": n_params, "train_rows": len(y["train"]), "val_rows": len(y["val"]),
            "test_rows": len(y["test"]), "train_seconds": time.time() - t_start,
        }
        if sel.status == "available" and sel.threshold is not None:
            for k, v in sel.val_metrics.items():
                row[f"val_{k}"] = v
            row.update({f"test_{k}": v for k, v in jaad._compute_metrics(y["test"], p_test, float(sel.threshold)).items()})
        results.append(row)
    print(f"  {variant} seed {seed}: best epoch {best_epoch}, val F1@0.5 {best_val_f1:.4f}, "
          f"params {n_params:,}, {time.time() - t_start:.0f}s")
    return results, curves, n_params


def anchor_table() -> pd.DataFrame:
    """Re-quote exact archive tabular reference rows (validation-selected thresholds)."""
    df = pd.read_csv(ANCHOR_SOURCE)
    keep = [
        ("bbox", "random_forest"),
        ("biu_bbox", "random_forest"),
        ("teacher_t2", "xgboost"),
    ]
    rows = df[[
        (r.feature_set, r.model_name) in keep for r in df.itertuples()
    ]].copy()
    rows["anchor_source"] = str(ANCHOR_SOURCE)
    return rows


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"device: {DEVICE}, torch {torch.__version__}")
    split_df = load_split_rows()
    print(f"frozen windows: {len(split_df)} "
          f"(train={int((split_df['split'] == 'train').sum())}, "
          f"val={int((split_df['split'] == 'val').sum())}, "
          f"test={int((split_df['split'] == 'test').sum())})")

    arrays, miss = assemble_streams(split_df)
    pd.DataFrame([miss]).to_csv(METRICS_DIR / "missing_data_stats.csv", index=False)
    print("missing-data stats:", miss)

    train_mask = (split_df["split"] == "train").to_numpy()
    scaler_df = standardize_streams(arrays, train_mask)
    scaler_df.to_csv(METRICS_DIR / "train_scalers.csv", index=False)

    split_idx = {s: split_df[split_df["split"] == s].reset_index(drop=True) for s in SPLITS}
    labels = {s: split_idx[s]["label"].astype(int).to_numpy() for s in SPLITS}
    data = {}
    for s in SPLITS:
        mask = (split_df["split"] == s).to_numpy()
        data[s] = {k: torch.tensor(arrays[k][mask], dtype=torch.float32, device=DEVICE) for k in arrays}

    stream_config = []
    for variant, streams in VARIANTS.items():
        stream_config.append({
            "variant": variant,
            "streams": "+".join(streams),
            "stream_dims": json.dumps({s: STREAM_DIMS[s] for s in streams}),
            "total_input_dim": sum(STREAM_DIMS[s] for s in streams),
            "excluded_streams": "ped_action, ped_look, ped_occlusion, age/gender, static_context, traffic_light "
                                "(manual annotations / not deployment-clean; MSLSTM_INPUT_SEMANTICS_AUDIT_REPORT.md)",
        })

    all_results, all_curves, param_rows = [], [], []
    for variant, streams in VARIANTS.items():
        for seed in SEEDS:
            res, curves, n_params = train_one(variant, streams, seed, data, labels, split_idx)
            all_results.extend(res)
            all_curves.extend(curves)
            param_rows.append({"variant": variant, "streams": "+".join(streams), "seed": seed, "n_params": n_params})

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(METRICS_DIR / "mslstm_retrain_results.csv", index=False)
    pd.DataFrame(all_curves).to_csv(METRICS_DIR / "training_curves.csv", index=False)
    pd.DataFrame(stream_config).to_csv(METRICS_DIR / "stream_configuration.csv", index=False)
    pd.DataFrame(param_rows).drop_duplicates(subset=["variant"]).to_csv(METRICS_DIR / "param_counts.csv", index=False)

    metric_cols = [c for c in results_df.columns if c.startswith("test_") or c.startswith("val_") or c == "selected_threshold"]
    agg = results_df.groupby(["variant", "streams", "threshold_rule", "threshold_status"], dropna=False)[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg.reset_index().to_csv(METRICS_DIR / "mslstm_retrain_results_mean_std.csv", index=False)

    anchor_table().to_csv(METRICS_DIR / "reference_tabular_anchors.csv", index=False)

    print(json.dumps({
        "results_rows": len(results_df),
        "prediction_files": len(list(PREDICTIONS_DIR.glob("*.csv"))),
    }, indent=2))


if __name__ == "__main__":
    main()
