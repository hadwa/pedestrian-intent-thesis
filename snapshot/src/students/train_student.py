"""Train one student configuration against frozen PIE splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.benchmark.metrics import compute_binary_metrics, find_best_threshold_on_validation
from src.students.datasets import StudentLoaderReport, TabularSequenceDataset, load_preprocessed_frames
from src.students.models import MLPStudent


def _parse_scalar(value: str):
    value = value.strip()
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [] if not inner else [_parse_scalar(item.strip()) for item in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip('"').strip("'")


def _fallback_yaml_load(path: Path) -> dict:
    result = {}
    stack = [(0, result)]
    last_key_at_indent = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        while stack and indent < stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if line.startswith("- "):
            key = last_key_at_indent.get(indent - 2)
            container = stack[-1][1]
            if not isinstance(container, list):
                raise ValueError(f"Unsupported YAML list placement in {path}: {raw}")
            container.append(_parse_scalar(line[2:]))
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            parent[key] = _parse_scalar(value)
        else:
            # Next indented block is a list for known list keys, otherwise a mapping.
            container = [] if key == "features" else {}
            parent[key] = container
            stack.append((indent + 2, container))
            last_key_at_indent[indent] = key
    return result


def read_yaml(path: Path) -> dict:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        config = _fallback_yaml_load(path)
    else:
        with path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return config

def build_frames(config: dict[str, Any], report: StudentLoaderReport | None = None) -> tuple[dict[str, pd.DataFrame], list[str], list[str]]:
    frames, feature_columns, metadata_columns, _ = load_preprocessed_frames(config, report=report, normalize=True)
    return frames, feature_columns, metadata_columns


def dry_run(config_path: Path) -> StudentLoaderReport:
    config = read_yaml(config_path)
    report = StudentLoaderReport(config_path=str(config_path), student_name=str(config.get("student_name", config_path.stem)))
    load_preprocessed_frames(config, report=report, normalize=False)
    return report


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[list[str], np.ndarray, np.ndarray]:
    model.eval()
    ids: list[str] = []
    labels: list[float] = []
    logits: list[float] = []
    with torch.no_grad():
        for x, y, sequence_ids in loader:
            x = x.to(device)
            batch_logits = model(x).cpu().numpy()
            ids.extend(sequence_ids)
            labels.extend(y.numpy().tolist())
            logits.extend(batch_logits.tolist())
    return ids, np.asarray(labels, dtype=int), np.asarray(logits, dtype=float)


def train(config_path: Path) -> dict[str, Any]:
    config = read_yaml(config_path)
    training = config.get("training", {})
    torch.manual_seed(int(training.get("seed", 42)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frames, feature_columns, metadata_columns = build_frames(config)
    datasets = {split: TabularSequenceDataset(df, feature_columns, metadata_columns) for split, df in frames.items()}
    batch_size = int(training.get("batch_size", 64))
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True),
        "val": DataLoader(datasets["val"], batch_size=batch_size, shuffle=False),
        "test": DataLoader(datasets["test"], batch_size=batch_size, shuffle=False),
    }

    model_cfg = config.get("model", {})
    model = MLPStudent(
        input_dim=len(feature_columns),
        hidden_dims=list(model_cfg.get("hidden_dims", [128, 64])),
        dropout=float(model_cfg.get("dropout", 0.2)),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training.get("learning_rate", 0.001)))
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(int(training.get("epochs", 50))):
        model.train()
        for x, y, _ in loaders["train"]:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

    outputs = {}
    for split in ["val", "test"]:
        ids, labels, logits = predict(model, loaders[split], device)
        probs = 1.0 / (1.0 + np.exp(-logits))
        outputs[split] = {"ids": ids, "labels": labels, "logits": logits, "probs": probs}

    threshold = find_best_threshold_on_validation(outputs["val"]["labels"], outputs["val"]["probs"], metric="f1")
    metrics = {
        "student_name": config.get("student_name", config_path.stem),
        "config": str(config_path),
        "validation": compute_binary_metrics(outputs["val"]["labels"], outputs["val"]["probs"], threshold),
        "test": compute_binary_metrics(outputs["test"]["labels"], outputs["test"]["probs"], threshold),
    }

    pred_rows = []
    for split in ["val", "test"]:
        data = outputs[split]
        preds = (data["probs"] >= threshold).astype(int)
        for sequence_id, label, logit, prob, pred in zip(data["ids"], data["labels"], data["logits"], data["probs"], preds):
            pred_rows.append(
                {
                    "sequence_id": sequence_id,
                    "split": split,
                    "label": int(label),
                    "student_logit": float(logit),
                    "student_prob": float(prob),
                    "student_pred": int(pred),
                }
            )
    pred_path = Path(config.get("outputs", {}).get("predictions", f"results/predictions/{config_path.stem}_predictions.csv"))
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pred_rows).to_csv(pred_path, index=False)

    metrics_path = Path(config.get("outputs", {}).get("metrics", f"results/{config_path.stem}_metrics.json"))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one student config using frozen split CSVs.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        report = dry_run(args.config)
        print(json.dumps({
            "student_name": report.student_name,
            "feature_count": report.feature_count,
            "row_counts": report.row_counts,
            "nan_before": report.nan_before,
            "nan_after": report.nan_after,
            "validation_messages": report.validation_messages,
        }, indent=2))
        return
    metrics = train(args.config)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
