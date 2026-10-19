#!/usr/bin/env python3
"""Resumable runner for the Exp053 BBox53 replacement experiment."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
REPO_ROOT = ROOT.parents[1]
EXP050_ROOT = ROOT.parents[0] / "050_mslstm_practical_behavior_replacement_pie_16f_tte30_60"
EXP051_ROOT = ROOT.parents[0] / "051_mslstm_normalized_fusion_pie_16f_tte30_60"
EXP050_RUNNER_PATH = EXP050_ROOT / "scripts/run_experiment.py"
EXP012_ROOT = ROOT.parents[0] / "012_pie_biu_v2_30_60_benchmark"
ARCHIVE_FEATURE_ROOT = Path(
    "/home/abdelrahman/Hadwa/BIU-PID-Distillation/data/processed/features_wacv_pcpa_30_60_overlap06"
)
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import exp053_core as core  # noqa: E402


def _load_exp050_runner():
    spec = importlib.util.spec_from_file_location("_exp050_runner_for_exp053", EXP050_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load Exp050 runner: {EXP050_RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = _load_exp050_runner()
base.core = core
base.ROOT = ROOT
base.REPO_ROOT = REPO_ROOT
BASE_CACHE = EXP050_ROOT / "cache/trainval_base"
TEST_CACHE = EXP050_ROOT / "cache/set03_final"
base.BASE_CACHE = BASE_CACHE
base.TEST_CACHE = TEST_CACHE
base.PREPARE_MARKER = ROOT / ".prepare.complete.json"
base.EVALUATION_MARKER = ROOT / ".evaluation.complete.json"
base.COMPLETE_MARKER = ROOT / "EXPERIMENT_COMPLETE"
_original_load_cache = base.load_cache


def required_training_identities() -> list[tuple[str, int]]:
    return [(variant, seed) for variant in core.VARIANTS for seed in core.SEEDS]


def training_marker(root: Path, variant: str, seed: int) -> Path:
    return Path(root) / f".train_{variant}_seed{seed}.complete.json"


def assert_training_complete(root: Path = ROOT) -> list[dict[str, object]]:
    authenticated = []
    for variant, seed in required_training_identities():
        marker = training_marker(root, variant, seed)
        if not marker.is_file():
            raise RuntimeError("Final evaluation requires all nine authenticated checkpoints.")
        payload = json.loads(marker.read_text(encoding="utf-8"))
        if payload.get("variant") != variant or int(payload.get("seed", -1)) != seed:
            raise RuntimeError("Final evaluation requires all nine authenticated checkpoints.")
        try:
            base.authenticate_artifacts(root, payload.get("artifacts", {}))
        except ValueError as error:
            raise RuntimeError("Final evaluation requires all nine authenticated checkpoints.") from error
        authenticated.append(payload)
    return authenticated


def _archive_table(splits: tuple[str, ...]) -> pd.DataFrame:
    tables = [pd.read_csv(ARCHIVE_FEATURE_ROOT / f"bbox_{split}.csv") for split in splits]
    table = pd.concat(tables, ignore_index=True)
    missing = [name for name in core.BBOX53_FEATURES if name not in table.columns]
    if missing:
        raise ValueError(f"Archived BBox53 columns missing: {missing}")
    return table


def _align_archive(rows: pd.DataFrame, splits: tuple[str, ...]) -> np.ndarray:
    table = _archive_table(splits).set_index("sequence_id", drop=False)
    ids = rows["sequence_id"].astype(str)
    if len(table) != len(rows) or set(table.index.astype(str)) != set(ids):
        raise ValueError("Archived BBox53 population does not match frozen rows.")
    aligned = table.loc[ids]
    if not np.array_equal(aligned["label"].to_numpy(dtype=np.int64), rows["label"].to_numpy(dtype=np.int64)):
        raise ValueError("Archived BBox53 labels do not match frozen rows.")
    return aligned[list(core.BBOX53_FEATURES)].to_numpy(dtype=np.float64)


def _scaler_payload(scaler: dict[str, np.ndarray]) -> dict[str, list[float]]:
    return {name: np.asarray(values, dtype=np.float64).tolist() for name, values in scaler.items()}


def _load_scaler() -> dict[str, np.ndarray]:
    payload = json.loads((ROOT / "metrics/bbox53_scaler.json").read_text(encoding="utf-8"))
    return {name: np.asarray(payload[name], dtype=np.float64) for name in ("median", "mean", "std")}


def _bbox53_cache_path(kind: str) -> Path:
    return ROOT / f"cache/bbox53_{kind}.npy"


def load_cache(path: Path):
    candidate = Path(path)
    try:
        relative = candidate.relative_to(ROOT / "cache")
    except ValueError:
        rows, arrays, metadata = _original_load_cache(candidate)
        if candidate == BASE_CACHE and _bbox53_cache_path("trainval").is_file():
            arrays["bbox53"] = np.load(_bbox53_cache_path("trainval"), mmap_mode="r")
        elif candidate == TEST_CACHE and _bbox53_cache_path("test").is_file():
            arrays["bbox53"] = np.load(_bbox53_cache_path("test"), mmap_mode="r")
        return rows, arrays, metadata
    source_path = EXP050_ROOT / "cache" / relative
    rows, arrays, metadata = _original_load_cache(source_path)
    if relative.name.startswith("augmented_seed"):
        seed = int(relative.name.replace("augmented_seed", ""))
        arrays["bbox53"] = np.load(_bbox53_cache_path(f"augmented_seed{seed}"), mmap_mode="r")
    return rows, arrays, metadata


base.load_cache = load_cache


def _recomputed_bbox53(rows: pd.DataFrame, boxes: np.ndarray, positions: np.ndarray) -> np.ndarray:
    output = np.empty((len(positions), 53), dtype=np.float64)
    for destination, position in enumerate(positions):
        row = rows.iloc[int(position)]
        output[destination] = core.bbox53_from_normalized_bbox(
            boxes[int(position)],
            width=int(row["frame_width"]),
            height=int(row["frame_height"]),
        )
    return output


def run_prepare() -> None:
    if base.PREPARE_MARKER.is_file():
        payload = json.loads(base.PREPARE_MARKER.read_text(encoding="utf-8"))
        base.authenticate_artifacts(ROOT, payload["artifacts"])
        base.log_line(ROOT / "logs/prepare.log", "status=already_complete_authenticated")
        return
    source_marker = EXP050_ROOT / ".prepare.complete.json"
    source = json.loads(source_marker.read_text(encoding="utf-8"))
    if source.get("test_accessed") is not False:
        raise RuntimeError("Exp050 preparation does not prove set03 exclusion.")
    base.authenticate_artifacts(EXP050_ROOT, source["artifacts"])
    base_rows, base_arrays, _ = _original_load_cache(BASE_CACHE)
    archived = _align_archive(base_rows, ("train", "val"))
    train_mask = base_rows["split"].eq("train").to_numpy()
    scaler = core.fit_bbox53_scaler(archived[train_mask])
    transformed = core.transform_bbox53(archived, scaler)
    base.atomic_npy(transformed, _bbox53_cache_path("trainval"))
    base.atomic_json(_scaler_payload(scaler), ROOT / "metrics/bbox53_scaler.json")

    all_positions = np.arange(len(base_rows), dtype=np.int64)
    recomputed = _recomputed_bbox53(base_rows, base_arrays["bbox"], all_positions)
    standardized_error = np.abs(core.transform_bbox53(recomputed, scaler) - transformed)
    audit_records = [
        {
            "scope": "trainval_original_roundtrip",
            "rows": len(base_rows),
            "features": 53,
            "max_standardized_abs_error": float(np.max(standardized_error)),
            "mean_standardized_abs_error": float(np.mean(standardized_error)),
            "archived_original_values_used": True,
        }
    ]
    for seed in core.SEEDS:
        manifest, augmented_arrays, _ = _original_load_cache(EXP050_ROOT / f"cache/augmented_seed{seed}")
        source_indices = manifest["source_cache_index"].to_numpy(dtype=np.int64)
        augmented_mask = manifest["augmented"].to_numpy(dtype=bool)
        raw = archived[source_indices].copy()
        augmented_positions = np.flatnonzero(augmented_mask)
        for position in augmented_positions:
            source_index = int(source_indices[position])
            row = base_rows.iloc[source_index]
            raw[position] = core.bbox53_from_normalized_bbox(
                augmented_arrays["bbox"][position],
                width=int(row["frame_width"]),
                height=int(row["frame_height"]),
            )
        augmented_transformed = core.transform_bbox53(raw, scaler)
        if not np.array_equal(augmented_transformed[~augmented_mask], transformed[source_indices[~augmented_mask]]):
            raise RuntimeError("Non-augmented BBox53 rows changed during augmented-cache construction.")
        base.atomic_npy(augmented_transformed, _bbox53_cache_path(f"augmented_seed{seed}"))
        audit_records.append(
            {
                "scope": f"augmented_seed{seed}",
                "rows": len(manifest),
                "features": 53,
                "max_standardized_abs_error": 0.0,
                "mean_standardized_abs_error": 0.0,
                "archived_original_values_used": True,
                "recomputed_augmented_rows": int(augmented_mask.sum()),
            }
        )
    audit_path = ROOT / "metrics/bbox53_provenance_audit.csv"
    base.atomic_csv(pd.DataFrame(audit_records), audit_path)
    artifacts = {
        "cache/bbox53_trainval.npy": core.sha256_file(_bbox53_cache_path("trainval")),
        "metrics/bbox53_scaler.json": core.sha256_file(ROOT / "metrics/bbox53_scaler.json"),
        "metrics/bbox53_provenance_audit.csv": core.sha256_file(audit_path),
    }
    for seed in core.SEEDS:
        relative = f"cache/bbox53_augmented_seed{seed}.npy"
        artifacts[relative] = core.sha256_file(ROOT / relative)
    base.atomic_json(
        {
            "phase": "prepare",
            "created_at": base.utc_now(),
            "train_rows": core.EXPECTED_SPLIT_COUNTS["train"],
            "val_rows": core.EXPECTED_SPLIT_COUNTS["val"],
            "test_accessed": False,
            "source_prepare_sha256": core.sha256_file(source_marker),
            "feature_names": list(core.BBOX53_FEATURES),
            "scalers": source["scalers"],
            "artifacts": artifacts,
        },
        base.PREPARE_MARKER,
    )
    base.log_line(ROOT / "logs/prepare.log", "phase=prepare status=complete test_accessed=false")


def run_train_all() -> None:
    for variant, seed in required_training_identities():
        base.run_training(variant, seed)
    assert_training_complete(ROOT)
    base.atomic_json({"phase": "train_all", "created_at": base.utc_now(), "run_count": 9}, ROOT / "TRAINING_COMPLETE")


def _prepare_test_bbox53(test_rows: pd.DataFrame) -> None:
    archived = _align_archive(test_rows, ("test",))
    transformed = core.transform_bbox53(archived, _load_scaler())
    base.atomic_npy(transformed, _bbox53_cache_path("test"))


def _condition(variant: str) -> str:
    return {"C": "core", "G": "core", "F": "biu_v1", "H": "biu_v1", "E": "biu_v2", "I": "biu_v2"}[variant]


def run_evaluate() -> None:
    if base.EVALUATION_MARKER.is_file():
        payload = json.loads(base.EVALUATION_MARKER.read_text(encoding="utf-8"))
        base.authenticate_artifacts(ROOT, payload["artifacts"])
        base.log_line(ROOT / "logs/evaluate.log", "status=already_complete_authenticated")
        return
    training = assert_training_complete(ROOT)
    if not torch.cuda.is_available():
        raise RuntimeError("Exp053 final inference is GPU-only; CUDA is unavailable.")
    test_rows, _, test_metadata = _original_load_cache(TEST_CACHE)
    if set(test_rows["split"].astype(str)) != {"test"} or set(test_rows["set_id"].astype(str)) != {"set03"}:
        raise RuntimeError("Final cache is not designated set03.")
    _prepare_test_bbox53(test_rows)
    test_rows, test_arrays, _ = load_cache(TEST_CACHE)
    labels = test_rows["label"].to_numpy(dtype=np.int64)
    device = torch.device("cuda")
    new_records = []
    predictions: dict[tuple[str, int], pd.DataFrame] = {}
    efficiency = []
    for variant, seed in required_training_identities():
        checkpoint = torch.load(ROOT / f"checkpoints/{variant}_seed{seed}.best.pt", map_location="cpu", weights_only=False)
        model = core.MslstmBbox53Model(variant).to(device)
        model.load_state_dict(checkpoint["model_state"])
        streams = core.VARIANT_STREAMS[variant]
        tensors = {name: base.tensor_for_stream(test_arrays[name], name, device) for name in streams}
        torch.cuda.synchronize()
        started = time.perf_counter()
        probability = base.probabilities(model, tensors, len(test_rows))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        metrics = core.compute_metrics(labels, probability)
        train_record = next(item for item in training if item["variant"] == variant and int(item["seed"]) == seed)
        new_records.append(
            {
                "condition": _condition(variant),
                "representation": "bbox53",
                "variant": variant,
                "seed": seed,
                "best_epoch": int(checkpoint["best_epoch"]),
                "best_validation_f1": float(checkpoint["best_validation_f1"]),
                "total_parameters": int(checkpoint["total_parameters"]),
                "training_seconds": float(train_record["training_seconds"]),
                **metrics,
            }
        )
        frame = test_rows[["sequence_id", "split", "set_id", "video_id", "ped_id", "start_frame", "end_frame", "event_frame", "time_to_event_frames", "label"]].copy()
        frame["variant"] = variant
        frame["seed"] = seed
        frame["threshold"] = core.FIXED_THRESHOLD
        frame["probability"] = probability
        frame["prediction"] = (probability >= core.FIXED_THRESHOLD).astype(np.int64)
        path = ROOT / f"predictions/set03_{variant}_seed{seed}.csv"
        base.atomic_csv(frame, path)
        predictions[(variant, seed)] = frame
        efficiency.append({"variant": variant, "seed": seed, "device": str(device), "total_parameters": int(checkpoint["total_parameters"]), "inference_seconds": elapsed, "seconds_per_sample": elapsed / len(test_rows)})
        base.log_line(ROOT / "logs/evaluate.log", f"variant={variant} seed={seed} f1={metrics['f1']:.6f} pr_auc={metrics['pr_auc']:.6f}")
        del model, tensors
        torch.cuda.empty_cache()

    exp051_metrics = pd.read_csv(EXP051_ROOT / "metrics/per_seed_results.csv").set_index(["variant", "seed"])
    raw_records = []
    comparisons = []
    transitions = []
    metric_names = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")
    new_frame = pd.DataFrame(new_records).set_index(["variant", "seed"])
    for new_variant, raw_variant in core.RAW_CONTROL_VARIANTS.items():
        for seed in core.SEEDS:
            raw_metric = exp051_metrics.loc[(raw_variant, seed)]
            raw_records.append(
                {
                    "condition": _condition(raw_variant),
                    "representation": "raw_bbox",
                    "variant": raw_variant,
                    "seed": seed,
                    "best_epoch": int(raw_metric["best_epoch"]),
                    "best_validation_f1": float(raw_metric["best_validation_f1"]),
                    "total_parameters": int(raw_metric["total_parameters"]),
                    "training_seconds": float(raw_metric["training_seconds"]),
                    **{name: float(raw_metric[name]) for name in metric_names},
                }
            )
            delta = {"comparison": f"{raw_variant}_to_{new_variant}", "condition": _condition(new_variant), "seed": seed}
            for name in metric_names:
                delta[f"{name}_delta"] = float(new_frame.loc[(new_variant, seed), name] - raw_metric[name])
            comparisons.append(delta)
            raw_prediction = pd.read_csv(EXP051_ROOT / f"predictions/set03_{raw_variant}_seed{seed}.csv")
            new_prediction = predictions[(new_variant, seed)]
            if not raw_prediction["sequence_id"].equals(new_prediction["sequence_id"]) or not raw_prediction["label"].equals(new_prediction["label"]):
                raise RuntimeError("Reused raw-control cohort does not match Exp053.")
            raw_correct = raw_prediction["prediction"].to_numpy() == raw_prediction["label"].to_numpy()
            new_correct = new_prediction["prediction"].to_numpy() == new_prediction["label"].to_numpy()
            transitions.append({"comparison": f"{raw_variant}_to_{new_variant}", "condition": _condition(new_variant), "seed": seed, "raw_wrong_bbox53_correct": int((~raw_correct & new_correct).sum()), "raw_correct_bbox53_wrong": int((raw_correct & ~new_correct).sum()), "both_wrong": int((~raw_correct & ~new_correct).sum()), "both_correct": int((raw_correct & new_correct).sum())})

    all_results = pd.concat([pd.DataFrame(raw_records), pd.DataFrame(new_records)], ignore_index=True)
    summary = all_results.groupby(["condition", "representation"], sort=False).agg(
        **{f"{metric}_{stat}": (metric, stat) for metric in (*metric_names, "total_parameters", "training_seconds") for stat in ("mean", "std")}
    ).reset_index()
    reference = pd.read_csv(EXP012_ROOT / "metrics/pie_biu_v2_benchmark_results.csv")
    reference = reference.loc[
        reference["threshold_rule"].eq("fixed_0_5")
        & (
            (reference["feature_set"].eq("bbox_only") & reference["model_name"].eq("rf"))
            | (reference["feature_set"].eq("biu_v1_plus_bbox") & reference["model_name"].eq("rf"))
            | (reference["feature_set"].eq("biu_v2_core_plus_bbox") & reference["model_name"].eq("xgb"))
        ),
        ["feature_set", "model_name", "seed", "test_f1", "test_roc_auc", "test_pr_auc"],
    ].reset_index(drop=True)
    output_tables = {
        "new_per_seed_results.csv": pd.DataFrame(new_records),
        "all_controlled_results.csv": all_results,
        "summary_results.csv": summary,
        "representation_paired_deltas.csv": pd.DataFrame(comparisons),
        "error_transitions_fixed_0p5.csv": pd.DataFrame(transitions),
        "efficiency.csv": pd.DataFrame(efficiency),
        "exp012_references.csv": reference,
    }
    for name, frame in output_tables.items():
        base.atomic_csv(frame, ROOT / f"metrics/{name}")
    report_path = ROOT / "FINAL_REPORT.md"
    report_path.write_text(
        "# Exp053 Final Report\n\nRaw versus 53-D BBox replacement with core, BIU-v1 and BIU-v2 conditions.\n\n"
        + base.frame_to_markdown(summary)
        + "\n\nRaw controls are authenticated Exp051 C/F/E artifacts; all 53-D variants are fresh Exp053 runs.\n",
        encoding="utf-8",
    )
    artifacts = {f"metrics/{name}": core.sha256_file(ROOT / f"metrics/{name}") for name in output_tables}
    artifacts["FINAL_REPORT.md"] = core.sha256_file(report_path)
    artifacts["cache/bbox53_test.npy"] = core.sha256_file(_bbox53_cache_path("test"))
    for variant, seed in required_training_identities():
        relative = f"predictions/set03_{variant}_seed{seed}.csv"
        artifacts[relative] = core.sha256_file(ROOT / relative)
    base.atomic_json({"phase": "evaluate", "created_at": base.utc_now(), "test_rows": len(test_rows), "test_sequence_id_sha256": test_metadata["sequence_id_sha256"], "artifacts": artifacts}, base.EVALUATION_MARKER)
    base.log_line(ROOT / "logs/evaluate.log", "phase=evaluate status=complete")


def run_verify() -> None:
    assert_training_complete(ROOT)
    if not base.EVALUATION_MARKER.is_file():
        raise RuntimeError("Evaluation marker is missing.")
    evaluation = json.loads(base.EVALUATION_MARKER.read_text(encoding="utf-8"))
    base.authenticate_artifacts(ROOT, evaluation["artifacts"])
    base.COMPLETE_MARKER.write_text(json.dumps({"experiment": 53, "completed_at": base.utc_now(), "training_runs": 9, "reused_raw_controls": 9, "test_rows": int(evaluation["test_rows"]), "evaluation_marker_sha256": core.sha256_file(base.EVALUATION_MARKER)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Exp053 verification passed; EXPERIMENT_COMPLETE written.")


def gpu_smoke() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in Exp053 execution environment.")
    model = core.MslstmBbox53Model("I").to("cuda")
    inputs = {name: torch.zeros((2, core.STREAM_DIMS[name]), device="cuda") if name in core.WINDOW_VECTOR_STREAMS else torch.zeros((2, core.SEQ_LEN, core.STREAM_DIMS[name]), device="cuda") for name in core.VARIANT_STREAMS["I"]}
    loss = model(inputs).sum()
    loss.backward()
    print(json.dumps({"device": "cuda", "gpu": torch.cuda.get_device_name(0), "loss": float(loss.item())}))


def status() -> None:
    completed = [f"{variant}{seed}" for variant, seed in required_training_identities() if training_marker(ROOT, variant, seed).is_file()]
    print(json.dumps({"prepare_complete": base.PREPARE_MARKER.is_file(), "completed_training_runs": completed, "completed_training_count": len(completed), "evaluation_complete": base.EVALUATION_MARKER.is_file(), "experiment_complete": base.COMPLETE_MARKER.is_file()}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare")
    train = subparsers.add_parser("train")
    train.add_argument("--variant", choices=core.VARIANTS, required=True)
    train.add_argument("--seed", type=int, choices=core.SEEDS, required=True)
    subparsers.add_parser("train-all")
    subparsers.add_parser("evaluate")
    subparsers.add_parser("verify")
    subparsers.add_parser("gpu-smoke")
    subparsers.add_parser("status")
    args = parser.parse_args()
    if args.command == "prepare":
        run_prepare()
    elif args.command == "train":
        base.run_training(args.variant, args.seed)
    elif args.command == "train-all":
        run_train_all()
    elif args.command == "evaluate":
        run_evaluate()
    elif args.command == "verify":
        run_verify()
    elif args.command == "gpu-smoke":
        gpu_smoke()
    elif args.command == "status":
        status()


if __name__ == "__main__":
    main()
