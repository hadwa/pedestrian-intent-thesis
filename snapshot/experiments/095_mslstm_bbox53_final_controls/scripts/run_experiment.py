#!/usr/bin/env python3
"""Resumable runner for the final matched MSLSTM BBox53 controls."""

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
EXPERIMENTS_ROOT = ROOT.parent
EXP050_ROOT = EXPERIMENTS_ROOT / "050_mslstm_practical_behavior_replacement_pie_16f_tte30_60"
EXP053_ROOT = EXPERIMENTS_ROOT / "053_mslstm_bbox53_replacement"
EXP050_RUNNER_PATH = EXP050_ROOT / "scripts/run_experiment.py"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import exp095_core as core  # noqa: E402


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


base = _load_module("_exp050_runner_for_exp095", EXP050_RUNNER_PATH)
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
_source_load_cache = base.load_cache

COMPARISONS = (
    ("G", "J"),
    ("G", "K"),
    ("G", "L"),
    ("J", "H"),
    ("J", "I"),
    ("J", "L"),
    ("K", "H"),
    ("K", "I"),
    ("K", "L"),
    ("H", "L"),
    ("I", "L"),
)
REFERENCE_VARIANTS = ("G", "H", "I")
METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")


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


def _authenticate_source_preparation() -> None:
    source050 = json.loads((EXP050_ROOT / ".prepare.complete.json").read_text(encoding="utf-8"))
    source053 = json.loads((EXP053_ROOT / ".prepare.complete.json").read_text(encoding="utf-8"))
    if source050.get("test_accessed") is not False or source053.get("test_accessed") is not False:
        raise RuntimeError("Source preparation does not prove set03 exclusion.")
    base.authenticate_artifacts(EXP050_ROOT, source050["artifacts"])
    base.authenticate_artifacts(EXP053_ROOT, source053["artifacts"])


def _add_control_arrays(arrays: dict[str, np.ndarray], bbox53: np.ndarray) -> dict[str, np.ndarray]:
    output = dict(arrays)
    output["bbox53"] = bbox53
    output["biu_combined"] = core.combined_biu(output["biu_v1"], output["biu_v2"])
    row_count = len(output["bbox53"])
    if output["bbox53"].shape != (row_count, 53) or output["biu_combined"].shape != (row_count, 135):
        raise RuntimeError("Derived Exp095 arrays have invalid shapes.")
    return output


def load_cache(path: Path):
    candidate = Path(path)
    if candidate == BASE_CACHE:
        rows, arrays, metadata = _source_load_cache(BASE_CACHE)
        bbox53 = np.load(EXP053_ROOT / "cache/bbox53_trainval.npy", mmap_mode="r")
        return rows, _add_control_arrays(arrays, bbox53), metadata
    if candidate == TEST_CACHE:
        rows, arrays, metadata = _source_load_cache(TEST_CACHE)
        bbox53 = np.load(EXP053_ROOT / "cache/bbox53_test.npy", mmap_mode="r")
        return rows, _add_control_arrays(arrays, bbox53), metadata
    try:
        relative = candidate.relative_to(ROOT / "cache")
    except ValueError:
        return _source_load_cache(candidate)
    if not relative.name.startswith("augmented_seed"):
        raise ValueError(f"Unknown Exp095 cache path: {candidate}")
    seed = int(relative.name.replace("augmented_seed", ""))
    source = EXP050_ROOT / "cache" / relative
    rows, arrays, metadata = _source_load_cache(source)
    bbox53 = np.load(EXP053_ROOT / f"cache/bbox53_augmented_seed{seed}.npy", mmap_mode="r")
    return rows, _add_control_arrays(arrays, bbox53), metadata


base.load_cache = load_cache


def run_prepare() -> None:
    if base.PREPARE_MARKER.is_file():
        payload = json.loads(base.PREPARE_MARKER.read_text(encoding="utf-8"))
        base.authenticate_artifacts(ROOT, payload["artifacts"])
        _authenticate_source_preparation()
        base.log_line(ROOT / "logs/prepare.log", "status=already_complete_authenticated")
        return
    _authenticate_source_preparation()
    rows, arrays, metadata = load_cache(BASE_CACHE)
    if rows["split"].value_counts().astype(int).to_dict() != {
        "val": core.EXPECTED_SPLIT_COUNTS["val"],
        "train": core.EXPECTED_SPLIT_COUNTS["train"],
    }:
        raise RuntimeError("Frozen train/validation cohort drifted.")
    audit = {
        "experiment": 95,
        "created_at": base.utc_now(),
        "protocol": "PIE 16 observed frames, TTE30--60, overlap 0.6, fixed threshold 0.5",
        "test_accessed": False,
        "train_rows": core.EXPECTED_SPLIT_COUNTS["train"],
        "validation_rows": core.EXPECTED_SPLIT_COUNTS["val"],
        "source_sequence_id_sha256": metadata["sequence_id_sha256"],
        "bbox53_shape": list(arrays["bbox53"].shape),
        "raw_pose_shape": list(arrays["pose"].shape),
        "action_shape": list(arrays["ped_action"].shape),
        "look_shape": list(arrays["ped_look"].shape),
        "biu_combined_shape": list(arrays["biu_combined"].shape),
        "biu_combined_feature_names": list(core.BIU_COMBINED_FEATURE_NAMES),
        "source_prepare_sha256": {
            "exp050": core.sha256_file(EXP050_ROOT / ".prepare.complete.json"),
            "exp053": core.sha256_file(EXP053_ROOT / ".prepare.complete.json"),
        },
    }
    for seed in core.SEEDS:
        augmented_rows, augmented, _ = load_cache(ROOT / f"cache/augmented_seed{seed}")
        if len(augmented_rows) != len(augmented["bbox53"]) or augmented["biu_combined"].shape[1] != 135:
            raise RuntimeError(f"Augmented source drifted for seed {seed}.")
    audit_path = ROOT / "metrics/source_audit.json"
    base.atomic_json(audit, audit_path)
    artifacts = {"metrics/source_audit.json": core.sha256_file(audit_path)}
    base.atomic_json(
        {
            "phase": "prepare",
            "created_at": base.utc_now(),
            "train_rows": core.EXPECTED_SPLIT_COUNTS["train"],
            "val_rows": core.EXPECTED_SPLIT_COUNTS["val"],
            "test_accessed": False,
            "artifacts": artifacts,
        },
        base.PREPARE_MARKER,
    )
    base.log_line(ROOT / "logs/prepare.log", "phase=prepare status=complete test_accessed=false")


def run_train_all() -> None:
    for variant, seed in required_training_identities():
        base.run_training(variant, seed)
    assert_training_complete(ROOT)
    base.atomic_json(
        {"phase": "train_all", "created_at": base.utc_now(), "run_count": 9},
        ROOT / "TRAINING_COMPLETE",
    )


def _reference_results_and_predictions():
    source = pd.read_csv(EXP053_ROOT / "metrics/new_per_seed_results.csv")
    source = source.loc[source["variant"].isin(REFERENCE_VARIANTS)].copy()
    if len(source) != 9:
        raise RuntimeError("Exp053 must provide all nine G/H/I reference results.")
    source["streams"] = source["variant"].map(
        {
            "G": "bbox53+ego_speed+ego_acc+yolop",
            "H": "bbox53+ego_speed+ego_acc+yolop+biu_v1",
            "I": "bbox53+ego_speed+ego_acc+yolop+biu_v2",
        }
    )
    predictions = {
        (variant, seed): pd.read_csv(EXP053_ROOT / f"predictions/set03_{variant}_seed{seed}.csv")
        for variant in REFERENCE_VARIANTS
        for seed in core.SEEDS
    }
    return source, predictions


def _paired_tables(results: pd.DataFrame, predictions: dict[tuple[str, int], pd.DataFrame]):
    indexed = results.set_index(["variant", "seed"])
    deltas = []
    transitions = []
    for left, right in COMPARISONS:
        for seed in core.SEEDS:
            record = {"comparison": f"{left}_to_{right}", "seed": seed}
            for metric in METRICS:
                record[f"{metric}_delta"] = float(
                    indexed.loc[(right, seed), metric] - indexed.loc[(left, seed), metric]
                )
            deltas.append(record)
            a = predictions[(left, seed)]
            b = predictions[(right, seed)]
            if not a["sequence_id"].equals(b["sequence_id"]) or not a["label"].equals(b["label"]):
                raise RuntimeError("Compared predictions do not share an identical cohort.")
            a_correct = a["prediction"].to_numpy() == a["label"].to_numpy()
            b_correct = b["prediction"].to_numpy() == b["label"].to_numpy()
            transitions.append(
                {
                    "comparison": f"{left}_to_{right}",
                    "seed": seed,
                    "left_wrong_right_correct": int((~a_correct & b_correct).sum()),
                    "left_correct_right_wrong": int((a_correct & ~b_correct).sum()),
                    "both_wrong": int((~a_correct & ~b_correct).sum()),
                    "both_correct": int((a_correct & b_correct).sum()),
                }
            )
    return pd.DataFrame(deltas), pd.DataFrame(transitions)


def run_evaluate() -> None:
    if base.EVALUATION_MARKER.is_file():
        payload = json.loads(base.EVALUATION_MARKER.read_text(encoding="utf-8"))
        base.authenticate_artifacts(ROOT, payload["artifacts"])
        base.log_line(ROOT / "logs/evaluate.log", "status=already_complete_authenticated")
        return
    training = assert_training_complete(ROOT)
    source_evaluation = json.loads((EXP053_ROOT / ".evaluation.complete.json").read_text(encoding="utf-8"))
    base.authenticate_artifacts(EXP053_ROOT, source_evaluation["artifacts"])
    if not torch.cuda.is_available():
        raise RuntimeError("Exp095 final inference is GPU-only; CUDA is unavailable.")
    test_rows, test_arrays, test_metadata = load_cache(TEST_CACHE)
    if set(test_rows["split"].astype(str)) != {"test"} or set(test_rows["set_id"].astype(str)) != {"set03"}:
        raise RuntimeError("Final cache is not designated set03.")
    labels = test_rows["label"].to_numpy(dtype=np.int64)
    device = torch.device("cuda")
    new_records = []
    new_predictions: dict[tuple[str, int], pd.DataFrame] = {}
    efficiency = []
    for variant, seed in required_training_identities():
        checkpoint = torch.load(
            ROOT / f"checkpoints/{variant}_seed{seed}.best.pt",
            map_location="cpu",
            weights_only=False,
        )
        model = core.MslstmBbox53ControlModel(variant).to(device)
        model.load_state_dict(checkpoint["model_state"])
        streams = core.VARIANT_STREAMS[variant]
        tensors = {name: base.tensor_for_stream(test_arrays[name], name, device) for name in streams}
        torch.cuda.synchronize()
        started = time.perf_counter()
        probability = base.probabilities(model, tensors, len(test_rows))
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        metrics = core.compute_metrics(labels, probability)
        train_record = next(
            item for item in training if item["variant"] == variant and int(item["seed"]) == seed
        )
        new_records.append(
            {
                "condition": {"J": "raw_pose", "K": "gt_action_look", "L": "biu_combined"}[variant],
                "representation": "bbox53",
                "variant": variant,
                "seed": seed,
                "streams": "+".join(streams),
                "best_epoch": int(checkpoint["best_epoch"]),
                "best_validation_f1": float(checkpoint["best_validation_f1"]),
                "total_parameters": int(checkpoint["total_parameters"]),
                "training_seconds": float(train_record["training_seconds"]),
                **metrics,
            }
        )
        frame = test_rows[
            [
                "sequence_id", "split", "set_id", "video_id", "ped_id", "start_frame",
                "end_frame", "event_frame", "time_to_event_frames", "label",
            ]
        ].copy()
        frame["variant"] = variant
        frame["seed"] = seed
        frame["threshold"] = core.FIXED_THRESHOLD
        frame["probability"] = probability
        frame["prediction"] = (probability >= core.FIXED_THRESHOLD).astype(np.int64)
        prediction_path = ROOT / f"predictions/set03_{variant}_seed{seed}.csv"
        base.atomic_csv(frame, prediction_path)
        new_predictions[(variant, seed)] = frame
        efficiency.append(
            {
                "variant": variant,
                "seed": seed,
                "device": str(device),
                "total_parameters": int(checkpoint["total_parameters"]),
                "inference_seconds": elapsed,
                "seconds_per_sample": elapsed / len(test_rows),
                "pose_estimator_runtime_included": False,
            }
        )
        base.log_line(
            ROOT / "logs/evaluate.log",
            f"variant={variant} seed={seed} f1={metrics['f1']:.6f} pr_auc={metrics['pr_auc']:.6f}",
        )
        del model, tensors
        torch.cuda.empty_cache()

    references, reference_predictions = _reference_results_and_predictions()
    all_results = pd.concat((references, pd.DataFrame(new_records)), ignore_index=True, sort=False)
    summary = all_results.groupby(["variant", "condition"], sort=False).agg(
        **{
            f"{metric}_{stat}": (metric, stat)
            for metric in (*METRICS, "total_parameters", "training_seconds")
            for stat in ("mean", "std")
        }
    ).reset_index()
    predictions = {**reference_predictions, **new_predictions}
    deltas, transitions = _paired_tables(all_results, predictions)
    output_tables = {
        "new_per_seed_results.csv": pd.DataFrame(new_records),
        "all_controlled_results.csv": all_results,
        "summary_results.csv": summary,
        "paired_deltas.csv": deltas,
        "error_transitions_fixed_0p5.csv": transitions,
        "efficiency.csv": pd.DataFrame(efficiency),
    }
    for name, frame in output_tables.items():
        base.atomic_csv(frame, ROOT / f"metrics/{name}")
    report_lines = [
        "# Exp095 Final Report",
        "",
        "Matched MSLSTM BBox53 controls on PIE: 16 observed frames, TTE30--60, fixed threshold 0.5.",
        "",
        base.frame_to_markdown(summary),
        "",
        "G/H/I are authenticated Exp053 references. J/K/L are fresh matched controls.",
        "This is a retrospective control study on the previously exposed set03 cohort, not pristine confirmation.",
        "Comparisons concern predictive utility under this architecture; they do not establish semantic equivalence or universal replacement.",
    ]
    report_path = ROOT / "FINAL_REPORT.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    artifacts = {f"metrics/{name}": core.sha256_file(ROOT / f"metrics/{name}") for name in output_tables}
    artifacts["FINAL_REPORT.md"] = core.sha256_file(report_path)
    for variant, seed in required_training_identities():
        relative = f"predictions/set03_{variant}_seed{seed}.csv"
        artifacts[relative] = core.sha256_file(ROOT / relative)
    base.atomic_json(
        {
            "phase": "evaluate",
            "created_at": base.utc_now(),
            "test_rows": len(test_rows),
            "test_sequence_id_sha256": test_metadata["sequence_id_sha256"],
            "source_exp053_evaluation_sha256": core.sha256_file(EXP053_ROOT / ".evaluation.complete.json"),
            "artifacts": artifacts,
        },
        base.EVALUATION_MARKER,
    )
    base.log_line(ROOT / "logs/evaluate.log", "phase=evaluate status=complete")


def run_verify() -> None:
    assert_training_complete(ROOT)
    if not base.EVALUATION_MARKER.is_file():
        raise RuntimeError("Evaluation marker is missing.")
    evaluation = json.loads(base.EVALUATION_MARKER.read_text(encoding="utf-8"))
    base.authenticate_artifacts(ROOT, evaluation["artifacts"])
    base.atomic_json(
        {
            "experiment": 95,
            "completed_at": base.utc_now(),
            "new_training_runs": 9,
            "reused_reference_runs": 9,
            "test_rows": int(evaluation["test_rows"]),
            "evaluation_marker_sha256": core.sha256_file(base.EVALUATION_MARKER),
        },
        base.COMPLETE_MARKER,
    )
    print("Exp095 verification passed; EXPERIMENT_COMPLETE written.")


def gpu_smoke() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in Exp095 execution environment.")
    device = torch.device("cuda")
    output = {}
    for variant in core.VARIANTS:
        model = core.MslstmBbox53ControlModel(variant).to(device)
        inputs = {
            name: torch.zeros((2, core.STREAM_DIMS[name]), device=device)
            if name in core.WINDOW_VECTOR_STREAMS
            else torch.zeros((2, core.SEQ_LEN, core.STREAM_DIMS[name]), device=device)
            for name in core.VARIANT_STREAMS[variant]
        }
        loss = model(inputs).sum()
        loss.backward()
        output[variant] = {"loss": float(loss.item()), "parameters": core.count_parameters(model)}
    print(json.dumps({"device": str(device), "gpu": torch.cuda.get_device_name(0), "variants": output}))


def status() -> None:
    completed = [
        f"{variant}{seed}"
        for variant, seed in required_training_identities()
        if training_marker(ROOT, variant, seed).is_file()
    ]
    active_states = sorted(path.name for path in (ROOT / "checkpoints").glob("*.state.pt"))
    print(
        json.dumps(
            {
                "prepare_complete": base.PREPARE_MARKER.is_file(),
                "completed_training_runs": completed,
                "completed_training_count": len(completed),
                "active_states": active_states,
                "evaluation_complete": base.EVALUATION_MARKER.is_file(),
                "experiment_complete": base.COMPLETE_MARKER.is_file(),
            },
            indent=2,
        )
    )


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
