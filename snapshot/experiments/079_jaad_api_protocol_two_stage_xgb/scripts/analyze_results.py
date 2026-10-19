"""Saved-prediction-only diagnostics and clean-protocol comparisons for Exp079."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_ROOT = Path(__file__).resolve().parents[3]
EXP031 = ACTIVE_ROOT / "experiments/031_jaad_api_label_protocol_sensitivity"
EXP077 = ACTIVE_ROOT / "experiments/077_jaad_two_stage_early_late_xgb_transfer"
EXP076_SCRIPTS = ACTIVE_ROOT / "experiments/076_pie_two_stage_early_late_xgb/scripts"
if str(EXP076_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(EXP076_SCRIPTS))

import evaluation  # noqa: E402
import two_stage_core  # noqa: E402


def _summarize_metric_rows(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "fpr", "fnr", "brier",
        "log_loss", "fp_ge_0p9_count", "fn_le_0p1_count",
    ]
    rows = []
    for model, group in frame.groupby("model", sort=False):
        row = {"model": model, "seed_count": int(group.seed.nunique())}
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def bbox_fixed_0p5() -> pd.DataFrame:
    predictions = pd.read_csv(EXP031 / "metrics/selected_model_predictions.csv")
    predictions = predictions.loc[
        predictions["protocol"].eq("api_style_reconstruction")
        & predictions["prediction_split"].eq("test")
        & predictions["model_name"].eq("xgb")
    ]
    rows = []
    for seed, group in predictions.groupby("seed", sort=True):
        rows.append(
            {
                "scope": "jaad_api_protocol_native_test",
                "model": "bbox53_xgb_exp031",
                "seed": int(seed),
                **evaluation.compute_metrics(group.label.to_numpy(int), group.probability.to_numpy(float)),
            }
        )
    return pd.DataFrame(rows)


def exact_overlap_feature_parity() -> list[dict[str, int | float | str]]:
    definitions = ACTIVE_ROOT / "experiments/012_pie_biu_v2_30_60_benchmark/metrics/pie_biu_v2_feature_set_definitions.csv"
    contract = two_stage_core.load_joint_contract(definitions)
    columns = [f"early__{name}" for name in contract["joint"]] + list(contract["portable"])
    overlap = pd.read_csv(EXP031 / "metrics/exact_overlap_window_comparison.csv")
    rows = []
    for split, new_name, old_name in (
        ("train", "train", "train"), ("val", "validation", "validation"), ("test", "test", "test")
    ):
        mapping = overlap.loc[overlap.split.eq(split), ["sequence_id_api", "sequence_id_clean"]]
        new = pd.read_csv(ROOT / f"metrics/prepared_jaad_{new_name}.csv", usecols=["sequence_id", *columns]).rename(
            columns={"sequence_id": "sequence_id_api"}
        )
        old = pd.read_csv(EXP077 / f"metrics/prepared_jaad_{old_name}.csv", usecols=["sequence_id", *columns]).rename(
            columns={"sequence_id": "sequence_id_clean"}
        )
        merged = mapping.merge(new, on="sequence_id_api", validate="one_to_one").merge(
            old, on="sequence_id_clean", validate="one_to_one", suffixes=("_api", "_clean")
        )
        differences = np.concatenate(
            [
                np.abs(merged[f"{column}_api"].to_numpy(float) - merged[f"{column}_clean"].to_numpy(float))
                for column in columns
            ]
        )
        rows.append(
            {
                "split": split,
                "overlap_rows": len(merged),
                "feature_values_compared": int(differences.size),
                "max_abs_difference": float(differences.max()),
                "mean_abs_difference": float(differences.mean()),
            }
        )
    return rows


def zero_shot_diagnostics() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    cohort = pd.read_csv(
        ROOT / "metrics/prepared_jaad_test.csv",
        usecols=["sequence_id", "ped_id", "label", "api_flaw_affected", "temporal_position"],
    )
    predictions = pd.read_csv(ROOT / "predictions/pie_zero_shot_test_all_seeds.csv").merge(
        cohort[["sequence_id", "ped_id"]], on="sequence_id", validate="many_to_one"
    )
    rows = []
    masks = {
        "all": pd.Series(True, index=predictions.index),
        "behavior_b": predictions.ped_id.str.endswith("b"),
        "non_behavior": ~predictions.ped_id.str.endswith("b"),
    }
    for group_name, mask in masks.items():
        for model, column in (("one_stage", "one_stage_probability"), ("two_stage", "two_stage_probability")):
            for seed, group in predictions.loc[mask].groupby("seed", sort=True):
                labels = group.label.to_numpy(int)
                probability = group[column].to_numpy(float)
                prediction = (probability >= 0.5).astype(int)
                rows.append(
                    {
                        "group": group_name,
                        "model": model,
                        "seed": int(seed),
                        "n": len(group),
                        "positives": int(labels.sum()),
                        "positive_rate": float(labels.mean()),
                        "mean_probability": float(probability.mean()),
                        "fp": int(((labels == 0) & (prediction == 1)).sum()),
                        "fn": int(((labels == 1) & (prediction == 0)).sum()),
                        "f1": float(f1_score(labels, prediction, zero_division=0)),
                        "roc_auc": float(roc_auc_score(labels, probability)) if len(np.unique(labels)) == 2 else np.nan,
                        "pr_auc": float(average_precision_score(labels, probability)) if labels.sum() else np.nan,
                    }
                )
    group_metrics = pd.DataFrame(rows)

    overlap = pd.read_csv(EXP031 / "metrics/exact_overlap_window_comparison.csv")
    overlap = overlap.loc[overlap.split.eq("test"), ["sequence_id_api", "sequence_id_clean"]]
    new = pd.read_csv(ROOT / "predictions/pie_zero_shot_test_all_seeds.csv").rename(
        columns={
            "sequence_id": "sequence_id_api",
            "label": "label_api",
            "one_stage_probability": "one_stage_probability_api",
            "two_stage_probability": "two_stage_probability_api",
        }
    )
    old = pd.read_csv(EXP077 / "predictions/pie_zero_shot_test_all_seeds.csv").rename(
        columns={
            "sequence_id": "sequence_id_clean",
            "label": "label_clean",
            "one_stage_probability": "one_stage_probability_clean",
            "two_stage_probability": "two_stage_probability_clean",
        }
    )
    merged = overlap.merge(new, on="sequence_id_api", validate="one_to_many").merge(
        old, on=["sequence_id_clean", "seed"], validate="one_to_one"
    )
    parity = {
        "overlap_windows": len(overlap),
        "seed_window_rows": len(merged),
        "labels_equal": bool((merged.label_api == merged.label_clean).all()),
        "one_stage_probability_max_abs_difference": float(
            np.max(np.abs(merged.one_stage_probability_api - merged.one_stage_probability_clean))
        ),
        "two_stage_probability_max_abs_difference": float(
            np.max(np.abs(merged.two_stage_probability_api - merged.two_stage_probability_clean))
        ),
    }
    population = (
        cohort.assign(track_type=np.where(cohort.ped_id.str.endswith("b"), "behavior_b", "non_behavior"))
        .groupby("track_type", as_index=False)
        .agg(
            windows=("sequence_id", "size"),
            positives=("label", "sum"),
            unique_pedestrians=("ped_id", "nunique"),
            api_flaw_affected=("api_flaw_affected", "sum"),
        )
    )
    return group_metrics, parity, population


def protocol_comparison(bbox: pd.DataFrame) -> pd.DataFrame:
    current = pd.read_csv(ROOT / "metrics/test_per_seed_results.csv")
    current = current.loc[current.scope.eq("jaad_api_protocol_native_test")].copy()
    clean = pd.read_csv(EXP077 / "metrics/test_per_seed_results.csv")
    clean = clean.loc[clean.scope.eq("jaad_native_test")].copy()
    columns = ["seed", "model", "f1", "pr_auc", "roc_auc", "accuracy", "fp_ge_0p9_count", "fn_le_0p1_count"]
    current = current[columns].assign(protocol="flawed_api", source="Exp079")
    clean = clean[columns].assign(protocol="clean_event_aligned", source="Exp077")
    bbox_rows = bbox[columns].assign(protocol="flawed_api", source="Exp031")
    return pd.concat([bbox_rows, current, clean], ignore_index=True)


def main() -> None:
    bbox = bbox_fixed_0p5()
    bbox.to_csv(ROOT / "metrics/flawed_protocol_bbox_fixed_0p5.csv", index=False)
    _summarize_metric_rows(bbox).to_csv(ROOT / "metrics/flawed_protocol_bbox_fixed_0p5_summary.csv", index=False)
    comparison = protocol_comparison(bbox)
    comparison.to_csv(ROOT / "metrics/protocol_comparison_per_seed.csv", index=False)
    comparison_summary = comparison.groupby(
        ["protocol", "source", "model"], as_index=False
    ).mean(numeric_only=True)
    comparison_summary.to_csv(ROOT / "metrics/protocol_comparison_summary.csv", index=False)
    bbox_mean = comparison_summary.loc[
        comparison_summary.model.eq("bbox53_xgb_exp031")
    ].iloc[0]
    one_mean = comparison_summary.loc[
        comparison_summary.protocol.eq("flawed_api") & comparison_summary.model.eq("one_stage")
    ].iloc[0]
    two_mean = comparison_summary.loc[
        comparison_summary.protocol.eq("flawed_api") & comparison_summary.model.eq("two_stage")
    ].iloc[0]
    delta_metrics = ("f1", "pr_auc", "roc_auc", "accuracy", "fp_ge_0p9_count", "fn_le_0p1_count")
    delta_rows = []
    for name, left, right in (
        ("one_stage_minus_bbox53", one_mean, bbox_mean),
        ("two_stage_minus_bbox53", two_mean, bbox_mean),
        ("two_stage_minus_one_stage", two_mean, one_mean),
    ):
        delta_rows.append({"comparison": name, **{f"delta_{metric}": float(left[metric] - right[metric]) for metric in delta_metrics}})
    pd.DataFrame(delta_rows).to_csv(ROOT / "metrics/primary_deltas.csv", index=False)
    parity = exact_overlap_feature_parity()
    (ROOT / "metrics/exact_overlap_feature_parity.json").write_text(json.dumps(parity, indent=2) + "\n")
    group_metrics, prediction_parity, population = zero_shot_diagnostics()
    group_metrics.to_csv(ROOT / "metrics/zero_shot_population_diagnostic.csv", index=False)
    population.to_csv(ROOT / "metrics/test_population_by_track_type.csv", index=False)
    (ROOT / "metrics/zero_shot_overlap_prediction_parity.json").write_text(
        json.dumps(prediction_parity, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"feature_parity": parity, "prediction_parity": prediction_parity}, indent=2))


if __name__ == "__main__":
    main()
