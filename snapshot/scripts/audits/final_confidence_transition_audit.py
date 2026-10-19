from __future__ import annotations

from pathlib import Path
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.benchmark.final_confidence_transition_audit import (
    assemble_transition_rows,
    build_primary_scalar_frame,
    compute_error_accounting,
    compute_transition_counts,
    compute_transition_distribution_summary,
    load_selected_prediction_payload,
    rebuild_fixed_threshold_predictions,
    select_seed_prediction_row,
)


CLEAN_ROOT = ROOT
METRICS_DIR = CLEAN_ROOT / "experiments/030_final_confidence_benchmark/metrics"
OUTPUT_ROOT = CLEAN_ROOT / "experiments/030_final_confidence_benchmark/transition_audit"
SAMPLE_ROOT = OUTPUT_ROOT / "sample_level"
PRIMARY_CONFIGS = [("pie", "30_60"), ("jaad", "30_60"), ("jaad", "5_15")]
SECONDARY_CONFIGS = [("jaad", "15_30")]
PRIMARY_COMPARISONS = [
    "KP-XY -> KP-XYC",
    "BIU-v2 -> BIU-v2 + BIU-specific reliability",
    "BBox + BIU-v2 -> BBox + BIU-v2 + reliability",
]
REQUIRED_SEEDS = (42, 43, 44)
COMPARISON_TO_REPRESENTATIONS = {
    "KP-XY -> KP-XYC": ("kp_xy", "kp_xyc"),
    "BIU-v2 -> BIU-v2 + BIU-specific reliability": (
        "biu_v2",
        "biu_v2_plus_reliability",
    ),
    "BBox + BIU-v2 -> BBox + BIU-v2 + reliability": (
        "bbox_plus_biu_v2",
        "bbox_plus_biu_v2_plus_reliability",
    ),
}
FIXED_0P5_ENABLED = True


def prediction_path(dataset: str, horizon: str, representation: str) -> Path:
    return METRICS_DIR / f"{dataset}_{horizon}_{representation}_selected_predictions.json"


def comparison_slug(comparison: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", comparison.lower()).strip("_")


def scope_label(dataset: str, horizon: str) -> str:
    return "primary" if (dataset, horizon) in PRIMARY_CONFIGS else "secondary"


def write_sample_level_csv(rows: pd.DataFrame) -> Path:
    SAMPLE_ROOT.mkdir(parents=True, exist_ok=True)
    first = rows.iloc[0]
    path = SAMPLE_ROOT / (
        f"{first['dataset']}_{first['horizon']}_{comparison_slug(first['comparison'])}_"
        f"{first['threshold_regime']}_seed{int(first['seed'])}.csv"
    )
    rows.to_csv(path, index=False)
    return path


def attach_metadata(frame: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    first = rows.iloc[0]
    enriched = frame.copy()
    enriched.insert(0, "scope", first["scope"])
    enriched.insert(1, "dataset", first["dataset"])
    enriched.insert(2, "horizon", first["horizon"])
    enriched.insert(3, "comparison", first["comparison"])
    enriched.insert(4, "threshold_regime", first["threshold_regime"])
    enriched.insert(5, "seed", int(first["seed"]))
    return enriched


def summarize_rows(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    count_summary = attach_metadata(compute_transition_counts(rows), rows)
    distribution_summary = attach_metadata(
        compute_transition_distribution_summary(rows), rows
    )
    error_summary = pd.DataFrame([compute_error_accounting(rows)])
    error_summary = attach_metadata(error_summary, rows)
    return count_summary, distribution_summary, error_summary


def summarize_error_rates(error_summary: pd.DataFrame) -> pd.DataFrame:
    rate_columns = [
        "correction_rate",
        "error_introduction_rate",
        "net_error_change",
    ]
    grouped = (
        error_summary.groupby(
            ["scope", "dataset", "horizon", "comparison", "threshold_regime"],
            dropna=False,
        )
        .agg(
            n_seeds=("seed", "nunique"),
            correction_rate_mean=("correction_rate", "mean"),
            correction_rate_std=("correction_rate", "std"),
            error_introduction_rate_mean=("error_introduction_rate", "mean"),
            error_introduction_rate_std=("error_introduction_rate", "std"),
            net_error_change_mean=("net_error_change", "mean"),
            net_error_change_std=("net_error_change", "std"),
            baseline_errors_mean=("baseline_errors", "mean"),
            corrected_errors_mean=("corrected_errors", "mean"),
            new_errors_mean=("new_errors", "mean"),
        )
        .reset_index()
    )
    grouped["seed_list"] = ",".join(str(seed) for seed in REQUIRED_SEEDS)
    return grouped


def write_markdown_report(
    count_summary: pd.DataFrame,
    distribution_summary: pd.DataFrame,
    error_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Final Confidence Transition Audit",
        "",
        "Read-only post-hoc audit over saved Experiment 030 prediction artifacts.",
        "",
        "Primary scope:",
        "- PIE `30_60`",
        "- JAAD `30_60`",
        "- JAAD `5_15`",
        "",
        "Secondary scope:",
        "- JAAD `15_30`",
        "",
        "Outputs:",
        "- `sample_level/*.csv` per dataset/horizon/comparison/threshold regime/seed",
        "- `transition_count_summary.csv`",
        "- `transition_distribution_summary.csv`",
        "- `error_accounting_summary.csv`",
        "",
        "Primary validation-selected error accounting:",
        "",
    ]
    primary_error = error_summary[error_summary["threshold_regime"] == "val_f1"].copy()
    for _, row in primary_error.iterrows():
        lines.append(
            f"- {row['scope'].upper()} {row['dataset'].upper()} {row['horizon']} {row['comparison']} "
            f"(seed {int(row['seed'])}): baseline_errors={int(row['baseline_errors'])}, "
            f"corrected={int(row['corrected_errors'])}, new_errors={int(row['new_errors'])}, "
            f"net_corrected={int(row['net_corrected_errors'])}, "
            f"correction_rate={row['correction_rate']:.4f}, "
            f"error_introduction_rate={row['error_introduction_rate']:.4f}, "
            f"net_error_change={int(row['net_error_change'])}"
        )
    lines.append("")
    lines.append("Transition-family summaries include counts and percentages of the relevant baseline population.")
    lines.append("Distribution summaries include primary reliability, conservative reliability, probability shift, and optional baseline/augmented probability summaries.")
    lines.append("Rate summaries emphasize mean ± SD across seeds; per-seed integer counts remain in `error_accounting_summary.csv`.")
    (OUTPUT_ROOT / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    SAMPLE_ROOT.mkdir(parents=True, exist_ok=True)
    count_summaries: list[pd.DataFrame] = []
    distribution_summaries: list[pd.DataFrame] = []
    error_summaries: list[pd.DataFrame] = []

    for dataset, horizon in PRIMARY_CONFIGS + SECONDARY_CONFIGS:
        for comparison in PRIMARY_COMPARISONS:
            baseline_name, augmented_name = COMPARISON_TO_REPRESENTATIONS[comparison]
            baseline_payload = load_selected_prediction_payload(
                prediction_path(dataset, horizon, baseline_name)
            )
            augmented_payload = load_selected_prediction_payload(
                prediction_path(dataset, horizon, augmented_name)
            )
            scalar_frame = build_primary_scalar_frame((dataset, horizon), comparison)
            for seed in REQUIRED_SEEDS:
                rows = assemble_transition_rows(
                    dataset,
                    horizon,
                    comparison,
                    seed,
                    baseline_payload,
                    augmented_payload,
                    threshold_regime="val_f1",
                    scalar_frame=scalar_frame,
                )
                rows["scope"] = scope_label(dataset, horizon)
                baseline_seed = select_seed_prediction_row(baseline_payload, seed)
                augmented_seed = select_seed_prediction_row(augmented_payload, seed)
                rows["baseline_model_name"] = baseline_seed["model_name"]
                rows["augmented_model_name"] = augmented_seed["model_name"]
                rows["baseline_selected_threshold"] = baseline_seed["selected_threshold"]
                rows["augmented_selected_threshold"] = augmented_seed["selected_threshold"]
                write_sample_level_csv(rows)
                count_summary, distribution_summary, error_summary = summarize_rows(rows)
                count_summaries.append(count_summary)
                distribution_summaries.append(distribution_summary)
                error_summaries.append(error_summary)

                if FIXED_0P5_ENABLED:
                    fixed_rows = assemble_transition_rows(
                        dataset,
                        horizon,
                        comparison,
                        seed,
                        rebuild_fixed_threshold_predictions(baseline_payload, 0.5),
                        rebuild_fixed_threshold_predictions(augmented_payload, 0.5),
                        threshold_regime="fixed_0p5",
                        scalar_frame=scalar_frame,
                    )
                    fixed_rows["scope"] = scope_label(dataset, horizon)
                    fixed_rows["baseline_model_name"] = baseline_seed["model_name"]
                    fixed_rows["augmented_model_name"] = augmented_seed["model_name"]
                    fixed_rows["baseline_selected_threshold"] = 0.5
                    fixed_rows["augmented_selected_threshold"] = 0.5
                    write_sample_level_csv(fixed_rows)
                    count_summary, distribution_summary, error_summary = summarize_rows(
                        fixed_rows
                    )
                    count_summaries.append(count_summary)
                    distribution_summaries.append(distribution_summary)
                    error_summaries.append(error_summary)

    count_df = pd.concat(count_summaries, ignore_index=True)
    distribution_df = pd.concat(distribution_summaries, ignore_index=True)
    error_df = pd.concat(error_summaries, ignore_index=True)
    count_df.to_csv(OUTPUT_ROOT / "transition_count_summary.csv", index=False)
    distribution_df.to_csv(
        OUTPUT_ROOT / "transition_distribution_summary.csv", index=False
    )
    error_df.to_csv(OUTPUT_ROOT / "error_accounting_summary.csv", index=False)
    summarize_error_rates(error_df).to_csv(
        OUTPUT_ROOT / "error_accounting_rate_summary.csv", index=False
    )
    write_markdown_report(count_df, distribution_df, error_df)


if __name__ == "__main__":
    main()
