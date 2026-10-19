from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


QUARTILE_LABELS = ["Q1_lowest", "Q2", "Q3", "Q4_highest"]
REPO_ROOT = Path(__file__).resolve().parents[2]
EXP089 = REPO_ROOT / "experiments/089_pedcmt_standardized_jaad_biu_transfer"
EXP094 = REPO_ROOT / "experiments/094_pedcmt_jaad_capped_raw_pose_control"
CONFIDENCE_PATH = (
    REPO_ROOT
    / "experiments/014_confidence_aware_biu_audit/metrics/window_confidence_30_60.csv"
)
ROWS_PATH = EXP089 / "cache/test_source/rows.csv"
POSE_PATHS = [
    REPO_ROOT / "data/jaad_keypoints_vitpose_s_extended_short_horizon.jsonl",
    REPO_ROOT
    / "experiments/030_final_confidence_benchmark/jaad30_repair/jaad30_missing_supplemental_keypoints.jsonl",
]
JAAD_IMAGE_ROOT = Path("/home/abdelrahman/Hadwa/datasets/JAAD/images")
OUTPUT_ROOT = REPO_ROOT / "experiments/097_jaad_pose_label_forensic_audit"
METRICS_ROOT = OUTPUT_ROOT / "metrics"
FIGURES_ROOT = OUTPUT_ROOT / "figures"
CASE_ROOT = OUTPUT_ROOT / "case_sheets"
SEEDS = (42, 43, 44)
MODEL_FILES = {"BBox": "A", "RawPose": "B", "BIUv2": "D", "Union": "E"}
COCO_EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def _require_unique(frame: pd.DataFrame, name: str) -> None:
    duplicates = frame.loc[frame["sequence_id"].duplicated(), "sequence_id"]
    if not duplicates.empty:
        example = duplicates.iloc[0]
        raise ValueError(f"{name} contains duplicate sequence_id {example!r}")


def join_predictions(
    metadata: pd.DataFrame,
    confidence: pd.DataFrame,
    predictions: Mapping[tuple[str, int], pd.DataFrame],
) -> pd.DataFrame:
    """Join matched saved predictions to metadata and pose confidence."""
    _require_unique(metadata, "metadata")
    _require_unique(confidence, "confidence")
    expected_ids = set(metadata["sequence_id"].astype(str))
    confidence_ids = set(confidence["sequence_id"].astype(str))
    if confidence_ids != expected_ids:
        raise ValueError("confidence sequence-ID mismatch")

    base = metadata.merge(
        confidence[["sequence_id", "skeleton_conf_mean"]],
        on="sequence_id",
        how="left",
        validate="one_to_one",
    )
    joined: list[pd.DataFrame] = []
    for (model, seed), prediction_frame in predictions.items():
        _require_unique(prediction_frame, f"predictions[{model}, {seed}]")
        prediction_ids = set(prediction_frame["sequence_id"].astype(str))
        if prediction_ids != expected_ids:
            raise ValueError(
                f"sequence-ID mismatch for model={model}, seed={seed}"
            )
        columns = ["sequence_id", "label", "probability"]
        part = base.merge(
            prediction_frame[columns],
            on="sequence_id",
            how="left",
            suffixes=("", "_prediction"),
            validate="one_to_one",
        )
        if not np.array_equal(
            part["label"].to_numpy(dtype=int),
            part["label_prediction"].to_numpy(dtype=int),
        ):
            raise ValueError(f"label mismatch for model={model}, seed={seed}")
        part = part.drop(columns="label_prediction")
        part["model"] = model
        part["seed"] = int(seed)
        joined.append(part)
    if not joined:
        raise ValueError("no prediction frames supplied")
    return pd.concat(joined, ignore_index=True)


def assign_quartiles(
    frame: pd.DataFrame,
    edges: Sequence[float],
) -> pd.DataFrame:
    if len(edges) != 3:
        raise ValueError("exactly three quartile edges are required")
    numeric_edges = [float(value) for value in edges]
    if numeric_edges != sorted(numeric_edges):
        raise ValueError("quartile edges must be sorted")
    out = frame.copy()
    out["confidence_quartile"] = pd.cut(
        out["skeleton_conf_mean"],
        bins=[-np.inf, *numeric_edges, np.inf],
        labels=QUARTILE_LABELS,
        include_lowest=True,
        ordered=True,
    )
    return out


def add_fixed_threshold_errors(
    frame: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    out = frame.copy()
    out["prediction"] = (out["probability"] >= threshold).astype(int)
    out["is_false_positive"] = (
        (out["label"] == 0) & (out["prediction"] == 1)
    ).astype(int)
    out["is_false_negative"] = (
        (out["label"] == 1) & (out["prediction"] == 0)
    ).astype(int)
    out["is_error"] = (out["label"] != out["prediction"]).astype(int)
    return out


def summarize_error_rates(frame: pd.DataFrame) -> pd.DataFrame:
    group_columns = [
        "model",
        "seed",
        "crossing_attr",
        "confidence_quartile",
    ]
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_columns, observed=True, dropna=False):
        model, seed, crossing_attr, confidence_quartile = keys
        negatives = int((group["label"] == 0).sum())
        positives = int((group["label"] == 1).sum())
        false_positives = int(group["is_false_positive"].sum())
        false_negatives = int(group["is_false_negative"].sum())
        rows.append(
            {
                "model": model,
                "seed": int(seed),
                "crossing_attr": int(crossing_attr),
                "confidence_quartile": str(confidence_quartile),
                "n_windows": int(len(group)),
                "n_tracks": int(group["track_id"].nunique()),
                "n_negative": negatives,
                "n_positive": positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "false_positive_rate": (
                    false_positives / negatives if negatives else np.nan
                ),
                "false_negative_rate": (
                    false_negatives / positives if positives else np.nan
                ),
                "error_rate": float(group["is_error"].mean()),
                "mean_probability": float(group["probability"].mean()),
            }
        )
    return pd.DataFrame(rows)


def ensemble_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Average probabilities across seeds before applying the fixed threshold."""
    invariant_columns = [
        column
        for column in frame.columns
        if column not in {"seed", "probability", "prediction", "is_false_positive", "is_false_negative", "is_error"}
    ]
    key_columns = ["sequence_id", "model"]
    check_columns = [column for column in invariant_columns if column not in key_columns]
    grouped = frame.groupby(key_columns, sort=False, observed=True, dropna=False)
    for column in check_columns:
        if (grouped[column].nunique(dropna=False) > 1).any():
            raise ValueError(f"{column} is not invariant across seeds")
    metadata = grouped[check_columns].first().reset_index()
    probabilities = grouped["probability"].agg(["mean", "count"]).reset_index()
    probabilities = probabilities.rename(
        columns={"mean": "probability", "count": "n_seeds"}
    )
    out = metadata.merge(probabilities, on=key_columns, validate="one_to_one")
    return add_fixed_threshold_errors(out)


def cluster_bootstrap_rate_difference(
    frame: pd.DataFrame,
    *,
    group_column: str,
    left_group: object,
    right_group: object,
    outcome_column: str,
    n_bootstrap: int = 5000,
    random_seed: int = 20260905,
) -> dict[str, float | int | str]:
    """Bootstrap a rate difference while resampling pedestrian tracks."""
    selected = frame.loc[
        frame[group_column].isin([left_group, right_group]),
        ["track_id", group_column, outcome_column],
    ].copy()
    left = selected.loc[selected[group_column] == left_group, outcome_column]
    right = selected.loc[selected[group_column] == right_group, outcome_column]
    if left.empty or right.empty:
        raise ValueError("both contrast groups must contain observations")
    observed = float(left.mean() - right.mean())

    tracks = pd.Index(selected["track_id"].drop_duplicates())
    aggregate = (
        selected.groupby(["track_id", group_column], observed=True)[outcome_column]
        .agg(["sum", "count"])
        .reset_index()
    )
    sum_pivot = aggregate.pivot(
        index="track_id", columns=group_column, values="sum"
    ).reindex(index=tracks, columns=[left_group, right_group], fill_value=0).fillna(0)
    count_pivot = aggregate.pivot(
        index="track_id", columns=group_column, values="count"
    ).reindex(index=tracks, columns=[left_group, right_group], fill_value=0).fillna(0)
    sums = sum_pivot.to_numpy(dtype=float)
    counts = count_pivot.to_numpy(dtype=float)

    rng = np.random.default_rng(random_seed)
    differences: list[float] = []
    probabilities = np.full(len(tracks), 1.0 / len(tracks))
    for _ in range(int(n_bootstrap)):
        weights = rng.multinomial(len(tracks), probabilities)
        denominators = weights @ counts
        if np.any(denominators == 0):
            continue
        rates = (weights @ sums) / denominators
        differences.append(float(rates[0] - rates[1]))
    if not differences:
        raise ValueError("no valid bootstrap replicates")
    lower, upper = np.quantile(differences, [0.025, 0.975])
    return {
        "left_group": str(left_group),
        "right_group": str(right_group),
        "observed_difference": observed,
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "n_tracks": int(len(tracks)),
        "valid_bootstrap_replicates": int(len(differences)),
    }


def select_forensic_cases(
    ensemble_frame: pd.DataFrame,
    *,
    cases_per_category: int = 2,
) -> pd.DataFrame:
    """Select non-overlapping tracks that expose distinct error mechanisms."""
    probabilities = ensemble_frame.pivot(
        index="sequence_id", columns="model", values="probability"
    )
    predictions = ensemble_frame.pivot(
        index="sequence_id", columns="model", values="prediction"
    )
    base = ensemble_frame.drop_duplicates("sequence_id").set_index("sequence_id")
    common_models = [
        model
        for model in ["BBox", "RawPose", "BIUv2", "Union"]
        if model in predictions.columns
    ]
    if "BBox" not in common_models or "BIUv2" not in common_models:
        raise ValueError("case selection requires BBox and BIUv2 predictions")
    high_confidence = base["confidence_quartile"].astype(str).eq("Q4_highest")
    low_confidence = base["confidence_quartile"].astype(str).eq("Q1_lowest")
    negative = base["label"].eq(0)
    positive = base["label"].eq(1)
    all_positive = predictions[common_models].eq(1).all(axis=1)
    all_negative = predictions[common_models].eq(0).all(axis=1)

    categories: list[tuple[str, pd.Series, pd.Series, bool]] = [
        (
            "all_models_high_confidence_false_positive",
            negative & high_confidence & all_positive,
            probabilities[common_models].min(axis=1),
            False,
        ),
        (
            "all_models_high_confidence_false_negative",
            positive & high_confidence & all_negative,
            probabilities[common_models].max(axis=1),
            True,
        ),
        (
            "bbox_false_positive_biuv2_true_negative_high_confidence",
            negative
            & high_confidence
            & predictions["BBox"].eq(1)
            & predictions["BIUv2"].eq(0),
            probabilities["BBox"] - probabilities["BIUv2"],
            False,
        ),
        (
            "bbox_true_negative_biuv2_false_positive_high_confidence",
            negative
            & high_confidence
            & predictions["BBox"].eq(0)
            & predictions["BIUv2"].eq(1),
            probabilities["BIUv2"] - probabilities["BBox"],
            False,
        ),
        (
            "biuv2_low_confidence_false_positive",
            negative & low_confidence & predictions["BIUv2"].eq(1),
            probabilities["BIUv2"],
            False,
        ),
        (
            "biuv2_high_confidence_false_positive",
            negative & high_confidence & predictions["BIUv2"].eq(1),
            probabilities["BIUv2"],
            False,
        ),
    ]

    used_tracks: set[str] = set()
    selected_rows: list[dict[str, object]] = []
    for category, mask, rank_score, ascending in categories:
        candidates = base.loc[mask].copy()
        candidates["rank_score"] = rank_score.loc[candidates.index]
        candidates = candidates.sort_values("rank_score", ascending=ascending)
        chosen = 0
        for sequence_id, row in candidates.iterrows():
            track_id = str(row["track_id"])
            if track_id in used_tracks:
                continue
            used_tracks.add(track_id)
            record = row.to_dict()
            record.update(
                {
                    "sequence_id": sequence_id,
                    "category": category,
                    "rank_score": float(row["rank_score"]),
                }
            )
            for model in common_models:
                record[f"{model}_probability"] = float(
                    probabilities.loc[sequence_id, model]
                )
                record[f"{model}_prediction"] = int(
                    predictions.loc[sequence_id, model]
                )
            selected_rows.append(record)
            chosen += 1
            if chosen >= cases_per_category:
                break
    return pd.DataFrame(selected_rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_primary_predictions() -> dict[tuple[str, int], pd.DataFrame]:
    predictions: dict[tuple[str, int], pd.DataFrame] = {}
    for model, variant in MODEL_FILES.items():
        for seed in SEEDS:
            path = EXP089 / f"predictions/test_{variant}_seed{seed}.csv"
            predictions[(model, seed)] = pd.read_csv(path)
    return predictions


def _load_capped_predictions() -> dict[tuple[str, int], pd.DataFrame]:
    mapping = {
        "CappedBBox": "A_probability",
        "CappedRawPose": "P10_probability",
        "CappedUnion": "U10_probability",
    }
    predictions: dict[tuple[str, int], pd.DataFrame] = {}
    for seed in SEEDS:
        path = EXP094 / f"predictions/test_aligned_A_P10_U10_seed{seed}.csv"
        wide = pd.read_csv(path)
        for model, probability_column in mapping.items():
            predictions[(model, seed)] = wide[
                ["sequence_id", "label", probability_column]
            ].rename(columns={probability_column: "probability"})
    return predictions


def _with_all_confidence_band(frame: pd.DataFrame) -> pd.DataFrame:
    all_band = frame.copy()
    all_band["confidence_quartile"] = "ALL"
    return pd.concat([frame, all_band], ignore_index=True)


def _bootstrap_contrasts(ensemble: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, model_frame in ensemble.groupby("model", sort=False):
        negatives = model_frame.loc[model_frame["label"] == 0]
        positives = model_frame.loc[model_frame["label"] == 1]
        contrasts = [
            (
                "attr0_minus_attr-1_fpr",
                negatives,
                "crossing_attr",
                0,
                -1,
                "is_false_positive",
            ),
            (
                "q1_minus_q4_fpr_all_negatives",
                negatives,
                "confidence_quartile",
                "Q1_lowest",
                "Q4_highest",
                "is_false_positive",
            ),
            (
                "q1_minus_q4_fnr_positives",
                positives,
                "confidence_quartile",
                "Q1_lowest",
                "Q4_highest",
                "is_false_negative",
            ),
        ]
        for attr in (-1, 0):
            contrasts.append(
                (
                    f"q1_minus_q4_fpr_attr{attr}",
                    negatives.loc[negatives["crossing_attr"] == attr],
                    "confidence_quartile",
                    "Q1_lowest",
                    "Q4_highest",
                    "is_false_positive",
                )
            )
        for contrast, subset, group_column, left, right, outcome in contrasts:
            result = cluster_bootstrap_rate_difference(
                subset,
                group_column=group_column,
                left_group=left,
                right_group=right,
                outcome_column=outcome,
            )
            rows.append({"model": model, "contrast": contrast, **result})

    errors = ensemble.pivot(
        index="sequence_id", columns="model", values="is_false_positive"
    )
    base = ensemble.drop_duplicates("sequence_id").set_index("sequence_id")
    for model in ("RawPose", "BIUv2", "Union"):
        paired = base[
            ["track_id", "label", "crossing_attr", "confidence_quartile"]
        ].copy()
        paired["relative_fp_error"] = errors[model] - errors["BBox"]
        paired = paired.loc[paired["label"] == 0]
        for contrast, group_column, left, right in [
            (
                "relative_to_bbox_q1_minus_q4_fpr",
                "confidence_quartile",
                "Q1_lowest",
                "Q4_highest",
            ),
            (
                "relative_to_bbox_attr0_minus_attr-1_fpr",
                "crossing_attr",
                0,
                -1,
            ),
        ]:
            result = cluster_bootstrap_rate_difference(
                paired,
                group_column=group_column,
                left_group=left,
                right_group=right,
                outcome_column="relative_fp_error",
            )
            rows.append({"model": model, "contrast": contrast, **result})
    return pd.DataFrame(rows)


def _confidence_composition(rows: pd.DataFrame) -> pd.DataFrame:
    working = rows.copy()
    working["bbox_height_mean_px"] = working["bbox_sequence_json"].map(
        lambda raw: float(
            np.mean(
                [
                    float(box[3]) - float(box[1])
                    for box in json.loads(raw)
                ]
            )
        )
    )
    return (
        working.groupby("confidence_quartile", observed=True)
        .agg(
            n_windows=("sequence_id", "size"),
            n_tracks=("track_id", "nunique"),
            positive_prevalence=("label", "mean"),
            attr_minus_one_windows=("crossing_attr", lambda x: int((x == -1).sum())),
            attr_zero_windows=("crossing_attr", lambda x: int((x == 0).sum())),
            attr_one_windows=("crossing_attr", lambda x: int((x == 1).sum())),
            skeleton_conf_mean=("skeleton_conf_mean", "mean"),
            bbox_height_mean_px=("bbox_height_mean_px", "mean"),
            bbox_height_median_px=("bbox_height_mean_px", "median"),
        )
        .reset_index()
    )


def _load_selected_pose_index(
    selected: pd.DataFrame,
) -> dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]]:
    required: set[tuple[str, str, int]] = set()
    for row in selected.itertuples(index=False):
        for frame_index in json.loads(row.observed_frames_json):
            required.add((str(row.video_id), str(row.ped_id), int(frame_index)))
    result: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]] = {}
    for path in POSE_PATHS:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                key = (
                    str(record["video"]),
                    str(record["pid"]),
                    int(record["frame_idx"]),
                )
                if key not in required or key in result:
                    continue
                result[key] = (
                    np.asarray(record["keypoints"], dtype=float),
                    np.asarray(record["kp_conf"], dtype=float),
                )
    missing = required.difference(result)
    if missing:
        raise ValueError(f"missing pose records for selected cases: {len(missing)}")
    return result


def _render_case_sheets(selected: pd.DataFrame) -> None:
    pose_index = _load_selected_pose_index(selected)
    font = ImageFont.load_default()
    tile_width, tile_height = 240, 190
    gap, title_height = 6, 78
    for case_number, row in enumerate(selected.itertuples(index=False), start=1):
        frames = [int(value) for value in json.loads(row.observed_frames_json)]
        boxes = json.loads(row.bbox_sequence_json)
        sheet = Image.new(
            "RGB",
            (4 * tile_width + 5 * gap, title_height + 4 * tile_height + 5 * gap),
            (242, 242, 242),
        )
        title_draw = ImageDraw.Draw(sheet)
        probabilities = " ".join(
            f"{model}={getattr(row, f'{model}_probability'):.3f}"
            for model in MODEL_FILES
        )
        title = (
            f"case_{case_number:02d} | {row.category}\n"
            f"{row.sequence_id} | attr={int(row.crossing_attr)} y={int(row.label)} "
            f"pose_conf={float(row.skeleton_conf_mean):.3f} {row.confidence_quartile}\n"
            f"ensemble probabilities: {probabilities} | fixed threshold=0.5"
        )
        title_draw.multiline_text((8, 6), title, fill=(0, 0, 0), font=font, spacing=3)
        for index, (frame_index, bbox) in enumerate(zip(frames, boxes)):
            image_path = JAAD_IMAGE_ROOT / str(row.video_id) / f"{frame_index:05d}.png"
            image = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(image)
            x1, y1, x2, y2 = map(float, bbox)
            draw.rectangle((x1, y1, x2, y2), outline=(255, 60, 40), width=4)
            coords, confidence = pose_index[
                (str(row.video_id), str(row.ped_id), frame_index)
            ]
            for start, end in COCO_EDGES:
                if confidence[start] >= 0.2 and confidence[end] >= 0.2:
                    draw.line(
                        [tuple(coords[start]), tuple(coords[end])],
                        fill=(0, 255, 120),
                        width=4,
                    )
            for point, point_confidence in zip(coords, confidence):
                if point_confidence < 0.2:
                    continue
                px, py = map(float, point)
                radius = 4
                draw.ellipse(
                    (px - radius, py - radius, px + radius, py + radius),
                    fill=(255, 225, 0),
                    outline=(0, 0, 0),
                )
            box_width = max(x2 - x1, 70.0)
            box_height = max(y2 - y1, 130.0)
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            crop_width = max(box_width * 3.2, box_height * 1.25)
            crop_height = max(box_height * 1.55, box_width * 2.0)
            crop = (
                max(0, int(center_x - crop_width / 2)),
                max(0, int(center_y - crop_height / 2)),
                min(image.width, int(center_x + crop_width / 2)),
                min(image.height, int(center_y + crop_height / 2)),
            )
            tile = image.crop(crop).resize(
                (tile_width, tile_height), Image.Resampling.BILINEAR
            )
            tile_draw = ImageDraw.Draw(tile)
            tile_draw.rectangle((0, 0, 132, 18), fill=(0, 0, 0))
            tile_draw.text(
                (3, 3),
                f"t={index:02d} fr={frame_index} kp={confidence.mean():.2f}",
                fill=(255, 255, 255),
                font=font,
            )
            column, grid_row = index % 4, index // 4
            x = gap + column * (tile_width + gap)
            y = title_height + gap + grid_row * (tile_height + gap)
            sheet.paste(tile, (x, y))
        filename = f"case_{case_number:02d}_{row.category}.png"
        sheet.save(CASE_ROOT / filename, optimize=True)


def _write_figures(ensemble_summary: pd.DataFrame) -> None:
    quartiles = QUARTILE_LABELS
    negative = ensemble_summary.loc[
        ensemble_summary["crossing_attr"].isin([-1, 0])
        & ensemble_summary["confidence_quartile"].isin(quartiles)
    ]
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for axis, model in zip(axes.flat, MODEL_FILES):
        subset = negative.loc[negative["model"] == model]
        for attr, label, color in [(-1, "attr=-1", "#3366cc"), (0, "attr=0", "#dc3912")]:
            values = (
                subset.loc[subset["crossing_attr"] == attr]
                .set_index("confidence_quartile")
                .reindex(quartiles)["false_positive_rate"]
            )
            axis.plot(quartiles, values, marker="o", label=label, color=color)
        axis.set_title(model)
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
        axis.tick_params(axis="x", rotation=25)
    axes[0, 0].set_ylabel("False-positive rate")
    axes[1, 0].set_ylabel("False-positive rate")
    axes[0, 0].legend()
    figure.suptitle("JAAD ensemble FPR by annotation subgroup and pose confidence")
    figure.tight_layout()
    figure.savefig(FIGURES_ROOT / "ensemble_fpr_by_attr_and_pose_confidence.png", dpi=180)
    plt.close(figure)


def main() -> None:
    for directory in (OUTPUT_ROOT, METRICS_ROOT, FIGURES_ROOT, CASE_ROOT):
        directory.mkdir(parents=True, exist_ok=True)

    full_rows = pd.read_csv(ROWS_PATH)
    metadata_columns = [
        "sequence_id",
        "video_id",
        "ped_id",
        "track_id",
        "start_frame",
        "end_frame",
        "event_frame",
        "tte_frames",
        "label",
        "crossing_attr",
    ]
    metadata = full_rows[metadata_columns].copy()
    confidence_all = pd.read_csv(CONFIDENCE_PATH)
    development_confidence = confidence_all.loc[
        confidence_all["split"].isin(["train", "val"]), "skeleton_conf_mean"
    ]
    edges = development_confidence.quantile([0.25, 0.50, 0.75]).to_numpy()
    confidence_test = confidence_all.loc[
        confidence_all["split"] == "test",
        ["sequence_id", "skeleton_conf_mean"],
    ]

    joined = join_predictions(metadata, confidence_test, _load_primary_predictions())
    joined = add_fixed_threshold_errors(assign_quartiles(joined, edges))
    per_seed_summary = summarize_error_rates(_with_all_confidence_band(joined))
    per_seed_summary.to_csv(
        METRICS_ROOT / "per_seed_error_rates_by_attr_confidence.csv", index=False
    )

    ensemble = ensemble_predictions(joined)
    ensemble_summary = summarize_error_rates(
        _with_all_confidence_band(ensemble.assign(seed=-1))
    ).drop(columns="seed")
    ensemble_summary.to_csv(
        METRICS_ROOT / "ensemble_error_rates_by_attr_confidence.csv", index=False
    )
    _bootstrap_contrasts(ensemble).to_csv(
        METRICS_ROOT / "track_cluster_bootstrap_contrasts.csv", index=False
    )

    composition_rows = full_rows.merge(
        confidence_test, on="sequence_id", validate="one_to_one"
    )
    composition_rows = assign_quartiles(composition_rows, edges)
    _confidence_composition(composition_rows).to_csv(
        METRICS_ROOT / "confidence_quartile_composition.csv", index=False
    )

    capped = join_predictions(metadata, confidence_test, _load_capped_predictions())
    capped = add_fixed_threshold_errors(assign_quartiles(capped, edges))
    capped_ensemble = ensemble_predictions(capped)
    capped_summary = summarize_error_rates(
        _with_all_confidence_band(capped_ensemble.assign(seed=-1))
    ).drop(columns="seed")
    capped_summary.to_csv(
        METRICS_ROOT / "capped_ensemble_error_rates_by_attr_confidence.csv",
        index=False,
    )

    selected = select_forensic_cases(ensemble, cases_per_category=2)
    selected = selected.merge(
        full_rows[
            [
                "sequence_id",
                "observed_frames_json",
                "bbox_sequence_json",
                "frame_width",
                "frame_height",
            ]
        ],
        on="sequence_id",
        validate="one_to_one",
    )
    selected.insert(0, "case_id", [f"case_{index:02d}" for index in range(1, len(selected) + 1)])
    selected.to_csv(METRICS_ROOT / "selected_forensic_cases.csv", index=False)
    _render_case_sheets(selected)
    _write_figures(ensemble_summary)

    provenance_paths = [ROWS_PATH, CONFIDENCE_PATH, *POSE_PATHS]
    provenance_paths.extend(
        EXP089 / f"predictions/test_{variant}_seed{seed}.csv"
        for variant in MODEL_FILES.values()
        for seed in SEEDS
    )
    provenance_paths.extend(
        EXP094 / f"predictions/test_aligned_A_P10_U10_seed{seed}.csv"
        for seed in SEEDS
    )
    integrity = {
        "status": "diagnostic_not_main_benchmark",
        "threshold": 0.5,
        "test_windows": int(len(metadata)),
        "test_tracks": int(metadata["track_id"].nunique()),
        "models": list(MODEL_FILES),
        "seeds": list(SEEDS),
        "development_derived_confidence_edges": [float(value) for value in edges],
        "primary_join_rows": int(len(joined)),
        "expected_primary_join_rows": int(len(metadata) * len(MODEL_FILES) * len(SEEDS)),
        "ensemble_rows": int(len(ensemble)),
        "selected_case_count": int(len(selected)),
        "source_sha256": {str(path): _sha256(path) for path in provenance_paths},
    }
    (METRICS_ROOT / "join_integrity_and_provenance.json").write_text(
        json.dumps(integrity, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in integrity.items() if key != "source_sha256"}, indent=2))


if __name__ == "__main__":
    main()
