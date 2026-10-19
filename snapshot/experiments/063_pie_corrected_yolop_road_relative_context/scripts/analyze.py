"""Post-hoc validation-only diagnostics for the completed frozen Exp063 run."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import ExperimentConfig
from .data import load_trainval_context
from .extract_trainval import EXP062_ROOT
from .runner import EXP_ROOT, _atomic_csv


def _transition_counts(labels: np.ndarray, source: np.ndarray, target: np.ndarray) -> dict[str, int]:
    source_class = np.asarray(source) >= 0.5
    target_class = np.asarray(target) >= 0.5
    truth = np.asarray(labels, dtype=np.int64).astype(bool)
    source_correct = source_class == truth
    target_correct = target_class == truth
    return {
        "source_wrong_to_target_correct": int((~source_correct & target_correct).sum()),
        "source_correct_to_target_wrong": int((source_correct & ~target_correct).sum()),
        "source_fp_to_target_tn": int((~truth & source_class & ~target_class).sum()),
        "source_fn_to_target_tp": int((truth & ~source_class & target_class).sum()),
        "source_tn_to_target_fp": int((~truth & ~source_class & target_class).sum()),
        "source_tp_to_target_fn": int((truth & source_class & ~target_class).sum()),
    }


def write_representation_diagnostics(root: Path = EXP_ROOT) -> Path:
    rows: list[dict[str, object]] = []
    for representation in ExperimentConfig().representations:
        data = load_trainval_context(representation)
        for split_name, mask in (("train", data.train_mask), ("validation", data.val_mask)):
            values = data.scene[mask]
            record: dict[str, object] = {
                "representation": representation,
                "split": split_name,
                "windows": int(mask.sum()),
                "frames": int(mask.sum() * 16),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "nonzero_fraction": float(np.mean(values != 0)),
            }
            if representation == "corrected_grid":
                record.update(
                    {
                        "drivable_occupancy_mean": float(values[:, :, :9].mean()),
                        "lane_occupancy_mean": float(values[:, :, 9:18].mean()),
                        "vehicle_count_mean": float(values[:, :, 18].mean()),
                        "pedestrian_count_mean": float(values[:, :, 19].mean()),
                    }
                )
            else:
                record.update(
                    {
                        "road_boundary_valid_fraction": float(values[:, :, 6].mean()),
                        "lane_geometry_valid_fraction": float(values[:, :, 12].mean()),
                        "motion_parallel_road_nonzero_fraction": float(
                            np.mean(values[:, :, 16] != 0)
                        ),
                        "motion_toward_road_nonzero_fraction": float(
                            np.mean(values[:, :, 17] != 0)
                        ),
                        "motion_parallel_lane_nonzero_fraction": float(
                            np.mean(values[:, :, 18] != 0)
                        ),
                    }
                )
            rows.append(record)
    return _atomic_csv(root / "metrics/representation_diagnostics.csv", pd.DataFrame(rows))


def write_error_transitions(root: Path = EXP_ROOT) -> Path:
    rows: list[dict[str, object]] = []
    historical_root = EXP062_ROOT / "predictions"
    for anchor in ExperimentConfig().anchor_families:
        for seed in ExperimentConfig().seeds:
            historical = pd.read_csv(
                historical_root / f"validation_{anchor}_seed{seed}.csv"
            ).set_index("sequence_id")
            current: dict[str, pd.DataFrame] = {}
            for representation in ExperimentConfig().representations:
                frame = pd.read_csv(
                    root
                    / "predictions"
                    / f"validation_{representation}__{anchor}__seed{seed}.csv"
                ).set_index("sequence_id")
                if not frame.index.equals(historical.index) or not np.array_equal(
                    frame["label"].to_numpy(), historical["label"].to_numpy()
                ):
                    raise RuntimeError("Post-hoc transition cohort identity drifted")
                current[representation] = frame
                rows.append(
                    {
                        "comparison": f"historical_to_{representation}",
                        "anchor_name": anchor,
                        "seed": seed,
                        **_transition_counts(
                            frame["label"].to_numpy(),
                            historical["corrected_probability"].to_numpy(),
                            frame["corrected_probability"].to_numpy(),
                        ),
                    }
                )
            grid = current["corrected_grid"]
            road = current["road_relative"]
            rows.append(
                {
                    "comparison": "corrected_grid_to_road_relative",
                    "anchor_name": anchor,
                    "seed": seed,
                    **_transition_counts(
                        grid["label"].to_numpy(),
                        grid["corrected_probability"].to_numpy(),
                        road["corrected_probability"].to_numpy(),
                    ),
                }
            )
    return _atomic_csv(
        root / "metrics/error_transitions_fixed_0p5.csv", pd.DataFrame(rows)
    )


def main() -> None:
    write_representation_diagnostics()
    write_error_transitions()


if __name__ == "__main__":
    main()
