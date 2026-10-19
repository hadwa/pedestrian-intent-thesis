# Exp063: Corrected YOLOP Road-Relative Context

Exp063 is a controlled PIE train/validation experiment testing whether the
historical YOLOP context problem came from incorrect segmentation
postprocessing, an unsuitable grid representation, or both.

The immutable scientific design is
`docs/superpowers/specs/2026-08-26-exp063-corrected-yolop-road-relative-context-design.md`.
The implementation plan is
`docs/superpowers/plans/2026-08-26-exp063-corrected-yolop-road-relative-context.md`.

## Variants

- **H:** authenticated Exp062 historical 20-D YOLOP reference; reused, not
  retrained.
- **G:** officially postprocessed YOLOP masks represented by the historical
  9+9+2 grid/count layout.
- **R:** the same corrected masks represented by a frozen 20-D
  YOLOP+BBox-derived road-relative sequence.

The two tree anchors, ego sequence, residual architecture, seeds, optimizer,
fixed threshold 0.5, and validation gates remain those of Exp062. Set03 is
forbidden until a candidate passes every frozen validation gate and is bound by
an authenticated final freeze.

## Literature

The YOLOP paper used to justify the shared detection/drivable-area/lane-line
perception source is already preserved at `papers/Wu_2021_YOLOP.pdf`. The
experiment does not claim that its new 20-D road-relative summary was proposed
by that paper; it is the predeclared Exp063 representation being tested.

## Commands

All commands run from the repository root with the Exp062-compatible Python
3.10 environment:

```bash
experiments/062_hybrid_tree_anchor_context_correction_skl152/.venv/bin/python -m pytest -q experiments/063_pie_corrected_yolop_road_relative_context/tests
experiments/062_hybrid_tree_anchor_context_correction_skl152/.venv/bin/python -m compileall experiments/063_pie_corrected_yolop_road_relative_context/scripts
```

The verified run completed validation-only. Neither corrected representation
passed the frozen gate, so set03 was not accessed. The result and rerun decision
are documented in `FINAL_REPORT.md`.

Executed commands:

```bash
/home/abdelrahman/miniconda3/envs/mslstm_pid_env/bin/python \
  -m experiments.063_pie_corrected_yolop_road_relative_context.scripts.runner \
  extract --device cuda:0

experiments/062_hybrid_tree_anchor_context_correction_skl152/.venv/bin/python \
  -m experiments.063_pie_corrected_yolop_road_relative_context.scripts.runner prepare

experiments/062_hybrid_tree_anchor_context_correction_skl152/.venv/bin/python \
  -m experiments.063_pie_corrected_yolop_road_relative_context.scripts.runner \
  train-all --device cuda:0

experiments/062_hybrid_tree_anchor_context_correction_skl152/.venv/bin/python \
  -m experiments.063_pie_corrected_yolop_road_relative_context.scripts.runner \
  validate --device cuda:0

experiments/062_hybrid_tree_anchor_context_correction_skl152/.venv/bin/python \
  -m experiments.063_pie_corrected_yolop_road_relative_context.scripts.runner verify
```

Generated caches, checkpoints, predictions, metrics, and logs become evidence
only through the hashes in `VALIDATION_COMPLETE.json`,
`VALIDATION_ONLY_COMPLETE`, and `EXPERIMENT_COMPLETE`.
