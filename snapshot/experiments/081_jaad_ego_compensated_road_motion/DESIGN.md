# Exp081 JAAD Ego-Compensated Road Motion Design

## Status

Approved design specification. Implementation and execution have not started.

## Experiment Identity

- Next free experiment number verified on 2026-08-28: `081`.
- Planned folder:
  `experiments/081_jaad_ego_compensated_road_motion/`.
- Category: controlled JAAD clean-benchmark experiment.
- Development protocol: validation only until the frozen primary gate passes.
- Final test protocol: one evaluation of the frozen primary candidate, and only
  after an authenticated validation decision to advance.

## Scientific Motivation

JAAD does not provide the continuous ego-speed telemetry available in PIE.
This is a dataset limitation, not a realistic-vehicle limitation: a deployed
vehicle may obtain speed and acceleration from CAN, wheel odometry, or IMU.
Exp081 must therefore call the camera-derived quantity a **visual ego-motion
proxy**, never measured speed or acceleration.

Previous PIE experiments already tested road-relative geometry:

- Exp063 computed a 20-channel representation containing road occupancy,
  boundary and lane geometry, raw foot velocity, motion toward the drivable
  area, and motion parallel to the road and lane. Its validation gains were
  small and unstable, and it did not advance to set03.
- Exp074 added 60 early/late/delta road-relative summaries to the strong Exp072
  XGBoost representation. Validation F1 and PR-AUC declined, so it did not
  advance.
- Exp075 replaced YOLOP with TwinLiteNet+ Large for the same representation.
  PIE set03 F1 increased slightly, but PR-AUC declined and the result was not a
  consistent overall improvement.

Those experiments used image-space pedestrian motion. They did not first
remove global camera translation, rotation, or scale change. Exp081 tests the
new, narrower hypothesis that road-relative motion becomes more useful after
camera motion is estimated from the background and removed from target
pedestrian motion.

## Primary Question

On the clean JAAD 16-frame, TTE 30--60 benchmark, does replacing raw
image-space motion in the existing road-relative representation with
background-motion-compensated pedestrian motion improve PR-AUC and fixed-0.5
F1 without increasing high-confidence errors?

## Secondary Question

Does a compact global visual ego-motion descriptor provide predictive value
on its own beyond the existing portable BBox/BIU/temporal representation?

## Frozen Benchmark And Anchor

- Dataset: clean event-aligned JAAD behavior-pedestrian cohort from Exp077.
- Observation: exactly 16 consecutive frames.
- TTE: exactly 30--60 frames under the authenticated Exp077 construction.
- Train, validation, and test IDs, ordering, labels, and frame windows: exact
  Exp077 lineage.
- Threshold: fixed `0.5`.
- Seeds: `42`, `43`, `44`.
- Classifier and training recipe: exact Exp077 one-stage 182-D XGBoost recipe
  with a 250-tree budget. The Exp077 early/late two-stage model is not used.
- Anchor input: exact 182-D Exp077 portable feature contract.
- No threshold selection, calibration, model-family search, feature selection,
  or hyperparameter search.
- No clean test IDs, features, predictions, or metrics may be loaded during
  development.

Exp080's RF result does not replace the anchor because its clean-test F1 and
ROC-AUC did not improve over Exp077 one-stage XGBoost. Exp081 changes the
representation, not the classifier family.

## Perception Source

Use the same authenticated TwinLiteNet+ Large lineage as Exp075:

- upstream project: `chequanghuy/TwinLiteNetPlus`;
- configuration: `Large`;
- checkpoint SHA-256:
  `5605d1a8ce762c1b6f7d23f4c2e7ec9384bfd6e6887b040b43e1f8902a817e22`;
- local paper: `papers/Che_2025_TwinLiteNetPlus.pdf`;
- adapter/postprocessing: reuse the verified Exp075 implementation unless a
  parity test proves that a repository-level extraction utility is identical.

Extract only unique JAAD train/validation frames before the validation gate.
Do not extract clean-test masks until the primary candidate is frozen and the
gate authorizes final evaluation. Cache masks or compact geometry by exact
video/frame identity so overlapping windows never repeat perception inference.

## Visual Ego-Motion Estimator

The estimator is deterministic and label-free:

1. Convert consecutive observed RGB frames to grayscale.
2. Detect Shi-Tomasi background corners.
3. Exclude every available annotated pedestrian BBox, expanded by 10% of its
   width and height on every side and clipped to the image, from corner
   detection and fitting.
4. Track corners with pyramidal Lucas-Kanade optical flow.
5. Apply a forward/backward tracking-consistency check.
6. Fit a four-degree-of-freedom partial affine transform with RANSAC. The
   transform represents translation, rotation, and uniform scale change.
7. Fit each transform in the prior-frame to current-frame direction. Compose
   and invert those transforms to map every later frame into the first-frame
   coordinate system.

The implementation plan must freeze all OpenCV numeric settings before any
validation metric is generated. Defaults may be inherited from the documented
OpenCV APIs where appropriate, but they must be written to the experiment
configuration rather than left implicit.

If a transition cannot be estimated, use an identity transform and set its
validity to zero. No sequence may be dropped. Record tracked-point count,
RANSAC inlier fraction, forward/backward error, and validity for audit. These
quality values may be used only in the compact proxy contract declared below;
they may not trigger manual sample exclusion.

## Visual Ego-Motion Proxy Contract

For each of the 15 frame transitions, derive:

- horizontal translation divided by image width;
- vertical translation divided by image height;
- rotation in radians;
- log uniform-scale change.

For each channel calculate three fixed summaries:

- mean over transitions 1--7;
- mean over transitions 8--15;
- late mean minus early mean.

Append:

- valid-transition fraction;
- mean RANSAC inlier fraction over valid transitions;
- magnitude of the final cumulative translation after dividing its horizontal
  and vertical components by image width and height, respectively.

This produces exactly 15 visual ego-motion features. No mapping to km/h or
m/s2 is allowed.

## Road-Relative Contracts

### Uncompensated Control

Reuse the exact 20-channel Exp063/074 road-relative definition and the exact
Exp074 summarization: each channel is summarized by observation frames 1--8
mean, frames 9--16 mean, and late minus early. This produces 60 features.

### Compensated Candidate

Keep the first 13 static road/lane geometry channels identical. Preserve the
same 20-channel names, ordering, and 60-feature summarization. Change only the
seven temporal channels:

- map each pedestrian foot point into the first-frame coordinate system using
  the cumulative background transform;
- calculate foot velocity and speed from those stabilized points;
- project stabilized velocity onto the contemporaneous road-boundary and lane
  tangents;
- calculate motion toward the drivable area from stabilized velocity;
- retain signed road-boundary distance change as the same-frame relational
  measure already defined by Exp063.

No future frame is used. No pose confidence, GT behavior, GT scene label,
privileged ego annotation, or manually selected example enters construction.

## Frozen Variants

| Variant | Input | Dimension | Purpose |
| --- | --- | ---: | --- |
| A | Exact Exp077 portable anchor | 182 | Reproduced reference |
| B | A + visual ego-motion proxy | 197 | Global visual-motion effect |
| C | A + uncompensated road-relative summaries | 242 | Existing road representation control on JAAD |
| D | A + compensated road-relative summaries | 242 | Primary camera-compensation candidate |

Variant D uses the visual ego-motion estimator to construct its road-relative
features but does not concatenate the 15 global proxy values. This makes C to
D a controlled comparison with identical dimensions and isolates compensation
instead of confounding it with extra inputs.

## Feasibility And Leakage Gates

Before training:

- prove exact Exp077 sample/frame/label parity for A--D;
- prove all new features use only the 16 observed frames;
- prove no sample is dropped when motion or road geometry is invalid;
- prove duplicate video/frame requests produce byte-identical cached outputs;
- prove masks and transforms resolve by exact video/frame identity;
- prove feature dimensions, names, and ordering match this specification;
- prove train-only imputation statistics are used;
- prove clean-test files are not enumerated by prepare/train/validate commands;
- report train and validation transform-validity rates, point counts, inlier
  fractions, road-boundary validity, and lane-geometry validity;
- require at least 95% valid motion transitions overall on both train and
  validation. If this fails, stop and report rather than tune against labels;
- on a deterministic train/validation-only audit sample, verify that partial
  affine compensation reduces median background reprojection displacement by
  at least 50% relative to no compensation. If this fails, stop and report.

As an auxiliary provenance check, calculate train/validation-only associations
between the same visual proxy and real PIE ego telemetry. This is descriptive
only. Do not tune the estimator to maximize those associations, and do not
claim that the proxy reconstructs physical speed.

## Training And Evaluation

Train A--D freshly through one runner with the exact same cohort, imputation,
XGBoost recipe, seeds, and data ordering. Variant A must reproduce the Exp077
validation result within deterministic numeric tolerance; otherwise stop and
resolve parity before interpreting new variants.

Always report per seed and mean +/- standard deviation:

- accuracy at `0.5`;
- precision at `0.5`;
- recall at `0.5`;
- F1 at `0.5`;
- ROC-AUC;
- PR-AUC;
- Brier score;
- log loss;
- FP count with probability at least `0.9`;
- FN count with probability at most `0.1`.

Produce paired saved-prediction transitions for A to B, A to C, C to D, and A
to D. Do not inspect individual validation errors and redesign the features.

## Primary Validation Gate

Only Variant D is eligible to advance. B and C are explanatory controls.
Freeze the following conditions before training:

1. D minus A mean PR-AUC is positive and the paired PR-AUC delta is positive
   for all three seeds.
2. D mean fixed-0.5 F1 is no worse than A by more than `0.005`.
3. D mean ROC-AUC is no worse than A by more than `0.005`.
4. D mean combined high-confidence FP+FN count does not exceed A.
5. C to D improves mean PR-AUC, establishing benefit from compensation rather
   than ordinary road-relative input alone.

If all conditions pass, freeze D and authorize one clean-test extraction and
evaluation. If any condition fails, write a validation-only negative result
and do not access the clean test.

The clean JAAD test has been evaluated in earlier experiments and is therefore
not historically pristine. Exp081 nonetheless treats it only as the designated
test: its design, feature contract, implementation settings, and advancement
gate are frozen without using its IDs, features, or outcomes.

## Efficiency

Measure separately for a 16-frame window:

- TwinLiteNet+ perception latency;
- sparse optical-flow/affine-estimation latency;
- raw and compensated road-feature construction latency;
- cached-feature XGBoost prediction latency;
- total incremental pipeline latency excluding ViTPose unless explicitly
  measured end to end;
- model size and serialized feature-cache size.

Do not claim real-time end-to-end operation from cached-predictor latency.

## Required Artifacts

The implementation must create, without overwriting prior experiments:

- `README.md`;
- `DESIGN.md` copied from this approved specification;
- `PROVENANCE_AUDIT.md`;
- `scripts/`;
- `metrics/feature_contract.json`;
- `metrics/cohort_parity.json`;
- `metrics/motion_quality.csv`;
- `metrics/road_geometry_quality.csv`;
- `metrics/pie_visual_proxy_telemetry_audit.csv`;
- `metrics/validation_per_seed_results.csv`;
- `metrics/validation_summary_results.csv`;
- `metrics/paired_deltas.csv`;
- `metrics/error_transitions_fixed_0p5.csv`;
- `metrics/validation_gate.json`;
- `metrics/efficiency.csv`;
- `predictions/validation_*.csv`;
- persistent logs and explicit completion markers.

Final-test predictions and metrics may exist only if the frozen validation gate
passes.

## Interpretation Boundaries

Safe positive wording:

> After estimating global camera motion from observed background features,
> expressing pedestrian motion relative to the detected road improved JAAD
> crossing-intention prediction over the same road representation using raw
> image-space motion.

Do not claim:

- recovery of physical ego speed or acceleration;
- that road context always helps;
- that optical flow replaces vehicle telemetry in a deployed system;
- untouched-test confirmation;
- cross-dataset generalization before it is directly tested.

For a real vehicle, measured ego telemetry remains the preferred source. The
visual estimator is a JAAD-compatible fallback and a portable camera-only
representation. A later experiment may define how measured telemetry and the
visual proxy coexist, but Exp081 must not add that hypothesis.

## References

- OpenCV Lucas-Kanade optical flow documentation:
  <https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html>
- OpenCV robust partial-affine estimation documentation:
  <https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html>
- Che et al., *TwinLiteNet+: An Enhanced Multi-Task Segmentation Model for
  Autonomous Driving*, local copy: `papers/Che_2025_TwinLiteNetPlus.pdf`.
- Exp063 final report:
  `experiments/063_pie_corrected_yolop_road_relative_context/FINAL_REPORT.md`.
- Exp074 final report:
  `experiments/074_pie_road_relative_yolop_xgb_extension/FINAL_REPORT.md`.
- Exp075 final report:
  `experiments/075_pie_twinlitenetplus_road_context/FINAL_REPORT.md`.
- Exp077 JAAD clean benchmark lineage:
  `experiments/077_jaad_two_stage_early_late_xgb_transfer/`.
