# JAAD qualitative ambiguous-negative casebook

## Scope
- Total rendered cases: 23
- Split: test only
- Horizons covered: `30_60`, `15_30`, `5_15`
- Per-sample archived model scores are available only for JAAD `30_60` BBox and BIU v1 fixed-0.5 predictions.
- BIU v2 per-sample scores were not saved in experiment 009, and this task does not retrain models, so BIU v2 score/prediction fields are marked `NA`.

## What types of ambiguous negatives were selected?
- `30_60` ambiguous negatives that were BIU v1 false positives on the archived fixed-0.5 run.
- `30_60` ambiguous negatives that were BBox false positives, including a disagreement subset where BBox was positive while BIU v1 stayed negative.
- Representative `15_30` and `5_15` ambiguous negatives chosen for strong lateral motion, scale change, and curb / crossing-scene relevance.
- Actual-positive and clean-negative references for contrast at all three horizons.

## Qualitative reading from the rendered sheets
- The ambiguous negatives are visually crossing-relevant, not random easy negatives. Many stand or walk near the curb, close to a crosswalk, or at an intersection edge with clear road-facing orientation.
- In `30_60` BBox-false-positive cases such as `case_007`, the geometry/context cue is obvious: the pedestrian is positioned at the crosswalk edge and grows slightly in scale, which can plausibly fool a BBox-heavy model even though BIU v1 remains negative.
- In `30_60` BIU v1 false-positive cases such as `case_001`, the subject looks crossing-relevant and maintains visible body configuration changes, but the future row suggests delayed or absent commitment rather than immediate crossing onset.
- The short-horizon `5_15` ambiguous negatives such as `case_012` are the most informative manual-inspection targets: they often look very close to actual crossing readiness, with strong curbside positioning and some forward body motion, yet they still remain event-negative.
- The `5_15` actual-positive references such as `case_018` show similar scene context but usually stronger final alignment into the roadway or clearer forward commitment in the last observed frames.

## In what cases does BIU v2 likely fix BIU v1?
- Experiment 009 showed BIU v2 helps most at `5_15`, but per-case BIU v2 scores are unavailable here.
- The best proxy cases are the `5_15` ambiguous negatives and the `5_15` actual-positive references viewed side by side.
- The intended manual question is whether the ambiguous negatives still lack decisive lower-body commitment even when they are already crossing-relevant in context.

## Are BIU v2 fixes visually consistent with “no lower-body commitment”?
- Tentatively yes, but this remains a human-inspection question rather than a scored claim in this casebook.
- Several ambiguous-negative sheets show curbside or intersection readiness without a clearly committed step sequence into the road during the observed 16-frame window.
- That pattern is consistent with the BIU v2 benchmark result: stronger short-horizon behavior modeling helps, but BBox still captures much of the scene-level crossing relevance.

## When BIU v2 is still likely wrong, what seems to confuse it?
- Hard ambiguous negatives often share the same crossing-relevant setup as actual positives: curb proximity, visible forward walking, scale change, and alignment with a crosswalk.
- In those cases the difference may be subtle timing rather than coarse posture, which explains why threshold/selective-prediction analysis is still important even after BIU v2 improves over BIU v1.

## Does BBox appear to rely on scale/approach while BIU v2 relies on body motion?
- The qualitative evidence supports that interpretation.
- The BBox false-positive examples are often driven by crosswalk location, approach, and size change.
- The pose overlays make the next manual question more specific: whether a pedestrian is only crossing-relevant, or whether the lower body is already committing to road entry.

## Which cases should be manually inspected by Hadwa/Abdelrahman?
- `case_001` to `case_003`: archived `30_60` BIU v1 false positives.
- `case_004` to `case_007`: archived `30_60` BBox false positives, especially the BBox-positive / BIU-v1-negative disagreement case.
- `case_012` to `case_015`: `5_15` ambiguous negatives selected for strong short-horizon motion.
- `case_018` and `case_019`: `5_15` actual positives for direct visual comparison.

## Limitations
- BIU v2 per-sample scores/predictions are unavailable because experiment 009 saved only aggregate benchmark CSVs, not per-case prediction files.
- `15_30` and `5_15` model-specific score columns therefore remain `NA` in the casebook CSV.
- This casebook is a qualitative aid, not a scored per-case BIU v2 error analysis.

## Next experiment
- The next experiment should be a staged `BBox/context -> BIU v2 confirmation` model or a threshold/selective-prediction audit using BIU v2 plus BBox.
- Another purely qualitative BIU redesign pass is not the highest-priority next step.
