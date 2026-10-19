# Exp096: PIE Exp072 Matched BIU Ablation

## Question

Within the exact lightweight XGBoost architecture selected by Exp072, which BIU representation
provides the strongest evidence: BIU v1, BIU v2, or their deduplicated union?

## Frozen comparison

All arms reuse the authenticated Exp072 PIE matrices, train/validation/test cohorts, three seeds,
XGBoost recipe, and fixed 0.5 threshold. The shared non-pose block contains BBox-53, the six
BBox-temporal contrasts, and five exact-frame ego-speed summaries. Both pose arms also receive the
three pose-confidence temporal summaries. The v2 arm retains the twelve v2-specific lower-body and
torso temporal contrasts. The combined arm is byte-for-feature identical to Exp072's 187-D
candidate.

| Arm | Feature count | Contents |
|---|---:|---|
| base | 64 | BBox-53 + BBox temporal + ego speed |
| v1 | 139 | base + BIU v1 + pose-quality temporal summaries |
| v2 | 135 | base + BIU v2 + pose-quality and v2 temporal summaries |
| combined | 187 | exact Exp072 joint BIU + temporal + ego candidate |

All twelve validation runs must complete before the runner can open its test source. No arm is
selected or removed at the validation gate. Since PIE set03 was exposed by earlier experiments,
this is explicitly a retrospective matched control rather than independent confirmation.

## Outputs

The experiment reports validation and test metrics per seed, mean/std summaries, paired deltas,
and track-cluster bootstrap intervals over seed-ensemble predictions. The bootstrap covers both
version-to-version comparisons and every BIU arm against the shared base. It does not alter or
overwrite Exp072.
