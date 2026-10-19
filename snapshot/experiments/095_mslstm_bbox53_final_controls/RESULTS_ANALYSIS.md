# Exp095 Results Analysis

## Scope

This is a retrospective matched-control analysis on PIE set03. All results use
16 observed frames, TTE30--60, overlap 0.6, three seeds, and a fixed 0.5
threshold. G/H/I are authenticated Exp053 references; J/K/L are the fresh
Exp095 controls. The test cohort had prior exposure in the project, so these
results are not pristine confirmation.

## Main Results

| Arm | Representation added to BBox53 + ego + YOLOP | F1 | ROC-AUC | PR-AUC | Parameters |
|---|---|---:|---:|---:|---:|
| G | none | 0.809927 | 0.935398 | 0.853650 | 6,765,321 |
| H | BIU-v1 | 0.822135 | 0.934830 | 0.846777 | 6,840,714 |
| I | BIU-v2 | **0.828921** | 0.938407 | 0.859921 | 6,839,562 |
| J | raw pose | 0.822746 | 0.937325 | 0.854871 | 9,006,091 |
| K | GT Action/Look | 0.827391 | **0.945671** | **0.862852** | 11,111,693 |
| L | BIU-Combined | 0.823058 | 0.936324 | 0.860430 | 6,848,778 |

Every added representation improved fixed-0.5 F1 over G in all three seeds.
The mean improvements were +0.012208 for BIU-v1, +0.018994 for BIU-v2,
+0.012819 for raw pose, +0.017464 for GT Action/Look, and +0.013131 for
BIU-Combined.

## Raw Pose Versus BIUs

Relative to raw pose, BIU-v2 changed mean F1 by +0.006175, ROC-AUC by
+0.001082, and PR-AUC by +0.005050. The F1 direction favored BIU-v2 in two of
three seeds, so the result supports model-specific predictive utility rather
than universal dominance.

BIU-v2 used 6,839,562 total parameters versus 9,006,091 for raw pose. Its added
pathway contained 74,241 parameters above G, versus 2,240,770 for raw pose: a
96.69% reduction in incremental pathway parameters.

## Privileged Action/Look Reference

BIU-v2 had mean F1 0.828921 versus 0.827391 for GT Action/Look, a descriptive
difference of +0.001530. GT Action/Look retained higher ROC-AUC by 0.007264 and
higher PR-AUC by 0.002930. The seed-level F1 ordering was mixed.

The defensible observation is that the automatic BIU-v2 pathway achieved a
similar mean fixed-threshold F1 in this benchmark-aligned architecture while
using fewer parameters. This does not establish semantic equivalence,
statistical non-inferiority, or general replacement of Action/Look.

## BIU-Combined

BIU-Combined did not improve over BIU-v2 in this MSLSTM configuration. Relative
to BIU-v2, it changed mean F1 by -0.005863, ROC-AUC by -0.002083, and PR-AUC by
+0.000509. It was only marginally different from raw pose in mean F1
(+0.000312).

Therefore the union benefit observed in PedCMT is architecture-dependent. The
MSLSTM result supports keeping BIU-v1, BIU-v2, and BIU-Combined as distinct
representations rather than redefining the union as BIU-v2. BIU-v2 is the
strongest MSLSTM BIU arm; BIU-Combined remains the strongest PedCMT arm under
the corresponding PIE experiment.

## Thesis-Safe Conclusions

1. Pose-derived BIUs add predictive value beyond the matched BBox53 core in the
   benchmark-aligned recurrent architecture.
2. BIU-v2 provides the strongest mean fixed-0.5 F1 and is substantially more
   parameter-efficient than the raw-pose pathway.
3. BIU-v2 reaches a similar mean F1 to the GT Action/Look reference, but GT
   Action/Look retains the strongest probability-ranking metrics.
4. Combining v1 and v2 is not universally beneficial; its advantage depends on
   the downstream architecture and fusion behavior.
5. These conclusions are specific to the frozen PIE protocol and do not claim
   exact reproduction of the earlier published MSLSTM result.

Exact per-seed values are in `metrics/all_controlled_results.csv`; paired
deltas are in `metrics/paired_deltas.csv`; predictions are in `predictions/`.
