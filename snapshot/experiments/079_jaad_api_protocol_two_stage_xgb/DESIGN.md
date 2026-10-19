# Exp079 Design

## Question

How do the frozen Exp077 one-stage and 8+8 two-stage XGBoost designs behave
when JAAD is constructed with Exp031's documented reconstruction of the
upstream API-style flawed protocol rather than the clean event-aligned
benchmark?

This is a protocol-sensitivity diagnostic. It does not replace the clean JAAD
result and is not presented as an exact replay of the unrecovered historical
JAAD snapshot.

## Frozen Comparison

The model and feature recipes are unchanged from Exp077:

- 16 ordered observations and nominal TTE 30--60;
- 161-D joint BBox/BIU-v1 representation;
- 21 fixed early/late contrasts, giving 182 full-window features;
- one-stage XGBoost with 250 trees;
- two-stage XGBoost with 125 trees per stage;
- track-grouped five-fold out-of-fold stage-1 probabilities in native training;
- seeds 42, 43, and 44;
- fixed threshold 0.5.

The frozen PIE-to-JAAD models are reused directly from Exp077. Protocol A
native models are retrained because their training population is different.

## Cohorts

Exp031 Protocol A is used exactly:

| Split | Windows |
| --- | ---: |
| Train | 8,613 |
| Validation | 1,265 |
| Test | 6,732 |

The test cohort is materialized only in the evaluation phase. No decision is
changed after evaluation.

## Pose Completion

Existing JAAD caches cover 12,997 of the 69,460 required exact pedestrian-frame
identities. Exp079 extracts only the remaining 56,463 identities using the
same SHA-pinned ViTPose-S checkpoint/configuration lineage used by Exp049.

Protocol A uses `sample_type="all"`, so most missing identities are non-`b`
JAAD people tracks. Their exact API-provided BBoxes are stored in Exp031 and
are used directly. They are not aliased to behavior-pedestrian tracks.

The extractor records model-load, image-decode, CUDA-synchronized pose
inference, write, throughput, and sequential 16-frame estimate statistics.
Pose inference excludes BIU and classifier runtime.

## Preserved Protocol Defects

Two Protocol A training windows contain a gap in raw frame numbers inside the
16 ordered API observations. They are preserved because repairing or dropping
them would change the reconstructed flawed cohort. Exp079 records their count
and feeds the observations in API trajectory order, matching the reconstructed
input semantics.

## Interpretation

Differences from Exp077 can reflect changed population, label semantics,
temporal anchoring, already-crossing/post-event positives, and the two gapped
API trajectories. They are not a causal estimate of label error alone.
