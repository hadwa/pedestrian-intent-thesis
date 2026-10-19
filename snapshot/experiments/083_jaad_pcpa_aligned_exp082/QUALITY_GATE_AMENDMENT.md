# Exp083 Development Context Quality-Gate Amendment

## Timing And Scope

This amendment was approved before feature materialization, model training,
validation, or test access. It changes no cohort, label, extracted value,
feature, model, hyperparameter, seed, or evaluation rule.

## Observed Defect

The inherited Exp081 quality function sorted each development split by
`sequence_id` and evaluated displacement reduction on only the first 128
windows. On the expanded Exp083 `JAAD_all` validation population, those 128
windows were concentrated in `video_0006`. Their median displacement-reduction
fraction was `-0.1173066455802651`, so this prefix failed even though:

- all 1,265 validation windows had a median of `0.7370826124124015`;
- all 8,613 training windows had a median of `0.7687736218799865`;
- valid-transition coverage was `1.0` on both complete splits.

The failure was therefore caused by a population-dependent lexical-prefix
sample, not by missing transitions or incomplete extraction.

## Approved Correction

Apply the same frozen Exp081 thresholds to each complete development split:

- minimum valid-transition fraction: `0.95`;
- minimum median displacement-reduction fraction: `0.5`.

The former 128-window lexical-prefix result remains stored in
`metrics/development_context_quality_gate.json` as
`legacy_lexical_prefix_diagnostic`. It is not used as the Exp083 pass/fail
decision.

Test features, test labels, test predictions, and test metrics remained sealed
when this correction was approved and implemented.
