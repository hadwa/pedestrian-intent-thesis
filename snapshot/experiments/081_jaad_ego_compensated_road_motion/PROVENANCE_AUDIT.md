# Exp081 Provenance Audit

## Benchmark lineage

- Cohort: clean JAAD benchmark inherited from Exp077.
- Observation: 16 frames.
- Time to event: 30--60 frames.
- Threshold: fixed 0.5.
- Development cohorts: 1,552 training windows and 274 validation windows.
- Test discipline: the designated test cohort is sealed until the frozen
  validation gate is authenticated.

The prepared Exp077 training and validation tables are reused byte-for-byte.
Their hashes are pinned in `pre_registration.json`. The underlying archive
split CSVs are read-only sources and are also pinned there. The test split is
intentionally not opened or hashed during development.

The Exp012 feature-definition CSV that defines Exp077's 182-D portable ABI and
Exp077's saved one-stage validation table are pinned separately. Fresh Variant
A must reproduce those saved per-seed validation metrics within `1e-12` before
the Exp081 gate is interpreted.

## Model lineage

Variant A is the exact one-stage 182-D Exp077 portable XGBoost anchor, not the
Exp077 two-stage diagnostic. It uses 250 trees and the frozen hyperparameters
recorded in `pre_registration.json`. Variants B--D change only the declared
additional feature block.

## Road-perception lineage

TwinLiteNet+ Large code, adapter, configuration, checkpoint, and paper are
pinned by SHA-256 in `pre_registration.json`. The checkpoint SHA-256 is:

`5605d1a8ce762c1b6f7d23f4c2e7ec9384bfd6e6887b040b43e1f8902a817e22`

No full-resolution segmentation masks will be persisted. Only compact derived
features and audit summaries are allowed as experiment outputs.

The exact Exp063 channel ABI is authenticated from its contracts and
representation modules. An initial pre-results draft used descriptive channel
labels in the JSON record; this was corrected before any real extraction or
validation metric, and a source-equality test now prevents recurrence.

## Visual-motion interpretation

The background affine estimate is an image-space camera-motion proxy. It is
not a measurement of physical ego speed or acceleration. Pedestrian regions
are masked before estimating motion. Later observations are mapped into the
first observation frame before compensated road-relative features are formed.

A separate descriptive audit uses only frozen PIE train/validation samples
from sets01/02/04/05/06 to compare proxy values with recorded OBD summaries.
It excludes set03, fits no mapping, and cannot alter the JAAD experiment.

## Exposure limitation

JAAD has been used in earlier repository work. Exp081 nevertheless keeps its
designated test cohort sealed during development: no test identifiers,
features, labels, predictions, or metrics may be accessed until the frozen
validation gate passes. Any eventual test result must retain the repository's
historical-exposure caveat.
