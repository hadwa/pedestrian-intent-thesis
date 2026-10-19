# Exp083 Protocol Amendment: Raw-Frame Continuity

## Timing

Approved before feature construction, training, validation, or test
evaluation. The discovery came from the official cohort-construction audit.

## Evidence

The official JAAD/PCPA sequence generator produces exactly two
`JAAD_all/train` windows from track `video_0098:0_98_544` whose 16 consecutive
track observations span one 134-frame annotation gap. All `JAAD_beh` windows
and all `JAAD_all` validation/test windows are raw-frame consecutive.

## Frozen Resolution

Preserve both official training windows for protocol alignment. Interpret the
observation contract as 16 consecutive official track observations, audit raw
frame continuity explicitly, and never hide or impute the gap. Every manifest
records `raw_frame_consecutive` and `max_raw_frame_gap`. The real audit must
fail if the nonconsecutive population differs from exactly these two windows
on the documented track.

Dropping the windows would alter the official training population. This
amendment changes no labels, model features, splits, thresholds, validation
rules, or test population.
