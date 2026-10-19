# Exp063 Design Binding

The authoritative design is:

`docs/superpowers/specs/2026-08-26-exp063-corrected-yolop-road-relative-context-design.md`

Implementation and generated artifacts cannot override that specification.
In particular, Exp063 changes only the YOLOP scene representation used by the
Exp062 context corrector. It does not change anchors, BIU definitions, BBox
features, ego inputs, neural architecture, training policy, fixed threshold,
validation gates, or benchmark cohorts.

Development is restricted to PIE set01, set02, set04, set05, and set06. Set03
is inaccessible before an authenticated final freeze. If neither corrected
representation passes every inherited gate, the valid outcome is
validation-only closure.
