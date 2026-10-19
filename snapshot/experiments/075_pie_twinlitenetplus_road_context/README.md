# Exp075: TwinLiteNet+ Large Road Context

Exp075 is a controlled perception-source ablation on the PIE 16-frame,
TTE 30--60 benchmark. It preserves Exp072's 187-D anchor, Exp063/074's exact
20-D road-relative equations and 60-D temporal summary, and Exp072's XGBoost
recipe. Only the drivable-area and lane-mask source changes from original
YOLOP to TwinLiteNet+ Large.

The approved design is in `DESIGN.md`. Set03 is gated and remains inaccessible
until the frozen validation rule passes. Primary mask extraction requires CUDA
FP16; CPU execution is limited to provenance and interface smoke tests.

## Perception Source

- Source: <https://github.com/chequanghuy/TwinLiteNetPlus>
- Commit: `90f1b8695ae311d5123b05f8534b2e11e42499d2`
- Configuration: Large
- Official checkpoint file ID: `1H8P-GrOUBOaVs5LEqXBfz0dC9gguUUio`
- Checkpoint SHA-256: `5605d1a8ce762c1b6f7d23f4c2e7ec9384bfd6e6887b040b43e1f8902a817e22`

## Paper

Che, Q.-H., Le, D.-T., Pham, M.-Q., Nguyen, V.-T., and Lam, D.-K.,
"TwinLiteNet+: An Enhanced Multi-Task Segmentation Model for Autonomous
Driving," arXiv:2403.16958, current repository copy v6, 2026.

The exact PDF and provenance note are stored at
`papers/Che_2025_TwinLiteNetPlus.pdf` and
`papers/Che_2025_TwinLiteNetPlus_README.md`.

## Latency Scope

Exp075 reports component latency, cached 247-D predictor latency, and online
incremental latency from one already-decoded 16-frame RGB window plus existing
Exp072 cues through TwinLiteNet+ masks, road-relative features, summaries, and
XGBoost. Frame/video decoding and ViTPose extraction are reported as excluded;
the result is not described as full camera-to-decision latency.
