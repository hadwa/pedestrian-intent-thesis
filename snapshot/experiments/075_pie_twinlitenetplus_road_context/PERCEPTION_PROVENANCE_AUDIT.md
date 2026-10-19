# Exp075 Perception Provenance Audit

## Source

- Official repository: `https://github.com/chequanghuy/TwinLiteNetPlus.git`
- Frozen commit: `90f1b8695ae311d5123b05f8534b2e11e42499d2`
- Commit date: 2026-05-05
- Local source: `external/TwinLiteNetPlus`
- License: MIT
- Configuration: Large, with `p=5`, `q=7`, channels
  `[32,64,128,256,512]`
- Official constructor ABI: `TwinLiteNetPlus(SimpleNamespace(config='large'))`

The repository is a pinned submodule and Exp075 does not edit it.

## Checkpoint

- Official pretrained-folder file: `large.pth`
- Google Drive file ID: `1H8P-GrOUBOaVs5LEqXBfz0dC9gguUUio`
- Local path: `experiments/075_pie_twinlitenetplus_road_context/checkpoints/large.pth`
- Size: 7,959,943 bytes
- SHA-256: `5605d1a8ce762c1b6f7d23f4c2e7ec9384bfd6e6887b040b43e1f8902a817e22`
- Container: PyTorch `OrderedDict`
- State entries: 408
- Strict load: all keys matched successfully
- Model parameters: 1,943,911
- Trainable parameters: 1,943,911

The strict-load audit used CPU only and did not run inference. The audited
environment was Python from `cv_env`, PyTorch 2.4.0+cu118, OpenCV 4.11.0, and
NumPy 1.26.4. CUDA was not visible during this audit, so primary extraction has
not started.

## Inference ABI

The pinned `demo.py` and `demoDataset.py` establish BGR OpenCV input,
stride-32 automatic letterboxing to target 640, padding value 114, BGR-to-RGB
CHW conversion, division by 255 without mean/std normalization, output order
drivable-area then lane-line, crop-before-bilinear-resize, and class-1
foreground masks. Exp075 adapts only the inverse resize to target exact original
PIE frame dimensions.

## Paper

- Local path: `papers/Che_2025_TwinLiteNetPlus.pdf`
- Source: `https://arxiv.org/pdf/2403.16958`
- Retrieved artifact: arXiv v6 dated 2026-04-27
- Size: 14,342,482 bytes
- SHA-256: `be71abe6d7aeaac128ac8beda408e39436f7c92bf2761abe6e874505df5c13ff`

The filename retains the predeclared Exp075 name even though the current arXiv
revision is dated 2026.
