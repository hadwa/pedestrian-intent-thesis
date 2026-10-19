# YOLOP Provenance And Historical-Postprocessing Audit

## Pinned Perception Lineage

| Artifact | Path | SHA-256 |
|---|---|---|
| checkpoint | `/home/abdelrahman/Hadwa/MSLSTM-PID/YOLOP/weights/End-to-end.pth` | `620e7495328cfc8e04807c14cbb8f8df9916953359d190cbf494b2d049f03b0c` |
| historical extraction notebook | `/home/abdelrahman/Hadwa/MSLSTM-PID/Offline Extraction/yolop-extraction.ipynb` | `2310285c26a88e9acc5aa019e1d301e9435b58bd8b69e7c2d9292951575f8b2f` |
| official local demo | `/home/abdelrahman/Hadwa/MSLSTM-PID/YOLOP/tools/demo.py` | `3a5a2faad6799589793b05fe59a7ecc9406f203af7bedc78cfc5ee1f5ba4971a` |
| local dataset label implementation | `/home/abdelrahman/Hadwa/MSLSTM-PID/YOLOP/lib/dataset/AutoDriveDataset.py` | `9f9f520fe36e7349a36780c351b486a4b6484502524575ae435ac250dbbec8fc` |
| paper | `papers/Wu_2021_YOLOP.pdf` | repository-preserved paper artifact |

The local YOLOP Git lineage is pinned to commit
`8d8f68df318c71f01d6f813c024df646c7d1978f`.

## Proven Historical Discrepancy

The historical notebook resizes the segmentation tensors directly to original
image size, then serializes `output[0][0].sigmoid() * 255`. This selects channel
zero and retains letterbox-padding influence. It does not apply a class-channel
argmax.

The official local demo first removes the recorded letterbox padding from both
two-channel segmentation tensors, resizes the cropped tensors, and applies
`torch.max(..., 1)` to select the predicted class. The dataset implementation
constructs segmentation targets as background/foreground channel pairs, and
the visualization treats class value one as foreground.

Therefore the historical cache is not an official foreground drivable/lane
mask cache. Its high, nearly static values are consistent with channel-zero
background activations. Exp063 does not overwrite it: H retains it as the
historical reference, while G and R use a new corrected extraction.

## Controlled Interpretation

H -> G isolates mask postprocessing because G retains the historical 9+9+2
feature layout and count definition. G -> R isolates representation because
both are composed from the same corrected per-frame inference output. R uses
the tracked pedestrian BBox for target point, spatial scale, and motion, so it
must be described as YOLOP+BBox-derived relational context.

## Development Boundary

The corrected extraction reads only authenticated train/validation windows
from set01, set02, set04, set05, and set06. It does not accept target labels as
an input and it rejects any path or row containing set03.

The completed extractor authenticated and processed:

- 34 train/validation videos;
- 31,131 unique source frames;
- 46,210 exact pedestrian/frame BBox records;
- 5,985 complete 16-frame windows;
- zero dropped or substituted samples;
- zero resumed parts on the original run;
- zero set03 access.

The pinned checkpoint reported 7,940,846 parameters. CUDA extraction used
batch size 8 and completed in 840.31 seconds on an NVIDIA GeForce RTX 3070
Laptop GPU. The authoritative per-video source paths, video/annotation hashes,
part hashes, counts, and runtime are stored in
`metrics/extraction_manifest.json`. Full-resolution masks were deliberately not
persisted; both 20-D representations were composed from the same in-memory
corrected masks and saved together in each atomic video part.
