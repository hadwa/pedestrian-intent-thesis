# Source preflight

## Result

Blocked before expensive recovery or extraction. The official archive central
directory was read using the existing validated RemoteZip helper and ETag
`"3868cb96ae54e00a99ad64d9e17bc701-325"`. Directory inspection transferred
694,159 bytes per check; it did not download any MP4 or annotation payload.

- Selected split: 108 train videos, 35 validation videos, matching Exp113.
- Exp113 populations: 26,880 train and 14,922 validation windows.
- PSI2 CV annotation members: 140/143 selected videos.
- Missing PSI2 CV members: `video_0032`, `video_0055`, `video_0093`.
- Affected train windows: 143, 141, and 301 respectively; total 585.
- Affected validation windows: 0.
- PSI2 cognitive files have only `video_name` and `pedestrians` at top level and
  contain no speed/GPS key in the pedestrian records.
- PSI1 has CV members for `video_0032` and `video_0055`, but no CV member for
  `video_0093`. PSI1 annotations are not silently substituted for PSI2.

Because the complete source-only S187 arm requires five summaries of official
frame speed, the missing source prevents both matched arms from being built on
the full frozen cohort. This is a source-data blocker, not a negative result.

## Avoided cost and feasibility

The live ZIP directory contains all 143 needed TrainVal MP4 members. Their exact
combined size is 4,543,821,150 bytes uncompressed and 4,544,559,755 compressed.
The 140 PSI2 CV members total 3,864,596,582 bytes uncompressed and 73,707,702
compressed. None was downloaded for Exp114.

An escalated host check found PyTorch 2.11.0+cu128, CUDA available, one NVIDIA
GeForce RTX 3070 Laptop GPU, compute capability 8.6. Thus GPU availability is
not the blocker. The Exp075 checkpoint remains at
`experiments/075_pie_twinlitenetplus_road_context/checkpoints/large.pth`; no
inference was launched.

## Required resolution

Proceed only if an official, frame-aligned numeric speed source for
`video_0093` and all other required PSI2 frames is provided and its release,
frame identity, units, and full observed-frame coverage are verified. After
that, rerun the preflight and freeze the resulting source hashes before any
download, extraction, feature preparation, or model fit.
