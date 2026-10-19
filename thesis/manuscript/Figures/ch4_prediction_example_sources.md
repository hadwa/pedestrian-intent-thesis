# PIE scene — Figure 4.2

Revised 20 October 2026. Two horizontal panels show the prediction at frame 16 and the annotated outcome two seconds later. The first-frame panel was removed at the user’s request to enlarge the images. The user requested a different scene showing the road in front of the ego vehicle, then explicitly requested a correctly predicted crossing. This is a qualitative illustration, not an estimate of performance.

## Sources

- Retained predictor: Exp116 PIE BIU-v2 road-context XGBoost, 195 inputs, seed 42: `experiments/116_xgboost_v2_core_completion/models/pie_road_seed42.joblib`.
- Saved predictions: `experiments/116_xgboost_v2_core_completion/predictions/pie_road_test_seed42.csv`.
- Exact input mapping: `experiments/116_xgboost_v2_core_completion/cache/pie_road/test/{rows.csv,features.npy,columns.json}`.
- Video: `/home/abdelrahman/Hadwa/PIE_Dataset/set03/video_0010.mp4`.
- Annotations: `/home/abdelrahman/Hadwa/MSLSTM-PID/PIE/annotations/annotations/set03/video_0010_annt.xml`.
- Cached estimated poses: `/home/abdelrahman/Hadwa/pedestrian-intent-biu/data/data_keypoints_all.jsonl`.
- Hashes, row identifiers, annotations, skeleton coordinates and probability replay: `ch4_prediction_example_sources.json` beside this document.

## Frame sequence and verification

Track `3_10_630`, sample `set03_video_0010_3_10_630_839_854_884_1`:

| Panel | Annotation frame | Meaning |
|---|---:|---|
| Frame 16 | 854 | Prediction uses all frames 839–854 |
| Future outcome | 914 | 60 frames / two seconds after prediction |

The first crossing onset is frame 884, 30 frames after prediction. All 16 observation frames are verified as not crossing. All displayed boxes are inside the image and unoccluded. Future crossing is verified from the original annotations.

The saved crossing probability is 0.97999007; replay on the exact 195-feature row gives 0.9799900650978088 (agreement within 1e-7). The graphic displays crossing 98.0% and not crossing 2.0%. The fixed saved decision is crossing. All original pose records for the observation are recovered; the final observed skeleton is displayed. No future pose or annotation enters the predictor.

Frame IDs map to decoded offsets `frame - 1`. Every panel uses the same viewport `[0,450,1920,1080]`, retaining the road and ego-vehicle bonnet while removing much of the sky. The raw PNGs preserve full decoded frames. Code draws annotation boxes and estimated skeletons; no image generation or retouching is used.

## Additional pedestrian B

Track `3_10_632` is the pedestrian on the right, shown in orange with ID B. The original annotations mark B as already crossing at frame 854 and still crossing at frame 914. The cached estimated skeleton for frame 854 is verified and displayed. There is no retained anticipatory test row for B ending at frame 854; no model probability is assigned. B's status is explicitly labelled as an annotation. A remains the cyan pedestrian with the verified 98.0% prediction. The script verifies 17 cached pose records in total (A's 16 observation frames and B's displayed frame).

## Selection scope

Candidate scenes were first screened for visibility, pedestrian size and an unobstructed view of crossing the road ahead. After the user's explicit request for a successful scenario, saved predictions were checked to choose a correctly predicted crossing. This is intentionally selected qualitative evidence, not representative sampling. Seed 42 is fixed. No metrics, models, experiment outputs, splits or predictions were modified. The source hashes were unchanged after all reads.

## Commands and outputs

Run from the active repository:

```bash
MPLCONFIGDIR=/tmp/thesis-scene-mpl /home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python thesis/manuscript/Figures/build_ch4_prediction_example.py
MPLCONFIGDIR=/tmp/thesis-scene-mpl /home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python thesis/manuscript/Figures/build_ch4_prediction_example.py --render
TEXMFVAR=/tmp/guc-thesis-texmf-var VARTEXFONTS=/tmp/guc-thesis-fonts latexmk -cd -pdf -interaction=nonstopmode -halt-on-error thesis/manuscript/Thesis.tex
```

Outputs in this figure folder: `pie_prediction_first.png`, `pie_prediction_observed.png`, `pie_prediction_future.png`, `ch4_prediction_example.pdf`, `.png`, `.svg`, `_sources.json`, and the `.tex` wrapper. The manuscript prose and result-source map reflect this frame sequence. No retraining, perception extraction, commit or push.
