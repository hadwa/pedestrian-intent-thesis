# Exp113 PSI-native BIU comparison

Approved: matched BBox53, BBox53+72-D v1, BBox53+63-D v2, XGBoost trained on PSI.
No road/ego stream, no PSI pose correction, no confidence redesign. Reuse pose.
Training/validation only until frozen models and validation report are reviewed.

Recovered baseline revision: PSI-Intention2022/PSI-Intent-Prediction
63c584b415e9b147c06d7aad9819d0df39feee91.
Unlike the earlier schema-only audit, actual baseline code establishes binary
mean voting: cross=1, not_sure=.5, not_cross=0, mean>=.5 is positive.
Do not drop uncertain judgments. Official extension initially fills missing intent
with not_sure, then forward-fills. Preserve its already-crossed truncation.
main.py sets 15 observations and one target frame, max_track_size=16 for intent.
The defaults .9 train overlap and 1 validation overlap both yield stride 1.
Official update_db_annotations admits contiguous tracks strictly longer than16.

Use TrainVal split for development (108 train/35 val). Both archive split files
agree here. Any test stage must explicitly resolve video_0192 discrepancy; test
annotations are not recovered in this stage. Video_0111 remains excluded because
neither split assigns it. Do not overwrite Exp108's historical audit.

Official source repositories:
https://github.com/PSI-Intention2022/PSI-Intent-Prediction
https://github.com/PSI-Intention2022/PSI-Dataset
Paper already present: papers/Jing_2026_PSI_arxiv_v3.pdf.
