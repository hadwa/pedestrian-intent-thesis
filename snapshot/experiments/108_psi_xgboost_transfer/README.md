# Exp108: PSI XGBoost transfer, data recovery gate

Approved scope: separate PSI transfer evaluation using existing compatible XGBoost
models, including a road-aware model. No training or PSI predictions have run.

The number108 was free on2026-09-10. Existing experiments remain unchanged.
The first step is to authenticate PSI release, BBoxes, cognitive labels, frame/track
mapping and official split before freezing a target cohort. Never substitute static
JSONL labels for native frame-level intention without verification. Never fabricate
ego speed. No claim of equivalence to the PIE16-frame/TTE30--60 benchmark.

## Recovery status

Local pose cache: `data/psi_vitpose_s_all_tracks.jsonl` (346 tracks,190 videos).
Previous inventory and SHA are in
`experiments/psi_and_deployment_feasibility_2026_09_10/psi_cache_inventory.json`.
`find /home/abdelrahman/Hadwa -maxdepth 6 -iname '*psi*'` did not recover the
official video/annotation/split directories. This bounded search does not establish
absence from the host or external drives.

The official URL uses HTTP, not HTTPS:
http://pedestriandataset.situated-intent.net/
HTTP succeeded2026-09-10; HTTPS timed out. The page links a dataset access form,
not public annotation archives:
https://docs.google.com/forms/d/e/1FAIpQLSfuzL_3E8pGEU0xI0pnRfX15fGqUgks4XVu2ClPQ8V05oU0Cg/viewform?usp=sf_link
No form submitted and no credentials changed. User asked for existing files/access.

Downloaded official page retained as `official_home_20260910.html`, SHA256
`b05ba7b29d04f3a9c53fb85d3eedc2c9bd16d314aaf3733d7c1aa047c74b9077`.
It also documents COCO-format pose estimations; this does not authenticate the
exported labels or substitute for matching our cache to the official annotations.

The older read-only `pedestrian-intent-biu/psi_jsonl_audit.md` concerns the29-track
export and a10-frame/5-frame-horizon task, not the larger346-track cache or a recovered
native PSI benchmark. Its negative recommendation is not generalized to PSI here.

Required: PSI videos, `cv_annotation`, cognitive annotations, `PSI2_split.json`,
and correspondence to cached `(video,pid,frame_idx)`. Until recovered, evaluation
fails closed. Missing boxes in our JSONL are not a defect of PSI itself.

Candidate compatible saved models exist in Exp081: A182-D, C242-D raw road and
D242-D compensated road. They use no measured ego speed. Source selection for PSI
must be documented before PSI predictions; the Exp109 A/C latency pair is selected
for a controlled road-cost comparison, not on PSI performance. Exp075's247-D saved
PIE predictor still requires ego telemetry and cannot simply accept zeros.

Protocol/PSI-native training is not frozen yet; this folder is an explicitly
incomplete recovery package, not a completed benchmark. See Exp109 for independent
detected-box GPU timing. No10% PedCMT correction is part of these XGBoost models.

Exp109 is now complete; no further training or experiment was launched afterward.
