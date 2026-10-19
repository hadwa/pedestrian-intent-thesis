# User-requested pause

Annotation recovery stopped with Ctrl-C (exit130) at the user's request.
102 pedestrian_intent.json files exist locally out of143 requested train/val
files. Resume recovery revalidates existing bytes/CRC and downloads only missing
files. No final recovery.json marker exists yet. No training or predictions ran.
Four native-label/sampling/dimension/leakage unit tests passed before pausing.

Resume from active repository with the existing Python environment:

```bash
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python -u experiments/113_psi_native_biu_xgb/recover.py > experiments/113_psi_native_biu_xgb/recovery_resume.log 2>&1
```

Then inspect preparation code, run prepare, check full coverage/invariants, and
only then run the nine fixed train/validation fits. Test remains closed. Source
clones are currently in /tmp/psi-intent-native and /tmp/psi-dataset-native; exact
required source files were already copied into official_code/. Recreate pinned
clones if /tmp is cleared before resuming. This Exp113 work is not yet committed.
