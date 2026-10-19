# Exp120: clean-JAAD adapted MSLSTM controls

Approved baseline: BBox53 + categorical vehicle-state sequence + YOLOP;
no acceleration, no measured-speed claim. Add raw pose, core56 or full63 in
three separate arms. Twelve matched fresh trainings, seeds42/43/44. See DESIGN.md.

No equivalent 16-frame JAAD-trained runs were found; Exp049 zero-shot30-frame
results are not reused as this baseline. Pose and BIU inputs reuse Exp089 caches.
YOLOP reuses exact Exp049 records where available; missing records are extracted
with its original mslstm_pid_env and pinned checkpoint. The main predictor runs
in biu-pid-distill. No environment is modified. A GPU one-frame extraction smoke
passed before launch; model parameter smoke counts A/B/C/D are
4,590,087 / 6,830,857 / 4,663,432 / 4,664,328.

```sh
env LD_LIBRARY_PATH=/usr/lib/wsl/lib PYTHONDONTWRITEBYTECODE=1 /home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python experiments/120_jaad_mslstm_v2_completion/scripts/launch.py
tail -n 30 experiments/120_jaad_mslstm_v2_completion/logs/pipeline.log
```

Detached, no tmux; host sleep/shutdown can interrupt. Extraction is cached per
video; completed model seeds are hash-verified and skipped on restart. Persistent
per-epoch logs and per-seed markers precede TRAINING_COMPLETE; test data remains
locked until all12 model checkpoints exist. EXPERIMENT_COMPLETE follows final
evaluation and source-immutability checks. No thesis edits or outcome-based search.
