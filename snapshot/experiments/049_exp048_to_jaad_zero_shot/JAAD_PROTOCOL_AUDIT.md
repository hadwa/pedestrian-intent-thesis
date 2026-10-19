# JAAD Protocol Audit

Status: complete. Test labels were sealed through feature preparation and the
pre-inference audit, then attached once for frozen final evaluation.

## Frozen Mapping

- Official source: `split_ids/default/{train,val,test}.txt`.
- Primary test cohort: official `default/test.txt`; test target labels remain
  sealed until the final authenticated evaluation stage.
- Unit of analysis: every behavior-annotated pedestrian track region containing
  31 consecutive raw frame IDs.
- Observation: the first 30 frames of each region.
- Target: the next raw frame, exactly one frame after the observation end.
- Target label: that target frame's binary `behavior.cross` value.
- Explicitly forbidden fallback: pedestrian-level `attributes.crossing`.
- Gapped regions are rejected rather than interpolated or substituted.

## Train/Validation Evidence

Command:

```bash
/home/abdelrahman/miniconda3/envs/biu-pid-distill/bin/python \
  experiments/049_exp048_to_jaad_zero_shot/scripts/run_experiment.py \
  audit-protocol --splits train val
```

| Split | Official videos | Videos with windows | Behavior tracks | Windows | Negative targets | Positive targets |
|---|---:|---:|---:|---:|---:|---:|
| train | 177 | 163 | 315 | 51,946 | 19,799 | 32,147 |
| val | 29 | 27 | 47 | 8,127 | 3,325 | 4,802 |

Both splits had zero continuity failures, temporal-contract failures,
duplicate sequence IDs, missing target behaviors, and nonbinary target
behaviors. The machine-readable output is
`metrics/protocol_audit_trainval.json` (generated locally and not required in
Git). The command accepts only `train` and `val`; it reported
`official_test_split_labels_parsed=false`.

## Final Test Evidence

After `AUDIT_COMPLETE` authenticated the unlabeled cohort, exact frame-level
target labels were attached. All 44,646 windows had binary labels; none were
excluded. The cohort contains 27,031 positive and 17,615 negative targets over
273 tracks and 111 videos. Independent saved-prediction verification confirmed
that all 18 runs have identical sequence IDs, ordering, target frames, labels,
and threshold 0.5.
