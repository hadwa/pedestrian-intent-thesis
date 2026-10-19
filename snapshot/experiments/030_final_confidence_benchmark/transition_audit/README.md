# Final Confidence Transition Audit

Read-only post-hoc audit over saved Experiment 030 prediction artifacts.

Primary scope:
- PIE `30_60`
- JAAD `30_60`
- JAAD `5_15`

Secondary scope:
- JAAD `15_30`

Outputs:
- `sample_level/*.csv` per dataset/horizon/comparison/threshold regime/seed
- `transition_count_summary.csv`
- `transition_distribution_summary.csv`
- `error_accounting_summary.csv`

Primary validation-selected error accounting:

- PRIMARY PIE 30_60 KP-XY -> KP-XYC (seed 42): baseline_errors=897, corrected=189, new_errors=128, net_corrected=61, correction_rate=0.2107, error_introduction_rate=0.0447, net_error_change=-61
- PRIMARY PIE 30_60 KP-XY -> KP-XYC (seed 43): baseline_errors=899, corrected=179, new_errors=111, net_corrected=68, correction_rate=0.1991, error_introduction_rate=0.0388, net_error_change=-68
- PRIMARY PIE 30_60 KP-XY -> KP-XYC (seed 44): baseline_errors=894, corrected=316, new_errors=194, net_corrected=122, correction_rate=0.3535, error_introduction_rate=0.0677, net_error_change=-122
- PRIMARY PIE 30_60 BIU-v2 -> BIU-v2 + BIU-specific reliability (seed 42): baseline_errors=362, corrected=51, new_errors=42, net_corrected=9, correction_rate=0.1409, error_introduction_rate=0.0124, net_error_change=-9
- PRIMARY PIE 30_60 BIU-v2 -> BIU-v2 + BIU-specific reliability (seed 43): baseline_errors=382, corrected=69, new_errors=45, net_corrected=24, correction_rate=0.1806, error_introduction_rate=0.0133, net_error_change=-24
- PRIMARY PIE 30_60 BIU-v2 -> BIU-v2 + BIU-specific reliability (seed 44): baseline_errors=370, corrected=49, new_errors=48, net_corrected=1, correction_rate=0.1324, error_introduction_rate=0.0142, net_error_change=-1
- PRIMARY PIE 30_60 BBox + BIU-v2 -> BBox + BIU-v2 + reliability (seed 42): baseline_errors=298, corrected=45, new_errors=98, net_corrected=-53, correction_rate=0.1510, error_introduction_rate=0.0283, net_error_change=53
- PRIMARY PIE 30_60 BBox + BIU-v2 -> BBox + BIU-v2 + reliability (seed 43): baseline_errors=288, corrected=47, new_errors=83, net_corrected=-36, correction_rate=0.1632, error_introduction_rate=0.0239, net_error_change=36
- PRIMARY PIE 30_60 BBox + BIU-v2 -> BBox + BIU-v2 + reliability (seed 44): baseline_errors=323, corrected=45, new_errors=21, net_corrected=24, correction_rate=0.1393, error_introduction_rate=0.0061, net_error_change=-24
- PRIMARY JAAD 30_60 KP-XY -> KP-XYC (seed 42): baseline_errors=584, corrected=182, new_errors=278, net_corrected=-96, correction_rate=0.3116, error_introduction_rate=0.3032, net_error_change=96
- PRIMARY JAAD 30_60 KP-XY -> KP-XYC (seed 43): baseline_errors=584, corrected=180, new_errors=277, net_corrected=-97, correction_rate=0.3082, error_introduction_rate=0.3021, net_error_change=97
- PRIMARY JAAD 30_60 KP-XY -> KP-XYC (seed 44): baseline_errors=584, corrected=179, new_errors=289, net_corrected=-110, correction_rate=0.3065, error_introduction_rate=0.3152, net_error_change=110
- PRIMARY JAAD 30_60 BIU-v2 -> BIU-v2 + BIU-specific reliability (seed 42): baseline_errors=722, corrected=23, new_errors=19, net_corrected=4, correction_rate=0.0319, error_introduction_rate=0.0244, net_error_change=-4
- PRIMARY JAAD 30_60 BIU-v2 -> BIU-v2 + BIU-specific reliability (seed 43): baseline_errors=722, corrected=23, new_errors=19, net_corrected=4, correction_rate=0.0319, error_introduction_rate=0.0244, net_error_change=-4
- PRIMARY JAAD 30_60 BIU-v2 -> BIU-v2 + BIU-specific reliability (seed 44): baseline_errors=722, corrected=23, new_errors=19, net_corrected=4, correction_rate=0.0319, error_introduction_rate=0.0244, net_error_change=-4
- PRIMARY JAAD 30_60 BBox + BIU-v2 -> BBox + BIU-v2 + reliability (seed 42): baseline_errors=543, corrected=51, new_errors=228, net_corrected=-177, correction_rate=0.0939, error_introduction_rate=0.2380, net_error_change=177
- PRIMARY JAAD 30_60 BBox + BIU-v2 -> BBox + BIU-v2 + reliability (seed 43): baseline_errors=579, corrected=141, new_errors=90, net_corrected=51, correction_rate=0.2435, error_introduction_rate=0.0976, net_error_change=-51
- PRIMARY JAAD 30_60 BBox + BIU-v2 -> BBox + BIU-v2 + reliability (seed 44): baseline_errors=579, corrected=136, new_errors=95, net_corrected=41, correction_rate=0.2349, error_introduction_rate=0.1030, net_error_change=-41
- PRIMARY JAAD 5_15 KP-XY -> KP-XYC (seed 42): baseline_errors=182, corrected=26, new_errors=94, net_corrected=-68, correction_rate=0.1429, error_introduction_rate=0.1808, net_error_change=68
- PRIMARY JAAD 5_15 KP-XY -> KP-XYC (seed 43): baseline_errors=223, corrected=47, new_errors=59, net_corrected=-12, correction_rate=0.2108, error_introduction_rate=0.1232, net_error_change=12
- PRIMARY JAAD 5_15 KP-XY -> KP-XYC (seed 44): baseline_errors=182, corrected=26, new_errors=97, net_corrected=-71, correction_rate=0.1429, error_introduction_rate=0.1865, net_error_change=71
- PRIMARY JAAD 5_15 BIU-v2 -> BIU-v2 + BIU-specific reliability (seed 42): baseline_errors=96, corrected=11, new_errors=17, net_corrected=-6, correction_rate=0.1146, error_introduction_rate=0.0281, net_error_change=6
- PRIMARY JAAD 5_15 BIU-v2 -> BIU-v2 + BIU-specific reliability (seed 43): baseline_errors=90, corrected=6, new_errors=20, net_corrected=-14, correction_rate=0.0667, error_introduction_rate=0.0327, net_error_change=14
- PRIMARY JAAD 5_15 BIU-v2 -> BIU-v2 + BIU-specific reliability (seed 44): baseline_errors=92, corrected=4, new_errors=17, net_corrected=-13, correction_rate=0.0435, error_introduction_rate=0.0279, net_error_change=13
- PRIMARY JAAD 5_15 BBox + BIU-v2 -> BBox + BIU-v2 + reliability (seed 42): baseline_errors=100, corrected=32, new_errors=24, net_corrected=8, correction_rate=0.3200, error_introduction_rate=0.0399, net_error_change=-8
- PRIMARY JAAD 5_15 BBox + BIU-v2 -> BBox + BIU-v2 + reliability (seed 43): baseline_errors=104, corrected=8, new_errors=48, net_corrected=-40, correction_rate=0.0769, error_introduction_rate=0.0803, net_error_change=40
- PRIMARY JAAD 5_15 BBox + BIU-v2 -> BBox + BIU-v2 + reliability (seed 44): baseline_errors=115, corrected=50, new_errors=75, net_corrected=-25, correction_rate=0.4348, error_introduction_rate=0.1278, net_error_change=25
- SECONDARY JAAD 15_30 KP-XY -> KP-XYC (seed 42): baseline_errors=42, corrected=14, new_errors=30, net_corrected=-16, correction_rate=0.3333, error_introduction_rate=0.3571, net_error_change=16
- SECONDARY JAAD 15_30 KP-XY -> KP-XYC (seed 43): baseline_errors=42, corrected=14, new_errors=30, net_corrected=-16, correction_rate=0.3333, error_introduction_rate=0.3571, net_error_change=16
- SECONDARY JAAD 15_30 KP-XY -> KP-XYC (seed 44): baseline_errors=42, corrected=14, new_errors=30, net_corrected=-16, correction_rate=0.3333, error_introduction_rate=0.3571, net_error_change=16
- SECONDARY JAAD 15_30 BIU-v2 -> BIU-v2 + BIU-specific reliability (seed 42): baseline_errors=49, corrected=0, new_errors=5, net_corrected=-5, correction_rate=0.0000, error_introduction_rate=0.0649, net_error_change=5
- SECONDARY JAAD 15_30 BIU-v2 -> BIU-v2 + BIU-specific reliability (seed 43): baseline_errors=37, corrected=3, new_errors=20, net_corrected=-17, correction_rate=0.0811, error_introduction_rate=0.2247, net_error_change=17
- SECONDARY JAAD 15_30 BIU-v2 -> BIU-v2 + BIU-specific reliability (seed 44): baseline_errors=36, corrected=6, new_errors=7, net_corrected=-1, correction_rate=0.1667, error_introduction_rate=0.0778, net_error_change=1
- SECONDARY JAAD 15_30 BBox + BIU-v2 -> BBox + BIU-v2 + reliability (seed 42): baseline_errors=31, corrected=4, new_errors=8, net_corrected=-4, correction_rate=0.1290, error_introduction_rate=0.0842, net_error_change=4
- SECONDARY JAAD 15_30 BBox + BIU-v2 -> BBox + BIU-v2 + reliability (seed 43): baseline_errors=34, corrected=7, new_errors=8, net_corrected=-1, correction_rate=0.2059, error_introduction_rate=0.0870, net_error_change=1
- SECONDARY JAAD 15_30 BBox + BIU-v2 -> BBox + BIU-v2 + reliability (seed 44): baseline_errors=31, corrected=4, new_errors=8, net_corrected=-4, correction_rate=0.1290, error_introduction_rate=0.0842, net_error_change=4

Transition-family summaries include counts and percentages of the relevant baseline population.
Distribution summaries include primary reliability, conservative reliability, probability shift, and optional baseline/augmented probability summaries.
Rate summaries emphasize mean ± SD across seeds; per-seed integer counts remain in `error_accounting_summary.csv`.
