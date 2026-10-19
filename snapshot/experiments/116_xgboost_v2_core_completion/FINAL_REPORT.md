# Exp116 final results

Retrospective controlled completion; fixed threshold 0.5. AP = average precision.
All 33 new fits, all parent probability replays, cohort and feature invariants, and saved-metric checks passed.

| Case | Core56 F1 | Parent F1 | Core56 AP | Parent AP | Core56 ROC | Parent ROC |
|---|---:|---:|---:|---:|---:|---:|
| jaad_clean_simple | 0.705147 | 0.710638 | 0.674278 | 0.680854 | 0.707973 | 0.708192 |
| psi_simple | 0.763880 | 0.759333 | 0.862163 | 0.863645 | 0.660432 | 0.660694 |
| pie_road | 0.882893 | 0.879739 | 0.930437 | 0.927432 | 0.973090 | 0.972513 |
| jaad_clean_road | 0.719082 | 0.735290 | 0.678997 | 0.659378 | 0.746164 | 0.750283 |
| jaad_beh_road | 0.771414 | 0.779342 | 0.769711 | 0.746922 | 0.685450 | 0.672254 |
| jaad_all_road | 0.725017 | 0.727548 | 0.811236 | 0.811555 | 0.945269 | 0.945511 |
| jaad_clean_road_vehicle | 0.728248 | 0.739896 | 0.680597 | 0.668882 | 0.752735 | 0.756633 |
| jaad_beh_road_vehicle | 0.775530 | 0.781154 | 0.781860 | 0.748605 | 0.696956 | 0.670970 |
| jaad_all_road_vehicle | 0.732697 | 0.733276 | 0.822673 | 0.820633 | 0.948732 | 0.949021 |
| psi_ego | 0.803998 | 0.801355 | 0.847409 | 0.824576 | 0.660772 | 0.629648 |
| psi_road_ego | 0.824709 | 0.817304 | 0.862200 | 0.849057 | 0.690414 | 0.666407 |

Full per-seed and mean/sample-SD tables: metrics/per_seed_results.csv and metrics/summary_results.csv.
Paired deltas: metrics/paired_deltas.csv; transitions: metrics/error_transitions.csv.
Simple parents use full63. Extended parents use historical combined static BIUs, not standalone full63.
Existing temporal/pose-quality summaries are retained in extended arms. PSI is a different intent task.
No neural reruns, test-driven selection, thesis edits, perception reruns or benchmark overwrites were performed.
Total measured tree fitting time: 25.97 seconds.
