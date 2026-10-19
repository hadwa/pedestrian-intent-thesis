# Seminar comparison inventory (2026-09-18)

| Dataset/cohort | Family | Baseline | Raw pose | v2 core56 | v2 + auxiliaries63 | Required action |
| --- | --- | --- | --- | --- | --- | --- |
| PIE | Adapted normalized-BBox53 MSLSTM | Exp053 G | Exp095 J | Exp117 mslstm_pie/core56 | Exp053 I | Reuse; no duplicate training |
| PIE | Direct-token PedCMT | Exp084 A | Exp084 B | Exp117 pedcmt_pie/core56 | Exp084 D | Reuse; separate from correction family |
| Clean JAAD | Direct-token PedCMT | Exp089 A | Exp089 B | Exp117 pedcmt_jaad/core56 | Exp089 D | Reuse |
| PIE | Capped correction PedCMT | Frozen Exp084 A | Missing | Missing | Missing | Exp119, nine new runs |
| Clean JAAD | Capped correction PedCMT | Frozen Exp089 A | Exp094 P10 | Missing | Missing | Exp119, six new runs; reuse P10 |
| Clean JAAD | Adapted normalized-BBox53 MSLSTM | Missing | Missing | Missing | Missing | Separate JAAD input contract and scene coverage required |

Exp049 is zero-shot evaluation of PIE-trained Exp048 under 30-frame/next-frame
construction; it cannot supply the missing 16-frame/TTE30--60 JAAD-trained MSLSTM
controls. Exp092/093 capped successes use union135 and cannot be relabelled v2.
Exp117 did not train capped branches. No API_beh/API_all expansion is included.

Exact metric sources for completed comparisons:

- `experiments/117_neural_v2_core_completion/metrics/summary_results.csv`
  (`mslstm_pie`, `pedcmt_pie`, `pedcmt_jaad` cases; explicit core56 and reused arms).
- `experiments/094_pedcmt_jaad_capped_raw_pose_control/metrics/summary_results.csv`.
- Protocols and architecture: Exp053/095, Exp084/089/117 DESIGN.md;
  Exp094 authoritative specification under docs/superpowers/specs.

The user approved categorical vehicle state with no acceleration stream for
JAAD MSLSTM. Exp120 implements this separate adaptation with four fresh arms.
All 6180 development images exist, but the Exp049 test-only scene cache covers
none of their 7844 pedestrian-frame identities; these are extracted with the
original compatible YOLOP environment, without new pose extraction.
