# Exp083: JAAD PCPA-Aligned Exp082-D

Fresh controlled training of the frozen Exp082-D design on separate official
PCPA-style `JAAD_beh` and `JAAD_all` cohorts.

Primary protocol: 16 observed frames, TTE 30--60 frames, JAAD sample overlap
0.8, fixed probability threshold 0.5, and seeds 42/43/44.

The experiment is complete. The primary fixed-0.5 means are:

| Cohort | Accuracy | F1 | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: |
| JAAD_beh | 0.678185 | 0.779342 | 0.672254 | 0.746922 |
| JAAD_all | 0.891414 | 0.727548 | 0.945511 | 0.811555 |

See `FINAL_REPORT.md` for per-seed uncertainty, operating-point results,
PCAP-SL comparison, efficiency scope, and protocol limitations. Earlier
experiments remain unchanged.
