# Exp083 Label And Cohort Audit

## Cohorts

| Cohort | Split | Windows | Tracks | Positive | Negative | Positive rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| JAAD_beh | train | 2,134 | 194 | 1,760 | 374 | 0.824743 |
| JAAD_beh | validation | 242 | 22 | 176 | 66 | 0.727273 |
| JAAD_beh | test | 1,881 | 171 | 1,177 | 704 | 0.625731 |
| JAAD_all | train | 8,613 | 783 | 1,760 | 6,853 | 0.204342 |
| JAAD_all | validation | 1,265 | 115 | 176 | 1,089 | 0.139130 |
| JAAD_all | test | 6,732 | 612 | 1,177 | 5,555 | 0.174837 |

The generated track counts exactly match the PCPA benchmark's published
JAAD_beh and JAAD_all split counts. Exact generated-window parity with PCAP-SL
cannot be proven because its window IDs/counts are not published.

## Label Sources

JAAD_beh contains only behavior-annotated tracks. Its labels come from the
official crossing-action construction. JAAD_all contains the same behavior
population plus official non-behavior tracks assigned the non-crossing class
by the upstream generator:

| Split | Additional non-behavior negative windows | Tracks |
| --- | ---: | ---: |
| train | 6,479 | 589 |
| validation | 1,023 | 93 |
| test | 4,851 | 441 |

These broader JAAD_all negatives generally lack the detailed frame-level
behavior attributes available for JAAD_beh. No label was manually changed,
and no sample was selected or removed using predictions.

## Endpoint Caveat

The official API-style population is not the repository's clean verified-onset
cohort. The audit records 1,441 JAAD_beh test windows and 6,292 JAAD_all test
windows whose nominal endpoint is not a verified frame-level crossing onset;
610 test windows contain observed crossing frames. These facts are preserved
for protocol alignment and must accompany any comparison with the clean
event-aligned JAAD benchmark.

Train, validation, and test track memberships are disjoint, sample IDs are
unique, and the only nonconsecutive raw-frame observations are the two approved
JAAD_all training windows documented in `PROTOCOL_AMENDMENT.md`.
