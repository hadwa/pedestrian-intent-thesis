# Interpretation of the completed no-noise control

Source: `FINAL_REPORT.md`, `metrics/per_seed_validation.csv` and
`metrics/paired_noise_removal_deltas.csv`. Validation only; no test claim.

Removing noise is not an overall fix. Relative to Exp121 AP-selected models,
mean AP decreases in every arm while mean ROC-AUC increases. Full63 mean F1
increases0.0168, but only one of three paired seeds improves F1; core56 mean F1
decreases0.0128. Thus the measurement perturbation audit did not establish that
removing noise would uniformly help learning, and this experiment does not
support such a conclusion.

Within Exp122, both BIU variants exceed baseline and raw pose on AP/ROC in all
three seeds. F1 advantages are not consistently positive. In particular,
baseline seed43 has AP0.823722 but F1 only0.486056 and recall0.358824 at0.5.
Its low F1 drives much of the mean representation-versus-baseline gap.
Do not advertise that mean gap as a robust resolution of the previous problem.

This demonstrates the opposite operating-point problem to Exp120's overly
positive seed44: selecting on ranking can retain a model with good ranking but
poor fixed-threshold classification. Neither the maximum-F1 nor maximum-AP
selection rule alone has yielded a uniformly satisfactory outcome here. The
secondary F1-selected rows remain diagnostic; they must not silently replace
the predeclared primary AP-selected models.

No threshold search, calibration, additional training, test evaluation or
automatic integration redesign was performed. Preserve this negative/mixed
intervention result and decide the next controlled development step separately.
