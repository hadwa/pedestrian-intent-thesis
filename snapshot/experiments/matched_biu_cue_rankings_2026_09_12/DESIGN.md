# Matched BIU cue rankings

Diagnostic analysis of existing models only. No fitting, feature removal, threshold changes, test-based model selection, or manuscript edits. Archive remains read-only.

Compare versions within six existing settings: PIE Exp096 (v1/core56/historical Combined108), clean JAAD Exp098 (v1/full63/full union135), clean JAAD with vehicle state Exp100 (v1/core56/historical Combined108), API JAAD_beh and JAAD_all Exp102 (v1/core56/full63/full union135), and PSI Exp113 (v1/full63; no matched union exists). Exp096/100 also have version-dependent pose temporal branches, documented separately from static BIUs. PSI's 15-frame observed / frame-16 human-rated intent task remains distinct from PIE/clean-JAAD future crossing.

Use seeds 42/43/44, validation and test separately, saved feature matrices and models; require sequence/label alignment and probability parity. Reuse existing full-union per-feature SHAP arrays for Exp098/102, compute exact XGBoost pred_contribs for other models. No expensive perception or raw pose extraction. Keep full-union v1 and v2 measurements separately named.

For cue c and window i: s_ic = sum of its feature SHAP contributions in log-odds; a_ic = (2*y_i-1)*s_ic. Positive a pushes toward the true label relative to the model's SHAP baseline; negative pushes away. It does not measure accuracy gain or the effect of removal. Importance = sum of mean absolute feature SHAP within cue, normalized over static BIU columns (including quality and full-v2 metadata, clearly labeled); temporal branches remain separate.

Report all-window and per-class toward/away/neutral percentages (epsilon 1e-10), mean positive support, mean opposing magnitude and net alignment. Also report equal-class mean net alignment and equal-track mean net alignment to expose class imbalance and repeated windows. Three-seed means/SD and validation/test sign agreement are descriptive; no independent-window significance claims. Rank by validation importance and validation equal-class net alignment. Test is an explanatory follow-up only; no deletion verdict from test. Store all-window, positive/negative and TP/TN/FP/FN rows; error-conditioned rows are descriptive.

Record exact commands, source SHA-256 hashes, cue membership, parity and additivity checks. Create all outputs here, preserving older results.
