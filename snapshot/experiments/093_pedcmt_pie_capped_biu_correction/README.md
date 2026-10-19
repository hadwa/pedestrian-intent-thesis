# Exp093: PIE Capped BIU Correction

Exploratory controlled transfer of Exp092's fixed 10% BIU correction to the
Exp084 standardized PIE 16-frame/TTE30--60 PedCMT parent.

No pose extraction or parent retraining is performed. See `DESIGN.md` for the
frozen design and `RESULTS.md` for the final outcome.

The fixed 10% correction improved mean PIE set03 PR-AUC from `0.8584` to
`0.8674` and mean fixed-0.5 F1 from `0.7926` to `0.7944`. Ranking improved in
all three seeds, while F1 improved in two of three seeds. This is a positive
exploratory transfer result, not a new best PIE model.
