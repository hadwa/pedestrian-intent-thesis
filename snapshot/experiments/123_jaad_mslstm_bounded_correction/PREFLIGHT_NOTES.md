# Pretraining implementation correction

The first preflight stopped at the parent probability replay assertion before
any branch training or design freeze. Its capture helper used batches32 across
the combined development cache, unlike Exp122's validation-only batches512.
Capture now uses each split separately with batch512, matching the historical
evaluation path. The original probability tolerance2e-6 was not relaxed.
The persistent log/FAILURE.json retain the initial attempt. Completion markers,
if subsequently produced, establish that the corrected preflight succeeded.
This is an implementation replay correction, not a changed model/cap/design.
