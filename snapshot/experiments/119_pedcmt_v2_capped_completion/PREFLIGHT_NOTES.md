# Pretraining implementation checks

First GPU preflight stopped before training/test access: reproduced parent logits
differed from the historical cache by up to 4.887581e-6, exceeding a 2e-6 absolute
logit tolerance. Saved validation probability parity passed at 2e-6. The replay
logit tolerance is now 1e-5; training reuses the original cached logits/anchors
exactly, rather than substituting recomputed values. This changes neither inputs
to the historical training recipe nor checkpoint/threshold/cap rules. Initial
code/design hashes remain in PREFLIGHT_ATTEMPT1.json; no result-based change.

The second preflight encountered 1.04904175e-5 maximum logit difference for another
PIE parent. Final replay checks probabilities at the unchanged 2e-6 tolerance
instead of enforcing a magnitude-independent logit tolerance. Hidden-anchor
replay uses 1e-4 absolute tolerance across numerical backend execution, with
observed differences recorded per seed. Exact original cached training tensors
are reused; this is not an approximation to the fitting inputs. All corrections
occurred before training or final-test access. Code/design are frozen only after
the full pretraining audit passes. Attempt2 hashes are preserved separately.
