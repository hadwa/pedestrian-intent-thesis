# Superseded CPU Partial Run

This directory preserves the incomplete Variant A, seed 42 state and log from the
initial sandboxed launch. The Codex sandbox hid WSL's `/dev/dxg`, so PyTorch used
CPU and completed epoch 1 before the device-isolation issue was identified.

This state is excluded from Exp047. Variant A, seed 42 is restarted from epoch 1
on CUDA so all nine frozen A/B/C runs use the same execution device path. No
scientific setting, cohort, training rule, or model definition was changed.
