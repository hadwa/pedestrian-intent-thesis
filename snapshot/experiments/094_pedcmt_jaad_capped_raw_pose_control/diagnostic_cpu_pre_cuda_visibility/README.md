# Superseded CPU diagnostic run

These artifacts preserve the first complete Exp094 training pass. The restricted
execution environment reported CUDA unavailable, so seeds 42, 43, and 44 were trained
on CPU. After confirming that CUDA was visible outside that restriction with
`LD_LIBRARY_PATH=/usr/lib/wsl/lib`, the CPU artifacts were moved here intact and the
authoritative run was repeated on the available NVIDIA GPU.

This folder is diagnostic/superseded. Do not quote its predictions or metrics as the
main Exp094 result.
