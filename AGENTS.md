# Publication workspace rules

This folder is independent of the research and archive repositories. Treat both as read-only. Never run copied historical scripts until their resolved inputs and outputs have been audited: absolute paths may point back to research files. All execution outputs and environments must remain inside this publication workspace. Do not retrain expensive models or push without explicit authorization. Preserve experiment IDs, splits, thresholds, model-selection policies and copied evidence. Do not present staging checks as benchmark reproduction. Audit files contain local paths and are staging provenance, not automatically public release content.

## Current author instruction

Do not retrain any model during packaging. Focus on preprocessing instructions, original training details, saved weights, and evaluation. Training commands may be documented for future readers but must not be executed unless the author explicitly changes this instruction. Preserve original research/thesis files.
