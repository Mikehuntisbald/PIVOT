from config.ablations.cfg_stageb_v15_decoupled_global_confidence import *  # noqa: F401,F403

# Absolute confidence is the learned image-expression validity gate itself.
# Frozen confidence phrase/patch logits remain available only as immutable
# features and candidate-pooling weights; they are never added to the output.
stage_b_v16_confidence_output_mode = "gate_only"
