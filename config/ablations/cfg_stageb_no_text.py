from config.cfg_patch_stage_b import *  # noqa: F401,F403

lambda_text = 0.0

# Inference also disables text fusion; keep text logits available so PostProcessStageB
# can run unchanged, but make them contribute exactly zero to slot scores.
stage_b_infer_text_beta = 0.0
stage_b_infer_canonical_weight = 0.0
