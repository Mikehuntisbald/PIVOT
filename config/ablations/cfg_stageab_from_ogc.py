from config.cfg_patch_stage_b import *  # noqa: F401,F403

# Joint Stage A+B ablation from the original GroundingDINO OGC checkpoint.
# Keep the Stage B local-TN/text objective and current inference fusion policy,
# but train the patch scorer from scratch instead of expecting a Stage A checkpoint.

lambda_patch = 1.0
lambda_text = 0.25

# Current Stage A+B fusion policy used by Stage B post-processing.
stage_b_infer_text_beta = 1.0
stage_b_infer_canonical_weight = 0.15
stage_b_infer_text_agg = "mean"
stage_b_infer_softmin_tau = 0.7
stage_b_infer_sigmoid_scores = False

# OGC has no trained patch branch. Match Stage A's decoder adaptation while also
# training the Stage B text projection/head.
unfreeze_decoder_last_n_layers = 3
freeze_keywords = [
    "backbone",
    "transformer",
    "bbox_embed",
    "bert",
]
only_train_keywords = None

# Align the decoder-open AB run with Stage A's box stabilization.
bbox_loss_coef = 5.0
giou_loss_coef = 2.0
