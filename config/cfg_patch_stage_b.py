from config.cfg_patch_stage_a import *  # noqa: F401,F403

# Stage B: keep Stage A patch-only training path, add text-only token supervision.
stage_b = True
patch_only = True
patch_matching = "hungarian"
patch_only_compute_text_logits = True
build_text_token_masks = True

lambda_patch = 1.0
lambda_text = 0.25
canonical_pos_weight = 0.15
attr_pos_weight = 1.0
attr_neg_weight = 1.0

# Freeze everything except the text projection / text head.
unfreeze_decoder_last_n_layers = 0
freeze_keywords = [
    "backbone",
    "transformer",
    "patch_encoder",
    "query_proj_for_patch",
    "patch_logit_scale",
    "bbox_embed",
    "bert",
]
only_train_keywords = [
    "feat_map",
    "class_embed",
]

# Box losses are disabled in Stage B because the corresponding modules stay frozen.
bbox_loss_coef = 0.0
giou_loss_coef = 0.0

# Stage B v1 default: keep captions aligned to real slot phrases / canonical fallbacks.
patch_text_augment = False
text_mask_skip_invalid_canonical = True
text_mask_warn_limit = 100

# Optional drift logging to verify Stage B preserves Stage A patch behavior.
log_stage_b_patch_drift = True
stage_b_patch_drift_interval = 200
stage_b_patch_drift_topk = 50
