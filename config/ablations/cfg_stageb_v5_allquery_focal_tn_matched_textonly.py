from config.cfg_patch_stage_b import *  # noqa: F401,F403

# Stage-B v5 text-only probe: start from Stage-B v2 checkpoint0003 and keep
# the v5 text objective, but keep the detector decoder and bbox heads frozen.
#
# Text objective:
# - positive/refexp rows use original GroundingDINO-like all-query sigmoid focal
# - TN rows stay on the v2 matched-query BCE negative-token objective
# - non-matched queries with IoU > stage_b_extra_iou_match_thr are supervised as
#   extra positives for the corresponding slot
#
# Optimization scope intentionally matches Stage-B v2: train only the feature
# map/text classification heads. The patch branch, bbox heads, decoder, visual
# encoder/backbone/input projection, and BERT stay frozen.

stage_b_text_loss_type = "allquery_focal_tn_matched_bce"
stage_b_text_focal_alpha = 0.25
stage_b_text_focal_gamma = 2.0
stage_b_extra_iou_match_thr = 0.5

lambda_text = 0.25
lambda_patch = 1.0

stage_b_enable_phrase_rank = False
stage_b_rank_loss_coef = 0.0
stage_b_score_calib_loss_coef = 0.0

# Keep detector localization frozen, same as Stage-B v2.
bbox_loss_coef = 0.0
giou_loss_coef = 0.0

only_train_keywords = [
    "feat_map",
    "class_embed",
]
freeze_keywords = [
    "backbone",
    "transformer",
    "patch_encoder",
    "query_proj_for_patch",
    "patch_logit_scale",
    "bbox_embed",
    "bert",
]
unfreeze_decoder_last_n_layers = 0

epochs = 1
lr_drop = 100
save_checkpoint_interval = 1
