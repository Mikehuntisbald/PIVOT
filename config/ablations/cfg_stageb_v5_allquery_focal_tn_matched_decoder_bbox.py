from config.cfg_patch_stage_b import *  # noqa: F401,F403

# Stage-B v5: start from Stage-B v2 checkpoint0003.
#
# Text objective:
# - positive/refexp rows use original GroundingDINO-like all-query sigmoid focal
# - TN rows stay on the v2 matched-query BCE negative-token objective
# - non-matched queries with IoU > stage_b_extra_iou_match_thr are supervised as
#   extra positives for the corresponding slot
#
# Optimization scope follows the decoder+bbox patch-frozen probe:
# train all detector decoder layers plus bbox/class/feat-map heads, while the
# patch branch, visual encoder/backbone/input projection, and BERT stay frozen.

stage_b_text_loss_type = "allquery_focal_tn_matched_bce"
stage_b_text_focal_alpha = 0.25
stage_b_text_focal_gamma = 2.0
stage_b_extra_iou_match_thr = 0.5

# Keep v2's text loss weight so TN matched BCE is not diluted like v4. The
# dense focal term is normalized by supervised positive queries rather than by
# all valid query-token entries.
lambda_text = 0.25
lambda_patch = 1.0

stage_b_enable_phrase_rank = False
stage_b_rank_loss_coef = 0.0
stage_b_score_calib_loss_coef = 0.0

# Open localization losses at the same caliber as the decoder+bbox probe.
bbox_loss_coef = 5.0
giou_loss_coef = 2.0

only_train_keywords = [
    "feat_map",
    "class_embed",
    "bbox_embed",
    "transformer.decoder.layers",
]
freeze_keywords = [
    "backbone",
    "input_proj",
    "transformer",
    "patch_encoder",
    "query_proj_for_patch",
    "patch_logit_scale",
    "patch_dn_tgt",
    "bert",
]
unfreeze_decoder_last_n_layers = 0

# Keep LR flat for the first probe from v2 ckpt0003.
epochs = 1
lr_drop = 100
save_checkpoint_interval = 1
