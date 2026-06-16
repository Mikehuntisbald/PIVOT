from config.cfg_patch_stage_b import *  # noqa: F401,F403

# Probe: keep Stage-B v2 data/text/TN objective, but let the detector decoder
# and box heads adapt with Hungarian bbox/GIoU losses. The patch branch stays
# frozen: support patch encoder, query-to-patch projection, patch temperature,
# patch DN token, shared visual backbone, and transformer encoder are fixed.
#
# Intended checkpoint lineage:
#   outputs/stageB_local_tn_v2_no_phrase_loss/checkpoint0003.pth

# Keep v2 behavior explicit.
stage_b_text_loss_type = "matched_bce"
stage_b_enable_phrase_rank = False
stage_b_rank_loss_coef = 0.0
stage_b_score_calib_loss_coef = 0.0
lambda_patch = 1.0
lambda_text = 0.25

# Open detector localization losses at the same caliber used by Stage A/GDINO.
bbox_loss_coef = 5.0
giou_loss_coef = 2.0

# Stage-B v2 only trained feat_map/class_embed through only_train_keywords.
# This probe keeps an explicit allow-list so bbox heads are definitely trainable
# while the patch/encoder/backbone side stays fixed.
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

# Keep LR flat for the short probe.
epochs = 1
lr_drop = 100
save_checkpoint_interval = 1
