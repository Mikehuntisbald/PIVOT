from config.ablations.cfg_stageb_v5_2_refcoco_patchpos_aux_from_v5_alltn00625 import *  # noqa: F401,F403

# Stage-B v6:
# - keep v5.2's patch CE path, including RefCOCO-family positive-only patch CE
#   and auxiliary decoder losses
# - make the text/detection branch match GDINO Stage-B data FT as closely as
#   possible: original-like all-query sigmoid focal, uniform token weights,
#   TN prompts as empty-det/all-negative text rows, and no extra IoU-positive
#   query expansion

stage_b_text_loss_type = "allquery_focal_tn_empty_det"
stage_b_text_focal_alpha = 0.25
stage_b_text_focal_gamma = 2.0
stage_b_extra_iou_match_thr = 0.0  # inert for this text-loss type; document no extra positives

lambda_patch = 1.0
# StageBCriterion weights text loss with lambda_text; set it to the original
# GDINO loss_ce coefficient and mirror the value in cls_loss_coef for provenance.
lambda_text = 2.0
cls_loss_coef = 2.0
canonical_pos_weight = 1.0
bbox_loss_coef = 5.0
giou_loss_coef = 2.0
aux_loss = True
use_checkpoint = False

stage_b_enable_phrase_rank = False
stage_b_rank_loss_coef = 0.0
stage_b_score_calib_loss_coef = 0.0
stage_b_score_calib_all_tn_neg_weight = 0.0

# Keep TN token-specific BCE knobs inert for the v6 all-query focal path.
lambda_tn_neg = 1.0
lambda_tn_content = 0.0
lambda_tn_canonical = 0.0
tn_content_target = 0.0
tn_canonical_target = 0.0

# Match GDINO FT's broader optimization scope while leaving BERT and explicit
# patch branch parameters frozen. The visual backbone/input projection remain
# trainable by design for this probe.
freeze_keywords = [
    "bert",
    "patch_encoder",
    "query_proj_for_patch",
    "patch_logit_scale",
    "patch_dn_tgt",
]
only_train_keywords = None
unfreeze_decoder_last_n_layers = 0

# Short probe defaults; override from CLI if needed.
batch_size = 4
epochs = 1
lr_drop = 100
save_checkpoint_interval = 1
