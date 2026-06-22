from config.cfg_patch_stage_b import *  # noqa: F401,F403

# Stage B v7: detached post-candidate predicate verifier.
#
# Candidate generation remains Stage A:
# - canonical class captions only
# - support patch/entity branch
# - patch-only Hungarian matching
#
# Verifier training:
# - full phrase/sentence captions
# - detached query feature + detached ROI feature + predicate text tokens
# - matched-query BCE after matched_iou > stage_b_v7_min_matched_iou

stage_b = False
stage_b_v7 = True
patch_only = True
patch_matching = "hungarian"
build_text_token_masks = True

stage_b_v7_canonical_token_weight = 0.15
stage_b_v7_tn_token_weight = 1.0
stage_b_v7_min_matched_iou = 0.5
stage_b_v7_use_neighbor_geometry = False

# Keep Stage A candidate generation frozen. Only train verifier-local modules;
# verifier BERT is explicitly excluded and is also frozen in main.py.
unfreeze_decoder_last_n_layers = 0
freeze_keywords = [
    "backbone",
    "transformer",
    "patch_encoder",
    "query_proj_for_patch",
    "patch_logit_scale",
    "bbox_embed",
    "class_embed",
    "feat_map",
    "bert",
]
only_train_keywords = ["stage_b_verifier"]
only_train_exclude_keywords = ["stage_b_verifier.bert"]

patch_only_compute_text_logits = False
lambda_patch = 0.0
lambda_text = 0.0
bbox_loss_coef = 0.0
giou_loss_coef = 0.0
aux_loss = False

log_stage_b_patch_drift = False
