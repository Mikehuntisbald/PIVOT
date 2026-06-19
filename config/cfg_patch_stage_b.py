from config.cfg_patch_stage_a import *  # noqa: F401,F403

# Stage B v2: keep Stage A patch-only training path, add text-only token
# supervision. Phrase ranking is retained only in ablation configs.
stage_b = True
patch_only = True
patch_matching = "hungarian"
patch_only_compute_text_logits = True
build_text_token_masks = True

lambda_patch = 1.0
lambda_text = 0.25
# Mainline v2 keeps the local matched-query BCE token objective. Ablations can
# set this to "allquery_focal" for original GroundingDINO-like dense token loss.
stage_b_text_loss_type = "matched_bce"
stage_b_text_focal_alpha = 0.25
stage_b_text_focal_gamma = 2.0
stage_b_enable_phrase_rank = False
stage_b_rank_margin = 0.3
stage_b_rank_loss_coef = 0.0
stage_b_rank_detach_patch = True
stage_b_score_calib_loss_coef = 0.0
stage_b_score_calib_tau_pos = 0.10
stage_b_score_calib_tau_neg = 1.40
stage_b_score_calib_margin = 0.30
stage_b_score_calib_topk = 10
stage_b_score_calib_neg_weight = 0.5
stage_b_score_calib_pos_weight = 0.1
stage_b_score_calib_gap_weight = 0.1
stage_b_score_calib_pos_query_weight = 0.1
stage_b_score_calib_detach_patch = True
tn_loss_profile = "standard"
canonical_pos_weight = 0.15

use_tn_category_weights = True
default_tn_category_weight = 1.0
tn_balance_sampling = True
tn_balance_cap = 5.0

# Deprecated/ignored in the content-token Stage-B loss:
# attr_pos_weight, tn_shared_attr_pos_weight, attr_neg_weight,
# use_phrase_tn_loss, phrase_score_type, softmin_tau, lambda_phrase.
# Non-canonical content-positive tokens and TN negative tokens both use weight 1.0.

# Inference only: slot-level fusion keeps model fused_logits untouched.
stage_b_infer_text_beta = 1.0
stage_b_infer_canonical_weight = 0.15
stage_b_infer_text_agg = "mean"
stage_b_infer_softmin_tau = 0.7
stage_b_infer_mean_softmin_alpha = 0.5
stage_b_infer_sigmoid_scores = False

skip_tn_if_neg_overlaps_canonical = True
skip_ambiguous_tn = True
skip_tn_if_changed_span_not_found = True
skip_tn_if_changed_span_empty_after_filter = True
skip_relation_like_tn_in_v1 = False

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

# Keep drift logging off for full training. Use cfg_patch_stage_b_drift_sanity.py
# for a short sanity run when patch-drift diagnostics are needed.
log_stage_b_patch_drift = False
stage_b_patch_drift_interval = 200
stage_b_patch_drift_topk = 50
