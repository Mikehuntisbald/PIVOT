from config.ablations.cfg_stageb_v7_post_candidate_verifier import *  # noqa: F401,F403

# Stage B v7 paired SAM3/TN verifier run.
#
# Data side:
# - no COCO/LVIS train images as normal detection supervision
# - no refcoco/refcoco+/refcocog positive-only phrase data
# - only the full SAM3-washed TN jsonl rows with valid try_tn
#
# Loss side:
# - the same matched query sees the positive phrase slot and the TN phrase slot
# - BCE is positive phrase BCE + negative phrase BCE for that matched query
# - add a 5.x-style positive/negative slot separation loss

stage_b_v7_canonical_token_weight = 0.15
stage_b_v7_tn_token_weight = 1.0
stage_b_v7_tn_shared_token_weight = 0.25
stage_b_v7_tn_length_reweight = False

stage_b_v7_candidate_residual_init = True
stage_b_v7_phrase_agg = "prob_mean"
stage_b_v7_phrase_mean_weight = 0.5
stage_b_v7_phrase_softmin_tau = 0.5
stage_b_v7_use_joint_phrase_head = True

stage_b_v7_phrase_focal_alpha = 0.25
stage_b_v7_phrase_focal_gamma = 2.0
stage_b_v7_phrase_focal_coef = 1.0
stage_b_v7_token_focal_alpha = 0.25
stage_b_v7_token_focal_gamma = 2.0
stage_b_v7_token_focal_coef = 0.25

stage_b_v7_pair_rank_loss_coef = 0.36
stage_b_v7_pair_rank_margin = 0.18
stage_b_v7_pair_rank_topk = 10
stage_b_v7_pair_rank_lse_tau = 0.1
stage_b_v7_pair_pos_weight = 0.0
stage_b_v7_pair_neg_weight = 0.0

text_mask_skip_invalid_canonical = False
text_mask_warn_limit = 100

save_checkpoint_interval = 1
amp_max_consecutive_skips = 8
