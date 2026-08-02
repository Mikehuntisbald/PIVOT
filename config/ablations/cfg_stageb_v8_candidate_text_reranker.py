from config.ablations.cfg_stageb_v7_sam3tn_pair_verifier import *  # noqa: F401,F403

# Stage A supplies a fixed candidate pool. Inside that pool, text alone determines
# ordering; patch confidence is available only as an explicit prior ablation.
stage_b_v7_candidate_topk = 50
stage_b_v7_patch_prior_weight = 0.0
stage_b_v7_context_scale = 2.0

# Candidate-conditional supervision: clear IoU positives, clear distractors, and
# local counterfactual TN labels on the same target candidates only.
stage_b_v7_negative_iou_max = 0.3
stage_b_v7_phrase_hard_negative_topk = 10
stage_b_v7_tn_shared_token_weight = 0.0
stage_b_v7_phrase_focal_coef = 1.0
stage_b_v7_token_focal_coef = 0.1
stage_b_v7_pair_rank_loss_coef = 0.5
stage_b_v7_pair_rank_margin = 0.3
stage_b_v7_pair_rank_topk = 10
stage_b_v7_pair_rank_lse_tau = 0.1

# Start AMP near the empirically stable scale and fail loudly on repeated skips.
amp_init_scale = 512.0
amp_max_consecutive_skips = 8

epochs = 8
lr_drop = 4
save_checkpoint_interval = 1
