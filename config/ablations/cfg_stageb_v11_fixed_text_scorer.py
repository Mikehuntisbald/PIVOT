from config.ablations.cfg_stageb_v10_deterministic_reranker import *  # noqa: F401,F403

# Stage A remains the deterministic localization/candidate generator. The v11
# decoder scores each full expression on the same fixed Top-K boxes and has no
# bbox head, patch branch, or private BERT copy.
stage_b_v11_fixed_text = True
stage_b_v11_num_layers = 3
stage_b_v11_candidate_topk = 50
stage_b_v11_expression_microbatch = 8
stage_b_v11_assert_fixed_candidates = False
support_num_patches_min = 1
support_num_patches_max = 1

# `stage_b_v7=True` is inherited only so existing evaluation tools read the
# compatibility score keys. The builder gives v11 priority and does not build
# the old verifier.
only_train_keywords = ["stage_b_fixed_text_scorer"]
only_train_exclude_keywords = []
unfreeze_decoder_last_n_layers = 0
stage_b_v11_trainable_params_min = 5_600_000
stage_b_v11_trainable_params_max = 5_700_000

# Candidate-set objectives: localize the positive expression within frozen
# candidates, then compare positive/TN text on the exact same IoU-positive boxes.
stage_b_v11_positive_iou_threshold = 0.5
stage_b_v11_negative_iou_threshold = 0.3
stage_b_v11_listwise_temperature = 0.2
stage_b_v11_listwise_weight = 0.2
stage_b_v11_local_tn_rank_margin = 0.3
stage_b_v11_local_tn_rank_weight = 1.0
stage_b_v11_local_anchor_weight = 0.5
stage_b_v11_positive_anchor_logit = 0.5
stage_b_v11_negative_anchor_logit = -0.5

# Local counterfactuals are not image-global negatives. Keep global-TN losses
# disabled until data carries an explicit global_tn_verified flag.
stage_b_v11_global_tn_negative_weight = 0.0
stage_b_v11_global_tn_tail_weight = 0.0
stage_b_v11_global_tn_tail_topk = 10
stage_b_v11_global_tn_tail_temperature = 0.2
stage_b_v11_global_tn_tail_target_logit = 0.0

# A small cross-batch tail term targets the observed positive/TN score overlap.
stage_b_v11_batch_tail_separation_weight = 0.25
stage_b_v11_batch_positive_quantile = 0.05
stage_b_v11_batch_negative_quantile = 0.95
stage_b_v11_batch_tail_margin = 0.3

# Decoder adaptation is deliberately conservative. Despite its historical
# name, this repository treats lr_linear_proj_mult as an absolute learning rate,
# not a multiplier; sampling_offsets/ref_point_head therefore use 2e-6.
lr = 2e-5
lr_linear_proj_mult = 2e-6
batch_size = 4
epochs = 8
lr_drop = 4
amp_init_scale = 512.0
amp_max_consecutive_skips = 8
log_patch_sanity = False
log_stage_b_patch_drift = False

# Initialize Stage A/v10 weights with --pretrain_model_path. Do not use --resume
# for that first load: v11 must start with a fresh optimizer and scheduler.
