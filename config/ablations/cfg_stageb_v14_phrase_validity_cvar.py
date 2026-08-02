from config.ablations.cfg_stageb_v12_predicate_token_rank import *  # noqa: F401,F403

# Stage A remains the immutable patch/localization tower. The deployed text
# score is a zero-initialized scalar residual over the detached legacy phrase
# logit, so initialization exactly preserves v12 ranking while all v14 losses
# optimize the score that inference actually consumes.
stage_b_v14_validity_head = True
stage_b_v11_assert_fixed_candidates = True
stage_b_v11_trainable_params_min = 5_690_000
stage_b_v11_trainable_params_max = 5_700_000

# Candidate ranking plus class-balanced absolute target/TN classification.
stage_b_v11_listwise_weight = 0.2
stage_b_v11_local_tn_rank_weight = 0.25
stage_b_v11_predicate_tn_rank_weight = 0.25
stage_b_v11_local_anchor_weight = 0.0
stage_b_v11_batch_tail_separation_weight = 0.0
stage_b_v14_local_absolute_weight = 1.0
stage_b_v14_local_absolute_gamma = 1.0
stage_b_v14_predicate_absolute_weight = 0.5
stage_b_v14_predicate_absolute_gamma = 1.0

# Cross-step target-candidate tails. The queue stores one best-IoU positive/TN
# score per sample, is synchronized across DDP ranks, and is committed only
# after a successful optimizer step.
stage_b_v14_tail_queue_weight = 0.25
stage_b_v14_tail_queue_size = 4096
stage_b_v14_tail_queue_min_count = 256
stage_b_v14_tail_queue_positive_quantile = 0.05
stage_b_v14_tail_queue_negative_quantile = 0.95
stage_b_v14_tail_queue_temperature = 0.1
stage_b_v14_tail_queue_margin = 0.3

# This base config is target-local and label-safe. The companion global config
# enables the historical benchmark's all-candidate TN semantics.
stage_b_v14_global_tn_all_candidates = False
stage_b_v11_global_tn_negative_weight = 0.0
stage_b_v11_global_tn_tail_weight = 0.0

# v11 already passed a two-rank batch-16 memory probe. The larger effective
# batch makes absolute/tail estimates less noisy and gives a fairer exposure
# budget than the earlier batch-4 v12 run.
batch_size = 16
stage_b_v11_expression_microbatch = 8
