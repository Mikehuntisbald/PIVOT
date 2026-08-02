from config.ablations.cfg_stageb_dense_duty_common_20260728 import *  # noqa: F401,F403

stage_b_dense_duty_phase = "confidence"
stage_b_v22_train_phase = "confidence"
stage_b_v15_scorer_init_checkpoint = ""
stage_b_dense_duty_dataset_config_path = (
    "/media/haoyi/T9/pivot/"
    "config/datasets_stageb_dense_duty_confidence_20260728.json"
)
stage_b_dense_duty_dataset_config_sha256 = (
    "09ad8048e89e60243c6a1397a13a0f24356de81022b1e90a22855c0e61ad114e"
)

# Phase 2 loads the completed phase-1 model and freezes rank_tower. The private
# confidence tower learns token edits plus image-expression absolute tails.
stage_b_v11_listwise_weight = 0.0
stage_b_v11_local_tn_rank_weight = 0.0
stage_b_v11_predicate_tn_rank_weight = 0.0
stage_b_v11_local_anchor_weight = 0.0
stage_b_v11_batch_tail_separation_weight = 0.0
stage_b_v14_local_absolute_weight = 1.0
stage_b_v14_local_absolute_gamma = 1.0
stage_b_v14_predicate_absolute_weight = 0.0
stage_b_v11_global_tn_negative_weight = 1.0
stage_b_v11_global_tn_tail_weight = 1.0
stage_b_v11_global_tn_tail_topk = 10
stage_b_v11_global_tn_tail_temperature = 0.2
stage_b_v11_global_tn_tail_target_logit = 0.0
stage_b_v14_global_tn_all_candidates = True
stage_b_v14_tail_queue_weight = 1.0
stage_b_v14_tail_queue_size = 4096
stage_b_v14_tail_queue_min_count = 256
stage_b_v14_tail_queue_positive_quantile = 0.05
stage_b_v14_tail_queue_negative_quantile = 0.95
stage_b_v14_tail_queue_temperature = 0.1
stage_b_v14_tail_queue_margin = 0.3
stage_b_v15_tail_queue_global_scores = True
stage_b_v15_tail_queue_objective = "fpr95"
stage_b_v15_tail_queue_pair_weight = 0.25
stage_b_v15_tail_queue_pair_margin = 0.05
stage_b_v15_tail_queue_positive_trust_weight = 1.0
stage_b_v15_tail_queue_positive_trust_margin = 0.02
stage_b_v11_trainable_params_min = 25597697
stage_b_v11_trainable_params_max = 25597697

lr = 2e-5
lr_linear_proj_mult = 2e-6
# The zero-variance candidate-pool backward path is explicitly stabilized. A
# full-objective B16/acc4 soak kept scale 256 through the active FPR95 queue.
amp_init_scale = 256.0
epochs = 24
lr_drop = 100
