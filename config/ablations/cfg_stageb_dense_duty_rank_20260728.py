from config.ablations.cfg_stageb_dense_duty_common_20260728 import *  # noqa: F401,F403

stage_b_dense_duty_phase = "rank"
stage_b_v22_train_phase = "rank"
stage_b_dense_duty_dataset_config_path = (
    "/media/haoyi/T9/pivot/"
    "config/datasets_stageb_dense_duty_rank_20260728.json"
)
stage_b_dense_duty_dataset_config_sha256 = (
    "6cc541d8347468c625ca0785a8a87c6a85ef9e85ac911a301feab4c25061ceba"
)

# Candidate-relative localization and positive/TN modifier ordering update only
# rank_tower. Every absolute-confidence loss is exactly disabled in phase 1.
stage_b_v11_positive_iou_threshold = 0.5
stage_b_v11_negative_iou_threshold = 0.499
stage_b_v11_listwise_temperature = 0.2
stage_b_v11_listwise_weight = 0.5
stage_b_v11_local_tn_rank_margin = 0.3
stage_b_v11_local_tn_rank_weight = 0.5
stage_b_v11_predicate_tn_rank_margin = 0.3
stage_b_v11_predicate_tn_rank_weight = 1.0
stage_b_v11_local_anchor_weight = 0.0
stage_b_v11_batch_tail_separation_weight = 0.0
stage_b_v14_local_absolute_weight = 0.0
stage_b_v14_predicate_absolute_weight = 0.0
stage_b_v11_global_tn_negative_weight = 0.0
stage_b_v11_global_tn_tail_weight = 0.0
stage_b_v14_tail_queue_weight = 0.0
stage_b_v15_tail_queue_pair_weight = 0.0
stage_b_v15_tail_queue_positive_trust_weight = 0.0
stage_b_v14_global_tn_all_candidates = False
stage_b_v11_trainable_params_min = 25464320
stage_b_v11_trainable_params_max = 25464320

epochs = 4
lr_drop = 100
