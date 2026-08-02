from config.ablations.cfg_stageb_gdino_score_adapter_semantic_verified import *  # noqa: F401,F403

# Independent confidence-only S_GDINO phase. The accepted negatives are tied to
# one frozen detector checkpoint and one extraction/deployment transform. Start
# only from an audited rank milestone with --pretrain_model_path; continuations
# inside this scope use --resume so optimizer and recent-q05 state remain exact.
stage_b_gdino_adapter_train_mode = "confidence_only"
stage_b_gdino_tn_scope = "image_global_topk_verified"

# Repeat the inherited P3 contract explicitly so the experiment config is
# readable without relying on parent-config defaults.
stage_b_gdino_rank_weight = 0.0
stage_b_gdino_confidence_weight = 1.0
stage_b_gdino_confidence_objective = "detached_recent_q05_trust"
stage_b_gdino_paired_margin_weight = 0.25
stage_b_gdino_paired_margin = 0.05
stage_b_gdino_positive_trust_margin = 0.02
stage_b_gdino_positive_trust_weight = 1.0
stage_b_gdino_queue_size = 512
stage_b_gdino_queue_min_count = 256

stage_b_gdino_gate_lr = 3e-4
data_aug_hflip_prob = 0.0
batch_size = 4
epochs = 1
skip_eval = True
