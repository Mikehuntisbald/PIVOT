from config.ablations.cfg_stageb_gdino_score_adapter_dataft import *  # noqa: F401,F403

# Independent confidence-only phase S.  Start this scope with
# --pretrain_model_path from an audited R or data-FT C milestone.  Never resume
# across scopes: a fresh criterion state is required so the q05 queue contains
# only image_global_topk_verified examples.
stage_b_gdino_adapter_train_mode = "confidence_only"
stage_b_gdino_tn_scope = "image_global_topk_verified"

# Keep the repaired all-query maximum, detached recent q05, positive trust, and
# paired margin from phase C.  At global batch 8, the queue warms after 32 steps.
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
batch_size = 4
epochs = 1
skip_eval = True
