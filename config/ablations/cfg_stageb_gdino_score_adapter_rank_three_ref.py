from config.ablations.cfg_stageb_gdino_score_adapter_dataft import *  # noqa: F401,F403

# Rank-only phase R. Initialize from the frozen pure-GDINO Stage-B data-FT
# checkpoint. The three positive ODVG sources are sampled equally by the
# accompanying dataset config; no TN caption or TN scope is consumed here.
stage_b_gdino_adapter_train_mode = "rank_only"
stage_b_gdino_tn_scope = ""

stage_b_gdino_rank_weight = 1.0
stage_b_gdino_confidence_weight = 0.0
stage_b_gdino_paired_margin_weight = 0.0
stage_b_gdino_queue_size = 0
stage_b_gdino_queue_min_count = 0

stage_b_gdino_rank_lr = 3e-5
lr = 3e-5
