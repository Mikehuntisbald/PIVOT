from config.ablations.cfg_stageb_v15_decoupled_global_confidence import *  # noqa: F401,F403

# Fast diagnostic for whether detached frozen features contain enough signal to
# separate valid expressions from proposal-verified global TNs. Box ranking is
# bitwise fixed to the initialization checkpoint throughout this run.
only_train_keywords = ["stage_b_fixed_text_scorer.validity_head"]
stage_b_v11_trainable_params_min = 66_049
stage_b_v11_trainable_params_max = 66_049

stage_b_v11_listwise_weight = 0.0
stage_b_v11_local_tn_rank_weight = 0.0
stage_b_v11_predicate_tn_rank_weight = 0.0

lr = 5e-4
lr_linear_proj_mult = 5e-4
weight_decay = 1e-4
