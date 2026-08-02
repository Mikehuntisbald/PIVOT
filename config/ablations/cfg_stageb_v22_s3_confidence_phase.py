from config.ablations.cfg_stageb_v22_s2_independent_joint import *  # noqa: F401,F403

# S3 phase 2/2: load the completed S3-rank model state and update only the
# validity gate. The confidence decoder remains the immutable warm-start base.
stage_b_v22_table_id = "S3-confidence"
stage_b_v22_score_ownership = "independent_decoders_two_phase"
stage_b_v22_train_phase = "confidence"
stage_b_v22_objective_fidelity = (
    "common_objective_ownership_ablation_split_schedule"
)
only_train_keywords = ["stage_b_fixed_text_scorer.validity_head"]
only_train_exclude_keywords = []
stage_b_v11_trainable_params_min = 66_049
stage_b_v11_trainable_params_max = 66_049

# Every loss weight remains identical to S2. Parameter ownership alone makes
# this a confidence phase: the loaded rank decoder is frozen by the trainable
# keyword contract above.

epochs = 4
lr_drop = 2
stage_b_v22_phase_index = 2
stage_b_v22_phase_count = 2
stage_b_v22_requires_rank_phase_checkpoint = True
