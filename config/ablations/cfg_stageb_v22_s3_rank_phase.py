from config.ablations.cfg_stageb_v22_s2_independent_joint import *  # noqa: F401,F403

# S3 phase 1/2: update only the rank decoder. The paper launcher assigns this
# phase exactly half of the fixed optimizer-update budget; confidence gets the
# other half.
stage_b_v22_table_id = "S3-rank"
stage_b_v22_score_ownership = "independent_decoders_two_phase"
stage_b_v22_train_phase = "rank"
stage_b_v22_objective_fidelity = (
    "common_objective_ownership_ablation_split_schedule"
)
only_train_keywords = ["stage_b_fixed_text_scorer.decoder"]
only_train_exclude_keywords = []
stage_b_v11_trainable_params_min = 5_626_240
stage_b_v11_trainable_params_max = 5_626_240
stage_b_v15_validity_lr = None
stage_b_v15_separate_grad_clip = False

epochs = 4
lr_drop = 2
stage_b_v22_phase_index = 1
stage_b_v22_phase_count = 2
stage_b_v22_requires_rank_phase_checkpoint = False
