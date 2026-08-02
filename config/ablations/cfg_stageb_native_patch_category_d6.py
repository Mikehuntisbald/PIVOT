from config.ablations.cfg_stageb_native_patch_category_d1 import *  # noqa: F401,F403

# D6 starts from the exact formal D1 U500 checkpoint. It optimizes the
# deployed Gap-3 decision directly: category-negative native winners are
# pushed outside the gate, while category-positive queries retain explicit
# category-level protection. No primary-instance identity enters the loss.
stage_b_native_patch_contract_version = 6
stage_b_native_patch_objective = "d6_direct_deployment_gap"

stage_b_native_patch_d6_weight = 1.0
stage_b_native_patch_gate_max_gap = 3.0
stage_b_native_patch_score_clip = 5.0
stage_b_native_patch_d6_keep_gap = 2.75
stage_b_native_patch_d6_drop_gap = 3.25
stage_b_native_patch_d6_drop_active_gap = 3.75
stage_b_native_patch_d6_temperature = 0.25
stage_b_native_patch_d6_drop_weight = 2.0
stage_b_native_patch_d6_critical_keep_weight = 1.0
stage_b_native_patch_d6_positive_active_gap = 2.0
stage_b_native_patch_d6_positive_target_gap = 2.5
stage_b_native_patch_d6_positive_barrier_weight = 2.0

stage_b_native_patch_lr = 5e-5
lr = 5e-5
amp_init_scale = 8.0
