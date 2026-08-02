from config.ablations.cfg_stageb_native_patch_category_d1 import *  # noqa: F401,F403

# D7 keeps D6's direct deployment-gap objective unchanged and adds one
# all-state category-positive anchor. The anchor protects the native-text-best
# category-positive query independently of native-winner state; no primary
# instance identity enters the loss.
stage_b_native_patch_contract_version = 7
stage_b_native_patch_objective = "d7_all_state_positive_anchor"

stage_b_native_patch_d7_weight = 1.0
stage_b_native_patch_gate_max_gap = 3.0
stage_b_native_patch_score_clip = 5.0
stage_b_native_patch_d7_keep_gap = 2.75
stage_b_native_patch_d7_drop_gap = 3.25
stage_b_native_patch_d7_drop_active_gap = 3.75
stage_b_native_patch_d7_temperature = 0.25
stage_b_native_patch_d7_drop_weight = 2.0
stage_b_native_patch_d7_critical_keep_weight = 1.0
stage_b_native_patch_d7_positive_active_gap = 2.0
stage_b_native_patch_d7_positive_target_gap = 2.5
stage_b_native_patch_d7_positive_barrier_weight = 2.0
stage_b_native_patch_d7_anchor_active_gap = 2.0
stage_b_native_patch_d7_anchor_target_gap = 2.5
stage_b_native_patch_d7_anchor_weight = 2.0

stage_b_native_patch_lr = 5e-5
lr = 5e-5
amp_init_scale = 8.0
