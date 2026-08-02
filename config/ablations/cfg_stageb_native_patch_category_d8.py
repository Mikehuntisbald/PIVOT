from config.ablations.cfg_stageb_native_patch_category_d1 import *  # noqa: F401,F403

# D8 keeps D6's three deployment-aligned terms unchanged. It replaces D7's
# pooled anchor with fixed native-winner-state budgets, each macro-averaged
# over present support classes. No primary identity or source metadata enters
# the objective.
stage_b_native_patch_contract_version = 8
stage_b_native_patch_objective = "d8_state_class_macro_anchor"

stage_b_native_patch_d8_weight = 1.0
stage_b_native_patch_gate_max_gap = 3.0
stage_b_native_patch_score_clip = 5.0
stage_b_native_patch_d8_keep_gap = 2.75
stage_b_native_patch_d8_drop_gap = 3.25
stage_b_native_patch_d8_drop_active_gap = 3.75
stage_b_native_patch_d8_temperature = 0.25
stage_b_native_patch_d8_drop_weight = 2.0
stage_b_native_patch_d8_critical_keep_weight = 1.0
stage_b_native_patch_d8_positive_active_gap = 2.0
stage_b_native_patch_d8_positive_target_gap = 2.5
stage_b_native_patch_d8_positive_barrier_weight = 2.0
stage_b_native_patch_d8_anchor_active_gap = 2.0
stage_b_native_patch_d8_anchor_target_gap = 2.5
stage_b_native_patch_d8_anchor_negative_weight = 1.0
stage_b_native_patch_d8_anchor_neutral_weight = 2.0
stage_b_native_patch_d8_anchor_positive_weight = 4.0

stage_b_native_patch_lr = 5e-5
lr = 5e-5
amp_init_scale = 8.0
