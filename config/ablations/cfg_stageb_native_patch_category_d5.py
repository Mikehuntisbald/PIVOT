from config.ablations.cfg_stageb_native_patch_category_d1 import *  # noqa: F401,F403

# D5 starts from the exact formal D1 U500 checkpoint. It retains D3's
# critical-winner geometry but replaces all-positive mean retention with an
# active-tail barrier, so already-safe positives contribute no gradient.
stage_b_native_patch_contract_version = 5
stage_b_native_patch_objective = "d5_active_tail_positive_barrier"

stage_b_native_patch_d5_weight = 1.0
stage_b_native_patch_gate_max_gap = 3.0
stage_b_native_patch_score_clip = 5.0
stage_b_native_patch_d5_keep_gap = 2.75
stage_b_native_patch_d5_separation_gap = 3.25
stage_b_native_patch_d5_temperature = 0.25
stage_b_native_patch_d5_critical_weight = 2.0
stage_b_native_patch_d5_critical_keep_weight = 1.0
stage_b_native_patch_d5_active_gap = 2.0
stage_b_native_patch_d5_target_gap = 2.5
stage_b_native_patch_d5_positive_barrier_weight = 2.0

stage_b_native_patch_lr = 5e-5
lr = 5e-5
amp_init_scale = 8.0
