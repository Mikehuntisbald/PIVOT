from config.ablations.cfg_stageb_native_patch_category_d1 import *  # noqa: F401,F403

# D4 starts from the exact formal D1 U500 checkpoint and preserves D3's
# deployed-score critical-winner objective. It differs only by strongly
# protecting native-positive winners while the same eight patch projection
# tensors are updated; native text scores and boxes remain detached.
stage_b_native_patch_contract_version = 4
stage_b_native_patch_objective = "d4_positive_protected_critical_winner"

stage_b_native_patch_d4_weight = 1.0
stage_b_native_patch_gate_max_gap = 3.0
stage_b_native_patch_score_clip = 5.0
stage_b_native_patch_d4_keep_gap = 2.75
stage_b_native_patch_d4_separation_gap = 3.25
stage_b_native_patch_d4_temperature = 0.25
stage_b_native_patch_d4_critical_weight = 2.0
stage_b_native_patch_d4_critical_keep_weight = 1.0
stage_b_native_patch_d4_positive_keep_weight = 32.0

# Keep the D3 continuation geometry but provide a twofold AMP safety margin
# for the positive-protection term, whose effective gradient matches the
# undiluted critical-row term after weighting.
stage_b_native_patch_lr = 5e-5
lr = 5e-5
amp_init_scale = 8.0
