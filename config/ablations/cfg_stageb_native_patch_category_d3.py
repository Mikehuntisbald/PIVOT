from config.ablations.cfg_stageb_native_patch_category_d1 import *  # noqa: F401,F403

# D3 starts from the exact formal D1 U500 checkpoint and uses the audited D2
# category-complete corpus unchanged. It trains only the same eight patch
# projection tensors; native full-text scores and boxes remain detached.
stage_b_native_patch_contract_version = 3
stage_b_native_patch_objective = "d3_critical_winner"

# Directly repair category-negative native winners in the deployed Gap-3,
# clip-5 standardized patch-score space. The three row sets are reduced
# independently so rare critical rows are not diluted by easy queries.
stage_b_native_patch_d3_weight = 1.0
stage_b_native_patch_gate_max_gap = 3.0
stage_b_native_patch_score_clip = 5.0
stage_b_native_patch_d3_keep_gap = 2.75
stage_b_native_patch_d3_separation_gap = 3.25
stage_b_native_patch_d3_temperature = 0.25
stage_b_native_patch_d3_critical_weight = 2.0
stage_b_native_patch_d3_critical_keep_weight = 1.0
stage_b_native_patch_d3_positive_keep_weight = 1.0

# D3 is a short, conservative fresh-optimizer continuation from D1.
stage_b_native_patch_lr = 5e-5
lr = 5e-5

# The undiluted critical-row loss has a larger FP16 gradient than D1/D2.  Scale
# 64 passed the short smoke but overflowed once before U50; scale 32 then stayed
# stable through U75.  Pin 16 for a twofold safety margin so formal training
# never relies on overflow-driven GradScaler backoff.
amp_init_scale = 16.0
