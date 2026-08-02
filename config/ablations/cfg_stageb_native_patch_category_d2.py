from config.ablations.cfg_stageb_native_patch_category_d1 import *  # noqa: F401,F403

# D2 continues from the pure-data D1 checkpoint and trains the same eight patch
# projection tensors.  Frozen b58 text/query/box tensors remain bitwise fixed.
stage_b_native_patch_contract_version = 2
stage_b_native_patch_objective = "d2_gate_aligned"

# Match the deployed category gate exactly, with a 0.25 safety band on either
# side of Gap-3. Native text scores select hard queries but are detached.
stage_b_native_patch_d2_weight = 1.0
stage_b_native_patch_gate_max_gap = 3.0
stage_b_native_patch_score_clip = 5.0
stage_b_native_patch_d2_keep_gap = 2.75
stage_b_native_patch_d2_drop_gap = 3.25
stage_b_native_patch_d2_temperature = 0.25
stage_b_native_patch_d2_native_hard_negatives = 16
stage_b_native_patch_d2_patch_hard_negatives = 4
stage_b_native_patch_d2_keep_weight = 2.0
stage_b_native_patch_d2_drop_weight = 1.0
stage_b_native_patch_d2_coverage_weight = 0.25

# D1 already learned useful category geometry. D2 resets optimizer/scaler and
# uses a conservative learning rate for the deployment-aligned continuation.
stage_b_native_patch_lr = 1e-4
lr = 1e-4
