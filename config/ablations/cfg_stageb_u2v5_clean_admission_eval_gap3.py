"""Val-only Gap3 evaluation for a clean U2-v5 admission milestone."""

from config.ablations.cfg_stageb_u2v5_clean_admission_u100 import *  # noqa: F401,F403

# The clean admission checkpoint intentionally retains the complete U2-v4
# training-replay mechanism and its ownership receipt.
stage_b_u2v4_legacy_training_replay = True
stage_b_u2v4_checkpoint_eval = True
stage_b_u0_category_preserving_patch_gate = True
stage_b_u0_category_gate_max_gap = 3.0
stage_b_u2v2_category_gate_max_gap = stage_b_u0_category_gate_max_gap
stage_b_gdino_ref_top1_guard = False
stage_b_gdino_ref_route_contract = "u2v5_clean_admission_gap3_frozen_r100_v1"

stage_b_u2v5_clean_anchor = True
stage_b_u2v2_initializer_path = (
    "outputs/u2v5_leakage_clean_anchor_20260817/initializer/"
    "checkpoint_clean_init.pth"
)
stage_b_u2v2_initializer_sha256 = (
    "ad7b3a563ef84356c6d952167ee6a48f615f8db887eba31bed92a81b0ba756a7"
)
