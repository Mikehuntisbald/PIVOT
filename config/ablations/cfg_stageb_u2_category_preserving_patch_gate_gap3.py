from config.ablations.cfg_stageb_u2_category_preserving_patch_gate import *  # noqa: F401,F403

# Fixed by the canonical B16 three-val sweep. The sealed selector chose gap 3
# without using any official test split.
stage_b_u0_category_gate_max_gap = 3.0
stage_b_u0_category_gate_selection_receipt = (
    "outputs/paper_cvpr_v1/u2_category_complete_seed42_b56_scale8192_v2/"
    "evaluations/category_gate_val_sweep_canonical_b16/selection_receipt.json"
)
stage_b_u0_category_gate_selection_payload_sha256 = (
    "aeff3ee6c55fcb49b0942ed9fd363c846e7a4d3d87e2c542e7feb69e4dcc6a4c"
)
