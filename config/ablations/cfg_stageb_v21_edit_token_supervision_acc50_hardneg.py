from config.ablations.cfg_stageb_v19_full_text_base_plus_gate import *  # noqa: F401,F403

# Combine v19's explicit immutable-base-plus-gate confidence contract with the
# v20 rank-path hard-negative boundary. Historical v20 alone is gate-only.
stage_b_v11_negative_iou_threshold = 0.499
stage_b_v20_acc50_aligned_hard_negatives = True
stage_b_v21_token_objective = "edit_bce"
stage_b_v21_token_weight = 1.0
stage_b_v21_token_positive_weight = 1.0
stage_b_v21_token_shared_weight = 0.25
stage_b_v21_token_edit_weight = 1.0
stage_b_v21_token_focal_alpha = 0.25
stage_b_v21_token_focal_gamma = 2.0
stage_b_v21_allow_legacy_token_diff_fallback = False
stage_b_v19_allow_scope_labeled_tn_ablation = True
stage_b_v19_table_b_id = "D3"
stage_b_v19_table_b_scope_allowlist = ["proposal_covered_verified"]
stage_b_v19_table_b_audit = (
    "data/ablations/stageb_tn_table_b_equal_exposure_20260717/audit.json"
)
stage_b_v19_table_b_audit_sha256 = (
    "7d74d541529a3e9abfbe84b192f2d0d3608d291bf46d19263c7c06a6ccb2291d"
)
stage_b_v19_table_b_allow_single_edit_token_provenance = True
