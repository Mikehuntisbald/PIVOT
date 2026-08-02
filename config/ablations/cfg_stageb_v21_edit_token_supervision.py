from config.ablations.cfg_stageb_v19_full_text_base_plus_gate import *  # noqa: F401,F403

# Edit-aware supervision acts only on IoU-positive fixed candidates. Positive
# phrase tokens stay positive; a paired TN keeps shared tokens positive and
# labels only the tokenizer-aligned changed tokens negative.
stage_b_v21_token_objective = "edit_bce"
stage_b_v21_token_weight = 1.0
stage_b_v21_token_positive_weight = 1.0
stage_b_v21_token_shared_weight = 0.25
stage_b_v21_token_edit_weight = 1.0
stage_b_v21_token_focal_alpha = 0.25
stage_b_v21_token_focal_gamma = 2.0
# Paper runs accept only rows certified by the strict single-edit manifest.
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
