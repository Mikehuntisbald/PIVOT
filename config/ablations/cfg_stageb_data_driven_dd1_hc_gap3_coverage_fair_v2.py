from config.ablations.cfg_stageb_data_driven_dd1_h_same_class_rank_fair_v2 import *  # noqa: F401,F403

# DD1-HC keeps H's primary and same-category auxiliary roles, then adds only
# the competitors admitted by the exact fixed-Gap3 inference route. The patch
# score is detached when this boolean coverage mask is built, so rank loss
# still cannot update the category branch.
stage_b_data_driven_variant_id = "DD1-HC"
stage_b_data_driven_rank_supervision = (
    "primary_vs_same_category_aux_plus_gap3_coverage_v1"
)
stage_b_data_driven_category_gate_max_gap = 3.0
stage_b_data_driven_patch_score_clip = 5.0
stage_b_data_driven_strict_sample_identity = True

_dd1_h_root = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "data_driven_dd1_h_same_class_rank_fair_v2_seed42_b64_u5020_v1"
)
_dd1_h_preflight = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "data_driven_dd1_h_same_class_rank_fair_v2_preflight"
)
stage_b_data_driven_hc_control_checkpoint_path = (
    f"{_dd1_h_root}/checkpoint_iter.pth"
)
stage_b_data_driven_hc_control_checkpoint_sha256 = (
    "f09beb98ffef7558e1cd0e59dab3a516d2f46e1f1152a0ab4d7d7841d230019a"
)
stage_b_data_driven_hc_control_result_path = (
    f"{_dd1_h_root}/evaluations/gap3_refcocog_val_b16_protocol_v1/"
    "formal_result_audit.json"
)
stage_b_data_driven_hc_control_result_sha256 = (
    "b108c24f1e55b2c5af1ce117e14fa30e841dbb4cbdc544ba68f2c18526eb3ffc"
)
stage_b_data_driven_hc_rank_diagnostic_path = (
    f"{_dd1_h_root}/evaluations/rank_only_refcocog_val_b16_protocol_v1/"
    "summary.json"
)
stage_b_data_driven_hc_rank_diagnostic_sha256 = (
    "97cddb071681001e0577d1fd9d12c22b607fd5f194b246c3f463214a366ffb46"
)
stage_b_data_driven_hc_query_diagnostic_path = (
    f"{_dd1_h_preflight}/query_diagnostics_analysis_seed600042_v1.json"
)
stage_b_data_driven_hc_query_diagnostic_sha256 = (
    "98ac00f3f732c27dd9e30cbd0cb0a218af894a1e610214631f46602d4d6c1c3d"
)
stage_b_data_driven_hc_source_snapshot_path = (
    f"{_dd1_h_preflight}/source_snapshot_pre_dd1hc_20260722.tar.gz"
)
stage_b_data_driven_hc_source_snapshot_sha256 = (
    "8ed9aefe08b6b32c74493963e361c04a06da8a461dbe3f71bc28c14ee2561cc9"
)
del _dd1_h_root, _dd1_h_preflight
