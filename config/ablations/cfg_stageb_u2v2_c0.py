"""U2-v2 C0: frozen Stage-A patch gate + sealed R100/C100."""

from config.ablations.cfg_stageb_u0_r100p50_patch_rank import *  # noqa: F401,F403

stage_b_u2v2 = True
stage_b_u2v2_contract = "pivot.stageb.u2v2_initializer/v1"
stage_b_u2v2_rank_residual = False
stage_b_u2v2_residual_limit = 0.1
stage_b_u2v2_category_gate_max_gap = 3.0
stage_b_u0_category_preserving_patch_gate = True
stage_b_u0_category_gate_max_gap = stage_b_u2v2_category_gate_max_gap

# U2-v2 deliberately exposes the unguarded full-expression R100 route.
stage_b_gdino_ref_top1_guard = False
stage_b_gdino_ref_route_contract = "u2v2_patch_gate_postgate_rank_v1"
stage_b_gdino_tn_scope = "benchmark_dataft_alltn"
stage_b_gdino_confidence_objective = "detached_recent_q05_total_trust"

# Runtime provenance is fail-closed by tools/build_stageb_u2v2_initializer.py.
stage_b_u2v2_stagea_checkpoint = (
    "/media/haoyi/T9/gdino/outputs/"
    "stageA_b58_trunk_patch0006_realign_bs38_formal_20260814/checkpoint0007.pth"
)
stage_b_u2v2_stagea_sha256 = (
    "fe20fe91f3c46b6d143db13c74817ff3aa810cc51d1579104913c3d23fec9a8b"
)
stage_b_u2v2_c100_checkpoint = (
    "/media/haoyi/T9/pivot/outputs/"
    "stagea_b58_patch0006_realign_r100_c100_sealed_evaluator_v3_20260815/"
    "confidence/milestones/checkpoint_iter_000100.pth"
)
stage_b_u2v2_c100_sha256 = (
    "c9737d6bcabec4325bd53b146782b82a4d1119237d01d87de9f8d2987e03000e"
)
stage_b_u2v2_r100_checkpoint = (
    "/media/haoyi/T9/pivot/outputs/"
    "stagea_b58_patch0006_realign_r100_c100_sealed_evaluator_v3_20260815/"
    "rank/milestones/checkpoint_iter_000100.pth"
)
stage_b_u2v2_r100_sha256 = (
    "346e847228f7a14a70ee772233c8d5fb2b090aebab76d7deda981901e74cc2b7"
)
stage_b_u2v2_initializer_path = (
    "/media/haoyi/T9/pivot/outputs/u2v2_diagnostic_20260816/"
    "c0/checkpoint_u2v2_c0.pth"
)
stage_b_u2v2_initializer_sha256 = (
    "2578c62a187948e7e459afba0f2c72d3de6901912abc3ac7770a19b86f177309"
)
