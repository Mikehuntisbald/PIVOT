from config.ablations.cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_rank_evidence_20260731 import *  # noqa: F401,F403

# v12 adds a zero-initialized intercept to v11's frozen-rank-logit scale. The
# affine path learns an absolute threshold while preserving exact U0 identity.
stage_b_dense_duty_confidence_revision = (
    "word_veto_gated_pool_rank_affine_v12"
)
stage_b_dense_duty_confidence_rank_evidence_contract = (
    "zero_init_rank_logit_affine_v2"
)
stage_b_v11_trainable_params_min = 185_927
stage_b_v11_trainable_params_max = 185_927

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u300_word_veto_gated_pool_rank_affine_strict1607_v12"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_veto_gated_pool_rank_affine_highmem_20260731/"
    "probe_evaluation/u000300_strict1607_report.json"
)

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_veto_gated_pool_rank_affine_"
    "trace_audit_20260731/receipt.json"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "6bdf8762d5931bff908c926f462d1a9d738e042dcb52b2c2d94cca1129c3d370"
)
