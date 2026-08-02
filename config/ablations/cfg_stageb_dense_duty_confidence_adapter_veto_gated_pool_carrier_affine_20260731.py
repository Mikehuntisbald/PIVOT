from config.ablations.cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_carrier_slope_20260731 import *  # noqa: F401,F403

# v15 adds one zero-initialized carrier-only intercept to v14's contextual
# token slope. It accelerates the shared carrier direction while the 64-D
# weight retains token-conditioned deviations; rank and base adapter stay fixed.
stage_b_dense_duty_confidence_revision = (
    "word_veto_gated_pool_carrier_affine_v15"
)
stage_b_dense_duty_confidence_rank_evidence_contract = (
    "zero_init_carrier_token_rank_affine_v5"
)
stage_b_dense_duty_confidence_residual_parameterization_gain = 0.25 / 0.03
stage_b_v11_trainable_params_min = 185_990
stage_b_v11_trainable_params_max = 185_990

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u300_word_veto_gated_pool_carrier_affine_strict1607_v15"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_veto_gated_pool_carrier_affine_highmem_20260731/"
    "probe_evaluation/u000300_strict1607_report.json"
)

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_veto_gated_pool_carrier_affine_"
    "trace_audit_20260731/receipt.json"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "0fac9107a1913ef6da65a3bf2b388fab8b135385e0298023b57de5d81ebf2df5"
)
