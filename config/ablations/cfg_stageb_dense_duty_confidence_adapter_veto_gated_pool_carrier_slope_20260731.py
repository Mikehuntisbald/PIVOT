from config.ablations.cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_rank_evidence_20260731 import *  # noqa: F401,F403

# v14 restores the unit-gain query-token adapter and adds one zero-initialized,
# token-conditioned rank slope only on the exact frozen-reference carrier.
# The carrier term alone is expressed in paired-margin/gate-ramp units.
stage_b_dense_duty_confidence_revision = (
    "word_veto_gated_pool_carrier_slope_v14"
)
stage_b_dense_duty_confidence_rank_evidence_contract = (
    "zero_init_carrier_token_rank_slope_v4"
)
stage_b_dense_duty_confidence_residual_parameterization_gain = 0.25 / 0.03
stage_b_v11_trainable_params_min = 185_989
stage_b_v11_trainable_params_max = 185_989

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u300_word_veto_gated_pool_carrier_slope_strict1607_v14"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_veto_gated_pool_carrier_slope_highmem_20260731/"
    "probe_evaluation/u000300_strict1607_report.json"
)

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_veto_gated_pool_carrier_slope_"
    "trace_audit_20260731/receipt.json"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "42082ac8c59f67ecb358afe47eb9b00ab172f2442ae240264eb9efcd08227768"
)
