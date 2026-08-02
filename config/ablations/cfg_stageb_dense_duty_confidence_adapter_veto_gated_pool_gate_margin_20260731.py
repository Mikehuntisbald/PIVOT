from config.ablations.cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_rank_evidence_20260731 import *  # noqa: F401,F403

# v13 keeps the query-token conditioned adapter and reparameterizes its complete
# zero-initialized residual in the units defined by the paired carrier margin
# and the inference gate ramp. This changes optimization conditioning without
# changing U0 outputs, rank ownership, data, losses, or model capacity.
stage_b_dense_duty_confidence_revision = (
    "word_veto_gated_pool_gate_margin_v13"
)
stage_b_dense_duty_confidence_rank_evidence_contract = (
    "zero_init_rank_logit_gate_margin_scale_v3"
)
stage_b_dense_duty_confidence_residual_parameterization_gain = 0.25 / 0.03
stage_b_v11_trainable_params_min = 185_926
stage_b_v11_trainable_params_max = 185_926

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u300_word_veto_gated_pool_gate_margin_strict1607_v13"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_veto_gated_pool_gate_margin_highmem_20260731/"
    "probe_evaluation/u000300_strict1607_report.json"
)

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_veto_gated_pool_gate_margin_"
    "trace_audit_20260731/receipt.json"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "73c03fd5812917bb2eb10db1e25992a0677630905f32d8a36001bcca61c274d3"
)
