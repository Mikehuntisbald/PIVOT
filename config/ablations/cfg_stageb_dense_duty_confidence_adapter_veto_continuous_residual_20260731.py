from config.ablations.cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_tail_paired_signed_rank_pool_20260731 import *  # noqa: F401,F403

# v21 removes the v20 absolute-cap readout that collapsed many TN and positive
# samples onto the same q05 score surface. The trace-supervised maximum
# modifier residual now supplies a continuous carrier gate, and the signed
# rank-query pool supplies a sample-specific delta:
#
#   confidence = frozen_rank_confidence + sigmoid(modifier_mismatch) * delta
#
# U0 remains exactly equal to U6551 because the pool delta is zero initialized.
# Positive raw-token supervision closes the gate; changed-TN supervision opens
# it. No edit annotation is consumed by the inference path.
stage_b_dense_duty_confidence_revision = (
    "word_veto_continuous_conditional_residual_v21"
)
stage_b_dense_duty_confidence_gate_gradient_contract = "continuous_sigmoid_v3"

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_veto_continuous_residual_"
    "trace_audit_20260731/receipt.json"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "db41b5606e3307514c23f750f32047c08399b15dfa4e86fc54c55e6c946a641c"
)

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u300_word_veto_continuous_conditional_residual_strict1607_v21"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_veto_continuous_residual_highmem_20260731/"
    "probe_evaluation/u000300_strict1607_report.json"
)
