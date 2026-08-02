from config.ablations.cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_20260731 import *  # noqa: F401,F403

# v6 changes only carrier-gate calibration. The positive carrier is explicitly
# trained below -0.10, while a TN carrier now reaches a fully open gate at a
# +0.05 residual instead of waiting for the full +0.15 raw hinge margin.
stage_b_dense_duty_confidence_revision = (
    "word_veto_gated_pool_calibrated_v6"
)
stage_b_dense_duty_confidence_veto_gate_offset = 0.02
stage_b_dense_duty_confidence_veto_gate_scale = 0.03

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u300_word_veto_gated_pool_calibrated_strict1607_v6"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_veto_gated_pool_calibrated_highmem_20260731/"
    "probe_evaluation/u000300_strict1607_report.json"
)

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_veto_gated_pool_calibrated_"
    "trace_audit_20260731/receipt.json"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "89323f04df3e445b5956b5d95d086ad9b6fd848ec0d49a64e8616d4877b3bad8"
)
