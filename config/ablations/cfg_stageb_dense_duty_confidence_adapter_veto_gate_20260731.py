from config.ablations.cfg_stageb_dense_duty_confidence_adapter_veto_penalty_20260730 import *  # noqa: F401,F403

# Final word-veto training contract: the deployed gate remains detached from
# global confidence losses, while direct trace supervision anchors its raw
# source below zero on positive target queries and above zero on changed TN
# words. This removes the all-off and all-on gate degeneracies.
stage_b_dense_duty_confidence_revision = "word_veto_raw_gate_margin_v3"
stage_b_dense_duty_raw_veto_gate_weight = 1.0
stage_b_dense_duty_raw_veto_positive_margin = 0.1
stage_b_dense_duty_raw_veto_tn_margin = 0.1

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u300_word_veto_gate_strict1607_v3"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_veto_gate_highmem_20260731/probe_evaluation/"
    "u000300_strict1607_report.json"
)

# Filled by the code-bound direct-trace audit before training starts.
stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_veto_gate_trace_audit_20260731/"
    "receipt.json"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "a3999b238a0e957236fa4f7b7b52dc3885ff8fd20a7a7c52a753e6cb3f419066"
)
