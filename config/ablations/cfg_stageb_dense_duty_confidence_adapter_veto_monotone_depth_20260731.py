from config.ablations.cfg_stageb_dense_duty_confidence_adapter_veto_continuous_residual_20260731 import *  # noqa: F401,F403

# v22 retains v21's continuous trace-supervised modifier gate but constrains
# the global pool output to a non-negative veto depth. The deployed confidence
# can only stay at or move below the frozen rank reference; a centered
# softplus straight-through derivative keeps the exactly-zero U0 depth
# trainable.
stage_b_dense_duty_confidence_revision = (
    "word_veto_continuous_monotone_depth_v22"
)
stage_b_dense_duty_confidence_gate_gradient_contract = (
    "continuous_sigmoid_monotone_depth_v4"
)

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_veto_monotone_depth_"
    "trace_audit_20260731/receipt.json"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "131cf7d6e4cb403d4a907704c9eb6577f1db2ef731568383cd16b5d1a932177a"
)

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u300_word_veto_continuous_monotone_depth_strict1607_v22"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_veto_monotone_depth_highmem_20260731/"
    "probe_evaluation/u000300_strict1607_report.json"
)
