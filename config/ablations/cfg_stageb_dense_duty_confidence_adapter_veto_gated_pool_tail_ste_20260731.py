from config.ablations.cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_carrier_affine_20260731 import *  # noqa: F401,F403

# v16 keeps v15's exact inference graph and parameter surface. During
# confidence training only, the post-veto scalar differentiates through a
# smooth carrier-gate surrogate so a hard-closed high-tail TN is not a dead row.
stage_b_dense_duty_confidence_revision = (
    "word_veto_gated_pool_tail_ste_v16"
)
stage_b_dense_duty_confidence_gate_gradient_contract = (
    "hard_forward_soft_backward_v2"
)

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u300_word_veto_gated_pool_tail_ste_strict1607_v16"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_veto_gated_pool_tail_ste_highmem_20260731/"
    "probe_evaluation/u000300_strict1607_report.json"
)

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_veto_gated_pool_tail_ste_"
    "trace_audit_20260731/receipt.json"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "7baa3ab4265e66886295895db9a0a8a912fa1270336236745a7e1f38c0df02f6"
)
