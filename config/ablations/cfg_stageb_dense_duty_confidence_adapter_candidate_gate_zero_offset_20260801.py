from config.ablations.cfg_stageb_dense_duty_confidence_adapter_candidate_tail_elementwise_20260801 import *  # noqa: F401,F403

# v39 keeps v38's bounded positive-tail gradient and changes only the hard
# lexical gate dead zone. Weak positive residuals now enter the veto ramp,
# while the learned scorer, losses, data, rank source, and gate scale stay fixed.
stage_b_dense_duty_confidence_veto_gate_offset = 0.0

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_candidate_gate_zero_offset_"
    "trace_audit_20260801"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "103c96c7114af0397f7fd640b7dd1dd43d3a067d83ccc4eebab8025b0453c0b8"
)

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u400_word_veto_candidate_gate_zero_offset_confidence_strict1607_v39"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_candidate_gate_zero_offset_highmem_20260801/"
    "probe_evaluation/u000400_strict1607_report.json"
)
