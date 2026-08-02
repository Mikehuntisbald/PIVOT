from config.ablations.cfg_stageb_dense_duty_confidence_adapter_candidate_split_positive_tail_20260801 import *  # noqa: F401,F403

# V47 is a fresh V46 single-variable ablation. The top-quarter reduction
# concentrates the deployed-routing gradient on the highest-veto positives and
# lowest-veto TNs while preserving the 0.1 routing weight and independent clips.
stage_b_dense_duty_confidence_revision = (
    "word_veto_candidate_split_boundary_routing_v47"
)
stage_b_dense_duty_deployed_veto_routing_reduction_contract = (
    "balanced_top_quarter_cvar_v2"
)

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_candidate_split_boundary_routing_"
    "trace_audit_20260801"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "56655e7c20f06eb79b552b766fd39519273f9efdc4414f52b9b9bcd751a60252"
)

# Formal admission remains unbound until this exact U400 probe beats the fixed
# GDINO Stage-B data-FT strict1607 false-accept count.
stage_b_dense_duty_confidence_probe_admission_contract = (
    "u400_word_veto_candidate_split_boundary_routing_confidence_strict1607_v47"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_candidate_split_boundary_routing_highmem_20260801/"
    "probe_evaluation/u000400_strict1607_report.json"
)
