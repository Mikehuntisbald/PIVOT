from config.ablations.cfg_stageb_dense_duty_confidence_adapter_candidate_split_boundary_routing_20260801 import *  # noqa: F401,F403

# V48 is a fresh V47 single-variable ablation. The FPR95 queue keeps its exact
# historical positive q05, margin, temperature, positive trust, and pair term;
# only already-rejected TNs are removed from the negative reduction.
stage_b_dense_duty_confidence_revision = (
    "word_veto_candidate_split_fpr_active_set_v48"
)
stage_b_v15_tail_queue_negative_reduction_contract = (
    "exact_fpr95_active_set_mean_v1"
)

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_candidate_split_fpr_active_set_"
    "trace_audit_20260801"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "e7a0ce50b9ca557d1b8f804cc2aa10bfbf3d9a548120c1fdeb7fdac08b974891"
)

# Formal admission remains unbound until this exact U400 probe beats the fixed
# GDINO Stage-B data-FT strict1607 false-accept count.
stage_b_dense_duty_confidence_probe_admission_contract = (
    "u400_word_veto_candidate_split_fpr_active_set_confidence_strict1607_v48"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_candidate_split_fpr_active_set_highmem_20260801/"
    "probe_evaluation/u000400_strict1607_report.json"
)
