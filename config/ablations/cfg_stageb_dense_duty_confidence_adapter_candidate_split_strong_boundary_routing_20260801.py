from config.ablations.cfg_stageb_dense_duty_confidence_adapter_candidate_split_boundary_routing_20260801 import *  # noqa: F401,F403

# V50 is a fresh V47 single-variable ablation. The deployed-carrier routing
# objective is the only behavioral change: its weight increases from 0.1 to
# 0.25 to target the false-veto positive tail observed in strict1607 without
# letting the routing auxiliary dominate token BCE.
stage_b_dense_duty_confidence_revision = (
    "word_veto_candidate_split_strong_boundary_routing_v50"
)
stage_b_dense_duty_deployed_veto_routing_weight = 0.25
# Seal V47's semantic defaults so a V50 checkpoint cannot silently change its
# cross-sample TN objective or edit-query scope under future defaults.
stage_b_v15_tail_queue_negative_reduction_contract = "all_mean_v1"
stage_b_v21_token_edit_query_scope = "target_iou_v1"

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_candidate_split_strong_boundary_"
    "routing_trace_audit_20260801/receipt.json"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "cada245135685743d88a315b28b55395c2e8e206cb57191a59999c47178b814d"
)

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u400_word_veto_candidate_split_strong_boundary_routing_confidence_"
    "strict1607_v50"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_candidate_split_strong_boundary_routing_highmem_"
    "20260801/probe_evaluation/u000400_strict1607_report.json"
)
