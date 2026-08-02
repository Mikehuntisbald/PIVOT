from config.ablations.cfg_stageb_dense_duty_confidence_adapter_candidate_split_heads_20260801 import *  # noqa: F401,F403

# V45 keeps V44's rank source, split parameter owners, inference surface, data,
# and update budget. It aligns each owner's training reduction with its deployed
# failure mode while restoring the confidence branch's total clip norm to 0.1.
stage_b_dense_duty_confidence_revision = (
    "word_veto_candidate_split_tail_aligned_v45"
)
stage_b_dense_duty_confidence_head_gradient_contract = (
    "split_token_veto_global_absolute_joint_clip_v3"
)
stage_b_dense_duty_deployed_veto_routing_weight = 1.0
stage_b_dense_duty_deployed_veto_routing_reduction_contract = (
    "balanced_top_quarter_cvar_v2"
)
stage_b_v15_tail_queue_positive_trust_reduction_contract = (
    "top_quarter_cvar_v2"
)
stage_b_dense_duty_positive_trust_contract = (
    "absolute_global_confidence_logit_v2"
)

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_candidate_split_tail_aligned_"
    "trace_audit_20260801"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "7a99aadfc3d90ede244d100d6ebac1406ea11e75d35654b3c38d04ca412b6bd3"
)

# This target is intentionally unbound until the isolated U400 strict1607 gate
# passes. Probe configs below disable promotion entirely.
stage_b_dense_duty_confidence_probe_admission_contract = (
    "u400_word_veto_candidate_split_tail_aligned_confidence_strict1607_v45"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_candidate_split_tail_aligned_highmem_20260801/"
    "probe_evaluation/u000400_strict1607_report.json"
)
