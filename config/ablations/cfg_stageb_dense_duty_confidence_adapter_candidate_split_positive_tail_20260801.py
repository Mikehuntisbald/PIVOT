from config.ablations.cfg_stageb_dense_duty_confidence_adapter_candidate_split_heads_20260801 import *  # noqa: F401,F403

# V46 is a fresh V44 ablation. It preserves the split-v2 owners and the
# deployed-routing mean objective, changing only positive trust to emphasize
# the lowest-confidence quarter of positive samples.
stage_b_dense_duty_confidence_revision = (
    "word_veto_candidate_split_positive_tail_v46"
)
stage_b_dense_duty_confidence_head_gradient_contract = (
    "split_token_veto_global_absolute_v2"
)
stage_b_dense_duty_deployed_veto_routing_weight = 0.1
stage_b_dense_duty_deployed_veto_routing_reduction_contract = "balanced_mean_v1"
stage_b_v15_tail_queue_positive_trust_reduction_contract = (
    "top_quarter_cvar_v2"
)
stage_b_dense_duty_positive_trust_contract = (
    "absolute_global_confidence_logit_v2"
)

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_candidate_split_positive_tail_"
    "trace_audit_20260801"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "ae59eee0791fa4b07377bda7e60bdc5e996ea6b04ed6b383d4491ac2690094f7"
)

# Formal admission remains unbound in main.py until the isolated U400 probe
# passes the strict1607 gate. The probe config below disables it entirely.
stage_b_dense_duty_confidence_probe_admission_contract = (
    "u400_word_veto_candidate_split_positive_tail_confidence_strict1607_v46"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_candidate_split_positive_tail_highmem_20260801/"
    "probe_evaluation/u000400_strict1607_report.json"
)
