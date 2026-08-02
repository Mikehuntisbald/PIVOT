from config.ablations.cfg_stageb_dense_duty_confidence_adapter_candidate_gate_zero_offset_20260801 import *  # noqa: F401,F403

# v43 keeps v39's scorer, data, update budget, and bidirectional carrier-pair
# loss. The exact deployed hard gates retain their forward values but expose a
# smooth backward path to a bounded winner/coverage routing objective.
stage_b_dense_duty_confidence_revision = (
    "word_veto_candidate_asymmetric_deployed_routing_v43"
)
stage_b_dense_duty_confidence_gate_gradient_contract = (
    "candidate_raw_patch_asymmetric_deployed_routing_st_v15"
)
stage_b_dense_duty_deployed_veto_routing_weight = 0.1
stage_b_dense_duty_deployed_veto_positive_max = 0.1
stage_b_dense_duty_deployed_veto_tn_min = 0.9
stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract = "bidirectional_v1"

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_candidate_deployed_routing_"
    "trace_audit_20260801"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "9d131174f40827d38e27768eb02343a1067e1479a3c6de626bb0d270c09e9d58"
)

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u400_word_veto_candidate_deployed_routing_confidence_strict1607_v43"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_candidate_deployed_routing_highmem_20260801/"
    "probe_evaluation/u000400_strict1607_report.json"
)
