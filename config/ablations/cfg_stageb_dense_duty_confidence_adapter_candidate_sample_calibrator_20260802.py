from config.ablations.cfg_stageb_dense_duty_confidence_adapter_candidate_split_independent_deployed_router_20260802 import *  # noqa: F401,F403

# V52 is a fresh U6551 migration, not a continuation of trained V51 confidence
# weights. It keeps the exact V47/V51 deployed score value at U0 while removing
# the router objective and assigning three disjoint gradient owners: token veto,
# query-local candidate absolute confidence, and sample-global calibration.
stage_b_dense_duty_confidence_revision = (
    "word_veto_candidate_sample_calibrator_split_v52"
)
stage_b_dense_duty_confidence_head_gradient_contract = (
    "split_token_veto_candidate_absolute_sample_calibrator_v6"
)
stage_b_dense_duty_deployed_veto_routing_weight = 0.0
stage_b_dense_duty_deployed_veto_routing_reduction_contract = (
    "balanced_top_quarter_cvar_v2"
)
stage_b_v15_tail_queue_positive_trust_reduction_contract = (
    "top_quarter_cvar_v2"
)
stage_b_v15_tail_queue_negative_reduction_contract = "all_mean_v1"
stage_b_v21_token_edit_query_scope = "target_iou_v1"
stage_b_v11_trainable_params_min = 535_945
stage_b_v11_trainable_params_max = 535_945

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_candidate_sample_calibrator_"
    "trace_audit_20260802/receipt.json"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "99c3123139d78ade289c9b1d84d48c5854d71d60b0621ce398b8366660b05120"
)

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u400_word_veto_candidate_sample_calibrator_confidence_strict1607_v52"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_candidate_sample_calibrator_highmem_20260802/"
    "probe_evaluation/u000400_strict1607_report.json"
)
