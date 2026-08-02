from config.ablations.cfg_stageb_dense_duty_confidence_adapter_candidate_split_boundary_routing_20260801 import *  # noqa: F401,F403

# V51 preserves the complete V47 boundary-routing objective and moves only its
# deployed routing residual into a third, independently clipped parameter owner.
stage_b_dense_duty_confidence_revision = (
    "word_veto_candidate_split_independent_deployed_router_v51"
)
stage_b_dense_duty_confidence_head_gradient_contract = (
    "split_token_veto_deployed_router_global_absolute_v5"
)
stage_b_dense_duty_deployed_veto_routing_weight = 0.1
stage_b_dense_duty_deployed_veto_routing_reduction_contract = (
    "balanced_top_quarter_cvar_v2"
)
stage_b_v15_tail_queue_positive_trust_reduction_contract = (
    "top_quarter_cvar_v2"
)
stage_b_v15_tail_queue_negative_reduction_contract = "all_mean_v1"
stage_b_v21_token_edit_query_scope = "target_iou_v1"
stage_b_v11_trainable_params_min = 536_734
stage_b_v11_trainable_params_max = 536_734

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_candidate_split_independent_"
    "deployed_router_trace_audit_20260802/receipt.json"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "4a51a2d9a79284763922747ba80f9588aafa2675467283ea06cd51798cebd027"
)

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u400_word_veto_candidate_split_independent_deployed_router_"
    "confidence_strict1607_v51"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_candidate_split_independent_deployed_router_highmem_"
    "20260802/probe_evaluation/u000400_strict1607_report.json"
)
