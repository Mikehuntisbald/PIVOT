from config.ablations.cfg_stageb_dense_duty_confidence_adapter_deployment_owned_global_20260802 import *  # noqa: F401,F403

# V58 preserves V56's deployment-owned representation and removes gradients
# from TNs already rejected by the exact historical positive-q05 boundary.
# Unlike the historical v1 active mean, normalization stays tied to all valid
# TNs so the gradient scale cannot grow as the active set shrinks.
stage_b_dense_duty_confidence_revision = (
    "word_veto_rank_full_expression_deployment_owned_global_"
    "stable_fpr95_active_set_v58"
)
stage_b_v15_tail_queue_negative_reduction_contract = (
    "exact_fpr95_active_set_all_count_mean_v2"
)
stage_b_dense_duty_deployed_global_absolute_weight = 0.0
stage_b_dense_duty_deployed_global_absolute_gamma = 1.0

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_deployment_owned_global_"
    "stable_fpr95_active_set_trace_audit_20260802/receipt.json"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "f4aa72e7c9fae7b740e4b0836c7b251b8873c9b7cc2b38a8c1f95732af1e8aa4"
)

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u400_word_veto_rank_full_expression_deployment_owned_global_"
    "stable_fpr95_active_set_confidence_strict1607_v58"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_deployment_owned_global_stable_fpr95_active_set_"
    "highmem_20260802/probe_evaluation/u000400_strict1607_report.json"
)
