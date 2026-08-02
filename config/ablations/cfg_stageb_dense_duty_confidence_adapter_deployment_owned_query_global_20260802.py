from config.ablations.cfg_stageb_dense_duty_confidence_adapter_deployment_owned_global_20260802 import *  # noqa: F401,F403

# V59 keeps V56's deployment-owned full-text representation, but makes the
# zero-initialized per-query absolute head part of the deployed sample-global
# score. Frozen rank supplies detached query preferences to a normalized,
# monotone log-sum-exp; no local candidate objective owns any trunk parameter.
stage_b_dense_duty_confidence_revision = (
    "word_veto_rank_full_expression_deployment_owned_query_global_v59"
)
stage_b_dense_duty_confidence_head_gradient_contract = (
    "split_token_veto_deployment_owned_query_global_absolute_v10"
)
stage_b_dense_duty_confidence_pool_feature_contract = (
    "detached_rank_full_expression_monotone_query_"
    "deployment_owned_global_pool_v14"
)
stage_b_dense_duty_positive_trust_contract = (
    "absolute_global_confidence_logit_v2"
)
stage_b_v15_tail_queue_negative_reduction_contract = "all_mean_v1"
stage_b_v14_local_absolute_weight = 0.0
stage_b_dense_duty_deployed_global_absolute_weight = 0.0
stage_b_v11_trainable_params_min = 534_725
stage_b_v11_trainable_params_max = 534_725

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_deployment_owned_query_global_"
    "trace_audit_20260802/receipt.json"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "197c1fb2d6680b9f1785c0f2c36eb053bbf13922712ed438ce88267f33c13396"
)

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u400_word_veto_rank_full_expression_deployment_owned_query_"
    "global_confidence_strict1607_v59"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_deployment_owned_query_global_highmem_20260802/"
    "probe_evaluation/u000400_strict1607_report.json"
)
