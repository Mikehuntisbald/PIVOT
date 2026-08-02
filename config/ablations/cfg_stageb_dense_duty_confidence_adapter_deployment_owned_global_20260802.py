from config.ablations.cfg_stageb_dense_duty_confidence_adapter_fulltext_global_independent_absolute_20260802 import *  # noqa: F401,F403

# V56 keeps V55's deployed independent global residual exactly, but transfers
# complete ownership of its representation to deployed sample-global losses.
# The serialized candidate head is detached, frozen, and diagnostic only.
stage_b_dense_duty_confidence_revision = (
    "word_veto_rank_full_expression_deployment_owned_global_v56"
)
stage_b_dense_duty_confidence_head_gradient_contract = (
    "split_token_veto_deployment_owned_global_absolute_v9"
)
stage_b_dense_duty_confidence_pool_feature_contract = (
    "detached_rank_full_expression_deployment_owned_global_pool_v13"
)
stage_b_v14_local_absolute_weight = 0.0
stage_b_v11_trainable_params_min = 468_164
stage_b_v11_trainable_params_max = 468_164

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_deployment_owned_global_"
    "trace_audit_20260802/receipt.json"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "a42a7ab25a9c3277dfe62f287507f24a21903530a0ef12b20261c99343b21f2c"
)

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u400_word_veto_rank_full_expression_deployment_owned_global_"
    "confidence_strict1607_v56"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_deployment_owned_global_highmem_20260802/"
    "probe_evaluation/u000400_strict1607_report.json"
)
