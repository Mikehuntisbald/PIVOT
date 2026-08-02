from config.ablations.cfg_stageb_dense_duty_confidence_adapter_fulltext_global_absolute_exact_residual_20260802 import *  # noqa: F401,F403

# V55 removes every additive frozen-rank carrier from confidence. Candidate
# focal supervision owns a local absolute logit, while the pool's zero-init
# scalar is the complete deployed sample-global absolute confidence. Frozen
# rank remains available only as detached full-text features and pool weights.
stage_b_dense_duty_confidence_revision = (
    "word_veto_rank_full_expression_global_independent_absolute_v55"
)
stage_b_dense_duty_confidence_head_gradient_contract = (
    "split_token_veto_local_candidate_global_absolute_v8"
)
stage_b_dense_duty_confidence_pool_feature_contract = (
    "detached_rank_full_expression_local_candidate_"
    "frozen_rank_global_pool_v12"
)
stage_b_dense_duty_positive_trust_contract = "absolute_global_pool_logit_v4"

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_fulltext_global_independent_"
    "absolute_trace_audit_20260802/receipt.json"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "8f835b6c39deba306483e6562a6b78683aad8cea3e851c96775841171271933f"
)

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u400_word_veto_rank_full_expression_global_independent_absolute_"
    "confidence_strict1607_v55"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_fulltext_global_independent_absolute_highmem_"
    "20260802/probe_evaluation/u000400_strict1607_report.json"
)
