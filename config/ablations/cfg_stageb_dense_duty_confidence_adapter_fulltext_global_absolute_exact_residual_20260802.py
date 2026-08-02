from config.ablations.cfg_stageb_dense_duty_confidence_adapter_fulltext_global_absolute_20260802 import *  # noqa: F401,F403

# V54 keeps the complete V53 deployed absolute-confidence path and parameter
# surface. Only positive-tail trust is measured as the learned residual above
# the exact detached frozen-rank candidate maximum. TN, pair, queue, and
# inference continue to use the deployed absolute confidence logit.
stage_b_dense_duty_confidence_revision = (
    "word_veto_rank_full_expression_global_absolute_exact_residual_v54"
)
stage_b_dense_duty_confidence_pool_feature_contract = (
    "detached_rank_full_expression_candidate_residual_global_pool_"
    "exact_rank_max_reference_v11"
)
stage_b_dense_duty_positive_trust_contract = (
    "exact_frozen_rank_max_confidence_delta_v3"
)

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_fulltext_global_absolute_"
    "exact_residual_trace_audit_20260802/receipt.json"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "70749fce1aed14db91d141e60d482797682b34e279c3719b973dfdd290c8924b"
)

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u400_word_veto_rank_full_expression_global_absolute_exact_residual_"
    "confidence_strict1607_v54"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_fulltext_global_absolute_exact_residual_highmem_"
    "20260802/probe_evaluation/u000400_strict1607_report.json"
)
