from config.ablations.cfg_stageb_dense_duty_confidence_adapter_independent_absolute_20260731 import *  # noqa: F401,F403

# v28 preserves v27's frozen rank tower, inherited token semantics, independent
# token BCE, and absolute (non-rank-carried) global confidence.  It replaces the
# parameter-free modifier-token mean with candidate-specific lightweight
# query-to-token cross-attention so token identity survives until verification.
stage_b_dense_duty_confidence_revision = (
    "word_veto_cross_attention_absolute_confidence_v28"
)
stage_b_dense_duty_confidence_pool_feature_contract = (
    "detached_rank_query_modifier_cross_attention_plus_"
    "patch_statistics_absolute_v4"
)
stage_b_dense_duty_confidence_gate_gradient_contract = (
    "cross_attention_independent_absolute_logit_v9"
)
stage_b_v11_trainable_params_min = 469_382
stage_b_v11_trainable_params_max = 469_382

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_cross_attention_absolute_"
    "trace_audit_20260731"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "612b788d588b933f4c810e9545cfff25612fa4a7d824287be09e134d14c36943"
)

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u300_word_veto_cross_attention_absolute_confidence_strict1607_v28"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_cross_attention_absolute_highmem_20260731/"
    "probe_evaluation/u000300_strict1607_report.json"
)
