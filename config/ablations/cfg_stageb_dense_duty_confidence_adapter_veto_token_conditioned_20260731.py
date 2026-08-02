from config.ablations.cfg_stageb_dense_duty_confidence_adapter_veto_monotone_depth_20260731 import *  # noqa: F401,F403

# v23 keeps the frozen rank tower, inherited token logits, independent token
# residual BCE, and one-sided monotone confidence veto. Unlike v22, the global
# veto-depth pool receives the full modifier-token identity and per-candidate
# inherited/learned token evidence instead of only a scalar max-residual gate.
stage_b_dense_duty_confidence_revision = (
    "word_veto_token_conditioned_monotone_depth_v23"
)
stage_b_dense_duty_confidence_pool_feature_contract = (
    "detached_rank_query_token_context_plus_patch_statistics_monotone_v3"
)

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_veto_token_conditioned_"
    "trace_audit_20260731"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "5f960c46e7539bba183357aa436de80cf40923cf6edd6ebbfe57f71fee98a191"
)

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u300_word_veto_token_conditioned_monotone_depth_strict1607_v23"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_veto_token_conditioned_highmem_20260731/"
    "probe_evaluation/u000300_strict1607_report.json"
)
