from config.ablations.cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_carrier_affine_20260731 import *  # noqa: F401,F403

# v17 leaves the v15 hard forward gate detached. Its existing 25% TN carrier
# budget is normalized over detached global-score q95 weights, so only exact
# trace-changed carrier words on FPR-tail samples receive the focused gradient.
stage_b_dense_duty_confidence_revision = (
    "word_veto_gated_pool_tail_carrier_v17"
)
stage_b_dense_duty_confidence_gate_gradient_contract = "hard_detached_v1"
stage_b_dense_duty_raw_veto_query_scope = (
    "tn_all_admitted_tail_weighted_carrier_positive_carrier_paired_v6"
)
stage_b_dense_duty_raw_veto_tail_quantile = 0.95
stage_b_dense_duty_raw_veto_tail_temperature = 0.1
stage_b_dense_duty_raw_veto_tail_min_count = 256

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u300_word_veto_gated_pool_tail_carrier_strict1607_v17"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_veto_gated_pool_tail_carrier_highmem_20260731/"
    "probe_evaluation/u000300_strict1607_report.json"
)

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_veto_gated_pool_tail_carrier_"
    "trace_audit_20260731/receipt.json"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "8deb4b3601851f838451f18a2340dc340a6d2f8d148d1dba4b7bcb6766b0ea32"
)
