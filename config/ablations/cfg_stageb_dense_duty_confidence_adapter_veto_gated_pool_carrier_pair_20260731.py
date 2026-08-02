from config.ablations.cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_carrier_quarter_20260731 import *  # noqa: F401,F403

# v9 keeps the v8 quarter-carrier TN objective and adds an exact-carrier
# paired margin. The paired term separates positive/TN carriers without
# shifting their shared midpoint.
stage_b_dense_duty_confidence_revision = (
    "word_veto_gated_pool_carrier_pair_v9"
)
stage_b_dense_duty_raw_veto_query_scope = (
    "tn_all_admitted_carrier_balanced_positive_carrier_paired_v4"
)
stage_b_dense_duty_raw_veto_tn_carrier_balance = 0.25
stage_b_dense_duty_raw_veto_carrier_pair_weight = 0.25
stage_b_dense_duty_raw_veto_carrier_pair_margin = 0.25

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u300_word_veto_gated_pool_carrier_pair_strict1607_v9"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_veto_gated_pool_carrier_pair_highmem_20260731/"
    "probe_evaluation/u000300_strict1607_report.json"
)

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_veto_gated_pool_carrier_pair_"
    "trace_audit_20260731/receipt.json"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "f1f357cbaa626d5c6fa160d97961c6a557c1440511c57a2e60e698bb47f1e56f"
)
