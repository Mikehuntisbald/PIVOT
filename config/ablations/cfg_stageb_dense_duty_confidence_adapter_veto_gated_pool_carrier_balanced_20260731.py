from config.ablations.cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_calibrated_20260731 import *  # noqa: F401,F403

# v7 aligns raw token supervision with the exact frozen-rank carrier used by
# global confidence. TN loss balances all admitted queries with that carrier;
# positive carrier protection and the monotone v5 readout remain unchanged.
stage_b_dense_duty_confidence_revision = (
    "word_veto_gated_pool_carrier_balanced_v7"
)
stage_b_dense_duty_raw_veto_query_scope = (
    "tn_all_admitted_carrier_balanced_positive_carrier_v3"
)
stage_b_dense_duty_raw_veto_tn_carrier_balance = 0.5
stage_b_dense_duty_confidence_carrier_selector_contract = (
    "final_layer_reference_argmax_exact_eligible_v1"
)

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u300_word_veto_gated_pool_carrier_balanced_strict1607_v7"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_veto_gated_pool_carrier_balanced_highmem_20260731/"
    "probe_evaluation/u000300_strict1607_report.json"
)

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_veto_gated_pool_carrier_balanced_"
    "trace_audit_20260731/receipt.json"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "28d695be719fd4df8c39eae51989d49d60b32cf343814b8ef5b4f0b95cd1533b"
)
