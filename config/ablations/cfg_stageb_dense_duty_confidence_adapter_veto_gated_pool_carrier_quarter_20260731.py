from config.ablations.cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_carrier_balanced_20260731 import *  # noqa: F401,F403

# v8 keeps the v7 carrier-aligned objective but reduces the TN carrier mixture
# after the b=0.5 U50 screen showed excessive positive-carrier veto leakage.
stage_b_dense_duty_confidence_revision = (
    "word_veto_gated_pool_carrier_quarter_v8"
)
stage_b_dense_duty_raw_veto_tn_carrier_balance = 0.25

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u300_word_veto_gated_pool_carrier_quarter_strict1607_v8"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_veto_gated_pool_carrier_quarter_highmem_20260731/"
    "probe_evaluation/u000300_strict1607_report.json"
)

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_veto_gated_pool_carrier_quarter_"
    "trace_audit_20260731/receipt.json"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "64d4cff8cb94c39097a2a9fe3da6213c437f4c7f1ef64ff87257c5e253b3545f"
)
