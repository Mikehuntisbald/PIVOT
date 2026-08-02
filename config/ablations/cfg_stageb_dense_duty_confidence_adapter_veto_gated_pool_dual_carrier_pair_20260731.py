from config.ablations.cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_carrier_pair_20260731 import *  # noqa: F401,F403

# v10 keeps v9's paired carrier separation and gives the positive inference
# carrier the same explicit 0.25 mixture already used on the TN side.
stage_b_dense_duty_confidence_revision = (
    "word_veto_gated_pool_dual_carrier_pair_v10"
)
stage_b_dense_duty_raw_veto_query_scope = (
    "tn_all_admitted_dual_carrier_balanced_paired_v5"
)
stage_b_dense_duty_raw_veto_positive_carrier_balance = 0.25

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u300_word_veto_gated_pool_dual_carrier_pair_strict1607_v10"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_veto_gated_pool_dual_carrier_pair_highmem_20260731/"
    "probe_evaluation/u000300_strict1607_report.json"
)

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_veto_gated_pool_dual_carrier_pair_"
    "trace_audit_20260731/receipt.json"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "ac44cda794084b04c47ea60c8faa45b293121649a4d607fc43f8ac07fd622cbd"
)
