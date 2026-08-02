from config.ablations.cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_carrier_pair_20260731 import *  # noqa: F401,F403

# v11 keeps v9's exact-carrier paired supervision. A single zero-initialized
# scalar lets the veto residual reuse frozen rank-token semantics immediately;
# the rank tower remains detached and frozen.
stage_b_dense_duty_confidence_revision = (
    "word_veto_gated_pool_rank_evidence_v11"
)
stage_b_dense_duty_confidence_rank_evidence_contract = (
    "zero_init_rank_logit_scale_v1"
)
stage_b_dense_duty_raw_veto_query_scope = (
    "tn_all_admitted_carrier_balanced_positive_carrier_paired_v4"
)
stage_b_dense_duty_raw_veto_tn_carrier_balance = 0.25
stage_b_dense_duty_raw_veto_positive_carrier_balance = 0.0
stage_b_dense_duty_raw_veto_carrier_pair_weight = 0.25
stage_b_dense_duty_raw_veto_carrier_pair_margin = 0.25
stage_b_v11_trainable_params_min = 185_926
stage_b_v11_trainable_params_max = 185_926

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u300_word_veto_gated_pool_rank_evidence_strict1607_v11"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_veto_gated_pool_rank_evidence_highmem_20260731/"
    "probe_evaluation/u000300_strict1607_report.json"
)

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_veto_gated_pool_rank_evidence_"
    "trace_audit_20260731/receipt.json"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "cd84ee0b26980ba423ba1d22e681a7dccfd739a75c79342e748e1def6d32484d"
)
