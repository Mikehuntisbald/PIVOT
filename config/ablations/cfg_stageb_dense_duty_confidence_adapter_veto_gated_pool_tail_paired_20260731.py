from config.ablations.cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_tail_carrier_20260731 import *  # noqa: F401,F403

# v18 keeps the exact detached hard gate and v17's q95-weighted changed-carrier
# objective. The same detached tail weights now focus the paired carrier margin,
# so a hard TN is contrasted against its trace-linked positive carrier instead
# of receiving only a one-sided source update.
stage_b_dense_duty_confidence_revision = (
    "word_veto_gated_pool_tail_paired_v18"
)
stage_b_dense_duty_raw_veto_query_scope = (
    "tn_all_admitted_tail_weighted_carrier_tail_paired_v7"
)

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u300_word_veto_gated_pool_tail_paired_strict1607_v18"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_veto_gated_pool_tail_paired_highmem_20260731/"
    "probe_evaluation/u000300_strict1607_report.json"
)

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_veto_gated_pool_tail_paired_"
    "trace_audit_20260731/receipt.json"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "a5834ad5ccc89d975d11da39dce20dbd5af0e59d9652d4fbfd499bd5448f55b3"
)
