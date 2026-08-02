from config.ablations.cfg_stageb_dense_duty_confidence_adapter_veto_cap_20260731 import *  # noqa: F401,F403

# CVPR confidence revision v5. The frozen reference argmax is the expression
# carrier. Its detached changed-word gate alone decides whether confidence may
# leave the U6551 reference; the shared pool only learns veto depth after that.
stage_b_dense_duty_confidence_revision = (
    "word_veto_gated_pool_absolute_cap_v5"
)
stage_b_dense_duty_confidence_phrase_aggregation = (
    "trace_activated_word_veto_gated_pool_absolute_cap_v5"
)

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u300_word_veto_gated_pool_absolute_cap_strict1607_v5"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_veto_gated_pool_highmem_20260731/probe_evaluation/"
    "u000300_strict1607_report.json"
)

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_veto_gated_pool_trace_audit_20260731/"
    "receipt.json"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "7deeca871c010731a30f4e426a454e3041796f455778b88662b6cf67bac6766b"
)
