from config.ablations.cfg_stageb_dense_duty_confidence_adapter_veto_20260730 import *  # noqa: F401,F403

# CVPR word-veto revision v2. The adapter still inherits U6551 token semantics
# exactly at U0, but a trace-activated mismatch adds a one-sided lexical
# log-likelihood penalty instead of replacing the inherited confidence logit.
stage_b_dense_duty_confidence_revision = "word_veto_incremental_penalty_v2"
stage_b_dense_duty_confidence_phrase_aggregation = (
    "trace_activated_word_veto_penalty_v2"
)

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_veto_penalty_trace_audit_20260730/"
    "receipt.json"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "86b6797af4c78852beb5b25d9ecda2e1d5f861adecd12a7ce8bcee3d4377f937"
)

# A v2 formal run is intentionally fail-closed until its own probe and strict
# TN evaluation have produced a promotion receipt.
stage_b_dense_duty_confidence_probe_admission_contract = (
    "u300_word_veto_penalty_strict1607_v2"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_veto_penalty_highmem_20260730/probe_evaluation/"
    "u000300_strict1607_report.json"
)
