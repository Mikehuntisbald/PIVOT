from config.ablations.cfg_stageb_dense_duty_confidence_adapter_20260730 import *  # noqa: F401,F403

# CVPR confidence revision: the trace-supervised token residual activates a
# lexical-word veto. Patch confidence and text confidence compose as a
# probability conjunction, so a mismatched modifier cannot be canceled by a
# high category score. The rank tower and its inference score remain unchanged.
stage_b_dense_duty_confidence_revision = "word_veto_net_trust_v1"
stage_b_dense_duty_confidence_phrase_aggregation = (
    "trace_activated_word_veto_product_v1"
)
stage_b_dense_duty_confidence_word_softmin_temperature = 0.1
stage_b_dense_duty_confidence_veto_gate_scale = 1.0
stage_b_dense_duty_positive_trust_contract = "net_total_confidence_delta_v1"
stage_b_dense_duty_confidence_tn_scope = "direct_trace_valid_v1"

# Formal promotion is allowed only after the same U300 architecture passes the
# sealed loss-health gate and the TN-only strict1607 diagnostic. The verified
# audit is rebound inside main.py and persisted in every formal checkpoint.
stage_b_dense_duty_confidence_probe_admission_contract = (
    "u300_word_veto_strict1607_v1"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_veto_highmem_20260730/probe_evaluation/"
    "u000300_strict1607_report.json"
)

# The direct-trace rows are unchanged, but this receipt rebinds their exact
# token-role audit to the current promotion-gated training source closure.
stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_veto_trace_audit_20260730/receipt.json"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "60047aa1a8b89e6789ef4e4817e4ce9c23e8fdb457e25bdde65daada07250da6"
)

# A broadcast sample-global confidence makes the old candidate mean and top-k
# tail losses identical. Keep one mild absolute TN term; the exact FPR95 queue
# remains the primary cross-sample objective.
stage_b_v11_global_tn_negative_weight = 0.25
stage_b_v11_global_tn_tail_weight = 0.0
