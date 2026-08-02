from config.ablations.cfg_stageb_dense_duty_confidence_adapter_veto_token_conditioned_20260731 import *  # noqa: F401,F403

# v27 makes the named AbsoluteConfidencePool genuinely absolute. Rank logits
# remain detached inputs/statistics but are no longer the output carrier:
# confidence is the pool head's independent sample logit. Token confidence
# still exactly inherits rank_token_logits minus its zero-init edit residual.
stage_b_dense_duty_confidence_revision = (
    "word_veto_independent_absolute_confidence_v27"
)
stage_b_dense_duty_confidence_gate_gradient_contract = (
    "token_conditioned_independent_absolute_logit_v8"
)
stage_b_dense_duty_positive_trust_contract = (
    "absolute_global_confidence_logit_v2"
)
stage_b_v15_tail_queue_positive_trust_weight = 1.0

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_independent_absolute_"
    "trace_audit_20260731"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "12ad86cdacb2fad6194ab4462a433c933402f5e11694d10eb45799391cd48686"
)

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u300_word_veto_independent_absolute_confidence_strict1607_v27"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_independent_absolute_highmem_20260731/"
    "probe_evaluation/u000300_strict1607_report.json"
)
