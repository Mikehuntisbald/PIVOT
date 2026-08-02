from config.ablations.cfg_stageb_dense_duty_confidence_adapter_cross_attention_absolute_20260731 import *  # noqa: F401,F403

# v29 keeps v28's detached query-to-modifier cross-attention, but moves the
# absolute decision before sample pooling.  Every admitted candidate receives
# its own independent confidence logit, so positive/TN local-absolute losses
# directly train the same verifier whose maximum determines FPR95.
stage_b_dense_duty_confidence_revision = (
    "word_veto_candidate_absolute_confidence_v29"
)
stage_b_dense_duty_confidence_pool_feature_contract = (
    "detached_query_modifier_cross_attention_candidate_absolute_logits_v5"
)
stage_b_dense_duty_confidence_gate_gradient_contract = (
    "candidate_cross_attention_independent_absolute_logit_v10"
)
stage_b_v11_trainable_params_min = 535_943
stage_b_v11_trainable_params_max = 535_943

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_candidate_absolute_"
    "trace_audit_20260731"
)
# Filled only from a freshly regenerated receipt after the v29 source closure
# is final; the controller refuses to train when this binding drifts.
stage_b_dense_duty_trace_audit_sha256 = (
    "70c449010be3a4bf5b10db75558afb5ac09fe87592936e4017bbe82d145721d4"
)

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u300_word_veto_candidate_absolute_confidence_strict1607_v29"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_candidate_absolute_highmem_20260731/"
    "probe_evaluation/u000300_strict1607_report.json"
)
