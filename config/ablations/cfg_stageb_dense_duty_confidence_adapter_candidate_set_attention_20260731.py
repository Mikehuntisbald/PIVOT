from config.ablations.cfg_stageb_dense_duty_confidence_adapter_candidate_asymmetric_20260731 import *  # noqa: F401,F403

# v33 preserves v32 token/patch/veto calibration and strengthens only the
# detached sample-level candidate-set pool with four learned attention seeds.
stage_b_dense_duty_confidence_revision = (
    "word_veto_candidate_set_attention_confidence_v33"
)
stage_b_dense_duty_confidence_pool_feature_contract = (
    "detached_candidate_set_attention_absolute_asymmetric_veto_logits_v9"
)
stage_b_dense_duty_confidence_gate_gradient_contract = (
    "candidate_set_attention_asymmetric_monotone_veto_absolute_logit_v14"
)
stage_b_v11_trainable_params_min = 1_329_033
stage_b_v11_trainable_params_max = 1_329_033

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_candidate_set_attention_"
    "trace_audit_20260731"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "e0025e7b76730df6ed8d2997dc0c9c5326e2ee1ec63a2384bb3bae107701e401"
)

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u400_word_veto_candidate_set_attention_confidence_strict1607_v33"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_candidate_set_attention_highmem_20260731/"
    "probe_evaluation/u000400_strict1607_report.json"
)
