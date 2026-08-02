from config.ablations.cfg_stageb_dense_duty_confidence_adapter_candidate_calibrated_20260731 import *  # noqa: F401,F403

# v32 keeps raw patch statistics for absolute category evidence, normalizes the
# cross-sample patch carrier by the configured logit clip, and separates local
# changed-token veto strength from sample-tail coverage strength.  The token
# adapter remains stop-gradient with an exact-zero residual at U0.
stage_b_dense_duty_confidence_revision = (
    "word_veto_candidate_asymmetric_confidence_v32"
)
stage_b_dense_duty_confidence_pool_feature_contract = (
    "detached_candidate_absolute_raw_patch_asymmetric_veto_logits_v8"
)
stage_b_dense_duty_confidence_gate_gradient_contract = (
    "candidate_raw_patch_asymmetric_monotone_veto_absolute_logit_v13"
)
stage_b_v11_trainable_params_min = 535_945
stage_b_v11_trainable_params_max = 535_945

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_candidate_asymmetric_"
    "trace_audit_20260731"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "a9864ff77e84510e13c9f7d07d973a39d218d4370000c20fd31cfbd297e88152"
)

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u400_word_veto_candidate_asymmetric_confidence_strict1607_v32"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_candidate_asymmetric_highmem_20260731/"
    "probe_evaluation/u000400_strict1607_report.json"
)
