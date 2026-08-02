from config.ablations.cfg_stageb_dense_duty_confidence_adapter_candidate_absolute_20260731 import *  # noqa: F401,F403

# v30 preserves v29's candidate-specific cross-modal verifier while separating
# within-image category evidence from cross-sample confidence calibration.
# Three sign-constrained scalars start at exact zero, so U0 remains bitwise
# neutral: patch row maxima can only be removed and learned token mismatch /
# coverage evidence can only veto confidence.
stage_b_dense_duty_confidence_revision = (
    "word_veto_candidate_patch_invariant_confidence_v30"
)
stage_b_dense_duty_confidence_pool_feature_contract = (
    "detached_candidate_absolute_patch_invariant_monotone_veto_logits_v6"
)
stage_b_dense_duty_confidence_gate_gradient_contract = (
    "candidate_patch_invariant_monotone_veto_absolute_logit_v11"
)
stage_b_v11_trainable_params_min = 535_946
stage_b_v11_trainable_params_max = 535_946

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_candidate_calibrated_"
    "trace_audit_20260731"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "a4db552470c1c9d09f93e0db929c17649d9c881d9c96d97bfc47ab4a5bd6417e"
)

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u400_word_veto_candidate_patch_invariant_confidence_strict1607_v30"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_candidate_calibrated_highmem_20260731/"
    "probe_evaluation/u000400_strict1607_report.json"
)
