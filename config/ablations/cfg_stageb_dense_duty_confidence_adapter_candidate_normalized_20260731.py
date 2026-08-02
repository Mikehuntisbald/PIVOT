from config.ablations.cfg_stageb_dense_duty_confidence_adapter_candidate_calibrated_20260731 import *  # noqa: F401,F403

# v31 removes patch absolute-scale inputs by construction. Patch still owns
# Top-50 admission and within-image standardized category evidence; only the
# detached token mismatch and coverage paths can veto cross-sample confidence.
# Their exact-zero parameterization gain equals the adapter bottleneck width
# (64), preserving a neutral U0 while avoiding scalar-gradient starvation.
stage_b_dense_duty_confidence_revision = (
    "word_veto_candidate_normalized_confidence_v31"
)
stage_b_dense_duty_confidence_pool_feature_contract = (
    "detached_candidate_absolute_normalized_patch_amplified_veto_logits_v7"
)
stage_b_dense_duty_confidence_gate_gradient_contract = (
    "candidate_normalized_patch_amplified_monotone_veto_absolute_logit_v12"
)
stage_b_v11_trainable_params_min = 535_945
stage_b_v11_trainable_params_max = 535_945

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_candidate_normalized_"
    "trace_audit_20260731"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "b7e61621e480bb494baa66b3a9947dd8e399bfbfdc92c58743aa0ce4ec7255dc"
)

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u400_word_veto_candidate_normalized_confidence_strict1607_v31"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_candidate_normalized_highmem_20260731/"
    "probe_evaluation/u000400_strict1607_report.json"
)
