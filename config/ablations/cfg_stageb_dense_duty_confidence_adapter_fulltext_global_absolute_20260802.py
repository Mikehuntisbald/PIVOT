from config.ablations.cfg_stageb_dense_duty_confidence_adapter_candidate_sample_calibrator_20260802 import *  # noqa: F401,F403

# V53 has exactly two trainable confidence owners. Token-veto remains anchored
# by edit-token supervision. The full global-absolute owner starts from the
# frozen rank tower's complete phrase logit and jointly trains its candidate
# residual and sample pool with local absolute, global TN, q05, and tail losses.
stage_b_dense_duty_confidence_revision = (
    "word_veto_rank_full_expression_global_absolute_v53"
)
stage_b_dense_duty_confidence_head_gradient_contract = (
    "split_token_veto_fulltext_global_absolute_v7"
)
stage_b_dense_duty_confidence_pool_feature_contract = (
    "detached_rank_full_expression_candidate_residual_global_pool_v10"
)
stage_b_dense_duty_confidence_gate_gradient_contract = (
    "candidate_raw_patch_asymmetric_monotone_veto_absolute_logit_v13"
)
stage_b_v11_trainable_params_min = 534_725
stage_b_v11_trainable_params_max = 534_725

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_fulltext_global_absolute_"
    "trace_audit_20260802/receipt.json"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "a678bc115f4a9994a18804eb261fe2765b2f98b59d9e5de79de08b9392d83912"
)

stage_b_dense_duty_confidence_probe_admission_contract = (
    "u400_word_veto_rank_full_expression_global_absolute_"
    "confidence_strict1607_v53"
)
stage_b_dense_duty_confidence_probe_admission_report = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_fulltext_global_absolute_highmem_20260802/"
    "probe_evaluation/u000400_strict1607_report.json"
)
