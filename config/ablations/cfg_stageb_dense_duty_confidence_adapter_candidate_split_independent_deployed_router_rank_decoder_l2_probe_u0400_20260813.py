from config.ablations.cfg_stageb_dense_duty_confidence_adapter_candidate_split_independent_deployed_router_20260802 import *  # noqa: F401,F403

# Controlled V51 ablation: preserve the complete V51 objective and deployment
# surface, but let confidence losses adapt only the final two private rank
# decoder layers. The suffix has its own 0.1x LR and independent gradient clip.
stage_b_dense_duty_execution_scope = "probe"
stage_b_dense_duty_evaluation_scope = "probe"
stage_b_dense_duty_confidence_expected_optimizer_updates = 400
stage_b_dense_duty_confidence_probe_admission_contract = "disabled_for_probe_v1"
stage_b_dense_duty_confidence_probe_admission_report = ""
stage_b_dense_duty_confidence_rank_decoder_unfreeze_last_n = 2
stage_b_dense_duty_confidence_rank_decoder_lr = 2.0e-6
stage_b_v11_trainable_params_min = 4_155_806
stage_b_v11_trainable_params_max = 4_155_806
stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_v51_rank_decoder_l2_trace_audit_"
    "20260813/receipt.json"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "8f279dd783d86b7e562aa96e803242f7a452f64a4ddcf74b64127f9e51d6bdf6"
)
epochs = 2
