from config.ablations.cfg_stageb_dense_duty_confidence_adapter_deployed_global_balanced_absolute_20260802 import *  # noqa: F401,F403

epochs = 24
stage_b_dense_duty_confidence_expected_optimizer_updates = 400
stage_b_dense_duty_max_train_iters = 400
stage_b_dense_duty_execution_scope = "probe"
stage_b_dense_duty_evaluation_scope = "probe"
stage_b_dense_duty_confidence_probe_admission_contract = "disabled_for_probe_v1"
stage_b_dense_duty_confidence_probe_admission_report = ""

output_dir = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "dense_duty_adapter_deployed_global_balanced_absolute_highmem_20260802/"
    "probe/u000400_fresh"
)
