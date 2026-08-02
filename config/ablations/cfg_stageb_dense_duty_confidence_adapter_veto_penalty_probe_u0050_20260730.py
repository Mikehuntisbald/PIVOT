from config.ablations.cfg_stageb_dense_duty_confidence_adapter_veto_penalty_20260730 import *  # noqa: F401,F403

# Short fail-fast trajectory screen. It uses the exact formal model, data,
# losses, packed B32 forward, E64 chunking, and effective batch 64.
stage_b_dense_duty_execution_scope = "probe"
stage_b_dense_duty_evaluation_scope = "probe"
stage_b_dense_duty_confidence_expected_optimizer_updates = 50
stage_b_dense_duty_confidence_probe_admission_contract = "disabled_for_probe_v1"
stage_b_dense_duty_confidence_probe_admission_report = ""
epochs = 1
