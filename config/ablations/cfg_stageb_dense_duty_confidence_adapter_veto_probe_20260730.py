from config.ablations.cfg_stageb_dense_duty_confidence_adapter_veto_20260730 import *  # noqa: F401,F403

# Bounded, non-paper training probe with the same B16 logical losses, packed
# B32 forward, E64 expression chunk, and effective batch 64 as the formal run.
stage_b_dense_duty_execution_scope = "probe"
stage_b_dense_duty_evaluation_scope = "probe"
stage_b_dense_duty_confidence_expected_optimizer_updates = 300
stage_b_dense_duty_confidence_probe_admission_contract = "disabled_for_probe_v1"
stage_b_dense_duty_confidence_probe_admission_report = ""
epochs = 2
