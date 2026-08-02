from config.ablations.cfg_stageb_dense_duty_confidence_adapter_veto_gate_20260731 import *  # noqa: F401,F403

# Bounded architecture-promotion probe. It preserves the formal data, batch,
# loss, optimizer, U6551 lineage, and word-veto gate contracts exactly.
stage_b_dense_duty_execution_scope = "probe"
stage_b_dense_duty_evaluation_scope = "probe"
stage_b_dense_duty_confidence_expected_optimizer_updates = 300
stage_b_dense_duty_confidence_probe_admission_contract = "disabled_for_probe_v1"
stage_b_dense_duty_confidence_probe_admission_report = ""
epochs = 2
