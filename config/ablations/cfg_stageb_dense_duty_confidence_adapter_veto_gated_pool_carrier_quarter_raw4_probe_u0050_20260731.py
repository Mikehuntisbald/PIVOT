from config.ablations.cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_carrier_quarter_20260731 import *  # noqa: F401,F403

# Development-only U50 screen: keep the v8 objective and increase only the
# absolute raw-veto loss scale so its margins settle earlier under grad clip.
stage_b_dense_duty_raw_veto_gate_weight = 4.0

stage_b_dense_duty_execution_scope = "probe"
stage_b_dense_duty_evaluation_scope = "probe"
stage_b_dense_duty_confidence_expected_optimizer_updates = 50
stage_b_dense_duty_confidence_probe_admission_contract = "disabled_for_probe_v1"
stage_b_dense_duty_confidence_probe_admission_report = ""
epochs = 1
