from config.ablations.cfg_stageb_dense_duty_confidence_adapter_candidate_absolute_20260731 import *  # noqa: F401,F403

# Convergence diagnostic for the unchanged v29 architecture. This starts
# fresh from the sealed U6551 rank source; it is not a continuation or a
# formal-training claim.
stage_b_dense_duty_execution_scope = "probe"
stage_b_dense_duty_evaluation_scope = "probe"
stage_b_dense_duty_confidence_expected_optimizer_updates = 600
stage_b_dense_duty_confidence_probe_admission_contract = "disabled_for_probe_v1"
stage_b_dense_duty_confidence_probe_admission_report = ""
epochs = 3
