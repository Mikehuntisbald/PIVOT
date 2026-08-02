from config.ablations.cfg_stageb_dense_duty_confidence_adapter_20260730 import *  # noqa: F401,F403

# One-update real-model contract probe. Formal training always uses the sibling
# confidence_adapter_20260730 config and its B16/acc4/U4412 runtime.
stage_b_dense_duty_execution_scope = "probe"
stage_b_dense_duty_evaluation_scope = "probe"
stage_b_dense_duty_forward_pack_factor = 1
stage_b_dense_duty_logical_loss_batch_size = 1
stage_b_dense_duty_expected_forward_batch_size = 1
stage_b_v11_expression_microbatch = 2
batch_size = 1
epochs = 1
