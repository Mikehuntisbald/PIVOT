from config.ablations.cfg_stageb_dense_duty_confidence_adapter_20260730 import *  # noqa: F401,F403

# Real-model packed-forward probe. It keeps the formal B16 logical loss groups,
# B32 forward, E64 tower chunk, and accumulation-two update composition.
stage_b_dense_duty_execution_scope = "probe"
stage_b_dense_duty_evaluation_scope = "probe"
epochs = 1
