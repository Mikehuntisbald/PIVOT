from config.ablations.cfg_stageb_dense_duty_confidence_20260728 import *  # noqa: F401,F403

# CVPR final Stage-B: preserve the selected full rank tower bitwise and train
# only a stop-gradient token-aware confidence adapter plus AbsoluteConfidencePool.
stage_b_v22_score_ownership = "rank_tower_stopgrad_token_adapter_two_phase"
stage_b_dense_duty_confidence_adapter_dim = 64
stage_b_dense_duty_confidence_init_seed = 42
stage_b_dense_duty_confidence_token_contract = (
    "detached_rank_token_minus_zero_init_residual_v1"
)
stage_b_dense_duty_confidence_pool_feature_contract = "patch_statistics_only_v1"

# The selected U6551 rank model already exceeds the GDINO data-FT RefCOCO
# baselines.  This is an explicit architecture migration, not a same-code rank
# continuation and not a Stage-B teacher initialization.
stage_b_dense_duty_rank_expected_optimizer_updates = 6551
stage_b_dense_duty_rank_source_checkpoint_path = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/dense_duty_20260728/"
    "formal/rank/checkpoint_iter.pth"
)
stage_b_dense_duty_rank_source_checkpoint_sha256 = (
    "50e60a1314f7f2908bee5eea84ede5549b908177b367609efdec1682caa67ed3"
)
stage_b_dense_duty_rank_source_optimizer_updates = 6551
stage_b_dense_duty_rank_source_checkpoint_reason = "signal"
stage_b_dense_duty_rank_source_rank_sha256 = (
    "e03219d5868004aa5cb9ff4fe68f1aa94d33f1f0f6e1290cb251d12f9c914045"
)
stage_b_dense_duty_rank_source_transferred_sha256 = (
    "5300b52061b2f441346fb81334268bc7d192881c819773e2076d96a36070fe96"
)

# Same TN data, token roles, global-TN/FPR95 queue, q05 trust, effective batch,
# and 4,412 confidence updates as the sealed dense-duty confidence phase.
# Two consecutive B16 data batches share one B32 model forward. Criterion,
# global-TN/q05 statistics, and queue payloads remain split into logical B16
# groups; two packed forwards still form the same effective batch of 64.
stage_b_dense_duty_forward_pack_factor = 2
stage_b_dense_duty_logical_loss_batch_size = 16
stage_b_dense_duty_expected_forward_batch_size = 32
stage_b_dense_duty_expected_logical_batches_per_epoch = 887
stage_b_dense_duty_expected_physical_forwards_per_epoch = 444
stage_b_dense_duty_expected_gradient_accumulation_steps = 2
stage_b_v11_expression_microbatch = 64
stage_b_dense_duty_expected_expression_microbatch = 64
stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_adapter_packed_trace_audit_20260730/receipt.json"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "2204a5678ebcccb3817d58dac32376835bb8b8ef4baa4106c1c694eae5da8d86"
)

# Exact TokenAwareConfidenceAdapter + AbsoluteConfidencePool ownership surface.
stage_b_v11_trainable_params_min = 185_924
stage_b_v11_trainable_params_max = 185_924
