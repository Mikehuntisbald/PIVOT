from config.ablations.cfg_stageb_dense_duty_confidence_adapter_deployment_owned_query_veto_20260802 import *  # noqa: F401,F403

# V61 capacity upper bound: a second complete 256-dim, six-layer expression
# tower is initialized by an audited tensor-for-tensor copy of U6551 rank.
# Rank remains frozen. The verifier exposes token entailment and a bounded,
# non-negative veto depth only; it never emits a free absolute score.
stage_b_dense_duty_confidence_full_decoder_verifier = True
stage_b_dense_duty_confidence_capacity_contract = (
    "rank_cloned_full_decoder_6layer_256d_v1"
)
stage_b_dense_duty_confidence_variant = (
    "full_decoder_token_entailment_nonnegative_veto_capacity_upper_bound_v61"
)

# Full verifier tower (25,464,320) + zero-init veto head (66,561) +
# deployment-owned AbsoluteConfidencePool (133,377).
stage_b_v11_trainable_params_min = 25_664_258
stage_b_v11_trainable_params_max = 25_664_258

# A second full expression tower keeps activation memory rather than adapter
# width as the limiting resource. Start conservatively and raise this only
# after measured headroom.
stage_b_v11_expression_microbatch = 4
stage_b_dense_duty_expected_expression_microbatch = 4
stage_b_dense_duty_forward_pack_factor = 1
stage_b_dense_duty_expected_forward_batch_size = 16
stage_b_dense_duty_expected_physical_forwards_per_epoch = 887
stage_b_dense_duty_expected_gradient_accumulation_steps = 4

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_full_decoder_verifier_"
    "trace_audit_20260803/receipt.json"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "05cb271a0c894b39e8063edbd3fd6d9b83266cb6e424f16f0973cc7a28f29a95"
)
