from config.ablations.cfg_stageb_dense_duty_confidence_full_decoder_verifier_20260803 import *  # noqa: F401,F403

# V62 removes the compensating text-rich AbsoluteConfidencePool from the
# deployed and trainable surface. Patch evidence supplies category-conditioned
# candidate weights; the independent full decoder supplies token entailment and
# a non-negative query veto. A normalized patch-weighted soft-min aggregates
# the existential query set, and the deployed logit is exactly -veto.
stage_b_dense_duty_confidence_veto_only_patch_softmin = True
stage_b_dense_duty_confidence_capacity_contract = (
    "rank_cloned_full_decoder_patch_softmin_veto_v2"
)
stage_b_dense_duty_confidence_variant = (
    "full_decoder_token_entailment_patch_weighted_existential_veto_v62"
)

# Full verifier tower (25,464,320) + zero-init veto head (66,561).
# The historical six-tensor pool remains serialized at zero for migration
# compatibility, but is frozen, dormant, and excluded from optimizer owners.
stage_b_v11_trainable_params_min = 25_530_881
stage_b_v11_trainable_params_max = 25_530_881

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_confidence_full_decoder_patch_softmin_veto_"
    "trace_audit_20260803/receipt.json"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "8ac86f33d3b8851d1bd76c12263fb7bb9fa3382f56232a33fd6255d159a397e2"
)
