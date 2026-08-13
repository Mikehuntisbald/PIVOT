from config.ablations.cfg_stageb_dense_duty_confidence_full_decoder_patch_softmin_veto_20260803 import *  # noqa: F401,F403

# C1 isolates the candidate-coverage hypothesis while retaining V62's free
# non-negative verifier head. The exact deployed Top-50 receives direct depth
# supervision; changed-token labels expand only when the row carries explicit
# global-word-absence or per-candidate provenance.
stage_b_dense_duty_confidence_candidate_trace_contract = (
    "candidate_complete_free_head_coverage_v1"
)
stage_b_dense_duty_confidence_capacity_contract = (
    "rank_cloned_full_decoder_candidate_complete_free_head_v3"
)
stage_b_dense_duty_confidence_variant = (
    "candidate_complete_trace_free_head_coverage_c1"
)
stage_b_dense_duty_confidence_token_depth_base_scale = 1.0

stage_b_v21_token_objective = "edit_bce_group_balanced"
stage_b_v21_token_edit_query_scope = "candidate_complete_trace_v4"

stage_b_dense_duty_candidate_depth_all_weight = 1.0
stage_b_dense_duty_candidate_depth_escape_weight = 1.0
stage_b_dense_duty_candidate_depth_positive_weight = 1.0
stage_b_dense_duty_candidate_depth_tn_margin = 0.5
stage_b_dense_duty_candidate_depth_escape_margin = 0.5
stage_b_dense_duty_candidate_depth_positive_max = 0.05
stage_b_dense_duty_candidate_depth_temperature = 0.1

# C1 keeps the V62 active topology: verifier tower plus the free veto head.
stage_b_v11_trainable_params_min = 25_530_881
stage_b_v11_trainable_params_max = 25_530_881

stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_candidate_complete_trace_audit_20260803/receipt.json"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "ca19a27da8154a0accb4d959940123add62fc1e2642f7b2f73b43f420c07ef18"
)
