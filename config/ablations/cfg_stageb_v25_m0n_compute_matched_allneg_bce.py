from config.ablations.cfg_stageb_v25_m0_compute_matched import *  # noqa: F401,F403

# Compute-matched validation-only control for the fixed M0 headline row.  This
# changes the complete token objective, not labels alone: every target-local
# positive token and every negative token logit is supervised by BCE, while
# the complementary predicate-pair rank term remains active.
stage_b_v25_main_id = "M0N"
stage_b_v25_control_of = "M0"
stage_b_v25_headline_eligible = False
stage_b_v25_matrix_validation_only = True
stage_b_v25_comparison_claim = "full_token_objective_control_not_labels_only"
stage_b_v25_token_objective_scope = (
    "target_local_positive_and_all_negative_token_logits"
)

stage_b_v21_token_objective = "targetlocal_allneg_bce"
stage_b_v11_predicate_tn_rank_weight = 1.0
