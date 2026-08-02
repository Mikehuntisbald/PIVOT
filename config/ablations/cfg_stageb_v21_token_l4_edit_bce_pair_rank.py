from config.ablations.cfg_stageb_v21_token_matrix_base import *  # noqa: F401,F403

# L4: main flat per-token edit-aware objective plus the complementary
# predicate-pair rank.
stage_b_v21_token_objective = "edit_bce"
stage_b_v11_predicate_tn_rank_weight = 1.0
