from config.ablations.cfg_stageb_v21_token_matrix_base import *  # noqa: F401,F403

# L3: provenance-certified flat per-token edit-role BCE without predicate-pair
# rank. Role coefficients are applied to individual tokens before one common
# denominator; they are not separately normalized group means.
stage_b_v21_token_objective = "edit_bce"
stage_b_v11_predicate_tn_rank_weight = 0.0
