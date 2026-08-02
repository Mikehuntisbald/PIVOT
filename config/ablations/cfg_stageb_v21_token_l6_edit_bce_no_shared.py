from config.ablations.cfg_stageb_v21_token_matrix_base import *  # noqa: F401,F403

# L6: L4 without the shared-context positive term.
stage_b_v21_token_objective = "edit_bce"
stage_b_v11_predicate_tn_rank_weight = 1.0
stage_b_v21_token_shared_weight = 0.0
