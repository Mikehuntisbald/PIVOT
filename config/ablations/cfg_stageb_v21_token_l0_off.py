from config.ablations.cfg_stageb_v21_token_matrix_base import *  # noqa: F401,F403

# L0: no token objective and no predicate-pair auxiliary rank loss.
stage_b_v21_token_objective = "off"
stage_b_v11_predicate_tn_rank_weight = 0.0
