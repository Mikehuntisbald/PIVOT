from config.ablations.cfg_stageb_v21_token_matrix_base import *  # noqa: F401,F403

# L7: BCE loss-family control for L2. Query scope and all-negative labels are
# identical; only focal modulation/alpha are removed.
stage_b_v21_token_objective = "targetlocal_allneg_bce"
stage_b_v11_predicate_tn_rank_weight = 0.0
