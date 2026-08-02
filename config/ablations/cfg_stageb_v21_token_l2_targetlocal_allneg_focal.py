from config.ablations.cfg_stageb_v21_token_matrix_base import *  # noqa: F401,F403

# L2: retain all-negative TN token labels but restrict them to IoU-positive
# target-local candidates.
stage_b_v21_token_objective = "targetlocal_allneg_focal"
stage_b_v11_predicate_tn_rank_weight = 0.0
