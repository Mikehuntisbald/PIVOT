from config.ablations.cfg_stageb_v21_token_matrix_base import *  # noqa: F401,F403

# L8: focal loss-family counterpart of L3. It uses the same target-local,
# edit-aware labels and flat per-token reduction.
stage_b_v21_token_objective = "edit_focal"
stage_b_v11_predicate_tn_rank_weight = 0.0
