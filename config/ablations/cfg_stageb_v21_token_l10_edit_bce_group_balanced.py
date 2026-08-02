from config.ablations.cfg_stageb_v21_token_matrix_base import *  # noqa: F401,F403

# L10: reduction control for L4. Each role is averaged independently before
# role weighting, reproducing the earlier group-balanced implementation.
stage_b_v21_token_objective = "edit_bce_group_balanced"
stage_b_v11_predicate_tn_rank_weight = 1.0
