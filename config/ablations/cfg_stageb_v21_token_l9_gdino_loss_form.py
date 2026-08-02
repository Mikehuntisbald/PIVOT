from config.ablations.cfg_stageb_v21_token_matrix_base import *  # noqa: F401,F403

# L9: fixed-Top50 analogue of GDINO's dense token focal loss form: alpha=.25,
# gamma=2, sum over scorer-visible tokens, normalized by positive target-query
# count. It is not an exact 900-query GDINO architecture reproduction.
stage_b_v21_token_objective = "gdino_allquery_allneg_focal"
stage_b_v11_predicate_tn_rank_weight = 0.0
