from config.ablations.cfg_stageb_v21_token_matrix_base import *  # noqa: F401,F403

# L1: all-query, all-negative focal with a flat mean over scorer-visible
# tokens. L9 separately preserves the original GDINO sum/positive-query
# normalization, so L1 -> L2 isolates query scope without a scale change.
stage_b_v21_token_objective = "allquery_allneg_focal"
stage_b_v11_predicate_tn_rank_weight = 0.0
