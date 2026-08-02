from config.ablations.cfg_stageb_v11_fixed_text_scorer import *  # noqa: F401,F403

# Auxiliary target-local contrast on only the independently tokenized words that
# differ between the positive and paired TN expressions. The deployed score
# remains the single-expression full-phrase score.
stage_b_v11_predicate_tn_rank_weight = 1.0
stage_b_v11_predicate_tn_rank_margin = 0.3
