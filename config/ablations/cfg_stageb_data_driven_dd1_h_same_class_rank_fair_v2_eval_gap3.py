from config.ablations.cfg_stageb_data_driven_dd1_h_same_class_rank_fair_v2 import *  # noqa: F401,F403

# Headline inference: patch first limits category candidates, then full-text
# rank resolves the referent. This gap is fixed before formal evaluation.
stage_b_data_driven_category_gate = True
stage_b_data_driven_category_gate_max_gap = 3.0
stage_b_data_driven_eval_expected_optimizer_updates = 5020
