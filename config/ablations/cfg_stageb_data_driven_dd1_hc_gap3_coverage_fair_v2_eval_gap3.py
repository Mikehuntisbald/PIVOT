from config.ablations.cfg_stageb_data_driven_dd1_hc_gap3_coverage_fair_v2 import *  # noqa: F401,F403

# Fixed headline route: patch narrows category candidates and full-text rank
# chooses the referent inside the exact same Gap3 set supervised by DD1-HC.
stage_b_data_driven_category_gate = True
stage_b_data_driven_category_gate_max_gap = 3.0
stage_b_data_driven_patch_score_clip = 5.0
stage_b_data_driven_eval_expected_optimizer_updates = 5020
