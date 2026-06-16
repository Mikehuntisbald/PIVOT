from config.ablations.cfg_stageb_v5_tnneg10_lse_top10_score_w025 import *  # noqa: F401,F403

# Middle point between w0125_tau02 (better acc, slightly weak TN) and w025
# (better TN, weaker acc). Use a harder logsumexp tail and a stronger positive
# guard to keep the ref score distribution from collapsing.

stage_b_score_calib_neg_lse_tau = 0.2
stage_b_score_calib_detach_patch = False

stage_b_score_calib_neg_weight = 0.1875
stage_b_score_calib_gap_weight = 0.1875
stage_b_score_calib_pos_weight = 0.10
stage_b_score_calib_pos_query_weight = 0.10
