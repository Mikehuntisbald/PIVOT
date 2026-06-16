from config.ablations.cfg_stageb_v5_tnneg10_lse_top10_score_w025 import *  # noqa: F401,F403

# Interpolate between w0125_tau02 (better ref/ref+ accuracy, weak TN) and
# w025 (better TN, weaker ref/ref+ accuracy) without raising the positive guard.

stage_b_score_calib_neg_lse_tau = 0.2
stage_b_score_calib_detach_patch = False

stage_b_score_calib_neg_weight = 0.1875
stage_b_score_calib_gap_weight = 0.1875
