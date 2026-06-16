from config.ablations.cfg_stageb_v5_tnneg10_lse_top10_score_w025 import *  # noqa: F401,F403

# Keep the stronger global TN top-10 pressure from w025, but reduce the
# pairwise gap term that can pull positive/ref score distributions around.

stage_b_score_calib_neg_lse_tau = 0.2
stage_b_score_calib_detach_patch = False

stage_b_score_calib_neg_weight = 0.25
stage_b_score_calib_gap_weight = 0.125
