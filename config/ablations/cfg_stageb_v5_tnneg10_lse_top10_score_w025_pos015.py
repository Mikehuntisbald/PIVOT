from config.ablations.cfg_stageb_v5_tnneg10_lse_top10_score_w025 import *  # noqa: F401,F403

# w025 already crossed the v2 TN FPR baseline, but lost ref/ref+ acc50.
# Keep the same TN hard-tail objective and strengthen the positive guards so
# the 95% TPR threshold and matched-query ranking do not drift down.

stage_b_score_calib_pos_weight = 0.15
stage_b_score_calib_pos_query_weight = 0.10
