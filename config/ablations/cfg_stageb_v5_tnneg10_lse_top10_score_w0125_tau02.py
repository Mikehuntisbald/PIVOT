from config.ablations.cfg_stageb_v5_tnneg10_lse_top10_score_w025 import *  # noqa: F401,F403

# Light hard-tail final-score calibration.
# w025 no-detach was the first probe to beat the v2 TN FPR, but it hurt
# ref/ref+ accuracy. Keep the same eval-aligned objective, make the top-10
# aggregate closer to max, and halve neg/gap weights to reduce positive drift.

stage_b_score_calib_neg_lse_tau = 0.2
stage_b_score_calib_detach_patch = False

stage_b_score_calib_neg_weight = 0.125
stage_b_score_calib_gap_weight = 0.125
