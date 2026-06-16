from config.ablations.cfg_stageb_v5_tnneg10_lse_top10_score_w025 import *  # noqa: F401,F403

# Keep the best current score-calibration weights, but stop the calibration
# loss from moving the patch score branch. The loss still sees the final
# Stage-B score; gradients are applied through the text branch only.

stage_b_score_calib_detach_patch = True
