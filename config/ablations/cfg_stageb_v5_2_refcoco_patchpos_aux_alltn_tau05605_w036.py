from config.ablations.cfg_stageb_v5_1_refcoco_patchpos_alltn_tau05605_w036 import *  # noqa: F401,F403

# Stage-B v5.2 normalized-score allTN setting.
#
# This is exactly the v5.1 normalized-score recipe plus decoder aux supervision.
# Score/allTN calibration follows aux as well, so TN samples participate in
# every supervised decoder layer.
aux_loss = True
use_checkpoint = False
stage_b_score_calib_aux_loss = True
