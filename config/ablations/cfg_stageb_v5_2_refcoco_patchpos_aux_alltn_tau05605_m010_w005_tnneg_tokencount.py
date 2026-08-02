from config.ablations.cfg_stageb_v5_1_refcoco_patchpos_alltn_tau05605_m018_w036_tnneg_tokencount import *  # noqa: F401,F403

# Stage-B v5.2 aux-aware calibrated allTN setting.
#
# v5.2 adds score/allTN calibration to the final decoder layer plus all aux
# layers, so use a smaller allTN weight than the no-aux v5.1 recipe while
# keeping tau_neg fixed.
aux_loss = True
use_checkpoint = False
stage_b_score_calib_aux_loss = True
stage_b_score_calib_margin = 0.10
stage_b_score_calib_all_tn_neg_weight = 0.05
