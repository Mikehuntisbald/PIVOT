from config.ablations.cfg_stageb_v5_1_refcoco_patchpos_from_v5_alltn00625 import *  # noqa: F401,F403

# Stage-B v5.1 calibrated allTN setting.
#
# This is the no-aux counterpart of
# cfg_stageb_v5_2_refcoco_patchpos_aux_alltn_tau05605_w036.py. Keep the v5.1
# RefCOCO-family patch-positive CE and v5 decoder/box-head optimization scope,
# but align TN text supervision and allTN tail calibration with the calibrated
# pure-GDINO allTN run.

stage_b_text_loss_type = "allquery_focal_tn_empty_det"
stage_b_extra_iou_match_thr = 0.0

stage_b_score_calib_topk = 10
stage_b_score_calib_neg_agg = "logsumexp"
stage_b_score_calib_neg_lse_tau = 0.2
stage_b_score_calib_tau_neg = 0.5605
stage_b_score_calib_all_tn_neg_weight = 0.36

# v5.1 is intentionally the no-aux member of the calibrated v5.x pair.
aux_loss = False
use_checkpoint = False
stage_b_score_calib_aux_loss = False

# TN text supervision comes from the all-query focal all-negative row, not from
# the older matched-query TN BCE token losses. Keep content/canonical token
# weight at 1.0; amplify changed TN negative tokens by the TN phrase token
# count. For "white shirt man", this effective TN-neg weight is 3.
lambda_tn_neg = 1.0
lambda_tn_content = 1.0
lambda_tn_canonical = 1.0
stage_b_tn_neg_weight_mode = "token_count"
tn_content_target = 0.0
tn_canonical_target = 0.0
