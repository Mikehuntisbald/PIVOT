from config.ablations.cfg_stageb_v5_2_refcoco_patchpos_aux_from_v5_alltn00625 import *  # noqa: F401,F403

# Stage-B v5.2 calibrated allTN setting.
#
# Keep the established v5.2 recipe:
# - all-query focal positives, with TN rows as empty-det/all-negative focal rows
# - decoder/box-head unfreeze, patch branch frozen
# - bbox/GIoU losses
# - RefCOCO-family patch CE positive-only
# - decoder aux losses, including TN/allTN score calibration on every aux layer
#
# Synchronize the allTN tail-suppression setting with the calibrated pure-GDINO
# allTN run. In Stage-B v5.x this term is part of loss_score_calib and operates
# on the final Stage-B slot score, so the parameter names differ from the
# pure-GDINO loss_tn_alltn config.

stage_b_text_loss_type = "allquery_focal_tn_empty_det"
stage_b_extra_iou_match_thr = 0.0

stage_b_score_calib_topk = 10
stage_b_score_calib_neg_agg = "logsumexp"
stage_b_score_calib_neg_lse_tau = 0.2
stage_b_score_calib_tau_neg = 0.5605
stage_b_score_calib_all_tn_neg_weight = 0.36
stage_b_score_calib_aux_loss = True

# TN text supervision now comes from the all-query focal all-negative row, not
# from the older matched-query TN BCE token losses.
lambda_tn_neg = 1.0
lambda_tn_content = 0.0
lambda_tn_canonical = 0.0
tn_content_target = 0.0
tn_canonical_target = 0.0
