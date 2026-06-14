from config.cfg_patch_stage_b import *  # noqa: F401,F403

# Stage-B v4: start from v2 checkpoint0003 and add inference-score calibration.
# This is not the historical v3 phrase-rank loss; v3 remains cfg_stageb_full.py.
stage_b_enable_phrase_rank = False
stage_b_rank_loss_coef = 0.0

# Values calibrated from v2 checkpoint0003 TN-val score quantiles at beta=1:
# outputs/stageb_tn_val_compare_quantiles_v2/v2_ckpt0003_beta1_score_quantiles.json
stage_b_score_calib_loss_coef = 1.0
stage_b_score_calib_tau_pos = 0.10
stage_b_score_calib_tau_neg = 1.40
stage_b_score_calib_margin = 0.30
stage_b_score_calib_topk = 10

# Weighted inside loss_score_calib:
# 0.5 * TN top-10 score rejection
# 0.1 * positive score floor
# 0.1 * matched positive-vs-TN gap
# 0.1 * positive matched query above other top-10 queries
stage_b_score_calib_neg_weight = 0.5
stage_b_score_calib_pos_weight = 0.1
stage_b_score_calib_gap_weight = 0.1
stage_b_score_calib_pos_query_weight = 0.1
stage_b_score_calib_detach_patch = True

