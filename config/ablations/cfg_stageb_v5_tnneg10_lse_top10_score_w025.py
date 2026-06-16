from config.ablations.cfg_stageb_v5_allquery_focal_tnneg10_decoder_bbox import *  # noqa: F401,F403

# Stage-B v5 tnneg10 + inference-score TN calibration probe.
# Penalize the global logsumexp(top10) final Stage-B score on TN prompts while
# keeping the v5/tnneg10 token objective unchanged.

stage_b_score_calib_loss_coef = 1.0
stage_b_score_calib_topk = 10
stage_b_score_calib_neg_agg = "logsumexp"
stage_b_score_calib_neg_lse_tau = 0.5
stage_b_score_calib_detach_patch = False

stage_b_score_calib_tau_pos = 0.1
stage_b_score_calib_tau_neg = -2.4
stage_b_score_calib_margin = 0.8
stage_b_score_calib_pos_weight = 0.05
stage_b_score_calib_neg_weight = 0.25
stage_b_score_calib_gap_weight = 0.25
stage_b_score_calib_pos_query_weight = 0.05
