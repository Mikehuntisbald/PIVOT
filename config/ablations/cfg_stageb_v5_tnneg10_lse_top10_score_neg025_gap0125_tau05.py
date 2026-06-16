from config.ablations.cfg_stageb_v5_tnneg10_lse_top10_score_w025 import *  # noqa: F401,F403

# Start from the only logsumexp(top10) final-score probe that beat v2 TN FPR
# (w025/tau0.5), and reduce only the pairwise gap term to limit positive-score
# drift while keeping the hard TN-tail pressure unchanged.

stage_b_score_calib_neg_lse_tau = 0.5
stage_b_score_calib_detach_patch = False

stage_b_score_calib_neg_weight = 0.25
stage_b_score_calib_gap_weight = 0.125
