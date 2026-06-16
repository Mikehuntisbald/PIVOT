from config.ablations.cfg_stageb_v5_tnneg10_lse_top10_score_w0125_tau02 import *  # noqa: F401,F403

# Add a light all-TN final-score constraint on top of the best accuracy-preserving
# logsumexp(top10) probe. This supervises every TN slot's actual inference score,
# not only sparse positive/TN rank pairs.

stage_b_score_calib_all_tn_neg_weight = 0.0625
