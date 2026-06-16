from config.ablations.cfg_stageb_v5_tnneg10_lse_top10_score_w025_detach import *  # noqa: F401,F403

# Make logsumexp(top10) closer to a hard max so the loss focuses on the
# highest-scoring TN queries instead of lowering the whole top-10 set.

stage_b_score_calib_neg_lse_tau = 0.2
