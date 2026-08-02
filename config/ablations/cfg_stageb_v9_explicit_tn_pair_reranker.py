from config.ablations.cfg_stageb_v8_candidate_text_reranker import *  # noqa: F401,F403

# Keep localization ranking and counterfactual text ranking as separate losses.
# The TN term compares the positive and edited phrase on the same frozen Stage-A
# candidate, so localization hard negatives cannot hide the text violation.
stage_b_v7_tn_pair_rank_loss_coef = 1.0
stage_b_v7_tn_pair_rank_margin = 0.3
stage_b_v7_tn_pair_rank_topk = 10
