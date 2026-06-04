from config.cfg_patch_stage_b import *  # noqa: F401,F403

# Isolate token-level TN/content supervision. Keep the Stage-B patch objective,
# but disable phrase-rank subbatches and loss_phrase_rank.
lambda_text = 0.25
stage_b_enable_phrase_rank = False
stage_b_rank_loss_coef = 0.0
