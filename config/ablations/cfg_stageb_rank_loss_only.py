from config.cfg_patch_stage_b import *  # noqa: F401,F403

# Isolate phrase-level TN ranking. Keep the Stage-B patch objective, but remove
# token-level text BCE so TN supervision enters only through loss_phrase_rank.
lambda_text = 0.0
stage_b_enable_phrase_rank = True
stage_b_rank_loss_coef = 1.0

# Keep inference scoring identical to the full Stage-B recipe.
