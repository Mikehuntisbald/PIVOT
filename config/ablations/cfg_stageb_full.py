from config.cfg_patch_stage_b import *  # noqa: F401,F403

# Historical rank-enabled Stage-B recipe. The current Stage-B v2 mainline keeps
# phrase ranking disabled; use this config only to reproduce the v3/rank probe.
stage_b_enable_phrase_rank = True
stage_b_rank_loss_coef = 1.0
