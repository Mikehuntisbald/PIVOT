from config.cfg_patch_stage_b import *  # noqa: F401,F403

# No-TN removes the TN dataset in tools/run_stageb_ablations.sh. Disable the
# TN-specific ranking path as well; positive RefCOCO token supervision remains.
stage_b_enable_phrase_rank = False
stage_b_rank_loss_coef = 0.0
