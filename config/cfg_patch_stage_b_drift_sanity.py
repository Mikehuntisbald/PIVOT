from config.cfg_patch_stage_b import *  # noqa: F401,F403

# Short diagnostic runs only. Full Stage-B training keeps drift logging disabled
# to avoid the extra cached-batch eval forward in the training loop.
log_stage_b_patch_drift = True
stage_b_patch_drift_interval = 200
stage_b_patch_drift_topk = 50
