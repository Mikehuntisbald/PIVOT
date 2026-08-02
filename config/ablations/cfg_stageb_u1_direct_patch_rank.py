from config.ablations.cfg_stageb_u0_r100p50_patch_rank import *  # noqa: F401,F403

# U1 continues the sealed U0-U100 rank branch with a zero-initialized,
# monotonic patch-category skip. R100, P50, b58, and the U0 residual remain
# present in one model; confidence never consumes this path.
stage_b_u1_direct_patch_skip = True
stage_b_u1_direct_patch_gain_limit = 0.5
stage_b_u1_direct_patch_gain_lr = 5e-2

batch_size = 56
