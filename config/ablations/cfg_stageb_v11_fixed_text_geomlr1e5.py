from config.ablations.cfg_stageb_v11_fixed_text_scorer import *  # noqa: F401,F403

# Controlled continuation from the same v11 checkpoint: only the absolute LR
# for sampling_offsets/ref_point_head changes from 2e-6 to 1e-5.
lr_linear_proj_mult = 1e-5
