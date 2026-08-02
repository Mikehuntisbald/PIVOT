from config.ablations.cfg_stageb_native_patch_category_d5_u100 import *  # noqa: F401,F403

# Same model, data, objective, AMP scale, and memory geometry as formal D5,
# without the formal path/runtime lock. The CLI supplies the smoke budget.
stage_b_native_patch_execution_scope = ""
