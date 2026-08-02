from config.ablations.cfg_stageb_native_patch_category_d8 import *  # noqa: F401,F403

# D9 keeps every D8 forward value and objective coefficient fixed. Its only
# experimental variable is loss-backward localization: row mean/std are
# detached so a selected query cannot update the other 899 queries through
# standardization statistics.
stage_b_native_patch_contract_version = 9
stage_b_native_patch_objective = "d9_loss_gradient_localized"
stage_b_native_patch_d9_detach_row_stats = True
