from config.ablations.cfg_stageb_data_driven_dd1_role_routed_clean_20260727 import *  # noqa: F401,F403

# v18 preserves every v17 forward value, mask, and deployment decision. Only
# the positive-side gradient of each drop contrast is distributed over the best
# reachable query for every annotated same-category instance.
stage_b_data_driven_patch_drop_positive_anchor_gradient_policy = (
    "reachable_instance_best_mean_straight_through_v1"
)
stage_b_data_driven_patch_row_balance_contract = (
    "gate_barrier_role_exclusive_plus_allnegative_active_severity_"
    "instance_balanced_zero_sum_no_raw_focal_v10"
)
