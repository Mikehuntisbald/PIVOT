from config.ablations.cfg_stageb_data_driven_dd1_role_routed_patch_topk_semantic_clean_20260727 import *  # noqa: F401,F403

# Training constructs the exact detached Gate3 mask inside the criterion. At
# evaluation, enable that same gate so patch category scores define eligibility
# and the independent full-expression rank branch selects within it.
stage_b_data_driven_category_gate = True
stage_b_data_driven_category_gate_max_gap = 3.0
stage_b_data_driven_patch_score_clip = 5.0
stage_b_data_driven_eval_expected_optimizer_updates = 1000
