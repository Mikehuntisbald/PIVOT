from config.ablations.cfg_stageb_u2_category_complete_patch_rank import *  # noqa: F401,F403

# Inference-only reuse of a trained U2 checkpoint. Patch scores define a hard
# category-eligible set; R100 remains the exact ordering score inside that set.
stage_b_u0_category_preserving_patch_gate = True

# A query is eligible when its standardized patch score is no more than this
# many within-row standard deviations below the best valid query. The maximum
# patch-score query is therefore always eligible.
stage_b_u0_category_gate_max_gap = 1.0
