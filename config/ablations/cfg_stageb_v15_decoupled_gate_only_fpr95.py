from config.ablations.cfg_stageb_v15_decoupled_gate_only import *  # noqa: F401,F403

# Direct surrogate for FPR@95TPR: compare every current verified TN global max
# against the history positive q05 operating threshold.
stage_b_v15_tail_queue_objective = "fpr95"
stage_b_v14_tail_queue_weight = 1.0

# Keep light absolute/global-negative anchors, but retire the old top-k q95
# surrogate as a competing primary objective.
stage_b_v11_global_tn_negative_weight = 0.25
stage_b_v11_global_tn_tail_weight = 0.0
