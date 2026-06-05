_base_ = "cfg_patch_stage_a.py"

# Deprecated Stage A v2 ablation. Do not use for Stage-A mainline training;
# retained only to reproduce the negative rank-only probe from checkpoint0004.
patch_rank_loss_coef = 0.25
patch_rank_margin = 0.3
patch_rank_hard_negatives = 16
patch_rank_include_wrong_slots = True
patch_rank_wrong_slot_weight = 0.5
