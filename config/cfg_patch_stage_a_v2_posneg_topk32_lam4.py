_base_ = "cfg_patch_stage_a.py"

# Deprecated Stage A v2 ablation. Do not use for Stage-A mainline training;
# retained only to reproduce the negative pos/neg top-k CE probe.
patch_ce_reduction = "posneg_topk"
patch_ce_neg_topk = 32
patch_ce_neg_topk_ratio = 0.0
patch_lambda_neg = 4.0
