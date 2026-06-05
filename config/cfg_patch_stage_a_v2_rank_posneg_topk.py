_base_ = "cfg_patch_stage_a_v2_rank.py"

# Deprecated Stage A v2 ablation. Do not use for Stage-A mainline training;
# retained only to reproduce the negative rank + pos/neg top-k CE probe.
# Keep pos/neg-separated CE, but avoid easy-negative dilution by averaging only
# the hardest negative patch entries per image.
patch_ce_reduction = "posneg_topk"
# Chosen from checkpoint0004 calibration over 200 train batches:
# avg_pos_count=5.15/image, avg_neg_count=1348.35/image.
# topk=16 covers ~3.1x positives and ~1.19% hardest negatives/image;
# smaller topk is noisy, larger topk quickly approaches all-neg mean.
patch_ce_neg_topk = 16
patch_ce_neg_topk_ratio = 0.0

# With lambda_neg=0.25, posneg_topk16 CE is only ~0.625x legacy dense CE.
# Calibrating to legacy scale gives (0.2666 - 0.1634) / 0.01253 ~= 8.24;
# use 8.0 to restore classification pressure without exceeding legacy scale.
patch_lambda_neg = 8.0
