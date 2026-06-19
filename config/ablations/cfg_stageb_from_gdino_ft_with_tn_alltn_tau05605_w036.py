from config.ablations.cfg_stageb_from_gdino_ft_with_tn_alltn00625_tnneg10 import *  # noqa: F401,F403

# Calibrated pure-GDINO Stage-B data-FT allTN setting.
#
# This remains the ordinary GroundingDINO path: patch_only=False, stage_b=False,
# enable_patch_branch=False. It keeps TN rows in the original all-query sigmoid
# focal objective as all-negative rows, keeps TN token BCE, and updates the
# top-k allTN score suppression plus TN-token weighting from the historical
# fixed-10 alltn00625 probe.
#
# Calibration evidence:
# outputs/gdino_alltn_calibration_stagea0001_tau05605_mix120/calibration.json
# outputs/text_gdino_alltn_tau05605_weight_probe303_eval/summary.md

# With topk=10 and lse_tau=0.2, a uniform per-query sigmoid-mean score of 0.1
# maps to 0.1 + 0.2 * log(10) ~= 0.5605 in the allTN aggregate space.
gdino_tn_alltn_tau_neg = 0.5605

# 120-batch train-mix calibration estimated weight=0.36178 for an initial
# effective allTN/base-loss ratio of 10%. The 303-iter probe kept TN FPR at
# least as good as the baseline while improving mean RefCOCO acc50.
gdino_tn_alltn_weight = 0.36

# Current TN-token contract: content/canonical token BCE weights stay 1.0.
# TN negative-token BCE uses one unit per negative token, so the effective
# sample weight is the TN phrase token count rather than a fixed 10.
lambda_tn_neg = 1.0
lambda_tn_content = 1.0
lambda_tn_canonical = 1.0
gdino_tn_token_neg_weight = lambda_tn_neg
gdino_tn_token_content_weight = lambda_tn_content
gdino_tn_token_canonical_weight = lambda_tn_canonical
gdino_tn_token_neg_weight_mode = "token_count"
