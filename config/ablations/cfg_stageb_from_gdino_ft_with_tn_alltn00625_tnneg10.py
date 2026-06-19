from config.ablations.cfg_stageb_from_gdino_ft_with_tn_alltn00625 import *  # noqa: F401,F403

# Pure GroundingDINO Stage-B data FT ablation from the Stage-A FT checkpoint.
#
# Keep the ordinary GroundingDINO model/training path: no Stage-B wrapper,
# no patch branch, no patch loss. TN rows stay in the original all-query
# sigmoid focal loss as all-negative rows. This config adds two TN-only
# constraints:
#   1. loss_tn_alltn: top-k all-query score suppression.
#   2. loss_tn_tokens: caption-token negative BCE on TN samples.

batch_size = 18

gdino_tn_loss_type = "alltn00625"
gdino_tn_alltn_weight = 0.0625
gdino_tn_alltn_topk = 10
gdino_tn_alltn_lse_tau = 0.2
# Query scores are probability-space mean(sigmoid(token logits)); suppress the
# hardest top-10 TN queries below a low probability threshold.
gdino_tn_alltn_tau_neg = 0.0625
gdino_tn_alltn_text_agg = "mean"

lambda_tn_neg = 10.0
lambda_tn_content = 1.0
lambda_tn_canonical = 1.0
gdino_tn_token_neg_weight = lambda_tn_neg
gdino_tn_token_content_weight = lambda_tn_content
gdino_tn_token_canonical_weight = lambda_tn_canonical
