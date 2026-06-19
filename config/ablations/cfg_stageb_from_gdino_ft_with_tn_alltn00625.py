from config.ablations.cfg_stageb_from_gdino_ft_with_tn import *  # noqa: F401,F403

# Pure GroundingDINO Stage-B data FT ablation.
#
# Keep the ordinary GroundingDINO model/training path: no Stage-B wrapper,
# no patch branch, no patch loss, and the same ODVG Stage-B data recipe.
# The only intended change from cfg_stageb_from_gdino_ft_with_tn is TN handling:
# TN empty rows are removed from the dense all-query token focal loss and
# replaced by the light all-TN top-k logsumexp constraint selected in
# Stage-B v5 alltn00625.

gdino_tn_loss_type = "alltn00625"
gdino_tn_alltn_weight = 0.0625
gdino_tn_alltn_topk = 10
gdino_tn_alltn_lse_tau = 0.2
# Query scores are probability-space mean(sigmoid(token logits)); the TN loss
# suppresses logsumexp(top-k query scores).
gdino_tn_alltn_tau_neg = 0.0625
gdino_tn_alltn_text_agg = "mean"
