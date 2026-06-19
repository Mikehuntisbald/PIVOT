from config.ablations.cfg_stageb_v5_2_refcoco_patchpos_aux_from_v5_alltn00625 import *  # noqa: F401,F403

# Stage-B v5.2 calibrated allTN setting.
#
# Keep the established v5.2 recipe:
# - all-query focal positives with matched-BCE TN text rows
# - decoder/box-head unfreeze, patch branch frozen
# - bbox/GIoU losses
# - RefCOCO-family patch CE positive-only
# - decoder aux losses
#
# Synchronize the allTN tail-suppression setting with the calibrated pure-GDINO
# allTN run. In Stage-B v5.x this term is part of loss_score_calib and operates
# on the final Stage-B slot score, so the parameter names differ from the
# pure-GDINO loss_tn_alltn config.

stage_b_score_calib_topk = 10
stage_b_score_calib_neg_agg = "logsumexp"
stage_b_score_calib_neg_lse_tau = 0.2
stage_b_score_calib_tau_neg = 0.5605
stage_b_score_calib_all_tn_neg_weight = 0.36

# Restore TN content/canonical token supervision as explicit negatives. Older
# v5 alltn00625 probes used 10/0/0; the current TN-token contract is 10/1/1.
lambda_tn_neg = 10.0
lambda_tn_content = 1.0
lambda_tn_canonical = 1.0
tn_content_target = 0.0
tn_canonical_target = 0.0
