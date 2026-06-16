from config.ablations.cfg_stageb_v5_allquery_focal_tnneg10_decoder_bbox import *  # noqa: F401,F403

# Stage-B v5 TN-negative weighted probe with TN content/canonical as explicit
# negatives instead of dropping those terms.
#
# Baseline tnneg10 used:
# - lambda_tn_neg = 10.0
# - lambda_tn_content = 0.0
# - lambda_tn_canonical = 0.0
#
# This keeps the same decoder+bbox patch-frozen scope and TN negative weight,
# but supervises TN content and canonical tokens toward 0 with weight 1.

lambda_tn_neg = 10.0
lambda_tn_content = 1.0
lambda_tn_canonical = 1.0
tn_content_target = 0.0
tn_canonical_target = 0.0
