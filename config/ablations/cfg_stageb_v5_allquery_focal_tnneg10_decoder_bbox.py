from config.ablations.cfg_stageb_v5_allquery_focal_tn_matched_decoder_bbox import *  # noqa: F401,F403

# Stage-B v5 TN-negative weighted probe.
#
# Keep the decoder+bbox patch-frozen optimization scope and all-query focal
# positive/refexp objective from the previous v5 decoder/bbox probe, but make
# TN rows strictly negative-only:
# - amplify TN negative-token BCE
# - remove TN content-positive BCE
# - remove TN canonical-positive BCE
#
# This tests whether the v5 TN regression came from content/canonical positive
# leakage on TN samples plus too-weak TN negative supervision.

lambda_tn_neg = 10.0
lambda_tn_content = 0.0
lambda_tn_canonical = 0.0

