from config.ablations.cfg_stageb_v5_allquery_focal_tnneg10_decoder_bbox import *  # noqa: F401,F403

# Same as the tnneg10 decoder+bbox probe, but amplify TN negative-token BCE by
# another order of magnitude to test whether TN rejection keeps improving or
# starts to over-regularize positive/refexp scoring.

lambda_tn_neg = 100.0

