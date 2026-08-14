from config.ablations.cfg_stageb_gdino_score_adapter_rank_three_ref import *  # noqa: F401,F403

# Stage-A/B58 lineage R100: train the ordinary raw rank residual exactly as in
# the historical rank phase, but deploy it only below B58's immutable top-1.
# The raw score remains the training target; this flag changes only the sealed
# Ref route and therefore cannot leak confidence gradients into ranking.
stage_b_gdino_ref_top1_guard = True
stage_b_gdino_ref_route_contract = "b58_top1_anchored_rank_tail_v1"
