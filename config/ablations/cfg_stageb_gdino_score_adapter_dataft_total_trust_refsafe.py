from config.ablations.cfg_stageb_gdino_score_adapter_dataft_total_trust import *  # noqa: F401,F403

# Confidence training remains total-trust C100.  This leaf carries forward the
# R100 deployment contract without making confidence depend on the rank tower.
stage_b_gdino_ref_top1_guard = True
stage_b_gdino_ref_route_contract = "b58_top1_anchored_rank_tail_v1"
