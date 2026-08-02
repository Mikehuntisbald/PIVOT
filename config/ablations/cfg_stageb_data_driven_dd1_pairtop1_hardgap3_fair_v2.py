from config.ablations.cfg_stageb_data_driven_dd1_pairtop1_fair_v2 import *  # noqa: F401,F403

# Pure-data deployment alignment. PairTop1 remains unchanged and this separate
# objective compares its GT-selected correct query with the current highest
# scoring incorrect query inside the exact Gap3 deployment candidate set.
stage_b_data_driven_variant_id = "DD1-PairTop1-HardGap3"
stage_b_data_driven_deployment_weight = 1.0
