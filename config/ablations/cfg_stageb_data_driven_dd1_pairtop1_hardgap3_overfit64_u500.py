from config.ablations.cfg_stageb_data_driven_dd1_pairtop1_overfit64_u500 import *  # noqa: F401,F403

# Same sealed Overfit64/U500 causal gate as PairTop1, changing only the
# separately weighted exact-deployment hard-negative objective.
stage_b_data_driven_variant_id = "DD1-PairTop1-HardGap3"
stage_b_data_driven_deployment_weight = 1.0
