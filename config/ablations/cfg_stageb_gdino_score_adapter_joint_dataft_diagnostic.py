from config.ablations.cfg_stageb_gdino_score_adapter_dataft import *  # noqa: F401,F403

# Diagnostic compatibility mode only. The primary experiment uses separate R
# and C phases so rank updates cannot change confidence training behavior.
stage_b_gdino_adapter_train_mode = "joint"
stage_b_gdino_rank_weight = 1.0
stage_b_gdino_confidence_weight = 1.0
