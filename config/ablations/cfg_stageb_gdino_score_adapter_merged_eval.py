from config.ablations.cfg_stageb_gdino_score_adapter_semantic_verified import *  # noqa: F401,F403

# This config only constructs the ordinary pure-GDINO model plus both isolated
# adapter branches for evaluation.  The legal confidence-only mode is retained
# because the shared model builder also constructs a criterion; the explicit
# marker below is the authority and merger tools must reject training use.
stage_b_gdino_adapter_merged_eval_only = True
stage_b_gdino_adapter_merged_eval_contract_version = 1

# No loss weight is authorized by this evaluation recipe.  Rank and confidence
# parameters come from separate audited checkpoints and are never optimized as
# one model.
stage_b_gdino_rank_weight = 0.0
stage_b_gdino_confidence_weight = 0.0
