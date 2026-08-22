"""B58 pure-GroundingDINO trunk for matched raw-query owner replay."""

from config.ablations.cfg_stageb_from_gdino_ft_with_tn import *  # noqa: F401,F403

# Evaluation-only construction.  B58 is loaded as the exact 938-tensor pure
# GroundingDINO trunk.  No historical patch or Stage-B scoring branch may run.
aux_loss = False
use_checkpoint = False
use_transformer_ckpt = False
use_coco_eval = False
skip_eval = False
data_aug_hflip_prob = 0.0
label_list = ["object"]

patch_only = False
stage_b = False
enable_patch_branch = False
stage_b_gdino_score_adapter = False
stage_b_u0_patch_rank = False
stage_b_data_driven_score = False
stage_b_native_patch_category = False

# Dedicated fail-closed marker used by the B58 cache extractor.
stage_b_b58_raw_query_ownership_eval = True
