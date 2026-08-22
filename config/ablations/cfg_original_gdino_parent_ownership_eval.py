"""Pure GroundingDINO-T pre-Stage-B parent for frozen-cache extraction."""

from config.ablations.cfg_stageb_from_gdino_ft_with_tn import *  # noqa: F401,F403

# Evaluation-only construction.  The checkpoint contributes only its 938-tensor
# pure GroundingDINO trunk; the historical 200 patch tensors are audited as
# unused and may not enter the forward.
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

# Dedicated fail-closed marker used by the cache extractor.
stage_b_original_gdino_parent_ownership_eval = True
