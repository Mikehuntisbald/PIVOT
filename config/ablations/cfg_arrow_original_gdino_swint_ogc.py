"""Expression-only replay of the original GroundingDINO-T OGC checkpoint."""

from config.cfg_odvg import *  # noqa: F401,F403

# The upstream OGC checkpoint was built with a 2,000-entry denoising label
# book.  Denoising is disabled at inference, but matching the construction
# contract avoids silently accepting a shape-different model.
dn_labelbook_size = 2000

# Evaluation-only runtime.  These switches do not alter checkpoint-owned
# tensors; they only remove unused training/checkpointing work.
aux_loss = False
use_checkpoint = False
use_transformer_ckpt = False
use_coco_eval = False
label_list = ["object"]
data_aug_hflip_prob = 0.0

# Fail-closed public marker consumed by the dedicated evaluator.
stage_b_arrow_original_ogc_eval = True

# Explicitly prohibit every ARROW/Stage-B branch.  The model sees only the
# image and the complete referring expression.
patch_only = False
stage_b = False
stage_b_v7 = False
stage_b_v11_fixed_text = False
stage_b_legacy_global_gate = False
stage_b_gdino_score_adapter = False
stage_b_u0_patch_rank = False
stage_b_data_driven_score = False
stage_b_native_patch_category = False
enable_patch_branch = False

