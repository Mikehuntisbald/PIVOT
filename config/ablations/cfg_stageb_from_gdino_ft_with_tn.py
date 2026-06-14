from config.cfg_odvg import *  # noqa: F401,F403

# Pure GroundingDINO continuation from the same-data FT checkpoint on the
# Stage-B data recipe. This ablation intentionally does not enable patch_only,
# stage_b, support patches, patch losses, or phrase-rank losses.

patch_only = False
stage_b = False
enable_patch_branch = False
patch_gate_with_text = True

batch_size = 19
fix_size = True

# One new Stage-B data epoch from the FT checkpoint; keep LR flat.
epochs = 1
lr_drop = 100
save_checkpoint_interval = 1

# Match the original pure GroundingDINO same-data FT policy.
freeze_keywords = ["bert"]
unfreeze_decoder_last_n_layers = 0
only_train_keywords = None

skip_eval = True
use_coco_eval = False
label_list = ["object"]
