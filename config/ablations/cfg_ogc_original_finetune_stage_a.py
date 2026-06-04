from config.cfg_odvg import *  # noqa: F401,F403

# Original GroundingDINO-style finetune ablation from the OGC checkpoint.
# Data should be the Stage-A train split converted to ODVG by:
#   tools/build_stagea_odvg_finetune_ablation.py
#
# This intentionally does not enable patch_only / patch losses. It uses the
# normal GroundingDINO text-token classification + box/GIoU/L1 objective.

patch_only = False
stage_b = False
patch_gate_with_text = True

# Keep the original ODVG finetune batch surface. Sample exposure is matched by
# running the same number of completed Stage-A epochs on the converted train set.
batch_size = 4

# Keep the same train-time augmentation shape policy as Stage A.
fix_size = True

# The run script overwrites epochs from the Stage-A log / MATCH_EPOCHS.
epochs = 15
lr_drop = 4
save_checkpoint_interval = 1

# Keep cfg_odvg's original finetune policy: freeze BERT, train the detector.
freeze_keywords = ["bert"]
unfreeze_decoder_last_n_layers = 0

# Do not run validation after each epoch during exposure-matched training. Run a
# separate eval job after training if needed.
skip_eval = True
use_coco_eval = False

# Filled by the run script from the generated canonical ODVG label map.
label_list = ["object"]
