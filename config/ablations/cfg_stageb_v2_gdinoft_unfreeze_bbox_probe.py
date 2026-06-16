from config.cfg_patch_stage_b import *  # noqa: F401,F403

# Probe: keep Stage-B v2's text/TN objective, but train the detector with the
# same freeze policy as the pure GroundingDINO same-data FT ablation.
#
# Intended checkpoint lineage:
#   outputs/stageB_local_tn_v2_no_phrase_loss/checkpoint0003.pth

# Keep v2 behavior explicit.
stage_b_text_loss_type = "matched_bce"
stage_b_enable_phrase_rank = False
stage_b_rank_loss_coef = 0.0
stage_b_score_calib_loss_coef = 0.0
lambda_patch = 1.0
lambda_text = 0.25

# Match GroundingDINO FT trainability: freeze BERT, train detector/backbone/
# transformer/box heads. `only_train_keywords=None` disables v2's text-head-only
# restriction.
freeze_keywords = ["bert"]
unfreeze_decoder_last_n_layers = 0
only_train_keywords = None

# Turn patch Hungarian box losses back on at the GroundingDINO loss caliber.
bbox_loss_coef = 5.0
giou_loss_coef = 2.0

# Full detector unfreezing makes Swin/transformer activation recomputation part
# of backward. The local Swin blocks store H/W as mutable module state, so
# checkpoint recompute can see the wrong resolution in patch-only batches.
use_checkpoint = False
use_transformer_ckpt = False

# Probe runtime defaults; override batch_size from CLI to fit the GPU.
epochs = 1
lr_drop = 100
save_checkpoint_interval = 1
