"""Replay the sealed legacy-U2 admission mechanism on the C100 C0 stack."""

from config.ablations.cfg_stageb_u2v2_c0 import *  # noqa: F401,F403

stage_b_u2v4_legacy_training_replay = True
stage_b_u2v4_checkpoint_eval = False
stage_b_u2v3_category_admission = False
stage_b_u2v2_rank_residual = False
stage_b_u0_category_preserving_patch_gate = False
stage_b_u2v2_initializer_path = (
    "outputs/u2v4_legacy_admission_replay_20260816/training_initializer/"
    "checkpoint_u2v4_init.pth"
)
stage_b_u2v2_initializer_sha256 = (
    "97edb922caf2461c147c83d58bcbae3cd759457c2f7f58305293c43422c5ce02"
)

# Exact legacy U2 category-complete objective. The auxiliary U0 residual and
# patch projection are one admission subsystem; trunk, R100 and C100 remain
# independently frozen.
stage_b_u2_category_complete_supervision = True
stage_b_u2_category_loss_weight = 1.0
stage_b_u2_category_negative_iou_threshold = 0.3
stage_b_u2_category_margin = 0.1
stage_b_u2_target_preserve_weight = 1.0
stage_b_u0_patch_rank_weight = 1.0
stage_b_u0_positive_iou_threshold = 0.5
stage_b_u0_fix_margin = 0.05
stage_b_u0_preserve_margin = 0.02
stage_b_u0_rank_temperature = 0.1
stage_b_u0_residual_weight = 1e-3

stage_b_u0_patch_rank_lr = 3e-4
stage_b_u0_patch_projection_lr = 3e-4
lr = 3e-4
weight_decay = 1e-4
clip_max_norm = 0.1
amp_init_scale = 8192.0

batch_size = 56
epochs = 1
lr_drop = 100
save_checkpoint_interval = 25
skip_eval = True
use_coco_eval = False
aux_loss = False
use_checkpoint = False
use_transformer_ckpt = False
freeze_keywords = None
only_train_keywords = None
unfreeze_decoder_last_n_layers = 0
