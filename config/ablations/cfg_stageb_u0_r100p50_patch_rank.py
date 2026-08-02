from config.ablations.cfg_stageb_gdino_score_adapter_rank_three_ref import *  # noqa: F401,F403

# U0 is a strict extension of the sealed all-900 R100/P50 model. The existing
# rank and confidence adapters remain frozen; only the patch-category residual
# and patch-specific projection layers are trainable.
stage_b_u0_patch_rank = True
enable_patch_branch = True
patch_gate_with_text = False
# U0 freezes the sealed P50 confidence branch trained on this TN scope.
stage_b_gdino_tn_scope = "image_global_topk_verified"

stage_b_u0_patch_rank_hidden_dim = 64
stage_b_u0_patch_rank_score_clip = 5.0
stage_b_u0_patch_rank_weight = 1.0
stage_b_u0_positive_iou_threshold = 0.5
stage_b_u0_fix_margin = 0.05
stage_b_u0_preserve_margin = 0.02
stage_b_u0_rank_temperature = 0.1
stage_b_u0_residual_weight = 1e-3

stage_b_u0_patch_rank_lr = 3e-4
stage_b_u0_patch_projection_lr = 3e-5
lr = 3e-4
weight_decay = 1e-4
clip_max_norm = 0.1

aux_loss = False
use_checkpoint = False
use_transformer_ckpt = False
freeze_keywords = None
only_train_keywords = None
unfreeze_decoder_last_n_layers = 0

batch_size = 32
epochs = 1
lr_drop = 100
save_checkpoint_interval = 1
skip_eval = True
use_coco_eval = False
