"""U2-v2 diagnostic U100 post-gate residual training."""

from config.ablations.cfg_stageb_u2v2_c0 import *  # noqa: F401,F403

stage_b_u2v2_rank_residual = True
stage_b_u2v2_training_dataset_binding = True
stage_b_u2v2_forward_microbatch = 19
stage_b_u2v2_hidden_dim = 64
stage_b_u2v2_residual_limit = 0.1
stage_b_u2v2_weight = 1.0
stage_b_u2v2_positive_iou_threshold = 0.5
stage_b_u2v2_fix_margin = 0.05
stage_b_u2v2_preserve_tolerance = 0.01
stage_b_u2v2_preserve_floor = 0.005
stage_b_u2v2_temperature = 0.05
stage_b_u2v2_fix_weight = 1.0
stage_b_u2v2_preserve_weight = 2.0
stage_b_u2v2_residual_weight = 0.05

stage_b_u2v2_rank_lr = 1e-4
lr = 1e-4
weight_decay = 1e-4
clip_max_norm = 0.1
amp_init_scale = 8192.0

fix_size = False
data_aug_train_deterministic_aspect_resize = True
strong_aug = False
data_aug_hflip_prob = 0.0
data_aug_scales = [800]
data_aug_max_size = 1333

batch_size = 38
epochs = 250
lr_drop = 1000
onecyclelr = False
multi_step_lr = False
save_checkpoint_interval = 25
skip_eval = True
use_coco_eval = False
aux_loss = False
use_checkpoint = False
use_transformer_ckpt = False
freeze_keywords = None
only_train_keywords = None
unfreeze_decoder_last_n_layers = 0
