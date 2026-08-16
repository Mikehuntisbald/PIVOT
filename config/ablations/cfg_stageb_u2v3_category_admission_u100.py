"""U2-v3 bridge: train only Stage-A category admission on the U2-v2 C0 stack."""

from config.ablations.cfg_stageb_u2v2_c0 import *  # noqa: F401,F403

stage_b_u2v3_category_admission = True
stage_b_u2v3_contract = "pivot.stageb.u2v3_category_admission/v1"
stage_b_u2v3_training_dataset_binding = True
stage_b_u2v3_forward_microbatch = 19

# The hard gate is disabled during training.  The criterion differentiably
# reproduces its standardized Gap3 boundary; the eval leaf enables it.
stage_b_u0_category_preserving_patch_gate = False
stage_b_u2v3_category_gate_max_gap = 3.0
stage_b_u0_category_gate_max_gap = stage_b_u2v3_category_gate_max_gap
stage_b_u2v2_category_gate_max_gap = stage_b_u2v3_category_gate_max_gap
stage_b_u2v2_rank_residual = False

stage_b_u2v3_weight = 1.0
stage_b_u2v3_positive_iou_threshold = 0.5
stage_b_u2v3_negative_iou_threshold = 0.3
stage_b_u2v3_patch_score_clip = 5.0
stage_b_u2v3_keep_gap = 2.75
stage_b_u2v3_drop_gap = 3.25
stage_b_u2v3_drop_active_gap = 3.75
stage_b_u2v3_temperature = 0.25
stage_b_u2v3_max_rank_blockers = 4
stage_b_u2v3_drop_weight = 2.0
stage_b_u2v3_critical_keep_weight = 1.0
stage_b_u2v3_positive_active_gap = 2.25
stage_b_u2v3_positive_target_gap = 2.5
stage_b_u2v3_positive_barrier_weight = 2.0
stage_b_u2v3_instance_active_gap = 2.25
stage_b_u2v3_instance_target_gap = 2.5
stage_b_u2v3_instance_coverage_weight = 2.0

stage_b_u2v3_patch_lr = 5e-5
lr = 5e-5
weight_decay = 1e-4
clip_max_norm = 0.1
amp_init_scale = 8.0

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
