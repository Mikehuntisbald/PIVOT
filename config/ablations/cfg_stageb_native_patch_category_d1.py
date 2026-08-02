from config.ablations.cfg_stageb_from_gdino_ft_with_tn import *  # noqa: F401,F403

# D1 keeps b58's native full-expression detector and text scores bitwise frozen.
# Patch evidence learns only same-category candidate eligibility from data.
stage_b_native_patch_category = True
stage_b_native_patch_contract_version = 1
stage_b_data_driven_score = False
stage_b_gdino_score_adapter = False
stage_b_u0_patch_rank = False
stage_b_v7 = False
stage_b_v11_fixed_text = False
stage_b_legacy_global_gate = False
patch_only = False
stage_b = False
enable_patch_branch = True
patch_gate_with_text = False

stage_b_native_patch_weight = 1.0
stage_b_native_patch_positive_iou_threshold = 0.5
stage_b_native_patch_negative_iou_threshold = 0.3
stage_b_native_patch_margin = 0.1
stage_b_native_patch_temperature = 0.1
stage_b_native_patch_lr = 3e-4
stage_b_native_patch_b58_path = (
    "/media/haoyi/T9/gdino/outputs/"
    "gdino_ft_stageb_from_gdino_ft_e1_with_tn_bs19_nopatchbranch/"
    "checkpoint0001.pth"
)

# Match deployed 800/max1333 geometry without random crop, flip, or scale jitter.
fix_size = False
data_aug_train_deterministic_aspect_resize = True
strong_aug = False
data_aug_hflip_prob = 0.0
data_aug_scales = [800]
data_aug_max_size = 1333

# Frozen b58 features run in eval mode; only eight patch-projection tensors train.
freeze_keywords = None
only_train_keywords = None
unfreeze_decoder_last_n_layers = 0
aux_loss = False
use_checkpoint = False
use_transformer_ckpt = False

batch_size = 48
gradient_accumulation_steps = 2
lr = 3e-4
weight_decay = 1e-4
clip_max_norm = 0.1
amp_init_scale = 8192.0
persistent_workers = False
epochs = 250
lr_drop = 1000
onecyclelr = False
multi_step_lr = False
save_checkpoint_interval = 250
skip_eval = True
use_coco_eval = False
