from config.ablations.cfg_stageb_gdino_score_adapter_rank_three_ref import *  # noqa: F401,F403

# Data-only native-residual learnability check.  The only checkpoint source is
# the sealed b58 data-FT model; the adapter trunk is random and both deployed
# residual outputs are exact zero at initialization.
stage_b_native_residual_data_only = True
stage_b_native_residual_contract_version = 1

stage_b_gdino_adapter_train_mode = "rank_only"
stage_b_gdino_tn_scope = ""
stage_b_gdino_rank_weight = 1.0
stage_b_gdino_confidence_weight = 0.0
stage_b_gdino_paired_margin_weight = 0.0
stage_b_gdino_queue_size = 0
stage_b_gdino_queue_min_count = 0

# This is a bounded memorization/gradient-flow check, not a headline training
# recipe.  A fixed resize keeps every one of the 128 directed rows reachable.
stage_b_gdino_rank_lr = 3e-4
lr = 3e-4
batch_size = 64
epochs = 250
save_checkpoint_interval = 250
lr_drop = 1000

fix_size = True
data_aug_hflip_prob = 0.0
data_aug_scales = [800]
data_aug_max_size = 1333

skip_eval = True
use_coco_eval = False
