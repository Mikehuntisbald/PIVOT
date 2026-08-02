from config.ablations.cfg_stageb_gdino_score_adapter_rank_three_ref import *  # noqa: F401,F403

# Fresh, model-score-free learnability gate for the b58 native full-expression
# score.  The adapter is random/identity initialized and only its rank branch is
# trainable.  This leaf deliberately matches the deployed val resize geometry.
stage_b_native_residual_data_only = True
stage_b_native_residual_contract_version = 2

stage_b_gdino_adapter_train_mode = "rank_only"
stage_b_gdino_tn_scope = ""
stage_b_gdino_rank_weight = 1.0
stage_b_gdino_confidence_weight = 0.0
stage_b_gdino_paired_margin_weight = 0.0
stage_b_gdino_queue_size = 0
stage_b_gdino_queue_min_count = 0

stage_b_gdino_rank_lr = 3e-4
lr = 3e-4
batch_size = 32
epochs = 250
save_checkpoint_interval = 250
lr_drop = 1000

# Unlike legacy fix_size=True, this performs scalar 800/max1333 resizing and
# preserves aspect ratio without entering the RandomSelect/crop train path.
fix_size = False
data_aug_train_deterministic_aspect_resize = True
strong_aug = False
data_aug_hflip_prob = 0.0
data_aug_scales = [800]
data_aug_max_size = 1333

skip_eval = True
use_coco_eval = False
