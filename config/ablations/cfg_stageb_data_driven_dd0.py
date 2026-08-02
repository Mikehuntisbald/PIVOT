from config.ablations.cfg_stageb_from_gdino_ft_with_tn import *  # noqa: F401,F403

# DD0: b58-only canonical query generator with random, absolute and disjoint
# full-text rank/confidence heads. This phase trains rank plus primary-instance
# patch supervision from ordinary RefCOCO/+/g rows. It never consumes a teacher
# score, R100/P50, Stage-A patch tensor, or TN row.
stage_b_data_driven_score = True
stage_b_data_driven_experiment_id = "DD0"
stage_b_data_driven_train_mode = "rank_patch_only"
stage_b_data_driven_category_complete = False
stage_b_data_driven_confidence_trained = False
stage_b_data_driven_rank_supervision = "all_nonpositive_negative_v1"
stage_b_data_driven_rank_negative_iou_threshold = 0.3
stage_b_data_driven_base_initializer_sha256 = (
    "99189fb802329765d13b7700b88b76c61a81d41222ad01736aaf98e337d65032"
)
stage_b_data_driven_base_initializer_path = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "data_driven_initializers/seed42/checkpoint_dd_init.pth"
)

stage_b_gdino_score_adapter = False
stage_b_u0_patch_rank = False
stage_b_v7 = False
stage_b_v11_fixed_text = False
stage_b_legacy_global_gate = False
patch_only = False
stage_b = False
enable_patch_branch = True
patch_gate_with_text = False

stage_b_data_driven_rank_dim = 128
stage_b_data_driven_rank_architecture = "absolute_token"
stage_b_data_driven_rank_num_heads = 4
stage_b_data_driven_rank_image_level_policy = "last"
stage_b_data_driven_rank_image_levels = 2
stage_b_data_driven_rank_image_pool_size = 8
stage_b_data_driven_rank_image_pool_policy = (
    "valid_extent_masked_adaptive_avg_v1"
)
stage_b_data_driven_rank_box_fourier_bands = 16
stage_b_data_driven_rank_ffn_dim = 512
stage_b_data_driven_rank_dropout = 0.0
stage_b_data_driven_head_init_seed = 42
stage_b_data_driven_confidence_dim = 128
stage_b_data_driven_token_temperature = 0.07
stage_b_data_driven_gate_hidden_dim = 128
stage_b_data_driven_gate_pool_temperature = 0.1
stage_b_data_driven_gate_topk = 10

# The category gate is inference-only and is enabled by a separate eval config.
stage_b_data_driven_category_gate = False
stage_b_data_driven_category_gate_max_gap = 3.0
stage_b_data_driven_category_gate_boundary_margin = 0.25
stage_b_data_driven_patch_score_clip = 5.0

stage_b_data_driven_rank_weight = 1.0
stage_b_data_driven_patch_weight = 1.0
stage_b_data_driven_positive_iou_threshold = 0.5
stage_b_data_driven_patch_negative_iou_threshold = 0.3
stage_b_data_driven_temperature = 0.1
stage_b_data_driven_rank_margin = 0.1
stage_b_data_driven_category_margin = 0.1

stage_b_data_driven_rank_lr = 3e-5
stage_b_data_driven_patch_lr = 3e-4
stage_b_data_driven_confidence_lr = 3e-4
stage_b_data_driven_sampling_contract = "deterministic_epoch_ledger_v1"
stage_b_data_driven_sampler_seed = 42
stage_b_data_driven_loader_seed = 1042
stage_b_data_driven_grad_clip_contract = "per_optimizer_branch_v1"
persistent_workers = False
lr = 3e-4
weight_decay = 1e-4
clip_max_norm = 0.1

data_aug_hflip_prob = 0.0
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
amp_init_scale = 8192.0
skip_eval = True
use_coco_eval = False
