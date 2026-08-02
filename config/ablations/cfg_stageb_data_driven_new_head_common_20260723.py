from config.ablations.cfg_stageb_data_driven_dd0 import *  # noqa: F401,F403

# Shared teacher-free D0/D1 contract. The b58 detector/query generator is
# frozen; only the independent rank and patch-scoring branches are optimized.
stage_b_data_driven_base_initializer_path = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/data_driven_initializers/"
    "fair_v2_seed42/checkpoint_dd_a0_absolute_v2_init.pth"
)
stage_b_data_driven_base_initializer_sha256 = (
    "c2c4ba71656054d3afc3d219ca2f6d56839396d6258bac0201878566b1937034"
)
stage_b_data_driven_initializer_pair_receipt_path = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/data_driven_initializers/"
    "fair_v2_seed42/a0_a1_v2_pair_receipt.json"
)
stage_b_data_driven_initializer_pair_receipt_sha256 = (
    "e304d2e8439f5714facf1b510795ba3a9874ec456433110afeef91f2d1dc7d8d"
)

stage_b_data_driven_rank_architecture = "absolute_token"
stage_b_data_driven_train_mode = "rank_patch_only"
stage_b_data_driven_confidence_trained = False
stage_b_data_driven_rank_supervision = "all_nonpositive_negative_v1"
stage_b_data_driven_strict_sample_identity = True
stage_b_data_driven_rank_negative_iou_threshold = 0.3
stage_b_data_driven_positive_iou_threshold = 0.5
stage_b_data_driven_patch_negative_iou_threshold = 0.3
stage_b_data_driven_rank_weight = 1.0
stage_b_data_driven_patch_weight = 1.0
stage_b_data_driven_temperature = 0.1
stage_b_data_driven_category_margin = 0.1

stage_b = False
stage_b_gdino_score_adapter = False
stage_b_u0_patch_rank = False
stage_b_v7 = False
stage_b_v11_fixed_text = False
stage_b_legacy_global_gate = False
patch_only = False
enable_patch_branch = True
patch_gate_with_text = False
stage_b_data_driven_category_gate = False

# LR is preregistered after the clean D1 dev-screen probe. Probe configs may
# override rank_lr only; D0 and D1 formal runs must inherit the same value.
stage_b_data_driven_rank_lr = 3e-4
stage_b_data_driven_patch_lr = 3e-4
stage_b_data_driven_sampling_contract = "deterministic_epoch_ledger_v1"
stage_b_data_driven_sampler_seed = 42
stage_b_data_driven_loader_seed = 1042
stage_b_data_driven_grad_clip_contract = "per_optimizer_branch_v1"
stage_b_data_driven_required_allocator_env = "PYTORCH_CUDA_ALLOC_CONF"
stage_b_data_driven_required_allocator_conf = "expandable_segments:True"

batch_size = 64
epochs = 3
lr_drop = 100
onecyclelr = False
multi_step_lr = False
lr_drop_list = [4, 8]
weight_decay = 1e-4
clip_max_norm = 0.1
fix_size = True
strong_aug = False
data_aug_hflip_prob = 0.0
aux_loss = False
use_checkpoint = False
use_transformer_ckpt = False
persistent_workers = False
amp_init_scale = 8192.0
save_checkpoint_interval = 1
skip_eval = True
use_coco_eval = False

stage_b_data_driven_new_head_partition_receipt_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_data_driven_new_head_partition_20260723/receipt.json"
)
stage_b_data_driven_new_head_partition_receipt_sha256 = (
    "56de31d883ed137f3f9332c34de846839d82c0724120f42a49c5c1c302f38506"
)
stage_b_data_driven_new_head_support_receipt_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_data_driven_support_partition_20260723/receipt.json"
)
stage_b_data_driven_new_head_support_receipt_sha256 = (
    "a0e6632182bc7c01ac6e6997b15f1f96e0fbb0bf6dd9d1e3fd8485ad39a6da62"
)
