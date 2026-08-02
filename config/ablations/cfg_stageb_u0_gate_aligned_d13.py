"""Stage-B-only conservative patch-category residual for the Gap2 composite."""

from config.ablations.cfg_stageb_u0_gate_aligned_d11 import *  # noqa: F401,F403

stage_b_u0_gate_aligned_d11 = False
stage_b_u0_gate_aligned_rank_residual = False
stage_b_u0_gate_aligned_d12 = False
stage_b_u0_gate_aligned_patch_residual = True
stage_b_u0_gate_aligned_d13 = True
stage_b_u0_gate_aligned_d13_contract_version = 13
stage_b_u0_category_preserving_patch_gate = True
stage_b_u0_category_gate_max_gap = 2.0

stage_b_u0_d13_initializer_path = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "data_only_composite_d9_r100p50_gap3_v1/checkpoint_eval_only.pth"
)
stage_b_u0_d13_initializer_sha256 = (
    "92f66b76f529e84fec66e602a712d071c81b32b5b8f6d2633cb0f89a948a3b3a"
)

stage_b_u0_d13_hidden_dim = 64
stage_b_u0_d13_residual_limit = 0.25
stage_b_u0_d13_weight = 1.0
stage_b_u0_d13_positive_iou_threshold = 0.5
stage_b_u0_d13_negative_iou_threshold = 0.3
stage_b_u0_d13_keep_gap = 1.95
stage_b_u0_d13_drop_gap = 2.05
stage_b_u0_d13_preserve_tolerance = 0.02
stage_b_u0_d13_temperature = 0.05
stage_b_u0_d13_keep_weight = 1.0
stage_b_u0_d13_drop_weight = 1.0
stage_b_u0_d13_preserve_weight = 4.0
stage_b_u0_d13_residual_weight = 0.05

stage_b_u0_d13_patch_lr = 1e-4
lr = 1e-4
weight_decay = 1e-4
clip_max_norm = 0.1
amp_init_scale = 8192.0

batch_size = 36
epochs = 250
lr_drop = 1000
onecyclelr = False
multi_step_lr = False
save_checkpoint_interval = 100
skip_eval = True
use_coco_eval = False
