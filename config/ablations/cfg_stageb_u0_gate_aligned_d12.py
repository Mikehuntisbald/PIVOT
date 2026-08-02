"""Frozen-R100, Gap2-aligned conditional rank-residual training."""

from config.ablations.cfg_stageb_u0_gate_aligned_d11 import *  # noqa: F401,F403

stage_b_u0_gate_aligned_d11 = False
stage_b_u0_gate_aligned_rank_residual = True
stage_b_u0_gate_aligned_d12 = True
stage_b_u0_gate_aligned_d12_contract_version = 12
stage_b_u0_category_preserving_patch_gate = True
stage_b_u0_category_gate_max_gap = 2.0

stage_b_u0_d12_initializer_path = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "data_only_composite_d9_r100p50_gap3_v1/checkpoint_eval_only.pth"
)
stage_b_u0_d12_initializer_sha256 = (
    "92f66b76f529e84fec66e602a712d071c81b32b5b8f6d2633cb0f89a948a3b3a"
)

stage_b_u0_d12_hidden_dim = 64
stage_b_u0_d12_residual_limit = 0.1
stage_b_u0_d12_weight = 1.0
stage_b_u0_d12_positive_iou_threshold = 0.5
stage_b_u0_d12_fix_margin = 0.05
stage_b_u0_d12_preserve_tolerance = 0.01
stage_b_u0_d12_preserve_floor = 0.005
stage_b_u0_d12_temperature = 0.05
stage_b_u0_d12_fix_weight = 1.0
stage_b_u0_d12_preserve_weight = 2.0
stage_b_u0_d12_residual_weight = 0.05

stage_b_u0_d12_rank_lr = 1e-4
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
