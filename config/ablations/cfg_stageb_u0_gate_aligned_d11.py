"""Stage-B-only Gap-2-eligible continuation of the R100 text ranker."""

from config.ablations.cfg_stageb_u0_r100p50_patch_rank import *  # noqa: F401,F403

# The exact D9 Gap2 gate runs in eval mode during training.  Only R100's final
# rank-output weight is optimized; its gap-invariant bias, P50 confidence, and
# all patch/U0 tensors stay frozen.
stage_b_u0_gate_aligned_d11 = True
stage_b_u0_gate_aligned_d11_contract_version = 11
stage_b_u0_gate_aligned_d10 = False
stage_b_u0_category_preserving_patch_gate = True
stage_b_u0_category_gate_max_gap = 2.0
stage_b_u2_category_complete_supervision = False
stage_b_u2_category_loss_weight = 0.0

stage_b_u0_d11_initializer_path = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "data_only_composite_d9_r100p50_gap3_v1/checkpoint_eval_only.pth"
)
stage_b_u0_d11_initializer_sha256 = (
    "92f66b76f529e84fec66e602a712d071c81b32b5b8f6d2633cb0f89a948a3b3a"
)

stage_b_u0_d11_weight = 1.0
stage_b_u0_d11_positive_iou_threshold = 0.5
stage_b_u0_d11_fix_margin = 0.05
stage_b_u0_d11_preserve_margin = 0.02
stage_b_u0_d11_temperature = 0.05
stage_b_u0_d11_fix_weight = 1.0
stage_b_u0_d11_preserve_weight = 1.0

# R100 used 3e-5 from its fresh phase.  D11 changes a mature ranker, so the
# head-only continuation starts three times lower and without decay-to-zero.
stage_b_u0_d11_rank_lr = 1e-5
lr = 1e-5
weight_decay = 0.0
clip_max_norm = 0.1
amp_init_scale = 8192.0

# Match official 800/max1333 inference geometry without stochastic transforms.
fix_size = False
data_aug_train_deterministic_aspect_resize = True
strong_aug = False
data_aug_hflip_prob = 0.0
data_aug_scales = [800]
data_aug_max_size = 1333

batch_size = 36
epochs = 250
lr_drop = 1000
onecyclelr = False
multi_step_lr = False
save_checkpoint_interval = 100
skip_eval = True
use_coco_eval = False
aux_loss = False
use_checkpoint = False
use_transformer_ckpt = False
freeze_keywords = None
only_train_keywords = None
unfreeze_decoder_last_n_layers = 0
