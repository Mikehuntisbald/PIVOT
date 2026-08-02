"""Stage-B-only Gap-2-aligned continuation of D9 + R100/P50."""

from config.ablations.cfg_stageb_u0_r100p50_patch_rank import *  # noqa: F401,F403

# D10 starts from the sealed data-only composite.  During training the hard
# gate must stay disabled because it is non-differentiable; the criterion
# reproduces the same normalized Gap-2 boundary with a smooth local barrier.
stage_b_u0_gate_aligned_d10 = True
stage_b_u0_gate_aligned_d10_contract_version = 10
stage_b_u0_category_preserving_patch_gate = False
stage_b_u0_category_gate_max_gap = 2.0
stage_b_u2_category_complete_supervision = False
stage_b_u2_category_loss_weight = 0.0

stage_b_u0_d10_initializer_path = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "data_only_composite_d9_r100p50_gap3_v1/checkpoint_eval_only.pth"
)
stage_b_u0_d10_initializer_sha256 = (
    "92f66b76f529e84fec66e602a712d071c81b32b5b8f6d2633cb0f89a948a3b3a"
)

# Exact deployment geometry with a 0.25 safety band around Gap-2.
stage_b_u0_d10_weight = 1.0
stage_b_u0_d10_positive_iou_threshold = 0.5
stage_b_u0_d10_negative_iou_threshold = 0.3
stage_b_u0_d10_gate_max_gap = 2.0
stage_b_u0_d10_patch_score_clip = 5.0
stage_b_u0_d10_keep_gap = 1.75
stage_b_u0_d10_drop_gap = 2.25
stage_b_u0_d10_drop_active_gap = 2.75
stage_b_u0_d10_temperature = 0.25
stage_b_u0_d10_max_rank_blockers = 4
stage_b_u0_d10_drop_weight = 2.0
stage_b_u0_d10_critical_keep_weight = 1.0
stage_b_u0_d10_positive_active_gap = 1.25
stage_b_u0_d10_positive_target_gap = 1.5
stage_b_u0_d10_positive_barrier_weight = 2.0
stage_b_u0_d10_instance_active_gap = 1.25
stage_b_u0_d10_instance_target_gap = 1.5
stage_b_u0_d10_instance_coverage_weight = 2.0

# D9 is already near the target regime, so D10 uses a conservative update.
stage_b_u0_d10_patch_lr = 5e-5
lr = 5e-5
weight_decay = 1e-4
clip_max_norm = 0.1
# The direct standardized-gap barriers have much larger initial gradients than
# U2's small continuous margins.  D9's proven scale avoids a first-step AMP
# overflow while retaining mixed-precision memory geometry.
amp_init_scale = 8.0

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
