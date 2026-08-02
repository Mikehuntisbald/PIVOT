from config.ablations.cfg_stageb_u0_r100p50_patch_rank import *  # noqa: F401,F403

# U2 restarts from the zero-residual U0 initializer. Its only behavioral
# change is category-complete patch supervision: every non-crowd COCO instance
# of the support category is positive, while the original Ref expression stays
# bound to primary_support_instance_index=0.
stage_b_u2_category_complete_supervision = True
stage_b_u2_category_loss_weight = 1.0
stage_b_u2_category_negative_iou_threshold = 0.3
stage_b_u2_category_margin = 0.1
stage_b_u2_target_preserve_weight = 1.0

# The imported Stage-A projection now consumes frozen b58 query features, so
# the category auxiliary must realign that basis instead of treating it as a
# nearly-frozen source tensor.
stage_b_u0_patch_projection_lr = 3e-4
amp_init_scale = 8192.0

batch_size = 56
