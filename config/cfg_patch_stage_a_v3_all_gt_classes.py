_base_ = "cfg_patch_stage_a.py"

# Stage A v3: remove random K support-class sampling. For each multi-patch
# episode, use every eligible GT class in the query image as a support slot.
support_use_all_gt_classes = True
support_min_count = 1
support_num_patches_min = 1
