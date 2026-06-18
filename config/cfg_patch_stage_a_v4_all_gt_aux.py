_base_ = "cfg_patch_stage_a_v3_all_gt_classes.py"

# Stage A v4: v3 all-GT-class support slots plus decoder auxiliary patch
# losses. This is Stage-A-only; Stage-B configs keep their own aux settings.
aux_loss = True
