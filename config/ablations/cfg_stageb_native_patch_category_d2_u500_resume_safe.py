from config.ablations.cfg_stageb_native_patch_category_d2_u500 import *  # noqa: F401,F403

# Same formal D2 trajectory as v1, with non-behavioral periodic checkpoints so
# an external host interruption loses at most 100 optimizer updates.
stage_b_native_patch_formal_config_path = (
    "/media/haoyi/T9/pivot/config/ablations/"
    "cfg_stageb_native_patch_category_d2_u500_resume_safe.py"
)
stage_b_native_patch_formal_output_dir = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "native_patch_category_d2_seed42_b36a2_lr1e4_u500_resume_safe_v2"
)
