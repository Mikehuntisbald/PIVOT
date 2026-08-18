"""Public ARROW wrapper for the sealed support-patch Admission route."""

from config.ablations.cfg_stageb_u2v5_clean_confidence_d3_u100 import *  # noqa: F401,F403

stage_b_u0_category_preserving_patch_gate = True
stage_b_u0_category_gate_max_gap = 3.0
stage_b_u2v2_category_gate_max_gap = 3.0
stage_b_gdino_ref_top1_guard = False
stage_b_arrow_admission_input_ablation = True
stage_b_arrow_admission_row_id = "AR_A_PATCH"
stage_b_arrow_admission_source = "support_patch"
stage_b_arrow_source_spatial_size = 7
