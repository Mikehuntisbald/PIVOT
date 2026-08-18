"""Gap3 evaluation for AR_B_TEXT."""

from config.ablations.cfg_stageb_u2v5_clean_confidence_d3_u100 import *  # noqa: F401,F403

stage_b_u0_category_preserving_patch_gate = True
stage_b_u0_category_gate_max_gap = 3.0
stage_b_u2v2_category_gate_max_gap = 3.0
stage_b_gdino_ref_top1_guard = False

stage_b_arrow_admission_input_ablation = True
stage_b_arrow_admission_row_id = "AR_B_TEXT"
stage_b_arrow_admission_source = "canonical_text"
stage_b_arrow_source_spatial_size = 7
