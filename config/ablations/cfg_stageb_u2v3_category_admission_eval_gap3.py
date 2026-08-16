"""Val-only U2-v3 deployment leaf with the fixed Gap3 category gate."""

from config.ablations.cfg_stageb_u2v3_category_admission_u100 import *  # noqa: F401,F403

stage_b_u2v3_training_dataset_binding = False
stage_b_u2v3_category_admission = False
stage_b_u2v3_checkpoint_eval = True
stage_b_u0_category_preserving_patch_gate = True
stage_b_u0_category_gate_max_gap = 3.0
stage_b_u2v2_category_gate_max_gap = 3.0
stage_b_u2v3_category_gate_max_gap = 3.0
