"""Gap3 evaluation leaf for a U2-v4 legacy-mechanism replay checkpoint."""

from config.ablations.cfg_stageb_u2v4_legacy_training_replay_u100 import *  # noqa: F401,F403

stage_b_u2v4_checkpoint_eval = True
stage_b_u0_category_preserving_patch_gate = True
stage_b_u0_category_gate_max_gap = 3.0
stage_b_u2v2_category_gate_max_gap = stage_b_u0_category_gate_max_gap
stage_b_gdino_ref_top1_guard = False
stage_b_gdino_ref_route_contract = "u2v4_replayed_admission_gap3_frozen_r100_v1"
