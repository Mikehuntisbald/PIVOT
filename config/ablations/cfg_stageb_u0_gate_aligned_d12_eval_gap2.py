"""Inference leaf for D12's persistent conditional residual and Gap2 gate."""

from config.ablations.cfg_stageb_u0_gate_aligned_d12 import *  # noqa: F401,F403

stage_b_u0_gate_aligned_d12 = False
stage_b_u0_gate_aligned_rank_residual = True
stage_b_u0_category_preserving_patch_gate = True
stage_b_u0_category_gate_max_gap = 2.0

batch_size = 16
skip_eval = True
use_coco_eval = False
