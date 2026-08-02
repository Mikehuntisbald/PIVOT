"""Inference leaf for D10 checkpoints under the frozen Gap-2 decision."""

from config.ablations.cfg_stageb_u0_gate_aligned_d10 import *  # noqa: F401,F403

# D10 is a training-only objective.  Evaluation applies the exact hard gate
# and keeps R100/P50 behavior unchanged.
stage_b_u0_gate_aligned_d10 = False
stage_b_u0_category_preserving_patch_gate = True
stage_b_u0_category_gate_max_gap = 2.0

batch_size = 16
skip_eval = True
use_coco_eval = False
