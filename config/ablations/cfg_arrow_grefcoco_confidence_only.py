"""Sealed ARROW confidence-only route for gRefCOCO transfer evaluation."""

from config.ablations.cfg_arrow_admission_a_patch_eval_gap3 import *  # noqa: F401,F403

stage_b_arrow_grefcoco_eval = True
stage_b_arrow_grefcoco_confidence_only = True
data_aug_scales = [800]
data_aug_max_size = 1333
data_aug_hflip_prob = 0.0
