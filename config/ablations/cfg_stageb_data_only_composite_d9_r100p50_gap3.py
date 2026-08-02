"""Inference-only Stage-B-data composite: D9 patch + R100 rank + P50 confidence."""

from config.ablations.cfg_stageb_u0_r100p50_patch_rank import *  # noqa: F401,F403

# This leaf intentionally starts from the Stage-B R100/P50 U0 architecture,
# not from any U2/Stage-A config.  The serialized composite replaces every
# patch-specific tensor imported by the historical U0 initializer with the
# corresponding b58-native, Stage-B-trained D9 tensor.
stage_b_data_only_composite = True
stage_b_data_only_composite_contract_version = 1
stage_b_data_only_composite_patch_source = "d9_stageb_native_patch_u100"
stage_b_data_only_composite_rank_source = "r100_stageb_three_ref_u100"
stage_b_data_only_composite_confidence_source = (
    "p50_stageb_traceable_semantic_tn_u50"
)

# Deployment keeps the U2 functional division without reusing U2 weights:
# D9 patch scores define category eligibility, R100 orders eligible queries,
# and P50 supplies the independent absolute confidence score.
stage_b_u0_category_preserving_patch_gate = True
stage_b_u0_category_gate_max_gap = 3.0

# This artifact is evaluation-only; no U2 category objective or Stage-A
# selection receipt is part of its dependency chain.
stage_b_u2_category_complete_supervision = False
stage_b_u2_category_loss_weight = 0.0
stage_b_u0_category_gate_selection_receipt = ""
stage_b_u0_category_gate_selection_payload_sha256 = ""

batch_size = 16
skip_eval = True
use_coco_eval = False
