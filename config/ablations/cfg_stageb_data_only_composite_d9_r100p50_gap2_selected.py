"""Val-selected deployment of the Stage-B-only D9 + R100/P50 composite."""

from config.ablations.cfg_stageb_u0_r100p50_patch_rank import *  # noqa: F401,F403

stage_b_data_only_composite = True
stage_b_data_only_composite_contract_version = 1
stage_b_data_only_composite_patch_source = "d9_stageb_native_patch_u100"
stage_b_data_only_composite_rank_source = "r100_stageb_three_ref_u100"
stage_b_data_only_composite_confidence_source = (
    "p50_stageb_traceable_semantic_tn_u50"
)

stage_b_u0_category_preserving_patch_gate = True
stage_b_u0_category_gate_max_gap = 2.0
stage_b_u2_category_complete_supervision = False
stage_b_u2_category_loss_weight = 0.0

# Gap2 is the unique candidate in the canonical 11-gap, three-val,
# single-forward sweep that strictly exceeds sealed U2 on every val split.
stage_b_data_only_gate_selection_receipt = (
    "outputs/paper_cvpr_v1/data_only_composite_d9_r100p50_gap3_v1/"
    "evaluations/canonical_val3_gap_sweep_b16/selection_receipt.json"
)
stage_b_data_only_gate_selection_receipt_sha256 = (
    "793de437e085afd90edf84f1f90e9cdfeef8a09fb5b4e0d6f26f5ea1fed417db"
)
stage_b_data_only_gate_selection_payload_sha256 = (
    "8ff29ab30a2f8287947f71b73e3fd68df9b15b78aba17196e49ce129e4cb5e2e"
)
stage_b_data_only_checkpoint_sha256 = (
    "92f66b76f529e84fec66e602a712d071c81b32b5b8f6d2633cb0f89a948a3b3a"
)

batch_size = 16
skip_eval = True
use_coco_eval = False
