from config.ablations.cfg_stageb_v21_edit_token_supervision_acc50_hardneg import *  # noqa: F401,F403

# Table-D rows inherit the exact same data, fixed Top-50 surface, edit-aware
# token labels, and optimizer/update defaults. Leaf configs change only score
# ownership (and, for S3, which half of the fixed total schedule is active).
stage_b_v22_ablation_table = "D"
stage_b_v22_data_contract = "v21_fixed_top50_traceable_single_edit"
stage_b_v22_token_contract = "v21_flat_per_token_edit_bce"
stage_b_v22_gradient_diagnostic_interval = 0

# Clean ownership block: S0/S1 have no broadcast gate, so the gate-residual
# trust hinge has no matching parameter. Disable it for every S0-S3 row and
# restore it only in the separately labelled S2F objective row.
stage_b_v15_tail_queue_positive_trust_weight = 0.0
stage_b_v22_common_objective_contract = "gate_specific_positive_trust_off"

# Table D is tied to the selected Table-C/L4 main objective. Keep the Acc@0.5
# boundary and complementary predicate-pair rank identical in every row.
stage_b_v11_negative_iou_threshold = 0.499
stage_b_v11_predicate_tn_rank_weight = 1.0
stage_b_v22_full_predicate_tn_rank_weight = 1.0
