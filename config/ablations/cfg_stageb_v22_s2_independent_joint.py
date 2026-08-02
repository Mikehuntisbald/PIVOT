from config.ablations.cfg_stageb_v22_score_decoupling_common import *  # noqa: F401,F403

# S2 is the exact v19 base-plus-gate scorer plus the Acc@0.5 hard-negative
# boundary and v21 token objective: a trainable rank decoder, an immutable
# copied confidence decoder/base, and a trainable broadcast validity gate.
# Rank decoder and gate are optimized jointly.
stage_b_v22_table_id = "S2"
stage_b_v22_score_ownership = "independent_decoders_joint"
stage_b_v22_train_phase = "joint"
stage_b_v22_objective_fidelity = "common_objective_ownership_ablation"
stage_b_v14_validity_head = True
stage_b_v15_decoupled_confidence = True
stage_b_v19_explicit_confidence_output_contract = True
only_train_keywords = [
    "stage_b_fixed_text_scorer.decoder",
    "stage_b_fixed_text_scorer.validity_head",
]
only_train_exclude_keywords = [
    "stage_b_fixed_text_scorer.confidence_decoder"
]
stage_b_v11_trainable_params_min = 5_692_289
stage_b_v11_trainable_params_max = 5_692_289
stage_b_v22_gradient_diagnostic_kind = "branch_isolation"
