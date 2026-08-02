from config.ablations.cfg_stageb_v22_score_decoupling_common import *  # noqa: F401,F403

# S1: the phrase aggregation is the rank head and the validity residual is the
# confidence head, but both consume the same trainable decoder features.
stage_b_v22_table_id = "S1"
stage_b_v22_score_ownership = "shared_trunk_two_heads"
stage_b_v22_train_phase = "joint"
stage_b_v22_objective_fidelity = "common_objective_ownership_ablation"
stage_b_v14_validity_head = True
stage_b_v15_decoupled_confidence = False
stage_b_v19_explicit_confidence_output_contract = False
only_train_keywords = [
    "stage_b_fixed_text_scorer.decoder",
    "stage_b_fixed_text_scorer.validity_head",
]
only_train_exclude_keywords = []
stage_b_v11_trainable_params_min = 5_692_289
stage_b_v11_trainable_params_max = 5_692_289
stage_b_v15_separate_grad_clip = False

# As in S0, there is no candidate-invariant broadcast gate on which to impose
# v19's gate-residual trust/translation term. All score-level q05/global-max
# terms remain enabled.
stage_b_v15_tail_queue_positive_trust_weight = 0.0
stage_b_v22_missing_gate_objective = "positive_residual_trust_and_translation"
stage_b_v22_gradient_diagnostic_kind = "shared_parameter_conflict"
