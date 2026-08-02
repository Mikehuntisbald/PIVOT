from config.ablations.cfg_stageb_v22_score_decoupling_common import *  # noqa: F401,F403

# S0: one literal score. The full-text phrase logit selects the box and is also
# the absolute/global-max confidence used by FPR95.
stage_b_v22_table_id = "S0"
stage_b_v22_score_ownership = "shared_score"
stage_b_v22_train_phase = "joint"
stage_b_v22_objective_fidelity = "common_objective_ownership_ablation"
stage_b_v14_validity_head = False
stage_b_v15_decoupled_confidence = False
stage_b_v19_explicit_confidence_output_contract = False
only_train_keywords = ["stage_b_fixed_text_scorer.decoder"]
only_train_exclude_keywords = []
stage_b_v11_trainable_params_min = 5_626_240
stage_b_v11_trainable_params_max = 5_626_240
stage_b_v15_validity_lr = None
stage_b_v15_separate_grad_clip = False

# S0 has no broadcast gate. It keeps the same global-max q05 negative and
# positive/TN pair terms, but the gate-specific positive-residual trust hinge
# has no corresponding parameter and is therefore explicitly disabled.
stage_b_v15_tail_queue_positive_trust_weight = 0.0
stage_b_v22_missing_gate_objective = "positive_residual_trust_and_translation"
stage_b_v22_gradient_diagnostic_kind = "shared_parameter_conflict"
