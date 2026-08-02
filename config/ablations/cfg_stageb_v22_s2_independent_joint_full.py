from config.ablations.cfg_stageb_v22_s2_independent_joint import *  # noqa: F401,F403

# Separate objective-completion row. It is not part of the pure ownership
# contrast because only architectures with a broadcast validity gate can use
# this positive-residual trust term.
stage_b_v22_table_id = "S2F"
stage_b_v22_objective_fidelity = "full_v19_base_plus_gate_objective"
stage_b_v15_tail_queue_positive_trust_weight = 1.0
