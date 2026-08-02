from config.ablations.cfg_stageb_v16_gate_only_confidence import *  # noqa: F401,F403

# The Stage-A branch receives only the canonical category/support prompt, while
# the independent scoring branch receives and scores the complete expression.
# Patch logits remain the explicit category prior; retaining the head noun here
# avoids discarding useful full-text decoder evidence.
stage_b_v15_exclude_canonical_from_score = False

# The trust term is composed inside loss_fixed_text_tail_queue, whose outer
# weight is 0.05 in v15. Use 20 here so low-tail positive gates receive an
# effective unit-weight hinge instead of the accidental 0.05x supervision.
stage_b_v15_tail_queue_positive_trust_weight = 20.0
