from config.ablations.cfg_stageb_data_driven_dd2_confidence import *  # noqa: F401,F403

# Evaluation only: patch category eligibility followed by the frozen DD1 text
# rank. TN scoring continues to read only the independently trained DD2
# confidence output.
stage_b_data_driven_category_gate = True
stage_b_data_driven_category_gate_max_gap = 3.0
