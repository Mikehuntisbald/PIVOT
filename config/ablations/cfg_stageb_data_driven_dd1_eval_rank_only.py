from config.ablations.cfg_stageb_data_driven_dd1_category_complete import *  # noqa: F401,F403

# Diagnostic only: isolate the learned full-expression rank head by disabling
# the parameter-free patch eligibility gate. The checkpoint and score head are
# otherwise identical to the formal DD1 gap3 evaluation.
stage_b_data_driven_category_gate = False
