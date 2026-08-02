from config.ablations.cfg_stageb_data_driven_dd3_trace_token import *  # noqa: F401,F403

# DD3 uses the same inference rule as DD2; only its training loss differs.
stage_b_data_driven_category_gate = True
stage_b_data_driven_category_gate_max_gap = 3.0
