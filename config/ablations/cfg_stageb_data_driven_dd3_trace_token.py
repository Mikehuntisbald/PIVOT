from config.ablations.cfg_stageb_data_driven_dd2_confidence import *  # noqa: F401,F403

# DD3 is paired with DD2 at the data, DD1 initializer, optimizer, and budget
# levels. Its sole treatment difference is direct trace-derived token roles.
stage_b_data_driven_experiment_id = "DD3"
stage_b_data_driven_token_weight = 1.0
