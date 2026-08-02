from config.ablations.cfg_stageb_data_driven_dd1_pairtop1_hardgap3_fair_v2 import *  # noqa: F401,F403

# Fresh-A1, full-official-data memory and causal probe. This leaf deliberately
# keeps the training-time category gate disabled; HardGap3 builds its exact
# Gap3 candidate mask from detached patch scores inside the criterion.
stage_b_data_driven_probe_scope = "full_official_assignment_u50_v1"
stage_b_data_driven_probe_fresh_start = True
stage_b_data_driven_category_gate = False

epochs = 1
batch_size = 64
persistent_workers = False

# These values are argparse-owned and must be repeated on the launch command.
# They are recorded here so the probe receipt can fail closed on runtime drift.
stage_b_data_driven_probe_expected_max_train_iters = 50
stage_b_data_driven_probe_expected_iter_checkpoint_interval = 50
stage_b_data_driven_probe_expected_num_workers = 4
stage_b_data_driven_probe_expected_prefetch_factor = 1
stage_b_data_driven_probe_expected_gradient_accumulation_steps = 1
stage_b_data_driven_probe_expected_amp = True
stage_b_data_driven_probe_expected_save_log = True
stage_b_data_driven_probe_expected_world_size = 1
stage_b_data_driven_probe_expected_distributed = False
