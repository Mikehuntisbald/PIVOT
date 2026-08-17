"""Common U2-v5 ownership exposure/schedule contract."""

from config.ablations.cfg_stageb_u2v5_clean_admission_u100 import *  # noqa: F401,F403

stage_b_u2v5_ablation = True
stage_b_u2v5_ablation_phase = "ownership"
stage_b_u2v4_legacy_training_replay = False
stage_b_u2v5_ablation_row_id = ""
stage_b_u2v5_ownership_schedule = "interleaved_100_admission_50_confidence"
stage_b_u2v5_ownership_confidence_every = 2
stage_b_u2v5_ownership_admission_updates = 100
stage_b_u2v5_ownership_confidence_updates = 50
stage_b_u2v5_ownership_gradient_diagnostic_interval = 10
