"""Shared, capacity-matched ARROW Admission-input training contract."""

from config.ablations.cfg_stageb_u2v5_clean_admission_u100 import *  # noqa: F401,F403

stage_b_arrow_admission_input_ablation = True
stage_b_arrow_source_spatial_size = 7
stage_b_arrow_admission_updates = 100
stage_b_arrow_admission_batch_size = 56

# All sources train the legacy surface8 + training-only auxiliary8.  The
# full-expression B58 geometry, R100, confidence12, patch backbone, and patch
# temperature remain frozen.
stage_b_u2v4_legacy_training_replay = True
stage_b_u2v4_checkpoint_eval = False

