from config.ablations.cfg_stageb_data_driven_dd0 import *  # noqa: F401,F403

# DD1 differs from DD0 only in the sealed category-complete annotations used by
# the patch loss. Text rank still uses exactly one primary referent per row.
stage_b_data_driven_experiment_id = "DD1"
stage_b_data_driven_category_complete = True

# This exact allocator setting passed the deterministic B64 multi-scale stress
# run on the publication environment and is recorded in every trained payload.
stage_b_data_driven_required_allocator_env = "PYTORCH_CUDA_ALLOC_CONF"
stage_b_data_driven_required_allocator_conf = "expandable_segments:True"
