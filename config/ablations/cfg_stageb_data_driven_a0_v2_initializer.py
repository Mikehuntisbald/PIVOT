from config.ablations.cfg_stageb_data_driven_dd1_category_complete import *  # noqa: F401,F403

# SHA-independent fair-v2 absolute control template. The builder is allowed to
# read only the sealed b58 checkpoint supplied on its command line.
stage_b_data_driven_rank_architecture = "absolute_token"
stage_b_data_driven_head_init_seed = 42
stage_b_data_driven_base_initializer_path = ""
stage_b_data_driven_base_initializer_sha256 = ""
