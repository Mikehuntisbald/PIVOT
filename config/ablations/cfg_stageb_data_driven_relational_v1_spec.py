# Relational-v1 changes only rank expressivity. Data, loss, optimization, patch,
# and confidence contracts remain identical to the DD1 absolute-token control.
stage_b_data_driven_rank_architecture = "relational_v1"
stage_b_data_driven_rank_dim = 128
stage_b_data_driven_rank_num_heads = 4
stage_b_data_driven_rank_image_level_policy = "last"
stage_b_data_driven_rank_image_levels = 2
stage_b_data_driven_rank_image_pool_size = 8
stage_b_data_driven_rank_image_pool_policy = (
    "valid_extent_masked_adaptive_avg_v1"
)
stage_b_data_driven_rank_box_fourier_bands = 16
stage_b_data_driven_rank_ffn_dim = 512
stage_b_data_driven_rank_dropout = 0.0
stage_b_data_driven_head_init_seed = 42
