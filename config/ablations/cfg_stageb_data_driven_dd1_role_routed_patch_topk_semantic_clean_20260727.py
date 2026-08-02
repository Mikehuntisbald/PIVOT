from config.ablations.cfg_stageb_data_driven_dd1_role_routed_patch_residual_raw_centered_clean_20260727 import *  # noqa: F401,F403

# The patch branch remains category-only and uses no GT or text inside the
# scorer. A zero-output context branch compares each query with the semantic
# prototype of the ten highest frozen base-patch competitors in the same row.
stage_b_data_driven_patch_training_surface = (
    "residual_only_6tensor_topk_semantic_v1"
)
stage_b_data_driven_patch_residual_contract = (
    "detached_qp_base_topk10_semantic_context16_"
    "query_raw_centered_tanh025_v3"
)
stage_b_data_driven_patch_residual_context_dim = 16
stage_b_data_driven_patch_residual_context_topk = 10

# The three 263k-row manifests expand into a large nested Python object graph.
# Synchronous loading keeps batch-64 GPU memory unchanged while avoiding host
# OOM from worker copy-on-write, whole-batch prefetching, and pinned copies.
stage_b_data_driven_role_expected_num_workers = 0
stage_b_data_driven_role_expected_pin_memory = False

stage_b_data_driven_initializer_contract = (
    "clean_dd1_u1000_model_only_patch_topksemantic128_ctx16_v1"
)
stage_b_data_driven_base_initializer_path = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "data_driven_role_routed_20260727/initializers/"
    "clean_dd1_u1000_lr3e4_patch_topksemantic128_ctx16_k10_seed42_v1/"
    "checkpoint_model_only.pth"
)
stage_b_data_driven_base_initializer_sha256 = (
    "1c6472a8694eb606ceec8d74b7a5b2a7e8b3776790eb0f88edb4d5880a72ca0b"
)
stage_b_data_driven_role_initializer_receipt_path = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "data_driven_role_routed_20260727/initializers/"
    "clean_dd1_u1000_lr3e4_patch_topksemantic128_ctx16_k10_seed42_v1/"
    "receipt.json"
)
stage_b_data_driven_role_initializer_receipt_sha256 = (
    "8d1ae999912f7c2dadb0d153e4e4c3baac31a2d91c3fa460425c787f83401fb4"
)
