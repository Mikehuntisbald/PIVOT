from config.ablations.cfg_stageb_data_driven_dd1_role_routed_patch_residual_clean_20260727 import *  # noqa: F401,F403

# v19 removes the residual MLP's deployment-inert query-wise raw offset before
# tanh. All data, supervision, optimizer settings, limits, and frozen features
# are identical to the uncentered residual capacity probe.
stage_b_data_driven_patch_residual_contract = (
    "detached_qp_mlp128_query_raw_centered_tanh025_v2"
)
stage_b_data_driven_patch_residual_center_raw = True
stage_b_data_driven_initializer_contract = (
    "clean_dd1_u1000_model_only_patch_residual128_raw_centered_v1"
)
stage_b_data_driven_base_initializer_path = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "data_driven_role_routed_20260727/initializers/"
    "clean_dd1_u1000_lr3e4_patch_residual128_raw_centered_seed42_v3/"
    "checkpoint_model_only.pth"
)
stage_b_data_driven_base_initializer_sha256 = (
    "e65f62f342f3c36f6d6bd9322841eb028806f320b898eaf390a38ead36942799"
)
stage_b_data_driven_role_initializer_receipt_path = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "data_driven_role_routed_20260727/initializers/"
    "clean_dd1_u1000_lr3e4_patch_residual128_raw_centered_seed42_v3/"
    "receipt.json"
)
stage_b_data_driven_role_initializer_receipt_sha256 = (
    "eb66b68b61e62f5a9d4ef6b2fe78b72154286c832ce63d98190ecbf48799e242"
)
