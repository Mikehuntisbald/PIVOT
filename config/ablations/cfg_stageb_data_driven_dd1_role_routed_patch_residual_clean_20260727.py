from config.ablations.cfg_stageb_data_driven_dd1_role_routed_balanced_anchor_clean_20260727 import *  # noqa: F401,F403

# Nonlinear category-capacity test. The canonical query generator, support
# patch encoder, base cosine scorer, full-text rank branch, confidence branch,
# boxes, and Gate3 remain unchanged and frozen. Only this zero-output residual
# and the already-disjoint rank branch are trainable.
stage_b_data_driven_patch_residual = True
stage_b_data_driven_patch_training_surface = "residual_only_3tensor_v1"
stage_b_data_driven_patch_residual_contract = "detached_qp_mlp128_tanh025_v1"
stage_b_data_driven_patch_residual_hidden_dim = 128
stage_b_data_driven_patch_residual_limit = 0.25
stage_b_data_driven_patch_residual_init_seed = 42
stage_b_data_driven_patch_residual_source_initializer_sha256 = (
    "5ae688008cf56130c69c152197911fa61fecb6a24956f425fdd5a7ac42e97bd1"
)

stage_b_data_driven_initializer_contract = (
    "clean_dd1_u1000_model_only_patch_residual128_v1"
)
stage_b_data_driven_base_initializer_path = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "data_driven_role_routed_20260727/initializers/"
    "clean_dd1_u1000_lr3e4_patch_residual128_seed42_v2/"
    "checkpoint_model_only.pth"
)
stage_b_data_driven_base_initializer_sha256 = (
    "c4275c575d8f7f3734806620b90572cf316adcd3fa8b42958ea6678d700c04c0"
)
stage_b_data_driven_role_initializer_receipt_path = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "data_driven_role_routed_20260727/initializers/"
    "clean_dd1_u1000_lr3e4_patch_residual128_seed42_v2/receipt.json"
)
stage_b_data_driven_role_initializer_receipt_sha256 = (
    "84e7362a96df640079d321cd2dc11e2de8a8d22bc9b98734cbfc7d5dd89ff008"
)
stage_b_data_driven_patch_lr = 3e-4
