from config.ablations.cfg_stageb_data_driven_dd0 import *  # noqa: F401,F403

# Clean Stage-B-only warm start. The source U1000 model changed only the rank
# and patch surfaces from the b58-only A0 initializer; this phase discards its
# optimizer/criterion and applies the v17 zero-sum drop-gradient contract.
stage_b_data_driven_variant_id = "DD1-RoleRouted-Clean"
stage_b_data_driven_experiment_id = "DD1"
stage_b_data_driven_train_mode = "rank_patch_only"
stage_b_data_driven_category_complete = True
stage_b_data_driven_confidence_trained = False
stage_b_data_driven_rank_architecture = "absolute_token"
stage_b_data_driven_rank_supervision = (
    "role_routed_official_assignment_all_exclusive_nonowned_v2"
)
stage_b_data_driven_strict_sample_identity = True
stage_b_data_driven_rank_negative_iou_threshold = 0.3
stage_b_data_driven_positive_iou_threshold = 0.5
stage_b_data_driven_patch_negative_iou_threshold = 0.3
stage_b_data_driven_category_gate = False
stage_b_data_driven_category_gate_max_gap = 3.0
stage_b_data_driven_category_gate_boundary_margin = 0.25
stage_b_data_driven_patch_active_unsafe_auxiliary_weight = 1.0
stage_b_data_driven_patch_dense_category_focal_weight = 0.0
stage_b_data_driven_patch_dense_category_focal_alpha = 0.25
stage_b_data_driven_patch_dense_category_focal_gamma = 2.0
stage_b_data_driven_patch_dense_category_focal_negative_weight = 1.0
stage_b_data_driven_patch_drop_positive_anchor_gradient_policy = (
    "global_max_positive_v1"
)
stage_b_data_driven_patch_residual = False
stage_b_data_driven_patch_training_surface = "base_projection_8tensor_v1"
stage_b_data_driven_patch_score_clip = 5.0
stage_b_data_driven_patch_row_balance_contract = (
    "gate_barrier_role_exclusive_plus_allnegative_active_severity_"
    "zero_sum_no_raw_focal_v9"
)
stage_b_data_driven_rank_margin = 0.1
stage_b_data_driven_temperature = 0.1
stage_b_data_driven_rank_weight = 0.0
stage_b_data_driven_patch_weight = 1.0
stage_b_data_driven_assignment_weight = 1.0
stage_b_data_driven_deployment_weight = 0.0

stage_b_data_driven_no_teacher_contract = (
    "clean_dd1_u1000_stageb_data_only_model_warm_start_v1"
)
stage_b_data_driven_role_fresh_optimizer = True
stage_b_data_driven_initializer_contract = (
    "clean_dd1_u1000_model_only_role_routed_v1"
)
stage_b_data_driven_base_initializer_path = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "data_driven_role_routed_20260727/initializers/clean_dd1_u1000_lr3e4/"
    "checkpoint_model_only.pth"
)
stage_b_data_driven_base_initializer_sha256 = (
    "5ae688008cf56130c69c152197911fa61fecb6a24956f425fdd5a7ac42e97bd1"
)
stage_b_data_driven_role_initializer_receipt_path = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "data_driven_role_routed_20260727/initializers/clean_dd1_u1000_lr3e4/"
    "receipt.json"
)
stage_b_data_driven_role_initializer_receipt_sha256 = (
    "5e4ed2e0730e3710300dd3dfdec44e5f56bc5082aba49e1c6f471039caba3f32"
)
stage_b_data_driven_role_initializer_source_checkpoint_sha256 = (
    "dcfd1bf29668b7190f509587f1c9664345da168a9ee874bd97a1a032c01a1aa6"
)
stage_b_data_driven_role_initializer_a0_sha256 = (
    "c2c4ba71656054d3afc3d219ca2f6d56839396d6258bac0201878566b1937034"
)
stage_b_data_driven_role_initializer_source_optimizer_updates = 1000
stage_b_data_driven_role_expected_max_train_iters = 1000
stage_b_data_driven_role_expected_iter_checkpoint_interval = 1000
stage_b_data_driven_role_expected_amp = True
stage_b_data_driven_role_expected_seed = 42
stage_b_data_driven_role_expected_num_workers = 8
stage_b_data_driven_role_expected_prefetch_factor = 1
stage_b_data_driven_role_expected_gradient_accumulation_steps = 1

stage_b_data_driven_assignment_dataset_scope = (
    "official_assignment_clean_train_263661_v1"
)
stage_b_data_driven_assignment_dataset_config_path = (
    "/media/haoyi/T9/pivot/config/"
    "datasets_stageb_data_driven_role_routed_clean_train_20260727.json"
)
stage_b_data_driven_assignment_dataset_config_sha256 = (
    "909f2eb39934e5a263850c1f742d41bdf3f89f819992192696dbe99dd36ea245"
)
stage_b_data_driven_assignment_receipt_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_data_driven_role_routed_clean_assignment_20260727/receipt.json"
)
stage_b_data_driven_assignment_receipt_sha256 = (
    "8154ecbb5933885adc972585e779fb40da4a3f755dd29f50fc17005dc6abcca5"
)
stage_b_data_driven_assignment_manifest_sha256 = {
    "refcoco_stageb_phrase_v1.jsonl": (
        "9cf00f8c1cead0b5741e9f3bf74b29a3a58000982c0c3bcf18f5762512de20cc"
    ),
    "refcocoplus_stageb_phrase_v1.jsonl": (
        "c4d6aec09049381d3d49688e9bd5337767515bc732d957f9727f4892ca8847d5"
    ),
    "refcocog_stageb_phrase_v1.jsonl": (
        "b530c4d838a85496b8713a14014e80fc71db342237fde93649b0d25adb43033a"
    ),
}
stage_b_data_driven_assignment_expected_rows = 263661
stage_b_data_driven_assignment_expected_valid_rows = 224723

stage_b_gdino_score_adapter = False
stage_b_u0_patch_rank = False
stage_b_v7 = False
stage_b_v11_fixed_text = False
stage_b_legacy_global_gate = False
stage_b = False
enable_patch_branch = True
patch_gate_with_text = False
fix_size = True
strong_aug = False
data_aug_hflip_prob = 0.0

stage_b_data_driven_rank_lr = 3e-4
stage_b_data_driven_patch_lr = 1e-4
stage_b_data_driven_confidence_lr = 3e-4
lr = 1e-4
batch_size = 64
epochs = 1
lr_drop = 100
save_checkpoint_interval = 1
skip_eval = True
use_coco_eval = False
