from config.ablations.cfg_stageb_native_patch_category_d3 import *  # noqa: F401,F403

# This parser-owned value is supplied explicitly by the formal command.
del gradient_accumulation_steps

# D3 is initialized from the exact formal D1 U500 checkpoint, never D2.
stage_b_native_patch_initializer_path = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "native_patch_category_d1_seed42_b36a2_lr3e4_u500_v1/"
    "checkpoint_iter.pth"
)
stage_b_native_patch_initializer_sha256 = (
    "ac8b29a8d8a5e5bb8877a7c21769ff08c0b5ca805c522f80549dfc99f55c5dc5"
)
stage_b_native_patch_d3_base_initializer_path = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "native_patch_category_initializers/d1_seed42/checkpoint_d1_init.pth"
)
stage_b_native_patch_d3_base_initializer_sha256 = (
    "addec47338c2e36a3121d999370349d2351535f6ff7334729424aeb1bcd880b4"
)

stage_b_native_patch_execution_scope = "native_patch_category_d3_u200_v1"
stage_b_native_patch_formal_config_path = (
    "/media/haoyi/T9/pivot/config/ablations/"
    "cfg_stageb_native_patch_category_d3_u200.py"
)
stage_b_native_patch_formal_output_dir = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "native_patch_category_d3_seed42_s43_b36a2_lr5e5_amp16_u200_v2"
)

# Reuse the audited category-complete D2 corpus byte-for-byte.
stage_b_native_patch_dataset_config_path = (
    "/media/haoyi/T9/pivot/"
    "config/datasets_stageb_native_patch_category_d2_train_20260724.json"
)
stage_b_native_patch_dataset_config_sha256 = (
    "f8d1eda36b663bfdba43e986ccef060dd461e0ae3400ae7813c8c7ba6d32a398"
)
stage_b_native_patch_dataset_manifest_sha256 = None
stage_b_native_patch_dataset_manifest_sha256_by_source = {
    "refcoco": "3551d439c80e169e5216b12d7359a33829014403e3067d0ad0b852a133bf37dd",
    "refcocoplus": "872dcbe3989a33f8870b7e502e45c842e6e93d27c43a13563bfeb8aff4d94e58",
    "refcocog": "cf943ac2b4d3328dca90cf47c2a2321acd86084f2e757ef0ca6a399e560ef309",
}
stage_b_native_patch_dataset_receipt_sha256 = (
    "96d11562d1a0f064bdf5de676b48409967b35b31441777b7dac662b792d7eb94"
)
stage_b_native_patch_dataset_receipt_canonical_sha256 = (
    "b791bbde32f891ac5e4c30b5e511c4cd06433ba1980dc4442f05184100d9dca1"
)

stage_b_data_driven_sampling_contract = "deterministic_epoch_ledger_v1"
stage_b_data_driven_sampler_seed = 43
stage_b_data_driven_loader_seed = 1043
stage_b_data_driven_required_allocator_env = "PYTORCH_CUDA_ALLOC_CONF"
stage_b_data_driven_required_allocator_conf = "expandable_segments:True"

stage_b_native_patch_expected_max_train_iters = 200
stage_b_native_patch_expected_gradient_accumulation_steps = 2
stage_b_native_patch_expected_num_workers = 8
stage_b_native_patch_expected_seed = 42

batch_size = 36
epochs = 250
save_checkpoint_interval = 200
