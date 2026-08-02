from config.ablations.cfg_stageb_native_patch_category_d1 import *  # noqa: F401,F403

# This parser-owned field is supplied explicitly on the command line.  Removing
# it here keeps the sealed base config immutable and avoids an argparse/config
# collision in main.py.
del gradient_accumulation_steps

stage_b_native_patch_initializer_sha256 = (
    "addec47338c2e36a3121d999370349d2351535f6ff7334729424aeb1bcd880b4"
)
stage_b_native_patch_initializer_path = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "native_patch_category_initializers/d1_seed42/checkpoint_d1_init.pth"
)
stage_b_native_patch_execution_scope = "native_patch_category_d1_u500_v1"
stage_b_native_patch_formal_config_path = (
    "/media/haoyi/T9/pivot/config/ablations/"
    "cfg_stageb_native_patch_category_d1_u500.py"
)
stage_b_native_patch_formal_output_dir = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "native_patch_category_d1_seed42_b36a2_lr3e4_u500_v1"
)
stage_b_data_driven_sampling_contract = "deterministic_epoch_ledger_v1"
stage_b_data_driven_sampler_seed = 42
stage_b_data_driven_loader_seed = 1042
stage_b_data_driven_required_allocator_env = "PYTORCH_CUDA_ALLOC_CONF"
stage_b_data_driven_required_allocator_conf = "expandable_segments:True"

# Immutable experiment bindings recorded again in every saved checkpoint args.
stage_b_native_patch_dataset_config_path = (
    "/media/haoyi/T9/pivot/"
    "config/datasets_stageb_native_patch_category_d1_train_20260724.json"
)
stage_b_native_patch_dataset_config_sha256 = (
    "7d7ef28db4139d4c421faa5ae1229bdc03021251d58ac8f5def5623a37b8229e"
)
stage_b_native_patch_dataset_manifest_sha256 = (
    "2c2158b8b4952cda18713866797d87db4ba7dd20d6cdf6b8ba70d73b509016fd"
)
stage_b_native_patch_dataset_receipt_sha256 = (
    "cc28e4ed35c8b7fe6705b28bad72fde0548b093386f41373f0f9705662e7ea59"
)
stage_b_native_patch_expected_max_train_iters = 500
stage_b_native_patch_expected_gradient_accumulation_steps = 2
stage_b_native_patch_expected_num_workers = 8
stage_b_native_patch_expected_seed = 42

batch_size = 36
epochs = 250
save_checkpoint_interval = 500
