from config.ablations.cfg_stageb_data_driven_dd1_pairtop1_fair_v2 import *  # noqa: F401,F403

# Deterministic 64-pair memorization gate. With one B64 batch per epoch,
# epochs=500 and --max_train_iters 500 produce exactly 500 optimizer updates.
stage_b_data_driven_assignment_dataset_scope = (
    "official_assignment_overfit64_u500_v1"
)
stage_b_data_driven_assignment_dataset_config_path = (
    "/media/haoyi/T9/pivot/"
    "config/datasets_stageb_data_driven_dd1_pairtop1_overfit64.json"
)
stage_b_data_driven_assignment_dataset_config_sha256 = (
    "01db83b787258c57fbf9797c5c18d6ab7537d35f14592b510a2d16d931f3a010"
)
stage_b_data_driven_assignment_receipt_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_data_driven_assignment_overfit64_20260722/receipt.json"
)
stage_b_data_driven_assignment_receipt_sha256 = (
    "359924240b43eea3052ae5e18d4afd014a5d0b7e094deac117d2a9d826d57521"
)
stage_b_data_driven_assignment_manifest_sha256 = {
    "overfit64.jsonl": (
        "c9a763428bfdeff14e910978ca4fb423bec84de5e8bf5cc2a9664ea2959a529b"
    )
}
stage_b_data_driven_assignment_expected_rows = 64
stage_b_data_driven_assignment_expected_valid_rows = 64
stage_b_data_driven_assignment_overfit_support_tsv_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_data_driven_assignment_overfit64_20260722/"
    "overfit64_support_clean.tsv"
)
stage_b_data_driven_assignment_overfit_support_tsv_sha256 = (
    "d77f67a5d8e31d1284844698c49782b937bfd33987d78fd535f750adca8e56dd"
)
stage_b_data_driven_assignment_overfit_member_stream_sha256 = (
    "5d39f51b03d425c466755ae38aa3802aec3b09010986b372fc4a5d28c65a90d3"
)
stage_b_data_driven_assignment_overfit_heldout_sha256 = (
    "563910f87ae683866b51c5cc4694829681fcb890d0d3f0ebaab7caab512ccbaf"
)

batch_size = 64
epochs = 500
lr_drop = 500
persistent_workers = False
save_checkpoint_interval = 500
stage_b_data_driven_epoch_checkpoint_interval = 500
stage_b_data_driven_required_allocator_env = "PYTORCH_ALLOC_CONF"
stage_b_data_driven_required_allocator_conf = "expandable_segments:True"

# These parser-owned values must be supplied on the command line. main.py
# rejects any drift before model construction.
stage_b_data_driven_pairtop1_u500_expected_max_train_iters = 500
stage_b_data_driven_pairtop1_u500_expected_num_workers = 0
stage_b_data_driven_pairtop1_u500_expected_pin_memory = False
stage_b_data_driven_pairtop1_u500_expected_iter_checkpoint_interval = 500
stage_b_data_driven_pairtop1_u500_expected_save_checkpoint_interval = 500
