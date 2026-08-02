from config.ablations.cfg_stageb_data_driven_dd1_a1_relational_fair_v2 import *  # noqa: F401,F403

# Official human expressions and COCO GT only. PairTop1 is the sole rank loss.
stage_b_data_driven_variant_id = "DD1-PairTop1"
stage_b_data_driven_rank_supervision = (
    "official_same_image_same_category_assignment_v1"
)
stage_b_data_driven_rank_weight = 0.0
stage_b_data_driven_assignment_weight = 1.0
stage_b_data_driven_deployment_weight = 0.0
stage_b_data_driven_rank_negative_iou_threshold = 0.3
stage_b_data_driven_category_gate_max_gap = 3.0
stage_b_data_driven_patch_score_clip = 5.0
stage_b_data_driven_strict_sample_identity = True
fix_size = True
strong_aug = False
data_aug_hflip_prob = 0.0
batch_size = 64

stage_b_data_driven_assignment_dataset_scope = (
    "official_assignment_full_321327_v1"
)
stage_b_data_driven_assignment_dataset_config_path = (
    "/media/haoyi/T9/pivot/"
    "config/datasets_stageb_data_driven_dd1_official_assignment_three_ref.json"
)
stage_b_data_driven_assignment_dataset_config_sha256 = (
    "5c659bb2de76f32d644af330b4284550c104ca24a96f9aeef6a004a698acf89a"
)
stage_b_data_driven_assignment_receipt_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_data_driven_assignment_pairs_20260722/receipt.json"
)
stage_b_data_driven_assignment_receipt_sha256 = (
    "7b9ce1c911a2e1f0b67464243df8290fc2baf0786a2a3b131ddc57a6a6d2ddaa"
)
stage_b_data_driven_assignment_manifest_sha256 = {
    "refcoco_stageb_phrase_v1.jsonl": "f253c8bec4d15e421b11c42d8114e17c41bc32ed28f2614e34fe341e4da32592",
    "refcocoplus_stageb_phrase_v1.jsonl": "69039abbd5baeb1173c849c19c55128aea8053271ab79f0cc16fa679000deaa8",
    "refcocog_stageb_phrase_v1.jsonl": "378c5e34899e4113cd5dca1fd60352b362924e7969a717ea729b8278ce97a553",
}
stage_b_data_driven_assignment_expected_rows = 321327
stage_b_data_driven_assignment_expected_valid_rows = 274582
stage_b_data_driven_no_teacher_contract = (
    "b58_only_random_independent_heads_v1"
)
stage_b_gdino_score_adapter = False
stage_b_u0_patch_rank = False
stage_b_v7 = False
stage_b_v11_fixed_text = False
stage_b_legacy_global_gate = False
