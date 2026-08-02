from config.ablations.cfg_stageb_data_driven_dd1_a1_relational_fair_v2 import *  # noqa: F401,F403

# DD1-H changes only the text-rank denominator. The primary referent is
# positive, queries that unambiguously hit another same-category instance are
# hard negatives, and all background/ambiguous queries are ignored by rank.
# Patch supervision, model tensors, initializer, and sampling ledger remain the
# fair-v2 DD1 control values.
stage_b_data_driven_variant_id = "DD1-H"
stage_b_data_driven_rank_supervision = "primary_vs_same_category_aux_v1"
stage_b_data_driven_rank_negative_iou_threshold = 0.3
stage_b_data_driven_strict_sample_identity = True

# Content-addressed control evidence makes the one-variable comparison
# reconstructable even though this workspace has no usable Git metadata.
_dd1_control_root = (
    "/media/haoyi/T9/pivot/outputs/paper_cvpr_v1/"
    "data_driven_dd1_a1_relational_fair_v2_seed42_b64_u5020_v1"
)
stage_b_data_driven_control_checkpoint_path = (
    f"{_dd1_control_root}/checkpoint_iter.pth"
)
stage_b_data_driven_control_checkpoint_sha256 = (
    "e5cc60c9b6639bec3a922c4c28bb99771a95ef7fc85b2c7a697be405ceafa9be"
)
stage_b_data_driven_control_resolved_args_path = (
    f"{_dd1_control_root}/config_args_all.json"
)
stage_b_data_driven_control_resolved_args_sha256 = (
    "857f821dcdd4ca7aca322b3390ef46e2b4f2f352c72f590fa0f64333581f6458"
)
stage_b_data_driven_control_rank_summary_path = (
    f"{_dd1_control_root}/evaluations/"
    "rank_only_refcocog_val_b16_protocol_v1/summary.json"
)
stage_b_data_driven_control_rank_summary_sha256 = (
    "6cb6e0b37e74f81ec2562b484ede980eebb73ddb809b658e75c89bc29300a089"
)
stage_b_data_driven_control_gap3_summary_path = (
    f"{_dd1_control_root}/evaluations/"
    "gap3_refcocog_val_b16_protocol_v1/summary.json"
)
stage_b_data_driven_control_gap3_summary_sha256 = (
    "4625305d79fce4495209083698c546c159db605eebe36a826d5b077a2a4227bc"
)
stage_b_data_driven_control_source_snapshot_path = (
    f"{_dd1_control_root}/source_snapshot_pre_dd1h_20260722.tar.gz"
)
stage_b_data_driven_control_source_snapshot_sha256 = (
    "3a361006a617776e57311ceff79bf367cea9fb3a6d9b21df06f2412cc5cce307"
)
stage_b_data_driven_control_source_snapshot_supplement_path = (
    f"{_dd1_control_root}/"
    "source_snapshot_pre_dd1h_20260722_import_chain_supplement.tar.gz"
)
stage_b_data_driven_control_source_snapshot_supplement_sha256 = (
    "00665773ef60dacdd338da7e58a6a97f527ce8ec587e030f39641f3245259765"
)
stage_b_data_driven_control_source_snapshot_supplement_receipt_path = (
    f"{_dd1_control_root}/"
    "source_snapshot_pre_dd1h_20260722_import_chain_supplement_receipt.json"
)
stage_b_data_driven_control_source_snapshot_supplement_receipt_sha256 = (
    "5a610911839f77213d7e78ce4da99bb502339a4e742f0ff2b586c31266040926"
)
del _dd1_control_root
