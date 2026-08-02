from config.ablations.cfg_stageb_v19_full_text_base_plus_gate import *  # noqa: F401,F403

# One checkpoint, two parameter-disjoint dense GroundingDINO text/image towers.
# Stage A owns boxes, patch-category admission, and category-only ranking.
# Modifier-bearing expressions use text ranking without an additive patch logit.
stage_b_dense_duty = True
stage_b_v11_fixed_text = True
stage_b_v11_num_layers = 6
stage_b_v11_candidate_topk = 50
stage_b_v11_expression_microbatch = 16
stage_b_v11_assert_fixed_candidates = True
stage_b_v15_decoupled_confidence = True
stage_b_v15_patch_rank_fusion = False
stage_b_v15_patch_rank_weight = 0.0
stage_b_v15_exclude_canonical_from_score = True
stage_b_v15_separate_grad_clip = False
stage_b_v15_validity_lr = None
stage_b_v22_score_ownership = "independent_decoders_two_phase"
stage_b_v22_gradient_diagnostic_interval = 0

stage_b_dense_duty_category_gate_max_gap = 3.0
stage_b_dense_duty_patch_score_clip = 5.0
stage_b_dense_duty_confidence_hidden_dim = 256
stage_b_dense_duty_confidence_pool_temperature = 0.2
stage_b_dense_duty_confidence_pool_topk = 10
text_encoder_type = (
    "/home/haoyi/.cache/huggingface/hub/models--bert-base-uncased/"
    "snapshots/86b5e0934494bd15c9632b12f734a8a67f723594"
)

# Trace-supervised token targets: positive/shared words remain positive while
# only the aligned edited word is negative in the paired TN expression.
stage_b_v21_token_objective = "edit_bce"
stage_b_v21_token_weight = 1.0
stage_b_v21_token_positive_weight = 1.0
stage_b_v21_token_shared_weight = 0.25
stage_b_v21_token_edit_weight = 1.0
stage_b_v21_allow_legacy_token_diff_fallback = False
stage_b_dense_duty_token_role_source = "exact_direct_trace_v1"
# Formal headline training admits TN shared/changed roles only when the trace
# alone reconstructs the TN. Invalid TN roles are skipped; source-independent
# positive-expression token BCE and sample-level objectives remain active.
stage_b_dense_duty_allow_incidental_trace_edits = False

# This experiment is outside the proposal-covered Table-B ablation. Its TN
# rows are required by the dataset to carry image_global_topk_verified scope.
stage_b_v19_allow_scope_labeled_tn_ablation = False
stage_b_v19_table_b_id = ""
stage_b_v19_table_b_scope_allowlist = []
stage_b_v19_table_b_audit = ""
stage_b_v19_table_b_audit_sha256 = ""
stage_b_v19_table_b_allow_single_edit_token_provenance = False

# Initialization is pretraining, not a Stage-B teacher. The rank phase checks
# these exact files and hashes; B58 is explicitly forbidden as a tensor source.
stage_b_dense_duty_no_stageb_teacher = True
stage_b_dense_duty_execution_scope = "formal"
stage_b_dense_duty_evaluation_scope = "formal"
stage_b_dense_duty_base_checkpoint_path = (
    "/media/haoyi/T9/gdino/outputs/stageA_coco_multipatch/checkpoint0006.pth"
)
stage_b_dense_duty_base_checkpoint_sha256 = (
    "a4f153c8cbd9b408b9479901e27ec486a10f393013193d44b0da1dcd1888cb91"
)
stage_b_dense_duty_text_checkpoint_path = (
    "/media/haoyi/T9/pivot/weights/groundingdino_swint_ogc.pth"
)
stage_b_dense_duty_text_checkpoint_sha256 = (
    "3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799"
)
stage_b_dense_duty_forbidden_checkpoint_sha256 = [
    "b58e5209dc07dbffb2e5ed3d792e0db70c3306ec2ed24389693b4aeeebab1157"
]
stage_b_dense_duty_rank_expected_optimizer_updates = 10295
stage_b_dense_duty_confidence_expected_optimizer_updates = 4412
stage_b_dense_duty_rank_dataset_config_sha256 = (
    "6cc541d8347468c625ca0785a8a87c6a85ef9e85ac911a301feab4c25061ceba"
)
stage_b_dense_duty_tn_manifest_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_gdino_adapter_semantic_partition_20260717/single_edit_train.jsonl"
)
stage_b_dense_duty_tn_manifest_sha256 = (
    "276dc5a67c6e7a6654d6daa6a88cb99b9c59b1c52f84ef93205a3d6326b1b529"
)
stage_b_dense_duty_trace_audit_path = (
    "/media/haoyi/T9/pivot/data/ablations/"
    "stageb_dense_duty_trace_audit_20260728/receipt.json"
)
stage_b_dense_duty_trace_audit_sha256 = (
    "7414a08cabfff598c27cc6bf4fb5c0f3817fb939a185e05ff9d3c8e6b1f8cf78"
)
stage_b_dense_duty_trace_total_rows = 14196
stage_b_dense_duty_trace_lexical_valid_rows = 13892
stage_b_dense_duty_trace_direct_token_valid_rows = 13890
stage_b_dense_duty_trace_direct_token_invalid_rows = 306
stage_b_dense_duty_trace_no_unique_reconstruction_rows = 265
stage_b_dense_duty_trace_deletion_only_rows = 39
stage_b_dense_duty_trace_canonical_surface_rejections = 2

stage_b_v15_scorer_init_checkpoint = stage_b_dense_duty_text_checkpoint_path
only_train_keywords = ["stage_b_fixed_text_scorer"]
only_train_exclude_keywords = []
stage_b_v11_trainable_params_min = 1
stage_b_v11_trainable_params_max = 100000000

# Measured on the RTX 5090 before formal training. Both rank and confidence
# phases keep 8+ GiB of headroom while exercising the dense text tower at high
# occupancy. Accumulation remains an argparse setting, so the formal validator
# binds the launch command to the values below.
stage_b_dense_duty_expected_physical_batch_size = 16
stage_b_dense_duty_expected_gradient_accumulation_steps = 4
stage_b_dense_duty_expected_expression_microbatch = 16
batch_size = stage_b_dense_duty_expected_physical_batch_size
lr = 1e-5
lr_linear_proj_mult = 1e-6
weight_decay = 1e-4
clip_max_norm = 0.1
amp_init_scale = 256.0
amp_max_consecutive_skips = 8
skip_eval = True
use_coco_eval = False
fix_size = True
strong_aug = False
data_aug_hflip_prob = 0.0
save_checkpoint_interval = 1
log_patch_sanity = False
log_stage_b_patch_drift = False
