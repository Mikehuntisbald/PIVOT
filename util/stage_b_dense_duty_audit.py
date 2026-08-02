import ast
import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch


STATE_FINGERPRINT_SCHEMA = "pivot.stageb.dense_duty_state_fingerprint/v1"
CHECKPOINT_AUDIT_SCHEMA = "pivot.stageb.dense_duty_checkpoint_audit/v1"
FINGERPRINT_ARG = "stage_b_dense_duty_initial_state_fingerprint"
TRAINING_CONTRACT_ARG = "stage_b_dense_duty_training_contract"
SOURCE_CLOSURE_ARG = "stage_b_dense_duty_source_closure"
CODE_SOURCE_CLOSURE_SCHEMA = "pivot.stageb.dense_duty_code_source_closure/v1"
CONFIG_SOURCE_CLOSURE_SCHEMA = "pivot.stageb.dense_duty_config_source_closure/v1"
SOURCE_CLOSURE_SCHEMA = "pivot.stageb.dense_duty_source_closure/v1"
STRICT_RESUME_CHECKPOINT_NAME = "checkpoint_iter.pth"
_REPO_ROOT = Path(__file__).resolve().parents[1]
_FORMAL_PHASE_CONFIGS = {
    "rank": "config/ablations/cfg_stageb_dense_duty_rank_20260728.py",
    "confidence": (
        "config/ablations/cfg_stageb_dense_duty_confidence_20260728.py"
    ),
}
_ADAPTER_OWNERSHIP = "rank_tower_stopgrad_token_adapter_two_phase"
_ADAPTER_FORMAL_CONFIDENCE_CONFIGS = (
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_20260730.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_veto_20260730.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_veto_gate_20260731.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_veto_cap_20260731.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_20260731.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_calibrated_20260731.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_carrier_balanced_20260731.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_carrier_quarter_20260731.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_carrier_pair_20260731.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_dual_carrier_pair_20260731.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_rank_evidence_20260731.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_rank_affine_20260731.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_gate_margin_20260731.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_carrier_slope_20260731.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_carrier_affine_20260731.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_tail_ste_20260731.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_tail_carrier_20260731.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_tail_paired_20260731.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_tail_paired_rank_channel_20260731.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_candidate_absolute_20260731.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_candidate_calibrated_20260731.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_candidate_normalized_20260731.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_candidate_asymmetric_20260731.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_candidate_set_attention_20260731.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_candidate_q05_20260801.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_candidate_tail_balanced_20260801.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_candidate_tail_quarter_20260801.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_candidate_tail_bounded_20260801.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_candidate_tail_elementwise_20260801.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_candidate_gate_zero_offset_20260801.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_candidate_hardest_edit_20260801.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_candidate_role_complete_carrier_20260801.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_candidate_tn_only_carrier_pair_20260801.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_candidate_deployed_routing_20260801.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_candidate_split_heads_20260801.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_candidate_split_positive_tail_20260801.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_candidate_split_boundary_routing_20260801.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_candidate_split_fpr_active_set_20260801.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_candidate_split_global_trust_veto_20260801.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_candidate_split_strong_boundary_routing_20260801.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_candidate_split_independent_deployed_router_20260802.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_candidate_sample_calibrator_20260802.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_fulltext_global_absolute_20260802.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_fulltext_global_absolute_exact_residual_20260802.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_fulltext_global_independent_absolute_20260802.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_deployment_owned_global_20260802.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_deployed_global_balanced_absolute_20260802.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_deployment_owned_global_stable_fpr95_active_set_20260802.py",
    "config/ablations/cfg_stageb_dense_duty_confidence_adapter_deployment_owned_query_global_20260802.py",
)
_V53_CONFIDENCE_REVISION = "word_veto_rank_full_expression_global_absolute_v53"
_V53_CONFIDENCE_HEAD_CONTRACT = (
    "split_token_veto_fulltext_global_absolute_v7"
)
_V53_CONFIDENCE_POOL_CONTRACT = (
    "detached_rank_full_expression_candidate_residual_global_pool_v10"
)
_V53_CONFIDENCE_GATE_CONTRACT = (
    "candidate_raw_patch_asymmetric_monotone_veto_absolute_logit_v13"
)
_V53_FORMAL_ADMISSION_CONTRACT = (
    "u400_word_veto_rank_full_expression_global_absolute_"
    "confidence_strict1607_v53"
)
_V54_CONFIDENCE_REVISION = (
    "word_veto_rank_full_expression_global_absolute_exact_residual_v54"
)
_V54_CONFIDENCE_POOL_CONTRACT = (
    "detached_rank_full_expression_candidate_residual_global_pool_"
    "exact_rank_max_reference_v11"
)
_V54_POSITIVE_TRUST_CONTRACT = (
    "exact_frozen_rank_max_confidence_delta_v3"
)
_V54_FORMAL_ADMISSION_CONTRACT = (
    "u400_word_veto_rank_full_expression_global_absolute_exact_residual_"
    "confidence_strict1607_v54"
)
_V55_CONFIDENCE_REVISION = (
    "word_veto_rank_full_expression_global_independent_absolute_v55"
)
_V55_CONFIDENCE_HEAD_CONTRACT = (
    "split_token_veto_local_candidate_global_absolute_v8"
)
_V55_CONFIDENCE_POOL_CONTRACT = (
    "detached_rank_full_expression_local_candidate_"
    "frozen_rank_global_pool_v12"
)
_V55_POSITIVE_TRUST_CONTRACT = "absolute_global_pool_logit_v4"
_V55_FORMAL_ADMISSION_CONTRACT = (
    "u400_word_veto_rank_full_expression_global_independent_absolute_"
    "confidence_strict1607_v55"
)
_V56_CONFIDENCE_REVISION = (
    "word_veto_rank_full_expression_deployment_owned_global_v56"
)
_V56_CONFIDENCE_HEAD_CONTRACT = (
    "split_token_veto_deployment_owned_global_absolute_v9"
)
_V56_CONFIDENCE_POOL_CONTRACT = (
    "detached_rank_full_expression_deployment_owned_global_pool_v13"
)
_V56_FORMAL_ADMISSION_CONTRACT = (
    "u400_word_veto_rank_full_expression_deployment_owned_global_"
    "confidence_strict1607_v56"
)
_V57_CONFIDENCE_REVISION = (
    "word_veto_rank_full_expression_deployed_global_balanced_absolute_v57"
)
_V57_FORMAL_ADMISSION_CONTRACT = (
    "u400_word_veto_rank_full_expression_deployed_global_balanced_absolute_"
    "confidence_strict1607_v57"
)
_V58_CONFIDENCE_REVISION = (
    "word_veto_rank_full_expression_deployment_owned_global_"
    "stable_fpr95_active_set_v58"
)
_V58_FORMAL_ADMISSION_CONTRACT = (
    "u400_word_veto_rank_full_expression_deployment_owned_global_"
    "stable_fpr95_active_set_confidence_strict1607_v58"
)
_V59_CONFIDENCE_REVISION = (
    "word_veto_rank_full_expression_deployment_owned_query_global_v59"
)
_V59_CONFIDENCE_HEAD_CONTRACT = (
    "split_token_veto_deployment_owned_query_global_absolute_v10"
)
_V59_CONFIDENCE_POOL_CONTRACT = (
    "detached_rank_full_expression_monotone_query_"
    "deployment_owned_global_pool_v14"
)
_V59_POSITIVE_TRUST_CONTRACT = "absolute_global_confidence_logit_v2"
_V59_FORMAL_ADMISSION_CONTRACT = (
    "u400_word_veto_rank_full_expression_deployment_owned_query_"
    "global_confidence_strict1607_v59"
)
_V60_CONFIDENCE_REVISION = (
    "word_veto_rank_full_expression_deployment_owned_query_veto_v60"
)
_V60_CONFIDENCE_HEAD_CONTRACT = (
    "split_token_veto_deployment_owned_query_veto_global_absolute_v11"
)
_V60_CONFIDENCE_POOL_CONTRACT = (
    "detached_rank_full_expression_token_conditioned_query_veto_"
    "deployment_owned_global_pool_v15"
)
_V60_POSITIVE_TRUST_CONTRACT = "absolute_global_confidence_logit_v2"
_V60_FORMAL_ADMISSION_CONTRACT = (
    "u400_word_veto_rank_full_expression_deployment_owned_query_"
    "veto_confidence_strict1607_v60"
)
_V56_ACTIVE_TENSOR_COUNT = 59
_V56_ACTIVE_ELEMENT_COUNT = 468_164
_V56_OWNER_TENSOR_COUNTS = {
    "token_veto": 21,
    "global_absolute": 38,
}
_V53_ACTIVE_TENSOR_COUNT = 65
_V53_ACTIVE_ELEMENT_COUNT = 534_725
_V53_OWNER_TENSOR_COUNTS = {
    "token_veto": 21,
    "global_absolute": 44,
}
_V61_FULL_DECODER_ACTIVE_ELEMENT_COUNT = 25_664_258
_V61_FULL_DECODER_ACTIVE_TENSOR_COUNT = 368
_V61_FULL_DECODER_OWNER_TENSOR_COUNTS = {
    "token_veto": 356,
    "global_absolute": 12,
}
STRICT_RESUME_REQUIRED_KEYS = frozenset(
    {
        "model",
        "criterion",
        "optimizer",
        "lr_scheduler",
        "scaler",
        "epoch",
        "iteration",
        "optimizer_updates",
        "epoch_finished",
        "rng_state",
        "epoch_rng_state",
        "args",
        "checkpoint_reason",
    }
)

_RESUME_CONTRACT_KEYS = (
    "stage_b_dense_duty",
    "stage_b_dense_duty_phase",
    "stage_b_dense_duty_no_stageb_teacher",
    "stage_b_dense_duty_execution_scope",
    "stage_b_dense_duty_rank_expected_optimizer_updates",
    "stage_b_dense_duty_confidence_expected_optimizer_updates",
    "stage_b_dense_duty_expected_physical_batch_size",
    "stage_b_dense_duty_expected_gradient_accumulation_steps",
    "stage_b_dense_duty_expected_expression_microbatch",
    SOURCE_CLOSURE_ARG,
    "stage_b_v22_score_ownership",
    "stage_b_v22_train_phase",
    "stage_b_dense_duty_base_checkpoint_sha256",
    "stage_b_dense_duty_text_checkpoint_sha256",
    "stage_b_dense_duty_tn_manifest_sha256",
    "stage_b_dense_duty_trace_audit_path",
    "stage_b_dense_duty_trace_audit_sha256",
    "stage_b_dense_duty_trace_total_rows",
    "stage_b_dense_duty_trace_lexical_valid_rows",
    "stage_b_dense_duty_trace_direct_token_valid_rows",
    "stage_b_dense_duty_trace_direct_token_invalid_rows",
    "stage_b_dense_duty_trace_no_unique_reconstruction_rows",
    "stage_b_dense_duty_trace_deletion_only_rows",
    "stage_b_dense_duty_trace_canonical_surface_rejections",
    "stage_b_dense_duty_dataset_config_path",
    "stage_b_dense_duty_dataset_config_sha256",
    "stage_b_dense_duty_rank_dataset_config_sha256",
    "stage_b_v11_fixed_text",
    "stage_b_v11_num_layers",
    "stage_b_v11_candidate_topk",
    "stage_b_v11_expression_microbatch",
    "stage_b_v11_assert_fixed_candidates",
    "stage_b_v11_trainable_params_min",
    "stage_b_v11_trainable_params_max",
    "stage_b_dense_duty_category_gate_max_gap",
    "stage_b_dense_duty_patch_score_clip",
    "stage_b_dense_duty_confidence_hidden_dim",
    "stage_b_dense_duty_confidence_pool_temperature",
    "stage_b_dense_duty_confidence_pool_topk",
    "stage_b_dense_duty_allow_incidental_trace_edits",
    "stage_b_dense_duty_token_role_source",
    "stage_b_v15_decoupled_confidence",
    "stage_b_v15_patch_rank_fusion",
    "stage_b_v15_patch_rank_weight",
    "stage_b_v15_exclude_canonical_from_score",
    "stage_b_v15_scorer_init_checkpoint",
    "stage_b_v15_separate_grad_clip",
    "stage_b_v15_validity_lr",
    "stage_b_v21_token_objective",
    "stage_b_v21_token_weight",
    "stage_b_v21_token_positive_weight",
    "stage_b_v21_token_shared_weight",
    "stage_b_v21_token_edit_weight",
    "stage_b_v21_allow_legacy_token_diff_fallback",
    "stage_b_v11_positive_iou_threshold",
    "stage_b_v11_negative_iou_threshold",
    "stage_b_v11_listwise_temperature",
    "stage_b_v11_listwise_weight",
    "stage_b_v11_local_tn_rank_margin",
    "stage_b_v11_local_tn_rank_weight",
    "stage_b_v11_predicate_tn_rank_margin",
    "stage_b_v11_predicate_tn_rank_weight",
    "stage_b_v11_local_anchor_weight",
    "stage_b_v11_positive_anchor_logit",
    "stage_b_v11_negative_anchor_logit",
    "stage_b_v11_global_tn_negative_weight",
    "stage_b_v11_global_tn_tail_weight",
    "stage_b_v11_global_tn_tail_topk",
    "stage_b_v11_global_tn_tail_temperature",
    "stage_b_v11_global_tn_tail_target_logit",
    "stage_b_v11_batch_tail_separation_weight",
    "stage_b_v11_batch_positive_quantile",
    "stage_b_v11_batch_negative_quantile",
    "stage_b_v11_batch_tail_margin",
    "stage_b_v14_local_absolute_weight",
    "stage_b_v14_local_absolute_gamma",
    "stage_b_v14_predicate_absolute_weight",
    "stage_b_v14_predicate_absolute_gamma",
    "stage_b_v14_tail_queue_weight",
    "stage_b_v14_tail_queue_size",
    "stage_b_v14_tail_queue_min_count",
    "stage_b_v14_tail_queue_positive_quantile",
    "stage_b_v14_tail_queue_negative_quantile",
    "stage_b_v14_tail_queue_temperature",
    "stage_b_v14_tail_queue_margin",
    "stage_b_v14_global_tn_all_candidates",
    "stage_b_v15_tail_queue_global_scores",
    "stage_b_v15_tail_queue_objective",
    "stage_b_v15_tail_queue_pair_weight",
    "stage_b_v15_tail_queue_pair_margin",
    "stage_b_v15_tail_queue_positive_trust_weight",
    "stage_b_v15_tail_queue_positive_trust_margin",
    "max_text_len",
    "hidden_dim",
    "enc_layers",
    "dec_layers",
    "dim_feedforward",
    "nheads",
    "dropout",
    "num_feature_levels",
    "text_encoder_type",
    "batch_size",
    "gradient_accumulation_steps",
    "amp",
    "amp_init_scale",
    "amp_max_consecutive_skips",
    "lr",
    "lr_backbone",
    "lr_linear_proj_mult",
    "param_dict_type",
    "weight_decay",
    "clip_max_norm",
    "epochs",
    "lr_drop",
    "lr_drop_list",
    "onecyclelr",
    "multi_step_lr",
    "seed",
    "datasets",
    "max_train_iters",
    "iter_checkpoint_interval",
    "save_checkpoint_interval",
    "skip_eval",
    "use_coco_eval",
    "use_ema",
    "fix_size",
    "data_aug_hflip_prob",
    "strong_aug",
    "remove_difficult",
    "num_workers",
    "prefetch_factor",
    "pin_memory",
    "persistent_workers",
    "world_size",
    "distributed",
    "find_unused_params",
)

_ADAPTER_RESUME_CONTRACT_KEYS = (
    "stage_b_dense_duty_confidence_adapter_dim",
    "stage_b_dense_duty_confidence_init_seed",
    "stage_b_dense_duty_confidence_token_contract",
    "stage_b_dense_duty_confidence_pool_feature_contract",
    "stage_b_dense_duty_rank_source_checkpoint_path",
    "stage_b_dense_duty_rank_source_checkpoint_sha256",
    "stage_b_dense_duty_rank_source_optimizer_updates",
    "stage_b_dense_duty_rank_source_checkpoint_reason",
    "stage_b_dense_duty_rank_source_rank_sha256",
    "stage_b_dense_duty_rank_source_transferred_sha256",
)

_WORD_VETO_RESUME_CONTRACT_KEYS = (
    "stage_b_dense_duty_confidence_revision",
    "stage_b_dense_duty_confidence_phrase_aggregation",
    "stage_b_dense_duty_confidence_word_softmin_temperature",
    "stage_b_dense_duty_confidence_veto_gate_scale",
    "stage_b_dense_duty_positive_trust_contract",
    "stage_b_dense_duty_confidence_tn_scope",
)

_POSITIVE_TAIL_GRADIENT_RESUME_CONTRACT_KEYS = (
    "stage_b_v15_tail_queue_positive_gradient_contract",
)

_TOKEN_EDIT_CARRIER_RESUME_CONTRACT_KEYS = (
    "stage_b_v21_token_edit_query_scope",
)

_RAW_VETO_GATE_RESUME_CONTRACT_KEYS = (
    "stage_b_dense_duty_raw_veto_gate_weight",
    "stage_b_dense_duty_raw_veto_positive_margin",
    "stage_b_dense_duty_raw_veto_tn_margin",
)

_ABSOLUTE_CAP_RESUME_CONTRACT_KEYS = (
    "stage_b_dense_duty_raw_veto_query_scope",
    "stage_b_dense_duty_confidence_veto_gate_offset",
    "stage_b_dense_duty_confidence_veto_coverage_offset",
    "stage_b_dense_duty_confidence_veto_coverage_ramp",
    "stage_b_dense_duty_confidence_veto_cap_temperature",
    "stage_b_dense_duty_confidence_veto_cap_initial_ceiling",
)

_CARRIER_BALANCED_RESUME_CONTRACT_KEYS = (
    "stage_b_dense_duty_raw_veto_tn_carrier_balance",
    "stage_b_dense_duty_confidence_carrier_selector_contract",
)

_CARRIER_PAIR_RESUME_CONTRACT_KEYS = (
    "stage_b_dense_duty_raw_veto_carrier_pair_weight",
    "stage_b_dense_duty_raw_veto_carrier_pair_margin",
)

_CARRIER_PAIR_GRADIENT_RESUME_CONTRACT_KEYS = (
    "stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract",
)

_DEPLOYED_VETO_ROUTING_RESUME_CONTRACT_KEYS = (
    "stage_b_dense_duty_confidence_gate_gradient_contract",
    "stage_b_dense_duty_deployed_veto_routing_weight",
    "stage_b_dense_duty_deployed_veto_positive_max",
    "stage_b_dense_duty_deployed_veto_tn_min",
    "stage_b_dense_duty_raw_veto_tn_carrier_balance",
    "stage_b_dense_duty_raw_veto_positive_carrier_balance",
    "stage_b_dense_duty_confidence_carrier_selector_contract",
    "stage_b_dense_duty_raw_veto_carrier_pair_weight",
    "stage_b_dense_duty_raw_veto_carrier_pair_margin",
    "stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract",
    "stage_b_dense_duty_raw_veto_tail_quantile",
    "stage_b_dense_duty_raw_veto_tail_temperature",
    "stage_b_dense_duty_raw_veto_tail_min_count",
)

_SPLIT_CONFIDENCE_HEAD_RESUME_CONTRACT_KEYS = (
    "stage_b_dense_duty_confidence_head_gradient_contract",
)

_V53_FULLTEXT_GLOBAL_ABSOLUTE_RESUME_CONTRACT_KEYS = (
    "stage_b_dense_duty_confidence_revision",
    "stage_b_dense_duty_confidence_head_gradient_contract",
    "stage_b_dense_duty_confidence_pool_feature_contract",
    "stage_b_dense_duty_confidence_gate_gradient_contract",
)

_V57_DEPLOYED_GLOBAL_ABSOLUTE_RESUME_CONTRACT_KEYS = (
    "stage_b_dense_duty_deployed_global_absolute_weight",
    "stage_b_dense_duty_deployed_global_absolute_gamma",
)

_TAIL_ALIGNED_SPLIT_RESUME_CONTRACT_KEYS = (
    "stage_b_dense_duty_deployed_veto_routing_reduction_contract",
    "stage_b_v15_tail_queue_positive_trust_reduction_contract",
)

_FPR_ACTIVE_SET_RESUME_CONTRACT_KEYS = (
    "stage_b_v15_tail_queue_negative_reduction_contract",
)

_GLOBAL_TRUST_VETO_RESUME_CONTRACT_KEYS = (
    "stage_b_v15_tail_queue_negative_reduction_contract",
)

_STRONG_BOUNDARY_ROUTING_RESUME_CONTRACT_KEYS = (
    "stage_b_v15_tail_queue_negative_reduction_contract",
    "stage_b_v21_token_edit_query_scope",
)

_DUAL_CARRIER_RESUME_CONTRACT_KEYS = (
    "stage_b_dense_duty_raw_veto_positive_carrier_balance",
)

_RANK_EVIDENCE_RESUME_CONTRACT_KEYS = (
    "stage_b_dense_duty_confidence_rank_evidence_contract",
)

_GATE_MARGIN_RESUME_CONTRACT_KEYS = (
    "stage_b_dense_duty_confidence_residual_parameterization_gain",
)

_GATE_GRADIENT_RESUME_CONTRACT_KEYS = (
    "stage_b_dense_duty_confidence_gate_gradient_contract",
)

_TAIL_CARRIER_RESUME_CONTRACT_KEYS = (
    "stage_b_dense_duty_raw_veto_tail_quantile",
    "stage_b_dense_duty_raw_veto_tail_temperature",
    "stage_b_dense_duty_raw_veto_tail_min_count",
)

_PROBE_ADMISSION_RESUME_CONTRACT_KEYS = (
    "stage_b_dense_duty_confidence_probe_admission_contract",
    "stage_b_dense_duty_confidence_probe_admission_report",
    "stage_b_dense_duty_confidence_probe_admission_audit",
)

_PACKED_FORWARD_CONTRACT_KEYS = (
    "stage_b_dense_duty_forward_pack_factor",
    "stage_b_dense_duty_logical_loss_batch_size",
    "stage_b_dense_duty_expected_forward_batch_size",
    "stage_b_dense_duty_expected_logical_batches_per_epoch",
    "stage_b_dense_duty_expected_physical_forwards_per_epoch",
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_root(path: Optional[Path] = None) -> Path:
    root = Path(_REPO_ROOT if path is None else path).expanduser().resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"dense-duty source root is not a real directory: {root}")
    return root


def _repo_relative_regular_file(path: Path, *, repo_root: Path) -> tuple[Path, str]:
    """Return a lexical in-repo regular file and its portable path."""
    root = _repo_root(repo_root)
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = Path(os.path.abspath(os.fspath(candidate)))
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(
            f"dense-duty source file is outside the repository: {candidate}"
        ) from exc
    if not relative.parts or ".." in relative.parts:
        raise RuntimeError(f"dense-duty source file path is invalid: {candidate}")

    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        try:
            mode = cursor.lstat().st_mode
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"dense-duty source file does not exist: {candidate}"
            ) from exc
        if stat.S_ISLNK(mode):
            raise RuntimeError(
                f"dense-duty source closure forbids symlinks: {cursor}"
            )
    if not stat.S_ISREG(candidate.lstat().st_mode):
        raise RuntimeError(
            f"dense-duty source closure accepts regular files only: {candidate}"
        )
    if candidate.resolve(strict=True) != candidate:
        raise RuntimeError(
            f"dense-duty source path does not resolve lexically: {candidate}"
        )
    portable = relative.as_posix()
    try:
        portable.encode("ascii")
    except UnicodeEncodeError as exc:
        raise RuntimeError(
            f"dense-duty source path is not ASCII: {portable!r}"
        ) from exc
    return candidate, portable


def _repo_python_files(directory: str, *, repo_root: Path) -> list[Path]:
    root = _repo_root(repo_root)
    source_root = root / directory
    if source_root.is_symlink() or not source_root.is_dir():
        raise RuntimeError(
            f"dense-duty source directory is missing or symlinked: {source_root}"
        )
    result: list[Path] = []
    for current, directory_names, file_names in os.walk(
        source_root, topdown=True, followlinks=False
    ):
        current_path = Path(current)
        for name in directory_names:
            child = current_path / name
            if child.is_symlink():
                raise RuntimeError(
                    "dense-duty source closure forbids symlinked source "
                    f"directories: {child}"
                )
        for name in file_names:
            if not name.endswith(".py"):
                continue
            candidate, _ = _repo_relative_regular_file(
                current_path / name, repo_root=root
            )
            result.append(candidate)
    return result


def _file_record(path: Path, *, repo_root: Path) -> dict[str, Any]:
    resolved, portable = _repo_relative_regular_file(path, repo_root=repo_root)
    return {
        "path": portable,
        "size_bytes": int(resolved.stat().st_size),
        "sha256": _file_sha256(resolved),
    }


def _records_digest(records: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(_canonical_json_bytes(list(records))).hexdigest()


def build_code_source_closure(
    *, repo_root: Optional[Path] = None
) -> dict[str, Any]:
    """Fingerprint every Python source that can participate in Stage-B training."""
    root = _repo_root(repo_root)
    paths = [root / "main.py", root / "engine.py"]
    for directory in ("models", "datasets", "groundingdino", "util"):
        paths.extend(_repo_python_files(directory, repo_root=root))
    records = sorted(
        (_file_record(path, repo_root=root) for path in set(paths)),
        key=lambda item: item["path"],
    )
    if not records:
        raise RuntimeError("dense-duty code source closure is empty")
    return {
        "schema": CODE_SOURCE_CLOSURE_SCHEMA,
        "file_count": len(records),
        "files": records,
        "sha256": _records_digest(records),
    }


def _config_module_path(module: str, *, repo_root: Path) -> Optional[Path]:
    if module != "config" and not module.startswith("config."):
        return None
    base = repo_root.joinpath(*module.split("."))
    candidates = (base.with_suffix(".py"), base / "__init__.py")
    matches = [candidate for candidate in candidates if candidate.exists()]
    if len(matches) > 1:
        raise RuntimeError(
            f"dense-duty config module is ambiguous: {module!r}"
        )
    if not matches:
        return None
    return _repo_relative_regular_file(matches[0], repo_root=repo_root)[0]


def _relative_config_module(
    current: Path, *, module: Optional[str], level: int, repo_root: Path
) -> str:
    _, portable = _repo_relative_regular_file(current, repo_root=repo_root)
    package_parts = Path(portable).with_suffix("").parts[:-1]
    if level <= 0 or level > len(package_parts):
        raise RuntimeError(
            f"dense-duty config has an invalid relative import: {portable}"
        )
    base = package_parts[: len(package_parts) - level + 1]
    suffix = tuple(str(module).split(".")) if module else ()
    return ".".join((*base, *suffix))


def _local_config_imports(path: Path, *, repo_root: Path) -> list[Path]:
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise RuntimeError(f"could not parse dense-duty config {path}: {exc}") from exc

    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base_module = _relative_config_module(
                    path,
                    module=node.module,
                    level=node.level,
                    repo_root=repo_root,
                )
            else:
                base_module = str(node.module or "")
            modules.add(base_module)
            # ``from config import child`` may resolve child as a submodule.
            for alias in node.names:
                if alias.name != "*":
                    modules.add(f"{base_module}.{alias.name}")

    imports: list[Path] = []
    for module in sorted(modules):
        imported = _config_module_path(module, repo_root=repo_root)
        if imported is not None:
            imports.append(imported)
        elif module == "config" or module.startswith("config."):
            parent = module.rpartition(".")[0]
            if not parent or _config_module_path(parent, repo_root=repo_root) is None:
                raise RuntimeError(
                    f"dense-duty config imports missing local module {module!r}"
                )
    return imports


def build_config_source_closure(
    config_path: Path, *, repo_root: Optional[Path] = None
) -> dict[str, Any]:
    root = _repo_root(repo_root)
    entry, entry_relative = _repo_relative_regular_file(
        config_path, repo_root=root
    )
    if entry.suffix != ".py":
        raise RuntimeError("dense-duty phase config must be a Python file")

    pending = [entry]
    visited: set[Path] = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        pending.extend(_local_config_imports(current, repo_root=root))
    records = sorted(
        (_file_record(path, repo_root=root) for path in visited),
        key=lambda item: item["path"],
    )
    digest_payload = {"entry": entry_relative, "files": records}
    return {
        "schema": CONFIG_SOURCE_CLOSURE_SCHEMA,
        "entry": entry_relative,
        "file_count": len(records),
        "files": records,
        "sha256": hashlib.sha256(
            _canonical_json_bytes(digest_payload)
        ).hexdigest(),
    }


def build_source_closure(
    config_path: Path, *, repo_root: Optional[Path] = None
) -> dict[str, Any]:
    root = _repo_root(repo_root)
    code = build_code_source_closure(repo_root=root)
    config = build_config_source_closure(config_path, repo_root=root)
    digest_payload = {
        "code_sha256": code["sha256"],
        "config_sha256": config["sha256"],
    }
    return {
        "schema": SOURCE_CLOSURE_SCHEMA,
        "code": code,
        "config": config,
        "sha256": hashlib.sha256(
            _canonical_json_bytes(digest_payload)
        ).hexdigest(),
    }


def _validate_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise RuntimeError(f"dense-duty {label} SHA256 is invalid")
    return value


def _validate_file_records(value: Any, *, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"dense-duty {label} file manifest is empty or invalid")
    records: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != {
            "path",
            "size_bytes",
            "sha256",
        }:
            raise RuntimeError(f"dense-duty {label} file record is invalid")
        path = item.get("path")
        if (
            not isinstance(path, str)
            or not path
            or Path(path).is_absolute()
            or ".." in Path(path).parts
            or Path(path).as_posix() != path
        ):
            raise RuntimeError(f"dense-duty {label} path is not repo-relative")
        try:
            path.encode("ascii")
        except UnicodeEncodeError as exc:
            raise RuntimeError(f"dense-duty {label} path is not ASCII") from exc
        size = item.get("size_bytes")
        if type(size) is not int or size < 0:
            raise RuntimeError(f"dense-duty {label} file size is invalid")
        records.append(
            {
                "path": path,
                "size_bytes": size,
                "sha256": _validate_sha256(
                    item.get("sha256"), label=f"{label} file"
                ),
            }
        )
    paths = [record["path"] for record in records]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise RuntimeError(
            f"dense-duty {label} manifest paths are unsorted or duplicated"
        )
    return records


def validate_code_source_closure(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "file_count",
        "files",
        "sha256",
    }:
        raise RuntimeError("dense-duty code source closure is invalid")
    closure = dict(value)
    if closure.get("schema") != CODE_SOURCE_CLOSURE_SCHEMA:
        raise RuntimeError("dense-duty code source closure schema is invalid")
    records = _validate_file_records(closure.get("files"), label="code source")
    if closure.get("file_count") != len(records):
        raise RuntimeError("dense-duty code source file count is invalid")
    if _validate_sha256(
        closure.get("sha256"), label="code source closure"
    ) != _records_digest(records):
        raise RuntimeError("dense-duty code source closure digest is invalid")
    closure["files"] = records
    return closure


def validate_config_source_closure(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "entry",
        "file_count",
        "files",
        "sha256",
    }:
        raise RuntimeError("dense-duty config source closure is invalid")
    closure = dict(value)
    if closure.get("schema") != CONFIG_SOURCE_CLOSURE_SCHEMA:
        raise RuntimeError("dense-duty config source closure schema is invalid")
    records = _validate_file_records(closure.get("files"), label="config source")
    entry = closure.get("entry")
    if not isinstance(entry, str) or entry not in {
        record["path"] for record in records
    }:
        raise RuntimeError("dense-duty config source entry is invalid")
    if closure.get("file_count") != len(records):
        raise RuntimeError("dense-duty config source file count is invalid")
    expected = hashlib.sha256(
        _canonical_json_bytes({"entry": entry, "files": records})
    ).hexdigest()
    if _validate_sha256(
        closure.get("sha256"), label="config source closure"
    ) != expected:
        raise RuntimeError("dense-duty config source closure digest is invalid")
    closure["files"] = records
    return closure


def validate_source_closure(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "code",
        "config",
        "sha256",
    }:
        raise RuntimeError("dense-duty phase source closure is invalid")
    closure = dict(value)
    if closure.get("schema") != SOURCE_CLOSURE_SCHEMA:
        raise RuntimeError("dense-duty phase source closure schema is invalid")
    code = validate_code_source_closure(closure.get("code"))
    config = validate_config_source_closure(closure.get("config"))
    expected = hashlib.sha256(
        _canonical_json_bytes(
            {
                "code_sha256": code["sha256"],
                "config_sha256": config["sha256"],
            }
        )
    ).hexdigest()
    if _validate_sha256(
        closure.get("sha256"), label="phase source closure"
    ) != expected:
        raise RuntimeError("dense-duty phase source closure digest is invalid")
    closure["code"] = code
    closure["config"] = config
    return closure


def validate_current_source_closure(
    saved: Any, current: Any, *, compare_config: bool
) -> dict[str, Any]:
    saved_closure = validate_source_closure(saved)
    current_closure = validate_source_closure(current)
    if saved_closure["code"]["sha256"] != current_closure["code"]["sha256"]:
        raise RuntimeError("dense-duty code source closure drifted")
    if compare_config and saved_closure["config"] != current_closure["config"]:
        raise RuntimeError("dense-duty phase config source closure drifted")
    if compare_config and saved_closure != current_closure:
        raise RuntimeError("dense-duty phase source closure drifted")
    return saved_closure


def validate_formal_invocation(
    args: Any, *, repo_root: Optional[Path] = None
) -> None:
    values = _argument_mapping(args)
    if str(values.get("stage_b_dense_duty_execution_scope", "")).strip().lower() != (
        "formal"
    ):
        return
    phase = _validate_phase(values.get("stage_b_dense_duty_phase", ""))
    if values.get("options") is not None:
        raise RuntimeError("formal dense-duty training forbids --options")
    root = _repo_root(repo_root)
    ownership = str(values.get("stage_b_v22_score_ownership", "")).strip()
    config_names = (
        _ADAPTER_FORMAL_CONFIDENCE_CONFIGS
        if phase == "confidence" and ownership == _ADAPTER_OWNERSHIP
        else (_FORMAL_PHASE_CONFIGS[phase],)
    )
    expected = tuple((root / config_name).resolve(strict=True) for config_name in config_names)
    observed_value = values.get("config_file")
    if not isinstance(observed_value, str) or not observed_value.strip():
        raise RuntimeError("formal dense-duty training lacks its exact phase config")
    observed = Path(observed_value).expanduser().resolve(strict=True)
    if observed not in expected:
        raise RuntimeError(
            "formal dense-duty training requires an allowlisted exact phase config: "
            f"expected={expected}, observed={observed}"
        )


def _tensor_group_fingerprint(
    state: Mapping[str, torch.Tensor], names: Sequence[str]
) -> dict[str, Any]:
    digest = hashlib.sha256()
    tensor_count = 0
    element_count = 0
    storage_bytes = 0
    nonfinite_count = 0
    for name in names:
        value = state.get(name)
        if not torch.is_tensor(value):
            raise RuntimeError(f"model state entry is not a tensor: {name}")
        tensor = value.detach().cpu().contiguous()
        header = _canonical_json_bytes(
            [name, str(tensor.dtype), list(tensor.shape)]
        )
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        if tensor.numel():
            raw = tensor.reshape(-1).view(torch.uint8).numpy()
            digest.update(memoryview(raw))
            if tensor.is_floating_point() or tensor.is_complex():
                nonfinite_count += int((~torch.isfinite(tensor)).sum().item())
        tensor_count += 1
        element_count += int(tensor.numel())
        storage_bytes += int(tensor.numel() * tensor.element_size())
    return {
        "sha256": digest.hexdigest(),
        "tensor_count": tensor_count,
        "element_count": element_count,
        "storage_bytes": storage_bytes,
        "nonfinite_count": nonfinite_count,
    }


def fingerprint_named_tensors(
    state: Mapping[str, torch.Tensor], names: Sequence[str]
) -> dict[str, Any]:
    """Return the canonical dense-duty fingerprint for an explicit tensor set."""
    return _tensor_group_fingerprint(state, sorted(str(name) for name in names))


def _validate_phase(phase: str) -> str:
    normalized = str(phase).strip().lower()
    if normalized not in {"rank", "confidence"}:
        raise RuntimeError(
            "dense-duty fingerprint phase must be exactly 'rank' or 'confidence'"
        )
    return normalized


def _argument_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, Mapping):
        return attributes
    raise RuntimeError("dense-duty arguments must be a mapping or namespace")


def _v53_fulltext_global_absolute_revision(values: Mapping[str, Any]) -> bool:
    return (
        str(values.get("stage_b_v22_score_ownership", "")).strip()
        == _ADAPTER_OWNERSHIP
        and str(
            values.get("stage_b_dense_duty_confidence_revision", "")
        ).strip()
        == _V53_CONFIDENCE_REVISION
    )


def _v53_fulltext_global_absolute_contract(values: Mapping[str, Any]) -> bool:
    """Fail closed unless a V53 revision declares its complete exact surface."""
    if not _v53_fulltext_global_absolute_revision(values):
        return False
    expected = {
        "stage_b_dense_duty_confidence_head_gradient_contract": (
            _V53_CONFIDENCE_HEAD_CONTRACT
        ),
        "stage_b_dense_duty_confidence_pool_feature_contract": (
            _V53_CONFIDENCE_POOL_CONTRACT
        ),
        "stage_b_dense_duty_confidence_gate_gradient_contract": (
            _V53_CONFIDENCE_GATE_CONTRACT
        ),
        "stage_b_v11_trainable_params_min": _V53_ACTIVE_ELEMENT_COUNT,
        "stage_b_v11_trainable_params_max": _V53_ACTIVE_ELEMENT_COUNT,
    }
    drift = {
        key: (values.get(key), expected_value)
        for key, expected_value in expected.items()
        if values.get(key) != expected_value
    }
    scope = str(
        values.get("stage_b_dense_duty_execution_scope", "")
    ).strip().lower()
    admission = str(
        values.get(
            "stage_b_dense_duty_confidence_probe_admission_contract", ""
        )
    ).strip()
    if scope == "formal" and admission != _V53_FORMAL_ADMISSION_CONTRACT:
        drift["stage_b_dense_duty_confidence_probe_admission_contract"] = (
            admission,
            _V53_FORMAL_ADMISSION_CONTRACT,
        )
    if drift:
        raise RuntimeError(
            "V53 full-text global-absolute confidence contract drifted: "
            f"{drift}"
        )
    return True


def _v54_fulltext_global_absolute_exact_residual_revision(
    values: Mapping[str, Any],
) -> bool:
    return (
        str(values.get("stage_b_v22_score_ownership", "")).strip()
        == _ADAPTER_OWNERSHIP
        and str(
            values.get("stage_b_dense_duty_confidence_revision", "")
        ).strip()
        == _V54_CONFIDENCE_REVISION
    )


def _v54_fulltext_global_absolute_exact_residual_contract(
    values: Mapping[str, Any],
) -> bool:
    """Fail closed unless V54 differs from V53 only at its trust reference."""
    if not _v54_fulltext_global_absolute_exact_residual_revision(values):
        return False
    expected = {
        "stage_b_dense_duty_confidence_head_gradient_contract": (
            _V53_CONFIDENCE_HEAD_CONTRACT
        ),
        "stage_b_dense_duty_confidence_pool_feature_contract": (
            _V54_CONFIDENCE_POOL_CONTRACT
        ),
        "stage_b_dense_duty_confidence_gate_gradient_contract": (
            _V53_CONFIDENCE_GATE_CONTRACT
        ),
        "stage_b_dense_duty_positive_trust_contract": (
            _V54_POSITIVE_TRUST_CONTRACT
        ),
        "stage_b_v11_trainable_params_min": _V53_ACTIVE_ELEMENT_COUNT,
        "stage_b_v11_trainable_params_max": _V53_ACTIVE_ELEMENT_COUNT,
    }
    drift = {
        key: (values.get(key), expected_value)
        for key, expected_value in expected.items()
        if values.get(key) != expected_value
    }
    scope = str(
        values.get("stage_b_dense_duty_execution_scope", "")
    ).strip().lower()
    admission = str(
        values.get(
            "stage_b_dense_duty_confidence_probe_admission_contract", ""
        )
    ).strip()
    if scope == "formal" and admission != _V54_FORMAL_ADMISSION_CONTRACT:
        drift["stage_b_dense_duty_confidence_probe_admission_contract"] = (
            admission,
            _V54_FORMAL_ADMISSION_CONTRACT,
        )
    if drift:
        raise RuntimeError(
            "V54 exact-residual confidence contract drifted: " f"{drift}"
        )
    return True


def _v55_fulltext_global_independent_absolute_revision(
    values: Mapping[str, Any],
) -> bool:
    return (
        str(values.get("stage_b_v22_score_ownership", "")).strip()
        == _ADAPTER_OWNERSHIP
        and str(
            values.get("stage_b_dense_duty_confidence_revision", "")
        ).strip()
        == _V55_CONFIDENCE_REVISION
    )


def _v55_fulltext_global_independent_absolute_contract(
    values: Mapping[str, Any],
) -> bool:
    """Bind the independent candidate-local and sample-global V55 routes."""
    if not _v55_fulltext_global_independent_absolute_revision(values):
        return False
    expected = {
        "stage_b_dense_duty_confidence_head_gradient_contract": (
            _V55_CONFIDENCE_HEAD_CONTRACT
        ),
        "stage_b_dense_duty_confidence_pool_feature_contract": (
            _V55_CONFIDENCE_POOL_CONTRACT
        ),
        "stage_b_dense_duty_confidence_gate_gradient_contract": (
            _V53_CONFIDENCE_GATE_CONTRACT
        ),
        "stage_b_dense_duty_positive_trust_contract": (
            _V55_POSITIVE_TRUST_CONTRACT
        ),
        "stage_b_v11_trainable_params_min": _V53_ACTIVE_ELEMENT_COUNT,
        "stage_b_v11_trainable_params_max": _V53_ACTIVE_ELEMENT_COUNT,
    }
    drift = {
        key: (values.get(key), expected_value)
        for key, expected_value in expected.items()
        if values.get(key) != expected_value
    }
    scope = str(
        values.get("stage_b_dense_duty_execution_scope", "")
    ).strip().lower()
    admission = str(
        values.get(
            "stage_b_dense_duty_confidence_probe_admission_contract", ""
        )
    ).strip()
    if scope == "formal" and admission != _V55_FORMAL_ADMISSION_CONTRACT:
        drift["stage_b_dense_duty_confidence_probe_admission_contract"] = (
            admission,
            _V55_FORMAL_ADMISSION_CONTRACT,
        )
    if drift:
        raise RuntimeError(
            "V55 independent global-absolute confidence contract drifted: "
            f"{drift}"
        )
    return True


def _v56_deployment_owned_global_revision(values: Mapping[str, Any]) -> bool:
    return (
        str(values.get("stage_b_v22_score_ownership", "")).strip()
        == _ADAPTER_OWNERSHIP
        and str(
            values.get("stage_b_dense_duty_confidence_revision", "")
        ).strip()
        == _V56_CONFIDENCE_REVISION
    )


def _v56_deployment_owned_global_contract(values: Mapping[str, Any]) -> bool:
    """Bind the frozen diagnostic head and deployment-owned V56 trunk."""
    if not _v56_deployment_owned_global_revision(values):
        return False
    expected = {
        "stage_b_dense_duty_confidence_head_gradient_contract": (
            _V56_CONFIDENCE_HEAD_CONTRACT
        ),
        "stage_b_dense_duty_confidence_pool_feature_contract": (
            _V56_CONFIDENCE_POOL_CONTRACT
        ),
        "stage_b_dense_duty_confidence_gate_gradient_contract": (
            _V53_CONFIDENCE_GATE_CONTRACT
        ),
        "stage_b_dense_duty_positive_trust_contract": (
            _V55_POSITIVE_TRUST_CONTRACT
        ),
        "stage_b_v14_local_absolute_weight": 0.0,
        "stage_b_v11_trainable_params_min": _V56_ACTIVE_ELEMENT_COUNT,
        "stage_b_v11_trainable_params_max": _V56_ACTIVE_ELEMENT_COUNT,
    }
    drift = {
        key: (values.get(key), expected_value)
        for key, expected_value in expected.items()
        if values.get(key) != expected_value
    }
    scope = str(
        values.get("stage_b_dense_duty_execution_scope", "")
    ).strip().lower()
    admission = str(
        values.get(
            "stage_b_dense_duty_confidence_probe_admission_contract", ""
        )
    ).strip()
    if scope == "formal" and admission != _V56_FORMAL_ADMISSION_CONTRACT:
        drift["stage_b_dense_duty_confidence_probe_admission_contract"] = (
            admission,
            _V56_FORMAL_ADMISSION_CONTRACT,
        )
    if drift:
        raise RuntimeError(
            "V56 deployment-owned global confidence contract drifted: "
            f"{drift}"
        )
    return True


def _v57_deployed_global_balanced_absolute_revision(
    values: Mapping[str, Any],
) -> bool:
    return (
        str(values.get("stage_b_v22_score_ownership", "")).strip()
        == _ADAPTER_OWNERSHIP
        and str(
            values.get("stage_b_dense_duty_confidence_revision", "")
        ).strip()
        == _V57_CONFIDENCE_REVISION
    )


def _v57_deployed_global_balanced_absolute_contract(
    values: Mapping[str, Any],
) -> bool:
    """Bind V57's balanced loss to the unchanged V56 deployed owner."""
    if not _v57_deployed_global_balanced_absolute_revision(values):
        return False
    expected = {
        "stage_b_dense_duty_confidence_head_gradient_contract": (
            _V56_CONFIDENCE_HEAD_CONTRACT
        ),
        "stage_b_dense_duty_confidence_pool_feature_contract": (
            _V56_CONFIDENCE_POOL_CONTRACT
        ),
        "stage_b_dense_duty_confidence_gate_gradient_contract": (
            _V53_CONFIDENCE_GATE_CONTRACT
        ),
        "stage_b_dense_duty_positive_trust_contract": (
            _V55_POSITIVE_TRUST_CONTRACT
        ),
        "stage_b_v14_local_absolute_weight": 0.0,
        "stage_b_dense_duty_deployed_global_absolute_weight": 1.0,
        "stage_b_dense_duty_deployed_global_absolute_gamma": 1.0,
        "stage_b_v11_trainable_params_min": _V56_ACTIVE_ELEMENT_COUNT,
        "stage_b_v11_trainable_params_max": _V56_ACTIVE_ELEMENT_COUNT,
    }
    drift = {
        key: (values.get(key), expected_value)
        for key, expected_value in expected.items()
        if values.get(key) != expected_value
    }
    scope = str(
        values.get("stage_b_dense_duty_execution_scope", "")
    ).strip().lower()
    admission = str(
        values.get(
            "stage_b_dense_duty_confidence_probe_admission_contract", ""
        )
    ).strip()
    if scope == "formal" and admission != _V57_FORMAL_ADMISSION_CONTRACT:
        drift["stage_b_dense_duty_confidence_probe_admission_contract"] = (
            admission,
            _V57_FORMAL_ADMISSION_CONTRACT,
        )
    if drift:
        raise RuntimeError(
            "V57 deployed-global balanced-absolute contract drifted: "
            f"{drift}"
        )
    return True


def _v58_deployment_owned_stable_fpr95_active_set_revision(
    values: Mapping[str, Any],
) -> bool:
    return (
        str(values.get("stage_b_v22_score_ownership", "")).strip()
        == _ADAPTER_OWNERSHIP
        and str(
            values.get("stage_b_dense_duty_confidence_revision", "")
        ).strip()
        == _V58_CONFIDENCE_REVISION
    )


def _v58_deployment_owned_stable_fpr95_active_set_contract(
    values: Mapping[str, Any],
) -> bool:
    """Bind stable active-set reduction to the unchanged V56 owner."""
    if not _v58_deployment_owned_stable_fpr95_active_set_revision(values):
        return False
    expected = {
        "stage_b_dense_duty_confidence_head_gradient_contract": (
            _V56_CONFIDENCE_HEAD_CONTRACT
        ),
        "stage_b_dense_duty_confidence_pool_feature_contract": (
            _V56_CONFIDENCE_POOL_CONTRACT
        ),
        "stage_b_dense_duty_confidence_gate_gradient_contract": (
            _V53_CONFIDENCE_GATE_CONTRACT
        ),
        "stage_b_dense_duty_positive_trust_contract": (
            _V55_POSITIVE_TRUST_CONTRACT
        ),
        "stage_b_v14_local_absolute_weight": 0.0,
        "stage_b_dense_duty_deployed_global_absolute_weight": 0.0,
        "stage_b_v15_tail_queue_negative_reduction_contract": (
            "exact_fpr95_active_set_all_count_mean_v2"
        ),
        "stage_b_v11_trainable_params_min": _V56_ACTIVE_ELEMENT_COUNT,
        "stage_b_v11_trainable_params_max": _V56_ACTIVE_ELEMENT_COUNT,
    }
    drift = {
        key: (values.get(key), expected_value)
        for key, expected_value in expected.items()
        if values.get(key) != expected_value
    }
    scope = str(
        values.get("stage_b_dense_duty_execution_scope", "")
    ).strip().lower()
    admission = str(
        values.get(
            "stage_b_dense_duty_confidence_probe_admission_contract", ""
        )
    ).strip()
    if scope == "formal" and admission != _V58_FORMAL_ADMISSION_CONTRACT:
        drift["stage_b_dense_duty_confidence_probe_admission_contract"] = (
            admission,
            _V58_FORMAL_ADMISSION_CONTRACT,
        )
    if drift:
        raise RuntimeError(
            "V58 deployment-owned stable FPR95 active-set contract drifted: "
            f"{drift}"
        )
    return True


def _v59_deployment_owned_query_global_revision(
    values: Mapping[str, Any],
) -> bool:
    return (
        str(values.get("stage_b_v22_score_ownership", "")).strip()
        == _ADAPTER_OWNERSHIP
        and str(
            values.get("stage_b_dense_duty_confidence_revision", "")
        ).strip()
        == _V59_CONFIDENCE_REVISION
    )


def _v59_deployment_owned_query_global_contract(
    values: Mapping[str, Any],
) -> bool:
    """Bind query-structured evidence exclusively to the deployed V59 owner."""
    if not _v59_deployment_owned_query_global_revision(values):
        return False
    expected = {
        "stage_b_dense_duty_confidence_head_gradient_contract": (
            _V59_CONFIDENCE_HEAD_CONTRACT
        ),
        "stage_b_dense_duty_confidence_pool_feature_contract": (
            _V59_CONFIDENCE_POOL_CONTRACT
        ),
        "stage_b_dense_duty_confidence_gate_gradient_contract": (
            _V53_CONFIDENCE_GATE_CONTRACT
        ),
        "stage_b_dense_duty_positive_trust_contract": (
            _V59_POSITIVE_TRUST_CONTRACT
        ),
        "stage_b_v14_local_absolute_weight": 0.0,
        "stage_b_dense_duty_deployed_global_absolute_weight": 0.0,
        "stage_b_v15_tail_queue_negative_reduction_contract": "all_mean_v1",
        "stage_b_v11_trainable_params_min": _V53_ACTIVE_ELEMENT_COUNT,
        "stage_b_v11_trainable_params_max": _V53_ACTIVE_ELEMENT_COUNT,
    }
    drift = {
        key: (values.get(key), expected_value)
        for key, expected_value in expected.items()
        if values.get(key) != expected_value
    }
    scope = str(
        values.get("stage_b_dense_duty_execution_scope", "")
    ).strip().lower()
    admission = str(
        values.get(
            "stage_b_dense_duty_confidence_probe_admission_contract", ""
        )
    ).strip()
    if scope == "formal" and admission != _V59_FORMAL_ADMISSION_CONTRACT:
        drift["stage_b_dense_duty_confidence_probe_admission_contract"] = (
            admission,
            _V59_FORMAL_ADMISSION_CONTRACT,
        )
    if drift:
        raise RuntimeError(
            "V59 deployment-owned query-global confidence contract drifted: "
            f"{drift}"
        )
    return True


def _v60_deployment_owned_query_veto_revision(
    values: Mapping[str, Any],
) -> bool:
    return (
        str(values.get("stage_b_v22_score_ownership", "")).strip()
        == _ADAPTER_OWNERSHIP
        and str(
            values.get("stage_b_dense_duty_confidence_revision", "")
        ).strip()
        == _V60_CONFIDENCE_REVISION
    )


def _v60_deployment_owned_query_veto_contract(
    values: Mapping[str, Any],
) -> bool:
    """Bind the bounded query veto exclusively to the deployed V60 owner."""
    if not _v60_deployment_owned_query_veto_revision(values):
        return False
    full_decoder_verifier = bool(
        values.get("stage_b_dense_duty_confidence_full_decoder_verifier", False)
    )
    active_element_count = (
        _V61_FULL_DECODER_ACTIVE_ELEMENT_COUNT
        if full_decoder_verifier
        else _V53_ACTIVE_ELEMENT_COUNT
    )
    expected = {
        "stage_b_dense_duty_confidence_head_gradient_contract": (
            _V60_CONFIDENCE_HEAD_CONTRACT
        ),
        "stage_b_dense_duty_confidence_pool_feature_contract": (
            _V60_CONFIDENCE_POOL_CONTRACT
        ),
        "stage_b_dense_duty_confidence_gate_gradient_contract": (
            _V53_CONFIDENCE_GATE_CONTRACT
        ),
        "stage_b_dense_duty_positive_trust_contract": (
            _V60_POSITIVE_TRUST_CONTRACT
        ),
        "stage_b_v14_local_absolute_weight": 0.0,
        "stage_b_dense_duty_deployed_global_absolute_weight": 0.0,
        "stage_b_v15_tail_queue_negative_reduction_contract": "all_mean_v1",
        "stage_b_v11_trainable_params_min": active_element_count,
        "stage_b_v11_trainable_params_max": active_element_count,
    }
    if full_decoder_verifier:
        expected.update(
            {
                "stage_b_dense_duty_confidence_capacity_contract": (
                    "rank_cloned_full_decoder_6layer_256d_v1"
                ),
                "stage_b_dense_duty_confidence_variant": (
                    "full_decoder_token_entailment_nonnegative_veto_"
                    "capacity_upper_bound_v61"
                ),
            }
        )
    drift = {
        key: (values.get(key), expected_value)
        for key, expected_value in expected.items()
        if values.get(key) != expected_value
    }
    scope = str(
        values.get("stage_b_dense_duty_execution_scope", "")
    ).strip().lower()
    admission = str(
        values.get(
            "stage_b_dense_duty_confidence_probe_admission_contract", ""
        )
    ).strip()
    if scope == "formal" and admission != _V60_FORMAL_ADMISSION_CONTRACT:
        drift["stage_b_dense_duty_confidence_probe_admission_contract"] = (
            admission,
            _V60_FORMAL_ADMISSION_CONTRACT,
        )
    if drift:
        raise RuntimeError(
            "V60 deployment-owned query-veto confidence contract drifted: "
            f"{drift}"
        )
    return True


def build_training_contract(args: Any) -> dict[str, Any]:
    values = _argument_mapping(args)
    adapter_contract = (
        str(values.get("stage_b_v22_score_ownership", "")).strip()
        == _ADAPTER_OWNERSHIP
    )
    packed_contract = adapter_contract and int(
        values.get("stage_b_dense_duty_forward_pack_factor", 1) or 1
    ) > 1
    word_veto_contract = adapter_contract and str(
        values.get("stage_b_dense_duty_confidence_phrase_aggregation", "")
    ).strip().lower() in {
        "trace_activated_word_veto_product_v1",
        "trace_activated_word_veto_penalty_v2",
        "trace_activated_word_veto_absolute_cap_v4",
        "trace_activated_word_veto_gated_pool_absolute_cap_v5",
    }
    absolute_cap_contract = word_veto_contract and str(
        values.get("stage_b_dense_duty_confidence_phrase_aggregation", "")
    ).strip().lower() in {
        "trace_activated_word_veto_absolute_cap_v4",
        "trace_activated_word_veto_gated_pool_absolute_cap_v5",
    }
    raw_veto_gate_contract = word_veto_contract and float(
        values.get("stage_b_dense_duty_raw_veto_gate_weight", 0.0) or 0.0
    ) > 0.0
    carrier_balanced_contract = word_veto_contract and str(
        values.get("stage_b_dense_duty_confidence_revision", "")
    ).strip() in {
        "word_veto_gated_pool_carrier_balanced_v7",
        "word_veto_gated_pool_carrier_quarter_v8",
        "word_veto_gated_pool_carrier_pair_v9",
        "word_veto_gated_pool_dual_carrier_pair_v10",
        "word_veto_gated_pool_rank_evidence_v11",
        "word_veto_gated_pool_rank_affine_v12",
        "word_veto_gated_pool_gate_margin_v13",
        "word_veto_gated_pool_carrier_slope_v14",
        "word_veto_gated_pool_carrier_affine_v15",
        "word_veto_gated_pool_tail_ste_v16",
        "word_veto_gated_pool_tail_carrier_v17",
        "word_veto_gated_pool_tail_paired_v18",
        "word_veto_gated_pool_tail_paired_rank_channel_v19",
        "word_veto_gated_pool_tail_paired_signed_rank_pool_v20",
    }
    carrier_pair_contract = word_veto_contract and str(
        values.get("stage_b_dense_duty_confidence_revision", "")
    ).strip() in {
        "word_veto_gated_pool_carrier_pair_v9",
        "word_veto_gated_pool_dual_carrier_pair_v10",
        "word_veto_gated_pool_rank_evidence_v11",
        "word_veto_gated_pool_rank_affine_v12",
        "word_veto_gated_pool_gate_margin_v13",
        "word_veto_gated_pool_carrier_slope_v14",
        "word_veto_gated_pool_carrier_affine_v15",
        "word_veto_gated_pool_tail_ste_v16",
        "word_veto_gated_pool_tail_carrier_v17",
        "word_veto_gated_pool_tail_paired_v18",
        "word_veto_gated_pool_tail_paired_rank_channel_v19",
        "word_veto_gated_pool_tail_paired_signed_rank_pool_v20",
    }
    dual_carrier_contract = word_veto_contract and str(
        values.get("stage_b_dense_duty_confidence_revision", "")
    ).strip() == "word_veto_gated_pool_dual_carrier_pair_v10"
    rank_evidence_contract = word_veto_contract and str(
        values.get("stage_b_dense_duty_confidence_revision", "")
    ).strip() == "word_veto_gated_pool_rank_evidence_v11"
    rank_affine_contract = word_veto_contract and str(
        values.get("stage_b_dense_duty_confidence_revision", "")
    ).strip() == "word_veto_gated_pool_rank_affine_v12"
    gate_margin_contract = word_veto_contract and str(
        values.get("stage_b_dense_duty_confidence_revision", "")
    ).strip() == "word_veto_gated_pool_gate_margin_v13"
    carrier_slope_contract = word_veto_contract and str(
        values.get("stage_b_dense_duty_confidence_revision", "")
    ).strip() == "word_veto_gated_pool_carrier_slope_v14"
    carrier_affine_contract = word_veto_contract and str(
        values.get("stage_b_dense_duty_confidence_revision", "")
    ).strip() in {
        "word_veto_gated_pool_carrier_affine_v15",
        "word_veto_gated_pool_tail_ste_v16",
        "word_veto_gated_pool_tail_carrier_v17",
        "word_veto_gated_pool_tail_paired_v18",
        "word_veto_gated_pool_tail_paired_rank_channel_v19",
        "word_veto_gated_pool_tail_paired_signed_rank_pool_v20",
    }
    tail_ste_contract = word_veto_contract and str(
        values.get("stage_b_dense_duty_confidence_revision", "")
    ).strip() == "word_veto_gated_pool_tail_ste_v16"
    tail_carrier_contract = word_veto_contract and str(
        values.get("stage_b_dense_duty_confidence_revision", "")
    ).strip() == "word_veto_gated_pool_tail_carrier_v17"
    tail_paired_contract = word_veto_contract and str(
        values.get("stage_b_dense_duty_confidence_revision", "")
    ).strip() in {
        "word_veto_gated_pool_tail_paired_v18",
        "word_veto_gated_pool_tail_paired_rank_channel_v19",
        "word_veto_gated_pool_tail_paired_signed_rank_pool_v20",
    }
    positive_tail_gradient_contract = adapter_contract and str(
        values.get(
            "stage_b_v15_tail_queue_positive_gradient_contract",
            "mean_translation_v1",
        )
    ).strip() in {
        "exact_batch_lower_tail_st_v2",
        "mean_plus_exact_lower_tail_st_v3",
        "mean_plus_quarter_exact_lower_tail_st_v4",
        "bounded_mean_plus_sixteenth_exact_lower_tail_st_v5",
        "elementwise_bounded_mean_plus_sixteenth_exact_lower_tail_st_v6",
    }
    token_edit_query_scope = str(
        values.get("stage_b_v21_token_edit_query_scope", "target_iou_v1")
    ).strip().lower()
    token_edit_carrier_contract = adapter_contract and token_edit_query_scope in {
        "target_iou_union_detached_final_confidence_base_argmax_v2",
        "target_iou_union_detached_role_complete_confidence_base_argmax_v3",
    }
    role_complete_carrier_contract = adapter_contract and token_edit_query_scope == (
        "target_iou_union_detached_role_complete_confidence_base_argmax_v3"
    )
    carrier_pair_gradient_contract = adapter_contract and str(
        values.get(
            "stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract",
            "bidirectional_v1",
        )
    ).strip().lower() != "bidirectional_v1"
    confidence_revision = str(
        values.get("stage_b_dense_duty_confidence_revision", "")
    ).strip()
    tail_aligned_split_contract = adapter_contract and confidence_revision == (
        "word_veto_candidate_split_tail_aligned_v45"
    )
    positive_tail_split_contract = adapter_contract and confidence_revision == (
        "word_veto_candidate_split_positive_tail_v46"
    )
    boundary_routing_split_contract = adapter_contract and confidence_revision == (
        "word_veto_candidate_split_boundary_routing_v47"
    )
    fpr_active_set_split_contract = adapter_contract and confidence_revision == (
        "word_veto_candidate_split_fpr_active_set_v48"
    )
    global_trust_veto_split_contract = adapter_contract and confidence_revision == (
        "word_veto_candidate_split_global_trust_veto_v49"
    )
    strong_boundary_routing_split_contract = (
        adapter_contract
        and confidence_revision
        == "word_veto_candidate_split_strong_boundary_routing_v50"
    )
    independent_deployed_router_split_contract = (
        adapter_contract
        and confidence_revision
        == "word_veto_candidate_split_independent_deployed_router_v51"
    )
    candidate_sample_calibrator_split_contract = (
        adapter_contract
        and confidence_revision
        == "word_veto_candidate_sample_calibrator_split_v52"
    )
    v53_fulltext_global_absolute_contract = (
        _v53_fulltext_global_absolute_contract(values)
    )
    v54_fulltext_global_absolute_exact_residual_contract = (
        _v54_fulltext_global_absolute_exact_residual_contract(values)
    )
    v55_fulltext_global_independent_absolute_contract = (
        _v55_fulltext_global_independent_absolute_contract(values)
    )
    v56_deployment_owned_global_contract = (
        _v56_deployment_owned_global_contract(values)
    )
    v57_deployed_global_balanced_absolute_contract = (
        _v57_deployed_global_balanced_absolute_contract(values)
    )
    v58_deployment_owned_stable_fpr95_active_set_contract = (
        _v58_deployment_owned_stable_fpr95_active_set_contract(values)
    )
    v59_deployment_owned_query_global_contract = (
        _v59_deployment_owned_query_global_contract(values)
    )
    v60_deployment_owned_query_veto_contract = (
        _v60_deployment_owned_query_veto_contract(values)
    )
    fulltext_global_absolute_contract = (
        v53_fulltext_global_absolute_contract
        or v54_fulltext_global_absolute_exact_residual_contract
        or v55_fulltext_global_independent_absolute_contract
        or v56_deployment_owned_global_contract
        or v57_deployed_global_balanced_absolute_contract
        or v58_deployment_owned_stable_fpr95_active_set_contract
        or v59_deployment_owned_query_global_contract
        or v60_deployment_owned_query_veto_contract
    )
    split_reduction_contract = (
        tail_aligned_split_contract
        or positive_tail_split_contract
        or boundary_routing_split_contract
        or fpr_active_set_split_contract
        or global_trust_veto_split_contract
        or strong_boundary_routing_split_contract
        or independent_deployed_router_split_contract
        or candidate_sample_calibrator_split_contract
        or fulltext_global_absolute_contract
    )
    deployed_veto_routing_contract = adapter_contract and confidence_revision in {
        "word_veto_candidate_asymmetric_deployed_routing_v43",
        "word_veto_candidate_split_tail_aligned_v45",
        "word_veto_candidate_split_positive_tail_v46",
        "word_veto_candidate_split_boundary_routing_v47",
        "word_veto_candidate_split_fpr_active_set_v48",
        "word_veto_candidate_split_global_trust_veto_v49",
        "word_veto_candidate_split_strong_boundary_routing_v50",
        "word_veto_candidate_split_independent_deployed_router_v51",
    }
    split_confidence_head_contract = adapter_contract and str(
        values.get(
            "stage_b_dense_duty_confidence_head_gradient_contract",
            "shared_token_veto_global_absolute_v1",
        )
    ).strip().lower() in {
        "split_token_veto_global_absolute_v2",
        "split_token_veto_global_absolute_joint_clip_v3",
        "split_token_veto_global_trust_veto_v4",
        "split_token_veto_deployed_router_global_absolute_v5",
        "split_token_veto_candidate_absolute_sample_calibrator_v6",
        _V53_CONFIDENCE_HEAD_CONTRACT,
        _V55_CONFIDENCE_HEAD_CONTRACT,
        _V56_CONFIDENCE_HEAD_CONTRACT,
    }
    probe_admission_contract = adapter_contract and str(
        values.get(
            "stage_b_dense_duty_confidence_probe_admission_contract", ""
        )
    ).strip() in {
        "u300_word_veto_strict1607_v1",
        "u300_word_veto_gate_strict1607_v3",
        "u300_word_veto_absolute_cap_strict1607_v4",
        "u300_word_veto_gated_pool_absolute_cap_strict1607_v5",
        "u300_word_veto_gated_pool_calibrated_strict1607_v6",
        "u300_word_veto_gated_pool_carrier_balanced_strict1607_v7",
        "u300_word_veto_gated_pool_carrier_quarter_strict1607_v8",
        "u300_word_veto_gated_pool_carrier_pair_strict1607_v9",
        "u300_word_veto_gated_pool_dual_carrier_pair_strict1607_v10",
        "u300_word_veto_gated_pool_rank_evidence_strict1607_v11",
        "u300_word_veto_gated_pool_rank_affine_strict1607_v12",
        "u300_word_veto_gated_pool_gate_margin_strict1607_v13",
        "u300_word_veto_gated_pool_carrier_slope_strict1607_v14",
        "u300_word_veto_gated_pool_carrier_affine_strict1607_v15",
        "u300_word_veto_gated_pool_tail_ste_strict1607_v16",
        "u300_word_veto_gated_pool_tail_carrier_strict1607_v17",
        "u300_word_veto_gated_pool_tail_paired_strict1607_v18",
        "u300_word_veto_gated_pool_tail_paired_rank_channel_strict1607_v19",
        "u300_word_veto_gated_pool_tail_paired_signed_rank_pool_strict1607_v20",
        "u400_word_veto_candidate_q05_confidence_strict1607_v34",
        "u400_word_veto_candidate_tail_balanced_confidence_strict1607_v35",
        "u400_word_veto_candidate_tail_quarter_confidence_strict1607_v36",
        "u400_word_veto_candidate_tail_bounded_confidence_strict1607_v37",
        "u400_word_veto_candidate_tail_elementwise_confidence_strict1607_v38",
        "u400_word_veto_candidate_gate_zero_offset_confidence_strict1607_v39",
        "u400_word_veto_candidate_hardest_edit_confidence_strict1607_v40",
        "u400_word_veto_candidate_role_complete_carrier_confidence_strict1607_v41",
        "u400_word_veto_candidate_tn_only_carrier_pair_confidence_strict1607_v42",
        "u400_word_veto_candidate_deployed_routing_confidence_strict1607_v43",
        "u400_word_veto_candidate_split_tail_aligned_confidence_strict1607_v45",
        "u400_word_veto_candidate_split_positive_tail_confidence_strict1607_v46",
        "u400_word_veto_candidate_split_boundary_routing_confidence_strict1607_v47",
        "u400_word_veto_candidate_split_fpr_active_set_confidence_strict1607_v48",
        "u400_word_veto_candidate_split_global_trust_veto_confidence_strict1607_v49",
        "u400_word_veto_candidate_split_strong_boundary_routing_confidence_strict1607_v50",
        "u400_word_veto_candidate_split_independent_deployed_router_confidence_strict1607_v51",
        "u400_word_veto_candidate_sample_calibrator_confidence_strict1607_v52",
        _V53_FORMAL_ADMISSION_CONTRACT,
        _V54_FORMAL_ADMISSION_CONTRACT,
        _V55_FORMAL_ADMISSION_CONTRACT,
        _V56_FORMAL_ADMISSION_CONTRACT,
        _V57_FORMAL_ADMISSION_CONTRACT,
        _V58_FORMAL_ADMISSION_CONTRACT,
    }
    contract_keys = (
        _RESUME_CONTRACT_KEYS
        + (_ADAPTER_RESUME_CONTRACT_KEYS if adapter_contract else ())
        + (_WORD_VETO_RESUME_CONTRACT_KEYS if word_veto_contract else ())
        + (
            _RAW_VETO_GATE_RESUME_CONTRACT_KEYS
            if raw_veto_gate_contract
            else ()
        )
        + (
            _ABSOLUTE_CAP_RESUME_CONTRACT_KEYS
            if absolute_cap_contract
            else ()
        )
        + (
            _CARRIER_BALANCED_RESUME_CONTRACT_KEYS
            if carrier_balanced_contract
            else ()
        )
        + (
            _DEPLOYED_VETO_ROUTING_RESUME_CONTRACT_KEYS
            if (
                deployed_veto_routing_contract
                or candidate_sample_calibrator_split_contract
                or fulltext_global_absolute_contract
            )
            else ()
        )
        + (
            _SPLIT_CONFIDENCE_HEAD_RESUME_CONTRACT_KEYS
            if split_confidence_head_contract
            else ()
        )
        + (
            _V53_FULLTEXT_GLOBAL_ABSOLUTE_RESUME_CONTRACT_KEYS
            if fulltext_global_absolute_contract
            else ()
        )
        + (
            _TAIL_ALIGNED_SPLIT_RESUME_CONTRACT_KEYS
            if split_reduction_contract
            else ()
        )
        + (
            _FPR_ACTIVE_SET_RESUME_CONTRACT_KEYS
            if (
                fpr_active_set_split_contract
                or v58_deployment_owned_stable_fpr95_active_set_contract
            )
            else ()
        )
        + (
            _GLOBAL_TRUST_VETO_RESUME_CONTRACT_KEYS
            if global_trust_veto_split_contract
            else ()
        )
        + (
            _STRONG_BOUNDARY_ROUTING_RESUME_CONTRACT_KEYS
            if (
                strong_boundary_routing_split_contract
                or independent_deployed_router_split_contract
                or candidate_sample_calibrator_split_contract
                or fulltext_global_absolute_contract
            )
            else ()
        )
        + (
            _CARRIER_PAIR_RESUME_CONTRACT_KEYS
            if carrier_pair_contract
            else ()
        )
        + (
            _DUAL_CARRIER_RESUME_CONTRACT_KEYS
            if dual_carrier_contract
            else ()
        )
        + (
            _RANK_EVIDENCE_RESUME_CONTRACT_KEYS
            if rank_evidence_contract
            or rank_affine_contract
            or gate_margin_contract
            or carrier_slope_contract
            or carrier_affine_contract
            else ()
        )
        + (
            _GATE_MARGIN_RESUME_CONTRACT_KEYS
            if gate_margin_contract
            or carrier_slope_contract
            or carrier_affine_contract
            else ()
        )
        + (
            _GATE_GRADIENT_RESUME_CONTRACT_KEYS
            if tail_ste_contract or tail_carrier_contract or tail_paired_contract
            else ()
        )
        + (
            _TAIL_CARRIER_RESUME_CONTRACT_KEYS
            if tail_carrier_contract or tail_paired_contract
            else ()
        )
        + (
            _POSITIVE_TAIL_GRADIENT_RESUME_CONTRACT_KEYS
            if positive_tail_gradient_contract
            else ()
        )
        + (
            _TOKEN_EDIT_CARRIER_RESUME_CONTRACT_KEYS
            if token_edit_carrier_contract
            else ()
        )
        + (
            _CARRIER_BALANCED_RESUME_CONTRACT_KEYS
            + _CARRIER_PAIR_RESUME_CONTRACT_KEYS
            + _TAIL_CARRIER_RESUME_CONTRACT_KEYS
            + _CARRIER_PAIR_GRADIENT_RESUME_CONTRACT_KEYS
            if carrier_pair_gradient_contract
            else ()
        )
        + (
            _V57_DEPLOYED_GLOBAL_ABSOLUTE_RESUME_CONTRACT_KEYS
            if v57_deployed_global_balanced_absolute_contract
            else ()
        )
        + (
            _PROBE_ADMISSION_RESUME_CONTRACT_KEYS
            if probe_admission_contract
            else ()
        )
        + (_PACKED_FORWARD_CONTRACT_KEYS if packed_contract else ())
    )
    missing = [key for key in contract_keys if key not in values]
    if missing:
        raise RuntimeError(
            f"dense-duty training contract lacks required keys: {missing}"
        )
    source_closure = validate_source_closure(values[SOURCE_CLOSURE_ARG])
    contract_values = {key: values[key] for key in contract_keys}
    contract_values[SOURCE_CLOSURE_ARG] = source_closure
    if v60_deployment_owned_query_veto_contract:
        schema = "pivot.stageb.dense_duty_training_contract/v42"
    elif v59_deployment_owned_query_global_contract:
        schema = "pivot.stageb.dense_duty_training_contract/v41"
    elif v58_deployment_owned_stable_fpr95_active_set_contract:
        schema = "pivot.stageb.dense_duty_training_contract/v40"
    elif v57_deployed_global_balanced_absolute_contract:
        schema = "pivot.stageb.dense_duty_training_contract/v39"
    elif v56_deployment_owned_global_contract:
        schema = "pivot.stageb.dense_duty_training_contract/v38"
    elif v55_fulltext_global_independent_absolute_contract:
        schema = "pivot.stageb.dense_duty_training_contract/v37"
    elif v54_fulltext_global_absolute_exact_residual_contract:
        schema = "pivot.stageb.dense_duty_training_contract/v36"
    elif v53_fulltext_global_absolute_contract:
        schema = "pivot.stageb.dense_duty_training_contract/v35"
    elif candidate_sample_calibrator_split_contract:
        schema = "pivot.stageb.dense_duty_training_contract/v34"
    elif independent_deployed_router_split_contract:
        schema = "pivot.stageb.dense_duty_training_contract/v33"
    elif strong_boundary_routing_split_contract:
        schema = "pivot.stageb.dense_duty_training_contract/v32"
    elif global_trust_veto_split_contract:
        schema = "pivot.stageb.dense_duty_training_contract/v31"
    elif fpr_active_set_split_contract:
        schema = "pivot.stageb.dense_duty_training_contract/v30"
    elif boundary_routing_split_contract:
        schema = "pivot.stageb.dense_duty_training_contract/v29"
    elif positive_tail_split_contract:
        schema = "pivot.stageb.dense_duty_training_contract/v28"
    elif tail_aligned_split_contract:
        schema = "pivot.stageb.dense_duty_training_contract/v27"
    elif split_confidence_head_contract:
        schema = "pivot.stageb.dense_duty_training_contract/v26"
    elif deployed_veto_routing_contract:
        schema = "pivot.stageb.dense_duty_training_contract/v25"
    elif carrier_pair_gradient_contract:
        schema = "pivot.stageb.dense_duty_training_contract/v24"
    elif role_complete_carrier_contract:
        schema = "pivot.stageb.dense_duty_training_contract/v23"
    elif token_edit_carrier_contract:
        schema = "pivot.stageb.dense_duty_training_contract/v22"
    elif positive_tail_gradient_contract:
        schema = "pivot.stageb.dense_duty_training_contract/v21"
    elif tail_paired_contract:
        schema = "pivot.stageb.dense_duty_training_contract/v20"
    elif tail_carrier_contract:
        schema = "pivot.stageb.dense_duty_training_contract/v19"
    elif tail_ste_contract:
        schema = "pivot.stageb.dense_duty_training_contract/v18"
    elif carrier_affine_contract:
        schema = "pivot.stageb.dense_duty_training_contract/v17"
    elif carrier_slope_contract:
        schema = "pivot.stageb.dense_duty_training_contract/v16"
    elif gate_margin_contract:
        schema = "pivot.stageb.dense_duty_training_contract/v15"
    elif rank_affine_contract:
        schema = "pivot.stageb.dense_duty_training_contract/v14"
    elif rank_evidence_contract:
        schema = "pivot.stageb.dense_duty_training_contract/v13"
    elif dual_carrier_contract:
        schema = "pivot.stageb.dense_duty_training_contract/v12"
    elif carrier_pair_contract:
        schema = "pivot.stageb.dense_duty_training_contract/v11"
    elif carrier_balanced_contract:
        schema = "pivot.stageb.dense_duty_training_contract/v10"
    elif absolute_cap_contract:
        schema = "pivot.stageb.dense_duty_training_contract/v9"
    elif raw_veto_gate_contract:
        schema = "pivot.stageb.dense_duty_training_contract/v8"
    elif probe_admission_contract:
        schema = "pivot.stageb.dense_duty_training_contract/v7"
    elif word_veto_contract:
        schema = "pivot.stageb.dense_duty_training_contract/v6"
    elif packed_contract:
        schema = "pivot.stageb.dense_duty_training_contract/v5"
    elif adapter_contract:
        schema = "pivot.stageb.dense_duty_training_contract/v4"
    else:
        schema = "pivot.stageb.dense_duty_training_contract/v3"
    return {
        "schema": schema,
        "sha256": hashlib.sha256(
            _canonical_json_bytes(contract_values)
        ).hexdigest(),
        "values": contract_values,
    }


def validate_resume_training_contract(
    current_args: Any, saved_args: Any
) -> dict[str, Any]:
    current = build_training_contract(current_args)
    saved = build_training_contract(saved_args)
    if current != saved:
        current_values = current["values"]
        saved_values = saved["values"]
        contract_keys = tuple(current_values)
        drift = {
            key: (saved_values[key], current_values[key])
            for key in contract_keys
            if key not in saved_values or saved_values[key] != current_values[key]
        }
        raise RuntimeError(
            f"dense-duty resume training contract drifted: {drift}"
        )
    return current


def _validate_active_names(
    state: Mapping[str, torch.Tensor], active_parameter_names: Sequence[str], phase: str
) -> list[str]:
    active_names = sorted(str(name) for name in active_parameter_names)
    if not active_names or len(active_names) != len(set(active_names)):
        raise RuntimeError("dense-duty active parameter names are empty or duplicated")
    missing = sorted(set(active_names).difference(state))
    if missing:
        raise RuntimeError(
            f"dense-duty active parameters are absent from model state: {missing[:20]}"
        )
    if phase == "rank":
        allowed_prefixes = ("stage_b_fixed_text_scorer.rank_tower.",)
    else:
        contract_value = state.get(
            "stage_b_fixed_text_scorer._dense_duty_contract_version"
        )
        contract_version = (
            int(contract_value.item())
            if torch.is_tensor(contract_value) and contract_value.numel() == 1
            else 0
        )
        allowed_prefixes = (
            (
                "stage_b_fixed_text_scorer.confidence_adapter.",
                "stage_b_fixed_text_scorer.confidence_pool.",
                "stage_b_fixed_text_scorer.confidence_veto_pool.",
            )
            if contract_version >= 2
            else (
                "stage_b_fixed_text_scorer.confidence_tower.",
                "stage_b_fixed_text_scorer.confidence_pool.",
            )
        )
    unexpected = [
        name for name in active_names if not name.startswith(allowed_prefixes)
    ]
    if unexpected:
        raise RuntimeError(
            "dense-duty active parameters violate phase ownership: "
            f"{unexpected[:20]}"
        )
    return active_names


def fingerprint_state(
    state: Mapping[str, torch.Tensor],
    *,
    active_parameter_names: Sequence[str],
    phase: str,
) -> dict[str, Any]:
    phase = _validate_phase(phase)
    active_names = _validate_active_names(state, active_parameter_names, phase)
    tensor_names = sorted(
        name for name, value in state.items() if torch.is_tensor(value)
    )
    if len(tensor_names) != len(state):
        unexpected = sorted(
            name for name, value in state.items() if not torch.is_tensor(value)
        )
        raise RuntimeError(
            f"dense-duty model state contains non-tensors: {unexpected[:20]}"
        )
    active_set = set(active_names)
    frozen_names = [name for name in tensor_names if name not in active_set]
    if not frozen_names:
        raise RuntimeError("dense-duty frozen model state is empty")
    names_sha256 = hashlib.sha256(_canonical_json_bytes(active_names)).hexdigest()
    return {
        "schema": STATE_FINGERPRINT_SCHEMA,
        "phase": phase,
        "active_parameter_names": active_names,
        "active_parameter_names_sha256": names_sha256,
        "active": _tensor_group_fingerprint(state, active_names),
        "frozen": _tensor_group_fingerprint(state, frozen_names),
    }


def fingerprint_model(model: torch.nn.Module, *, phase: str) -> dict[str, Any]:
    active_names = sorted(
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    return fingerprint_state(
        model.state_dict(),
        active_parameter_names=active_names,
        phase=phase,
    )


def validate_initial_fingerprint(
    value: Any, *, expected_phase: Optional[str] = None
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("dense-duty checkpoint lacks its initial state fingerprint")
    fingerprint = dict(value)
    if fingerprint.get("schema") != STATE_FINGERPRINT_SCHEMA:
        raise RuntimeError("dense-duty initial state fingerprint schema is invalid")
    phase = _validate_phase(fingerprint.get("phase", ""))
    if expected_phase is not None and phase != _validate_phase(expected_phase):
        raise RuntimeError("dense-duty initial state fingerprint phase mismatch")
    names = fingerprint.get("active_parameter_names")
    if not isinstance(names, list):
        raise RuntimeError("dense-duty initial fingerprint lacks active parameter names")
    expected_names_sha = hashlib.sha256(_canonical_json_bytes(names)).hexdigest()
    if fingerprint.get("active_parameter_names_sha256") != expected_names_sha:
        raise RuntimeError("dense-duty active parameter-name digest is invalid")
    for group_name in ("active", "frozen"):
        group = fingerprint.get(group_name)
        if not isinstance(group, Mapping):
            raise RuntimeError(
                f"dense-duty initial fingerprint lacks {group_name} record"
            )
        sha256 = group.get("sha256")
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(char not in "0123456789abcdef" for char in sha256)
        ):
            raise RuntimeError(
                f"dense-duty initial {group_name} fingerprint SHA256 is invalid"
            )
        for key in (
            "tensor_count",
            "element_count",
            "storage_bytes",
            "nonfinite_count",
        ):
            item = group.get(key)
            if isinstance(item, bool) or not isinstance(item, int) or item < 0:
                raise RuntimeError(
                    f"dense-duty initial {group_name}.{key} is invalid"
                )
    return fingerprint


def write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="ascii") as handle:
            json.dump(dict(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def audit_checkpoint_payload(
    payload: Mapping[str, Any],
    *,
    checkpoint_path: Optional[Path] = None,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise RuntimeError("dense-duty checkpoint payload must be a mapping")
    state = payload.get("model")
    args = payload.get("args")
    if not isinstance(state, Mapping) or not isinstance(args, Mapping):
        raise RuntimeError("dense-duty checkpoint lacks model state or saved args")
    if args.get("stage_b_dense_duty") is not True:
        raise RuntimeError("checkpoint is not labeled as dense-duty Stage B")
    phase = _validate_phase(args.get("stage_b_dense_duty_phase", ""))
    initial = validate_initial_fingerprint(
        args.get(FINGERPRINT_ARG), expected_phase=phase
    )
    current = fingerprint_state(
        state,
        active_parameter_names=initial["active_parameter_names"],
        phase=phase,
    )
    optimizer_updates = payload.get("optimizer_updates")
    if (
        isinstance(optimizer_updates, bool)
        or not isinstance(optimizer_updates, int)
        or optimizer_updates <= 0
    ):
        raise RuntimeError("dense-duty checkpoint has no successful optimizer update")
    fulltext_global_absolute_contract = (
        _v53_fulltext_global_absolute_contract(args)
        or _v54_fulltext_global_absolute_exact_residual_contract(args)
        or _v55_fulltext_global_independent_absolute_contract(args)
        or _v56_deployment_owned_global_contract(args)
        or _v57_deployed_global_balanced_absolute_contract(args)
        or _v58_deployment_owned_stable_fpr95_active_set_contract(args)
        or _v59_deployment_owned_query_global_contract(args)
        or _v60_deployment_owned_query_veto_contract(args)
    )
    if fulltext_global_absolute_contract:
        full_decoder_verifier = bool(
            args.get(
                "stage_b_dense_duty_confidence_full_decoder_verifier", False
            )
        )
        query_global_contract = (
            _v59_deployment_owned_query_global_revision(args)
            or _v60_deployment_owned_query_veto_revision(args)
        )
        deployment_owned_contract = (
            _v56_deployment_owned_global_revision(args)
            or _v57_deployed_global_balanced_absolute_revision(args)
            or _v58_deployment_owned_stable_fpr95_active_set_revision(args)
        )
        expected_active_tensor_count = (
            _V61_FULL_DECODER_ACTIVE_TENSOR_COUNT
            if full_decoder_verifier
            else _V53_ACTIVE_TENSOR_COUNT
            if query_global_contract
            else _V56_ACTIVE_TENSOR_COUNT
            if deployment_owned_contract
            else _V53_ACTIVE_TENSOR_COUNT
        )
        expected_active_element_count = (
            _V61_FULL_DECODER_ACTIVE_ELEMENT_COUNT
            if full_decoder_verifier
            else _V53_ACTIVE_ELEMENT_COUNT
            if query_global_contract
            else _V56_ACTIVE_ELEMENT_COUNT
            if deployment_owned_contract
            else _V53_ACTIVE_ELEMENT_COUNT
        )
        expected_owner_tensor_counts = (
            _V61_FULL_DECODER_OWNER_TENSOR_COUNTS
            if full_decoder_verifier
            else _V53_OWNER_TENSOR_COUNTS
            if query_global_contract
            else _V56_OWNER_TENSOR_COUNTS
            if deployment_owned_contract
            else _V53_OWNER_TENSOR_COUNTS
        )
        contract_label = (
            "V61-full-decoder"
            if full_decoder_verifier
            else "V59-V60"
            if query_global_contract
            else "V56-V58"
            if deployment_owned_contract
            else "V53-V55"
        )
        _validate_fulltext_active_fingerprint(
            initial["active"],
            current["active"],
            expected_tensor_count=expected_active_tensor_count,
            expected_element_count=expected_active_element_count,
            contract_label=contract_label,
        )
        _validate_fulltext_global_absolute_runtime_audit(
            args.get("stage_b_dense_duty_runtime_audit"),
            expected_steps=optimizer_updates,
            expected_owner_tensor_counts=expected_owner_tensor_counts,
            contract_label=contract_label,
        )
    frozen_unchanged = current["frozen"] == initial["frozen"]
    active_changed = current["active"]["sha256"] != initial["active"]["sha256"]
    if not frozen_unchanged:
        raise RuntimeError("dense-duty frozen state changed during training")
    if not active_changed:
        raise RuntimeError("dense-duty active state did not change during training")
    if current["active"]["nonfinite_count"] or current["frozen"]["nonfinite_count"]:
        raise RuntimeError("dense-duty checkpoint contains non-finite model state")

    checkpoint_record = None
    if checkpoint_path is not None:
        resolved = Path(checkpoint_path).expanduser().resolve(strict=True)
        checkpoint_record = {
            "path": str(resolved),
            "size_bytes": int(resolved.stat().st_size),
            "sha256": _file_sha256(resolved),
        }
    return {
        "schema": CHECKPOINT_AUDIT_SCHEMA,
        "status": "passed",
        "phase": phase,
        "optimizer_updates": optimizer_updates,
        "checkpoint": checkpoint_record,
        "ownership": {
            "active_parameter_names_sha256": initial[
                "active_parameter_names_sha256"
            ],
            "active_parameter_count": len(initial["active_parameter_names"]),
            "active_changed": active_changed,
            "frozen_unchanged": frozen_unchanged,
        },
        "initial": {
            "active": initial["active"],
            "frozen": initial["frozen"],
        },
        "current": {
            "active": current["active"],
            "frozen": current["frozen"],
        },
        "lineage": args.get("stage_b_dense_duty_lineage_audit"),
        "scorer_init": args.get("stage_b_v15_scorer_init_audit"),
        "runtime": args.get("stage_b_dense_duty_runtime_audit"),
        "source_closure": args.get(SOURCE_CLOSURE_ARG),
    }


def validate_rank_handoff_audit(
    rank_source: Any,
    *,
    execution_scope: str,
    rank_dataset_sha256: str,
    required_optimizer_updates: Optional[int] = None,
    code_source_sha256: Optional[str] = None,
) -> dict[str, Any]:
    """Validate and return the immutable rank lineage carried by confidence."""
    scope = str(execution_scope or "").strip().lower()
    if scope not in {"formal", "probe"}:
        raise RuntimeError("dense-duty rank handoff has an invalid execution scope")
    if not isinstance(rank_source, Mapping):
        raise RuntimeError(
            "dense-duty confidence checkpoint lacks its audited rank handoff"
        )
    optimizer_updates = rank_source.get("optimizer_updates")
    expected_updates = (
        int(required_optimizer_updates)
        if required_optimizer_updates is not None
        else optimizer_updates
    )
    ownership = rank_source.get("ownership", {})
    lineage = rank_source.get("lineage", {})
    dataset_config = (
        lineage.get("dataset_config", {}) if isinstance(lineage, Mapping) else {}
    )
    if code_source_sha256 is not None:
        expected_code_sha256 = _validate_sha256(
            code_source_sha256, label="rank handoff current code source"
        )
        try:
            rank_closure = validate_source_closure(
                rank_source.get("source_closure")
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "dense-duty confidence checkpoint has an invalid rank handoff audit"
            ) from exc
        if rank_closure["code"]["sha256"] != expected_code_sha256:
            raise RuntimeError(
                "dense-duty rank-to-confidence code source closure drifted"
            )
    if (
        rank_source.get("schema") != CHECKPOINT_AUDIT_SCHEMA
        or rank_source.get("status") != "passed"
        or rank_source.get("phase") != "rank"
        or isinstance(expected_updates, bool)
        or not isinstance(expected_updates, int)
        or expected_updates <= 0
        or optimizer_updates != expected_updates
        or not isinstance(ownership, Mapping)
        or ownership.get("active_changed") is not True
        or ownership.get("frozen_unchanged") is not True
        or not isinstance(lineage, Mapping)
        or not isinstance(dataset_config, Mapping)
        or lineage.get("execution_scope") != scope
        or lineage.get("no_stage_b_teacher") is not True
        or dataset_config.get("sha256") != str(rank_dataset_sha256)
    ):
        raise RuntimeError(
            "dense-duty confidence checkpoint has an invalid rank handoff audit"
        )
    return dict(rank_source)


def validate_confidence_adapter_rank_source_audit(
    value: Any,
    args: Any,
) -> dict[str, Any]:
    """Validate the legacy rank audit paired with an adapter migration receipt."""
    values = _argument_mapping(args)
    if str(values.get("stage_b_v22_score_ownership", "")).strip() != (
        _ADAPTER_OWNERSHIP
    ):
        raise RuntimeError(
            "confidence-adapter rank lineage requires the adapter ownership contract"
        )
    return validate_rank_handoff_audit(
        value,
        execution_scope="formal",
        rank_dataset_sha256=str(
            values.get("stage_b_dense_duty_rank_dataset_config_sha256", "")
        ),
        required_optimizer_updates=int(
            values.get("stage_b_dense_duty_rank_source_optimizer_updates", 0)
        ),
        # The rank was trained under the sealed v1 source. Its current-code
        # bridge is the bitwise migration receipt, not same-source equality.
        code_source_sha256=None,
    )


def _validate_resume_rng_state(value: Any, *, label: str) -> None:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"dense-duty strict resume has invalid {label} mapping")
    missing = sorted({"python", "numpy", "torch", "cuda"}.difference(value))
    if missing:
        raise RuntimeError(
            f"dense-duty strict resume {label} lacks RNG fields: {missing}"
        )
    if not isinstance(value["python"], tuple) or not isinstance(
        value["numpy"], tuple
    ):
        raise RuntimeError(
            f"dense-duty strict resume {label} has invalid Python/NumPy RNG state"
        )
    if not torch.is_tensor(value["torch"]):
        raise RuntimeError(
            f"dense-duty strict resume {label} has invalid torch RNG state"
        )
    cuda_state = value["cuda"]
    if cuda_state is not None and (
        not isinstance(cuda_state, (list, tuple))
        or any(not torch.is_tensor(item) for item in cuda_state)
    ):
        raise RuntimeError(
            f"dense-duty strict resume {label} has invalid CUDA RNG state"
        )


def _candidate_sample_runtime_contract(values: Mapping[str, Any]) -> bool:
    return (
        str(values.get("stage_b_dense_duty_confidence_revision", "")).strip()
        == "word_veto_candidate_sample_calibrator_split_v52"
        and str(
            values.get(
                "stage_b_dense_duty_confidence_head_gradient_contract", ""
            )
        ).strip().lower()
        == "split_token_veto_candidate_absolute_sample_calibrator_v6"
    )


def _validate_candidate_sample_runtime_audit(
    runtime: Mapping[str, Any], *, expected_steps: int
) -> dict[str, Any]:
    """Validate every optimizer boundary of the V52 three-owner contract."""
    if type(expected_steps) is not int or expected_steps <= 0:
        raise RuntimeError("V52 runtime audit requires positive expected steps")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("schema") != "pivot.stageb.dense_duty_runtime_audit/v1"
        or runtime.get("clip_contract_schema")
        != "pivot.stageb.dense_duty_three_owner_clip_contract/v1"
        or type(runtime.get("clip_contract_checked_steps")) is not int
        or runtime.get("clip_contract_checked_steps") != expected_steps
    ):
        raise RuntimeError(
            "V52 runtime audit lacks one three-owner clip check per update"
        )

    violation_fields = (
        "owner_clip_violation_steps",
        "active_pre_decomposition_violation_steps",
        "active_post_decomposition_violation_steps",
        "live_tensor_count_violation_steps",
        "active_monotonic_violation_steps",
    )
    invalid_violations = {
        field: runtime.get(field)
        for field in violation_fields
        if type(runtime.get(field)) is not int or runtime.get(field) != 0
    }
    expected_live = {
        "token_veto": 21,
        "candidate_absolute": 39,
        "sample_calibrator": 7,
    }
    invalid_counts = {
        owner: {
            "expected": runtime.get(f"expected_{owner}_tensor_count"),
            "observed": runtime.get(f"last_observed_{owner}_tensor_count"),
        }
        for owner, count in expected_live.items()
        if type(runtime.get(f"expected_{owner}_tensor_count")) is not int
        or runtime.get(f"expected_{owner}_tensor_count") != count
        or type(runtime.get(f"last_observed_{owner}_tensor_count")) is not int
        or runtime.get(f"last_observed_{owner}_tensor_count") != count
    }
    invalid_owners = {}
    for owner in expected_live:
        last = runtime.get(f"last_{owner}_grad_norm_preclip")
        maximum = runtime.get(f"max_{owner}_grad_norm_preclip")
        nonfinite = runtime.get(f"nonfinite_{owner}_gradient_boundaries", 0)
        zero = runtime.get(f"zero_{owner}_gradient_successful_steps", 0)
        valid_norms = (
            isinstance(last, (int, float))
            and not isinstance(last, bool)
            and isinstance(maximum, (int, float))
            and not isinstance(maximum, bool)
            and math.isfinite(float(last))
            and math.isfinite(float(maximum))
            and float(last) > 0.0
            and float(maximum) > 0.0
        )
        valid_counters = (
            type(nonfinite) is int
            and nonfinite == 0
            and type(zero) is int
            and zero == 0
        )
        if not valid_norms or not valid_counters:
            invalid_owners[owner] = {
                "last": last,
                "maximum": maximum,
                "nonfinite_boundaries": nonfinite,
                "zero_successful_steps": zero,
            }

    tolerance = runtime.get("clip_contract_tolerance")
    max_norm = runtime.get("clip_contract_max_norm")
    residual_fields = (
        "max_active_pre_decomposition_residual",
        "max_active_post_decomposition_residual",
        "max_owner_clip_residual",
        "max_active_monotonic_residual",
    )
    valid_tolerance = (
        isinstance(tolerance, (int, float))
        and not isinstance(tolerance, bool)
        and math.isfinite(float(tolerance))
        and float(tolerance) > 0.0
    )
    invalid_residuals = {
        field: runtime.get(field)
        for field in residual_fields
        if not isinstance(runtime.get(field), (int, float))
        or isinstance(runtime.get(field), bool)
        or not math.isfinite(float(runtime.get(field)))
        or float(runtime.get(field)) < 0.0
        or (valid_tolerance and float(runtime.get(field)) > float(tolerance))
    }
    if (
        invalid_violations
        or invalid_counts
        or invalid_owners
        or not valid_tolerance
        or not isinstance(max_norm, (int, float))
        or isinstance(max_norm, bool)
        or not math.isclose(float(max_norm), 0.1, rel_tol=0.0, abs_tol=1e-12)
        or invalid_residuals
    ):
        raise RuntimeError(
            "V52 three-owner runtime audit is invalid: "
            f"violations={invalid_violations}, counts={invalid_counts}, "
            f"owners={invalid_owners}, residuals={invalid_residuals}"
        )
    return dict(runtime)


def _validate_fulltext_active_fingerprint(
    initial: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    expected_tensor_count: int,
    expected_element_count: int,
    contract_label: str,
) -> None:
    invalid = {}
    for label, group in (("initial", initial), ("current", current)):
        tensor_count = group.get("tensor_count") if isinstance(group, Mapping) else None
        element_count = (
            group.get("element_count") if isinstance(group, Mapping) else None
        )
        if (
            tensor_count != expected_tensor_count
            or element_count != expected_element_count
        ):
            invalid[label] = {
                "tensor_count": tensor_count,
                "element_count": element_count,
            }
    if invalid:
        raise RuntimeError(
            f"{contract_label} confidence checkpoint violates its exact "
            f"{expected_tensor_count}-tensor/{expected_element_count}-element "
            f"production ownership: {invalid}"
        )


def _validate_v53_active_fingerprint(
    initial: Mapping[str, Any], current: Mapping[str, Any]
) -> None:
    """Backward-compatible exact V53-V55 ownership validator."""
    _validate_fulltext_active_fingerprint(
        initial,
        current,
        expected_tensor_count=_V53_ACTIVE_TENSOR_COUNT,
        expected_element_count=_V53_ACTIVE_ELEMENT_COUNT,
        contract_label="V53",
    )


def _validate_fulltext_global_absolute_runtime_audit(
    runtime: Mapping[str, Any],
    *,
    expected_steps: int,
    expected_owner_tensor_counts: Mapping[str, int] = _V53_OWNER_TENSOR_COUNTS,
    contract_label: str = "V53",
) -> dict[str, Any]:
    """Validate every successful update of a two-owner full-text contract."""
    if type(expected_steps) is not int or expected_steps <= 0:
        raise RuntimeError(
            f"{contract_label} runtime audit requires positive expected steps"
        )
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("schema") != "pivot.stageb.dense_duty_runtime_audit/v1"
        or runtime.get("clip_contract_schema")
        != "pivot.stageb.dense_duty_two_owner_clip_contract/v1"
        or type(runtime.get("successful_optimizer_steps")) is not int
        or runtime.get("successful_optimizer_steps") != expected_steps
        or type(runtime.get("optimizer_step_boundaries")) is not int
        or runtime.get("optimizer_step_boundaries") < expected_steps
        or type(runtime.get("clip_contract_checked_steps")) is not int
        or runtime.get("clip_contract_checked_steps") != expected_steps
    ):
        raise RuntimeError(
            f"{contract_label} runtime audit lacks one two-owner clip check per "
            "successful update"
        )

    violation_fields = (
        "owner_clip_violation_steps",
        "active_pre_decomposition_violation_steps",
        "active_post_decomposition_violation_steps",
        "live_tensor_count_violation_steps",
        "active_monotonic_violation_steps",
    )
    invalid_violations = {
        field: runtime.get(field)
        for field in violation_fields
        if type(runtime.get(field)) is not int or runtime.get(field) != 0
    }
    invalid_global_gradients = {
        field: runtime.get(field)
        for field in (
            "nonfinite_gradient_boundaries",
            "zero_gradient_successful_steps",
        )
        if type(runtime.get(field)) is not int or runtime.get(field) != 0
    }
    active_maximum = runtime.get("max_active_grad_norm_preclip")
    if (
        not isinstance(active_maximum, (int, float))
        or isinstance(active_maximum, bool)
        or not math.isfinite(float(active_maximum))
        or float(active_maximum) <= 0.0
    ):
        invalid_global_gradients["max_active_grad_norm_preclip"] = active_maximum
    invalid_counts = {
        owner: {
            "expected": runtime.get(f"expected_{owner}_tensor_count"),
            "observed": runtime.get(f"last_observed_{owner}_tensor_count"),
        }
        for owner, count in expected_owner_tensor_counts.items()
        if type(runtime.get(f"expected_{owner}_tensor_count")) is not int
        or runtime.get(f"expected_{owner}_tensor_count") != count
        or type(runtime.get(f"last_observed_{owner}_tensor_count")) is not int
        or runtime.get(f"last_observed_{owner}_tensor_count") != count
    }
    invalid_owners = {}
    for owner in expected_owner_tensor_counts:
        last = runtime.get(f"last_{owner}_grad_norm_preclip")
        maximum = runtime.get(f"max_{owner}_grad_norm_preclip")
        nonfinite = runtime.get(f"nonfinite_{owner}_gradient_boundaries", 0)
        zero = runtime.get(f"zero_{owner}_gradient_successful_steps", 0)
        valid_norms = (
            isinstance(last, (int, float))
            and not isinstance(last, bool)
            and isinstance(maximum, (int, float))
            and not isinstance(maximum, bool)
            and math.isfinite(float(last))
            and math.isfinite(float(maximum))
            and float(last) > 0.0
            and float(maximum) > 0.0
        )
        valid_counters = (
            type(nonfinite) is int
            and nonfinite == 0
            and type(zero) is int
            and zero == 0
        )
        if not valid_norms or not valid_counters:
            invalid_owners[owner] = {
                "last": last,
                "maximum": maximum,
                "nonfinite_boundaries": nonfinite,
                "zero_successful_steps": zero,
            }

    tolerance = runtime.get("clip_contract_tolerance")
    max_norm = runtime.get("clip_contract_max_norm")
    residual_fields = (
        "max_active_pre_decomposition_residual",
        "max_active_post_decomposition_residual",
        "max_owner_clip_residual",
        "max_active_monotonic_residual",
    )
    valid_tolerance = (
        isinstance(tolerance, (int, float))
        and not isinstance(tolerance, bool)
        and math.isfinite(float(tolerance))
        and float(tolerance) > 0.0
    )
    invalid_residuals = {
        field: runtime.get(field)
        for field in residual_fields
        if not isinstance(runtime.get(field), (int, float))
        or isinstance(runtime.get(field), bool)
        or not math.isfinite(float(runtime.get(field)))
        or float(runtime.get(field)) < 0.0
        or (valid_tolerance and float(runtime.get(field)) > float(tolerance))
    }
    if (
        invalid_violations
        or invalid_global_gradients
        or invalid_counts
        or invalid_owners
        or not valid_tolerance
        or not isinstance(max_norm, (int, float))
        or isinstance(max_norm, bool)
        or not math.isclose(float(max_norm), 0.1, rel_tol=0.0, abs_tol=1e-12)
        or invalid_residuals
    ):
        raise RuntimeError(
            f"{contract_label} two-owner runtime audit is invalid: "
            f"violations={invalid_violations}, global={invalid_global_gradients}, "
            f"counts={invalid_counts}, "
            f"owners={invalid_owners}, residuals={invalid_residuals}"
        )
    return dict(runtime)


def validate_strict_resume_checkpoint_path(
    current_args: Any, checkpoint_path: Path
) -> Path:
    values = _argument_mapping(current_args)
    resolved_path = Path(checkpoint_path).expanduser().resolve(strict=True)
    if resolved_path.name != STRICT_RESUME_CHECKPOINT_NAME:
        raise RuntimeError(
            "dense-duty strict resume trusts only the atomic "
            f"{STRICT_RESUME_CHECKPOINT_NAME} snapshot"
        )
    output_dir = values.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise RuntimeError("dense-duty strict resume requires an exact output_dir")
    expected_path = (
        Path(output_dir).expanduser().resolve(strict=True)
        / STRICT_RESUME_CHECKPOINT_NAME
    )
    if resolved_path != expected_path:
        raise RuntimeError(
            "dense-duty strict resume checkpoint must be the atomic snapshot in "
            f"its output_dir: expected={expected_path}, observed={resolved_path}"
        )
    return resolved_path


def validate_strict_resume_checkpoint_payload(
    payload: Mapping[str, Any],
    current_args: Any,
    *,
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Fail closed on a same-phase dense-duty optimizer-boundary snapshot."""
    if not isinstance(payload, Mapping):
        raise RuntimeError("dense-duty strict resume payload must be a mapping")
    values = _argument_mapping(current_args)
    resolved_path = validate_strict_resume_checkpoint_path(
        current_args, checkpoint_path
    )

    missing = sorted(STRICT_RESUME_REQUIRED_KEYS.difference(payload))
    if missing:
        raise RuntimeError(
            "dense-duty strict resume is missing complete training state: "
            f"{missing}"
        )
    for key in ("model", "criterion", "optimizer", "lr_scheduler", "scaler"):
        if not isinstance(payload[key], Mapping):
            raise RuntimeError(
                f"dense-duty strict resume has invalid {key} state mapping"
            )
    for key in ("epoch", "iteration", "optimizer_updates"):
        value = payload[key]
        if type(value) is not int or value < 0:
            raise RuntimeError(
                f"dense-duty strict resume has invalid {key}={value!r}"
            )
    if type(payload["epoch_finished"]) is not bool:
        raise RuntimeError(
            "dense-duty strict resume epoch_finished must be an exact bool"
        )
    _validate_resume_rng_state(payload["rng_state"], label="rng_state")
    _validate_resume_rng_state(
        payload["epoch_rng_state"], label="epoch_rng_state"
    )

    saved_args = payload["args"]
    if not isinstance(saved_args, Mapping):
        raise RuntimeError(
            "dense-duty strict resume requires its complete saved argument mapping"
        )
    phase = _validate_phase(values.get("stage_b_dense_duty_phase", ""))
    initial_fingerprint = validate_initial_fingerprint(
        saved_args.get(FINGERPRINT_ARG), expected_phase=phase
    )
    saved_contract = saved_args.get(TRAINING_CONTRACT_ARG)
    rebuilt_saved_contract = build_training_contract(saved_args)
    if not isinstance(saved_contract, Mapping) or dict(saved_contract) != (
        rebuilt_saved_contract
    ):
        raise RuntimeError(
            "dense-duty strict resume saved training contract is absent or invalid"
        )
    training_contract = validate_resume_training_contract(current_args, saved_args)

    if values.get("distributed") is not False or values.get("world_size") != 1:
        raise RuntimeError(
            "dense-duty strict resume requires single-process training because "
            "the atomic checkpoint stores one process RNG state"
        )

    accumulation = values.get("gradient_accumulation_steps")
    update_budget = values.get("max_train_iters")
    if type(accumulation) is not int or accumulation <= 0:
        raise RuntimeError(
            "dense-duty strict resume has an invalid accumulation contract"
        )
    if type(update_budget) is not int or update_budget <= 0:
        raise RuntimeError("dense-duty strict resume has an invalid update budget")
    optimizer_updates = payload["optimizer_updates"]
    if optimizer_updates >= update_budget:
        raise RuntimeError(
            "dense-duty strict resume accepts only an unfinished phase checkpoint"
        )
    runtime_audit = saved_args.get("stage_b_dense_duty_runtime_audit")
    if (
        not isinstance(runtime_audit, Mapping)
        or runtime_audit.get("schema")
        != "pivot.stageb.dense_duty_runtime_audit/v1"
        or type(runtime_audit.get("successful_optimizer_steps")) is not int
        or runtime_audit.get("successful_optimizer_steps") != optimizer_updates
        or type(runtime_audit.get("optimizer_step_boundaries")) is not int
        or runtime_audit.get("optimizer_step_boundaries") < optimizer_updates
    ):
        raise RuntimeError(
            "dense-duty strict resume runtime audit disagrees with optimizer progress"
        )
    if _candidate_sample_runtime_contract(values):
        _validate_candidate_sample_runtime_audit(
            runtime_audit, expected_steps=optimizer_updates
        )
    if (
        _v53_fulltext_global_absolute_contract(values)
        or _v54_fulltext_global_absolute_exact_residual_contract(values)
        or _v55_fulltext_global_independent_absolute_contract(values)
        or _v56_deployment_owned_global_contract(values)
        or _v57_deployed_global_balanced_absolute_contract(values)
        or _v58_deployment_owned_stable_fpr95_active_set_contract(values)
        or _v59_deployment_owned_query_global_contract(values)
        or _v60_deployment_owned_query_veto_contract(values)
    ):
        full_decoder_verifier = bool(
            values.get(
                "stage_b_dense_duty_confidence_full_decoder_verifier", False
            )
        )
        current_fingerprint = fingerprint_state(
            payload["model"],
            active_parameter_names=initial_fingerprint[
                "active_parameter_names"
            ],
            phase=phase,
        )
        query_global_contract = (
            _v59_deployment_owned_query_global_revision(values)
            or _v60_deployment_owned_query_veto_revision(values)
        )
        deployment_owned_contract = (
            _v56_deployment_owned_global_revision(values)
            or _v57_deployed_global_balanced_absolute_revision(values)
            or _v58_deployment_owned_stable_fpr95_active_set_revision(values)
        )
        expected_active_tensor_count = (
            _V61_FULL_DECODER_ACTIVE_TENSOR_COUNT
            if full_decoder_verifier
            else _V53_ACTIVE_TENSOR_COUNT
            if query_global_contract
            else _V56_ACTIVE_TENSOR_COUNT
            if deployment_owned_contract
            else _V53_ACTIVE_TENSOR_COUNT
        )
        expected_active_element_count = (
            _V61_FULL_DECODER_ACTIVE_ELEMENT_COUNT
            if full_decoder_verifier
            else _V53_ACTIVE_ELEMENT_COUNT
            if query_global_contract
            else _V56_ACTIVE_ELEMENT_COUNT
            if deployment_owned_contract
            else _V53_ACTIVE_ELEMENT_COUNT
        )
        expected_owner_tensor_counts = (
            _V61_FULL_DECODER_OWNER_TENSOR_COUNTS
            if full_decoder_verifier
            else _V53_OWNER_TENSOR_COUNTS
            if query_global_contract
            else _V56_OWNER_TENSOR_COUNTS
            if deployment_owned_contract
            else _V53_OWNER_TENSOR_COUNTS
        )
        contract_label = (
            "V61-full-decoder"
            if full_decoder_verifier
            else "V59-V60"
            if query_global_contract
            else "V56-V58"
            if deployment_owned_contract
            else "V53-V55"
        )
        _validate_fulltext_active_fingerprint(
            initial_fingerprint["active"],
            current_fingerprint["active"],
            expected_tensor_count=expected_active_tensor_count,
            expected_element_count=expected_active_element_count,
            contract_label=contract_label,
        )
        _validate_fulltext_global_absolute_runtime_audit(
            runtime_audit,
            expected_steps=optimizer_updates,
            expected_owner_tensor_counts=expected_owner_tensor_counts,
            contract_label=contract_label,
        )

    reason = payload["checkpoint_reason"]
    epoch_finished = payload["epoch_finished"]
    iteration = payload["iteration"]
    if type(reason) is not str or reason not in {
        "interval",
        "signal",
        "interval_epoch",
        "signal_after_epoch",
    }:
        raise RuntimeError(
            "dense-duty strict resume checkpoint_reason is not a supported "
            "nonterminal atomic-save reason"
        )
    if epoch_finished:
        if iteration != 0 or reason not in {
            "signal",
            "interval_epoch",
            "signal_after_epoch",
        }:
            raise RuntimeError(
                "dense-duty strict resume has an invalid epoch-boundary snapshot"
            )
    elif iteration <= 0 or iteration % accumulation != 0 or reason not in {
        "interval",
        "signal",
    }:
        raise RuntimeError(
            "dense-duty strict resume is not at an exact optimizer-update boundary"
        )
    forward_pack_factor = int(
        values.get("stage_b_dense_duty_forward_pack_factor", 1) or 1
    )
    if not epoch_finished and forward_pack_factor > 1:
        expected_logical_batches = values.get(
            "stage_b_dense_duty_expected_logical_batches_per_epoch"
        )
        expected_physical_forwards = values.get(
            "stage_b_dense_duty_expected_physical_forwards_per_epoch"
        )
        if (
            type(expected_logical_batches) is not int
            or expected_logical_batches <= 0
            or type(expected_physical_forwards) is not int
            or expected_physical_forwards
            != (
                expected_logical_batches + forward_pack_factor - 1
            )
            // forward_pack_factor
        ):
            raise RuntimeError(
                "dense-duty packed epoch geometry contract is invalid"
            )
        if iteration >= expected_physical_forwards:
            raise RuntimeError(
                "dense-duty strict resume iteration is outside the packed "
                "physical-forward epoch boundary"
            )
    if reason in {"interval", "interval_epoch"}:
        interval = values.get("iter_checkpoint_interval")
        if (
            type(interval) is not int
            or interval <= 0
            or optimizer_updates <= 0
            or optimizer_updates % interval != 0
        ):
            raise RuntimeError(
                "dense-duty strict resume interval checkpoint violates its save cadence"
            )

    rank_handoff = None
    if phase == "confidence":
        execution_scope = str(
            values.get("stage_b_dense_duty_execution_scope", "") or ""
        ).strip().lower()
        if str(values.get("stage_b_v22_score_ownership", "")).strip() == (
            _ADAPTER_OWNERSHIP
        ):
            from util.stage_b_confidence_adapter_migration import (
                validate_confidence_adapter_migration_audit,
            )

            rank_handoff = validate_confidence_adapter_migration_audit(
                saved_args.get(
                    "stage_b_dense_duty_confidence_adapter_migration_audit"
                ),
                source_checkpoint_sha256=str(
                    values["stage_b_dense_duty_rank_source_checkpoint_sha256"]
                ),
                source_optimizer_updates=int(
                    values["stage_b_dense_duty_rank_source_optimizer_updates"]
                ),
                source_checkpoint_reason=str(
                    values["stage_b_dense_duty_rank_source_checkpoint_reason"]
                ),
                rank_sha256=str(
                    values["stage_b_dense_duty_rank_source_rank_sha256"]
                ),
                transferred_sha256=str(
                    values["stage_b_dense_duty_rank_source_transferred_sha256"]
                ),
            )
            validate_confidence_adapter_rank_source_audit(
                saved_args.get("stage_b_dense_duty_rank_source_checkpoint_audit"),
                values,
            )
        else:
            rank_handoff = validate_rank_handoff_audit(
                saved_args.get("stage_b_dense_duty_rank_source_checkpoint_audit"),
                execution_scope=execution_scope,
                rank_dataset_sha256=str(
                    values.get("stage_b_dense_duty_rank_dataset_config_sha256", "")
                ),
                required_optimizer_updates=(
                    int(values["stage_b_dense_duty_rank_expected_optimizer_updates"])
                    if execution_scope == "formal"
                    else None
                ),
                code_source_sha256=validate_source_closure(
                    values.get(SOURCE_CLOSURE_ARG)
                )["code"]["sha256"],
            )
    return {
        "phase": phase,
        "checkpoint_path": str(resolved_path),
        "epoch": payload["epoch"],
        "iteration": iteration,
        "optimizer_updates": optimizer_updates,
        "epoch_finished": epoch_finished,
        "checkpoint_reason": reason,
        "training_contract": training_contract,
        "runtime_audit": dict(runtime_audit),
        "rank_handoff": rank_handoff,
    }


def validate_evaluation_checkpoint_payload(
    payload: Mapping[str, Any],
    cfg: Any,
    *,
    checkpoint_path: Optional[Path] = None,
    current_code_source_closure: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    audit = audit_checkpoint_payload(payload, checkpoint_path=checkpoint_path)
    if audit["phase"] != "confidence":
        raise RuntimeError(
            "dense-duty evaluation requires a completed confidence checkpoint"
        )
    saved_args = payload["args"]
    evaluation_scope = str(
        getattr(cfg, "stage_b_dense_duty_evaluation_scope", "formal") or ""
    ).strip().lower()
    if evaluation_scope not in {"formal", "probe"}:
        raise RuntimeError(
            "stage_b_dense_duty_evaluation_scope must be 'formal' or 'probe'"
        )
    if saved_args.get("stage_b_dense_duty_execution_scope") != evaluation_scope:
        raise RuntimeError(
            "dense-duty checkpoint execution scope does not match evaluation scope"
        )
    if payload.get("checkpoint_reason") != "max_train_iters":
        raise RuntimeError(
            "dense-duty evaluation requires a terminal max_train_iters checkpoint"
        )

    expected_rank_updates = int(
        getattr(cfg, "stage_b_dense_duty_rank_expected_optimizer_updates", 0)
    )
    expected_confidence_updates = int(
        getattr(cfg, "stage_b_dense_duty_confidence_expected_optimizer_updates", 0)
    )
    if expected_rank_updates <= 0 or expected_confidence_updates <= 0:
        raise RuntimeError(
            "dense-duty evaluation config lacks positive phase update contracts"
        )
    if evaluation_scope == "formal":
        current_code = validate_code_source_closure(
            current_code_source_closure
            if current_code_source_closure is not None
            else build_code_source_closure()
        )
        saved_source = validate_source_closure(
            saved_args.get(SOURCE_CLOSURE_ARG)
        )
        if saved_source["code"]["sha256"] != current_code["sha256"]:
            raise RuntimeError(
                "dense-duty formal evaluation code source closure drifted"
            )
        audit["source_closure"] = saved_source
        if (
            payload.get("optimizer_updates") != expected_confidence_updates
            or saved_args.get("max_train_iters") != expected_confidence_updates
        ):
            raise RuntimeError(
                "dense-duty formal evaluation requires the exact terminal "
                f"confidence checkpoint at {expected_confidence_updates} updates"
            )
        expected_runtime = {
            "batch_size": int(
                getattr(
                    cfg,
                    "stage_b_dense_duty_expected_physical_batch_size",
                    0,
                )
            ),
            "gradient_accumulation_steps": int(
                getattr(
                    cfg,
                    "stage_b_dense_duty_expected_gradient_accumulation_steps",
                    0,
                )
            ),
            "stage_b_v11_expression_microbatch": int(
                getattr(
                    cfg,
                    "stage_b_dense_duty_expected_expression_microbatch",
                    0,
                )
            ),
        }
        if str(getattr(cfg, "stage_b_v22_score_ownership", "")).strip() == (
            _ADAPTER_OWNERSHIP
        ) and int(
            getattr(cfg, "stage_b_dense_duty_forward_pack_factor", 1) or 1
        ) > 1:
            expected_runtime.update(
                {
                    key: int(getattr(cfg, key, 0))
                    for key in _PACKED_FORWARD_CONTRACT_KEYS
                }
            )
        observed_runtime = {
            key: int(saved_args.get(key, 0)) for key in expected_runtime
        }
        if (
            any(value <= 0 for value in expected_runtime.values())
            or observed_runtime != expected_runtime
        ):
            raise RuntimeError(
                "dense-duty formal checkpoint violates the measured runtime "
                f"contract: expected={expected_runtime}, "
                f"observed={observed_runtime}"
            )
        runtime = saved_args.get("stage_b_dense_duty_runtime_audit")
        if (
            not isinstance(runtime, Mapping)
            or runtime.get("schema")
            != "pivot.stageb.dense_duty_runtime_audit/v1"
            or runtime.get("successful_optimizer_steps")
            != expected_confidence_updates
            or int(runtime.get("optimizer_step_boundaries", -1))
            < expected_confidence_updates
            or int(runtime.get("zero_gradient_successful_steps", -1)) != 0
            or float(runtime.get("max_active_grad_norm_preclip", 0.0)) <= 0.0
            or int(runtime.get("peak_reserved_bytes", 0)) <= 0
        ):
            raise RuntimeError(
                "dense-duty formal checkpoint lacks a valid confidence runtime "
                "audit"
            )
        if (
            str(
                saved_args.get("stage_b_dense_duty_confidence_revision", "")
            ).strip()
            == "word_veto_candidate_split_global_trust_veto_v49"
            and str(
                saved_args.get(
                    "stage_b_dense_duty_confidence_head_gradient_contract", ""
                )
            ).strip().lower()
            == "split_token_veto_global_trust_veto_v4"
        ):
            invalid_subowners = {}
            for owner in ("global_trust", "global_veto"):
                last = runtime.get(f"last_{owner}_grad_norm_preclip")
                maximum = runtime.get(f"max_{owner}_grad_norm_preclip")
                nonfinite = runtime.get(
                    f"nonfinite_{owner}_gradient_boundaries", 0
                )
                zero = runtime.get(
                    f"zero_{owner}_gradient_successful_steps", 0
                )
                valid_norms = (
                    isinstance(last, (int, float))
                    and not isinstance(last, bool)
                    and isinstance(maximum, (int, float))
                    and not isinstance(maximum, bool)
                    and math.isfinite(float(last))
                    and math.isfinite(float(maximum))
                    and float(last) > 0.0
                    and float(maximum) > 0.0
                )
                valid_counters = (
                    type(nonfinite) is int
                    and nonfinite == 0
                    and type(zero) is int
                    and zero == 0
                )
                if not valid_norms or not valid_counters:
                    invalid_subowners[owner] = {
                        "last": last,
                        "maximum": maximum,
                        "nonfinite_boundaries": nonfinite,
                        "zero_successful_steps": zero,
                    }
            if invalid_subowners:
                raise RuntimeError(
                    "dense-duty formal v31 checkpoint lacks continuously live "
                    f"global trust/veto gradients: {invalid_subowners}"
                )
        if (
            str(
                saved_args.get("stage_b_dense_duty_confidence_revision", "")
            ).strip()
            == "word_veto_candidate_sample_calibrator_split_v52"
            and str(
                saved_args.get(
                    "stage_b_dense_duty_confidence_head_gradient_contract", ""
                )
            ).strip().lower()
            == "split_token_veto_candidate_absolute_sample_calibrator_v6"
        ):
            _validate_candidate_sample_runtime_audit(
                runtime, expected_steps=expected_confidence_updates
            )
        if (
            str(
                saved_args.get("stage_b_dense_duty_confidence_revision", "")
            ).strip()
            == "word_veto_candidate_split_independent_deployed_router_v51"
            and str(
                saved_args.get(
                    "stage_b_dense_duty_confidence_head_gradient_contract", ""
                )
            ).strip().lower()
            == "split_token_veto_deployed_router_global_absolute_v5"
        ):
            invalid_owners = {}
            for owner in ("token_veto", "deployed_router", "global_absolute"):
                last = runtime.get(f"last_{owner}_grad_norm_preclip")
                maximum = runtime.get(f"max_{owner}_grad_norm_preclip")
                nonfinite = runtime.get(
                    f"nonfinite_{owner}_gradient_boundaries", 0
                )
                zero = runtime.get(
                    f"zero_{owner}_gradient_successful_steps", 0
                )
                valid_norms = (
                    isinstance(last, (int, float))
                    and not isinstance(last, bool)
                    and isinstance(maximum, (int, float))
                    and not isinstance(maximum, bool)
                    and math.isfinite(float(last))
                    and math.isfinite(float(maximum))
                    and float(last) > 0.0
                    and float(maximum) > 0.0
                )
                valid_counters = (
                    type(nonfinite) is int
                    and nonfinite == 0
                    and type(zero) is int
                    and zero == 0
                )
                if not valid_norms or not valid_counters:
                    invalid_owners[owner] = {
                        "last": last,
                        "maximum": maximum,
                        "nonfinite_boundaries": nonfinite,
                        "zero_successful_steps": zero,
                    }
            if invalid_owners:
                raise RuntimeError(
                    "dense-duty formal v33 checkpoint lacks continuously live "
                    f"three-owner gradients: {invalid_owners}"
                )
        audit["runtime"] = dict(runtime)
    elif int(payload.get("optimizer_updates", 0)) <= 0:
        raise RuntimeError(
            "dense-duty probe evaluation requires confidence training progress"
        )

    required_equal_args = (
        "stage_b_dense_duty_no_stageb_teacher",
        "stage_b_v22_score_ownership",
        "stage_b_dense_duty_base_checkpoint_sha256",
        "stage_b_dense_duty_text_checkpoint_sha256",
        "stage_b_dense_duty_tn_manifest_sha256",
        "stage_b_dense_duty_dataset_config_sha256",
        "stage_b_v11_candidate_topk",
        "stage_b_v11_num_layers",
        "stage_b_v15_patch_rank_fusion",
        "stage_b_v15_patch_rank_weight",
    )
    adapter_contract = str(
        getattr(cfg, "stage_b_v22_score_ownership", "")
    ).strip() == _ADAPTER_OWNERSHIP
    if adapter_contract:
        required_equal_args = required_equal_args + _ADAPTER_RESUME_CONTRACT_KEYS
        if str(
            getattr(
                cfg,
                "stage_b_dense_duty_confidence_phrase_aggregation",
                "",
            )
        ).strip().lower() in {
            "trace_activated_word_veto_product_v1",
            "trace_activated_word_veto_penalty_v2",
            "trace_activated_word_veto_absolute_cap_v4",
            "trace_activated_word_veto_gated_pool_absolute_cap_v5",
        }:
            required_equal_args += _WORD_VETO_RESUME_CONTRACT_KEYS
            if float(
                getattr(cfg, "stage_b_dense_duty_raw_veto_gate_weight", 0.0)
                or 0.0
            ) > 0.0:
                required_equal_args += _RAW_VETO_GATE_RESUME_CONTRACT_KEYS
            confidence_revision = str(
                getattr(
                    cfg,
                    "stage_b_dense_duty_confidence_revision",
                    "",
                )
            ).strip()
            if confidence_revision in {
                "word_veto_candidate_asymmetric_deployed_routing_v43",
                "word_veto_candidate_split_tail_aligned_v45",
                "word_veto_candidate_split_positive_tail_v46",
                "word_veto_candidate_split_boundary_routing_v47",
                "word_veto_candidate_split_fpr_active_set_v48",
                "word_veto_candidate_split_global_trust_veto_v49",
                "word_veto_candidate_split_strong_boundary_routing_v50",
                "word_veto_candidate_split_independent_deployed_router_v51",
                "word_veto_candidate_sample_calibrator_split_v52",
                _V53_CONFIDENCE_REVISION,
                _V54_CONFIDENCE_REVISION,
                _V55_CONFIDENCE_REVISION,
                _V56_CONFIDENCE_REVISION,
                _V57_CONFIDENCE_REVISION,
                _V58_CONFIDENCE_REVISION,
            }:
                required_equal_args += _DEPLOYED_VETO_ROUTING_RESUME_CONTRACT_KEYS
            if str(
                getattr(
                    cfg,
                    "stage_b_dense_duty_confidence_head_gradient_contract",
                    "shared_token_veto_global_absolute_v1",
                )
            ).strip().lower() in {
                "split_token_veto_global_absolute_v2",
                "split_token_veto_global_absolute_joint_clip_v3",
                "split_token_veto_global_trust_veto_v4",
                "split_token_veto_deployed_router_global_absolute_v5",
                "split_token_veto_candidate_absolute_sample_calibrator_v6",
                _V53_CONFIDENCE_HEAD_CONTRACT,
                _V55_CONFIDENCE_HEAD_CONTRACT,
                _V56_CONFIDENCE_HEAD_CONTRACT,
            }:
                required_equal_args += _SPLIT_CONFIDENCE_HEAD_RESUME_CONTRACT_KEYS
            if confidence_revision in {
                "word_veto_candidate_split_tail_aligned_v45",
                "word_veto_candidate_split_positive_tail_v46",
                "word_veto_candidate_split_boundary_routing_v47",
                "word_veto_candidate_split_fpr_active_set_v48",
                "word_veto_candidate_split_global_trust_veto_v49",
                "word_veto_candidate_split_strong_boundary_routing_v50",
                "word_veto_candidate_split_independent_deployed_router_v51",
                "word_veto_candidate_sample_calibrator_split_v52",
                _V53_CONFIDENCE_REVISION,
                _V54_CONFIDENCE_REVISION,
                _V55_CONFIDENCE_REVISION,
                _V56_CONFIDENCE_REVISION,
                _V57_CONFIDENCE_REVISION,
                _V58_CONFIDENCE_REVISION,
            }:
                required_equal_args += _TAIL_ALIGNED_SPLIT_RESUME_CONTRACT_KEYS
            if confidence_revision in {
                "word_veto_candidate_split_fpr_active_set_v48",
                _V58_CONFIDENCE_REVISION,
            }:
                required_equal_args += _FPR_ACTIVE_SET_RESUME_CONTRACT_KEYS
            if confidence_revision == (
                "word_veto_candidate_split_global_trust_veto_v49"
            ):
                required_equal_args += _GLOBAL_TRUST_VETO_RESUME_CONTRACT_KEYS
            if confidence_revision in {
                "word_veto_candidate_split_strong_boundary_routing_v50",
                "word_veto_candidate_split_independent_deployed_router_v51",
                "word_veto_candidate_sample_calibrator_split_v52",
                _V53_CONFIDENCE_REVISION,
                _V54_CONFIDENCE_REVISION,
                _V55_CONFIDENCE_REVISION,
                _V56_CONFIDENCE_REVISION,
                _V57_CONFIDENCE_REVISION,
                _V58_CONFIDENCE_REVISION,
            }:
                required_equal_args += (
                    _STRONG_BOUNDARY_ROUTING_RESUME_CONTRACT_KEYS
                )
            if confidence_revision == _V57_CONFIDENCE_REVISION:
                required_equal_args += (
                    _V57_DEPLOYED_GLOBAL_ABSOLUTE_RESUME_CONTRACT_KEYS
                )
            if str(
                getattr(
                    cfg,
                    "stage_b_dense_duty_confidence_phrase_aggregation",
                    "",
                )
            ).strip().lower() in {
                "trace_activated_word_veto_absolute_cap_v4",
                "trace_activated_word_veto_gated_pool_absolute_cap_v5",
            }:
                required_equal_args += _ABSOLUTE_CAP_RESUME_CONTRACT_KEYS
            if str(
                getattr(
                    cfg,
                    "stage_b_dense_duty_confidence_revision",
                    "",
                )
            ).strip() in {
                "word_veto_gated_pool_carrier_balanced_v7",
                "word_veto_gated_pool_carrier_quarter_v8",
                "word_veto_gated_pool_carrier_pair_v9",
                "word_veto_gated_pool_dual_carrier_pair_v10",
                "word_veto_gated_pool_rank_evidence_v11",
                "word_veto_gated_pool_rank_affine_v12",
                "word_veto_gated_pool_gate_margin_v13",
                "word_veto_gated_pool_carrier_slope_v14",
                "word_veto_gated_pool_carrier_affine_v15",
                "word_veto_gated_pool_tail_ste_v16",
                "word_veto_gated_pool_tail_carrier_v17",
                "word_veto_gated_pool_tail_paired_v18",
                "word_veto_gated_pool_tail_paired_rank_channel_v19",
                "word_veto_gated_pool_tail_paired_signed_rank_pool_v20",
            }:
                required_equal_args += _CARRIER_BALANCED_RESUME_CONTRACT_KEYS
                if str(
                    getattr(
                        cfg,
                        "stage_b_dense_duty_confidence_revision",
                        "",
                    )
                ).strip() in {
                    "word_veto_gated_pool_carrier_pair_v9",
                    "word_veto_gated_pool_dual_carrier_pair_v10",
                    "word_veto_gated_pool_rank_evidence_v11",
                    "word_veto_gated_pool_rank_affine_v12",
                    "word_veto_gated_pool_gate_margin_v13",
                    "word_veto_gated_pool_carrier_slope_v14",
                    "word_veto_gated_pool_carrier_affine_v15",
                    "word_veto_gated_pool_tail_ste_v16",
                    "word_veto_gated_pool_tail_carrier_v17",
                    "word_veto_gated_pool_tail_paired_v18",
                    "word_veto_gated_pool_tail_paired_rank_channel_v19",
                    "word_veto_gated_pool_tail_paired_signed_rank_pool_v20",
                }:
                    required_equal_args += _CARRIER_PAIR_RESUME_CONTRACT_KEYS
                if str(
                    getattr(
                        cfg,
                        "stage_b_dense_duty_confidence_revision",
                        "",
                    )
                ).strip() == "word_veto_gated_pool_dual_carrier_pair_v10":
                    required_equal_args += _DUAL_CARRIER_RESUME_CONTRACT_KEYS
                if str(
                    getattr(
                        cfg,
                        "stage_b_dense_duty_confidence_revision",
                        "",
                    )
                ).strip() in {
                    "word_veto_gated_pool_rank_evidence_v11",
                    "word_veto_gated_pool_rank_affine_v12",
                    "word_veto_gated_pool_gate_margin_v13",
                    "word_veto_gated_pool_carrier_slope_v14",
                    "word_veto_gated_pool_carrier_affine_v15",
                    "word_veto_gated_pool_tail_ste_v16",
                    "word_veto_gated_pool_tail_carrier_v17",
                    "word_veto_gated_pool_tail_paired_v18",
                    "word_veto_gated_pool_tail_paired_rank_channel_v19",
                    "word_veto_gated_pool_tail_paired_signed_rank_pool_v20",
                }:
                    required_equal_args += _RANK_EVIDENCE_RESUME_CONTRACT_KEYS
                    if str(
                        getattr(
                            cfg,
                            "stage_b_dense_duty_confidence_revision",
                            "",
                        )
                    ).strip() in {
                        "word_veto_gated_pool_gate_margin_v13",
                        "word_veto_gated_pool_carrier_slope_v14",
                        "word_veto_gated_pool_carrier_affine_v15",
                        "word_veto_gated_pool_tail_ste_v16",
                        "word_veto_gated_pool_tail_carrier_v17",
                        "word_veto_gated_pool_tail_paired_v18",
                        "word_veto_gated_pool_tail_paired_rank_channel_v19",
                        "word_veto_gated_pool_tail_paired_signed_rank_pool_v20",
                    }:
                        required_equal_args += _GATE_MARGIN_RESUME_CONTRACT_KEYS
                    if str(
                        getattr(
                            cfg,
                            "stage_b_dense_duty_confidence_revision",
                            "",
                        )
                    ).strip() in {
                        "word_veto_gated_pool_tail_ste_v16",
                        "word_veto_gated_pool_tail_carrier_v17",
                        "word_veto_gated_pool_tail_paired_v18",
                        "word_veto_gated_pool_tail_paired_rank_channel_v19",
                        "word_veto_gated_pool_tail_paired_signed_rank_pool_v20",
                    }:
                        required_equal_args += _GATE_GRADIENT_RESUME_CONTRACT_KEYS
                    if str(
                        getattr(
                            cfg,
                            "stage_b_dense_duty_confidence_revision",
                            "",
                        )
                    ).strip() in {
                        "word_veto_gated_pool_tail_carrier_v17",
                        "word_veto_gated_pool_tail_paired_v18",
                        "word_veto_gated_pool_tail_paired_rank_channel_v19",
                        "word_veto_gated_pool_tail_paired_signed_rank_pool_v20",
                    }:
                        required_equal_args += _TAIL_CARRIER_RESUME_CONTRACT_KEYS
        if str(
            getattr(
                cfg,
                "stage_b_dense_duty_confidence_probe_admission_contract",
                "",
            )
        ).strip() in {
            "u300_word_veto_strict1607_v1",
            "u300_word_veto_gate_strict1607_v3",
            "u300_word_veto_absolute_cap_strict1607_v4",
            "u300_word_veto_gated_pool_absolute_cap_strict1607_v5",
            "u300_word_veto_gated_pool_calibrated_strict1607_v6",
            "u300_word_veto_gated_pool_carrier_balanced_strict1607_v7",
            "u300_word_veto_gated_pool_carrier_quarter_strict1607_v8",
            "u300_word_veto_gated_pool_carrier_pair_strict1607_v9",
            "u300_word_veto_gated_pool_dual_carrier_pair_strict1607_v10",
            "u300_word_veto_gated_pool_rank_evidence_strict1607_v11",
            "u300_word_veto_gated_pool_rank_affine_strict1607_v12",
            "u300_word_veto_gated_pool_gate_margin_strict1607_v13",
            "u300_word_veto_gated_pool_carrier_slope_strict1607_v14",
            "u300_word_veto_gated_pool_carrier_affine_strict1607_v15",
            "u300_word_veto_gated_pool_tail_ste_strict1607_v16",
            "u300_word_veto_gated_pool_tail_carrier_strict1607_v17",
            "u300_word_veto_gated_pool_tail_paired_strict1607_v18",
            "u300_word_veto_gated_pool_tail_paired_rank_channel_strict1607_v19",
            "u300_word_veto_gated_pool_tail_paired_signed_rank_pool_strict1607_v20",
            "u400_word_veto_candidate_tn_only_carrier_pair_confidence_strict1607_v42",
            "u400_word_veto_candidate_deployed_routing_confidence_strict1607_v43",
            "u400_word_veto_candidate_split_tail_aligned_confidence_strict1607_v45",
            "u400_word_veto_candidate_split_positive_tail_confidence_strict1607_v46",
            "u400_word_veto_candidate_split_boundary_routing_confidence_strict1607_v47",
            "u400_word_veto_candidate_split_fpr_active_set_confidence_strict1607_v48",
            "u400_word_veto_candidate_split_global_trust_veto_confidence_strict1607_v49",
            "u400_word_veto_candidate_split_strong_boundary_routing_confidence_strict1607_v50",
            "u400_word_veto_candidate_split_independent_deployed_router_confidence_strict1607_v51",
            "u400_word_veto_candidate_sample_calibrator_confidence_strict1607_v52",
            _V53_FORMAL_ADMISSION_CONTRACT,
            _V54_FORMAL_ADMISSION_CONTRACT,
            _V55_FORMAL_ADMISSION_CONTRACT,
            _V56_FORMAL_ADMISSION_CONTRACT,
            _V57_FORMAL_ADMISSION_CONTRACT,
            _V58_FORMAL_ADMISSION_CONTRACT,
        }:
            required_equal_args += _PROBE_ADMISSION_RESUME_CONTRACT_KEYS
        if int(
            getattr(cfg, "stage_b_dense_duty_forward_pack_factor", 1) or 1
        ) > 1:
            required_equal_args = (
                required_equal_args + _PACKED_FORWARD_CONTRACT_KEYS
            )
        if str(
            getattr(
                cfg,
                "stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract",
                "bidirectional_v1",
            )
        ).strip().lower() != "bidirectional_v1":
            required_equal_args += (
                _CARRIER_BALANCED_RESUME_CONTRACT_KEYS
                + _CARRIER_PAIR_RESUME_CONTRACT_KEYS
                + _TAIL_CARRIER_RESUME_CONTRACT_KEYS
                + _CARRIER_PAIR_GRADIENT_RESUME_CONTRACT_KEYS
            )
    drift = {
        key: (saved_args.get(key), getattr(cfg, key, None))
        for key in required_equal_args
        if saved_args.get(key) != getattr(cfg, key, None)
    }
    if drift:
        raise RuntimeError(
            f"dense-duty evaluation configuration drifted from training: {drift}"
        )

    if adapter_contract:
        from util.stage_b_confidence_adapter_migration import (
            validate_confidence_adapter_migration_audit,
        )

        rank_source = validate_confidence_adapter_migration_audit(
            saved_args.get(
                "stage_b_dense_duty_confidence_adapter_migration_audit"
            ),
            source_checkpoint_sha256=str(
                getattr(cfg, "stage_b_dense_duty_rank_source_checkpoint_sha256")
            ),
            source_optimizer_updates=int(
                getattr(cfg, "stage_b_dense_duty_rank_source_optimizer_updates")
            ),
            source_checkpoint_reason=str(
                getattr(cfg, "stage_b_dense_duty_rank_source_checkpoint_reason")
            ),
            rank_sha256=str(
                getattr(cfg, "stage_b_dense_duty_rank_source_rank_sha256")
            ),
            transferred_sha256=str(
                getattr(cfg, "stage_b_dense_duty_rank_source_transferred_sha256")
            ),
        )
        audit["rank_source_checkpoint_audit"] = (
            validate_confidence_adapter_rank_source_audit(
                saved_args.get("stage_b_dense_duty_rank_source_checkpoint_audit"),
                vars(cfg) if hasattr(cfg, "__dict__") else cfg,
            )
        )
    else:
        rank_source = validate_rank_handoff_audit(
            saved_args.get("stage_b_dense_duty_rank_source_checkpoint_audit"),
            execution_scope=evaluation_scope,
            rank_dataset_sha256=getattr(
                cfg, "stage_b_dense_duty_rank_dataset_config_sha256", ""
            ),
            required_optimizer_updates=(
                expected_rank_updates if evaluation_scope == "formal" else None
            ),
            code_source_sha256=(
                current_code["sha256"] if evaluation_scope == "formal" else None
            ),
        )
    audit["evaluation_scope"] = evaluation_scope
    audit["rank_handoff"] = rank_source
    return audit
