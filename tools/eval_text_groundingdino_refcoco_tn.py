#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import build_dataset  # noqa: E402
from groundingdino.util import box_ops  # noqa: E402
from tools.eval_refcoco_stageb import (  # noqa: E402
    _NATIVE_PATCH_GATE_CLIP,
    _NATIVE_PATCH_GATE_MAX_GAP,
    _NATIVE_PATCH_RANK_SCORE_KEY,
    _NATIVE_PATCH_REF_SCORE_CONTRACT,
    _V43_DEPLOYED_ROUTING_GATE_CONTRACT,
    _V43_DEPLOYED_ROUTING_REVISION,
    _V45_SPLIT_TAIL_ALIGNED_REVISION,
    _V46_SPLIT_POSITIVE_TAIL_REVISION,
    _V47_SPLIT_BOUNDARY_ROUTING_REVISION,
    _V48_SPLIT_FPR_ACTIVE_SET_REVISION,
    _V49_SPLIT_GLOBAL_TRUST_VETO_REVISION,
    _V50_SPLIT_STRONG_BOUNDARY_ROUTING_REVISION,
    _V51_SPLIT_INDEPENDENT_DEPLOYED_ROUTER_REVISION,
    _V52_CANDIDATE_SAMPLE_CALIBRATOR_REVISION,
    _V53_FULLTEXT_GLOBAL_ABSOLUTE_REVISION,
    _V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_REVISION,
    _V55_FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_REVISION,
    _V56_DEPLOYMENT_OWNED_GLOBAL_REVISION,
    _V57_DEPLOYED_GLOBAL_BALANCED_ABSOLUTE_REVISION,
    _V58_DEPLOYMENT_OWNED_STABLE_FPR95_ACTIVE_SET_REVISION,
    _V59_DEPLOYMENT_OWNED_QUERY_GLOBAL_REVISION,
    _V60_DEPLOYMENT_OWNED_QUERY_VETO_REVISION,
    _build_split_jsonl,
    _canonical_ref_split_seed_map,
    _ckpt_run_prefix,
    _default_splits,
    _forward as _stage_b_ref_forward,
    _load_canonical_name_maps,
    _load_model,
    _load_phrase_maps,
    _safe_name,
    _slot_scores as _stage_b_ref_slot_scores,
    _uses_stage_b_post_candidate_scorer,
    _validate_native_patch_category_ref_request,
    _validate_v43_deployed_routing_config,
    _validate_v45_split_tail_aligned_config,
    _validate_v46_split_positive_tail_config,
    _validate_v47_split_boundary_routing_config,
    _validate_v48_split_fpr_active_set_config,
    _validate_v49_split_global_trust_veto_config,
    _validate_v50_split_strong_boundary_routing_config,
    _validate_v51_split_independent_deployed_router_config,
    _validate_v52_candidate_sample_calibrator_config,
    _validate_v53_fulltext_global_absolute_config,
    _validate_v54_fulltext_global_absolute_exact_residual_config,
    _validate_v55_fulltext_global_independent_absolute_config,
    _validate_v56_deployment_owned_global_config,
    _validate_v57_deployed_global_balanced_absolute_config,
    _validate_v58_deployment_owned_stable_fpr95_active_set_config,
    _validate_v59_deployment_owned_query_global_config,
    _validate_v60_deployment_owned_query_veto_config,
    _validate_v61_full_decoder_verifier_config,
    _validate_v62_patch_softmin_veto_config,
    _validate_c1_candidate_complete_trace_config,
    _validate_c2_candidate_complete_trace_config,
    _verify_v39_immutable_archived_diagnostic_files,
    _verify_v40_immutable_archived_diagnostic_files,
    _verify_v41_immutable_archived_diagnostic_files,
    _verify_v42_immutable_archived_diagnostic_files,
)
from tools.eval_stagea_patch_checkpoints import _set_seed  # noqa: E402
from tools.eval_stageb_tn_val import (  # noqa: E402
    _build_tn_eval_jsonl,
    _forward_pair as _stage_b_tn_forward_pair,
    _is_stage_b_u0_model,
    _prepare_stage_b_u0_patch_batch,
    _make_datasetinfo as _make_tn_datasetinfo,
    _slot_scores as _stage_b_tn_slot_scores,
    _validate_adapter_tn_eval_manifest,
)
from tools.stageb_eval_holdout import load_holdout_keys  # noqa: E402
from tools.merge_stageb_gdino_adapter_eval import (  # noqa: E402
    CONTRACT_SCHEMA as MERGED_EVAL_CONTRACT_SCHEMA,
    LINEAGE_SCHEMA as MERGED_EVAL_LINEAGE_SCHEMA,
    verify_merged_eval_checkpoint,
)
from tools.stageb_gdino_adapter_probe_audit import (  # noqa: E402
    file_record,
    load_checkpoint,
)
from tools.stageb_eval_records import (  # noqa: E402
    EvalManifest,
    extract_adapter_tn_pair_captions,
    load_eval_manifest,
    make_eval_record,
    tn_manifest_binding_summary_fields,
    validate_eval_manifest_batch_alignment as _validate_eval_manifest_batch_alignment,
    write_eval_records,
)
from tools.stageb_screen_calibration import (  # noqa: E402
    DEFAULT_AUDIT as SCREEN_CALIBRATION_AUDIT,
    ScreenCalibrationBinding,
    build_manifest as build_screen_calibration_manifest,
    meta_rows as screen_calibration_meta_rows,
    summary_fields as screen_calibration_summary_fields,
)
from tools.stageb_table_b_matched_eval_surface import (  # noqa: E402
    DECLARED_SCOPE as MATCHED_EVAL_SCOPE,
    MatchedEvalSurfaceBinding,
    SURFACE_ROW_SCHEMA as MATCHED_EVAL_ROW_SCHEMA,
    load_binding as load_matched_eval_surface_binding,
    meta_rows as matched_eval_surface_meta_rows,
    summary_fields as matched_eval_surface_summary_fields,
)
from tools.stageb_ref_split_contract import REF_SPLITS as CANONICAL_REF_SPLITS  # noqa: E402
from util import misc as utils  # noqa: E402
from util.slconfig import SLConfig  # noqa: E402


MERGED_EVAL_SUMMARY_FIELDS = (
    "merged_eval_checkpoint_sha256",
    "merged_eval_contract_schema",
    "merged_eval_base_tensor_sha256",
    "merged_eval_rank_tensor_sha256",
    "merged_eval_confidence_tensor_sha256",
    "merged_eval_full_model_tensor_sha256",
    "merged_eval_rank_source_checkpoint_sha256",
    "merged_eval_confidence_source_checkpoint_sha256",
)
CATEGORY_GATE_SWEEP_CONTRACT = (
    "stageb-u2-category-gate-sweep-lexicographic-v1"
)
CATEGORY_GATE_BASE_EXPERT_SWEEP_CONTRACT = (
    "stageb-u2-category-gate-sweep-base-expert-lexicographic-v1"
)
_DATA_DRIVEN_RANK_SCORE_KEY = "stage_b_data_driven_rank_score"
_DATA_DRIVEN_CONFIDENCE_SCORE_KEY = "stage_b_data_driven_confidence_score"
_PARTIAL_CONFIDENCE_CONFIG = (
    REPO_ROOT
    / "config/ablations/cfg_stageb_dense_duty_confidence_adapter_20260730.py"
)
_VETO_PROBE_CONFIDENCE_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_veto_probe_20260730.py"
)
_VETO_GATE_PROBE_CONFIDENCE_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_veto_gate_probe_20260731.py"
)
_VETO_CAP_PROBE_CONFIDENCE_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_veto_cap_probe_20260731.py"
)
_VETO_GATED_POOL_PROBE_CONFIDENCE_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_probe_20260731.py"
)
_VETO_GATED_POOL_CALIBRATED_PROBE_CONFIDENCE_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_calibrated_probe_20260731.py"
)
_VETO_GATED_POOL_CARRIER_BALANCED_PROBE_CONFIDENCE_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_carrier_balanced_probe_20260731.py"
)
_VETO_GATED_POOL_CARRIER_QUARTER_PROBE_CONFIDENCE_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_carrier_quarter_probe_20260731.py"
)
_VETO_GATED_POOL_CARRIER_PAIR_PROBE_CONFIDENCE_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_carrier_pair_probe_20260731.py"
)
_VETO_GATED_POOL_CARRIER_AFFINE_U0050_CONFIDENCE_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_carrier_affine_probe_u0050_20260731.py"
)
_VETO_GATED_POOL_TAIL_CARRIER_U0050_CONFIDENCE_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_tail_carrier_probe_u0050_20260731.py"
)
_VETO_GATED_POOL_TAIL_PAIRED_U0050_CONFIDENCE_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_tail_paired_probe_u0050_20260731.py"
)
_VETO_GATED_POOL_TAIL_PAIRED_U0100_CONFIDENCE_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_tail_paired_probe_u0100_20260731.py"
)
_VETO_GATED_POOL_TAIL_PAIRED_RANK_CHANNEL_U0050_CONFIDENCE_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_tail_paired_rank_channel_probe_u0050_20260731.py"
)
_VETO_GATED_POOL_TAIL_PAIRED_SIGNED_RANK_POOL_U0050_CONFIDENCE_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_tail_paired_signed_rank_pool_probe_u0050_20260731.py"
)
_VETO_CONTINUOUS_RESIDUAL_U0050_CONFIDENCE_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_veto_continuous_residual_probe_u0050_20260731.py"
)
_VETO_MONOTONE_DEPTH_U0050_CONFIDENCE_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_veto_monotone_depth_probe_u0050_20260731.py"
)
_VETO_TOKEN_CONDITIONED_U0050_CONFIDENCE_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_veto_token_conditioned_probe_u0050_20260731.py"
)
_VETO_TOKEN_CONDITIONED_U0300_CONFIDENCE_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_veto_token_conditioned_probe_u0300_20260731.py"
)
_COMPLEMENTARY_TRUST_VETO_U0300_CONFIDENCE_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_complementary_trust_veto_probe_u0300_20260731.py"
)
_UNGATED_MONOTONE_TAIL_VETO_U0300_CONFIDENCE_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_ungated_monotone_tail_veto_probe_u0300_20260731.py"
)
_FLOOR_GATED_MONOTONE_TAIL_VETO_U0300_CONFIDENCE_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_floor_gated_monotone_tail_veto_probe_u0300_20260731.py"
)
_INDEPENDENT_ABSOLUTE_CONFIDENCE_U0300_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_independent_absolute_probe_u0300_20260731.py"
)
_CROSS_ATTENTION_ABSOLUTE_CONFIDENCE_U0300_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_cross_attention_absolute_probe_u0300_20260731.py"
)
_CANDIDATE_ABSOLUTE_CONFIDENCE_U0300_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_absolute_probe_u0300_20260731.py"
)
_CANDIDATE_ABSOLUTE_CONFIDENCE_U0600_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_absolute_probe_u0600_20260731.py"
)
_CANDIDATE_CALIBRATED_CONFIDENCE_U0400_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_calibrated_probe_u0400_20260731.py"
)
_CANDIDATE_NORMALIZED_CONFIDENCE_U0400_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_normalized_probe_u0400_20260731.py"
)
_CANDIDATE_ASYMMETRIC_CONFIDENCE_U0400_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_asymmetric_probe_u0400_20260731.py"
)
_CANDIDATE_Q05_CONFIDENCE_U0400_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_q05_probe_u0400_20260801.py"
)
_CANDIDATE_TAIL_BALANCED_CONFIDENCE_U0400_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_tail_balanced_probe_u0400_20260801.py"
)
_CANDIDATE_TAIL_QUARTER_CONFIDENCE_U0400_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_tail_quarter_probe_u0400_20260801.py"
)
_CANDIDATE_TAIL_BOUNDED_CONFIDENCE_U0400_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_tail_bounded_probe_u0400_20260801.py"
)
_CANDIDATE_TAIL_ELEMENTWISE_CONFIDENCE_U0400_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_tail_elementwise_probe_u0400_20260801.py"
)
_CANDIDATE_GATE_ZERO_OFFSET_CONFIDENCE_U0400_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_gate_zero_offset_probe_u0400_20260801.py"
)
_CANDIDATE_HARDEST_EDIT_CONFIDENCE_U0400_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_hardest_edit_probe_u0400_20260801.py"
)
_CANDIDATE_ROLE_COMPLETE_CARRIER_CONFIDENCE_U0400_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_role_complete_carrier_probe_u0400_20260801.py"
)
_CANDIDATE_TN_ONLY_CARRIER_PAIR_CONFIDENCE_U0400_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_tn_only_carrier_pair_probe_u0400_20260801.py"
)
_CANDIDATE_DEPLOYED_ROUTING_CONFIDENCE_U0400_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_deployed_routing_probe_u0400_20260801.py"
)
_CANDIDATE_SPLIT_HEADS_CONFIDENCE_U0400_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_split_heads_probe_u0400_20260801.py"
)
_CANDIDATE_SPLIT_TAIL_ALIGNED_CONFIDENCE_U0400_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_"
    "candidate_split_tail_aligned_probe_u0400_20260801.py"
)
_CANDIDATE_SPLIT_POSITIVE_TAIL_CONFIDENCE_U0400_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_"
    "candidate_split_positive_tail_probe_u0400_20260801.py"
)
_CANDIDATE_SPLIT_BOUNDARY_ROUTING_CONFIDENCE_U0400_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_"
    "candidate_split_boundary_routing_probe_u0400_20260801.py"
)
_CANDIDATE_SPLIT_FPR_ACTIVE_SET_CONFIDENCE_U0400_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_"
    "candidate_split_fpr_active_set_probe_u0400_20260801.py"
)
_CANDIDATE_SPLIT_GLOBAL_TRUST_VETO_CONFIDENCE_U0400_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_"
    "candidate_split_global_trust_veto_probe_u0400_20260801.py"
)
_CANDIDATE_SPLIT_STRONG_BOUNDARY_ROUTING_CONFIDENCE_U0400_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_"
    "candidate_split_strong_boundary_routing_probe_u0400_20260801.py"
)
_CANDIDATE_SPLIT_INDEPENDENT_DEPLOYED_ROUTER_CONFIDENCE_U0400_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_"
    "candidate_split_independent_deployed_router_probe_u0400_20260802.py"
)
_CANDIDATE_SAMPLE_CALIBRATOR_CONFIDENCE_U0400_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_"
    "candidate_sample_calibrator_probe_u0400_20260802.py"
)
_FULLTEXT_GLOBAL_ABSOLUTE_CONFIDENCE_U0400_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_"
    "fulltext_global_absolute_probe_u0400_20260802.py"
)
_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_CONFIDENCE_U0400_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_"
    "fulltext_global_absolute_exact_residual_probe_u0400_20260802.py"
)
_FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_CONFIDENCE_U0400_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_"
    "fulltext_global_independent_absolute_probe_u0400_20260802.py"
)
_DEPLOYMENT_OWNED_GLOBAL_CONFIDENCE_U0400_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_deployment_owned_global_"
    "probe_u0400_20260802.py"
)
_DEPLOYED_GLOBAL_BALANCED_ABSOLUTE_CONFIDENCE_U0400_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_deployed_global_"
    "balanced_absolute_probe_u0400_20260802.py"
)
_DEPLOYMENT_OWNED_GLOBAL_STABLE_FPR95_ACTIVE_SET_CONFIDENCE_U0400_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_deployment_owned_global_"
    "stable_fpr95_active_set_probe_u0400_20260802.py"
)
_DEPLOYMENT_OWNED_QUERY_GLOBAL_CONFIDENCE_U0400_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_deployment_owned_query_global_"
    "probe_u0400_20260802.py"
)
_DEPLOYMENT_OWNED_QUERY_VETO_CONFIDENCE_U0400_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_deployment_owned_query_veto_"
    "probe_u0400_20260802.py"
)
_FULL_DECODER_VERIFIER_CONFIDENCE_U0400_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_full_decoder_verifier_"
    "probe_u0400_20260803.py"
)
_FULL_DECODER_PATCH_SOFTMIN_VETO_CONFIDENCE_U0400_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_full_decoder_patch_softmin_veto_"
    "probe_u0400_20260803.py"
)
_CANDIDATE_COMPLETE_TRACE_C1_CONFIDENCE_U0400_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_candidate_complete_trace_c1_"
    "probe_u0400_20260803.py"
)
_CANDIDATE_COMPLETE_TRACE_C2_CONFIDENCE_U0400_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_candidate_complete_trace_c2_"
    "probe_u0400_20260803.py"
)
_V39_IMMUTABLE_ARCHIVED_SNAPSHOT_ROOT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/"
    "dense_duty_adapter_candidate_gate_zero_offset_highmem_20260801/"
    "probe/intermediate_snapshots"
)
_V39_IMMUTABLE_ARCHIVED_SNAPSHOT_PATHS = {
    update: _V39_IMMUTABLE_ARCHIVED_SNAPSHOT_ROOT
    / f"u{update:06d}/checkpoint_iter.pth"
    for update in (100, 200, 300)
}
_V39_IMMUTABLE_ARCHIVED_SCREEN_ROOT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/"
    "dense_duty_adapter_candidate_gate_zero_offset_highmem_20260801/"
    "probe_evaluation/immutable_archived_snapshot_screen"
)
_V39_IMMUTABLE_ARCHIVED_SCREEN_OUTPUTS = {
    update: _V39_IMMUTABLE_ARCHIVED_SCREEN_ROOT
    / f"u{update:06d}_strict1607"
    for update in (100, 200, 300)
}
_V40_IMMUTABLE_ARCHIVED_SNAPSHOT_ROOT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/"
    "dense_duty_adapter_candidate_hardest_edit_highmem_20260801/"
    "probe/intermediate_snapshots"
)
_V40_IMMUTABLE_ARCHIVED_SNAPSHOT_PATHS = {
    update: _V40_IMMUTABLE_ARCHIVED_SNAPSHOT_ROOT
    / f"u{update:06d}/checkpoint_iter.pth"
    for update in (100, 200, 300)
}
_V40_IMMUTABLE_ARCHIVED_SCREEN_ROOT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/"
    "dense_duty_adapter_candidate_hardest_edit_highmem_20260801/"
    "probe_evaluation/immutable_archived_snapshot_screen"
)
_V40_IMMUTABLE_ARCHIVED_SCREEN_OUTPUTS = {
    update: _V40_IMMUTABLE_ARCHIVED_SCREEN_ROOT
    / f"u{update:06d}_strict1607"
    for update in (100, 200, 300)
}
_V41_IMMUTABLE_ARCHIVED_SNAPSHOT_ROOT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/"
    "dense_duty_adapter_candidate_role_complete_carrier_highmem_20260801/"
    "probe/intermediate_snapshots"
)
_V41_IMMUTABLE_ARCHIVED_SNAPSHOT_PATHS = {
    update: _V41_IMMUTABLE_ARCHIVED_SNAPSHOT_ROOT
    / f"u{update:06d}/checkpoint_iter.pth"
    for update in (100, 200, 300)
}
_V41_IMMUTABLE_ARCHIVED_SCREEN_ROOT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/"
    "dense_duty_adapter_candidate_role_complete_carrier_highmem_20260801/"
    "probe_evaluation/immutable_archived_snapshot_screen"
)
_V41_IMMUTABLE_ARCHIVED_SCREEN_OUTPUTS = {
    update: _V41_IMMUTABLE_ARCHIVED_SCREEN_ROOT
    / f"u{update:06d}_strict1607"
    for update in (100, 200, 300)
}
_V42_IMMUTABLE_ARCHIVED_SNAPSHOT_ROOT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/"
    "dense_duty_adapter_candidate_tn_only_carrier_pair_highmem_20260801/"
    "probe"
)
_V42_IMMUTABLE_ARCHIVED_SNAPSHOT_PATHS = {
    100: _V42_IMMUTABLE_ARCHIVED_SNAPSHOT_ROOT
    / "intermediate_snapshots/u000100/checkpoint_iter.pth",
    200: _V42_IMMUTABLE_ARCHIVED_SNAPSHOT_ROOT
    / "intermediate_snapshots/u000200/checkpoint_iter.pth",
    300: _V42_IMMUTABLE_ARCHIVED_SNAPSHOT_ROOT
    / "intermediate_snapshots/u000300/checkpoint_iter.pth",
    400: _V42_IMMUTABLE_ARCHIVED_SNAPSHOT_ROOT
    / "u000400_fresh/checkpoint_iter.pth",
}
_CANDIDATE_SET_ATTENTION_CONFIDENCE_U0400_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_candidate_set_attention_probe_u0400_20260731.py"
)
_PARTIAL_CONFIDENCE_STRICT_ROOT = (
    REPO_ROOT
    / "data/eval_manifests/stageb_vlm_verified_strict_ann_umd_val_20260711"
)
_PARTIAL_CONFIDENCE_TN_SPECS = {
    "strict2031": {
        "path": _PARTIAL_CONFIDENCE_STRICT_ROOT / "eval_manifest.jsonl",
        "sha256": "0e47763c01178d63ee22430a6c93d4fc6a210848d43f32aafbb2e6cd7243e918",
    },
    "strict1607": {
        "path": (
            _PARTIAL_CONFIDENCE_STRICT_ROOT
            / "semantic_stageb_union_image_disjoint_manifest.jsonl"
        ),
        "sha256": "f2dc97d58884b8de3ae2c8b4cefd281432e15c8952b23b5e0252eb8e5be36d25",
    },
}


def _requested_ref_split_seed_map(
    requested: Sequence[str], base_seed: int
) -> Dict[str, int]:
    """Keep each Ref split's RNG stream stable across subset evaluations."""

    canonical = _canonical_ref_split_seed_map(int(base_seed))
    unknown = [str(name) for name in requested if str(name) not in canonical]
    if unknown:
        raise KeyError(f"Unknown Ref split names for seed mapping: {unknown}")
    return {str(name): canonical[str(name)] for name in requested}


def _merged_eval_checkpoint_summary_fields(
    cfg, checkpoint: str | Path
) -> Dict[str, str]:
    """Verify and extract compact provenance for a merged eval-only checkpoint."""

    if not bool(
        getattr(cfg, "stage_b_gdino_adapter_merged_eval_only", False)
    ):
        return {}
    if not bool(getattr(cfg, "stage_b_gdino_score_adapter", False)):
        raise RuntimeError(
            "merged eval-only config must enable stage_b_gdino_score_adapter"
        )
    checkpoint_path = Path(checkpoint).resolve()
    receipt = verify_merged_eval_checkpoint(checkpoint_path)
    if (
        receipt.get("status") != "verified"
        or receipt.get("schema") != MERGED_EVAL_CONTRACT_SCHEMA
    ):
        raise RuntimeError("merged eval verifier returned an invalid receipt")
    payload = load_checkpoint(checkpoint_path)
    if set(payload) != {"model", "lineage", "contract"}:
        raise RuntimeError(
            "verified merged eval checkpoint lost its exact model/lineage/contract payload"
        )
    lineage = payload.get("lineage")
    contract = payload.get("contract")
    if (
        not isinstance(lineage, Mapping)
        or lineage.get("schema") != MERGED_EVAL_LINEAGE_SCHEMA
        or not isinstance(contract, Mapping)
        or contract.get("schema") != MERGED_EVAL_CONTRACT_SCHEMA
    ):
        raise RuntimeError("verified merged eval checkpoint schema drifted")

    def source_sha256(name: str) -> Any:
        source = lineage.get(name)
        checkpoint_record = (
            source.get("checkpoint") if isinstance(source, Mapping) else None
        )
        return (
            checkpoint_record.get("sha256")
            if isinstance(checkpoint_record, Mapping)
            else None
        )

    checkpoint_record = receipt.get("checkpoint")
    if (
        not isinstance(checkpoint_record, Mapping)
        or file_record(checkpoint_path) != dict(checkpoint_record)
    ):
        raise RuntimeError("merged eval checkpoint changed after verification")
    values: Dict[str, Any] = {
        "merged_eval_checkpoint_sha256": (
            checkpoint_record.get("sha256")
        ),
        "merged_eval_contract_schema": contract.get("schema"),
        "merged_eval_base_tensor_sha256": contract.get("base_tensor_sha256"),
        "merged_eval_rank_tensor_sha256": contract.get("rank_tensor_sha256"),
        "merged_eval_confidence_tensor_sha256": contract.get(
            "confidence_tensor_sha256"
        ),
        "merged_eval_full_model_tensor_sha256": contract.get(
            "full_model_tensor_sha256"
        ),
        "merged_eval_rank_source_checkpoint_sha256": source_sha256(
            "rank_source"
        ),
        "merged_eval_confidence_source_checkpoint_sha256": source_sha256(
            "confidence_source"
        ),
    }
    invalid = {
        key: value
        for key, value in values.items()
        if not isinstance(value, str)
        or (
            key != "merged_eval_contract_schema"
            and re.fullmatch(r"[0-9a-f]{64}", value) is None
        )
    }
    if invalid or set(values) != set(MERGED_EVAL_SUMMARY_FIELDS):
        raise RuntimeError(
            "verified merged eval checkpoint has incomplete compact provenance: "
            f"{sorted(invalid)}"
        )
    if values["merged_eval_contract_schema"] != MERGED_EVAL_CONTRACT_SCHEMA:
        raise RuntimeError("merged eval contract schema changed after verification")
    return {key: str(values[key]) for key in MERGED_EVAL_SUMMARY_FIELDS}


def _load_model_with_checkpoint_contract(
    cfg, checkpoint: str | Path, device: torch.device
):
    summary_fields = _merged_eval_checkpoint_summary_fields(cfg, checkpoint)
    model = _load_model(cfg, str(checkpoint), device)
    if bool(
        getattr(cfg, "stage_b_dense_duty_confidence_full_decoder_verifier", False)
    ):
        _attach_full_decoder_decomposition_diagnostics(model)
    return model, summary_fields


def _attach_full_decoder_decomposition_diagnostics(model: torch.nn.Module) -> None:
    """Expose scorer-only full-verifier diagnostics without changing train code.

    Terminal probes validate the checkpoint's exact recursive training-source
    closure before this helper runs.  Keeping this evaluator-only hook outside
    ``models/`` preserves that fail-closed check while making the pool/veto
    decomposition auditable in per-example records.
    """
    scorer = getattr(model, "stage_b_fixed_text_scorer", None)
    if not isinstance(scorer, torch.nn.Module) or not bool(
        getattr(scorer, "confidence_full_decoder_verifier", False)
    ):
        raise RuntimeError("full-decoder decomposition requires a verifier scorer")
    captured: Dict[str, Mapping[str, torch.Tensor]] = {}

    def capture(_module, _inputs, output):
        if not isinstance(output, Mapping):
            raise RuntimeError("full-decoder scorer returned a non-mapping output")
        captured["scorer"] = output

    scorer.register_forward_hook(capture)
    original_forward = model.forward

    def forward_with_decomposition(*args, **kwargs):
        captured.clear()
        output = original_forward(*args, **kwargs)
        scorer_output = captured.pop("scorer", None)
        if not isinstance(output, dict) or not isinstance(scorer_output, Mapping):
            raise RuntimeError("full-decoder decomposition hook did not observe a forward")
        mapping = {
            "final_deployed_query_veto_depth": (
                "stage_b_dense_duty_deployed_query_veto_depth"
            ),
            "final_deployed_query_veto_gate": (
                "stage_b_dense_duty_deployed_query_veto_gate"
            ),
        }
        for scorer_key, output_key in mapping.items():
            value = scorer_output.get(scorer_key)
            if not torch.is_tensor(value):
                raise RuntimeError(
                    f"full-decoder decomposition is missing {scorer_key!r}"
                )
            output[output_key] = value
        return output

    model.forward = forward_with_decomposition


def _bind_checkpoint_summary_fields(
    row: Dict[str, Any], fields: Optional[Mapping[str, str]]
) -> None:
    if not fields:
        return
    if set(fields) != set(MERGED_EVAL_SUMMARY_FIELDS):
        raise RuntimeError("merged eval summary provenance field set is incomplete")
    collisions = sorted(set(row).intersection(fields))
    if collisions:
        raise RuntimeError(
            f"summary row already contains merged checkpoint fields: {collisions}"
        )
    row.update(fields)


def _evaluation_summary_provenance(
    *,
    cfg,
    args: argparse.Namespace,
    checkpoint: str | Path,
    data_root: Path,
) -> Dict[str, Any]:
    config_path = Path(args.config).expanduser().resolve(strict=True)
    checkpoint_path = Path(checkpoint).expanduser().resolve(strict=True)
    provenance = {
        "config": str(config_path),
        "config_sha256": str(file_record(config_path)["sha256"]),
        "checkpoint_sha256": str(file_record(checkpoint_path)["sha256"]),
        "amp": bool(args.amp),
        "device": str(args.device),
        "data_root": str(data_root.resolve(strict=True)),
    }
    if bool(getattr(cfg, "stage_b_dense_duty", False)):
        provenance.update(
            {
                "ref_score_key": "stage_b_v15_dense_rank_score",
                "tn_score_key": "stage_b_v7_final_score",
                "score_ownership": str(
                    getattr(
                        cfg,
                        "stage_b_v22_score_ownership",
                        "independent_decoders_two_phase",
                    )
                ),
            }
        )
        if bool(
            getattr(
                cfg,
                "stage_b_dense_duty_partial_rank_diagnostic",
                False,
            )
        ):
            provenance.update(
                {
                    "diagnostic_only": True,
                    "formal_gate_eligible": False,
                    "confidence_evaluated": False,
                    "training_phase": "rank",
                    "optimizer_updates": int(
                        getattr(
                            cfg,
                            "stage_b_dense_duty_partial_rank_diagnostic_optimizer_updates",
                        )
                    ),
                    "checkpoint_reason": str(
                        getattr(
                            cfg,
                            "stage_b_dense_duty_partial_rank_diagnostic_checkpoint_reason",
                        )
                    ),
                    "torch_multiprocessing_sharing_strategy": (
                        torch.multiprocessing.get_sharing_strategy()
                    ),
                }
            )
        elif bool(
            getattr(
                cfg,
                "stage_b_dense_duty_partial_confidence_diagnostic",
                False,
            )
        ):
            optimizer_updates = int(
                getattr(
                    cfg,
                    "stage_b_dense_duty_partial_confidence_diagnostic_optimizer_updates",
                )
            )
            expected_optimizer_updates = int(
                getattr(
                    cfg,
                    "stage_b_dense_duty_partial_confidence_diagnostic_expected_optimizer_updates",
                )
            )
            provenance.update(
                {
                    "diagnostic_only": True,
                    "formal_gate_eligible": False,
                    "confidence_evaluated": True,
                    "training_phase": "confidence",
                    "terminal_checkpoint": bool(
                        getattr(
                            cfg,
                            "stage_b_dense_duty_partial_confidence_diagnostic_terminal_checkpoint",
                            False,
                        )
                    ),
                    "optimizer_updates": optimizer_updates,
                    "expected_optimizer_updates": expected_optimizer_updates,
                    "remaining_optimizer_updates": (
                        expected_optimizer_updates - optimizer_updates
                    ),
                    "checkpoint_reason": str(
                        getattr(
                            cfg,
                            "stage_b_dense_duty_partial_confidence_diagnostic_checkpoint_reason",
                        )
                    ),
                    "torch_multiprocessing_sharing_strategy": (
                        torch.multiprocessing.get_sharing_strategy()
                    ),
                }
            )
            if bool(
                getattr(
                    cfg,
                    "stage_b_dense_duty_immutable_v39_archived_snapshot_diagnostic",
                    False,
                )
            ):
                immutable_audit = getattr(
                    cfg,
                    "stage_b_dense_duty_immutable_v39_archived_snapshot_audit",
                    None,
                )
                immutable_provenance = (
                    immutable_audit.get("immutable_archived_provenance")
                    if isinstance(immutable_audit, Mapping)
                    else None
                )
                if (
                    not isinstance(immutable_audit, Mapping)
                    or immutable_audit.get(
                        "immutable_archived_snapshot_diagnostic"
                    )
                    is not True
                    or not isinstance(immutable_provenance, Mapping)
                    or immutable_provenance.get("optimizer_updates")
                    != optimizer_updates
                ):
                    raise RuntimeError(
                        "v39 immutable archived diagnostic provenance is incomplete"
                    )
                provenance.update(
                    {
                        "immutable_v39_archived_snapshot_diagnostic": True,
                        "immutable_archived_snapshot_provenance": copy.deepcopy(
                            dict(immutable_provenance)
                        ),
                    }
                )
            elif bool(
                getattr(
                    cfg,
                    "stage_b_dense_duty_immutable_v40_archived_snapshot_diagnostic",
                    False,
                )
            ):
                immutable_audit = getattr(
                    cfg,
                    "stage_b_dense_duty_immutable_v40_archived_snapshot_audit",
                    None,
                )
                immutable_provenance = (
                    immutable_audit.get("immutable_archived_provenance")
                    if isinstance(immutable_audit, Mapping)
                    else None
                )
                if (
                    not isinstance(immutable_audit, Mapping)
                    or immutable_audit.get(
                        "immutable_archived_snapshot_diagnostic"
                    )
                    is not True
                    or immutable_audit.get("immutable_archived_snapshot_version")
                    != "v40"
                    or not isinstance(immutable_provenance, Mapping)
                    or immutable_provenance.get("schema")
                    != "pivot.stageb.v40_immutable_archived_diagnostic/v1"
                    or immutable_provenance.get("optimizer_updates")
                    != optimizer_updates
                ):
                    raise RuntimeError(
                        "v40 immutable archived diagnostic provenance is incomplete"
                    )
                provenance.update(
                    {
                        "immutable_v40_archived_snapshot_diagnostic": True,
                        "immutable_archived_snapshot_provenance": copy.deepcopy(
                            dict(immutable_provenance)
                        ),
                    }
                )
            elif bool(
                getattr(
                    cfg,
                    "stage_b_dense_duty_immutable_v41_archived_snapshot_diagnostic",
                    False,
                )
            ):
                immutable_audit = getattr(
                    cfg,
                    "stage_b_dense_duty_immutable_v41_archived_snapshot_audit",
                    None,
                )
                immutable_provenance = (
                    immutable_audit.get("immutable_archived_provenance")
                    if isinstance(immutable_audit, Mapping)
                    else None
                )
                if (
                    not isinstance(immutable_audit, Mapping)
                    or immutable_audit.get(
                        "immutable_archived_snapshot_diagnostic"
                    )
                    is not True
                    or immutable_audit.get("immutable_archived_snapshot_version")
                    != "v41"
                    or not isinstance(immutable_provenance, Mapping)
                    or immutable_provenance.get("schema")
                    != "pivot.stageb.v41_immutable_archived_diagnostic/v1"
                    or immutable_provenance.get("optimizer_updates")
                    != optimizer_updates
                ):
                    raise RuntimeError(
                        "v41 immutable archived diagnostic provenance is incomplete"
                    )
                provenance.update(
                    {
                        "immutable_v41_archived_snapshot_diagnostic": True,
                        "immutable_archived_snapshot_provenance": copy.deepcopy(
                            dict(immutable_provenance)
                        ),
                    }
                )
            elif bool(
                getattr(
                    cfg,
                    "stage_b_dense_duty_immutable_v42_archived_snapshot_diagnostic",
                    False,
                )
            ):
                immutable_audit = getattr(
                    cfg,
                    "stage_b_dense_duty_immutable_v42_archived_snapshot_audit",
                    None,
                )
                immutable_provenance = (
                    immutable_audit.get("immutable_archived_provenance")
                    if isinstance(immutable_audit, Mapping)
                    else None
                )
                if (
                    not isinstance(immutable_audit, Mapping)
                    or immutable_audit.get(
                        "immutable_archived_snapshot_diagnostic"
                    )
                    is not True
                    or immutable_audit.get("immutable_archived_snapshot_version")
                    != "v42"
                    or not isinstance(immutable_provenance, Mapping)
                    or immutable_provenance.get("schema")
                    != "pivot.stageb.v42_immutable_archived_diagnostic/v1"
                    or immutable_provenance.get("optimizer_updates")
                    != optimizer_updates
                ):
                    raise RuntimeError(
                        "v42 immutable archived diagnostic provenance is incomplete"
                    )
                provenance.update(
                    {
                        "immutable_v42_archived_snapshot_diagnostic": True,
                        "immutable_archived_snapshot_provenance": copy.deepcopy(
                            dict(immutable_provenance)
                        ),
                    }
                )
    elif bool(getattr(cfg, "stage_b_gdino_score_adapter", False)):
        ref_top1_guard = bool(
            getattr(cfg, "stage_b_gdino_ref_top1_guard", False)
        )
        provenance.update(
            {
                "ref_score_key": (
                    "stage_b_gdino_ref_safe_rank_score"
                    if ref_top1_guard
                    else "stage_b_gdino_rank_score"
                ),
                "tn_score_key": "stage_b_gdino_confidence_score",
                "score_ownership": (
                    "shared_frozen_gdino_trunk_independent_rank_confidence_adapters_"
                    "b58_top1_anchored_rank_tail"
                    if ref_top1_guard
                    else "shared_frozen_gdino_trunk_independent_rank_confidence_adapters"
                ),
            }
        )
    return provenance


def _bind_evaluation_summary_provenance(
    row: Dict[str, Any], provenance: Mapping[str, Any]
) -> None:
    values = dict(provenance)
    collisions = sorted(set(row).intersection(values))
    if collisions:
        raise RuntimeError(
            f"evaluation summary provenance fields already exist: {collisions}"
        )
    row.update(values)


def _validate_direct_prebuilt_tn_args(args: argparse.Namespace) -> None:
    """Fail closed before direct-prebuilt evaluation creates any output."""

    if not bool(getattr(args, "direct_prebuilt_tn", False)):
        return
    errors = []
    if bool(getattr(args, "screen_calibration_manifest", False)):
        errors.append("screen calibration mode is mutually exclusive")
    if bool(getattr(args, "skip_tn", False)) or not bool(
        getattr(args, "skip_ref", False)
    ):
        errors.append("direct-prebuilt mode requires TN-only evaluation")
    if not getattr(args, "tn_jsonl", None) or not getattr(
        args, "direct_prebuilt_tn_binding", None
    ):
        errors.append("manifest and binding are both required")
    if getattr(args, "holdout_level", None) != "none":
        errors.append("holdout filtering is forbidden")
    if int(getattr(args, "max_tn_batches", -1)) != 0:
        errors.append("partial TN batches are forbidden")
    if bool(getattr(args, "no_per_example_records", False)):
        errors.append("full per-example records are required")
    if int(getattr(args, "candidate_count_control", 0)) != 0:
        errors.append("candidate-count subsampling is forbidden")
    if len(list(getattr(args, "ckpts", ()))) != 1:
        errors.append("exactly one checkpoint is required")
    output_dir = Path(str(getattr(args, "output_dir", ""))).expanduser()
    if output_dir.exists():
        errors.append("output directory must be fresh")
    if errors:
        raise ValueError("invalid --direct_prebuilt_tn contract: " + "; ".join(errors))


def _validate_partial_dense_duty_rank_diagnostic_args(
    args: argparse.Namespace, cfg
) -> None:
    if not bool(getattr(args, "partial_dense_duty_rank_diagnostic", False)):
        return
    expected_config = (
        REPO_ROOT
        / "config/ablations/cfg_stageb_dense_duty_rank_20260728.py"
    ).resolve(strict=True)
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    errors = []
    if Path(args.config).expanduser().resolve(strict=True) != expected_config:
        errors.append("the fixed dense-duty rank config is required")
    if not bool(getattr(cfg, "stage_b_dense_duty", False)) or str(
        getattr(cfg, "stage_b_dense_duty_phase", "")
    ) != "rank":
        errors.append("config must build the dense-duty rank phase")
    if len(args.ckpts) != 1:
        errors.append("exactly one rank checkpoint is required")
    if not bool(args.skip_tn) or bool(args.skip_ref):
        errors.append("rank diagnostics require Ref-only --skip_tn")
    requested = tuple(str(value) for value in args.ref_splits)
    if len(requested) != len(CANONICAL_REF_SPLITS) or set(requested) != set(
        CANONICAL_REF_SPLITS
    ):
        errors.append("all eight official Ref splits are required exactly once")
    if (
        str(args.device) != "cuda:0"
        or int(args.batch_size) != 16
        or int(args.num_workers) != 4
        or int(args.seed) != 42
        or not bool(args.amp)
    ):
        errors.append("runtime must be cuda:0/B16/W4/AMP/seed42")
    if (
        list(args.topk) != [1]
        or int(args.max_ref_batches) != 0
        or int(args.max_tn_batches) != 0
        or bool(args.no_per_example_records)
    ):
        errors.append("Top1/full batches/per-example records are required")
    if (
        args.screen_calibration_manifest
        or args.direct_prebuilt_tn
        or args.category_gate_max_gaps is not None
        or args.category_gate_include_base_expert
        or int(args.candidate_count_control) != 0
        or args.holdout_level != "none"
        or bool(args.exclude_train_jsonl)
    ):
        errors.append("rank diagnostics forbid calibration, sweeps, and filtering")
    if output_dir.exists() and any(output_dir.iterdir()):
        errors.append("output_dir must be absent or empty")
    if errors:
        raise ValueError(
            "--partial_dense_duty_rank_diagnostic contract failed: "
            + "; ".join(errors)
        )


def _validate_immutable_v39_archived_snapshot_diagnostic_args(
    args: argparse.Namespace, cfg
) -> None:
    if not bool(
        getattr(
            args,
            "immutable_v39_archived_snapshot_diagnostic",
            False,
        )
    ):
        return
    errors = []
    if bool(
        getattr(args, "immutable_v40_archived_snapshot_diagnostic", False)
    ) or bool(
        getattr(args, "immutable_v41_archived_snapshot_diagnostic", False)
    ):
        errors.append(
            "v39, v40, and v41 immutable diagnostics are mutually exclusive"
        )
    if not bool(
        getattr(args, "partial_dense_duty_confidence_diagnostic", False)
    ):
        errors.append(
            "--partial_dense_duty_confidence_diagnostic is also required"
        )
    try:
        observed_config = Path(args.config).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        observed_config = None
    expected_config = _CANDIDATE_GATE_ZERO_OFFSET_CONFIDENCE_U0400_CONFIG.resolve(
        strict=True
    )
    if observed_config != expected_config:
        errors.append("the fixed v39 U400 probe config is required")

    checkpoint_update = None
    if len(list(getattr(args, "ckpts", ()))) != 1:
        errors.append("exactly one archived checkpoint is required")
    else:
        try:
            observed_checkpoint = Path(args.ckpts[0]).expanduser().resolve(
                strict=True
            )
        except (OSError, RuntimeError, ValueError):
            observed_checkpoint = None
        for update, path in _V39_IMMUTABLE_ARCHIVED_SNAPSHOT_PATHS.items():
            if observed_checkpoint == path.resolve(strict=True):
                checkpoint_update = update
                break
        if checkpoint_update is None:
            errors.append("the exact archived v39 U100, U200, or U300 path is required")

    if checkpoint_update is not None:
        observed_output = Path(args.output_dir).expanduser().resolve(strict=False)
        expected_output = _V39_IMMUTABLE_ARCHIVED_SCREEN_OUTPUTS[
            checkpoint_update
        ].resolve(strict=False)
        if observed_output != expected_output:
            errors.append(
                "output_dir must be the fixed immutable snapshot screen directory"
            )

    if bool(getattr(args, "skip_tn", False)) or not bool(
        getattr(args, "skip_ref", False)
    ):
        errors.append("the immutable snapshot screen is TN-only")
    expected_tn = _PARTIAL_CONFIDENCE_TN_SPECS["strict1607"]
    try:
        observed_tn = Path(str(getattr(args, "tn_jsonl", ""))).expanduser()
        observed_tn = observed_tn.resolve(strict=True)
        observed_tn_sha256 = str(file_record(observed_tn)["sha256"])
    except (OSError, KeyError, RuntimeError, TypeError, ValueError):
        observed_tn = None
        observed_tn_sha256 = None
    if (
        observed_tn != Path(expected_tn["path"]).resolve(strict=True)
        or observed_tn_sha256 != expected_tn["sha256"]
        or list(getattr(args, "tn_splits", ()))
        != ["refcocop_val", "refcocog_umd_val"]
    ):
        errors.append("the exact strict1607 manifest and split filter are required")
    if (
        str(getattr(args, "device", "")) != "cuda:0"
        or int(getattr(args, "batch_size", 0)) != 16
        or int(getattr(args, "num_workers", 0)) != 4
        or int(getattr(args, "seed", -1)) != 42
        or not bool(getattr(args, "amp", False))
        or list(getattr(args, "topk", ())) != [1]
        or list(getattr(args, "threshold_tprs", ())) != [0.75, 0.9, 0.95]
        or list(getattr(args, "score_thresholds", ())) != [0.5]
        or int(getattr(args, "max_ref_batches", -1)) != 0
        or int(getattr(args, "max_tn_batches", -1)) != 0
        or int(getattr(args, "log_every", -1)) != 50
        or bool(getattr(args, "no_per_example_records", False))
    ):
        errors.append(
            "runtime must be cuda:0/B16/W4/AMP/seed42/Top1/full strict1607"
        )
    if str(
        getattr(
            cfg,
            "stage_b_dense_duty_confidence_probe_admission_contract",
            "",
        )
    ) != "disabled_for_probe_v1" or str(
        getattr(
            cfg,
            "stage_b_dense_duty_confidence_probe_admission_report",
            "",
        )
        or ""
    ):
        errors.append("promotion/admission must remain disabled")
    if errors:
        raise ValueError(
            "--immutable_v39_archived_snapshot_diagnostic contract failed: "
            + "; ".join(errors)
        )


def _validate_immutable_v40_archived_snapshot_diagnostic_args(
    args: argparse.Namespace, cfg
) -> None:
    if not bool(
        getattr(
            args,
            "immutable_v40_archived_snapshot_diagnostic",
            False,
        )
    ):
        return
    errors = []
    if bool(
        getattr(args, "immutable_v39_archived_snapshot_diagnostic", False)
    ) or bool(
        getattr(args, "immutable_v41_archived_snapshot_diagnostic", False)
    ):
        errors.append(
            "v39, v40, and v41 immutable diagnostics are mutually exclusive"
        )
    if not bool(
        getattr(args, "partial_dense_duty_confidence_diagnostic", False)
    ):
        errors.append(
            "--partial_dense_duty_confidence_diagnostic is also required"
        )
    try:
        observed_config = Path(args.config).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        observed_config = None
    expected_config = _CANDIDATE_HARDEST_EDIT_CONFIDENCE_U0400_CONFIG.resolve(
        strict=True
    )
    if observed_config != expected_config:
        errors.append("the fixed v40 U400 probe config is required")

    checkpoint_update = None
    if len(list(getattr(args, "ckpts", ()))) != 1:
        errors.append("exactly one archived checkpoint is required")
    else:
        try:
            observed_checkpoint = Path(args.ckpts[0]).expanduser().resolve(
                strict=True
            )
        except (OSError, RuntimeError, ValueError):
            observed_checkpoint = None
        for update, path in _V40_IMMUTABLE_ARCHIVED_SNAPSHOT_PATHS.items():
            if observed_checkpoint == path.resolve(strict=True):
                checkpoint_update = update
                break
        if checkpoint_update is None:
            errors.append("the exact archived v40 U100, U200, or U300 path is required")

    if checkpoint_update is not None:
        observed_output = Path(args.output_dir).expanduser().resolve(strict=False)
        expected_output = _V40_IMMUTABLE_ARCHIVED_SCREEN_OUTPUTS[
            checkpoint_update
        ].resolve(strict=False)
        if observed_output != expected_output:
            errors.append(
                "output_dir must be the fixed immutable snapshot screen directory"
            )

    if bool(getattr(args, "skip_tn", False)) or not bool(
        getattr(args, "skip_ref", False)
    ):
        errors.append("the immutable snapshot screen is TN-only")
    expected_tn = _PARTIAL_CONFIDENCE_TN_SPECS["strict1607"]
    try:
        observed_tn = Path(str(getattr(args, "tn_jsonl", ""))).expanduser()
        observed_tn = observed_tn.resolve(strict=True)
        observed_tn_sha256 = str(file_record(observed_tn)["sha256"])
    except (OSError, KeyError, RuntimeError, TypeError, ValueError):
        observed_tn = None
        observed_tn_sha256 = None
    if (
        observed_tn != Path(expected_tn["path"]).resolve(strict=True)
        or observed_tn_sha256 != expected_tn["sha256"]
        or list(getattr(args, "tn_splits", ()))
        != ["refcocop_val", "refcocog_umd_val"]
    ):
        errors.append("the exact strict1607 manifest and split filter are required")
    if (
        str(getattr(args, "device", "")) != "cuda:0"
        or int(getattr(args, "batch_size", 0)) != 16
        or int(getattr(args, "num_workers", 0)) != 4
        or int(getattr(args, "seed", -1)) != 42
        or not bool(getattr(args, "amp", False))
        or list(getattr(args, "topk", ())) != [1]
        or list(getattr(args, "threshold_tprs", ())) != [0.75, 0.9, 0.95]
        or list(getattr(args, "score_thresholds", ())) != [0.5]
        or int(getattr(args, "max_ref_batches", -1)) != 0
        or int(getattr(args, "max_tn_batches", -1)) != 0
        or int(getattr(args, "log_every", -1)) != 50
        or bool(getattr(args, "no_per_example_records", False))
    ):
        errors.append(
            "runtime must be cuda:0/B16/W4/AMP/seed42/Top1/full strict1607"
        )
    if str(
        getattr(
            cfg,
            "stage_b_dense_duty_confidence_probe_admission_contract",
            "",
        )
    ) != "disabled_for_probe_v1" or str(
        getattr(
            cfg,
            "stage_b_dense_duty_confidence_probe_admission_report",
            "",
        )
        or ""
    ):
        errors.append("promotion/admission must remain disabled")
    if errors:
        raise ValueError(
            "--immutable_v40_archived_snapshot_diagnostic contract failed: "
            + "; ".join(errors)
        )


def _validate_immutable_v41_archived_snapshot_diagnostic_args(
    args: argparse.Namespace, cfg
) -> None:
    if not bool(
        getattr(
            args,
            "immutable_v41_archived_snapshot_diagnostic",
            False,
        )
    ):
        return
    errors = []
    if bool(
        getattr(args, "immutable_v39_archived_snapshot_diagnostic", False)
    ) or bool(
        getattr(args, "immutable_v40_archived_snapshot_diagnostic", False)
    ):
        errors.append(
            "v39, v40, and v41 immutable diagnostics are mutually exclusive"
        )
    if not bool(
        getattr(args, "partial_dense_duty_confidence_diagnostic", False)
    ):
        errors.append(
            "--partial_dense_duty_confidence_diagnostic is also required"
        )
    try:
        observed_config = Path(args.config).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        observed_config = None
    expected_config = (
        _CANDIDATE_ROLE_COMPLETE_CARRIER_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        )
    )
    if observed_config != expected_config:
        errors.append("the fixed v41 U400 probe config is required")

    checkpoint_update = None
    if len(list(getattr(args, "ckpts", ()))) != 1:
        errors.append("exactly one archived checkpoint is required")
    else:
        try:
            observed_checkpoint = Path(args.ckpts[0]).expanduser().resolve(
                strict=True
            )
        except (OSError, RuntimeError, ValueError):
            observed_checkpoint = None
        for update, path in _V41_IMMUTABLE_ARCHIVED_SNAPSHOT_PATHS.items():
            if observed_checkpoint == path.resolve(strict=True):
                checkpoint_update = update
                break
        if checkpoint_update is None:
            errors.append(
                "the exact archived v41 U100, U200, or U300 path is required"
            )

    if checkpoint_update is not None:
        observed_output = Path(args.output_dir).expanduser().resolve(strict=False)
        expected_output = _V41_IMMUTABLE_ARCHIVED_SCREEN_OUTPUTS[
            checkpoint_update
        ].resolve(strict=False)
        if observed_output != expected_output:
            errors.append(
                "output_dir must be the fixed immutable snapshot screen directory"
            )

    if bool(getattr(args, "skip_tn", False)) or not bool(
        getattr(args, "skip_ref", False)
    ):
        errors.append("the immutable snapshot screen is TN-only")
    expected_tn = _PARTIAL_CONFIDENCE_TN_SPECS["strict1607"]
    try:
        observed_tn = Path(str(getattr(args, "tn_jsonl", ""))).expanduser()
        observed_tn = observed_tn.resolve(strict=True)
        observed_tn_sha256 = str(file_record(observed_tn)["sha256"])
    except (OSError, KeyError, RuntimeError, TypeError, ValueError):
        observed_tn = None
        observed_tn_sha256 = None
    if (
        observed_tn != Path(expected_tn["path"]).resolve(strict=True)
        or observed_tn_sha256 != expected_tn["sha256"]
        or list(getattr(args, "tn_splits", ()))
        != ["refcocop_val", "refcocog_umd_val"]
    ):
        errors.append("the exact strict1607 manifest and split filter are required")
    if (
        str(getattr(args, "device", "")) != "cuda:0"
        or int(getattr(args, "batch_size", 0)) != 16
        or int(getattr(args, "num_workers", 0)) != 4
        or int(getattr(args, "seed", -1)) != 42
        or not bool(getattr(args, "amp", False))
        or list(getattr(args, "topk", ())) != [1]
        or list(getattr(args, "threshold_tprs", ())) != [0.75, 0.9, 0.95]
        or list(getattr(args, "score_thresholds", ())) != [0.5]
        or int(getattr(args, "max_ref_batches", -1)) != 0
        or int(getattr(args, "max_tn_batches", -1)) != 0
        or int(getattr(args, "log_every", -1)) != 50
        or bool(getattr(args, "no_per_example_records", False))
    ):
        errors.append(
            "runtime must be cuda:0/B16/W4/AMP/seed42/Top1/full strict1607"
        )
    if str(
        getattr(
            cfg,
            "stage_b_dense_duty_confidence_probe_admission_contract",
            "",
        )
    ) != "disabled_for_probe_v1" or str(
        getattr(
            cfg,
            "stage_b_dense_duty_confidence_probe_admission_report",
            "",
        )
        or ""
    ):
        errors.append("promotion/admission must remain disabled")
    if errors:
        raise ValueError(
            "--immutable_v41_archived_snapshot_diagnostic contract failed: "
            + "; ".join(errors)
        )


def _validate_immutable_v42_archived_snapshot_diagnostic_args(
    args: argparse.Namespace, cfg
) -> None:
    if not bool(
        getattr(
            args,
            "immutable_v42_archived_snapshot_diagnostic",
            False,
        )
    ):
        return
    errors = []
    if any(
        bool(getattr(args, name, False))
        for name in (
            "immutable_v39_archived_snapshot_diagnostic",
            "immutable_v40_archived_snapshot_diagnostic",
            "immutable_v41_archived_snapshot_diagnostic",
        )
    ):
        errors.append(
            "v39, v40, v41, and v42 immutable diagnostics are mutually exclusive"
        )
    if not bool(
        getattr(args, "partial_dense_duty_confidence_diagnostic", False)
    ):
        errors.append(
            "--partial_dense_duty_confidence_diagnostic is also required"
        )
    try:
        observed_config = Path(args.config).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        observed_config = None
    if observed_config != (
        _CANDIDATE_TN_ONLY_CARRIER_PAIR_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        )
    ):
        errors.append("the fixed v42 TN-only carrier-pair probe config is required")
    checkpoint_update = None
    if len(list(getattr(args, "ckpts", ()))) != 1:
        errors.append("exactly one archived checkpoint is required")
    else:
        try:
            observed_checkpoint = Path(args.ckpts[0]).expanduser().resolve(
                strict=True
            )
        except (OSError, RuntimeError, ValueError):
            observed_checkpoint = None
        for update, path in _V42_IMMUTABLE_ARCHIVED_SNAPSHOT_PATHS.items():
            if observed_checkpoint == path.resolve(strict=True):
                checkpoint_update = update
                break
        if checkpoint_update is None:
            errors.append(
                "the exact archived v42 U100, U200, U300, or U400 path is required"
            )
    if bool(getattr(args, "skip_tn", False)) or not bool(
        getattr(args, "skip_ref", False)
    ):
        errors.append("the v42 immutable diagnostic is TN-only")
    if str(
        getattr(
            cfg,
            "stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract",
            "",
        )
    ) != "tn_only_positive_detached_v2":
        errors.append("the v42 immutable diagnostic requires TN-only pair gradients")
    if errors:
        raise ValueError(
            "--immutable_v42_archived_snapshot_diagnostic contract failed: "
            + "; ".join(errors)
        )
    cfg.stage_b_dense_duty_immutable_v42_archived_snapshot_diagnostic = True


def _validate_partial_dense_duty_confidence_diagnostic_args(
    args: argparse.Namespace, cfg
) -> None:
    if not bool(
        getattr(args, "partial_dense_duty_confidence_diagnostic", False)
    ):
        return
    observed_config = Path(args.config).expanduser().resolve(strict=True)
    supported_configs = {
        _PARTIAL_CONFIDENCE_CONFIG.resolve(strict=True),
        _VETO_PROBE_CONFIDENCE_CONFIG.resolve(strict=True),
        _VETO_GATE_PROBE_CONFIDENCE_CONFIG.resolve(strict=True),
        _VETO_CAP_PROBE_CONFIDENCE_CONFIG.resolve(strict=True),
        _VETO_GATED_POOL_PROBE_CONFIDENCE_CONFIG.resolve(strict=True),
        _VETO_GATED_POOL_CALIBRATED_PROBE_CONFIDENCE_CONFIG.resolve(strict=True),
        _VETO_GATED_POOL_CARRIER_BALANCED_PROBE_CONFIDENCE_CONFIG.resolve(strict=True),
        _VETO_GATED_POOL_CARRIER_QUARTER_PROBE_CONFIDENCE_CONFIG.resolve(strict=True),
        _VETO_GATED_POOL_CARRIER_PAIR_PROBE_CONFIDENCE_CONFIG.resolve(strict=True),
        _VETO_GATED_POOL_CARRIER_AFFINE_U0050_CONFIDENCE_CONFIG.resolve(strict=True),
        _VETO_GATED_POOL_TAIL_CARRIER_U0050_CONFIDENCE_CONFIG.resolve(strict=True),
        _VETO_GATED_POOL_TAIL_PAIRED_U0050_CONFIDENCE_CONFIG.resolve(strict=True),
        _VETO_GATED_POOL_TAIL_PAIRED_U0100_CONFIDENCE_CONFIG.resolve(strict=True),
        _VETO_GATED_POOL_TAIL_PAIRED_RANK_CHANNEL_U0050_CONFIDENCE_CONFIG.resolve(
            strict=True
        ),
        _VETO_GATED_POOL_TAIL_PAIRED_SIGNED_RANK_POOL_U0050_CONFIDENCE_CONFIG.resolve(
            strict=True
        ),
        _VETO_CONTINUOUS_RESIDUAL_U0050_CONFIDENCE_CONFIG.resolve(strict=True),
        _VETO_MONOTONE_DEPTH_U0050_CONFIDENCE_CONFIG.resolve(strict=True),
        _VETO_TOKEN_CONDITIONED_U0050_CONFIDENCE_CONFIG.resolve(strict=True),
        _VETO_TOKEN_CONDITIONED_U0300_CONFIDENCE_CONFIG.resolve(strict=True),
        _COMPLEMENTARY_TRUST_VETO_U0300_CONFIDENCE_CONFIG.resolve(strict=True),
        _UNGATED_MONOTONE_TAIL_VETO_U0300_CONFIDENCE_CONFIG.resolve(strict=True),
        _FLOOR_GATED_MONOTONE_TAIL_VETO_U0300_CONFIDENCE_CONFIG.resolve(strict=True),
        _INDEPENDENT_ABSOLUTE_CONFIDENCE_U0300_CONFIG.resolve(strict=True),
        _CROSS_ATTENTION_ABSOLUTE_CONFIDENCE_U0300_CONFIG.resolve(strict=True),
        _CANDIDATE_ABSOLUTE_CONFIDENCE_U0300_CONFIG.resolve(strict=True),
        _CANDIDATE_ABSOLUTE_CONFIDENCE_U0600_CONFIG.resolve(strict=True),
        _CANDIDATE_CALIBRATED_CONFIDENCE_U0400_CONFIG.resolve(strict=True),
        _CANDIDATE_NORMALIZED_CONFIDENCE_U0400_CONFIG.resolve(strict=True),
        _CANDIDATE_ASYMMETRIC_CONFIDENCE_U0400_CONFIG.resolve(strict=True),
        _CANDIDATE_Q05_CONFIDENCE_U0400_CONFIG.resolve(strict=True),
        _CANDIDATE_TAIL_BALANCED_CONFIDENCE_U0400_CONFIG.resolve(strict=True),
        _CANDIDATE_TAIL_QUARTER_CONFIDENCE_U0400_CONFIG.resolve(strict=True),
        _CANDIDATE_TAIL_BOUNDED_CONFIDENCE_U0400_CONFIG.resolve(strict=True),
        _CANDIDATE_TAIL_ELEMENTWISE_CONFIDENCE_U0400_CONFIG.resolve(strict=True),
        _CANDIDATE_GATE_ZERO_OFFSET_CONFIDENCE_U0400_CONFIG.resolve(strict=True),
        _CANDIDATE_HARDEST_EDIT_CONFIDENCE_U0400_CONFIG.resolve(strict=True),
        _CANDIDATE_ROLE_COMPLETE_CARRIER_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        ),
        _CANDIDATE_TN_ONLY_CARRIER_PAIR_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        ),
        _CANDIDATE_DEPLOYED_ROUTING_CONFIDENCE_U0400_CONFIG.resolve(strict=True),
        _CANDIDATE_SPLIT_HEADS_CONFIDENCE_U0400_CONFIG.resolve(strict=True),
        _CANDIDATE_SPLIT_TAIL_ALIGNED_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        ),
        _CANDIDATE_SPLIT_POSITIVE_TAIL_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        ),
        _CANDIDATE_SPLIT_BOUNDARY_ROUTING_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        ),
        _CANDIDATE_SPLIT_FPR_ACTIVE_SET_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        ),
        _CANDIDATE_SPLIT_GLOBAL_TRUST_VETO_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        ),
        _CANDIDATE_SPLIT_STRONG_BOUNDARY_ROUTING_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        ),
        _CANDIDATE_SPLIT_INDEPENDENT_DEPLOYED_ROUTER_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        ),
        _CANDIDATE_SAMPLE_CALIBRATOR_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        ),
        _FULLTEXT_GLOBAL_ABSOLUTE_CONFIDENCE_U0400_CONFIG.resolve(strict=True),
        _FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        ),
        _FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        ),
        _DEPLOYMENT_OWNED_GLOBAL_CONFIDENCE_U0400_CONFIG.resolve(strict=True),
        _DEPLOYED_GLOBAL_BALANCED_ABSOLUTE_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        ),
        _DEPLOYMENT_OWNED_GLOBAL_STABLE_FPR95_ACTIVE_SET_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        ),
        _DEPLOYMENT_OWNED_QUERY_GLOBAL_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        ),
        _DEPLOYMENT_OWNED_QUERY_VETO_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        ),
        _FULL_DECODER_VERIFIER_CONFIDENCE_U0400_CONFIG.resolve(strict=True),
        _FULL_DECODER_PATCH_SOFTMIN_VETO_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        ),
        _CANDIDATE_COMPLETE_TRACE_C1_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        ),
        _CANDIDATE_COMPLETE_TRACE_C2_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        ),
        _CANDIDATE_SET_ATTENTION_CONFIDENCE_U0400_CONFIG.resolve(strict=True),
    }
    veto_probe = observed_config in {
        _VETO_PROBE_CONFIDENCE_CONFIG.resolve(strict=True),
        _VETO_GATE_PROBE_CONFIDENCE_CONFIG.resolve(strict=True),
        _VETO_CAP_PROBE_CONFIDENCE_CONFIG.resolve(strict=True),
        _VETO_GATED_POOL_PROBE_CONFIDENCE_CONFIG.resolve(strict=True),
        _VETO_GATED_POOL_CALIBRATED_PROBE_CONFIDENCE_CONFIG.resolve(strict=True),
        _VETO_GATED_POOL_CARRIER_BALANCED_PROBE_CONFIDENCE_CONFIG.resolve(strict=True),
        _VETO_GATED_POOL_CARRIER_QUARTER_PROBE_CONFIDENCE_CONFIG.resolve(strict=True),
        _VETO_GATED_POOL_CARRIER_PAIR_PROBE_CONFIDENCE_CONFIG.resolve(strict=True),
        _VETO_GATED_POOL_CARRIER_AFFINE_U0050_CONFIDENCE_CONFIG.resolve(strict=True),
        _VETO_GATED_POOL_TAIL_CARRIER_U0050_CONFIDENCE_CONFIG.resolve(strict=True),
        _VETO_GATED_POOL_TAIL_PAIRED_U0050_CONFIDENCE_CONFIG.resolve(strict=True),
        _VETO_GATED_POOL_TAIL_PAIRED_U0100_CONFIDENCE_CONFIG.resolve(strict=True),
        _VETO_GATED_POOL_TAIL_PAIRED_RANK_CHANNEL_U0050_CONFIDENCE_CONFIG.resolve(
            strict=True
        ),
        _VETO_GATED_POOL_TAIL_PAIRED_SIGNED_RANK_POOL_U0050_CONFIDENCE_CONFIG.resolve(
            strict=True
        ),
        _VETO_CONTINUOUS_RESIDUAL_U0050_CONFIDENCE_CONFIG.resolve(strict=True),
        _VETO_MONOTONE_DEPTH_U0050_CONFIDENCE_CONFIG.resolve(strict=True),
        _VETO_TOKEN_CONDITIONED_U0050_CONFIDENCE_CONFIG.resolve(strict=True),
        _VETO_TOKEN_CONDITIONED_U0300_CONFIDENCE_CONFIG.resolve(strict=True),
        _COMPLEMENTARY_TRUST_VETO_U0300_CONFIDENCE_CONFIG.resolve(strict=True),
        _UNGATED_MONOTONE_TAIL_VETO_U0300_CONFIDENCE_CONFIG.resolve(strict=True),
        _FLOOR_GATED_MONOTONE_TAIL_VETO_U0300_CONFIDENCE_CONFIG.resolve(strict=True),
        _INDEPENDENT_ABSOLUTE_CONFIDENCE_U0300_CONFIG.resolve(strict=True),
        _CROSS_ATTENTION_ABSOLUTE_CONFIDENCE_U0300_CONFIG.resolve(strict=True),
        _CANDIDATE_ABSOLUTE_CONFIDENCE_U0300_CONFIG.resolve(strict=True),
        _CANDIDATE_ABSOLUTE_CONFIDENCE_U0600_CONFIG.resolve(strict=True),
        _CANDIDATE_CALIBRATED_CONFIDENCE_U0400_CONFIG.resolve(strict=True),
        _CANDIDATE_NORMALIZED_CONFIDENCE_U0400_CONFIG.resolve(strict=True),
        _CANDIDATE_ASYMMETRIC_CONFIDENCE_U0400_CONFIG.resolve(strict=True),
        _CANDIDATE_Q05_CONFIDENCE_U0400_CONFIG.resolve(strict=True),
        _CANDIDATE_TAIL_BALANCED_CONFIDENCE_U0400_CONFIG.resolve(strict=True),
        _CANDIDATE_TAIL_QUARTER_CONFIDENCE_U0400_CONFIG.resolve(strict=True),
        _CANDIDATE_TAIL_BOUNDED_CONFIDENCE_U0400_CONFIG.resolve(strict=True),
        _CANDIDATE_TAIL_ELEMENTWISE_CONFIDENCE_U0400_CONFIG.resolve(strict=True),
        _CANDIDATE_GATE_ZERO_OFFSET_CONFIDENCE_U0400_CONFIG.resolve(strict=True),
        _CANDIDATE_HARDEST_EDIT_CONFIDENCE_U0400_CONFIG.resolve(strict=True),
        _CANDIDATE_ROLE_COMPLETE_CARRIER_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        ),
        _CANDIDATE_TN_ONLY_CARRIER_PAIR_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        ),
        _CANDIDATE_DEPLOYED_ROUTING_CONFIDENCE_U0400_CONFIG.resolve(strict=True),
        _CANDIDATE_SPLIT_HEADS_CONFIDENCE_U0400_CONFIG.resolve(strict=True),
        _CANDIDATE_SPLIT_TAIL_ALIGNED_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        ),
        _CANDIDATE_SPLIT_POSITIVE_TAIL_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        ),
        _CANDIDATE_SPLIT_BOUNDARY_ROUTING_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        ),
        _CANDIDATE_SPLIT_FPR_ACTIVE_SET_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        ),
        _CANDIDATE_SPLIT_GLOBAL_TRUST_VETO_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        ),
        _CANDIDATE_SPLIT_STRONG_BOUNDARY_ROUTING_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        ),
        _CANDIDATE_SPLIT_INDEPENDENT_DEPLOYED_ROUTER_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        ),
        _CANDIDATE_SAMPLE_CALIBRATOR_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        ),
        _FULLTEXT_GLOBAL_ABSOLUTE_CONFIDENCE_U0400_CONFIG.resolve(strict=True),
        _FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        ),
        _FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        ),
        _DEPLOYMENT_OWNED_GLOBAL_CONFIDENCE_U0400_CONFIG.resolve(strict=True),
        _DEPLOYED_GLOBAL_BALANCED_ABSOLUTE_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        ),
        _DEPLOYMENT_OWNED_GLOBAL_STABLE_FPR95_ACTIVE_SET_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        ),
        _DEPLOYMENT_OWNED_QUERY_GLOBAL_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        ),
        _CANDIDATE_SET_ATTENTION_CONFIDENCE_U0400_CONFIG.resolve(strict=True),
    }
    veto_gate_probe = observed_config == (
        _VETO_GATE_PROBE_CONFIDENCE_CONFIG.resolve(strict=True)
    )
    veto_cap_probe = observed_config == (
        _VETO_CAP_PROBE_CONFIDENCE_CONFIG.resolve(strict=True)
    )
    veto_gated_pool_probe = observed_config == (
        _VETO_GATED_POOL_PROBE_CONFIDENCE_CONFIG.resolve(strict=True)
    )
    veto_gated_pool_calibrated_probe = observed_config == (
        _VETO_GATED_POOL_CALIBRATED_PROBE_CONFIDENCE_CONFIG.resolve(strict=True)
    )
    veto_gated_pool_carrier_balanced_probe = observed_config == (
        _VETO_GATED_POOL_CARRIER_BALANCED_PROBE_CONFIDENCE_CONFIG.resolve(
            strict=True
        )
    )
    veto_gated_pool_carrier_quarter_probe = observed_config == (
        _VETO_GATED_POOL_CARRIER_QUARTER_PROBE_CONFIDENCE_CONFIG.resolve(
            strict=True
        )
    )
    veto_gated_pool_carrier_pair_probe = observed_config == (
        _VETO_GATED_POOL_CARRIER_PAIR_PROBE_CONFIDENCE_CONFIG.resolve(
            strict=True
        )
    )
    veto_gated_pool_carrier_affine_u0050 = observed_config == (
        _VETO_GATED_POOL_CARRIER_AFFINE_U0050_CONFIDENCE_CONFIG.resolve(
            strict=True
        )
    )
    veto_gated_pool_tail_carrier_u0050 = observed_config == (
        _VETO_GATED_POOL_TAIL_CARRIER_U0050_CONFIDENCE_CONFIG.resolve(
            strict=True
        )
    )
    veto_gated_pool_tail_paired_u0050 = observed_config == (
        _VETO_GATED_POOL_TAIL_PAIRED_U0050_CONFIDENCE_CONFIG.resolve(
            strict=True
        )
    )
    veto_gated_pool_tail_paired_u0100 = observed_config == (
        _VETO_GATED_POOL_TAIL_PAIRED_U0100_CONFIDENCE_CONFIG.resolve(
            strict=True
        )
    )
    veto_gated_pool_tail_paired_rank_channel_u0050 = observed_config == (
        _VETO_GATED_POOL_TAIL_PAIRED_RANK_CHANNEL_U0050_CONFIDENCE_CONFIG.resolve(
            strict=True
        )
    )
    veto_gated_pool_tail_paired_signed_rank_pool_u0050 = observed_config == (
        _VETO_GATED_POOL_TAIL_PAIRED_SIGNED_RANK_POOL_U0050_CONFIDENCE_CONFIG.resolve(
            strict=True
        )
    )
    veto_continuous_residual_u0050 = observed_config == (
        _VETO_CONTINUOUS_RESIDUAL_U0050_CONFIDENCE_CONFIG.resolve(strict=True)
    )
    veto_monotone_depth_u0050 = observed_config == (
        _VETO_MONOTONE_DEPTH_U0050_CONFIDENCE_CONFIG.resolve(strict=True)
    )
    veto_token_conditioned_u0050 = observed_config == (
        _VETO_TOKEN_CONDITIONED_U0050_CONFIDENCE_CONFIG.resolve(strict=True)
    )
    veto_token_conditioned_u0300 = observed_config == (
        _VETO_TOKEN_CONDITIONED_U0300_CONFIDENCE_CONFIG.resolve(strict=True)
    )
    complementary_trust_veto_u0300 = observed_config == (
        _COMPLEMENTARY_TRUST_VETO_U0300_CONFIDENCE_CONFIG.resolve(strict=True)
    )
    ungated_monotone_tail_veto_u0300 = observed_config == (
        _UNGATED_MONOTONE_TAIL_VETO_U0300_CONFIDENCE_CONFIG.resolve(strict=True)
    )
    floor_gated_monotone_tail_veto_u0300 = observed_config == (
        _FLOOR_GATED_MONOTONE_TAIL_VETO_U0300_CONFIDENCE_CONFIG.resolve(
            strict=True
        )
    )
    independent_absolute_confidence_u0300 = observed_config == (
        _INDEPENDENT_ABSOLUTE_CONFIDENCE_U0300_CONFIG.resolve(strict=True)
    )
    cross_attention_absolute_confidence_u0300 = observed_config == (
        _CROSS_ATTENTION_ABSOLUTE_CONFIDENCE_U0300_CONFIG.resolve(strict=True)
    )
    candidate_absolute_confidence_u0300 = observed_config == (
        _CANDIDATE_ABSOLUTE_CONFIDENCE_U0300_CONFIG.resolve(strict=True)
    )
    candidate_absolute_confidence_u0600 = observed_config == (
        _CANDIDATE_ABSOLUTE_CONFIDENCE_U0600_CONFIG.resolve(strict=True)
    )
    candidate_calibrated_confidence_u0400 = observed_config == (
        _CANDIDATE_CALIBRATED_CONFIDENCE_U0400_CONFIG.resolve(strict=True)
    )
    candidate_normalized_confidence_u0400 = observed_config == (
        _CANDIDATE_NORMALIZED_CONFIDENCE_U0400_CONFIG.resolve(strict=True)
    )
    candidate_asymmetric_confidence_u0400 = observed_config == (
        _CANDIDATE_ASYMMETRIC_CONFIDENCE_U0400_CONFIG.resolve(strict=True)
    )
    candidate_q05_confidence_u0400 = observed_config == (
        _CANDIDATE_Q05_CONFIDENCE_U0400_CONFIG.resolve(strict=True)
    )
    candidate_tail_balanced_confidence_u0400 = observed_config == (
        _CANDIDATE_TAIL_BALANCED_CONFIDENCE_U0400_CONFIG.resolve(strict=True)
    )
    candidate_tail_quarter_confidence_u0400 = observed_config == (
        _CANDIDATE_TAIL_QUARTER_CONFIDENCE_U0400_CONFIG.resolve(strict=True)
    )
    candidate_tail_bounded_confidence_u0400 = observed_config == (
        _CANDIDATE_TAIL_BOUNDED_CONFIDENCE_U0400_CONFIG.resolve(strict=True)
    )
    candidate_tail_elementwise_confidence_u0400 = observed_config == (
        _CANDIDATE_TAIL_ELEMENTWISE_CONFIDENCE_U0400_CONFIG.resolve(strict=True)
    )
    candidate_gate_zero_offset_confidence_u0400 = observed_config == (
        _CANDIDATE_GATE_ZERO_OFFSET_CONFIDENCE_U0400_CONFIG.resolve(strict=True)
    )
    candidate_hardest_edit_confidence_u0400 = observed_config == (
        _CANDIDATE_HARDEST_EDIT_CONFIDENCE_U0400_CONFIG.resolve(strict=True)
    )
    candidate_role_complete_carrier_confidence_u0400 = observed_config == (
        _CANDIDATE_ROLE_COMPLETE_CARRIER_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        )
    )
    candidate_tn_only_carrier_pair_confidence_u0400 = observed_config == (
        _CANDIDATE_TN_ONLY_CARRIER_PAIR_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        )
    )
    candidate_deployed_routing_confidence_u0400 = observed_config == (
        _CANDIDATE_DEPLOYED_ROUTING_CONFIDENCE_U0400_CONFIG.resolve(strict=True)
    )
    candidate_split_heads_confidence_u0400 = observed_config == (
        _CANDIDATE_SPLIT_HEADS_CONFIDENCE_U0400_CONFIG.resolve(strict=True)
    )
    candidate_split_tail_aligned_confidence_u0400 = observed_config == (
        _CANDIDATE_SPLIT_TAIL_ALIGNED_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        )
    )
    candidate_split_positive_tail_confidence_u0400 = observed_config == (
        _CANDIDATE_SPLIT_POSITIVE_TAIL_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        )
    )
    candidate_split_boundary_routing_confidence_u0400 = observed_config == (
        _CANDIDATE_SPLIT_BOUNDARY_ROUTING_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        )
    )
    candidate_split_fpr_active_set_confidence_u0400 = observed_config == (
        _CANDIDATE_SPLIT_FPR_ACTIVE_SET_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        )
    )
    candidate_split_global_trust_veto_confidence_u0400 = observed_config == (
        _CANDIDATE_SPLIT_GLOBAL_TRUST_VETO_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        )
    )
    candidate_split_strong_boundary_routing_confidence_u0400 = observed_config == (
        _CANDIDATE_SPLIT_STRONG_BOUNDARY_ROUTING_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        )
    )
    candidate_split_independent_deployed_router_confidence_u0400 = (
        observed_config
        == _CANDIDATE_SPLIT_INDEPENDENT_DEPLOYED_ROUTER_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        )
    )
    candidate_sample_calibrator_confidence_u0400 = observed_config == (
        _CANDIDATE_SAMPLE_CALIBRATOR_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        )
    )
    fulltext_global_absolute_confidence_u0400 = observed_config == (
        _FULLTEXT_GLOBAL_ABSOLUTE_CONFIDENCE_U0400_CONFIG.resolve(strict=True)
    )
    fulltext_global_absolute_exact_residual_confidence_u0400 = (
        observed_config
        == _FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        )
    )
    fulltext_global_independent_absolute_confidence_u0400 = (
        observed_config
        == _FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        )
    )
    deployed_global_balanced_absolute_confidence_u0400 = observed_config == (
        _DEPLOYED_GLOBAL_BALANCED_ABSOLUTE_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        )
    )
    deployment_owned_global_stable_fpr95_active_set_confidence_u0400 = (
        observed_config
        == _DEPLOYMENT_OWNED_GLOBAL_STABLE_FPR95_ACTIVE_SET_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        )
    )
    deployment_owned_query_global_confidence_u0400 = (
        observed_config
        == _DEPLOYMENT_OWNED_QUERY_GLOBAL_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        )
    )
    full_decoder_verifier_confidence_u0400 = observed_config == (
        _FULL_DECODER_VERIFIER_CONFIDENCE_U0400_CONFIG.resolve(strict=True)
    )
    full_decoder_patch_softmin_veto_confidence_u0400 = observed_config == (
        _FULL_DECODER_PATCH_SOFTMIN_VETO_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        )
    )
    candidate_complete_trace_c1_confidence_u0400 = observed_config == (
        _CANDIDATE_COMPLETE_TRACE_C1_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        )
    )
    candidate_complete_trace_c2_confidence_u0400 = observed_config == (
        _CANDIDATE_COMPLETE_TRACE_C2_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        )
    )
    deployment_owned_query_veto_confidence_u0400 = (
        observed_config
        == _DEPLOYMENT_OWNED_QUERY_VETO_CONFIDENCE_U0400_CONFIG.resolve(
            strict=True
        )
        or full_decoder_verifier_confidence_u0400
        or full_decoder_patch_softmin_veto_confidence_u0400
        or candidate_complete_trace_c1_confidence_u0400
        or candidate_complete_trace_c2_confidence_u0400
    )
    deployment_owned_global_confidence_u0400 = (
        observed_config
        == _DEPLOYMENT_OWNED_GLOBAL_CONFIDENCE_U0400_CONFIG.resolve(strict=True)
        or deployed_global_balanced_absolute_confidence_u0400
        or deployment_owned_global_stable_fpr95_active_set_confidence_u0400
        or deployment_owned_query_global_confidence_u0400
        or deployment_owned_query_veto_confidence_u0400
    )
    candidate_gate_zero_offset_family = (
        candidate_gate_zero_offset_confidence_u0400
        or candidate_hardest_edit_confidence_u0400
        or candidate_role_complete_carrier_confidence_u0400
        or candidate_tn_only_carrier_pair_confidence_u0400
        or candidate_deployed_routing_confidence_u0400
        or candidate_split_heads_confidence_u0400
        or candidate_split_tail_aligned_confidence_u0400
        or candidate_split_positive_tail_confidence_u0400
        or candidate_split_boundary_routing_confidence_u0400
        or candidate_split_fpr_active_set_confidence_u0400
        or candidate_split_global_trust_veto_confidence_u0400
        or candidate_split_strong_boundary_routing_confidence_u0400
        or candidate_split_independent_deployed_router_confidence_u0400
        or candidate_sample_calibrator_confidence_u0400
    )
    candidate_set_attention_confidence_u0400 = observed_config == (
        _CANDIDATE_SET_ATTENTION_CONFIDENCE_U0400_CONFIG.resolve(strict=True)
    )
    candidate_absolute_confidence_probe = (
        candidate_absolute_confidence_u0300
        or candidate_absolute_confidence_u0600
        or candidate_calibrated_confidence_u0400
        or candidate_normalized_confidence_u0400
        or candidate_asymmetric_confidence_u0400
        or candidate_q05_confidence_u0400
        or candidate_tail_balanced_confidence_u0400
        or candidate_tail_quarter_confidence_u0400
        or candidate_tail_bounded_confidence_u0400
        or candidate_tail_elementwise_confidence_u0400
        or candidate_gate_zero_offset_family
        or candidate_set_attention_confidence_u0400
        or fulltext_global_absolute_confidence_u0400
        or fulltext_global_absolute_exact_residual_confidence_u0400
        or fulltext_global_independent_absolute_confidence_u0400
        or deployment_owned_global_confidence_u0400
    )
    veto_token_conditioned_probe = (
        veto_token_conditioned_u0050
        or veto_token_conditioned_u0300
        or complementary_trust_veto_u0300
        or ungated_monotone_tail_veto_u0300
        or floor_gated_monotone_tail_veto_u0300
        or independent_absolute_confidence_u0300
        or cross_attention_absolute_confidence_u0300
        or candidate_absolute_confidence_probe
    )
    absolute_cap_probe = (
        veto_cap_probe
        or veto_gated_pool_probe
        or veto_gated_pool_calibrated_probe
        or veto_gated_pool_carrier_balanced_probe
        or veto_gated_pool_carrier_quarter_probe
        or veto_gated_pool_carrier_pair_probe
        or veto_gated_pool_carrier_affine_u0050
        or veto_gated_pool_tail_carrier_u0050
        or veto_gated_pool_tail_paired_u0050
        or veto_gated_pool_tail_paired_u0100
        or veto_gated_pool_tail_paired_rank_channel_u0050
        or veto_gated_pool_tail_paired_signed_rank_pool_u0050
        or veto_continuous_residual_u0050
        or veto_monotone_depth_u0050
        or veto_token_conditioned_probe
    )
    fixed_u0050_probe = (
        veto_gated_pool_carrier_affine_u0050
        or veto_gated_pool_tail_carrier_u0050
        or veto_gated_pool_tail_paired_u0050
        or veto_gated_pool_tail_paired_rank_channel_u0050
        or veto_gated_pool_tail_paired_signed_rank_pool_u0050
        or veto_continuous_residual_u0050
        or veto_monotone_depth_u0050
        or veto_token_conditioned_u0050
    )
    fixed_u0100_probe = veto_gated_pool_tail_paired_u0100
    carrier_affine_fixed_probe = fixed_u0050_probe or fixed_u0100_probe
    tail_carrier_fixed_probe = (
        veto_gated_pool_tail_carrier_u0050
        or veto_gated_pool_tail_paired_u0050
        or veto_gated_pool_tail_paired_u0100
        or veto_gated_pool_tail_paired_rank_channel_u0050
        or veto_gated_pool_tail_paired_signed_rank_pool_u0050
        or veto_continuous_residual_u0050
        or veto_monotone_depth_u0050
        or veto_token_conditioned_probe
    )
    tail_paired_fixed_probe = (
        veto_gated_pool_tail_paired_u0050
        or veto_gated_pool_tail_paired_u0100
        or veto_gated_pool_tail_paired_rank_channel_u0050
        or veto_gated_pool_tail_paired_signed_rank_pool_u0050
        or veto_continuous_residual_u0050
        or veto_monotone_depth_u0050
        or veto_token_conditioned_probe
    )
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)
    errors = []
    if (
        candidate_deployed_routing_confidence_u0400
        or candidate_split_heads_confidence_u0400
    ):
        try:
            _validate_v43_deployed_routing_config(cfg)
        except (RuntimeError, TypeError, ValueError) as exc:
            errors.append(str(exc))
    if candidate_split_tail_aligned_confidence_u0400:
        try:
            _validate_v45_split_tail_aligned_config(cfg)
        except (RuntimeError, TypeError, ValueError) as exc:
            errors.append(str(exc))
    if candidate_split_positive_tail_confidence_u0400:
        try:
            _validate_v46_split_positive_tail_config(cfg)
        except (RuntimeError, TypeError, ValueError) as exc:
            errors.append(str(exc))
    if candidate_split_boundary_routing_confidence_u0400:
        try:
            _validate_v47_split_boundary_routing_config(cfg)
        except (RuntimeError, TypeError, ValueError) as exc:
            errors.append(str(exc))
    if candidate_split_fpr_active_set_confidence_u0400:
        try:
            _validate_v48_split_fpr_active_set_config(cfg)
        except (RuntimeError, TypeError, ValueError) as exc:
            errors.append(str(exc))
    if candidate_split_global_trust_veto_confidence_u0400:
        try:
            _validate_v49_split_global_trust_veto_config(cfg)
        except (RuntimeError, TypeError, ValueError) as exc:
            errors.append(str(exc))
    if candidate_split_strong_boundary_routing_confidence_u0400:
        try:
            _validate_v50_split_strong_boundary_routing_config(cfg)
        except (RuntimeError, TypeError, ValueError) as exc:
            errors.append(str(exc))
    if candidate_split_independent_deployed_router_confidence_u0400:
        try:
            _validate_v51_split_independent_deployed_router_config(cfg)
        except (RuntimeError, TypeError, ValueError) as exc:
            errors.append(str(exc))
    if candidate_sample_calibrator_confidence_u0400:
        try:
            _validate_v52_candidate_sample_calibrator_config(cfg)
        except (RuntimeError, TypeError, ValueError) as exc:
            errors.append(str(exc))
    if fulltext_global_absolute_confidence_u0400:
        try:
            _validate_v53_fulltext_global_absolute_config(cfg)
        except (RuntimeError, TypeError, ValueError) as exc:
            errors.append(str(exc))
    if fulltext_global_absolute_exact_residual_confidence_u0400:
        try:
            _validate_v54_fulltext_global_absolute_exact_residual_config(cfg)
        except (RuntimeError, TypeError, ValueError) as exc:
            errors.append(str(exc))
    if fulltext_global_independent_absolute_confidence_u0400:
        try:
            _validate_v55_fulltext_global_independent_absolute_config(cfg)
        except (RuntimeError, TypeError, ValueError) as exc:
            errors.append(str(exc))
    if deployment_owned_global_confidence_u0400:
        try:
            if candidate_complete_trace_c2_confidence_u0400:
                if not _validate_c2_candidate_complete_trace_config(cfg):
                    raise RuntimeError(
                        "candidate-complete C2 config does not satisfy its exact "
                        "monotone contract"
                    )
            elif candidate_complete_trace_c1_confidence_u0400:
                _validate_c1_candidate_complete_trace_config(cfg)
            elif full_decoder_patch_softmin_veto_confidence_u0400:
                _validate_v62_patch_softmin_veto_config(cfg)
            elif full_decoder_verifier_confidence_u0400:
                _validate_v61_full_decoder_verifier_config(cfg)
            elif deployment_owned_query_veto_confidence_u0400:
                _validate_v60_deployment_owned_query_veto_config(cfg)
            elif deployment_owned_query_global_confidence_u0400:
                _validate_v59_deployment_owned_query_global_config(cfg)
            elif deployed_global_balanced_absolute_confidence_u0400:
                _validate_v57_deployed_global_balanced_absolute_config(cfg)
            elif deployment_owned_global_stable_fpr95_active_set_confidence_u0400:
                _validate_v58_deployment_owned_stable_fpr95_active_set_config(cfg)
            else:
                _validate_v56_deployment_owned_global_config(cfg)
        except (RuntimeError, TypeError, ValueError) as exc:
            errors.append(str(exc))
    if candidate_split_heads_confidence_u0400 and str(
        getattr(
            cfg,
            "stage_b_dense_duty_confidence_head_gradient_contract",
            "",
        )
    ).strip() != "split_token_veto_global_absolute_v2":
        errors.append(
            "candidate split-head probe requires its exact head-gradient contract"
        )
    if bool(getattr(args, "partial_dense_duty_rank_diagnostic", False)):
        errors.append("rank and confidence diagnostics are mutually exclusive")
    if observed_config not in supported_configs:
        errors.append(
            "a fixed dense-duty confidence-adapter diagnostic config is required"
        )
    if candidate_q05_confidence_u0400 and str(
        getattr(
            cfg,
            "stage_b_v15_tail_queue_positive_gradient_contract",
            "",
        )
    ).strip() != "exact_batch_lower_tail_st_v2":
        errors.append(
            "candidate-q05 probe requires exact batch lower-tail gradient routing"
        )
    if candidate_tail_balanced_confidence_u0400 and str(
        getattr(
            cfg,
            "stage_b_v15_tail_queue_positive_gradient_contract",
            "",
        )
    ).strip() != "mean_plus_exact_lower_tail_st_v3":
        errors.append(
            "candidate-tail-balanced probe requires mean-plus-exact-tail "
            "gradient routing"
        )
    if candidate_tail_quarter_confidence_u0400 and str(
        getattr(
            cfg,
            "stage_b_v15_tail_queue_positive_gradient_contract",
            "",
        )
    ).strip() != "mean_plus_quarter_exact_lower_tail_st_v4":
        errors.append(
            "candidate-tail-quarter probe requires quarter-strength exact-tail "
            "gradient routing"
        )
    if candidate_tail_bounded_confidence_u0400 and str(
        getattr(
            cfg,
            "stage_b_v15_tail_queue_positive_gradient_contract",
            "",
        )
    ).strip() != "bounded_mean_plus_sixteenth_exact_lower_tail_st_v5":
        errors.append(
            "candidate-tail-bounded probe requires bounded mean plus sixteenth-tail "
            "gradient routing"
        )
    if candidate_tail_elementwise_confidence_u0400 and str(
        getattr(
            cfg,
            "stage_b_v15_tail_queue_positive_gradient_contract",
            "",
        )
    ).strip() != "elementwise_bounded_mean_plus_sixteenth_exact_lower_tail_st_v6":
        errors.append(
            "candidate-tail-elementwise probe requires elementwise bounded "
            "mean plus sixteenth-tail gradient routing"
        )
    if candidate_tail_elementwise_confidence_u0400 and float(
        getattr(
            cfg,
            "stage_b_dense_duty_confidence_veto_gate_offset",
            -1.0,
        )
    ) != 0.02:
        errors.append("candidate-tail-elementwise probe requires gate offset 0.02")
    if candidate_gate_zero_offset_family and str(
        getattr(
            cfg,
            "stage_b_v15_tail_queue_positive_gradient_contract",
            "",
        )
    ).strip() != "elementwise_bounded_mean_plus_sixteenth_exact_lower_tail_st_v6":
        errors.append(
            "candidate-gate-zero-offset probe requires elementwise bounded "
            "mean plus sixteenth-tail gradient routing"
        )
    if candidate_gate_zero_offset_family and float(
        getattr(
            cfg,
            "stage_b_dense_duty_confidence_veto_gate_offset",
            -1.0,
        )
    ) != 0.0:
        errors.append("candidate-gate-zero-offset probe requires gate offset 0.0")
    observed_token_edit_scope = str(
        getattr(cfg, "stage_b_v21_token_edit_query_scope", "target_iou_v1")
    ).strip().lower()
    if (
        candidate_hardest_edit_confidence_u0400
        and observed_token_edit_scope
        != "target_iou_union_detached_final_confidence_base_argmax_v2"
    ):
        errors.append(
            "candidate-hardest-edit probe requires its exact detached base-logit "
            "carrier scope"
        )
    if (
        candidate_role_complete_carrier_confidence_u0400
        and observed_token_edit_scope
        != "target_iou_union_detached_role_complete_confidence_base_argmax_v3"
    ):
        errors.append(
            "candidate-role-complete-carrier probe requires its exact detached "
            "role-complete base-logit carrier scope"
        )
    if (
        (
            candidate_gate_zero_offset_confidence_u0400
            or candidate_tn_only_carrier_pair_confidence_u0400
            or candidate_deployed_routing_confidence_u0400
            or candidate_split_heads_confidence_u0400
            or candidate_split_tail_aligned_confidence_u0400
            or candidate_split_positive_tail_confidence_u0400
            or candidate_split_boundary_routing_confidence_u0400
            or candidate_split_fpr_active_set_confidence_u0400
            or candidate_split_global_trust_veto_confidence_u0400
            or candidate_split_strong_boundary_routing_confidence_u0400
            or candidate_split_independent_deployed_router_confidence_u0400
            or candidate_sample_calibrator_confidence_u0400
            or fulltext_global_absolute_confidence_u0400
            or fulltext_global_absolute_exact_residual_confidence_u0400
            or fulltext_global_independent_absolute_confidence_u0400
            or deployment_owned_global_confidence_u0400
        )
        and not candidate_complete_trace_c1_confidence_u0400
        and not candidate_complete_trace_c2_confidence_u0400
        and observed_token_edit_scope != "target_iou_v1"
    ):
        errors.append(
            "candidate-gate-zero-offset v39/v42/v43 requires target-IoU token scope"
        )
    observed_carrier_pair_gradient_contract = str(
        getattr(
            cfg,
            "stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract",
            "bidirectional_v1",
        )
    ).strip()
    if (
        candidate_tn_only_carrier_pair_confidence_u0400
        and observed_carrier_pair_gradient_contract
        != "tn_only_positive_detached_v2"
    ):
        errors.append(
            "candidate-tn-only-carrier-pair v42 requires "
            "tn_only_positive_detached_v2 carrier-pair gradients"
        )
    if (
        (
            candidate_gate_zero_offset_confidence_u0400
            or candidate_hardest_edit_confidence_u0400
            or candidate_role_complete_carrier_confidence_u0400
            or candidate_deployed_routing_confidence_u0400
            or candidate_split_heads_confidence_u0400
            or candidate_split_tail_aligned_confidence_u0400
            or candidate_split_positive_tail_confidence_u0400
            or candidate_split_boundary_routing_confidence_u0400
            or candidate_split_fpr_active_set_confidence_u0400
            or candidate_split_global_trust_veto_confidence_u0400
            or candidate_split_strong_boundary_routing_confidence_u0400
            or candidate_split_independent_deployed_router_confidence_u0400
            or candidate_sample_calibrator_confidence_u0400
            or fulltext_global_absolute_confidence_u0400
            or fulltext_global_absolute_exact_residual_confidence_u0400
            or fulltext_global_independent_absolute_confidence_u0400
            or deployment_owned_global_confidence_u0400
        )
        and observed_carrier_pair_gradient_contract != "bidirectional_v1"
    ):
        errors.append(
            "candidate v39-v41 require bidirectional_v1 carrier-pair gradients; "
            "v43 requires the same contract"
        )
    if (
        not bool(getattr(cfg, "stage_b_dense_duty", False))
        or str(getattr(cfg, "stage_b_dense_duty_phase", "")) != "confidence"
        or str(getattr(cfg, "stage_b_v22_train_phase", "")) != "confidence"
        or str(getattr(cfg, "stage_b_v22_score_ownership", ""))
        != "rank_tower_stopgrad_token_adapter_two_phase"
    ):
        errors.append("config must build the dense-duty confidence-adapter phase")
    if candidate_absolute_confidence_probe:
        expected_veto_aggregation = (
            "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        )
        expected_veto_revision = (
            "word_veto_candidate_patch_invariant_confidence_v30"
            if candidate_calibrated_confidence_u0400
            else (
                _V43_DEPLOYED_ROUTING_REVISION
                if (
                    candidate_deployed_routing_confidence_u0400
                    or candidate_split_heads_confidence_u0400
                )
                else (
                    "word_veto_candidate_asymmetric_confidence_v32"
                    if (
                        candidate_asymmetric_confidence_u0400
                        or candidate_q05_confidence_u0400
                        or candidate_tail_balanced_confidence_u0400
                        or candidate_tail_quarter_confidence_u0400
                        or candidate_tail_bounded_confidence_u0400
                        or candidate_tail_elementwise_confidence_u0400
                        or candidate_gate_zero_offset_family
                    )
                    else (
                        "word_veto_candidate_set_attention_confidence_v33"
                        if candidate_set_attention_confidence_u0400
                        else (
                            "word_veto_candidate_normalized_confidence_v31"
                            if candidate_normalized_confidence_u0400
                            else "word_veto_candidate_absolute_confidence_v29"
                        )
                    )
                )
            )
        )
        if candidate_split_tail_aligned_confidence_u0400:
            expected_veto_revision = _V45_SPLIT_TAIL_ALIGNED_REVISION
        elif candidate_split_positive_tail_confidence_u0400:
            expected_veto_revision = _V46_SPLIT_POSITIVE_TAIL_REVISION
        elif candidate_split_boundary_routing_confidence_u0400:
            expected_veto_revision = _V47_SPLIT_BOUNDARY_ROUTING_REVISION
        elif candidate_split_fpr_active_set_confidence_u0400:
            expected_veto_revision = _V48_SPLIT_FPR_ACTIVE_SET_REVISION
        elif candidate_split_global_trust_veto_confidence_u0400:
            expected_veto_revision = _V49_SPLIT_GLOBAL_TRUST_VETO_REVISION
        elif candidate_split_strong_boundary_routing_confidence_u0400:
            expected_veto_revision = _V50_SPLIT_STRONG_BOUNDARY_ROUTING_REVISION
        elif candidate_split_independent_deployed_router_confidence_u0400:
            expected_veto_revision = _V51_SPLIT_INDEPENDENT_DEPLOYED_ROUTER_REVISION
        elif candidate_sample_calibrator_confidence_u0400:
            expected_veto_revision = _V52_CANDIDATE_SAMPLE_CALIBRATOR_REVISION
        elif fulltext_global_absolute_confidence_u0400:
            expected_veto_revision = _V53_FULLTEXT_GLOBAL_ABSOLUTE_REVISION
        elif fulltext_global_absolute_exact_residual_confidence_u0400:
            expected_veto_revision = (
                _V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_REVISION
            )
        elif fulltext_global_independent_absolute_confidence_u0400:
            expected_veto_revision = (
                _V55_FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_REVISION
            )
        elif deployment_owned_global_confidence_u0400:
            expected_veto_revision = (
                _V60_DEPLOYMENT_OWNED_QUERY_VETO_REVISION
                if deployment_owned_query_veto_confidence_u0400
                else _V59_DEPLOYMENT_OWNED_QUERY_GLOBAL_REVISION
                if deployment_owned_query_global_confidence_u0400
                else _V58_DEPLOYMENT_OWNED_STABLE_FPR95_ACTIVE_SET_REVISION
                if deployment_owned_global_stable_fpr95_active_set_confidence_u0400
                else (
                    _V57_DEPLOYED_GLOBAL_BALANCED_ABSOLUTE_REVISION
                    if deployed_global_balanced_absolute_confidence_u0400
                    else _V56_DEPLOYMENT_OWNED_GLOBAL_REVISION
                )
            )
    elif cross_attention_absolute_confidence_u0300:
        expected_veto_aggregation = (
            "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        )
        expected_veto_revision = (
            "word_veto_cross_attention_absolute_confidence_v28"
        )
    elif independent_absolute_confidence_u0300:
        expected_veto_aggregation = (
            "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        )
        expected_veto_revision = (
            "word_veto_independent_absolute_confidence_v27"
        )
    elif floor_gated_monotone_tail_veto_u0300:
        expected_veto_aggregation = (
            "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        )
        expected_veto_revision = (
            "word_veto_floor_gated_monotone_tail_veto_v26"
        )
    elif ungated_monotone_tail_veto_u0300:
        expected_veto_aggregation = (
            "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        )
        expected_veto_revision = "word_veto_ungated_monotone_tail_veto_v25"
    elif complementary_trust_veto_u0300:
        expected_veto_aggregation = (
            "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        )
        expected_veto_revision = "word_veto_complementary_trust_veto_v24"
    elif veto_token_conditioned_probe:
        expected_veto_aggregation = (
            "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        )
        expected_veto_revision = (
            "word_veto_token_conditioned_monotone_depth_v23"
        )
    elif veto_monotone_depth_u0050:
        expected_veto_aggregation = (
            "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        )
        expected_veto_revision = "word_veto_continuous_monotone_depth_v22"
    elif veto_continuous_residual_u0050:
        expected_veto_aggregation = (
            "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        )
        expected_veto_revision = (
            "word_veto_continuous_conditional_residual_v21"
        )
    elif veto_gated_pool_tail_paired_signed_rank_pool_u0050:
        expected_veto_aggregation = (
            "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        )
        expected_veto_revision = (
            "word_veto_gated_pool_tail_paired_signed_rank_pool_v20"
        )
    elif veto_gated_pool_tail_paired_rank_channel_u0050:
        expected_veto_aggregation = (
            "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        )
        expected_veto_revision = (
            "word_veto_gated_pool_tail_paired_rank_channel_v19"
        )
    elif tail_paired_fixed_probe:
        expected_veto_aggregation = (
            "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        )
        expected_veto_revision = "word_veto_gated_pool_tail_paired_v18"
    elif veto_gated_pool_tail_carrier_u0050:
        expected_veto_aggregation = (
            "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        )
        expected_veto_revision = "word_veto_gated_pool_tail_carrier_v17"
    elif veto_gated_pool_carrier_affine_u0050:
        expected_veto_aggregation = (
            "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        )
        expected_veto_revision = "word_veto_gated_pool_carrier_affine_v15"
    elif veto_gated_pool_carrier_pair_probe:
        expected_veto_aggregation = (
            "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        )
        expected_veto_revision = "word_veto_gated_pool_carrier_pair_v9"
    elif veto_gated_pool_carrier_quarter_probe:
        expected_veto_aggregation = (
            "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        )
        expected_veto_revision = "word_veto_gated_pool_carrier_quarter_v8"
    elif veto_gated_pool_carrier_balanced_probe:
        expected_veto_aggregation = (
            "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        )
        expected_veto_revision = "word_veto_gated_pool_carrier_balanced_v7"
    elif veto_gated_pool_calibrated_probe:
        expected_veto_aggregation = (
            "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        )
        expected_veto_revision = "word_veto_gated_pool_calibrated_v6"
    elif veto_gated_pool_probe:
        expected_veto_aggregation = (
            "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        )
        expected_veto_revision = "word_veto_gated_pool_absolute_cap_v5"
    elif veto_cap_probe:
        expected_veto_aggregation = "trace_activated_word_veto_absolute_cap_v4"
        expected_veto_revision = "word_veto_coverage_absolute_cap_v4"
    elif veto_gate_probe:
        expected_veto_aggregation = "trace_activated_word_veto_penalty_v2"
        expected_veto_revision = "word_veto_raw_gate_margin_v3"
    else:
        expected_veto_aggregation = "trace_activated_word_veto_product_v1"
        expected_veto_revision = "word_veto_net_trust_v1"
    if veto_probe and (
        str(getattr(cfg, "stage_b_dense_duty_execution_scope", "")) != "probe"
        or str(getattr(cfg, "stage_b_dense_duty_evaluation_scope", "")) != "probe"
        or int(
            getattr(
                cfg,
                "stage_b_dense_duty_confidence_expected_optimizer_updates",
                0,
            )
        )
        != (
            400
            if (
                candidate_calibrated_confidence_u0400
                or candidate_normalized_confidence_u0400
                or candidate_asymmetric_confidence_u0400
                or candidate_q05_confidence_u0400
                or candidate_tail_balanced_confidence_u0400
                or candidate_tail_quarter_confidence_u0400
                or candidate_tail_bounded_confidence_u0400
                or candidate_tail_elementwise_confidence_u0400
                or candidate_gate_zero_offset_family
                or candidate_set_attention_confidence_u0400
                or fulltext_global_absolute_confidence_u0400
                or fulltext_global_absolute_exact_residual_confidence_u0400
                or fulltext_global_independent_absolute_confidence_u0400
                or deployment_owned_global_confidence_u0400
            )
            else (
                600
                if candidate_absolute_confidence_u0600
                else (
                    100
                    if fixed_u0100_probe
                    else (50 if fixed_u0050_probe else 300)
                )
            )
        )
        or str(
            getattr(
                cfg,
                "stage_b_dense_duty_confidence_phrase_aggregation",
                "",
            )
        )
        != expected_veto_aggregation
        or str(
            getattr(cfg, "stage_b_dense_duty_confidence_revision", "")
        )
        != expected_veto_revision
        or str(
            getattr(
                cfg,
                "stage_b_dense_duty_confidence_pool_feature_contract",
                "",
            )
        ).strip()
        != (
            (
                "detached_rank_full_expression_token_conditioned_query_veto_"
                "deployment_owned_global_pool_v15"
                if deployment_owned_query_veto_confidence_u0400
                else (
                    "detached_rank_full_expression_monotone_query_"
                    "deployment_owned_global_pool_v14"
                    if deployment_owned_query_global_confidence_u0400
                else (
                    "detached_rank_full_expression_deployment_owned_global_pool_v13"
                    if deployment_owned_global_confidence_u0400
                    else (
                        "detached_rank_full_expression_local_candidate_"
                        "frozen_rank_global_pool_v12"
                        if fulltext_global_independent_absolute_confidence_u0400
                        else (
                            "detached_rank_full_expression_candidate_residual_global_pool_"
                            "exact_rank_max_reference_v11"
                            if fulltext_global_absolute_exact_residual_confidence_u0400
                            else (
                                "detached_rank_full_expression_candidate_residual_global_pool_v10"
                                if fulltext_global_absolute_confidence_u0400
                                else (
                                    "detached_candidate_absolute_patch_invariant_"
                                    "monotone_veto_logits_v6"
                                    if candidate_calibrated_confidence_u0400
                                    else (
                                        "detached_candidate_absolute_raw_patch_"
                                        "asymmetric_veto_logits_v8"
                                        if (
                                            candidate_asymmetric_confidence_u0400
                                            or candidate_q05_confidence_u0400
                                            or candidate_tail_balanced_confidence_u0400
                                            or candidate_tail_quarter_confidence_u0400
                                            or candidate_tail_bounded_confidence_u0400
                                            or candidate_tail_elementwise_confidence_u0400
                                            or candidate_gate_zero_offset_family
                                        )
                                        else (
                                            "detached_candidate_set_attention_absolute_"
                                            "asymmetric_veto_logits_v9"
                                            if candidate_set_attention_confidence_u0400
                                            else (
                                                "detached_candidate_absolute_normalized_patch_"
                                                "amplified_veto_logits_v7"
                                                if candidate_normalized_confidence_u0400
                                                else (
                                                    "detached_query_modifier_cross_attention_"
                                                    "candidate_absolute_logits_v5"
                                                )
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
                )
            )
            if candidate_absolute_confidence_probe
            else (
                "detached_rank_query_modifier_cross_attention_plus_"
                "patch_statistics_absolute_v4"
                if cross_attention_absolute_confidence_u0300
                else (
                    "detached_rank_query_token_context_plus_patch_statistics_monotone_v3"
                    if veto_token_conditioned_probe
                    else (
                        "detached_rank_query_plus_patch_statistics_signed_residual_v2"
                        if (
                            veto_gated_pool_tail_paired_signed_rank_pool_u0050
                            or veto_continuous_residual_u0050
                            or veto_monotone_depth_u0050
                        )
                        else "patch_statistics_only_v1"
                    )
                )
            )
        )
        or (
            (veto_gate_probe or absolute_cap_probe)
            and (
                float(
                    getattr(
                        cfg,
                        "stage_b_dense_duty_raw_veto_gate_weight",
                        0.0,
                    )
                    or 0.0
                )
                <= 0.0
                or float(
                    getattr(
                        cfg,
                        "stage_b_dense_duty_raw_veto_positive_margin",
                        0.0,
                    )
                    or 0.0
                )
                <= 0.0
                or float(
                    getattr(
                        cfg,
                        "stage_b_dense_duty_raw_veto_tn_margin",
                        0.0,
                    )
                    or 0.0
                )
                <= 0.0
            )
        )
        or float(
            getattr(
                cfg,
                "stage_b_dense_duty_confidence_word_softmin_temperature",
                -1.0,
            )
        )
        != 0.1
        or float(
            getattr(cfg, "stage_b_dense_duty_confidence_veto_gate_scale", -1.0)
        )
        != (
            0.03
            if (
                veto_gated_pool_calibrated_probe
                or veto_gated_pool_carrier_balanced_probe
                or veto_gated_pool_carrier_quarter_probe
                or veto_gated_pool_carrier_pair_probe
                or veto_gated_pool_carrier_affine_u0050
                or veto_gated_pool_tail_carrier_u0050
                or veto_gated_pool_tail_paired_u0050
                or veto_gated_pool_tail_paired_u0100
                or veto_gated_pool_tail_paired_rank_channel_u0050
                or veto_gated_pool_tail_paired_signed_rank_pool_u0050
                or veto_continuous_residual_u0050
                or veto_monotone_depth_u0050
                or veto_token_conditioned_probe
            )
            else (0.1 if absolute_cap_probe else 1.0)
        )
        or (
            absolute_cap_probe
            and (
                float(
                    getattr(
                        cfg,
                        "stage_b_dense_duty_raw_veto_gate_weight",
                        -1.0,
                    )
                )
                != (0.0 if candidate_complete_trace_c2_confidence_u0400 else 1.0)
                or float(
                    getattr(
                        cfg,
                        "stage_b_dense_duty_raw_veto_positive_margin",
                        -1.0,
                    )
                )
                != 0.1
                or float(
                    getattr(
                        cfg,
                        "stage_b_dense_duty_raw_veto_tn_margin",
                        -1.0,
                    )
                )
                != 0.15
                or str(
                    getattr(
                        cfg,
                        "stage_b_dense_duty_raw_veto_query_scope",
                        "",
                    )
                ).strip()
                != (
                    "tn_all_admitted_tail_weighted_carrier_tail_paired_v7"
                    if tail_paired_fixed_probe
                    else (
                        "tn_all_admitted_tail_weighted_carrier_positive_carrier_paired_v6"
                        if veto_gated_pool_tail_carrier_u0050
                        else (
                            "tn_all_admitted_carrier_balanced_positive_carrier_paired_v4"
                            if (
                                veto_gated_pool_carrier_pair_probe
                                or veto_gated_pool_carrier_affine_u0050
                            )
                            else (
                                "tn_all_admitted_carrier_balanced_positive_carrier_v3"
                                if (
                                    veto_gated_pool_carrier_balanced_probe
                                    or veto_gated_pool_carrier_quarter_probe
                                )
                                else "tn_all_admitted_positive_carrier_v2"
                            )
                        )
                    )
                )
                or float(
                    getattr(
                        cfg,
                        "stage_b_dense_duty_confidence_veto_gate_offset",
                        -1.0,
                    )
                )
                != (
                    0.0
                    if (
                        candidate_gate_zero_offset_family
                        or fulltext_global_absolute_confidence_u0400
                        or fulltext_global_absolute_exact_residual_confidence_u0400
                        or fulltext_global_independent_absolute_confidence_u0400
                        or deployment_owned_global_confidence_u0400
                    )
                    else (
                        0.02
                        if (
                            veto_gated_pool_calibrated_probe
                            or veto_gated_pool_carrier_balanced_probe
                            or veto_gated_pool_carrier_quarter_probe
                            or veto_gated_pool_carrier_pair_probe
                            or veto_gated_pool_carrier_affine_u0050
                            or veto_gated_pool_tail_carrier_u0050
                            or veto_gated_pool_tail_paired_u0050
                            or veto_gated_pool_tail_paired_u0100
                            or veto_gated_pool_tail_paired_rank_channel_u0050
                            or veto_gated_pool_tail_paired_signed_rank_pool_u0050
                            or veto_continuous_residual_u0050
                            or veto_monotone_depth_u0050
                            or veto_token_conditioned_probe
                        )
                        else 0.05
                    )
                )
                or float(
                    getattr(
                        cfg,
                        "stage_b_dense_duty_confidence_veto_coverage_offset",
                        -1.0,
                    )
                )
                != 0.1
                or float(
                    getattr(
                        cfg,
                        "stage_b_dense_duty_confidence_veto_coverage_ramp",
                        -1.0,
                    )
                )
                != 0.8
                or float(
                    getattr(
                        cfg,
                        "stage_b_dense_duty_confidence_veto_cap_temperature",
                        -1.0,
                    )
                )
                != 0.1
                or float(
                    getattr(
                        cfg,
                        "stage_b_dense_duty_confidence_veto_cap_initial_ceiling",
                        0.0,
                    )
                )
                != -0.1
                or (
                    (
                        veto_gated_pool_carrier_balanced_probe
                        or veto_gated_pool_carrier_quarter_probe
                        or veto_gated_pool_carrier_pair_probe
                        or veto_gated_pool_carrier_affine_u0050
                        or veto_gated_pool_tail_carrier_u0050
                        or veto_gated_pool_tail_paired_u0050
                        or veto_gated_pool_tail_paired_u0100
                        or veto_gated_pool_tail_paired_rank_channel_u0050
                        or veto_gated_pool_tail_paired_signed_rank_pool_u0050
                        or veto_continuous_residual_u0050
                        or veto_monotone_depth_u0050
                        or veto_token_conditioned_probe
                    )
                    and (
                        float(
                            getattr(
                                cfg,
                                "stage_b_dense_duty_raw_veto_tn_carrier_balance",
                                -1.0,
                            )
                        )
                        != (
                            0.25
                            if (
                                veto_gated_pool_carrier_quarter_probe
                                or veto_gated_pool_carrier_pair_probe
                                or veto_gated_pool_carrier_affine_u0050
                                or veto_gated_pool_tail_carrier_u0050
                                or veto_gated_pool_tail_paired_u0050
                                or veto_gated_pool_tail_paired_u0100
                                or veto_gated_pool_tail_paired_rank_channel_u0050
                                or veto_gated_pool_tail_paired_signed_rank_pool_u0050
                                or veto_continuous_residual_u0050
                                or veto_monotone_depth_u0050
                                or veto_token_conditioned_probe
                            )
                            else 0.5
                        )
                        or str(
                            getattr(
                                cfg,
                                "stage_b_dense_duty_confidence_carrier_selector_contract",
                                "",
                            )
                        ).strip()
                        != "final_layer_reference_argmax_exact_eligible_v1"
                        or (
                            (
                                veto_gated_pool_carrier_pair_probe
                                or veto_gated_pool_carrier_affine_u0050
                                or veto_gated_pool_tail_carrier_u0050
                                or veto_gated_pool_tail_paired_u0050
                                or veto_gated_pool_tail_paired_u0100
                                or veto_gated_pool_tail_paired_rank_channel_u0050
                                or veto_gated_pool_tail_paired_signed_rank_pool_u0050
                                or veto_continuous_residual_u0050
                                or veto_monotone_depth_u0050
                                or veto_token_conditioned_probe
                            )
                            and (
                                float(
                                    getattr(
                                        cfg,
                                        "stage_b_dense_duty_raw_veto_carrier_pair_weight",
                                        -1.0,
                                    )
                                )
                                != (
                                    0.0
                                    if candidate_complete_trace_c2_confidence_u0400
                                    else 0.25
                                )
                                or float(
                                    getattr(
                                        cfg,
                                        "stage_b_dense_duty_raw_veto_carrier_pair_margin",
                                        -1.0,
                                    )
                                )
                                != 0.25
                            )
                        )
                    )
                )
                or (
                    carrier_affine_fixed_probe
                    and (
                        str(
                            getattr(
                                cfg,
                                "stage_b_dense_duty_confidence_rank_evidence_contract",
                                "",
                            )
                        ).strip()
                        != (
                            "zero_init_carrier_token_rank_affine_sparse_rank_channel_v6"
                            if (
                                veto_gated_pool_tail_paired_rank_channel_u0050
                                or veto_gated_pool_tail_paired_signed_rank_pool_u0050
                                or veto_continuous_residual_u0050
                                or veto_monotone_depth_u0050
                                or veto_token_conditioned_probe
                            )
                            else "zero_init_carrier_token_rank_affine_v5"
                        )
                        or float(
                            getattr(
                                cfg,
                                "stage_b_dense_duty_confidence_residual_parameterization_gain",
                                -1.0,
                            )
                        )
                        != (25.0 / 3.0)
                    )
                )
                or (
                    tail_carrier_fixed_probe
                    and (
                        str(
                            getattr(
                                cfg,
                                "stage_b_dense_duty_confidence_gate_gradient_contract",
                                "",
                            )
                        ).strip()
                        != (
                            (
                                (
                                    (
                                        "token_conditioned_floor_gated_monotone_depth_v7"
                                        if floor_gated_monotone_tail_veto_u0300
                                        else (
                                            (
                                                (
                                                    "candidate_patch_invariant_monotone_veto_absolute_logit_v11"
                                                    if candidate_calibrated_confidence_u0400
                                                    else (
                                                        _V43_DEPLOYED_ROUTING_GATE_CONTRACT
                                                        if (
                                                            candidate_deployed_routing_confidence_u0400
                                                            or candidate_split_heads_confidence_u0400
                                                            or candidate_split_tail_aligned_confidence_u0400
                                                                or candidate_split_positive_tail_confidence_u0400
                                                                or candidate_split_boundary_routing_confidence_u0400
                                                                or candidate_split_fpr_active_set_confidence_u0400
                                                                or candidate_split_global_trust_veto_confidence_u0400
                                                                or candidate_split_strong_boundary_routing_confidence_u0400
                                                                or candidate_split_independent_deployed_router_confidence_u0400
                                                                or candidate_sample_calibrator_confidence_u0400
                                                        )
                                                        else (
                                                            "candidate_raw_patch_asymmetric_monotone_veto_absolute_logit_v13"
                                                            if (
                                                                candidate_asymmetric_confidence_u0400
                                                                or candidate_q05_confidence_u0400
                                                                or candidate_tail_balanced_confidence_u0400
                                                                or candidate_tail_quarter_confidence_u0400
                                                                or candidate_tail_bounded_confidence_u0400
                                                                or candidate_tail_elementwise_confidence_u0400
                                                                or candidate_gate_zero_offset_family
                                                                or fulltext_global_absolute_confidence_u0400
                                                                or fulltext_global_absolute_exact_residual_confidence_u0400
                                                                or fulltext_global_independent_absolute_confidence_u0400
                                                                or deployment_owned_global_confidence_u0400
                                                            )
                                                            else (
                                                                "candidate_set_attention_asymmetric_monotone_veto_absolute_logit_v14"
                                                                if candidate_set_attention_confidence_u0400
                                                                else (
                                                                    "candidate_normalized_patch_amplified_monotone_veto_absolute_logit_v12"
                                                                    if candidate_normalized_confidence_u0400
                                                                    else "candidate_cross_attention_independent_absolute_logit_v10"
                                                                )
                                                            )
                                                        )
                                                    )
                                                )
                                                if candidate_absolute_confidence_probe
                                                else (
                                                    "cross_attention_independent_absolute_logit_v9"
                                                    if cross_attention_absolute_confidence_u0300
                                                    else "token_conditioned_independent_absolute_logit_v8"
                                                )
                                            )
                                            if (
                                                independent_absolute_confidence_u0300
                                                or cross_attention_absolute_confidence_u0300
                                                or candidate_absolute_confidence_probe
                                            )
                                            else "token_conditioned_ungated_monotone_depth_v6"
                                        )
                                    )
                                    if (
                                        ungated_monotone_tail_veto_u0300
                                        or floor_gated_monotone_tail_veto_u0300
                                        or independent_absolute_confidence_u0300
                                        or cross_attention_absolute_confidence_u0300
                                        or candidate_absolute_confidence_probe
                                    )
                                    else (
                                        "continuous_sigmoid_complementary_trust_veto_v5"
                                        if complementary_trust_veto_u0300
                                        else "continuous_sigmoid_monotone_depth_v4"
                                    )
                                )
                            )
                            if (
                                veto_monotone_depth_u0050
                                or veto_token_conditioned_probe
                            )
                            else (
                                "continuous_sigmoid_v3"
                                if veto_continuous_residual_u0050
                                else "hard_detached_v1"
                            )
                        )
                        or float(
                            getattr(
                                cfg,
                                "stage_b_dense_duty_raw_veto_tail_quantile",
                                -1.0,
                            )
                        )
                        != 0.95
                        or float(
                            getattr(
                                cfg,
                                "stage_b_dense_duty_raw_veto_tail_temperature",
                                -1.0,
                            )
                        )
                        != 0.1
                        or int(
                            getattr(
                                cfg,
                                "stage_b_dense_duty_raw_veto_tail_min_count",
                                -1,
                            )
                        )
                        != 256
                    )
                )
            )
        )
        or str(
            getattr(cfg, "stage_b_dense_duty_positive_trust_contract", "")
        )
        != (
            "absolute_global_confidence_logit_v2"
            if (
                deployment_owned_query_global_confidence_u0400
                or deployment_owned_query_veto_confidence_u0400
            )
            else (
                "absolute_global_pool_logit_v4"
                if (
                fulltext_global_independent_absolute_confidence_u0400
                or deployment_owned_global_confidence_u0400
                )
                else (
                    "exact_frozen_rank_max_confidence_delta_v3"
                    if fulltext_global_absolute_exact_residual_confidence_u0400
                    else (
                        "absolute_global_confidence_logit_v2"
                        if (
                            independent_absolute_confidence_u0300
                            or cross_attention_absolute_confidence_u0300
                            or candidate_absolute_confidence_probe
                        )
                        else "net_total_confidence_delta_v1"
                    )
                )
            )
        )
        or str(getattr(cfg, "stage_b_dense_duty_confidence_tn_scope", ""))
        != "direct_trace_valid_v1"
        or not bool(getattr(cfg, "stage_b_v15_exclude_canonical_from_score", False))
        or str(
            getattr(
                cfg,
                "stage_b_dense_duty_confidence_probe_admission_contract",
                "",
            )
        )
        != "disabled_for_probe_v1"
        or str(
            getattr(
                cfg,
                "stage_b_dense_duty_confidence_probe_admission_report",
                "",
            )
            or ""
        )
        != ""
    ):
        errors.append("word-veto probe config contract is incomplete")
    if len(args.ckpts) != 1:
        errors.append("exactly one confidence checkpoint is required")
    if bool(args.skip_tn):
        errors.append("confidence diagnostics require TN evaluation")
    if veto_probe and not bool(args.skip_ref):
        errors.append("word-veto probe diagnostics require TN-only strict1607")
    if not bool(args.skip_ref):
        requested = tuple(str(value) for value in args.ref_splits)
        if len(requested) != len(CANONICAL_REF_SPLITS) or set(requested) != set(
            CANONICAL_REF_SPLITS
        ):
            errors.append(
                "a joint confidence diagnostic requires all eight official Ref "
                "splits exactly once"
            )
    tn_jsonl = getattr(args, "tn_jsonl", None)
    expected_tn_name = "strict1607" if bool(args.skip_ref) else "strict2031"
    if not tn_jsonl:
        errors.append(f"the fixed {expected_tn_name} TN manifest is required")
    else:
        expected_tn = _PARTIAL_CONFIDENCE_TN_SPECS[expected_tn_name]
        try:
            observed_tn_path = Path(tn_jsonl).expanduser().resolve(strict=True)
            expected_tn_path = Path(expected_tn["path"]).resolve(strict=True)
            observed_tn_sha256 = str(file_record(observed_tn_path)["sha256"])
        except (OSError, KeyError, RuntimeError, TypeError, ValueError):
            errors.append(f"the fixed {expected_tn_name} TN manifest is unavailable")
        else:
            if (
                observed_tn_path != expected_tn_path
                or observed_tn_sha256 != expected_tn["sha256"]
            ):
                errors.append(
                    f"the fixed {expected_tn_name} TN manifest path/SHA256 is required"
                )
    if list(getattr(args, "tn_splits", ())) != [
        "refcocop_val",
        "refcocog_umd_val",
    ]:
        errors.append("the fixed strict TN split filter is required")
    if (
        str(args.device) != "cuda:0"
        or int(args.batch_size) != 16
        or int(args.num_workers) != 4
        or int(args.seed) != 42
        or not bool(args.amp)
    ):
        errors.append("runtime must be cuda:0/B16/W4/AMP/seed42")
    if (
        list(args.topk) != [1]
        or list(args.threshold_tprs) != [0.75, 0.9, 0.95]
        or list(args.score_thresholds) != [0.5]
        or int(args.max_ref_batches) != 0
        or int(args.max_tn_batches) != 0
        or bool(args.no_per_example_records)
    ):
        errors.append(
            "Top1/fixed thresholds/full batches/per-example records are required"
        )
    if (
        args.screen_calibration_manifest
        or args.direct_prebuilt_tn
        or args.category_gate_max_gaps is not None
        or args.category_gate_include_base_expert
        or int(args.candidate_count_control) != 0
        or args.holdout_level != "none"
        or bool(args.exclude_train_jsonl)
    ):
        errors.append(
            "confidence diagnostics forbid calibration, sweeps, and filtering"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        errors.append("output_dir must be absent or empty")
    if errors:
        raise ValueError(
            "--partial_dense_duty_confidence_diagnostic contract failed: "
            + "; ".join(errors)
        )


def _validate_direct_prebuilt_tn_rows(
    rows: Sequence[Mapping[str, Any]], *, declared_scope: str
) -> str:
    if declared_scope != MATCHED_EVAL_SCOPE or not rows:
        raise ValueError("direct-prebuilt TN scope/rows are invalid")
    sample_ids = set()
    pair_ids = set()
    for index, row in enumerate(rows):
        if not (
            row.get("matched_eval_surface_schema") == MATCHED_EVAL_ROW_SCHEMA
            and row.get("tn_scope") == declared_scope
            and row.get("global_tn_verified") is False
            and row.get("proposal_covered_verified") is True
            and row.get("tn_eval_split") == "matched_calibration"
            and row.get("table_b_id") == "D3m"
        ):
            raise ValueError(
                f"direct-prebuilt TN row {index} changed its matched surface scope"
            )
        sample_id = str(row.get("sample_id") or "")
        pair_id = str(row.get("matched_pair_id") or "")
        if not sample_id or sample_id in sample_ids or not pair_id or pair_id in pair_ids:
            raise ValueError(
                f"direct-prebuilt TN row {index} has missing/duplicate identities"
            )
        sample_ids.add(sample_id)
        pair_ids.add(pair_id)
    return declared_scope


def _seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _support_input_identity(
    target: Mapping[str, Any], *, expected_class_id: Optional[int] = None
) -> Dict[str, Any]:
    kinds = [
        key
        for key in ("patch", "patches", "patch_global")
        if torch.is_tensor(target.get(key))
    ]
    if len(kinds) != 1:
        raise RuntimeError(
            "matched direct evaluation requires exactly one support tensor kind"
        )
    kind = kinds[0]
    tensor = target[kind].detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"kind": kind, "dtype": str(tensor.dtype), "shape": list(tensor.shape)},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    )
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    integer_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    if "support_classes" in target:
        raw_classes = target["support_classes"]
        if not torch.is_tensor(raw_classes):
            raise RuntimeError(
                "matched direct evaluation support_classes must be an integer tensor"
            )
        if raw_classes.dtype not in integer_dtypes or raw_classes.numel() < 1:
            raise RuntimeError(
                "matched direct evaluation support_classes must be a non-empty "
                "integer tensor"
            )
        class_ids = [
            int(value) for value in raw_classes.detach().cpu().view(-1).tolist()
        ]
    else:
        raw_class = target.get("support_class")
        if type(raw_class) is int:
            class_ids = [raw_class]
        elif torch.is_tensor(raw_class):
            if raw_class.dtype not in integer_dtypes or raw_class.numel() != 1:
                raise RuntimeError(
                    "matched direct evaluation support_class must be one integer"
                )
            class_ids = [int(raw_class.detach().cpu().item())]
        else:
            raise RuntimeError("matched direct evaluation lacks support class identity")
    if any(class_id < 0 for class_id in class_ids):
        raise RuntimeError(
            "matched direct evaluation support class IDs must be non-negative"
        )
    if expected_class_id is not None:
        if type(expected_class_id) is not int or expected_class_id < 0:
            raise RuntimeError(
                "matched direct evaluation manifest class_id must be a non-negative int"
            )
        if class_ids != [expected_class_id]:
            raise RuntimeError(
                "matched direct evaluation support class differs from its manifest row"
            )
    return {
        "support_input_kind": kind,
        "support_input_sha256": digest.hexdigest(),
        "support_class_ids": class_ids,
    }


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "run"


def _make_datasetinfo(
    data_root: Path,
    name: str,
    anno: Path,
    *,
    adapter_no_support: bool = False,
    u0_patch_rank: bool = False,
) -> Dict[str, Any]:
    if adapter_no_support and u0_patch_rank:
        raise ValueError("Stage-B U0 Ref evaluation requires its support patch")
    datasetinfo = {
        "name": name,
        "dataset_mode": "patch_episode",
        "root": "/",
        "anno": str(anno),
        "box_format": "xywh",
        "canonical_classes_json": str(data_root / "canonical_classes_with_aliases.json"),
        "keep_only_support_gt": True,
        "neg_episode_prob": 0.0,
        "support_min_count": 1 if adapter_no_support else 2,
        "support_patch_size": 224,
        "support_num_patches_min": 1,
        "support_num_patches_max": 1,
        "build_text_token_masks": True,
        "text_mask_skip_invalid_canonical": False,
        "text_mask_warn_limit": 0,
        "tn_balance_sampling": False,
    }
    if adapter_no_support:
        datasetinfo.update(
            stage_b_gdino_adapter_no_support=True,
            stage_b_gdino_adapter_ref_eval=True,
        )
    else:
        if u0_patch_rank:
            datasetinfo["stage_b_gdino_adapter_ref_eval"] = True
        datasetinfo.update(
            support_patch_tsv=str(
                data_root / "patches_quality_emb" / "emb_index_from_quality.tsv"
            ),
            support_patch_bucket="clean",
            support_patch_use_embedding=False,
            support_patch_image_root=str(data_root / "patches_quality"),
            support_patch_max_per_class=200,
            patch_emb_cache_size=4096,
        )
    return datasetinfo


def _adapter_ref_score_key(cfg) -> Optional[str]:
    adapter_enabled = bool(getattr(cfg, "stage_b_gdino_score_adapter", False))
    u0_patch_rank = bool(getattr(cfg, "stage_b_u0_patch_rank", False))
    data_driven_score = bool(getattr(cfg, "stage_b_data_driven_score", False))
    native_patch_category = bool(
        getattr(cfg, "stage_b_native_patch_category", False)
    )
    active_routes = sum(
        bool(value)
        for value in (
            adapter_enabled,
            data_driven_score,
            native_patch_category,
        )
    )
    if active_routes > 1:
        raise ValueError(
            "native/data-driven/GDINO-adapter evaluation score routes are "
            "mutually exclusive"
        )
    if u0_patch_rank and not adapter_enabled:
        raise ValueError(
            "stage_b_u0_patch_rank evaluation requires stage_b_gdino_score_adapter"
        )
    if u0_patch_rank:
        return "stage_b_u0_rank_score"
    if data_driven_score:
        return _DATA_DRIVEN_RANK_SCORE_KEY
    if adapter_enabled:
        if bool(getattr(cfg, "stage_b_gdino_ref_top1_guard", False)):
            if (
                getattr(cfg, "stage_b_gdino_ref_route_contract", None)
                != "b58_top1_anchored_rank_tail_v1"
            ):
                raise ValueError("guarded GDINO Ref route contract drifted")
            return "stage_b_gdino_ref_safe_rank_score"
        return "stage_b_gdino_rank_score"
    return None


def _validate_native_patch_category_joint_request(args, cfg) -> bool:
    native_patch_category = _validate_native_patch_category_ref_request(
        cfg,
        args.ckpts,
        extra_score_source_requested=bool(
            getattr(args, "category_gate_max_gaps", None) is not None
            or getattr(args, "category_gate_include_base_expert", False)
        ),
    )
    if not native_patch_category:
        return False
    if not bool(getattr(args, "skip_tn", False)) and not bool(
        getattr(cfg, "stage_b_native_patch_confidence_trained", False)
    ):
        raise RuntimeError(
            "native patch-category official evaluation is Ref-only until an "
            "independent confidence head is explicitly marked trained; pass --skip_tn"
        )
    return True


def _normalize_category_gate_sweep_gaps(
    values: Optional[Sequence[float]],
) -> Optional[Tuple[float, ...]]:
    if values is None:
        return None
    gaps = tuple(float(value) for value in values)
    if len(gaps) < 2:
        raise ValueError("category gate sweep requires at least two max_gap values")
    if any(not math.isfinite(gap) or gap < 0.0 for gap in gaps):
        raise ValueError("category gate sweep max_gap values must be finite and non-negative")
    normalized = tuple(0.0 if gap == 0.0 else gap for gap in gaps)
    if len(set(normalized)) != len(normalized):
        raise ValueError("category gate sweep max_gap values must be unique")
    slugs = [_category_gate_gap_slug(gap) for gap in normalized]
    if len(set(slugs)) != len(slugs):
        raise ValueError("category gate sweep max_gap labels collide")
    return normalized


def _category_gate_gap_slug(max_gap: float) -> str:
    rendered = format(float(max_gap), ".12g")
    return (
        rendered.replace("-", "m")
        .replace("+", "p")
        .replace(".", "p")
    )


def _validate_category_gate_sweep_mode(
    args: argparse.Namespace,
    cfg,
    ref_splits: Sequence[str],
) -> Optional[Tuple[float, ...]]:
    gaps = _normalize_category_gate_sweep_gaps(
        getattr(args, "category_gate_max_gaps", None)
    )
    if gaps is None:
        if bool(getattr(args, "category_gate_include_base_expert", False)):
            raise ValueError(
                "category gate base expert requires --category_gate_max_gaps"
            )
        return None
    errors: List[str] = []
    if bool(getattr(args, "skip_ref", False)):
        errors.append("Ref evaluation cannot be skipped")
    if not bool(getattr(args, "skip_tn", False)):
        errors.append("TN evaluation must be skipped")
    if bool(getattr(args, "no_per_example_records", False)):
        errors.append("per-example records are required")
    if not bool(getattr(cfg, "stage_b_u0_patch_rank", False)):
        errors.append("stage_b_u0_patch_rank must be enabled")
    if not bool(
        getattr(cfg, "stage_b_u0_category_preserving_patch_gate", False)
    ):
        errors.append("category-preserving patch gate config must be enabled")
    invalid_splits = [
        str(name) for name in ref_splits if not str(name).endswith("_val")
    ]
    if invalid_splits:
        errors.append(f"only Ref val splits are allowed: {invalid_splits}")
    if not ref_splits:
        errors.append("at least one Ref val split is required")
    if errors:
        raise ValueError("category gate sweep contract failed: " + "; ".join(errors))
    return gaps


def _category_gate_sweep_query_scores(
    outputs: Mapping[str, Any],
    max_gaps: Sequence[float],
    *,
    rank_score_key: str = "stage_b_u0_teacher_rank_score",
) -> List[Tuple[float, torch.Tensor, torch.Tensor]]:
    gaps = _normalize_category_gate_sweep_gaps(max_gaps)
    if gaps is None:
        raise ValueError("category gate sweep max_gap values are required")
    required = {
        rank_score_key,
        "stage_b_u0_category_gate_patch_score",
        "stage_b_u0_candidate_mask",
    }
    missing = sorted(required.difference(outputs))
    if missing:
        raise KeyError(f"category gate sweep outputs are missing {missing}")
    teacher = outputs[rank_score_key]
    patch_score = outputs["stage_b_u0_category_gate_patch_score"]
    candidate_mask = outputs["stage_b_u0_candidate_mask"]
    if not all(torch.is_tensor(value) for value in (teacher, patch_score, candidate_mask)):
        raise TypeError("category gate sweep outputs must be tensors")
    if teacher.dim() != 2 or tuple(patch_score.shape) != tuple(teacher.shape):
        raise ValueError("category gate teacher and patch scores must have shape (B,Q)")
    if tuple(candidate_mask.shape) != tuple(teacher.shape):
        raise ValueError("category gate candidate mask must align with scores")
    if candidate_mask.dtype != torch.bool:
        raise TypeError("category gate candidate mask must be boolean")
    if not teacher.is_floating_point() or not patch_score.is_floating_point():
        raise TypeError("category gate teacher and patch scores must be floating point")
    if not (
        teacher.device == patch_score.device == candidate_mask.device
    ):
        raise ValueError("category gate outputs must share a device")
    mask = candidate_mask
    if bool((~mask.any(dim=1)).any().item()):
        raise ValueError("every category gate row must contain a candidate")
    if not bool(torch.isfinite(teacher[mask]).all().item()):
        raise ValueError("category gate teacher scores must be finite")
    if not bool(torch.isfinite(patch_score[mask]).all().item()):
        raise ValueError("category gate patch scores must be finite")

    best_patch = patch_score.masked_fill(~mask, -torch.inf).amax(
        dim=1, keepdim=True
    )
    teacher_min = teacher.masked_fill(~mask, torch.inf).amin(
        dim=1, keepdim=True
    )
    teacher_max = teacher.masked_fill(~mask, -torch.inf).amax(
        dim=1, keepdim=True
    )
    below_teacher_min = torch.nextafter(
        teacher_min, torch.full_like(teacher_min, -torch.inf)
    )
    if not bool(torch.isfinite(below_teacher_min).all().item()):
        raise RuntimeError(
            "cannot construct a finite category gate score below teacher minimum"
        )
    teacher_delta = torch.where(mask, teacher, teacher_max) - teacher_max
    ineligible_score = below_teacher_min + teacher_delta
    if not bool(torch.isfinite(ineligible_score).all().item()):
        raise RuntimeError("category gate demotion produced non-finite scores")

    result: List[Tuple[float, torch.Tensor, torch.Tensor]] = []
    for gap in gaps:
        eligible = mask & (best_patch - patch_score <= float(gap))
        if bool((~eligible.any(dim=1)).any().item()):
            raise RuntimeError("category gate sweep produced an empty eligible row")
        scores = torch.where(eligible, teacher, ineligible_score)
        if not torch.equal(scores[eligible], teacher[eligible]):
            raise RuntimeError("category gate changed an eligible teacher score")
        for row in range(int(scores.shape[0])):
            rejected = ~eligible[row]
            if bool(rejected.any().item()) and not bool(
                (
                    scores[row, rejected]
                    < teacher[row, eligible[row]].min()
                ).all().item()
            ):
                raise RuntimeError("category gate failed strict eligibility precedence")
        result.append((float(gap), scores, eligible))
    return result


def _build_loader(
    cfg,
    datasetinfo: Dict[str, Any],
    batch_size: int,
    num_workers: int,
    device: torch.device,
    seed: int,
) -> DataLoader:
    _set_seed(seed)
    dataset_cfg = cfg
    if bool(getattr(cfg, "stage_b_data_driven_score", False)):
        # The data-driven receipt/variant contract describes training manifests.
        # Benchmark JSONLs are independently sealed by load_eval_manifest below.
        dataset_cfg = copy.copy(cfg)
        dataset_cfg.stage_b_data_driven_score = False
    dataset = build_dataset(
        image_set="val", args=dataset_cfg, datasetinfo=datasetinfo
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        sampler=SequentialSampler(dataset),
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
        worker_init_fn=_seed_worker,
        generator=generator,
    )


def _first_mask(target: Dict[str, Any], key: str, device: torch.device, tmax: int) -> Optional[torch.Tensor]:
    mask = target.get(key)
    if not torch.is_tensor(mask):
        return None
    if mask.dim() == 2:
        mask = mask[0]
    elif mask.dim() != 1:
        return None
    out = torch.zeros((tmax,), dtype=torch.bool, device=device)
    cols = min(tmax, int(mask.shape[-1]))
    if cols > 0:
        out[:cols] = mask[:cols].to(device=device, dtype=torch.bool)
    if not bool(out.any().item()):
        return None
    return out


def _phrase_scores(
    outputs: Dict[str, torch.Tensor],
    targets: List[Dict[str, Any]],
    mask_key: str,
    *,
    adapter_score_key: Optional[str] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if adapter_score_key in {
        _DATA_DRIVEN_RANK_SCORE_KEY,
        _DATA_DRIVEN_CONFIDENCE_SCORE_KEY,
    }:
        boxes = outputs.get("pred_boxes")
        score = outputs.get(adapter_score_key)
        expression_mask = outputs.get(
            "stage_b_data_driven_expression_token_mask"
        )
        if not torch.is_tensor(boxes) or boxes.dim() != 3:
            raise RuntimeError(
                "data-driven evaluation requires pred_boxes with shape (B,Q,4)"
            )
        expected_shape = tuple(boxes.shape[:2])
        if not torch.is_tensor(score) or tuple(score.shape) != expected_shape:
            shape = tuple(score.shape) if torch.is_tensor(score) else type(score).__name__
            raise ValueError(
                f"{adapter_score_key} must have shape {expected_shape}, got {shape}"
            )
        if (
            not torch.is_tensor(expression_mask)
            or expression_mask.dim() != 2
            or int(expression_mask.shape[0]) != int(boxes.shape[0])
            or bool((~expression_mask.to(dtype=torch.bool).any(dim=1)).any().item())
        ):
            raise RuntimeError(
                "data-driven evaluation requires one non-empty independent "
                "full-expression token mask per row"
            )
        score = score.detach().float()
        if not bool(torch.isfinite(score).all().item()):
            raise ValueError(f"{adapter_score_key} contains non-finite values")
        return score, torch.ones(
            (int(score.shape[0]),), dtype=torch.bool, device=score.device
        )

    logits = outputs["pred_logits"].detach().float()
    boxes = outputs["pred_boxes"].detach().float()
    device = logits.device
    bsz, num_queries, tmax = logits.shape
    if adapter_score_key is not None:
        if adapter_score_key not in outputs:
            raise KeyError(
                f"Stage-B GDINO adapter output is missing {adapter_score_key}"
            )
        expression_mask = outputs.get(
            "stage_b_gdino_expression_token_mask", None
        )
        if (
            not torch.is_tensor(expression_mask)
            or expression_mask.dim() != 2
            or tuple(expression_mask.shape) != (bsz, tmax)
        ):
            raise RuntimeError(
                "Stage-B GDINO adapter evaluation requires its generated "
                "full-expression token mask"
            )
        expression_mask = expression_mask.to(device=device, dtype=torch.bool)
        token_count = expression_mask.sum(dim=-1)
        if bool((token_count == 0).any().item()):
            raise RuntimeError(
                "Stage-B GDINO adapter generated an empty expression token mask"
            )
        authoritative = (
            logits.sigmoid()
            .masked_fill(~expression_mask[:, None, :], 0.0)
            .sum(dim=-1)
            / token_count[:, None].to(dtype=torch.float32)
        )
        adapter_base = outputs.get("stage_b_gdino_base_score", None)
        if (
            not torch.is_tensor(adapter_base)
            or adapter_base.shape != authoritative.shape
        ):
            raise RuntimeError(
                "Stage-B GDINO adapter evaluation requires an aligned base score"
            )
        adapter_base = adapter_base.detach().float()
        if not torch.equal(adapter_base, authoritative):
            max_error = float((adapter_base - authoritative).abs().max().item())
            raise RuntimeError(
                "Stage-B GDINO adapter base score drifted from the authoritative "
                f"pure-GDINO aggregation (max_abs_error={max_error:g})"
            )
        adapter_score = outputs[adapter_score_key]
        if not torch.is_tensor(adapter_score):
            raise TypeError(f"{adapter_score_key} must be a tensor")
        adapter_score = adapter_score.detach().float()
        if adapter_score.shape != authoritative.shape:
            raise ValueError(
                f"{adapter_score_key} must have shape {tuple(authoritative.shape)}, "
                f"got {tuple(adapter_score.shape)}"
            )
        return adapter_score, torch.ones((bsz,), dtype=torch.bool, device=device)

    scores = torch.full((bsz, num_queries), -1.0, dtype=torch.float32, device=device)
    valid = torch.zeros((bsz,), dtype=torch.bool, device=device)
    for b, target in enumerate(targets):
        mask = _first_mask(target, mask_key, device, tmax)
        if mask is None:
            continue
        valid[b] = True
        denom = mask.to(torch.float32).sum().clamp(min=1.0)
        scores[b] = logits[b].sigmoid().masked_fill(~mask[None, :], 0.0).sum(dim=-1) / denom
    del boxes
    return scores, valid


def _single_post_candidate_slot_scores(
    outputs: Dict[str, torch.Tensor],
    cfg,
    *,
    branch: str,
) -> torch.Tensor:
    if branch == "rank":
        slot_scores = _stage_b_ref_slot_scores(outputs, cfg, 1.0)
    elif branch == "confidence":
        slot_scores = _stage_b_tn_slot_scores(outputs, cfg, 1.0)
    else:
        raise ValueError(f"unsupported post-candidate score branch: {branch!r}")
    if not torch.is_tensor(slot_scores) or slot_scores.dim() != 3:
        raise RuntimeError(
            f"Stage-B post-candidate {branch} score must be a (B,Q,K) tensor"
        )
    if int(slot_scores.shape[-1]) != 1:
        raise RuntimeError(
            f"Stage-B post-candidate {branch} evaluation requires exactly one "
            f"expression slot, got K={int(slot_scores.shape[-1])}"
        )
    boxes = outputs.get("pred_boxes")
    if not torch.is_tensor(boxes) or boxes.dim() != 3:
        raise RuntimeError(
            f"Stage-B post-candidate {branch} evaluation requires pred_boxes (B,Q,4)"
        )
    query_scores = slot_scores[..., 0].detach().float()
    if tuple(query_scores.shape) != tuple(boxes.shape[:2]):
        raise RuntimeError(
            f"Stage-B post-candidate {branch} score shape {tuple(query_scores.shape)} "
            f"does not align with pred_boxes {tuple(boxes.shape[:2])}"
        )
    if not bool(torch.isfinite(query_scores).all().item()):
        raise RuntimeError(
            f"Stage-B post-candidate {branch} score contains non-finite values"
        )
    return query_scores


def _forward_ref_batch(cfg, model, batch, device: torch.device, *, amp: bool):
    adapter_enabled = bool(getattr(cfg, "stage_b_gdino_score_adapter", False))
    u0_patch_rank = bool(getattr(cfg, "stage_b_u0_patch_rank", False))
    data_driven_score = bool(getattr(cfg, "stage_b_data_driven_score", False))
    native_patch_category = bool(
        getattr(cfg, "stage_b_native_patch_category", False)
    )
    if u0_patch_rank and not adapter_enabled:
        raise ValueError(
            "stage_b_u0_patch_rank evaluation requires stage_b_gdino_score_adapter"
        )
    if u0_patch_rank and not _is_stage_b_u0_model(model):
        raise RuntimeError("Stage-B U0 Ref evaluation model is missing its U0 head")
    if native_patch_category:
        outputs, targets = _stage_b_ref_forward(
            model, batch, device, amp=amp, cfg=cfg
        )
        return (
            outputs,
            targets,
            _single_post_candidate_slot_scores(
                outputs, cfg, branch="rank"
            ),
        )
    if data_driven_score:
        outputs, targets = _stage_b_ref_forward(
            model, batch, device, amp=amp, cfg=cfg
        )
        return outputs, targets, None
    if _uses_stage_b_post_candidate_scorer(cfg) and not adapter_enabled:
        outputs, targets = _stage_b_ref_forward(
            model, batch, device, amp=amp, cfg=cfg
        )
        return (
            outputs,
            targets,
            _single_post_candidate_slot_scores(
                outputs, cfg, branch="rank"
            ),
        )

    if u0_patch_rank:
        (
            samples,
            targets,
            captions,
            patches,
            patch_global,
            patch_mask,
        ) = _prepare_stage_b_u0_patch_batch(batch, device)
        raw_targets = list(batch[1])
        for index, target in enumerate(raw_targets):
            caption = target.get("caption")
            if not isinstance(caption, str) or not caption.strip():
                raise KeyError(
                    f"Stage-B U0 Ref target {index} requires a non-empty caption"
                )
        captions = [str(caption) for caption in captions]
    else:
        samples, raw_targets = batch
        samples = samples.to(device)
        targets = list(raw_targets)
        captions = [str(target.get("caption", "object .")) for target in targets]
        patches = None
        patch_global = None
        patch_mask = None
    with torch.cuda.amp.autocast(enabled=bool(amp) and device.type == "cuda"):
        if u0_patch_rank:
            outputs = model(
                samples,
                targets=targets,
                captions=captions,
                patches=patches,
                patch_global=patch_global,
                patch_mask=patch_mask,
                patch_only=False,
                disable_patch_dn=True,
            )
        else:
            outputs = model(samples, captions=captions)
    return outputs, targets, None


def _forward_tn_batch(cfg, model, batch, device: torch.device, *, amp: bool):
    raw_targets = list(batch[1])
    adapter_enabled = bool(getattr(cfg, "stage_b_gdino_score_adapter", False))
    u0_patch_rank = bool(getattr(cfg, "stage_b_u0_patch_rank", False))
    data_driven_score = bool(getattr(cfg, "stage_b_data_driven_score", False))
    native_patch_category = bool(
        getattr(cfg, "stage_b_native_patch_category", False)
    )
    if u0_patch_rank and not adapter_enabled:
        raise ValueError(
            "stage_b_u0_patch_rank evaluation requires stage_b_gdino_score_adapter"
        )
    if u0_patch_rank and not _is_stage_b_u0_model(model):
        raise RuntimeError("Stage-B U0 TN evaluation model is missing its U0 head")
    if native_patch_category:
        if not bool(
            getattr(cfg, "stage_b_native_patch_confidence_trained", False)
        ):
            raise RuntimeError(
                "native patch-category TN/FPR evaluation is forbidden until an "
                "independent confidence head is explicitly marked trained"
            )
        neg_outputs, pos_outputs, targets, valid_pos = _stage_b_tn_forward_pair(
            model, batch, device, amp=amp
        )
        neg_scores = _stage_b_tn_slot_scores(neg_outputs, cfg, 0.0)[..., 0]
        pos_scores = _stage_b_tn_slot_scores(pos_outputs, cfg, 0.0)[..., 0]
        valid_neg = torch.ones(
            (int(neg_scores.shape[0]),),
            dtype=torch.bool,
            device=neg_scores.device,
        )
        valid_pos_mask = torch.ones_like(valid_neg)
        return (
            neg_outputs,
            pos_outputs,
            targets,
            valid_pos.to(device=neg_scores.device, dtype=torch.bool),
            neg_scores,
            valid_neg,
            pos_scores,
            valid_pos_mask,
        )
    if data_driven_score:
        if not bool(
            getattr(cfg, "stage_b_data_driven_confidence_trained", False)
        ):
            raise RuntimeError(
                "data-driven TN/FPR evaluation is forbidden before DD2 trains "
                "the independent confidence head"
            )
        neg_outputs, pos_outputs, targets, valid_pos = _stage_b_tn_forward_pair(
            model, batch, device, amp=amp
        )
        neg_scores, valid_neg = _phrase_scores(
            neg_outputs,
            raw_targets,
            "phrase_to_token_mask",
            adapter_score_key=_DATA_DRIVEN_CONFIDENCE_SCORE_KEY,
        )
        pos_scores, valid_pos_mask = _phrase_scores(
            pos_outputs,
            raw_targets,
            "rank_positive_phrase_to_token_mask",
            adapter_score_key=_DATA_DRIVEN_CONFIDENCE_SCORE_KEY,
        )
        return (
            neg_outputs,
            pos_outputs,
            targets,
            valid_pos.to(device=valid_neg.device, dtype=torch.bool),
            neg_scores,
            valid_neg,
            pos_scores,
            valid_pos_mask,
        )
    # U2-v2 confidence is owned by the standalone frozen C100 path. Preserve
    # its separate negative/positive forwards; the legacy U0 helper packs a
    # 2B forward and changes cross-example image padding numerically.
    if u0_patch_rank and not bool(getattr(cfg, "stage_b_u2v2", False)):
        neg_outputs, pos_outputs, targets, valid_pos = _stage_b_tn_forward_pair(
            model, batch, device, amp=amp
        )
        neg_scores, valid_neg = _phrase_scores(
            neg_outputs,
            raw_targets,
            "phrase_to_token_mask",
            adapter_score_key="stage_b_gdino_confidence_score",
        )
        pos_scores, valid_pos_mask = _phrase_scores(
            pos_outputs,
            raw_targets,
            "rank_positive_phrase_to_token_mask",
            adapter_score_key="stage_b_gdino_confidence_score",
        )
        return (
            neg_outputs,
            pos_outputs,
            targets,
            valid_pos.to(device=valid_neg.device, dtype=torch.bool),
            neg_scores,
            valid_neg,
            pos_scores,
            valid_pos_mask,
        )
    if _uses_stage_b_post_candidate_scorer(cfg) and not adapter_enabled:
        neg_outputs, pos_outputs, targets, valid_pos = _stage_b_tn_forward_pair(
            model, batch, device, amp=amp
        )
        neg_scores = _single_post_candidate_slot_scores(
            neg_outputs, cfg, branch="confidence"
        )
        pos_scores = _single_post_candidate_slot_scores(
            pos_outputs, cfg, branch="confidence"
        )
        valid_neg = torch.ones(
            (int(neg_scores.shape[0]),), dtype=torch.bool, device=neg_scores.device
        )
        valid_pos_mask = torch.ones_like(valid_neg)
        return (
            neg_outputs,
            pos_outputs,
            targets,
            valid_pos.to(device=neg_scores.device, dtype=torch.bool),
            neg_scores,
            valid_neg,
            pos_scores,
            valid_pos_mask,
        )

    samples = batch[0].to(device)
    targets = raw_targets
    if adapter_enabled:
        pos_captions, neg_captions, valid_pos = extract_adapter_tn_pair_captions(
            targets
        )
    else:
        neg_captions = [
            str(target.get("caption", "object .")) for target in targets
        ]
        pos_captions, valid_pos = _positive_captions(targets)
    with torch.cuda.amp.autocast(enabled=bool(amp) and device.type == "cuda"):
        u2v2_confidence_kwargs = (
            {"stage_b_u2v2_confidence_only": True}
            if bool(getattr(cfg, "stage_b_u2v2", False)) else {}
        )
        neg_outputs = model(
            samples, captions=neg_captions, **u2v2_confidence_kwargs
        )
        pos_outputs = model(
            samples, captions=pos_captions, **u2v2_confidence_kwargs
        )
    neg_scores, valid_neg = _phrase_scores(
        neg_outputs,
        targets,
        "phrase_to_token_mask",
        adapter_score_key=(
            "stage_b_gdino_confidence_score" if adapter_enabled else None
        ),
    )
    pos_scores, valid_pos_mask = _phrase_scores(
        pos_outputs,
        targets,
        "rank_positive_phrase_to_token_mask",
        adapter_score_key=(
            "stage_b_gdino_confidence_score" if adapter_enabled else None
        ),
    )
    return (
        neg_outputs,
        pos_outputs,
        targets,
        valid_pos.to(device=valid_neg.device, dtype=torch.bool),
        neg_scores,
        valid_neg,
        pos_scores,
        valid_pos_mask,
    )


def _validate_adapter_tn_eval_scope(
    cfg, rows: List[Dict[str, Any]]
) -> Optional[str]:
    return _validate_adapter_tn_eval_manifest(cfg, rows)


def _top_iou(outputs: Dict[str, torch.Tensor], targets: List[Dict[str, Any]], scores: torch.Tensor) -> np.ndarray:
    pred_xyxy = box_ops.box_cxcywh_to_xyxy(outputs["pred_boxes"].detach().float()).clamp(0.0, 1.0)
    topq = scores.argmax(dim=1)
    ious: List[float] = []
    for b, target in enumerate(targets):
        gt_boxes = target.get("boxes")
        if (not torch.is_tensor(gt_boxes)) or gt_boxes.numel() == 0:
            ious.append(float("nan"))
            continue
        gt = box_ops.box_cxcywh_to_xyxy(gt_boxes[:1].to(pred_xyxy.device).detach().float()).clamp(0.0, 1.0)[0]
        q = int(topq[b].item())
        iou = box_ops.box_iou(pred_xyxy[b, q : q + 1], gt.view(1, 4))[0].view(-1)[0]
        ious.append(float(iou.item()))
    return np.asarray(ious, dtype=np.float32)


class RefCocoTextAccumulator:
    def __init__(
        self,
        topks: Iterable[int],
        *,
        manifest: EvalManifest,
        run_id: str,
        adapter_score_key: Optional[str] = None,
    ) -> None:
        self.topks = sorted({max(1, int(k)) for k in topks})
        self.manifest = manifest
        self.run_id = str(run_id)
        self.adapter_score_key = adapter_score_key
        self.total = 0
        self.valid_masks = 0
        self.correct50 = {k: 0 for k in self.topks}
        self.correct25 = {k: 0 for k in self.topks}
        self.iou_sum = {k: 0.0 for k in self.topks}
        self.all_query_correct50 = 0
        self.all_query_correct25 = 0
        self.all_query_iou_sum = 0.0
        self.eligible_query_correct50 = 0
        self.eligible_query_iou_sum = 0.0
        self.eligible_query_rows = 0
        self.eval_records: List[Dict[str, Any]] = []

    def update(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, Any]],
        *,
        query_scores: Optional[torch.Tensor] = None,
        record_extras: Optional[Sequence[Mapping[str, Any]]] = None,
        eligibility_mask: Optional[torch.Tensor] = None,
    ) -> None:
        start_index = int(self.total)
        if query_scores is None:
            scores, valid = _phrase_scores(
                outputs,
                targets,
                "phrase_to_token_mask",
                adapter_score_key=self.adapter_score_key,
            )
        else:
            scores = query_scores.detach().float()
            if tuple(scores.shape) != tuple(outputs["pred_boxes"].shape[:2]):
                raise RuntimeError(
                    "post-candidate Ref query scores must align with pred_boxes"
                )
            if not bool(torch.isfinite(scores).all().item()):
                raise RuntimeError("post-candidate Ref query scores must be finite")
            valid = torch.ones(
                (int(scores.shape[0]),), dtype=torch.bool, device=scores.device
            )
        pred_xyxy = box_ops.box_cxcywh_to_xyxy(outputs["pred_boxes"].detach().float()).clamp(0.0, 1.0)
        bsz, num_queries = scores.shape
        if record_extras is not None and len(record_extras) != int(bsz):
            raise ValueError("Ref record extras must align with the batch")
        if eligibility_mask is not None:
            eligibility_mask = torch.as_tensor(
                eligibility_mask, device=scores.device, dtype=torch.bool
            )
            if tuple(eligibility_mask.shape) != tuple(scores.shape) or bool(
                (~eligibility_mask.any(dim=1)).any().item()
            ):
                raise ValueError("Ref eligibility mask must be aligned and nonempty")

        def record_values(index: int, base: Dict[str, Any]) -> Dict[str, Any]:
            if record_extras is None:
                return base
            extras = dict(record_extras[index])
            collisions = sorted(set(base).intersection(extras))
            if collisions:
                raise ValueError(
                    f"Ref record extras collide with metric fields: {collisions}"
                )
            base.update(extras)
            return base

        max_topk = min(max(self.topks), int(num_queries))
        top_idx = torch.topk(scores, k=max_topk, dim=1, largest=True).indices
        for b, target in enumerate(targets):
            self.total += 1
            if not bool(valid[b].item()):
                self.eval_records.append(
                    make_eval_record(
                        self.manifest,
                        index=start_index + b,
                        run_id=self.run_id,
                        valid=False,
                        values=record_values(
                            b,
                            {
                                "top1_iou": None,
                                "all_query_best_iou": None,
                                "correct50": None,
                            },
                        ),
                    )
                )
                continue
            self.valid_masks += 1
            gt_boxes = target.get("boxes")
            if (not torch.is_tensor(gt_boxes)) or gt_boxes.numel() == 0:
                self.eval_records.append(
                    make_eval_record(
                        self.manifest,
                        index=start_index + b,
                        run_id=self.run_id,
                        valid=False,
                        values=record_values(
                            b,
                            {
                                "top1_iou": None,
                                "all_query_best_iou": None,
                                "correct50": None,
                            },
                        ),
                    )
                )
                continue
            gt = box_ops.box_cxcywh_to_xyxy(gt_boxes[:1].to(pred_xyxy.device).detach().float()).clamp(0.0, 1.0)[0]
            all_query_ious = box_ops.box_iou(
                pred_xyxy[b], gt.view(1, 4)
            )[0][:, 0]
            query_ious = all_query_ious.index_select(0, top_idx[b])
            all_query_best_iou = float(all_query_ious.max().item())
            self.all_query_iou_sum += all_query_best_iou
            if all_query_best_iou >= 0.5:
                self.all_query_correct50 += 1
            if all_query_best_iou >= 0.25:
                self.all_query_correct25 += 1
            eligible_best_iou = None
            if eligibility_mask is not None:
                eligible_ious = all_query_ious[eligibility_mask[b]]
                eligible_best_iou = float(eligible_ious.max().item())
                self.eligible_query_rows += 1
                self.eligible_query_iou_sum += eligible_best_iou
                if eligible_best_iou >= 0.5:
                    self.eligible_query_correct50 += 1
            ranked_best_iou: Dict[str, float] = {}
            for k in self.topks:
                local = query_ious[: min(k, int(query_ious.numel()))]
                best_iou = float(local.max().item()) if local.numel() else 0.0
                ranked_best_iou[str(k)] = best_iou
                self.iou_sum[k] += best_iou
                if best_iou >= 0.5:
                    self.correct50[k] += 1
                if best_iou >= 0.25:
                    self.correct25[k] += 1
            top1_iou = (
                float(query_ious[0].item()) if query_ious.numel() else float("nan")
            )
            record_valid = math.isfinite(top1_iou)
            self.eval_records.append(
                make_eval_record(
                    self.manifest,
                    index=start_index + b,
                    run_id=self.run_id,
                    valid=record_valid,
                    values=record_values(
                        b,
                        {
                            "top1_iou": top1_iou,
                            "all_query_best_iou": all_query_best_iou,
                            "correct50": (
                                bool(top1_iou >= 0.5) if record_valid else None
                            ),
                            "ranked_best_iou": ranked_best_iou,
                            "eligible_query_best_iou": eligible_best_iou,
                        },
                    ),
                )
            )

    def result(self) -> Dict[str, Any]:
        denom = max(1, int(self.valid_masks))
        out: Dict[str, Any] = {
            "num_expressions": int(self.total),
            "valid_mask_expressions": int(self.valid_masks),
            "invalid_mask_expressions": int(self.total - self.valid_masks),
            "recall50@all_queries": float(self.all_query_correct50 / denom),
            "recall25@all_queries": float(self.all_query_correct25 / denom),
            "mean_best_iou@all_queries": float(self.all_query_iou_sum / denom),
        }
        if self.eligible_query_rows:
            eligible_denom = int(self.eligible_query_rows)
            out["recall50@eligible_queries"] = float(
                self.eligible_query_correct50 / eligible_denom
            )
            out["mean_best_iou@eligible_queries"] = float(
                self.eligible_query_iou_sum / eligible_denom
            )
        for k in self.topks:
            suffix = "" if k == 1 else f"@{k}"
            out[f"acc50{suffix}"] = float(self.correct50[k] / denom)
            out[f"acc25{suffix}"] = float(self.correct25[k] / denom)
            out[f"mean_iou{suffix}"] = float(self.iou_sum[k] / denom)
        return out


def _threshold_for_tpr(pos_scores: np.ndarray, target_tpr: float) -> float:
    pos_scores = pos_scores[np.isfinite(pos_scores)]
    if pos_scores.size == 0:
        return float("inf")
    target_tpr = min(1.0, max(0.0, float(target_tpr)))
    if target_tpr <= 0.0:
        return float(np.nextafter(pos_scores.max(), np.inf))
    accepted = max(1, int(math.ceil(target_tpr * int(pos_scores.size))))
    ascending_index = int(pos_scores.size) - accepted
    return float(np.partition(pos_scores, ascending_index)[ascending_index])


def _score_independent_query_subset(
    *, query_count: int, subset_count: int, seed: int, repeat: int, sample_id: str
) -> np.ndarray:
    """Choose a deterministic subset without inspecting any model score."""

    query_count = int(query_count)
    subset_count = int(subset_count)
    if query_count <= 0 or not 0 < subset_count <= query_count:
        raise ValueError("candidate-count control requires 0 < subset <= queries")
    payload = f"{int(seed)}\x1f{int(repeat)}\x1f{sample_id}".encode("utf-8")
    local_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    rng = np.random.default_rng(local_seed)
    return np.sort(
        rng.choice(query_count, size=subset_count, replace=False).astype(np.int64)
    )


def _roc_auc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """Tie-aware Mann-Whitney ROC AUC without an optional dependency."""

    pos = np.asarray(pos_scores, dtype=np.float64)
    neg = np.asarray(neg_scores, dtype=np.float64)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if not pos.size or not neg.size:
        return 0.0
    values = np.concatenate((pos, neg))
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    rank_sum = float(ranks[: pos.size].sum())
    statistic = rank_sum - pos.size * (pos.size + 1) / 2.0
    return statistic / float(pos.size * neg.size)


def _summarize_candidate_count_control(
    *,
    pos_by_repeat: Sequence[Sequence[float]],
    neg_by_repeat: Sequence[Sequence[float]],
    query_count: int,
    subset_count: int,
    seed: int,
    threshold_tprs: List[float],
    score_thresholds: List[float],
) -> Dict[str, Any]:
    if len(pos_by_repeat) != len(neg_by_repeat) or not pos_by_repeat:
        raise ValueError("candidate-count control repeats are empty or misaligned")
    repeats = []
    for repeat, (pos_values, neg_values) in enumerate(
        zip(pos_by_repeat, neg_by_repeat)
    ):
        pos = np.asarray(pos_values, dtype=np.float32)
        neg = np.asarray(neg_values, dtype=np.float32)
        if pos.shape != neg.shape:
            raise ValueError("candidate-count control positive/TN arrays differ")
        row = _summarize_tn_arrays(
            pos,
            neg,
            np.zeros_like(pos),
            np.zeros_like(neg),
            threshold_tprs,
            score_thresholds,
        )
        row.pop("pos_top1_iou50", None)
        row.pop("tn_top1_iou50", None)
        row["roc_auc"] = _roc_auc(pos, neg)
        row["repeat"] = repeat
        repeats.append(row)
    metric_names = ["fpr95tpr", "fpr90tpr", "pair_win_rate", "roc_auc"]
    aggregate = {}
    for name in metric_names:
        values = np.asarray([float(row.get(name, 0.0)) for row in repeats])
        aggregate[f"{name}_mean"] = float(values.mean())
        aggregate[f"{name}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return {
        "schema": "stageb-gdino-candidate-count-control-v1",
        "diagnostic_only": True,
        "formal_baseline_eligible": False,
        "purpose": "isolate global-max multiplicity from architecture/training",
        "selection": "score_independent_sha256_seeded_without_replacement",
        "query_count": int(query_count),
        "subset_count": int(subset_count),
        "seed": int(seed),
        "repeats": repeats,
        "aggregate": aggregate,
    }


def _summarize_tn_arrays(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    pos_iou: np.ndarray,
    neg_iou: np.ndarray,
    threshold_tprs: List[float],
    score_thresholds: List[float],
) -> Dict[str, Any]:
    valid = np.isfinite(pos_scores) & np.isfinite(neg_scores)
    pos_scores = pos_scores[valid]
    neg_scores = neg_scores[valid]
    pos_iou = pos_iou[valid]
    neg_iou = neg_iou[valid]
    gap = pos_scores - neg_scores
    out: Dict[str, Any] = {
        "num_pairs": int(pos_scores.size),
        "pair_win_rate": float(np.mean(pos_scores > neg_scores)) if pos_scores.size else 0.0,
        "pair_tie_rate": float(np.mean(pos_scores == neg_scores)) if pos_scores.size else 0.0,
        "score_gap_mean": float(gap.mean()) if gap.size else 0.0,
        "score_gap_median": float(np.median(gap)) if gap.size else 0.0,
        "pos_score_mean": float(pos_scores.mean()) if pos_scores.size else 0.0,
        "tn_score_mean": float(neg_scores.mean()) if neg_scores.size else 0.0,
        "pos_top1_iou50": float(np.mean(pos_iou >= 0.5)) if pos_iou.size else 0.0,
        "tn_top1_iou50": float(np.mean(neg_iou >= 0.5)) if neg_iou.size else 0.0,
    }
    for tpr in threshold_tprs:
        key = f"{int(round(float(tpr) * 100)):02d}"
        threshold = _threshold_for_tpr(pos_scores, float(tpr))
        out[f"threshold_at_{key}tpr"] = threshold
        out[f"actual_tpr_at_{key}tpr"] = float(np.mean(pos_scores >= threshold)) if pos_scores.size else 0.0
        out[f"fpr{key}tpr"] = float(np.mean(neg_scores >= threshold)) if neg_scores.size else 0.0
    for threshold in score_thresholds:
        key = f"{float(threshold):.2f}".replace(".", "p")
        out[f"tpr_at_score_{key}"] = float(np.mean(pos_scores >= float(threshold))) if pos_scores.size else 0.0
        out[f"fpr_at_score_{key}"] = float(np.mean(neg_scores >= float(threshold))) if neg_scores.size else 0.0
    out.setdefault("fpr95tpr", 0.0)
    out["tn_fpr"] = float(out.get("fpr95tpr", 0.0))
    return out


def _summarize_tn_by_meta(
    records: List[Dict[str, float]],
    metas: List[Dict[str, Any]],
    key: str,
    threshold_tprs: List[float],
    score_thresholds: List[float],
):
    groups: Dict[str, List[int]] = {}
    for i, meta in enumerate(metas):
        groups.setdefault(str(meta.get(key, "unknown")), []).append(i)
    out: Dict[str, Any] = {}
    for name, idxs in groups.items():
        out[name] = _summarize_tn_arrays(
            np.asarray([records[i]["pos_score"] for i in idxs], dtype=np.float32),
            np.asarray([records[i]["tn_score"] for i in idxs], dtype=np.float32),
            np.asarray([records[i]["pos_iou"] for i in idxs], dtype=np.float32),
            np.asarray([records[i]["tn_iou"] for i in idxs], dtype=np.float32),
            threshold_tprs,
            score_thresholds,
        )
    return out


def _positive_captions(targets: List[Dict[str, Any]]) -> Tuple[List[str], torch.Tensor]:
    captions: List[str] = []
    valid: List[bool] = []
    for target in targets:
        rank_caps = target.get("rank_positive_captions", None)
        has_rank = target.get("has_rank_positive", None)
        cap = None
        ok = False
        if isinstance(rank_caps, list) and rank_caps:
            maybe = rank_caps[0]
            if isinstance(maybe, str) and maybe.strip():
                cap = maybe
                ok = True
        if torch.is_tensor(has_rank):
            ok = ok and bool(has_rank.view(-1)[0].item()) if has_rank.numel() > 0 else False
        captions.append(cap if cap is not None else str(target.get("caption", "object .")))
        valid.append(bool(ok))
    return captions, torch.as_tensor(valid, dtype=torch.bool)


@torch.no_grad()
def evaluate_refcoco_dataset(
    *,
    cfg,
    model,
    ckpt_path: str,
    datasetinfo: Dict[str, Any],
    dataset_name: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
    topks: List[int],
    amp: bool,
    max_batches: int,
    log_every: int,
    records_output_dir: Optional[Path] = None,
    checkpoint_summary_fields: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    loader = _build_loader(cfg, datasetinfo, batch_size, num_workers, device, seed)
    manifest = load_eval_manifest(
        Path(datasetinfo["anno"]),
        task="ref",
        split=dataset_name,
    )
    run_id = _ckpt_run_prefix(ckpt_path)
    acc = RefCocoTextAccumulator(
        topks,
        manifest=manifest,
        run_id=run_id,
        adapter_score_key=(
            _adapter_ref_score_key(cfg)
        ),
    )
    causal_route_keys = {
        "b58_base": "stage_b_gdino_base_score",
        "raw_r100": "stage_b_gdino_rank_score",
        "patch_r100": "stage_b_u2v2_patch_r100_rank_score",
    } if bool(getattr(cfg, "stage_b_u2v2_emit_causal_ref_routes", False)) else {}
    causal_accumulators = {
        name: RefCocoTextAccumulator(
            topks,
            manifest=manifest,
            run_id=f"{run_id}__causal_{name}",
            adapter_score_key=score_key,
        )
        for name, score_key in causal_route_keys.items()
    }
    start = time.time()

    print(
        f"[INFO] text RefCOCO eval ckpt={Path(ckpt_path).name} dataset={dataset_name} "
        f"expressions={len(loader.dataset)} batches={len(loader)} batch_size={batch_size}",
        flush=True,
    )
    for batch_i, batch in enumerate(loader):
        if max_batches > 0 and batch_i >= int(max_batches):
            break
        targets = list(batch[1])
        _validate_eval_manifest_batch_alignment(
            targets, manifest, int(acc.total)
        )
        outputs, model_targets, query_scores = _forward_ref_batch(
            cfg, model, batch, device, amp=amp
        )
        eligibility_mask = None
        record_extras = None
        if bool(getattr(cfg, "stage_b_u2v2", False)):
            eligibility_mask = outputs.get(
                "stage_b_u2v2_eligible_mask",
                outputs.get("stage_b_u0_category_gate_eligible_mask"),
            )
            if not torch.is_tensor(eligibility_mask) or eligibility_mask.dtype != torch.bool:
                raise RuntimeError("U2-v2 evaluation lacks its boolean eligibility mask")
            record_extras = []
            for row_mask in eligibility_mask:
                mask_bytes = (
                    row_mask.detach().to(device="cpu", dtype=torch.uint8)
                    .contiguous().numpy().tobytes()
                )
                record_extras.append(
                    {
                        "stage_b_u2v2_eligible_queries": int(row_mask.sum().item()),
                        "stage_b_u2v2_eligible_mask_sha256": hashlib.sha256(
                            mask_bytes
                        ).hexdigest(),
                    }
                )
        acc.update(
            outputs,
            model_targets,
            query_scores=query_scores,
            record_extras=record_extras,
            eligibility_mask=eligibility_mask,
        )
        for route_name, route_accumulator in causal_accumulators.items():
            route_score = outputs.get(causal_route_keys[route_name])
            deployed_score = (
                query_scores if torch.is_tensor(query_scores)
                else outputs.get(_adapter_ref_score_key(cfg))
            )
            if (
                not torch.is_tensor(deployed_score)
                or not torch.is_tensor(route_score)
                or route_score.shape != deployed_score.shape
            ):
                raise RuntimeError(f"U2-v2 causal route {route_name} is unavailable")
            route_accumulator.update(
                outputs,
                model_targets,
                query_scores=route_score,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if log_every > 0 and (batch_i == 0 or (batch_i + 1) % int(log_every) == 0):
            elapsed = time.time() - start
            done = batch_i + 1
            total = min(len(loader), max_batches) if max_batches > 0 else len(loader)
            eta = elapsed / max(1, done) * max(0, total - done)
            print(
                f"[INFO] {dataset_name} {Path(ckpt_path).name}: batch {done}/{total}, "
                f"expressions={acc.total}, elapsed={elapsed/60:.1f}m, eta={eta/60:.1f}m",
                flush=True,
            )
    row = acc.result()
    if causal_accumulators:
        row["stage_b_u2v2_causal_ref_routes"] = {
            "b58_base": causal_accumulators["b58_base"].result(),
            "raw_r100": causal_accumulators["raw_r100"].result(),
            "patch_r100": causal_accumulators["patch_r100"].result(),
            "patch_residual": dict(row),
        }
    row.update(
        {
            "run_id": _ckpt_run_prefix(ckpt_path),
            "checkpoint": str(ckpt_path),
            "checkpoint_name": Path(ckpt_path).name,
            "dataset": dataset_name,
            "seconds": float(time.time() - start),
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "seed": int(seed),
            "max_batches": int(max_batches),
        }
    )
    if bool(getattr(cfg, "stage_b_native_patch_category", False)):
        row.update(
            {
                "ref_score_contract": _NATIVE_PATCH_REF_SCORE_CONTRACT,
                "ref_score_key": _NATIVE_PATCH_RANK_SCORE_KEY,
                "native_patch_category_gate_max_gap": (
                    _NATIVE_PATCH_GATE_MAX_GAP
                ),
                "native_patch_category_gate_clip": _NATIVE_PATCH_GATE_CLIP,
                "native_patch_category_full_expression": True,
                "native_patch_category_single_checkpoint": True,
            }
        )
    if records_output_dir is not None:
        records_path = Path(records_output_dir) / (
            f"{run_id}__{_safe_name(dataset_name)}.records.jsonl"
        )
        write_eval_records(records_path, acc.eval_records)
        row.update(
            {
                "records_jsonl": str(records_path),
                "manifest_sha256": manifest.sha256,
                "manifest_n": manifest.size,
                "invalid_records": int(
                    sum(not bool(record.get("valid")) for record in acc.eval_records)
                ),
            }
        )
    _bind_checkpoint_summary_fields(row, checkpoint_summary_fields)
    return row


@torch.no_grad()
def evaluate_refcoco_category_gate_sweep(
    *,
    cfg,
    model,
    ckpt_path: str,
    datasetinfo: Dict[str, Any],
    dataset_name: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
    topks: List[int],
    max_gaps: Sequence[float],
    amp: bool,
    max_batches: int,
    log_every: int,
    records_output_dir: Path,
    include_base_expert: bool = False,
    checkpoint_summary_fields: Optional[Mapping[str, str]] = None,
) -> List[Dict[str, Any]]:
    gaps = _normalize_category_gate_sweep_gaps(max_gaps)
    if gaps is None:
        raise ValueError("category gate sweep max_gap values are required")
    if not bool(getattr(cfg, "stage_b_u0_patch_rank", False)) or not bool(
        getattr(cfg, "stage_b_u0_category_preserving_patch_gate", False)
    ):
        raise ValueError("category gate sweep requires a gate-enabled U0 config")
    configured_gap = float(getattr(cfg, "stage_b_u0_category_gate_max_gap"))
    if not math.isfinite(configured_gap) or configured_gap < 0.0:
        raise ValueError("configured category gate max_gap must be finite and non-negative")

    loader = _build_loader(cfg, datasetinfo, batch_size, num_workers, device, seed)
    manifest = load_eval_manifest(
        Path(datasetinfo["anno"]),
        task="ref",
        split=dataset_name,
    )
    base_run_id = _ckpt_run_prefix(ckpt_path)
    accumulators = {
        gap: RefCocoTextAccumulator(
            topks,
            manifest=manifest,
            run_id=(
                f"{base_run_id}__category_gate_gap_"
                f"{_category_gate_gap_slug(gap)}"
            ),
        )
        for gap in gaps
    }
    base_accumulators = (
        {
            gap: RefCocoTextAccumulator(
                topks,
                manifest=manifest,
                run_id=(
                    f"{base_run_id}__category_gate_base_b58_gap_"
                    f"{_category_gate_gap_slug(gap)}"
                ),
            )
            for gap in gaps
        }
        if bool(include_base_expert)
        else {}
    )
    start = time.time()
    print(
        f"[INFO] category gate Ref sweep ckpt={Path(ckpt_path).name} "
        f"dataset={dataset_name} gaps={list(gaps)} expressions={len(loader.dataset)} "
        f"batches={len(loader)} batch_size={batch_size} "
        f"experts={'teacher_r100+base_b58' if include_base_expert else 'teacher_r100'}",
        flush=True,
    )
    for batch_i, batch in enumerate(loader):
        if max_batches > 0 and batch_i >= int(max_batches):
            break
        first_accumulator = accumulators[gaps[0]]
        targets = list(batch[1])
        _validate_eval_manifest_batch_alignment(
            targets, manifest, int(first_accumulator.total)
        )
        outputs, model_targets, query_scores = _forward_ref_batch(
            cfg, model, batch, device, amp=amp
        )
        if query_scores is not None:
            raise RuntimeError("category gate sweep requires model-native U0 outputs")

        evaluation_gaps = gaps
        if configured_gap not in evaluation_gaps:
            evaluation_gaps = gaps + (configured_gap,)
        score_rows = _category_gate_sweep_query_scores(
            outputs, evaluation_gaps
        )
        by_gap = {
            gap: (scores, eligible)
            for gap, scores, eligible in score_rows
        }
        base_by_gap: Dict[float, Tuple[torch.Tensor, torch.Tensor]] = {}
        if include_base_expert:
            base_score, base_valid = _phrase_scores(
                outputs,
                model_targets,
                "phrase_to_token_mask",
                adapter_score_key="stage_b_gdino_base_score",
            )
            if not bool(base_valid.all().item()):
                raise RuntimeError("base-b58 category gate contains an invalid row")
            base_outputs = dict(outputs)
            base_outputs["stage_b_gdino_base_score"] = base_score
            base_score_rows = _category_gate_sweep_query_scores(
                base_outputs,
                evaluation_gaps,
                rank_score_key="stage_b_gdino_base_score",
            )
            base_by_gap = {
                gap: (scores, eligible)
                for gap, scores, eligible in base_score_rows
            }
        observed_rank = outputs.get("stage_b_u0_rank_score")
        if not torch.is_tensor(observed_rank) or not torch.equal(
            observed_rank, by_gap[configured_gap][0]
        ):
            raise RuntimeError(
                "configured model category gate score drifted from sweep reconstruction"
            )

        teacher = outputs["stage_b_u0_teacher_rank_score"]
        patch_score = outputs["stage_b_u0_category_gate_patch_score"]
        candidate_mask = outputs["stage_b_u0_candidate_mask"]
        teacher_winner = teacher.masked_fill(
            ~candidate_mask, -torch.inf
        ).argmax(dim=1)
        patch_winner = patch_score.masked_fill(
            ~candidate_mask, -torch.inf
        ).argmax(dim=1)
        base_winner = None
        if include_base_expert:
            base_winner = base_score.masked_fill(
                ~candidate_mask, -torch.inf
            ).argmax(dim=1)
        for gap in gaps:
            scores, eligible = by_gap[gap]
            gate_winner = scores.argmax(dim=1)
            extras = []
            for index in range(int(scores.shape[0])):
                extra = {
                    "category_gate_sweep_contract": CATEGORY_GATE_SWEEP_CONTRACT,
                    "category_gate_max_gap": float(gap),
                    "category_gate_eligible_queries": int(
                        eligible[index].sum().item()
                    ),
                    "category_gate_winner_query": int(gate_winner[index].item()),
                    "category_gate_teacher_winner_query": int(
                        teacher_winner[index].item()
                    ),
                    "category_gate_patch_winner_query": int(
                        patch_winner[index].item()
                    ),
                }
                if include_base_expert:
                    extra.update(
                        {
                            "category_gate_rank_expert": "teacher_r100",
                            "category_gate_base_winner_query": int(
                                base_winner[index].item()
                            ),
                        }
                    )
                extras.append(extra)
            accumulators[gap].update(
                outputs,
                model_targets,
                query_scores=scores,
                record_extras=extras,
                eligibility_mask=eligible,
            )
            if include_base_expert:
                base_scores, base_eligible = base_by_gap[gap]
                if not torch.equal(base_eligible, eligible):
                    raise RuntimeError(
                        "teacher and base category gates reconstructed different eligibility"
                    )
                base_gate_winner = base_scores.argmax(dim=1)
                base_extras = [
                    {
                        "category_gate_sweep_contract": (
                            CATEGORY_GATE_BASE_EXPERT_SWEEP_CONTRACT
                        ),
                        "category_gate_rank_expert": "base_b58",
                        "category_gate_rank_score_key": "stage_b_gdino_base_score",
                        "category_gate_max_gap": float(gap),
                        "category_gate_eligible_queries": int(
                            base_eligible[index].sum().item()
                        ),
                        "category_gate_winner_query": int(
                            base_gate_winner[index].item()
                        ),
                        "category_gate_rank_expert_winner_query": int(
                            base_winner[index].item()
                        ),
                        "category_gate_base_winner_query": int(
                            base_winner[index].item()
                        ),
                        "category_gate_teacher_winner_query": int(
                            teacher_winner[index].item()
                        ),
                        "category_gate_patch_winner_query": int(
                            patch_winner[index].item()
                        ),
                    }
                    for index in range(int(base_scores.shape[0]))
                ]
                base_accumulators[gap].update(
                    outputs,
                    model_targets,
                    query_scores=base_scores,
                    record_extras=base_extras,
                    eligibility_mask=base_eligible,
                )
        totals = {
            accumulator.total
            for accumulator in (
                list(accumulators.values()) + list(base_accumulators.values())
            )
        }
        if len(totals) != 1:
            raise RuntimeError("category gate sweep accumulators lost row alignment")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if log_every > 0 and (
            batch_i == 0 or (batch_i + 1) % int(log_every) == 0
        ):
            elapsed = time.time() - start
            done = batch_i + 1
            total = min(len(loader), max_batches) if max_batches > 0 else len(loader)
            eta = elapsed / max(1, done) * max(0, total - done)
            print(
                f"[INFO] category gate {dataset_name} {Path(ckpt_path).name}: "
                f"batch {done}/{total}, expressions={first_accumulator.total}, "
                f"elapsed={elapsed/60:.1f}m, eta={eta/60:.1f}m",
                flush=True,
            )

    elapsed = float(time.time() - start)
    rows: List[Dict[str, Any]] = []
    for gap in gaps:
        accumulator = accumulators[gap]
        row = accumulator.result()
        row.update(
            {
                "run_id": accumulator.run_id,
                "checkpoint": str(ckpt_path),
                "checkpoint_name": Path(ckpt_path).name,
                "dataset": dataset_name,
                "seconds": elapsed,
                "batch_size": int(batch_size),
                "num_workers": int(num_workers),
                "seed": int(seed),
                "max_batches": int(max_batches),
                "category_gate_sweep_contract": CATEGORY_GATE_SWEEP_CONTRACT,
                "category_gate_max_gap": float(gap),
                "category_gate_configured_max_gap": configured_gap,
                "category_gate_single_forward_gap_count": len(gaps),
            }
        )
        records_path = Path(records_output_dir) / (
            f"{accumulator.run_id}__{_safe_name(dataset_name)}.records.jsonl"
        )
        write_eval_records(records_path, accumulator.eval_records)
        row.update(
            {
                "records_jsonl": str(records_path),
                "manifest_sha256": manifest.sha256,
                "manifest_n": manifest.size,
                "invalid_records": int(
                    sum(
                        not bool(record.get("valid"))
                        for record in accumulator.eval_records
                    )
                ),
            }
        )
        _bind_checkpoint_summary_fields(row, checkpoint_summary_fields)
        rows.append(row)
    for gap in gaps:
        if not include_base_expert:
            break
        accumulator = base_accumulators[gap]
        row = accumulator.result()
        row.update(
            {
                "run_id": accumulator.run_id,
                "checkpoint": str(ckpt_path),
                "checkpoint_name": Path(ckpt_path).name,
                "dataset": dataset_name,
                "seconds": elapsed,
                "batch_size": int(batch_size),
                "num_workers": int(num_workers),
                "seed": int(seed),
                "max_batches": int(max_batches),
                "category_gate_sweep_contract": (
                    CATEGORY_GATE_BASE_EXPERT_SWEEP_CONTRACT
                ),
                "category_gate_rank_expert": "base_b58",
                "category_gate_rank_score_key": "stage_b_gdino_base_score",
                "category_gate_max_gap": float(gap),
                "category_gate_configured_max_gap": configured_gap,
                "category_gate_single_forward_gap_count": len(gaps),
                "category_gate_single_forward_expert_count": 2,
            }
        )
        records_path = Path(records_output_dir) / (
            f"{accumulator.run_id}__{_safe_name(dataset_name)}.records.jsonl"
        )
        write_eval_records(records_path, accumulator.eval_records)
        row.update(
            {
                "records_jsonl": str(records_path),
                "manifest_sha256": manifest.sha256,
                "manifest_n": manifest.size,
                "invalid_records": int(
                    sum(
                        not bool(record.get("valid"))
                        for record in accumulator.eval_records
                    )
                ),
            }
        )
        _bind_checkpoint_summary_fields(row, checkpoint_summary_fields)
        rows.append(row)
    if include_base_expert:
        for row in rows[: len(gaps)]:
            row["category_gate_rank_expert"] = "teacher_r100"
            row["category_gate_rank_score_key"] = (
                "stage_b_u0_teacher_rank_score"
            )
            row["category_gate_single_forward_expert_count"] = 2
    return rows


@torch.no_grad()
def evaluate_tn_dataset(
    *,
    cfg,
    model,
    ckpt_path: str,
    datasetinfo: Dict[str, Any],
    meta_rows: List[Dict[str, Any]],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    seed: int,
    threshold_tprs: List[float],
    score_thresholds: List[float],
    amp: bool,
    max_batches: int,
    log_every: int,
    candidate_count_control: int = 0,
    candidate_count_repeats: int = 0,
    records_output_dir: Optional[Path] = None,
    checkpoint_summary_fields: Optional[Mapping[str, str]] = None,
    declared_eval_scope: Optional[str] = None,
) -> Dict[str, Any]:
    loader = _build_loader(cfg, datasetinfo, batch_size, num_workers, device, seed)
    manifest = load_eval_manifest(
        Path(datasetinfo["anno"]),
        task="tn",
        split="global",
        manifest_key="tn_global",
    )
    eval_scope = (
        _validate_direct_prebuilt_tn_rows(
            manifest.rows, declared_scope=str(declared_eval_scope)
        )
        if declared_eval_scope is not None
        else _validate_adapter_tn_eval_scope(cfg, manifest.rows)
    )
    train_scope = (
        str(getattr(cfg, "stage_b_gdino_tn_scope", "")).strip() or None
        if bool(getattr(cfg, "stage_b_gdino_score_adapter", False))
        else None
    )
    run_id = _ckpt_run_prefix(ckpt_path)
    records: List[Dict[str, float]] = []
    eval_records: List[Dict[str, Any]] = []
    valid_metas: List[Dict[str, Any]] = []
    invalid_positive = 0
    invalid_negative = 0
    control_pos: List[List[float]] = [
        [] for _ in range(max(0, int(candidate_count_repeats)))
    ]
    control_neg: List[List[float]] = [
        [] for _ in range(max(0, int(candidate_count_repeats)))
    ]
    control_query_count: Optional[int] = None
    if (int(candidate_count_control) > 0) != (int(candidate_count_repeats) > 0):
        raise ValueError(
            "candidate-count control count and repeats must both be positive or zero"
        )
    start = time.time()

    def diagnostic_value(
        outputs: Mapping[str, Any], key: str, index: int
    ) -> Optional[float]:
        value = outputs.get(key)
        if not torch.is_tensor(value) or value.dim() < 1:
            return None
        row = value[index].detach().float().reshape(-1)
        if row.numel() == 0 or not bool(torch.isfinite(row[0]).item()):
            return None
        return float(row[0].item())

    def diagnostic_argmax_value(
        outputs: Mapping[str, Any], score_key: str, value_key: str, index: int
    ) -> Optional[float]:
        score = outputs.get(score_key)
        value = outputs.get(value_key)
        if not torch.is_tensor(score) or not torch.is_tensor(value):
            return None
        score_row = score[index].detach().float().reshape(-1)
        value_row = value[index].detach().float().reshape(-1)
        if score_row.numel() == 0 or score_row.numel() != value_row.numel():
            return None
        argmax = int(score_row.argmax().item())
        selected = value_row[argmax]
        if not bool(torch.isfinite(selected).item()):
            return None
        return float(selected.item())

    def diagnostic_max_value(
        outputs: Mapping[str, Any], key: str, index: int
    ) -> Optional[float]:
        value = outputs.get(key)
        if not torch.is_tensor(value) or value.dim() < 1:
            return None
        row = value[index].detach().float().reshape(-1)
        finite = row[torch.isfinite(row)]
        if finite.numel() == 0:
            return None
        return float(finite.max().item())

    print(
        f"[INFO] text TN eval ckpt={Path(ckpt_path).name} pairs={len(loader.dataset)} "
        f"batches={len(loader)} batch_size={batch_size}",
        flush=True,
    )
    offset = 0
    for batch_i, batch in enumerate(loader):
        if max_batches > 0 and batch_i >= int(max_batches):
            break
        raw_targets = list(batch[1])
        raw_bsz = len(raw_targets)
        manifest_start_index = offset
        _validate_eval_manifest_batch_alignment(
            raw_targets, manifest, manifest_start_index
        )
        metas = meta_rows[offset : offset + raw_bsz]
        offset += raw_bsz
        support_identities = (
            [
                _support_input_identity(
                    target,
                    expected_class_id=manifest.rows[
                        manifest_start_index + index
                    ].get("class_id"),
                )
                for index, target in enumerate(raw_targets)
            ]
            if declared_eval_scope is not None
            else [{} for _target in raw_targets]
        )
        (
            neg_outputs,
            pos_outputs,
            targets,
            valid_pos,
            neg_scores_q,
            valid_neg,
            pos_scores_q,
            valid_pos_mask,
        ) = _forward_tn_batch(
            cfg, model, batch, device, amp=amp
        )
        valid = valid_pos.to(valid_neg.device) & valid_neg & valid_pos_mask
        invalid_positive += int((~(valid_pos.to(valid_neg.device) & valid_pos_mask)).sum().item())
        invalid_negative += int((~valid_neg).sum().item())
        neg_best = neg_scores_q.max(dim=1).values.detach().cpu().numpy()
        pos_best = pos_scores_q.max(dim=1).values.detach().cpu().numpy()
        neg_iou = _top_iou(neg_outputs, targets, neg_scores_q)
        pos_iou = _top_iou(pos_outputs, targets, pos_scores_q)
        valid_np = valid.detach().cpu().numpy().astype(bool)
        pos_scores_cpu = pos_scores_q.detach().float().cpu().numpy()
        neg_scores_cpu = neg_scores_q.detach().float().cpu().numpy()
        if int(candidate_count_control) > 0:
            query_count = int(pos_scores_cpu.shape[1])
            if control_query_count is None:
                control_query_count = query_count
            elif control_query_count != query_count:
                raise RuntimeError("candidate-count control query count changed by batch")
            if int(candidate_count_control) > query_count:
                raise ValueError(
                    "candidate-count control subset exceeds model query count"
                )
        for i, ok in enumerate(valid_np):
            finite = bool(
                ok
                and np.isfinite(pos_best[i])
                and np.isfinite(neg_best[i])
                and np.isfinite(pos_iou[i])
                and np.isfinite(neg_iou[i])
            )
            eval_records.append(
                make_eval_record(
                    manifest,
                    index=manifest_start_index + i,
                    run_id=run_id,
                    valid=finite,
                    meta=metas[i],
                    values={
                        "train_scope": train_scope,
                        "eval_scope": eval_scope,
                        "pos_score": float(pos_best[i]),
                        "neg_score": float(neg_best[i]),
                        "pos_iou": float(pos_iou[i]),
                        "neg_iou": float(neg_iou[i]),
                        "pos_reference_global_logit": diagnostic_value(
                            pos_outputs,
                            "stage_b_dense_duty_reference_global_confidence_logits",
                            i,
                        ),
                        "neg_reference_global_logit": diagnostic_value(
                            neg_outputs,
                            "stage_b_dense_duty_reference_global_confidence_logits",
                            i,
                        ),
                        "pos_frozen_rank_full_expression_global_logit": diagnostic_value(
                            pos_outputs,
                            "stage_b_dense_duty_frozen_rank_full_expression_global_logits",
                            i,
                        ),
                        "neg_frozen_rank_full_expression_global_logit": diagnostic_value(
                            neg_outputs,
                            "stage_b_dense_duty_frozen_rank_full_expression_global_logits",
                            i,
                        ),
                        "pos_global_logit": diagnostic_value(
                            pos_outputs,
                            "stage_b_dense_duty_global_confidence_logits",
                            i,
                        ),
                        "neg_global_logit": diagnostic_value(
                            neg_outputs,
                            "stage_b_dense_duty_global_confidence_logits",
                            i,
                        ),
                        "pos_pool_absolute_logit": diagnostic_value(
                            pos_outputs,
                            "stage_b_dense_duty_confidence_pool_absolute_logits",
                            i,
                        ),
                        "neg_pool_absolute_logit": diagnostic_value(
                            neg_outputs,
                            "stage_b_dense_duty_confidence_pool_absolute_logits",
                            i,
                        ),
                        "pos_deployed_query_veto_depth": diagnostic_value(
                            pos_outputs,
                            "stage_b_dense_duty_deployed_query_veto_depth",
                            i,
                        ),
                        "neg_deployed_query_veto_depth": diagnostic_value(
                            neg_outputs,
                            "stage_b_dense_duty_deployed_query_veto_depth",
                            i,
                        ),
                        "pos_deployed_query_veto_gate": diagnostic_value(
                            pos_outputs,
                            "stage_b_dense_duty_deployed_query_veto_gate",
                            i,
                        ),
                        "neg_deployed_query_veto_gate": diagnostic_value(
                            neg_outputs,
                            "stage_b_dense_duty_deployed_query_veto_gate",
                            i,
                        ),
                        "pos_veto_sample_gate": diagnostic_value(
                            pos_outputs,
                            "stage_b_dense_duty_confidence_veto_sample_gate",
                            i,
                        ),
                        "neg_veto_sample_gate": diagnostic_value(
                            neg_outputs,
                            "stage_b_dense_duty_confidence_veto_sample_gate",
                            i,
                        ),
                        "pos_veto_coverage": diagnostic_value(
                            pos_outputs,
                            "stage_b_dense_duty_confidence_veto_coverage",
                            i,
                        ),
                        "neg_veto_coverage": diagnostic_value(
                            neg_outputs,
                            "stage_b_dense_duty_confidence_veto_coverage",
                            i,
                        ),
                        "pos_candidate_max_base_logit": diagnostic_argmax_value(
                            pos_outputs,
                            "stage_b_dense_duty_confidence_base_logits",
                            "stage_b_dense_duty_confidence_base_logits",
                            i,
                        ),
                        "neg_candidate_max_base_logit": diagnostic_argmax_value(
                            neg_outputs,
                            "stage_b_dense_duty_confidence_base_logits",
                            "stage_b_dense_duty_confidence_base_logits",
                            i,
                        ),
                        "pos_candidate_max_mismatch_gate": diagnostic_argmax_value(
                            pos_outputs,
                            "stage_b_dense_duty_confidence_base_logits",
                            "stage_b_dense_duty_confidence_mismatch_gate",
                            i,
                        ),
                        "neg_candidate_max_mismatch_gate": diagnostic_argmax_value(
                            neg_outputs,
                            "stage_b_dense_duty_confidence_base_logits",
                            "stage_b_dense_duty_confidence_mismatch_gate",
                            i,
                        ),
                        "pos_patch_candidate_max_logit": diagnostic_max_value(
                            pos_outputs,
                            "stage_b_v15_candidate_patch_logits",
                            i,
                        ),
                        "neg_patch_candidate_max_logit": diagnostic_max_value(
                            neg_outputs,
                            "stage_b_v15_candidate_patch_logits",
                            i,
                        ),
                        **support_identities[i],
                    },
                )
            )
            if not ok:
                continue
            if int(candidate_count_control) > 0:
                sample_id = str(
                    metas[i].get("sample_id")
                    or f"manifest-index:{manifest_start_index + i}"
                )
                for repeat in range(int(candidate_count_repeats)):
                    subset = _score_independent_query_subset(
                        query_count=int(pos_scores_cpu.shape[1]),
                        subset_count=int(candidate_count_control),
                        seed=int(seed),
                        repeat=repeat,
                        sample_id=sample_id,
                    )
                    control_pos[repeat].append(
                        float(pos_scores_cpu[i, subset].max())
                    )
                    control_neg[repeat].append(
                        float(neg_scores_cpu[i, subset].max())
                    )
            records.append(
                {
                    "pos_score": float(pos_best[i]),
                    "tn_score": float(neg_best[i]),
                    "pos_iou": float(pos_iou[i]),
                    "tn_iou": float(neg_iou[i]),
                }
            )
            valid_metas.append(metas[i])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if log_every > 0 and (batch_i == 0 or (batch_i + 1) % int(log_every) == 0):
            elapsed = time.time() - start
            done = batch_i + 1
            total = min(len(loader), max_batches) if max_batches > 0 else len(loader)
            eta = elapsed / max(1, done) * max(0, total - done)
            print(
                f"[INFO] TN {Path(ckpt_path).name}: batch {done}/{total}, valid_pairs={len(records)}, "
                f"elapsed={elapsed/60:.1f}m, eta={eta/60:.1f}m",
                flush=True,
            )
    row = _summarize_tn_arrays(
        np.asarray([r["pos_score"] for r in records], dtype=np.float32),
        np.asarray([r["tn_score"] for r in records], dtype=np.float32),
        np.asarray([r["pos_iou"] for r in records], dtype=np.float32),
        np.asarray([r["tn_iou"] for r in records], dtype=np.float32),
        threshold_tprs,
        score_thresholds,
    )
    row.update(
        {
            "run_id": _ckpt_run_prefix(ckpt_path),
            "checkpoint": str(ckpt_path),
            "checkpoint_name": Path(ckpt_path).name,
            "seconds": float(time.time() - start),
            "batch_size": int(batch_size),
            "num_workers": int(num_workers),
            "seed": int(seed),
            "max_batches": int(max_batches),
            "train_scope": train_scope,
            "eval_scope": eval_scope,
            "invalid_positive_pairs": int(invalid_positive),
            "invalid_negative_pairs": int(invalid_negative),
            "by_split": _summarize_tn_by_meta(records, valid_metas, "eval_split", threshold_tprs, score_thresholds),
            "by_category": _summarize_tn_by_meta(records, valid_metas, "category", threshold_tprs, score_thresholds),
        }
    )
    if int(candidate_count_control) > 0:
        if control_query_count is None:
            raise RuntimeError("candidate-count control observed no valid query batch")
        row["candidate_count_matched_control"] = _summarize_candidate_count_control(
            pos_by_repeat=control_pos,
            neg_by_repeat=control_neg,
            query_count=control_query_count,
            subset_count=int(candidate_count_control),
            seed=int(seed),
            threshold_tprs=threshold_tprs,
            score_thresholds=score_thresholds,
        )
    if records_output_dir is not None:
        records_path = Path(records_output_dir) / f"{run_id}__tn_global.records.jsonl"
        write_eval_records(records_path, eval_records)
        row.update(
            {
                "records_jsonl": str(records_path),
                "manifest_sha256": manifest.sha256,
                "manifest_n": manifest.size,
                **tn_manifest_binding_summary_fields(manifest),
                "invalid_records": int(
                    sum(not bool(record.get("valid")) for record in eval_records)
                ),
            }
        )
    _bind_checkpoint_summary_fields(row, checkpoint_summary_fields)
    return row


def _mean_metric(rows: List[Dict[str, Any]], run_id: str, metric: str) -> float:
    vals = [float(row.get(metric, 0.0)) for row in rows if row["run_id"] == run_id]
    return sum(vals) / max(1, len(vals))


def _write_summary(output_dir: Path, ref_rows: List[Dict[str, Any]], tn_rows: List[Dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"refcoco": ref_rows, "tn": tn_rows}
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    run_ids: List[str] = []
    seen = set()
    for row in ref_rows:
        if row["run_id"] not in seen:
            seen.add(row["run_id"])
            run_ids.append(row["run_id"])
    for row in tn_rows:
        if row["run_id"] not in seen:
            seen.add(row["run_id"])
            run_ids.append(row["run_id"])
    datasets: List[str] = []
    seen_ds = set()
    for row in ref_rows:
        if row["dataset"] not in seen_ds:
            seen_ds.add(row["dataset"])
            datasets.append(row["dataset"])
    by_run_ds = {(row["run_id"], row["dataset"]): row for row in ref_rows}
    tn_by_run = {row["run_id"]: row for row in tn_rows}
    ranked = sorted(run_ids, key=lambda rid: _mean_metric(ref_rows, rid, "acc50"), reverse=True)
    dataset_header = "".join(f" | {ds} acc50" for ds in datasets)
    dataset_align = "".join("|---:" for _ in datasets)
    lines = [
        "# Text GroundingDINO RefCOCO/TN Evaluation",
        "",
        "| rank | run | mean RefCOCO acc50 | TN fpr@95tpr | TN fpr@90tpr | TN fpr@score0.50 | TN tpr@score0.50 | TN pair win | TN gap"
        + dataset_header
        + " |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:"
        + dataset_align
        + "|",
    ]
    for i, run_id in enumerate(ranked, start=1):
        tn = tn_by_run.get(run_id, {})
        ds_vals = [f"{float(by_run_ds.get((run_id, ds), {}).get('acc50', 0.0)):.6f}" for ds in datasets]
        lines.append(
            f"| {i} | `{run_id}` | {float(_mean_metric(ref_rows, run_id, 'acc50')):.6f} | "
            f"{float(tn.get('fpr95tpr', 0.0)):.6f} | {float(tn.get('fpr90tpr', 0.0)):.6f} | "
            f"{float(tn.get('fpr_at_score_0p50', 0.0)):.6f} | {float(tn.get('tpr_at_score_0p50', 0.0)):.6f} | "
            f"{float(tn.get('pair_win_rate', 0.0)):.6f} | {float(tn.get('score_gap_mean', 0.0)):.6f}"
            + (" | " + " | ".join(ds_vals) if ds_vals else "")
            + " |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ordinary text GroundingDINO on RefCOCO val and TN pairs.")
    parser.add_argument("--config", default="outputs/ogc_original_finetune_stage_a/cfg_ogc_original_finetune_stage_a.generated.py")
    parser.add_argument("--ckpts", nargs="+", required=True)
    parser.add_argument("--output_dir", default="outputs/text_groundingdino_refcoco_tn_eval")
    parser.add_argument("--data_root", default=os.environ.get("DATA_ROOT", "/media/haoyi/T9/data"))
    parser.add_argument("--tn_jsonl", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--ref_splits", nargs="+", default=["refcoco_val", "refcocop_val", "refcocog_val"])
    parser.add_argument("--tn_splits", nargs="+", default=["refcocop_val", "refcocog_val"])
    parser.add_argument(
        "--screen_calibration_manifest",
        action="store_true",
        help=(
            "Consume every row of the sealed scope-preserving single-edit "
            "calibration manifest instead of filtering by official Ref split."
        ),
    )
    parser.add_argument(
        "--screen_calibration_audit",
        default=str(SCREEN_CALIBRATION_AUDIT),
        help="Audit bound to --screen_calibration_manifest.",
    )
    parser.add_argument(
        "--direct_prebuilt_tn",
        action="store_true",
        help=(
            "Consume every row of a prebuilt, independently bound matched TN "
            "surface without Ref split filtering."
        ),
    )
    parser.add_argument(
        "--direct_prebuilt_tn_binding",
        default=None,
        help="Binding sidecar required by --direct_prebuilt_tn.",
    )
    parser.add_argument("--skip_ref", action="store_true", help="Only run TN pair evaluation and skip RefCOCO splits.")
    parser.add_argument("--skip_tn", action="store_true", help="Only run RefCOCO splits and skip TN pair evaluation.")
    parser.add_argument(
        "--partial_dense_duty_rank_diagnostic",
        action="store_true",
        help=(
            "Evaluate a non-terminal dense-duty rank checkpoint on the full "
            "Ref8 contract only; never admits TN/FPR or formal-gate claims."
        ),
    )
    parser.add_argument(
        "--partial_dense_duty_confidence_diagnostic",
        action="store_true",
        help=(
            "Evaluate a partial confidence-adapter checkpoint or a fixed "
            "terminal word-veto probe; always diagnostic-only and never "
            "formal-gate eligible."
        ),
    )
    parser.add_argument(
        "--immutable_v39_archived_snapshot_diagnostic",
        action="store_true",
        help=(
            "Screen only the fixed archived v39 U100/U200/U300 checkpoints. "
            "Requires --partial_dense_duty_confidence_diagnostic and cannot "
            "produce promotion or admission evidence."
        ),
    )
    parser.add_argument(
        "--immutable_v40_archived_snapshot_diagnostic",
        action="store_true",
        help=(
            "Screen only the fixed archived v40 U100/U200/U300 checkpoints. "
            "Requires --partial_dense_duty_confidence_diagnostic and cannot "
            "produce promotion or admission evidence."
        ),
    )
    parser.add_argument(
        "--immutable_v41_archived_snapshot_diagnostic",
        action="store_true",
        help=(
            "Screen only the fixed archived v41 U100/U200/U300 checkpoints. "
            "Requires --partial_dense_duty_confidence_diagnostic and cannot "
            "produce promotion or admission evidence."
        ),
    )
    parser.add_argument(
        "--immutable_v42_archived_snapshot_diagnostic",
        action="store_true",
        help=(
            "Evaluate only the fixed archived v42 U100/U200/U300/U400 "
            "checkpoints. Requires --partial_dense_duty_confidence_diagnostic "
            "and cannot produce promotion or admission evidence."
        ),
    )
    parser.add_argument("--topk", nargs="+", type=int, default=[1])
    parser.add_argument(
        "--category_gate_max_gaps",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Ref-val-only single-forward sweep of category-preserving gate "
            "max_gap values; requires a gate-enabled U0 config and --skip_tn."
        ),
    )
    parser.add_argument(
        "--category_gate_include_base_expert",
        action="store_true",
        help=(
            "During a category-gate sweep, also rank the identical patch-eligible "
            "query set with the authoritative pure-GDINO base score (b58). The "
            "teacher-R100 and base-b58 records are emitted from the same forward."
        ),
    )
    parser.add_argument("--threshold_tprs", nargs="+", type=float, default=[0.75, 0.9, 0.95])
    parser.add_argument("--score_thresholds", nargs="+", type=float, default=[0.5])
    parser.add_argument("--max_ref_batches", type=int, default=0)
    parser.add_argument("--max_tn_batches", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument(
        "--candidate_count_control",
        type=int,
        default=0,
        help=(
            "Score-independent query subset size for the Table-A global-max "
            "multiplicity diagnostic; 0 disables it."
        ),
    )
    parser.add_argument(
        "--candidate_count_repeats",
        type=int,
        default=32,
        help="Number of deterministic subset repeats when the control is enabled.",
    )
    parser.add_argument("--exclude_train_jsonl", nargs="*", default=[])
    parser.add_argument("--holdout_level", choices=["none", "ann", "image"], default="none")
    parser.add_argument(
        "--no_per_example_records",
        action="store_true",
        help="Disable canonical *.records.jsonl output used by the paired final gate.",
    )
    args = parser.parse_args()

    _validate_direct_prebuilt_tn_args(args)

    if args.screen_calibration_manifest and (args.skip_tn or not args.tn_jsonl):
        raise ValueError(
            "--screen_calibration_manifest requires TN evaluation and an explicit --tn_jsonl"
        )
    if args.screen_calibration_manifest and args.holdout_level != "none":
        raise ValueError(
            "the sealed screen calibration manifest cannot be filtered by holdout options"
        )

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    device = torch.device(args.device)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    tn_jsonl = Path(args.tn_jsonl) if args.tn_jsonl else data_root / "patch_episode_prebuilt" / "refexp_tn_stageb_v1.jsonl"

    cfg = SLConfig.fromfile(args.config)
    _validate_partial_dense_duty_rank_diagnostic_args(args, cfg)
    _validate_partial_dense_duty_confidence_diagnostic_args(args, cfg)
    _validate_immutable_v39_archived_snapshot_diagnostic_args(args, cfg)
    _validate_immutable_v40_archived_snapshot_diagnostic_args(args, cfg)
    _validate_immutable_v41_archived_snapshot_diagnostic_args(args, cfg)
    _validate_immutable_v42_archived_snapshot_diagnostic_args(args, cfg)
    if args.partial_dense_duty_rank_diagnostic:
        cfg.stage_b_dense_duty_partial_rank_diagnostic = True
        torch.multiprocessing.set_sharing_strategy("file_system")
        print(
            "[INFO] torch multiprocessing sharing strategy: file_system",
            flush=True,
        )
    if args.partial_dense_duty_confidence_diagnostic:
        cfg.stage_b_dense_duty_partial_confidence_diagnostic = True
        torch.multiprocessing.set_sharing_strategy("file_system")
        print(
            "[INFO] torch multiprocessing sharing strategy: file_system",
            flush=True,
        )
    if args.immutable_v39_archived_snapshot_diagnostic:
        cfg.stage_b_dense_duty_immutable_v39_archived_snapshot_diagnostic = True
    if args.immutable_v40_archived_snapshot_diagnostic:
        cfg.stage_b_dense_duty_immutable_v40_archived_snapshot_diagnostic = True
    if args.immutable_v41_archived_snapshot_diagnostic:
        cfg.stage_b_dense_duty_immutable_v41_archived_snapshot_diagnostic = True
    cfg.device = str(device)
    adapter_enabled = bool(getattr(cfg, "stage_b_gdino_score_adapter", False))
    u0_patch_rank = bool(getattr(cfg, "stage_b_u0_patch_rank", False))
    data_driven_score = bool(getattr(cfg, "stage_b_data_driven_score", False))
    _validate_native_patch_category_joint_request(args, cfg)
    if u0_patch_rank and not adapter_enabled:
        raise ValueError(
            "stage_b_u0_patch_rank evaluation requires stage_b_gdino_score_adapter"
        )
    cfg.patch_only = False
    cfg.use_coco_eval = False
    # Evaluation has no backward pass. Checkpointing only retains allocator
    # pressure here and can OOM on the longer strict semantic expressions.
    cfg.use_checkpoint = False
    cfg.use_transformer_ckpt = False
    cfg.batch_size = int(args.batch_size)
    cfg.build_text_token_masks = True
    cfg.text_mask_warn_limit = 0
    if args.direct_prebuilt_tn and bool(
        getattr(cfg, "stage_b_gdino_score_adapter", False)
    ):
        raise ValueError(
            "--direct_prebuilt_tn is restricted to the PIVOT post-candidate scorer"
        )

    canonical_json = data_root / "canonical_classes_with_aliases.json"
    name_to_id, id_to_name = _load_canonical_name_maps(canonical_json)
    phrase_maps = _load_phrase_maps(
        [
            data_root / "refcoco_text_pairs" / "refcoco_unc_pairs.jsonl",
            data_root / "data_proc" / "refcoco_text_pairs" / "refcoco+_unc_pairs.jsonl",
            data_root / "data_proc" / "refcoco_text_pairs" / "refcocog_google_pairs.jsonl",
            data_root / "patch_episode_prebuilt" / "refcocoplus_stageb_phrase_v1.jsonl",
            data_root / "patch_episode_prebuilt" / "refcocog_stageb_phrase_v1.jsonl",
        ]
    )
    holdout_ann_keys, holdout_image_ids = load_holdout_keys(args.exclude_train_jsonl)
    if args.holdout_level != "none":
        print(
            f"[INFO] holdout level={args.holdout_level} "
            f"ann_keys={len(holdout_ann_keys)} image_ids={len(holdout_image_ids)}",
            flush=True,
        )
    split_specs = {spec["name"]: spec for spec in _default_splits()}
    wanted_ref = list(args.ref_splits)
    if wanted_ref == ["all"]:
        wanted_ref = list(split_specs)
    unknown = [name for name in wanted_ref if name not in split_specs]
    if unknown:
        raise KeyError(f"Unknown ref split names: {unknown}; available={list(split_specs)}")
    ref_split_seeds = _requested_ref_split_seed_map(wanted_ref, int(args.seed))
    category_gate_sweep_gaps = _validate_category_gate_sweep_mode(
        args, cfg, wanted_ref
    )

    ref_datasetinfos = []
    if not bool(args.skip_ref):
        for name in wanted_ref:
            spec = split_specs[name]
            jsonl_path, count = _build_split_jsonl(
                data_root=data_root,
                output_dir=output_dir,
                dataset=spec["dataset"],
                splitby=spec["splitby"],
                split=spec["split"],
                phrase_sources=list(spec["sources"]),
                phrase_maps=phrase_maps,
                name_to_id=name_to_id,
                id_to_name=id_to_name,
                holdout_level=args.holdout_level,
                holdout_ann_keys=holdout_ann_keys,
                holdout_image_ids=holdout_image_ids,
            )
            print(f"[INFO] built RefCOCO split {name}: {count} expressions -> {jsonl_path}", flush=True)
            ref_datasetinfos.append(
                (
                    name,
                    _make_datasetinfo(
                        data_root,
                        name,
                        jsonl_path,
                        adapter_no_support=bool(
                            getattr(cfg, "stage_b_gdino_score_adapter", False)
                        ) and not u0_patch_rank,
                        u0_patch_rank=u0_patch_rank,
                    ),
                )
            )

    tn_meta_rows: List[Dict[str, Any]] = []
    tn_datasetinfo = None
    screen_calibration_binding: Optional[ScreenCalibrationBinding] = None
    matched_eval_binding: Optional[MatchedEvalSurfaceBinding] = None
    direct_eval_scope: Optional[str] = None
    if not bool(args.skip_tn):
        if args.direct_prebuilt_tn:
            tn_eval_jsonl = tn_jsonl.resolve(strict=True)
            matched_eval_binding = load_matched_eval_surface_binding(
                Path(args.direct_prebuilt_tn_binding),
                expected_derived=tn_eval_jsonl,
            )
            tn_meta_rows = matched_eval_surface_meta_rows(matched_eval_binding)
            tn_counts = {"matched_calibration": len(tn_meta_rows)}
            direct_eval_scope = MATCHED_EVAL_SCOPE
        elif args.screen_calibration_manifest:
            tn_eval_jsonl = (
                output_dir / "tn_eval_inputs/tn_screen_calibration.jsonl"
            )
            screen_calibration_binding = build_screen_calibration_manifest(
                source_path=tn_jsonl,
                audit_path=Path(args.screen_calibration_audit),
                derived_path=tn_eval_jsonl,
                data_root=data_root,
            )
            tn_meta_rows = screen_calibration_meta_rows(
                screen_calibration_binding
            )
            tn_counts = {"screen_calibration": len(tn_meta_rows)}
        else:
            tn_eval_jsonl, tn_meta_rows, tn_counts = _build_tn_eval_jsonl(
                data_root=data_root,
                output_dir=output_dir,
                tn_jsonl=tn_jsonl,
                splits=list(args.tn_splits),
                max_pairs=0,
                holdout_level=args.holdout_level,
                holdout_ann_keys=holdout_ann_keys,
                holdout_image_ids=holdout_image_ids,
            )
        print(f"[INFO] built TN split rows={len(tn_meta_rows)} counts={tn_counts} -> {tn_eval_jsonl}", flush=True)
        with Path(tn_eval_jsonl).open("r", encoding="utf-8") as handle:
            selected_tn_rows = [
                json.loads(line) for line in handle if line.strip()
            ]
        tn_eval_scope = (
            _validate_direct_prebuilt_tn_rows(
                selected_tn_rows, declared_scope=MATCHED_EVAL_SCOPE
            )
            if args.direct_prebuilt_tn
            else _validate_adapter_tn_eval_manifest(cfg, selected_tn_rows)
        )
        tn_eval_protocol = (
            "stageb_vlm_verified_strict_tn_v2"
            if tn_eval_scope is not None
            and all(
                row.get("manifest_schema", None)
                == "stageb_vlm_verified_strict_tn_v2"
                for row in selected_tn_rows
            )
            else (
                "adapter_training_pair_schema"
                if tn_eval_scope is not None
                else None
            )
        )
        tn_datasetinfo = _make_tn_datasetinfo(
            data_root,
            tn_eval_jsonl,
            adapter_eval_scope=(
                None if args.direct_prebuilt_tn else tn_eval_scope
            ),
            adapter_eval_protocol=tn_eval_protocol,
            u0_patch_rank=u0_patch_rank,
            data_driven_score=data_driven_score,
        )
        if args.direct_prebuilt_tn:
            # Cache presence/mtime must not change bank construction or RNG
            # consumption between the independently evaluated conditions.
            tn_datasetinfo.update(
                patch_bank_cache=False,
                patch_bank_cache_write=False,
            )

    ref_rows: List[Dict[str, Any]] = []
    tn_rows: List[Dict[str, Any]] = []
    for ckpt_i, ckpt_path in enumerate(args.ckpts):
        ckpt_path = str(ckpt_path)
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(ckpt_path)
        print(f"[INFO] loading checkpoint {ckpt_i + 1}/{len(args.ckpts)}: {ckpt_path}", flush=True)
        _set_seed(int(args.seed))
        model, checkpoint_summary_fields = _load_model_with_checkpoint_contract(
            cfg, ckpt_path, device
        )
        evaluation_summary_provenance = _evaluation_summary_provenance(
            cfg=cfg,
            args=args,
            checkpoint=ckpt_path,
            data_root=data_root,
        )
        for name, datasetinfo in ref_datasetinfos:
            if category_gate_sweep_gaps is None:
                split_rows = [
                    evaluate_refcoco_dataset(
                        cfg=cfg,
                        model=model,
                        ckpt_path=ckpt_path,
                        datasetinfo=datasetinfo,
                        dataset_name=name,
                        device=device,
                        batch_size=int(args.batch_size),
                        num_workers=int(args.num_workers),
                        seed=ref_split_seeds[name],
                        topks=list(args.topk),
                        amp=bool(args.amp),
                        max_batches=int(args.max_ref_batches),
                        log_every=int(args.log_every),
                        records_output_dir=(
                            None
                            if args.no_per_example_records
                            else output_dir / "per_example_records"
                        ),
                        checkpoint_summary_fields=checkpoint_summary_fields,
                    )
                ]
            else:
                split_rows = evaluate_refcoco_category_gate_sweep(
                    cfg=cfg,
                    model=model,
                    ckpt_path=ckpt_path,
                    datasetinfo=datasetinfo,
                    dataset_name=name,
                    device=device,
                    batch_size=int(args.batch_size),
                    num_workers=int(args.num_workers),
                    seed=ref_split_seeds[name],
                    topks=list(args.topk),
                    max_gaps=category_gate_sweep_gaps,
                    amp=bool(args.amp),
                    max_batches=int(args.max_ref_batches),
                    log_every=int(args.log_every),
                    records_output_dir=output_dir / "per_example_records",
                    include_base_expert=bool(
                        args.category_gate_include_base_expert
                    ),
                    checkpoint_summary_fields=checkpoint_summary_fields,
                )
            for row in split_rows:
                _bind_evaluation_summary_provenance(
                    row,
                    evaluation_summary_provenance,
                )
                ref_rows.append(row)
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / f"{row['run_id']}__{name}.json").write_text(
                    json.dumps(row, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                _write_summary(output_dir, ref_rows, tn_rows)
                print(
                    f"[RESULT] {row['run_id']} {name}: "
                    f"acc50={row['acc50']:.6f} "
                    f"mean_iou={row['mean_iou']:.6f}",
                    flush=True,
                )
        if not bool(args.skip_tn):
            assert tn_datasetinfo is not None
            tn_row = evaluate_tn_dataset(
                cfg=cfg,
                model=model,
                ckpt_path=ckpt_path,
                datasetinfo=tn_datasetinfo,
                meta_rows=tn_meta_rows,
                device=device,
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                seed=int(args.seed),
                threshold_tprs=list(args.threshold_tprs),
                score_thresholds=list(args.score_thresholds),
                amp=bool(args.amp),
                max_batches=int(args.max_tn_batches),
                log_every=int(args.log_every),
                records_output_dir=(
                    None if args.no_per_example_records else output_dir / "per_example_records"
                ),
                checkpoint_summary_fields=checkpoint_summary_fields,
                declared_eval_scope=direct_eval_scope,
            )
            if bool(args.immutable_v39_archived_snapshot_diagnostic):
                post_evaluation = (
                    _verify_v39_immutable_archived_diagnostic_files(
                        Path(ckpt_path)
                    )
                )
                immutable_audit = getattr(
                    cfg,
                    "stage_b_dense_duty_immutable_v39_archived_snapshot_audit",
                )
                immutable_audit["immutable_archived_provenance"][
                    "snapshot_after_evaluation"
                ] = post_evaluation["snapshot"]
                immutable_audit["immutable_archived_provenance"][
                    "terminal_after_evaluation"
                ] = post_evaluation["terminal"]
                evaluation_summary_provenance = _evaluation_summary_provenance(
                    cfg=cfg,
                    args=args,
                    checkpoint=ckpt_path,
                    data_root=data_root,
                )
            elif bool(args.immutable_v40_archived_snapshot_diagnostic):
                post_evaluation = (
                    _verify_v40_immutable_archived_diagnostic_files(
                        Path(ckpt_path)
                    )
                )
                immutable_audit = getattr(
                    cfg,
                    "stage_b_dense_duty_immutable_v40_archived_snapshot_audit",
                )
                immutable_audit["immutable_archived_provenance"][
                    "snapshot_after_evaluation"
                ] = post_evaluation["snapshot"]
                immutable_audit["immutable_archived_provenance"][
                    "terminal_after_evaluation"
                ] = post_evaluation["terminal"]
                cfg.stage_b_dense_duty_immutable_v40_archived_snapshot_audit = (
                    immutable_audit
                )
                evaluation_summary_provenance = _evaluation_summary_provenance(
                    cfg=cfg,
                    args=args,
                    checkpoint=ckpt_path,
                    data_root=data_root,
                )
            elif bool(args.immutable_v41_archived_snapshot_diagnostic):
                post_evaluation = (
                    _verify_v41_immutable_archived_diagnostic_files(
                        Path(ckpt_path)
                    )
                )
                immutable_audit = getattr(
                    cfg,
                    "stage_b_dense_duty_immutable_v41_archived_snapshot_audit",
                )
                immutable_audit["immutable_archived_provenance"][
                    "snapshot_after_evaluation"
                ] = post_evaluation["snapshot"]
                immutable_audit["immutable_archived_provenance"][
                    "terminal_after_evaluation"
                ] = post_evaluation["terminal"]
                cfg.stage_b_dense_duty_immutable_v41_archived_snapshot_audit = (
                    immutable_audit
                )
                evaluation_summary_provenance = _evaluation_summary_provenance(
                    cfg=cfg,
                    args=args,
                    checkpoint=ckpt_path,
                    data_root=data_root,
                )
            elif bool(args.immutable_v42_archived_snapshot_diagnostic):
                post_evaluation = (
                    _verify_v42_immutable_archived_diagnostic_files(
                        Path(ckpt_path)
                    )
                )
                immutable_audit = getattr(
                    cfg,
                    "stage_b_dense_duty_immutable_v42_archived_snapshot_audit",
                )
                immutable_audit["immutable_archived_provenance"][
                    "snapshot_after_evaluation"
                ] = post_evaluation["snapshot"]
                immutable_audit["immutable_archived_provenance"][
                    "terminal_after_evaluation"
                ] = post_evaluation["terminal"]
                cfg.stage_b_dense_duty_immutable_v42_archived_snapshot_audit = (
                    immutable_audit
                )
                evaluation_summary_provenance = _evaluation_summary_provenance(
                    cfg=cfg,
                    args=args,
                    checkpoint=ckpt_path,
                    data_root=data_root,
                )
            _bind_evaluation_summary_provenance(
                tn_row,
                evaluation_summary_provenance,
            )
            if screen_calibration_binding is not None:
                tn_row.update(
                    screen_calibration_summary_fields(screen_calibration_binding)
                )
            if matched_eval_binding is not None:
                tn_row.update(matched_eval_surface_summary_fields(matched_eval_binding))
            tn_rows.append(tn_row)
            (output_dir / f"{_ckpt_run_prefix(ckpt_path)}__tn_val.json").write_text(
                json.dumps(tn_row, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            _write_summary(output_dir, ref_rows, tn_rows)
            print(
                f"[RESULT] {tn_row['run_id']} TN: fpr95={tn_row.get('fpr95tpr', 0.0):.6f} "
                f"fpr90={tn_row.get('fpr90tpr', 0.0):.6f} pair_win={tn_row.get('pair_win_rate', 0.0):.6f}",
                flush=True,
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _write_summary(output_dir, ref_rows, tn_rows)
    print(f"[INFO] wrote {output_dir / 'summary.json'}", flush=True)
    print(f"[INFO] wrote {output_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
