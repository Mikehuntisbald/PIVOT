#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import random
import re
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_HISTORICAL_DENSE_DUTY_U50_SOURCES = {
    "word_veto_gated_pool_carrier_affine_v15": {
        "config": (
            REPO_ROOT
            / "config/ablations/"
            "cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_"
            "carrier_affine_probe_u0050_20260731.py"
        ),
        "checkpoint": (
            REPO_ROOT
            / "outputs/paper_cvpr_v1/"
            "dense_duty_adapter_veto_gated_pool_carrier_affine_highmem_20260731/"
            "probe/u000050/checkpoint_iter.pth"
        ),
        "checkpoint_sha256": (
            "0068856a6b3d7be1fe5f975f8d94fafa5df199ed511a6534e52258e657fdfad2"
        ),
        "archive": (
            REPO_ROOT
            / "outputs/paper_cvpr_v1/"
            "dense_duty_adapter_veto_gated_pool_carrier_affine_highmem_20260731/"
            "probe/u000050/artifacts/"
            "v15_u50_exact_source_snapshot_20260731_074911.tar.gz"
        ),
        "archive_sha256": (
            "d55a4f45bed4c7a2dc474583d38f556dfcba82be3237de57582f591945ce03a1"
        ),
        "source_closure_sha256": (
            "4dec83ff050f36aa8c2615ab8c3b2935e8fca2c5f61239cb302620e8a18c0718"
        ),
    },
    "word_veto_gated_pool_tail_carrier_v17": {
        "config": (
            REPO_ROOT
            / "config/ablations/"
            "cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_"
            "tail_carrier_probe_u0050_20260731.py"
        ),
        "checkpoint": (
            REPO_ROOT
            / "outputs/paper_cvpr_v1/"
            "dense_duty_adapter_veto_gated_pool_tail_carrier_highmem_20260731/"
            "probe/u000050/checkpoint_iter.pth"
        ),
        "checkpoint_sha256": (
            "277a31ee77c8922cbf13d3cac45c8e4f250e69aab5016883ea106a33d5209b3b"
        ),
        "archive": (
            REPO_ROOT
            / "outputs/paper_cvpr_v1/"
            "dense_duty_adapter_veto_gated_pool_tail_carrier_highmem_20260731/"
            "probe/u000050/artifacts/"
            "v17_u50_exact_source_snapshot_20260731_082900.tar.gz"
        ),
        "archive_sha256": (
            "650b13e76e781690536e75b6931e576af3ac3f13bca7859cd9226c4c95795373"
        ),
        "source_closure_sha256": (
            "6890db0e2acc927a69771d739fd46ef145de8744d6badab7529f247ef5529562"
        ),
    },
    "word_veto_gated_pool_tail_paired_rank_channel_v19": {
        "config": (
            REPO_ROOT
            / "config/ablations/"
            "cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_"
            "tail_paired_rank_channel_probe_u0050_20260731.py"
        ),
        "checkpoint": (
            REPO_ROOT
            / "outputs/paper_cvpr_v1/"
            "dense_duty_adapter_veto_gated_pool_tail_paired_rank_channel_"
            "highmem_20260731/probe/u000050/checkpoint_iter.pth"
        ),
        "checkpoint_sha256": (
            "cb396b60c6bc3052f4e7501a5989cba9e7df19669ce61eca5336df9a2d3a40ed"
        ),
        "archive": (
            REPO_ROOT
            / "outputs/paper_cvpr_v1/"
            "dense_duty_adapter_veto_gated_pool_tail_paired_rank_channel_"
            "highmem_20260731/probe/u000050/artifacts/"
            "v19_u50_exact_source_snapshot_20260731_102309.tar.gz"
        ),
        "archive_sha256": (
            "37d45dc185589d138efbaa09de0e5c9f4f300e0f974177497a1e532049b5021b"
        ),
        "source_closure_sha256": (
            "5b9802600bfd82d40965cbc06e1b027ae3b0a01492f92de27840ef415d7702f4"
        ),
    },
    "word_veto_gated_pool_tail_paired_signed_rank_pool_v20": {
        "config": (
            REPO_ROOT
            / "config/ablations/"
            "cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_"
            "tail_paired_signed_rank_pool_probe_u0050_20260731.py"
        ),
        "checkpoint": (
            REPO_ROOT
            / "outputs/paper_cvpr_v1/"
            "dense_duty_adapter_veto_gated_pool_tail_paired_signed_rank_pool_"
            "highmem_20260731/probe/u000050/checkpoint_iter.pth"
        ),
        "checkpoint_sha256": (
            "31230a09417e8f7d3ab01d2d09fdfd673a6bdcd80151729c7998363535a2cd62"
        ),
        "archive": (
            REPO_ROOT
            / "outputs/paper_cvpr_v1/"
            "dense_duty_adapter_veto_gated_pool_tail_paired_signed_rank_pool_"
            "highmem_20260731/probe/u000050/artifacts/"
            "v20_u50_exact_source_snapshot_20260731_110500.tar.gz"
        ),
        "archive_sha256": (
            "6a763fa6aa934de93a35b1348fef8484ab058488574f90df049b7d98981fdca6"
        ),
        "source_closure_sha256": (
            "34dcbed896fa8614ceb22b773d8f4f4a68b7736e96fc5308bdc00dd8aeb6e09f"
        ),
    },
}

_HISTORICAL_DENSE_DUTY_U100_SOURCES = {
    "word_veto_gated_pool_tail_paired_v18": {
        "config": (
            REPO_ROOT
            / "config/ablations/"
            "cfg_stageb_dense_duty_confidence_adapter_veto_gated_pool_"
            "tail_paired_probe_u0100_20260731.py"
        ),
        "checkpoint": (
            REPO_ROOT
            / "outputs/paper_cvpr_v1/"
            "dense_duty_adapter_veto_gated_pool_tail_paired_highmem_20260731/"
            "probe/u000100/checkpoint_iter.pth"
        ),
        "checkpoint_sha256": (
            "3db0eec0901c1d7e99fbc4de42391907188b9862761cf4d93518553572e38ae6"
        ),
        "archive": (
            REPO_ROOT
            / "outputs/paper_cvpr_v1/"
            "dense_duty_adapter_veto_gated_pool_tail_paired_highmem_20260731/"
            "probe/u000100/artifacts/"
            "v18_u100_exact_source_snapshot_20260731_095030.tar.gz"
        ),
        "archive_sha256": (
            "2ff302b462033a2f53583409d7522e9f214ad0e37b8576d109bad0b594a46c36"
        ),
        "source_closure_sha256": (
            "fa43dd5f4ca4ee430be07da8307165b1c124cc2da876a9d593090b9ccd95e59b"
        ),
    },
}

from datasets import build_dataset  # noqa: E402
from tools.stageb_eval_holdout import is_excluded, load_holdout_keys  # noqa: E402
from groundingdino.util import box_ops  # noqa: E402
from groundingdino.util.utils import clean_state_dict  # noqa: E402
from models.GroundingDINO.stage_b_fixed_text_scorer import (  # noqa: E402
    validate_stage_b_fixed_text_scorer_checkpoint,
)
from models.GroundingDINO.stage_b_gdino_score_adapter import (  # noqa: E402
    aggregate_gdino_full_expression_score,
)
from models.GroundingDINO.stage_b_native_patch_category import (  # noqa: E402
    NATIVE_PATCH_CATEGORY_CONTRACT_VERSION,
    apply_native_patch_category_gate,
)
from models.GroundingDINO.stage_b_native_patch_category_d2 import (  # noqa: E402
    NATIVE_PATCH_CATEGORY_D2_CONTRACT_VERSION,
)
from models.GroundingDINO.stage_b_native_patch_category_d3 import (  # noqa: E402
    NATIVE_PATCH_CATEGORY_D3_CONTRACT_VERSION,
)
from models.GroundingDINO.stage_b_native_patch_category_d4 import (  # noqa: E402
    NATIVE_PATCH_CATEGORY_D4_CONTRACT_VERSION,
)
from models.GroundingDINO.stage_b_native_patch_category_d5 import (  # noqa: E402
    NATIVE_PATCH_CATEGORY_D5_CONTRACT_VERSION,
)
from models.GroundingDINO.stage_b_native_patch_category_d6 import (  # noqa: E402
    NATIVE_PATCH_CATEGORY_D6_CONTRACT_VERSION,
)
from models.GroundingDINO.stage_b_native_patch_category_d7 import (  # noqa: E402
    NATIVE_PATCH_CATEGORY_D7_CONTRACT_VERSION,
)
from models.GroundingDINO.stage_b_native_patch_category_d8 import (  # noqa: E402
    NATIVE_PATCH_CATEGORY_D8_CONTRACT_VERSION,
)
from models.GroundingDINO.stage_b_native_patch_category_d9 import (  # noqa: E402
    NATIVE_PATCH_CATEGORY_D9_CONTRACT_VERSION,
)
from models.GroundingDINO.stage_b_score import compute_stage_b_slot_logits  # noqa: E402
from models.registry import MODULE_BUILD_FUNCS  # noqa: E402
from tools.eval_stagea_patch_checkpoints import _prepare_patch_batch, _set_seed  # noqa: E402
from tools.stageb_eval_records import (  # noqa: E402
    EvalManifest,
    RECORD_SCHEMA,
    load_eval_manifest,
    make_eval_record,
    validate_eval_manifest_batch_alignment,
    write_eval_records,
)
from tools.stageb_external_rank_transfer_artifact import (  # noqa: E402
    CAPTION_PROVENANCE_CONTRACT,
    REF_SPLIT_ORDER,
    SPLIT_SEED_PROTOCOL,
    evaluator_settings_from_artifact,
    stable_ref_split_seed_map,
)
from tools.stageb_canonical_caption_route_artifact import (  # noqa: E402
    CANDIDATE_DESCRIPTOR_IDS as ROUTE_CANDIDATE_DESCRIPTOR_IDS,
    DEFAULT_DESCRIPTOR_ID as ROUTE_DEFAULT_DESCRIPTOR_ID,
    DESCRIPTOR_REGISTRY as ROUTE_DESCRIPTOR_REGISTRY,
)
from tools.stageb_fulltext_route_gate_artifact import (  # noqa: E402
    GATED_CAPTION as FULLTEXT_GATED_CAPTION,
    GATED_DESCRIPTOR_ID as FULLTEXT_GATED_DESCRIPTOR_ID,
    TOKEN_COUNT_CONTRACT as FULLTEXT_TOKEN_COUNT_CONTRACT,
)
from util import misc as utils  # noqa: E402
from util.slconfig import SLConfig  # noqa: E402

_WS_RE = re.compile(r"\s+")

_EXTERNAL_GDINO_QUERY_COUNT = 900
_DATA_DRIVEN_QUERY_DIAGNOSTIC_CONTRACT = (
    "data_driven_gap3_query_diagnostics_v1"
)
_EXTERNAL_GDINO_RANK_SCORE_KEY = "stage_b_gdino_rank_score"
_EXTERNAL_GDINO_BASE_SCORE_KEY = "stage_b_gdino_base_score"
_U0_RANK_SCORE_KEY = "stage_b_u0_rank_score"
_DATA_DRIVEN_RANK_SCORE_KEY = "stage_b_data_driven_rank_score"
_NATIVE_PATCH_BASE_SCORE_KEY = "stage_b_native_patch_base_score"
_NATIVE_PATCH_RANK_SCORE_KEY = "stage_b_native_patch_rank_score"
_NATIVE_PATCH_ELIGIBLE_MASK_KEY = (
    "stage_b_native_patch_category_eligible_mask"
)
_NATIVE_PATCH_STANDARDIZED_SCORE_KEY = (
    "stage_b_native_patch_category_patch_score"
)
_NATIVE_PATCH_EXPRESSION_MASK_KEY = (
    "stage_b_native_patch_expression_token_mask"
)
_NATIVE_PATCH_GATE_MAX_GAP = 3.0
_NATIVE_PATCH_GATE_CLIP = 5.0
_NATIVE_PATCH_QUERY_COUNT = 900
_NATIVE_PATCH_REF_SCORE_CONTRACT = (
    "native-full-expression-patch-category-gap3-clip5-v1"
)
_EXTERNAL_RANK_TRANSFER_MODES = (
    "nearest_iou",
    "max_score_iou_power",
    "max_score_iou_power_external_box",
    "top_query_nearest_candidate",
    "top_query_nearest_candidate_external_box",
)
_EXTERNAL_RANK_TRANSFER_CONTRACT_VERSION = 1
_PATCH_INTERNAL_RANK_IDENTITY_KIND = "patch_internal_rank_identity"
_PATCH_INTERNAL_RANK_IDENTITY_CONTRACT_VERSION = 1
_EXTERNAL_GDINO_BASE_IDENTITY_KIND = "external_gdino_base_identity"
_EXTERNAL_GDINO_BASE_IDENTITY_ID = "external_gdino_base_direct"
_EXTERNAL_GDINO_BASE_IDENTITY_CONTRACT_VERSION = 1
_FULLTEXT_LEXICAL_TOKEN_RE = re.compile(r"[a-z0-9]+")
_FULLTEXT_ROUTE_GATE_CONTRACT_VERSION = 1
_FULLTEXT_CAPTIONS_OUTPUT_KEY = "validated_full_expression_captions"

_V43_DEPLOYED_ROUTING_REVISION = (
    "word_veto_candidate_asymmetric_deployed_routing_v43"
)
_V43_DEPLOYED_ROUTING_GATE_CONTRACT = (
    "candidate_raw_patch_asymmetric_deployed_routing_st_v15"
)
_V43_DEPLOYED_ROUTING_WEIGHT = 0.1
_V43_DEPLOYED_ROUTING_POSITIVE_MAX = 0.1
_V43_DEPLOYED_ROUTING_TN_MIN = 0.9

_V45_SPLIT_TAIL_ALIGNED_REVISION = (
    "word_veto_candidate_split_tail_aligned_v45"
)
_V45_SPLIT_TAIL_ALIGNED_HEAD_CONTRACT = (
    "split_token_veto_global_absolute_joint_clip_v3"
)
_V45_SPLIT_TAIL_ALIGNED_ROUTING_REDUCTION = (
    "balanced_top_quarter_cvar_v2"
)
_V45_SPLIT_TAIL_ALIGNED_TRUST_REDUCTION = "top_quarter_cvar_v2"
_V45_SPLIT_TAIL_ALIGNED_POSITIVE_TRUST = (
    "absolute_global_confidence_logit_v2"
)

_V46_SPLIT_POSITIVE_TAIL_REVISION = (
    "word_veto_candidate_split_positive_tail_v46"
)
_V46_SPLIT_POSITIVE_TAIL_HEAD_CONTRACT = (
    "split_token_veto_global_absolute_v2"
)
_V46_SPLIT_POSITIVE_TAIL_ROUTING_REDUCTION = "balanced_mean_v1"
_V46_SPLIT_POSITIVE_TAIL_TRUST_REDUCTION = "top_quarter_cvar_v2"
_V46_SPLIT_POSITIVE_TAIL_POSITIVE_TRUST = (
    "absolute_global_confidence_logit_v2"
)

_V47_SPLIT_BOUNDARY_ROUTING_REVISION = (
    "word_veto_candidate_split_boundary_routing_v47"
)
_V47_SPLIT_BOUNDARY_ROUTING_HEAD_CONTRACT = (
    "split_token_veto_global_absolute_v2"
)
_V47_SPLIT_BOUNDARY_ROUTING_ROUTING_REDUCTION = (
    "balanced_top_quarter_cvar_v2"
)
_V47_SPLIT_BOUNDARY_ROUTING_TRUST_REDUCTION = "top_quarter_cvar_v2"
_V47_SPLIT_BOUNDARY_ROUTING_POSITIVE_TRUST = (
    "absolute_global_confidence_logit_v2"
)

_V48_SPLIT_FPR_ACTIVE_SET_REVISION = (
    "word_veto_candidate_split_fpr_active_set_v48"
)
_V48_SPLIT_FPR_ACTIVE_SET_NEGATIVE_REDUCTION = (
    "exact_fpr95_active_set_mean_v1"
)

_V49_SPLIT_GLOBAL_TRUST_VETO_REVISION = (
    "word_veto_candidate_split_global_trust_veto_v49"
)
_V49_SPLIT_GLOBAL_TRUST_VETO_HEAD_CONTRACT = (
    "split_token_veto_global_trust_veto_v4"
)
_V49_SPLIT_GLOBAL_TRUST_VETO_NEGATIVE_REDUCTION = "all_mean_v1"

_V50_SPLIT_STRONG_BOUNDARY_ROUTING_REVISION = (
    "word_veto_candidate_split_strong_boundary_routing_v50"
)
_V50_SPLIT_STRONG_BOUNDARY_ROUTING_HEAD_CONTRACT = (
    "split_token_veto_global_absolute_v2"
)
_V50_SPLIT_STRONG_BOUNDARY_ROUTING_WEIGHT = 0.25
_V50_SPLIT_STRONG_BOUNDARY_ROUTING_ROUTING_REDUCTION = (
    "balanced_top_quarter_cvar_v2"
)
_V50_SPLIT_STRONG_BOUNDARY_ROUTING_TRUST_REDUCTION = "top_quarter_cvar_v2"
_V50_SPLIT_STRONG_BOUNDARY_ROUTING_POSITIVE_TRUST = (
    "absolute_global_confidence_logit_v2"
)
_V50_SPLIT_STRONG_BOUNDARY_ROUTING_NEGATIVE_REDUCTION = "all_mean_v1"
_V50_SPLIT_STRONG_BOUNDARY_ROUTING_TRAINING_CONTRACT_SCHEMA = (
    "pivot.stageb.dense_duty_training_contract/v32"
)

_V51_SPLIT_INDEPENDENT_DEPLOYED_ROUTER_REVISION = (
    "word_veto_candidate_split_independent_deployed_router_v51"
)
_V51_SPLIT_INDEPENDENT_DEPLOYED_ROUTER_HEAD_CONTRACT = (
    "split_token_veto_deployed_router_global_absolute_v5"
)
_V51_SPLIT_INDEPENDENT_DEPLOYED_ROUTER_WEIGHT = 0.1
_V51_SPLIT_INDEPENDENT_DEPLOYED_ROUTER_ROUTING_REDUCTION = (
    "balanced_top_quarter_cvar_v2"
)
_V51_SPLIT_INDEPENDENT_DEPLOYED_ROUTER_TRUST_REDUCTION = (
    "top_quarter_cvar_v2"
)
_V51_SPLIT_INDEPENDENT_DEPLOYED_ROUTER_POSITIVE_TRUST = (
    "absolute_global_confidence_logit_v2"
)
_V51_SPLIT_INDEPENDENT_DEPLOYED_ROUTER_NEGATIVE_REDUCTION = "all_mean_v1"
_V51_SPLIT_INDEPENDENT_DEPLOYED_ROUTER_TRAINING_CONTRACT_SCHEMA = (
    "pivot.stageb.dense_duty_training_contract/v33"
)

_V52_CANDIDATE_SAMPLE_CALIBRATOR_REVISION = (
    "word_veto_candidate_sample_calibrator_split_v52"
)
_V52_CANDIDATE_SAMPLE_CALIBRATOR_HEAD_CONTRACT = (
    "split_token_veto_candidate_absolute_sample_calibrator_v6"
)
_V52_CANDIDATE_SAMPLE_CALIBRATOR_ROUTING_WEIGHT = 0.0
_V52_CANDIDATE_SAMPLE_CALIBRATOR_ROUTING_REDUCTION = (
    "balanced_top_quarter_cvar_v2"
)
_V52_CANDIDATE_SAMPLE_CALIBRATOR_TRUST_REDUCTION = "top_quarter_cvar_v2"
_V52_CANDIDATE_SAMPLE_CALIBRATOR_POSITIVE_TRUST = (
    "absolute_global_confidence_logit_v2"
)
_V52_CANDIDATE_SAMPLE_CALIBRATOR_NEGATIVE_REDUCTION = "all_mean_v1"
_V52_CANDIDATE_SAMPLE_CALIBRATOR_POOL_FEATURE_CONTRACT = (
    "detached_candidate_absolute_raw_patch_asymmetric_veto_logits_v8"
)
_V52_CANDIDATE_SAMPLE_CALIBRATOR_SOURCE_UPDATES = 6551
_V52_CANDIDATE_SAMPLE_CALIBRATOR_TRAINABLE_PARAMS = 535_945
_V52_CANDIDATE_SAMPLE_CALIBRATOR_TRAINING_CONTRACT_SCHEMA = (
    "pivot.stageb.dense_duty_training_contract/v34"
)
_V52_CANDIDATE_SAMPLE_CALIBRATOR_MIGRATION_SCHEMA = (
    "pivot.stageb.rank_to_token_confidence_adapter_candidate_sample_calibrator/v19"
)
_V52_THREE_OWNER_CLIP_CONTRACT_SCHEMA = (
    "pivot.stageb.dense_duty_three_owner_clip_contract/v1"
)

_V53_FULLTEXT_GLOBAL_ABSOLUTE_REVISION = (
    "word_veto_rank_full_expression_global_absolute_v53"
)
_V53_FULLTEXT_GLOBAL_ABSOLUTE_HEAD_CONTRACT = (
    "split_token_veto_fulltext_global_absolute_v7"
)
_V53_FULLTEXT_GLOBAL_ABSOLUTE_GATE_CONTRACT = (
    "candidate_raw_patch_asymmetric_monotone_veto_absolute_logit_v13"
)
_V53_FULLTEXT_GLOBAL_ABSOLUTE_POOL_FEATURE_CONTRACT = (
    "detached_rank_full_expression_candidate_residual_global_pool_v10"
)
_V53_FULLTEXT_GLOBAL_ABSOLUTE_ROUTING_WEIGHT = 0.0
_V53_FULLTEXT_GLOBAL_ABSOLUTE_ROUTING_REDUCTION = (
    "balanced_top_quarter_cvar_v2"
)
_V53_FULLTEXT_GLOBAL_ABSOLUTE_TRUST_REDUCTION = "top_quarter_cvar_v2"
_V53_FULLTEXT_GLOBAL_ABSOLUTE_POSITIVE_TRUST = (
    "absolute_global_confidence_logit_v2"
)
_V53_FULLTEXT_GLOBAL_ABSOLUTE_NEGATIVE_REDUCTION = "all_mean_v1"
_V53_FULLTEXT_GLOBAL_ABSOLUTE_CARRIER_SELECTOR = (
    "final_layer_reference_argmax_exact_eligible_v1"
)
_V53_FULLTEXT_GLOBAL_ABSOLUTE_SOURCE_UPDATES = 6551
_V53_FULLTEXT_GLOBAL_ABSOLUTE_TRAINABLE_PARAMS = 534_725
_V53_FULLTEXT_GLOBAL_ABSOLUTE_TRAINING_CONTRACT_SCHEMA = (
    "pivot.stageb.dense_duty_training_contract/v35"
)
_V53_FULLTEXT_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA = (
    "pivot.stageb.rank_to_token_confidence_adapter_fulltext_global_absolute/v20"
)
_V53_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT = (
    "token_adapter_rank_full_expression_candidate_residual_global_absolute_v18"
)
_V53_TWO_OWNER_CLIP_CONTRACT_SCHEMA = (
    "pivot.stageb.dense_duty_two_owner_clip_contract/v1"
)
_V53_TOKEN_VETO_TENSOR_COUNT = 21
_V53_GLOBAL_ABSOLUTE_TENSOR_COUNT = 44

_V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_REVISION = (
    "word_veto_rank_full_expression_global_absolute_exact_residual_v54"
)
_V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_HEAD_CONTRACT = (
    _V53_FULLTEXT_GLOBAL_ABSOLUTE_HEAD_CONTRACT
)
_V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_GATE_CONTRACT = (
    _V53_FULLTEXT_GLOBAL_ABSOLUTE_GATE_CONTRACT
)
_V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_POOL_FEATURE_CONTRACT = (
    "detached_rank_full_expression_candidate_residual_global_pool_"
    "exact_rank_max_reference_v11"
)
_V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_ROUTING_WEIGHT = (
    _V53_FULLTEXT_GLOBAL_ABSOLUTE_ROUTING_WEIGHT
)
_V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_ROUTING_REDUCTION = (
    _V53_FULLTEXT_GLOBAL_ABSOLUTE_ROUTING_REDUCTION
)
_V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_TRUST_REDUCTION = (
    _V53_FULLTEXT_GLOBAL_ABSOLUTE_TRUST_REDUCTION
)
_V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_POSITIVE_TRUST = (
    "exact_frozen_rank_max_confidence_delta_v3"
)
_V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_NEGATIVE_REDUCTION = (
    _V53_FULLTEXT_GLOBAL_ABSOLUTE_NEGATIVE_REDUCTION
)
_V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_CARRIER_SELECTOR = (
    _V53_FULLTEXT_GLOBAL_ABSOLUTE_CARRIER_SELECTOR
)
_V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_SOURCE_UPDATES = (
    _V53_FULLTEXT_GLOBAL_ABSOLUTE_SOURCE_UPDATES
)
_V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_TRAINABLE_PARAMS = (
    _V53_FULLTEXT_GLOBAL_ABSOLUTE_TRAINABLE_PARAMS
)
_V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_TRAINING_CONTRACT_SCHEMA = (
    "pivot.stageb.dense_duty_training_contract/v36"
)
_V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_MIGRATION_SCHEMA = (
    "pivot.stageb.rank_to_token_confidence_adapter_"
    "fulltext_global_absolute_exact_residual/v21"
)
_V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_FRESH_CONFIDENCE_CONTRACT = (
    "token_adapter_rank_full_expression_candidate_residual_global_absolute_"
    "exact_rank_max_residual_trust_v19"
)
_V54_TWO_OWNER_CLIP_CONTRACT_SCHEMA = _V53_TWO_OWNER_CLIP_CONTRACT_SCHEMA
_V54_TOKEN_VETO_TENSOR_COUNT = _V53_TOKEN_VETO_TENSOR_COUNT
_V54_GLOBAL_ABSOLUTE_TENSOR_COUNT = _V53_GLOBAL_ABSOLUTE_TENSOR_COUNT

_V55_FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_REVISION = (
    "word_veto_rank_full_expression_global_independent_absolute_v55"
)
_V55_FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_HEAD_CONTRACT = (
    "split_token_veto_local_candidate_global_absolute_v8"
)
_V55_FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_GATE_CONTRACT = (
    _V53_FULLTEXT_GLOBAL_ABSOLUTE_GATE_CONTRACT
)
_V55_FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_POOL_FEATURE_CONTRACT = (
    "detached_rank_full_expression_local_candidate_"
    "frozen_rank_global_pool_v12"
)
_V55_FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_ROUTING_WEIGHT = (
    _V53_FULLTEXT_GLOBAL_ABSOLUTE_ROUTING_WEIGHT
)
_V55_FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_ROUTING_REDUCTION = (
    _V53_FULLTEXT_GLOBAL_ABSOLUTE_ROUTING_REDUCTION
)
_V55_FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_TRUST_REDUCTION = (
    _V53_FULLTEXT_GLOBAL_ABSOLUTE_TRUST_REDUCTION
)
_V55_FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_POSITIVE_TRUST = (
    "absolute_global_pool_logit_v4"
)
_V55_FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_NEGATIVE_REDUCTION = (
    _V53_FULLTEXT_GLOBAL_ABSOLUTE_NEGATIVE_REDUCTION
)
_V55_FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_CARRIER_SELECTOR = (
    _V53_FULLTEXT_GLOBAL_ABSOLUTE_CARRIER_SELECTOR
)
_V55_FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_SOURCE_UPDATES = (
    _V53_FULLTEXT_GLOBAL_ABSOLUTE_SOURCE_UPDATES
)
_V55_FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_TRAINABLE_PARAMS = (
    _V53_FULLTEXT_GLOBAL_ABSOLUTE_TRAINABLE_PARAMS
)
_V55_FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_TRAINING_CONTRACT_SCHEMA = (
    "pivot.stageb.dense_duty_training_contract/v37"
)
_V55_FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_MIGRATION_SCHEMA = (
    "pivot.stageb.rank_to_token_confidence_adapter_"
    "fulltext_global_independent_absolute/v22"
)
_V55_FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT = (
    "token_adapter_rank_full_expression_local_candidate_"
    "global_independent_absolute_v20"
)
_V55_TWO_OWNER_CLIP_CONTRACT_SCHEMA = _V53_TWO_OWNER_CLIP_CONTRACT_SCHEMA
_V55_TOKEN_VETO_TENSOR_COUNT = _V53_TOKEN_VETO_TENSOR_COUNT
_V55_GLOBAL_ABSOLUTE_TENSOR_COUNT = _V53_GLOBAL_ABSOLUTE_TENSOR_COUNT

_V39_IMMUTABLE_ARCHIVED_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_"
    "candidate_gate_zero_offset_probe_u0400_20260801.py"
)
_V39_IMMUTABLE_ARCHIVED_ROOT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/"
    "dense_duty_adapter_candidate_gate_zero_offset_highmem_20260801/"
    "probe"
)
_V39_IMMUTABLE_ARCHIVED_TERMINAL = {
    "path": _V39_IMMUTABLE_ARCHIVED_ROOT
    / "u000400_fresh/checkpoint_iter.pth",
    "sha256": "202b067beb7cdd71343599872a1ae911b45bbb7f375b168739dab659b611c6c0",
    "optimizer_updates": 400,
    "checkpoint_reason": "max_train_iters",
}
_V39_IMMUTABLE_ARCHIVED_SNAPSHOTS = {
    100: {
        "path": _V39_IMMUTABLE_ARCHIVED_ROOT
        / "intermediate_snapshots/u000100/checkpoint_iter.pth",
        "sha256": "0f38783c015d544a9ff9f6ede47a7d66ac158d791a2f6a7f70cef2896fb3b584",
        "checkpoint_reason": "interval",
    },
    200: {
        "path": _V39_IMMUTABLE_ARCHIVED_ROOT
        / "intermediate_snapshots/u000200/checkpoint_iter.pth",
        "sha256": "b091b29b9076c933a4190c308ba041f8af0227da3275ef6bba4aca005b8286af",
        "checkpoint_reason": "interval",
    },
    300: {
        "path": _V39_IMMUTABLE_ARCHIVED_ROOT
        / "intermediate_snapshots/u000300/checkpoint_iter.pth",
        "sha256": "82822bc271950f1cdb31f791102b414d0ce8a986974a93ecc728645f6a409ab4",
        "checkpoint_reason": "interval",
    },
}

_V40_IMMUTABLE_ARCHIVED_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_"
    "candidate_hardest_edit_probe_u0400_20260801.py"
)
_V40_IMMUTABLE_ARCHIVED_ROOT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/"
    "dense_duty_adapter_candidate_hardest_edit_highmem_20260801/"
    "probe"
)
_V40_IMMUTABLE_ARCHIVED_TERMINAL = {
    "path": _V40_IMMUTABLE_ARCHIVED_ROOT
    / "u000400_fresh/checkpoint_iter.pth",
    "sha256": "898f2be8545b71dd8ba0ee2d0d2f7a669d3680b95d1952bfe3ccd5f2e905dc00",
    "optimizer_updates": 400,
    "checkpoint_reason": "max_train_iters",
}
_V40_IMMUTABLE_ARCHIVED_SNAPSHOTS = {
    100: {
        "path": _V40_IMMUTABLE_ARCHIVED_ROOT
        / "intermediate_snapshots/u000100/checkpoint_iter.pth",
        "sha256": "a97b63b94cc5391a92a58a202ef3d4239b792fe6694747ad2e72a593be372b88",
        "checkpoint_reason": "interval",
    },
    200: {
        "path": _V40_IMMUTABLE_ARCHIVED_ROOT
        / "intermediate_snapshots/u000200/checkpoint_iter.pth",
        "sha256": "261fee3242a5b93067e8c8f04d1f7d79ef54a0cc6897389a920bc8cf7b378e8f",
        "checkpoint_reason": "interval",
    },
    300: {
        "path": _V40_IMMUTABLE_ARCHIVED_ROOT
        / "intermediate_snapshots/u000300/checkpoint_iter.pth",
        "sha256": "71c96eef83e780f71d39f67502d47b975b3799ae89e080ed23de06f77e700da4",
        "checkpoint_reason": "interval",
    },
}

_V41_IMMUTABLE_ARCHIVED_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_"
    "candidate_role_complete_carrier_probe_u0400_20260801.py"
)
_V41_IMMUTABLE_ARCHIVED_ROOT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/"
    "dense_duty_adapter_candidate_role_complete_carrier_highmem_20260801/"
    "probe"
)
_V41_IMMUTABLE_ARCHIVED_TERMINAL = {
    "path": _V41_IMMUTABLE_ARCHIVED_ROOT
    / "u000400_fresh/checkpoint_iter.pth",
    "sha256": "c6dd000dcb10fbfc1aec5c897b64628130fd9213f7c8cbe13b0725c8283d840a",
    "optimizer_updates": 400,
    "checkpoint_reason": "max_train_iters",
}
_V41_IMMUTABLE_ARCHIVED_SNAPSHOTS = {
    100: {
        "path": _V41_IMMUTABLE_ARCHIVED_ROOT
        / "intermediate_snapshots/u000100/checkpoint_iter.pth",
        "sha256": "b0f80ec833bd65be8f25f9a6e6360025845f5023555a22ebbfcd38d9fe26ad31",
        "checkpoint_reason": "interval",
    },
    200: {
        "path": _V41_IMMUTABLE_ARCHIVED_ROOT
        / "intermediate_snapshots/u000200/checkpoint_iter.pth",
        "sha256": "c3ccdd5f9cdbc4ed250f4067daa7043e69f23cfee1814784e5ad49a475c44fc2",
        "checkpoint_reason": "interval",
    },
    300: {
        "path": _V41_IMMUTABLE_ARCHIVED_ROOT
        / "intermediate_snapshots/u000300/checkpoint_iter.pth",
        "sha256": "cae68b5cab47f6158e4419ef3afcd56e66b6b5bfd31bc39193621f60395d0808",
        "checkpoint_reason": "interval",
    },
}

_V42_IMMUTABLE_ARCHIVED_CONFIG = (
    REPO_ROOT
    / "config/ablations/"
    "cfg_stageb_dense_duty_confidence_adapter_"
    "candidate_tn_only_carrier_pair_probe_u0400_20260801.py"
)
_V42_IMMUTABLE_ARCHIVED_ROOT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/"
    "dense_duty_adapter_candidate_tn_only_carrier_pair_highmem_20260801/"
    "probe"
)
_V42_IMMUTABLE_ARCHIVED_TERMINAL = {
    "path": _V42_IMMUTABLE_ARCHIVED_ROOT
    / "u000400_fresh/checkpoint_iter.pth",
    "sha256": "e730b0298390587b575f4f7bec1e1bd1ee539d05859b3208a416d7dd21774069",
    "optimizer_updates": 400,
    "checkpoint_reason": "max_train_iters",
}
_V42_IMMUTABLE_ARCHIVED_SNAPSHOTS = {
    100: {
        "path": _V42_IMMUTABLE_ARCHIVED_ROOT
        / "intermediate_snapshots/u000100/checkpoint_iter.pth",
        "sha256": "507f42528d1f84fe94bf55bf15f7404302c64e2abd840b9ece5b7207fe1fafed",
        "checkpoint_reason": "interval",
    },
    200: {
        "path": _V42_IMMUTABLE_ARCHIVED_ROOT
        / "intermediate_snapshots/u000200/checkpoint_iter.pth",
        "sha256": "9063727529096a98da50d7892450aadbe9890defb6b64d0de75b91fe53400dba",
        "checkpoint_reason": "interval",
    },
    300: {
        "path": _V42_IMMUTABLE_ARCHIVED_ROOT
        / "intermediate_snapshots/u000300/checkpoint_iter.pth",
        "sha256": "cabe4fb0c040bca5fcfdba68569700464ff9cba191e096f5712fc81c116543e5",
        "checkpoint_reason": "interval",
    },
    400: dict(_V42_IMMUTABLE_ARCHIVED_TERMINAL),
}


def _torch_load_compat(path: str, *, map_location: str = "cpu"):
    try:
        return torch.load(path, map_location=map_location)
    except Exception as e:
        msg = str(e)
        if "Weights only load failed" not in msg and "weights_only" not in msg:
            raise
        return torch.load(path, map_location=map_location, weights_only=False)


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            return ckpt["model"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
    return ckpt


def _validate_native_patch_category_config(cfg) -> bool:
    enabled = bool(getattr(cfg, "stage_b_native_patch_category", False))
    if not enabled:
        return False
    version = getattr(cfg, "stage_b_native_patch_contract_version", None)
    supported_versions = {
        NATIVE_PATCH_CATEGORY_CONTRACT_VERSION,
        NATIVE_PATCH_CATEGORY_D2_CONTRACT_VERSION,
        NATIVE_PATCH_CATEGORY_D3_CONTRACT_VERSION,
        NATIVE_PATCH_CATEGORY_D4_CONTRACT_VERSION,
        NATIVE_PATCH_CATEGORY_D5_CONTRACT_VERSION,
        NATIVE_PATCH_CATEGORY_D6_CONTRACT_VERSION,
        NATIVE_PATCH_CATEGORY_D7_CONTRACT_VERSION,
        NATIVE_PATCH_CATEGORY_D8_CONTRACT_VERSION,
        NATIVE_PATCH_CATEGORY_D9_CONTRACT_VERSION,
    }
    if isinstance(version, bool) or version not in supported_versions:
        raise ValueError(
            "stage_b_native_patch_category evaluation requires contract version "
            f"in {sorted(supported_versions)}"
        )
    if version == NATIVE_PATCH_CATEGORY_D2_CONTRACT_VERSION:
        objective = str(
            getattr(cfg, "stage_b_native_patch_objective", "") or ""
        ).strip().lower()
        if objective != "d2_gate_aligned":
            raise ValueError(
                "stage_b_native_patch_category contract v2 requires "
                "stage_b_native_patch_objective='d2_gate_aligned'"
            )
        d2_gate = getattr(cfg, "stage_b_native_patch_gate_max_gap", None)
        if (
            isinstance(d2_gate, bool)
            or not isinstance(d2_gate, (int, float))
            or not math.isfinite(float(d2_gate))
            or float(d2_gate) != _NATIVE_PATCH_GATE_MAX_GAP
        ):
            raise ValueError(
                "stage_b_native_patch_category contract v2 requires exact "
                "gate max gap 3"
            )
        d2_clip = getattr(cfg, "stage_b_native_patch_score_clip", None)
        if (
            isinstance(d2_clip, bool)
            or not isinstance(d2_clip, (int, float))
            or not math.isfinite(float(d2_clip))
            or float(d2_clip) != _NATIVE_PATCH_GATE_CLIP
        ):
            raise ValueError(
                "stage_b_native_patch_category contract v2 requires exact "
                "patch-score clip 5"
            )
    if version == NATIVE_PATCH_CATEGORY_D3_CONTRACT_VERSION:
        objective = str(
            getattr(cfg, "stage_b_native_patch_objective", "") or ""
        ).strip().lower()
        if objective != "d3_critical_winner":
            raise ValueError(
                "stage_b_native_patch_category contract v3 requires "
                "stage_b_native_patch_objective='d3_critical_winner'"
            )
        d3_gate = getattr(cfg, "stage_b_native_patch_gate_max_gap", None)
        if (
            isinstance(d3_gate, bool)
            or not isinstance(d3_gate, (int, float))
            or not math.isfinite(float(d3_gate))
            or float(d3_gate) != _NATIVE_PATCH_GATE_MAX_GAP
        ):
            raise ValueError(
                "stage_b_native_patch_category contract v3 requires exact "
                "gate max gap 3"
            )
        d3_clip = getattr(cfg, "stage_b_native_patch_score_clip", None)
        if (
            isinstance(d3_clip, bool)
            or not isinstance(d3_clip, (int, float))
            or not math.isfinite(float(d3_clip))
            or float(d3_clip) != _NATIVE_PATCH_GATE_CLIP
        ):
            raise ValueError(
                "stage_b_native_patch_category contract v3 requires exact "
                "patch-score clip 5"
            )
    if version == NATIVE_PATCH_CATEGORY_D4_CONTRACT_VERSION:
        objective = str(
            getattr(cfg, "stage_b_native_patch_objective", "") or ""
        ).strip().lower()
        if objective != "d4_positive_protected_critical_winner":
            raise ValueError(
                "stage_b_native_patch_category contract v4 requires "
                "stage_b_native_patch_objective="
                "'d4_positive_protected_critical_winner'"
            )
        exact_d4_values = {
            "stage_b_native_patch_gate_max_gap": 3.0,
            "stage_b_native_patch_score_clip": 5.0,
            "stage_b_native_patch_d4_critical_weight": 2.0,
            "stage_b_native_patch_d4_critical_keep_weight": 1.0,
            "stage_b_native_patch_d4_positive_keep_weight": 32.0,
        }
        for name, expected in exact_d4_values.items():
            observed = getattr(cfg, name, None)
            if (
                isinstance(observed, bool)
                or not isinstance(observed, (int, float))
                or not math.isfinite(float(observed))
                or float(observed) != expected
            ):
                raise ValueError(
                    "stage_b_native_patch_category contract v4 requires "
                    f"exact {name}={expected:g}"
                )
    if version == NATIVE_PATCH_CATEGORY_D5_CONTRACT_VERSION:
        objective = str(
            getattr(cfg, "stage_b_native_patch_objective", "") or ""
        ).strip().lower()
        if objective != "d5_active_tail_positive_barrier":
            raise ValueError(
                "stage_b_native_patch_category contract v5 requires "
                "stage_b_native_patch_objective="
                "'d5_active_tail_positive_barrier'"
            )
        exact_d5_values = {
            "stage_b_native_patch_gate_max_gap": 3.0,
            "stage_b_native_patch_score_clip": 5.0,
            "stage_b_native_patch_d5_keep_gap": 2.75,
            "stage_b_native_patch_d5_separation_gap": 3.25,
            "stage_b_native_patch_d5_temperature": 0.25,
            "stage_b_native_patch_d5_critical_weight": 2.0,
            "stage_b_native_patch_d5_critical_keep_weight": 1.0,
            "stage_b_native_patch_d5_active_gap": 2.0,
            "stage_b_native_patch_d5_target_gap": 2.5,
            "stage_b_native_patch_d5_positive_barrier_weight": 2.0,
        }
        for name, expected in exact_d5_values.items():
            observed = getattr(cfg, name, None)
            if (
                isinstance(observed, bool)
                or not isinstance(observed, (int, float))
                or not math.isfinite(float(observed))
                or float(observed) != expected
            ):
                raise ValueError(
                    "stage_b_native_patch_category contract v5 requires "
                    f"exact {name}={expected:g}"
                )
    if version == NATIVE_PATCH_CATEGORY_D6_CONTRACT_VERSION:
        objective = str(
            getattr(cfg, "stage_b_native_patch_objective", "") or ""
        ).strip().lower()
        if objective != "d6_direct_deployment_gap":
            raise ValueError(
                "stage_b_native_patch_category contract v6 requires "
                "stage_b_native_patch_objective='d6_direct_deployment_gap'"
            )
        exact_d6_values = {
            "stage_b_native_patch_gate_max_gap": 3.0,
            "stage_b_native_patch_score_clip": 5.0,
            "stage_b_native_patch_positive_iou_threshold": 0.5,
            "stage_b_native_patch_negative_iou_threshold": 0.3,
            "stage_b_native_patch_d6_weight": 1.0,
            "stage_b_native_patch_d6_keep_gap": 2.75,
            "stage_b_native_patch_d6_drop_gap": 3.25,
            "stage_b_native_patch_d6_drop_active_gap": 3.75,
            "stage_b_native_patch_d6_temperature": 0.25,
            "stage_b_native_patch_d6_drop_weight": 2.0,
            "stage_b_native_patch_d6_critical_keep_weight": 1.0,
            "stage_b_native_patch_d6_positive_active_gap": 2.0,
            "stage_b_native_patch_d6_positive_target_gap": 2.5,
            "stage_b_native_patch_d6_positive_barrier_weight": 2.0,
        }
        for name, expected in exact_d6_values.items():
            observed = getattr(cfg, name, None)
            if (
                isinstance(observed, bool)
                or not isinstance(observed, (int, float))
                or not math.isfinite(float(observed))
                or float(observed) != expected
            ):
                raise ValueError(
                    "stage_b_native_patch_category contract v6 requires "
                    f"exact {name}={expected:g}"
                )
    if version == NATIVE_PATCH_CATEGORY_D7_CONTRACT_VERSION:
        objective = str(
            getattr(cfg, "stage_b_native_patch_objective", "") or ""
        ).strip().lower()
        if objective != "d7_all_state_positive_anchor":
            raise ValueError(
                "stage_b_native_patch_category contract v7 requires "
                "stage_b_native_patch_objective='d7_all_state_positive_anchor'"
            )
        exact_d7_values = {
            "stage_b_native_patch_gate_max_gap": 3.0,
            "stage_b_native_patch_score_clip": 5.0,
            "stage_b_native_patch_positive_iou_threshold": 0.5,
            "stage_b_native_patch_negative_iou_threshold": 0.3,
            "stage_b_native_patch_d7_weight": 1.0,
            "stage_b_native_patch_d7_keep_gap": 2.75,
            "stage_b_native_patch_d7_drop_gap": 3.25,
            "stage_b_native_patch_d7_drop_active_gap": 3.75,
            "stage_b_native_patch_d7_temperature": 0.25,
            "stage_b_native_patch_d7_drop_weight": 2.0,
            "stage_b_native_patch_d7_critical_keep_weight": 1.0,
            "stage_b_native_patch_d7_positive_active_gap": 2.0,
            "stage_b_native_patch_d7_positive_target_gap": 2.5,
            "stage_b_native_patch_d7_positive_barrier_weight": 2.0,
            "stage_b_native_patch_d7_anchor_active_gap": 2.0,
            "stage_b_native_patch_d7_anchor_target_gap": 2.5,
            "stage_b_native_patch_d7_anchor_weight": 2.0,
        }
        for name, expected in exact_d7_values.items():
            observed = getattr(cfg, name, None)
            if (
                isinstance(observed, bool)
                or not isinstance(observed, (int, float))
                or not math.isfinite(float(observed))
                or float(observed) != expected
            ):
                raise ValueError(
                    "stage_b_native_patch_category contract v7 requires "
                    f"exact {name}={expected:g}"
                )
    if version == NATIVE_PATCH_CATEGORY_D8_CONTRACT_VERSION:
        objective = str(
            getattr(cfg, "stage_b_native_patch_objective", "") or ""
        ).strip().lower()
        if objective != "d8_state_class_macro_anchor":
            raise ValueError(
                "stage_b_native_patch_category contract v8 requires "
                "stage_b_native_patch_objective='d8_state_class_macro_anchor'"
            )
        exact_d8_values = {
            "stage_b_native_patch_gate_max_gap": 3.0,
            "stage_b_native_patch_score_clip": 5.0,
            "stage_b_native_patch_positive_iou_threshold": 0.5,
            "stage_b_native_patch_negative_iou_threshold": 0.3,
            "stage_b_native_patch_d8_weight": 1.0,
            "stage_b_native_patch_d8_keep_gap": 2.75,
            "stage_b_native_patch_d8_drop_gap": 3.25,
            "stage_b_native_patch_d8_drop_active_gap": 3.75,
            "stage_b_native_patch_d8_temperature": 0.25,
            "stage_b_native_patch_d8_drop_weight": 2.0,
            "stage_b_native_patch_d8_critical_keep_weight": 1.0,
            "stage_b_native_patch_d8_positive_active_gap": 2.0,
            "stage_b_native_patch_d8_positive_target_gap": 2.5,
            "stage_b_native_patch_d8_positive_barrier_weight": 2.0,
            "stage_b_native_patch_d8_anchor_active_gap": 2.0,
            "stage_b_native_patch_d8_anchor_target_gap": 2.5,
            "stage_b_native_patch_d8_anchor_negative_weight": 1.0,
            "stage_b_native_patch_d8_anchor_neutral_weight": 2.0,
            "stage_b_native_patch_d8_anchor_positive_weight": 4.0,
        }
        for name, expected in exact_d8_values.items():
            observed = getattr(cfg, name, None)
            if (
                isinstance(observed, bool)
                or not isinstance(observed, (int, float))
                or not math.isfinite(float(observed))
                or float(observed) != expected
            ):
                raise ValueError(
                    "stage_b_native_patch_category contract v8 requires "
                    f"exact {name}={expected:g}"
                )
    if version == NATIVE_PATCH_CATEGORY_D9_CONTRACT_VERSION:
        objective = str(
            getattr(cfg, "stage_b_native_patch_objective", "") or ""
        ).strip().lower()
        if objective != "d9_loss_gradient_localized":
            raise ValueError(
                "stage_b_native_patch_category contract v9 requires "
                "stage_b_native_patch_objective="
                "'d9_loss_gradient_localized'"
            )
        exact_d9_values = {
            "stage_b_native_patch_gate_max_gap": 3.0,
            "stage_b_native_patch_score_clip": 5.0,
            "stage_b_native_patch_positive_iou_threshold": 0.5,
            "stage_b_native_patch_negative_iou_threshold": 0.3,
            "stage_b_native_patch_d8_weight": 1.0,
            "stage_b_native_patch_d8_keep_gap": 2.75,
            "stage_b_native_patch_d8_drop_gap": 3.25,
            "stage_b_native_patch_d8_drop_active_gap": 3.75,
            "stage_b_native_patch_d8_temperature": 0.25,
            "stage_b_native_patch_d8_drop_weight": 2.0,
            "stage_b_native_patch_d8_critical_keep_weight": 1.0,
            "stage_b_native_patch_d8_positive_active_gap": 2.0,
            "stage_b_native_patch_d8_positive_target_gap": 2.5,
            "stage_b_native_patch_d8_positive_barrier_weight": 2.0,
            "stage_b_native_patch_d8_anchor_active_gap": 2.0,
            "stage_b_native_patch_d8_anchor_target_gap": 2.5,
            "stage_b_native_patch_d8_anchor_negative_weight": 1.0,
            "stage_b_native_patch_d8_anchor_neutral_weight": 2.0,
            "stage_b_native_patch_d8_anchor_positive_weight": 4.0,
        }
        for name, expected in exact_d9_values.items():
            observed = getattr(cfg, name, None)
            if (
                isinstance(observed, bool)
                or not isinstance(observed, (int, float))
                or not math.isfinite(float(observed))
                or float(observed) != expected
            ):
                raise ValueError(
                    "stage_b_native_patch_category contract v9 requires "
                    f"exact {name}={expected:g}"
                )
        if (
            getattr(
                cfg, "stage_b_native_patch_d9_detach_row_stats", None
            )
            is not True
        ):
            raise ValueError(
                "stage_b_native_patch_category contract v9 requires exact "
                "stage_b_native_patch_d9_detach_row_stats=True"
            )
    incompatible = [
        name
        for name in (
            "stage_b_data_driven_score",
            "stage_b_gdino_score_adapter",
            "stage_b_u0_patch_rank",
            "stage_b_v7",
            "stage_b_v11_fixed_text",
            "stage_b_legacy_global_gate",
            "stage_b",
        )
        if bool(getattr(cfg, name, False))
    ]
    if incompatible:
        raise ValueError(
            "stage_b_native_patch_category Ref evaluation forbids other score "
            f"routes: {incompatible}"
        )
    if not bool(getattr(cfg, "enable_patch_branch", False)):
        raise ValueError(
            "stage_b_native_patch_category Ref evaluation requires enable_patch_branch=True"
        )
    if bool(getattr(cfg, "patch_gate_with_text", False)):
        raise ValueError(
            "stage_b_native_patch_category Ref evaluation requires patch_gate_with_text=False"
        )
    return True


def _validate_native_patch_category_ref_request(
    cfg,
    checkpoint_paths: Iterable[str | Path],
    *,
    extra_score_source_requested: bool = False,
) -> bool:
    enabled = _validate_native_patch_category_config(cfg)
    if not enabled:
        return False
    checkpoints = list(checkpoint_paths)
    if len(checkpoints) != 1:
        raise ValueError(
            "stage_b_native_patch_category official Ref evaluation requires "
            "exactly one checkpoint"
        )
    if extra_score_source_requested:
        raise ValueError(
            "stage_b_native_patch_category official Ref evaluation forbids "
            "external checkpoints, score-transfer artifacts, and score sweeps"
        )
    return True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_historical_dense_duty_source_archive(
    *,
    revision: str,
    expected_updates: int,
    registry: Mapping[str, Mapping[str, Any]],
    config_path: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    source_closure: Mapping[str, Any],
) -> Dict[str, Any]:
    spec = registry.get(revision)
    if spec is None:
        raise RuntimeError(
            "historical dense-duty terminal diagnostic revision is not registered"
        )
    expected_config = Path(spec["config"]).resolve(strict=True)
    if config_path != expected_config:
        raise RuntimeError(
            "historical dense-duty terminal diagnostic config path drifted"
        )
    expected_checkpoint = Path(spec["checkpoint"]).resolve(strict=True)
    if (
        checkpoint_path != expected_checkpoint
        or checkpoint_sha256 != spec["checkpoint_sha256"]
    ):
        raise RuntimeError(
            "historical dense-duty terminal diagnostic checkpoint path/SHA256 drifted"
        )
    if source_closure.get("sha256") != spec["source_closure_sha256"]:
        raise RuntimeError(
            "historical dense-duty terminal diagnostic source closure drifted"
        )

    expected_files: Dict[str, Mapping[str, Any]] = {}
    for section_name in ("code", "config"):
        section = source_closure.get(section_name)
        if not isinstance(section, Mapping):
            raise RuntimeError(
                "historical dense-duty terminal source closure lacks code/config"
            )
        for record in section.get("files", ()):
            if not isinstance(record, Mapping):
                raise RuntimeError(
                    "historical dense-duty terminal source closure has an invalid file record"
                )
            path = str(record.get("path", ""))
            pure_path = PurePosixPath(path)
            if (
                not path
                or pure_path.is_absolute()
                or any(part in {"", ".", ".."} for part in pure_path.parts)
                or pure_path.as_posix() != path
                or path in expected_files
            ):
                raise RuntimeError(
                    "historical dense-duty terminal source closure has unsafe/duplicate paths"
                )
            expected_files[path] = record

    archive = Path(spec["archive"]).resolve(strict=True)
    archive_sha256 = _sha256_file(archive)
    if archive_sha256 != spec["archive_sha256"]:
        raise RuntimeError(
            "historical dense-duty terminal exact-source archive SHA256 drifted"
        )
    observed_paths = set()
    try:
        with tarfile.open(archive, mode="r:gz") as handle:
            for member in handle.getmembers():
                path = member.name
                pure_path = PurePosixPath(path)
                if (
                    not member.isfile()
                    or pure_path.is_absolute()
                    or any(part in {"", ".", ".."} for part in pure_path.parts)
                    or pure_path.as_posix() != path
                    or path in observed_paths
                    or path not in expected_files
                ):
                    raise RuntimeError(
                        "historical dense-duty terminal exact-source archive has "
                        "unsafe, duplicate, or unexpected members"
                    )
                stream = handle.extractfile(member)
                if stream is None:
                    raise RuntimeError(
                        "historical dense-duty terminal exact-source archive member "
                        "cannot be read"
                    )
                digest = hashlib.sha256()
                observed_size = 0
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
                    observed_size += len(chunk)
                expected = expected_files[path]
                if (
                    observed_size != int(expected.get("size_bytes", -1))
                    or digest.hexdigest() != expected.get("sha256")
                ):
                    raise RuntimeError(
                        "historical dense-duty terminal exact-source archive member "
                        f"drifted: {path}"
                    )
                observed_paths.add(path)
    except (OSError, tarfile.TarError) as error:
        raise RuntimeError(
            "historical dense-duty terminal exact-source archive is unreadable"
        ) from error
    if observed_paths != set(expected_files):
        raise RuntimeError(
            "historical dense-duty terminal exact-source archive is incomplete"
        )
    return {
        "schema": "pivot.stageb.historical_terminal_source_archive/v1",
        "revision": revision,
        "optimizer_updates": expected_updates,
        "config": str(expected_config),
        "checkpoint": str(expected_checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "archive": str(archive),
        "archive_sha256": archive_sha256,
        "source_closure_sha256": source_closure["sha256"],
        "file_count": len(observed_paths),
        "status": "verified",
    }


def _validate_historical_dense_duty_u50_source_archive(
    *,
    revision: str,
    config_path: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    source_closure: Mapping[str, Any],
) -> Dict[str, Any]:
    return _validate_historical_dense_duty_source_archive(
        revision=revision,
        expected_updates=50,
        registry=_HISTORICAL_DENSE_DUTY_U50_SOURCES,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        source_closure=source_closure,
    )


def _validate_historical_dense_duty_u100_source_archive(
    *,
    revision: str,
    config_path: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    source_closure: Mapping[str, Any],
) -> Dict[str, Any]:
    return _validate_historical_dense_duty_source_archive(
        revision=revision,
        expected_updates=100,
        registry=_HISTORICAL_DENSE_DUTY_U100_SOURCES,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
        source_closure=source_closure,
    )


def _validate_dense_duty_partial_rank_diagnostic_checkpoint(
    payload: Mapping[str, Any],
    cfg,
    *,
    checkpoint_path: Path,
) -> Dict[str, Any]:
    """Admit an immutable rank snapshot for Ref-only diagnostics."""
    from util.stage_b_dense_duty_audit import (
        SOURCE_CLOSURE_ARG,
        audit_checkpoint_payload,
        build_code_source_closure,
        validate_code_source_closure,
        validate_source_closure,
        validate_strict_resume_checkpoint_payload,
    )

    resolved = checkpoint_path.expanduser().resolve(strict=True)
    saved_args = payload.get("args")
    if not isinstance(saved_args, Mapping):
        raise RuntimeError("dense-duty rank diagnostic lacks saved args")
    output_dir = saved_args.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise RuntimeError("dense-duty rank diagnostic lacks its training output_dir")
    canonical = Path(output_dir).expanduser().resolve(strict=True) / "checkpoint_iter.pth"
    canonical = canonical.resolve(strict=True)
    snapshot_sha256 = _sha256_file(resolved)
    canonical_sha256 = _sha256_file(canonical)
    if snapshot_sha256 != canonical_sha256:
        raise RuntimeError(
            "dense-duty rank diagnostic snapshot differs from the canonical "
            "strict-resume checkpoint"
        )

    resume = validate_strict_resume_checkpoint_payload(
        payload,
        saved_args,
        checkpoint_path=canonical,
    )
    audit = audit_checkpoint_payload(payload, checkpoint_path=resolved)
    expected_updates = int(
        getattr(cfg, "stage_b_dense_duty_rank_expected_optimizer_updates", 0)
    )
    observed_updates = payload.get("optimizer_updates")
    if (
        audit.get("status") != "passed"
        or audit.get("phase") != "rank"
        or resume.get("phase") != "rank"
        or isinstance(observed_updates, bool)
        or not isinstance(observed_updates, int)
        or observed_updates <= 0
        or expected_updates <= 0
        or observed_updates >= expected_updates
        or payload.get("checkpoint_reason") not in {"signal", "interval"}
        or resume.get("optimizer_updates") != observed_updates
        or resume.get("checkpoint_reason") != payload.get("checkpoint_reason")
    ):
        raise RuntimeError(
            "dense-duty partial rank diagnostic requires a valid non-terminal "
            "rank optimizer-boundary checkpoint"
        )

    evaluation_scope = str(
        getattr(cfg, "stage_b_dense_duty_evaluation_scope", "formal") or ""
    ).strip().lower()
    if evaluation_scope not in {"formal", "probe"} or saved_args.get(
        "stage_b_dense_duty_execution_scope"
    ) != evaluation_scope:
        raise RuntimeError(
            "dense-duty rank diagnostic execution/evaluation scope mismatch"
        )
    if evaluation_scope == "formal":
        current_code = validate_code_source_closure(build_code_source_closure())
        saved_source = validate_source_closure(saved_args.get(SOURCE_CLOSURE_ARG))
        if saved_source["code"]["sha256"] != current_code["sha256"]:
            raise RuntimeError(
                "dense-duty rank diagnostic training code source closure drifted"
            )
        audit["source_closure"] = saved_source

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
    drift = {
        key: (saved_args.get(key), getattr(cfg, key, None))
        for key in required_equal_args
        if saved_args.get(key) != getattr(cfg, key, None)
    }
    runtime = saved_args.get("stage_b_dense_duty_runtime_audit")
    lineage = audit.get("lineage")
    if drift:
        raise RuntimeError(
            f"dense-duty rank diagnostic configuration drifted from training: {drift}"
        )
    if (
        saved_args.get("stage_b_dense_duty_no_stageb_teacher") is not True
        or saved_args.get("stage_b_v22_score_ownership")
        != "independent_decoders_two_phase"
        or not isinstance(lineage, Mapping)
        or lineage.get("no_stage_b_teacher") is not True
        or lineage.get("execution_scope") != evaluation_scope
        or not isinstance(runtime, Mapping)
        or runtime.get("successful_optimizer_steps") != observed_updates
        or runtime.get("optimizer_step_boundaries") != observed_updates
        or int(runtime.get("amp_skipped_optimizer_steps", -1)) != 0
        or int(runtime.get("nonfinite_gradient_boundaries", -1)) != 0
        or int(runtime.get("zero_gradient_successful_steps", -1)) != 0
    ):
        raise RuntimeError(
            "dense-duty rank diagnostic lacks valid no-teacher lineage/runtime evidence"
        )

    audit.update(
        {
            "checkpoint_reason": payload["checkpoint_reason"],
            "evaluation_scope": evaluation_scope,
            "diagnostic_only": True,
            "formal_gate_eligible": False,
            "confidence_evaluated": False,
            "training_phase": "rank",
            "canonical_checkpoint": str(canonical),
            "canonical_checkpoint_sha256": canonical_sha256,
            "evaluation_checkpoint_sha256": snapshot_sha256,
            "strict_resume": resume,
        }
    )
    return audit


def _v39_immutable_archived_file_record(path: Path) -> Dict[str, Any]:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise RuntimeError(
            "v39 immutable archived diagnostic files must not be symlinks"
        )
    resolved = candidate.resolve(strict=True)
    return {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": _sha256_file(resolved),
    }


def _v39_immutable_archived_snapshot_spec(
    checkpoint_path: Path,
) -> Tuple[int, Dict[str, Any]]:
    resolved = Path(checkpoint_path).expanduser().resolve(strict=True)
    for optimizer_updates, raw_spec in _V39_IMMUTABLE_ARCHIVED_SNAPSHOTS.items():
        spec = dict(raw_spec)
        if resolved == Path(spec["path"]).resolve(strict=True):
            spec["optimizer_updates"] = optimizer_updates
            return optimizer_updates, spec
    raise RuntimeError(
        "v39 immutable archived diagnostic requires the exact U100, U200, or "
        "U300 checkpoint path"
    )


def _validate_v39_immutable_archived_snapshot_metadata(
    payload: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    optimizer_updates: int,
    checkpoint_reason: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise RuntimeError("v39 immutable archived snapshot payload is invalid")
    saved_args = payload.get("args")
    if not isinstance(saved_args, Mapping):
        raise RuntimeError("v39 immutable archived snapshot lacks saved args")
    expected_output_dir = Path(
        _V39_IMMUTABLE_ARCHIVED_TERMINAL["path"]
    ).parent.resolve(strict=True)
    try:
        saved_output_dir = Path(str(saved_args.get("output_dir", ""))).expanduser()
        saved_output_dir = saved_output_dir.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        saved_output_dir = None
    if (
        payload.get("optimizer_updates") != optimizer_updates
        or payload.get("checkpoint_reason") != checkpoint_reason
        or saved_args.get("max_train_iters") != 400
        or saved_args.get(
            "stage_b_dense_duty_confidence_expected_optimizer_updates"
        )
        != 400
        or saved_args.get("stage_b_dense_duty_execution_scope") != "probe"
        or saved_args.get("stage_b_dense_duty_evaluation_scope") != "probe"
        or saved_args.get("stage_b_dense_duty_phase") != "confidence"
        or saved_args.get("stage_b_dense_duty_confidence_revision")
        != "word_veto_candidate_asymmetric_confidence_v32"
        or saved_output_dir != expected_output_dir
        or checkpoint_path.parent.name != f"u{optimizer_updates:06d}"
        or checkpoint_path.parent.parent.name != "intermediate_snapshots"
        or checkpoint_path.name != "checkpoint_iter.pth"
    ):
        raise RuntimeError(
            "v39 immutable archived snapshot directory/update/reason metadata "
            "does not match its fixed contract"
        )


def _validate_v39_immutable_archived_terminal_metadata(
    payload: Mapping[str, Any], *, checkpoint_path: Path
) -> None:
    if not isinstance(payload, Mapping):
        raise RuntimeError("v39 immutable archived terminal payload is invalid")
    saved_args = payload.get("args")
    if not isinstance(saved_args, Mapping):
        raise RuntimeError("v39 immutable archived terminal lacks saved args")
    expected_path = Path(_V39_IMMUTABLE_ARCHIVED_TERMINAL["path"]).resolve(
        strict=True
    )
    try:
        saved_output_dir = Path(str(saved_args.get("output_dir", ""))).expanduser()
        saved_output_dir = saved_output_dir.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        saved_output_dir = None
    if (
        checkpoint_path != expected_path
        or payload.get("optimizer_updates") != 400
        or payload.get("checkpoint_reason") != "max_train_iters"
        or saved_args.get("max_train_iters") != 400
        or saved_args.get(
            "stage_b_dense_duty_confidence_expected_optimizer_updates"
        )
        != 400
        or saved_args.get("stage_b_dense_duty_execution_scope") != "probe"
        or saved_args.get("stage_b_dense_duty_evaluation_scope") != "probe"
        or saved_args.get("stage_b_dense_duty_phase") != "confidence"
        or saved_output_dir != expected_path.parent
    ):
        raise RuntimeError(
            "v39 immutable archived diagnostic requires the exact terminal U400 "
            "max_train_iters checkpoint"
        )


def _torch_load_v39_immutable_metadata(path: Path) -> Mapping[str, Any]:
    try:
        return torch.load(
            str(path),
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except TypeError:
        return _torch_load_compat(str(path), map_location="cpu")


def _prepare_v39_immutable_archived_diagnostic(
    payload: Mapping[str, Any], cfg, *, checkpoint_path: Path
) -> Dict[str, Any]:
    config_path = Path(str(getattr(cfg, "_filename", ""))).expanduser()
    if config_path.resolve(strict=True) != _V39_IMMUTABLE_ARCHIVED_CONFIG.resolve(
        strict=True
    ):
        raise RuntimeError(
            "v39 immutable archived diagnostic requires its fixed probe config"
        )
    resolved = Path(checkpoint_path).expanduser().resolve(strict=True)
    optimizer_updates, snapshot_spec = _v39_immutable_archived_snapshot_spec(
        resolved
    )
    snapshot_before = _v39_immutable_archived_file_record(resolved)
    if snapshot_before["sha256"] != snapshot_spec["sha256"]:
        raise RuntimeError("v39 immutable archived snapshot SHA256 mismatch")
    _validate_v39_immutable_archived_snapshot_metadata(
        payload,
        checkpoint_path=resolved,
        optimizer_updates=optimizer_updates,
        checkpoint_reason=str(snapshot_spec["checkpoint_reason"]),
    )

    terminal_path = Path(_V39_IMMUTABLE_ARCHIVED_TERMINAL["path"]).resolve(
        strict=True
    )
    terminal_before = _v39_immutable_archived_file_record(terminal_path)
    if terminal_before["sha256"] != _V39_IMMUTABLE_ARCHIVED_TERMINAL["sha256"]:
        raise RuntimeError("v39 immutable archived terminal U400 SHA256 mismatch")
    terminal_payload = _torch_load_v39_immutable_metadata(terminal_path)
    try:
        _validate_v39_immutable_archived_terminal_metadata(
            terminal_payload,
            checkpoint_path=terminal_path,
        )
    finally:
        del terminal_payload

    return {
        "schema": "pivot.stageb.v39_immutable_archived_diagnostic/v1",
        "optimizer_updates": optimizer_updates,
        "checkpoint_reason": str(snapshot_spec["checkpoint_reason"]),
        "snapshot_before_validation": snapshot_before,
        "terminal_before_validation": terminal_before,
    }


def _verify_v39_immutable_archived_diagnostic_files(
    checkpoint_path: Path,
) -> Dict[str, Any]:
    resolved = Path(checkpoint_path).expanduser().resolve(strict=True)
    optimizer_updates, snapshot_spec = _v39_immutable_archived_snapshot_spec(
        resolved
    )
    snapshot = _v39_immutable_archived_file_record(resolved)
    terminal_path = Path(_V39_IMMUTABLE_ARCHIVED_TERMINAL["path"]).resolve(
        strict=True
    )
    terminal = _v39_immutable_archived_file_record(terminal_path)
    if snapshot["sha256"] != snapshot_spec["sha256"]:
        raise RuntimeError("v39 immutable archived snapshot changed during evaluation")
    if terminal["sha256"] != _V39_IMMUTABLE_ARCHIVED_TERMINAL["sha256"]:
        raise RuntimeError("v39 immutable archived terminal changed during evaluation")
    return {
        "optimizer_updates": optimizer_updates,
        "snapshot": snapshot,
        "terminal": terminal,
    }


def _v40_immutable_archived_file_record(path: Path) -> Dict[str, Any]:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise RuntimeError(
            "v40 immutable archived diagnostic files must not be symlinks"
        )
    resolved = candidate.resolve(strict=True)
    return {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": _sha256_file(resolved),
    }


def _v40_immutable_archived_snapshot_spec(
    checkpoint_path: Path,
) -> Tuple[int, Dict[str, Any]]:
    resolved = Path(checkpoint_path).expanduser().resolve(strict=True)
    for optimizer_updates, raw_spec in _V40_IMMUTABLE_ARCHIVED_SNAPSHOTS.items():
        spec = dict(raw_spec)
        if resolved == Path(spec["path"]).resolve(strict=True):
            spec["optimizer_updates"] = optimizer_updates
            return optimizer_updates, spec
    raise RuntimeError(
        "v40 immutable archived diagnostic requires the exact U100, U200, or "
        "U300 checkpoint path"
    )


def _validate_v40_immutable_archived_snapshot_metadata(
    payload: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    optimizer_updates: int,
    checkpoint_reason: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise RuntimeError("v40 immutable archived snapshot payload is invalid")
    saved_args = payload.get("args")
    if not isinstance(saved_args, Mapping):
        raise RuntimeError("v40 immutable archived snapshot lacks saved args")
    expected_output_dir = Path(
        _V40_IMMUTABLE_ARCHIVED_TERMINAL["path"]
    ).parent.resolve(strict=True)
    try:
        saved_output_dir = Path(str(saved_args.get("output_dir", ""))).expanduser()
        saved_output_dir = saved_output_dir.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        saved_output_dir = None
    if (
        payload.get("optimizer_updates") != optimizer_updates
        or payload.get("checkpoint_reason") != checkpoint_reason
        or saved_args.get("max_train_iters") != 400
        or saved_args.get(
            "stage_b_dense_duty_confidence_expected_optimizer_updates"
        )
        != 400
        or saved_args.get("stage_b_dense_duty_execution_scope") != "probe"
        or saved_args.get("stage_b_dense_duty_evaluation_scope") != "probe"
        or saved_args.get("stage_b_dense_duty_phase") != "confidence"
        or saved_args.get("stage_b_dense_duty_confidence_revision")
        != "word_veto_candidate_asymmetric_confidence_v32"
        or saved_args.get("stage_b_v21_token_edit_query_scope")
        != "target_iou_union_detached_final_confidence_base_argmax_v2"
        or saved_output_dir != expected_output_dir
        or checkpoint_path.parent.name != f"u{optimizer_updates:06d}"
        or checkpoint_path.parent.parent.name != "intermediate_snapshots"
        or checkpoint_path.name != "checkpoint_iter.pth"
    ):
        raise RuntimeError(
            "v40 immutable archived snapshot directory/update/reason metadata "
            "does not match its fixed contract"
        )


def _validate_v40_immutable_archived_terminal_metadata(
    payload: Mapping[str, Any], *, checkpoint_path: Path
) -> None:
    if not isinstance(payload, Mapping):
        raise RuntimeError("v40 immutable archived terminal payload is invalid")
    saved_args = payload.get("args")
    if not isinstance(saved_args, Mapping):
        raise RuntimeError("v40 immutable archived terminal lacks saved args")
    expected_path = Path(_V40_IMMUTABLE_ARCHIVED_TERMINAL["path"]).resolve(
        strict=True
    )
    try:
        saved_output_dir = Path(str(saved_args.get("output_dir", ""))).expanduser()
        saved_output_dir = saved_output_dir.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        saved_output_dir = None
    if (
        checkpoint_path != expected_path
        or payload.get("optimizer_updates") != 400
        or payload.get("checkpoint_reason") != "max_train_iters"
        or saved_args.get("max_train_iters") != 400
        or saved_args.get(
            "stage_b_dense_duty_confidence_expected_optimizer_updates"
        )
        != 400
        or saved_args.get("stage_b_dense_duty_execution_scope") != "probe"
        or saved_args.get("stage_b_dense_duty_evaluation_scope") != "probe"
        or saved_args.get("stage_b_dense_duty_phase") != "confidence"
        or saved_args.get("stage_b_dense_duty_confidence_revision")
        != "word_veto_candidate_asymmetric_confidence_v32"
        or saved_args.get("stage_b_v21_token_edit_query_scope")
        != "target_iou_union_detached_final_confidence_base_argmax_v2"
        or saved_output_dir != expected_path.parent
    ):
        raise RuntimeError(
            "v40 immutable archived diagnostic requires the exact terminal U400 "
            "max_train_iters checkpoint"
        )


def _torch_load_v40_immutable_metadata(path: Path) -> Mapping[str, Any]:
    try:
        return torch.load(
            str(path),
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except TypeError:
        return _torch_load_compat(str(path), map_location="cpu")


def _prepare_v40_immutable_archived_diagnostic(
    payload: Mapping[str, Any], cfg, *, checkpoint_path: Path
) -> Dict[str, Any]:
    config_path = Path(str(getattr(cfg, "_filename", ""))).expanduser()
    if config_path.resolve(strict=True) != _V40_IMMUTABLE_ARCHIVED_CONFIG.resolve(
        strict=True
    ):
        raise RuntimeError(
            "v40 immutable archived diagnostic requires its fixed probe config"
        )
    resolved = Path(checkpoint_path).expanduser().resolve(strict=True)
    optimizer_updates, snapshot_spec = _v40_immutable_archived_snapshot_spec(
        resolved
    )
    snapshot_before = _v40_immutable_archived_file_record(resolved)
    if snapshot_before["sha256"] != snapshot_spec["sha256"]:
        raise RuntimeError("v40 immutable archived snapshot SHA256 mismatch")
    _validate_v40_immutable_archived_snapshot_metadata(
        payload,
        checkpoint_path=resolved,
        optimizer_updates=optimizer_updates,
        checkpoint_reason=str(snapshot_spec["checkpoint_reason"]),
    )

    terminal_path = Path(_V40_IMMUTABLE_ARCHIVED_TERMINAL["path"]).resolve(
        strict=True
    )
    terminal_before = _v40_immutable_archived_file_record(terminal_path)
    if terminal_before["sha256"] != _V40_IMMUTABLE_ARCHIVED_TERMINAL["sha256"]:
        raise RuntimeError("v40 immutable archived terminal U400 SHA256 mismatch")
    terminal_payload = _torch_load_v40_immutable_metadata(terminal_path)
    try:
        _validate_v40_immutable_archived_terminal_metadata(
            terminal_payload,
            checkpoint_path=terminal_path,
        )
    finally:
        del terminal_payload

    return {
        "schema": "pivot.stageb.v40_immutable_archived_diagnostic/v1",
        "optimizer_updates": optimizer_updates,
        "checkpoint_reason": str(snapshot_spec["checkpoint_reason"]),
        "snapshot_before_validation": snapshot_before,
        "terminal_before_validation": terminal_before,
    }


def _verify_v40_immutable_archived_diagnostic_files(
    checkpoint_path: Path,
) -> Dict[str, Any]:
    resolved = Path(checkpoint_path).expanduser().resolve(strict=True)
    optimizer_updates, snapshot_spec = _v40_immutable_archived_snapshot_spec(
        resolved
    )
    snapshot = _v40_immutable_archived_file_record(resolved)
    terminal_path = Path(_V40_IMMUTABLE_ARCHIVED_TERMINAL["path"]).resolve(
        strict=True
    )
    terminal = _v40_immutable_archived_file_record(terminal_path)
    if snapshot["sha256"] != snapshot_spec["sha256"]:
        raise RuntimeError("v40 immutable archived snapshot changed during evaluation")
    if terminal["sha256"] != _V40_IMMUTABLE_ARCHIVED_TERMINAL["sha256"]:
        raise RuntimeError("v40 immutable archived terminal changed during evaluation")
    return {
        "optimizer_updates": optimizer_updates,
        "snapshot": snapshot,
        "terminal": terminal,
    }


def _v41_immutable_archived_file_record(path: Path) -> Dict[str, Any]:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise RuntimeError(
            "v41 immutable archived diagnostic files must not be symlinks"
        )
    resolved = candidate.resolve(strict=True)
    return {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": _sha256_file(resolved),
    }


def _v41_immutable_archived_snapshot_spec(
    checkpoint_path: Path,
) -> Tuple[int, Dict[str, Any]]:
    resolved = Path(checkpoint_path).expanduser().resolve(strict=True)
    for optimizer_updates, raw_spec in _V41_IMMUTABLE_ARCHIVED_SNAPSHOTS.items():
        spec = dict(raw_spec)
        if resolved == Path(spec["path"]).resolve(strict=True):
            spec["optimizer_updates"] = optimizer_updates
            return optimizer_updates, spec
    raise RuntimeError(
        "v41 immutable archived diagnostic requires the exact U100, U200, or "
        "U300 checkpoint path"
    )


def _validate_v41_immutable_archived_snapshot_metadata(
    payload: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    optimizer_updates: int,
    checkpoint_reason: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise RuntimeError("v41 immutable archived snapshot payload is invalid")
    saved_args = payload.get("args")
    if not isinstance(saved_args, Mapping):
        raise RuntimeError("v41 immutable archived snapshot lacks saved args")
    expected_output_dir = Path(
        _V41_IMMUTABLE_ARCHIVED_TERMINAL["path"]
    ).parent.resolve(strict=True)
    try:
        saved_output_dir = Path(str(saved_args.get("output_dir", ""))).expanduser()
        saved_output_dir = saved_output_dir.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        saved_output_dir = None
    if (
        payload.get("optimizer_updates") != optimizer_updates
        or payload.get("checkpoint_reason") != checkpoint_reason
        or saved_args.get("max_train_iters") != 400
        or saved_args.get(
            "stage_b_dense_duty_confidence_expected_optimizer_updates"
        )
        != 400
        or saved_args.get("stage_b_dense_duty_execution_scope") != "probe"
        or saved_args.get("stage_b_dense_duty_evaluation_scope") != "probe"
        or saved_args.get("stage_b_dense_duty_phase") != "confidence"
        or saved_args.get("stage_b_dense_duty_confidence_revision")
        != "word_veto_candidate_asymmetric_confidence_v32"
        or saved_args.get("stage_b_v21_token_edit_query_scope")
        != "target_iou_union_detached_role_complete_confidence_base_argmax_v3"
        or saved_output_dir != expected_output_dir
        or checkpoint_path.parent.name != f"u{optimizer_updates:06d}"
        or checkpoint_path.parent.parent.name != "intermediate_snapshots"
        or checkpoint_path.name != "checkpoint_iter.pth"
    ):
        raise RuntimeError(
            "v41 immutable archived snapshot directory/update/reason metadata "
            "does not match its fixed contract"
        )


def _validate_v41_immutable_archived_terminal_metadata(
    payload: Mapping[str, Any], *, checkpoint_path: Path
) -> None:
    if not isinstance(payload, Mapping):
        raise RuntimeError("v41 immutable archived terminal payload is invalid")
    saved_args = payload.get("args")
    if not isinstance(saved_args, Mapping):
        raise RuntimeError("v41 immutable archived terminal lacks saved args")
    expected_path = Path(_V41_IMMUTABLE_ARCHIVED_TERMINAL["path"]).resolve(
        strict=True
    )
    try:
        saved_output_dir = Path(str(saved_args.get("output_dir", ""))).expanduser()
        saved_output_dir = saved_output_dir.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        saved_output_dir = None
    if (
        checkpoint_path != expected_path
        or payload.get("optimizer_updates") != 400
        or payload.get("checkpoint_reason") != "max_train_iters"
        or saved_args.get("max_train_iters") != 400
        or saved_args.get(
            "stage_b_dense_duty_confidence_expected_optimizer_updates"
        )
        != 400
        or saved_args.get("stage_b_dense_duty_execution_scope") != "probe"
        or saved_args.get("stage_b_dense_duty_evaluation_scope") != "probe"
        or saved_args.get("stage_b_dense_duty_phase") != "confidence"
        or saved_args.get("stage_b_dense_duty_confidence_revision")
        != "word_veto_candidate_asymmetric_confidence_v32"
        or saved_args.get("stage_b_v21_token_edit_query_scope")
        != "target_iou_union_detached_role_complete_confidence_base_argmax_v3"
        or saved_output_dir != expected_path.parent
    ):
        raise RuntimeError(
            "v41 immutable archived diagnostic requires the exact terminal U400 "
            "max_train_iters checkpoint"
        )


def _torch_load_v41_immutable_metadata(path: Path) -> Mapping[str, Any]:
    try:
        return torch.load(
            str(path),
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except TypeError:
        return _torch_load_compat(str(path), map_location="cpu")


def _prepare_v41_immutable_archived_diagnostic(
    payload: Mapping[str, Any], cfg, *, checkpoint_path: Path
) -> Dict[str, Any]:
    config_path = Path(str(getattr(cfg, "_filename", ""))).expanduser()
    if config_path.resolve(strict=True) != _V41_IMMUTABLE_ARCHIVED_CONFIG.resolve(
        strict=True
    ):
        raise RuntimeError(
            "v41 immutable archived diagnostic requires its fixed probe config"
        )
    resolved = Path(checkpoint_path).expanduser().resolve(strict=True)
    optimizer_updates, snapshot_spec = _v41_immutable_archived_snapshot_spec(
        resolved
    )
    snapshot_before = _v41_immutable_archived_file_record(resolved)
    if snapshot_before["sha256"] != snapshot_spec["sha256"]:
        raise RuntimeError("v41 immutable archived snapshot SHA256 mismatch")
    _validate_v41_immutable_archived_snapshot_metadata(
        payload,
        checkpoint_path=resolved,
        optimizer_updates=optimizer_updates,
        checkpoint_reason=str(snapshot_spec["checkpoint_reason"]),
    )

    terminal_path = Path(_V41_IMMUTABLE_ARCHIVED_TERMINAL["path"]).resolve(
        strict=True
    )
    terminal_before = _v41_immutable_archived_file_record(terminal_path)
    if terminal_before["sha256"] != _V41_IMMUTABLE_ARCHIVED_TERMINAL["sha256"]:
        raise RuntimeError("v41 immutable archived terminal U400 SHA256 mismatch")
    terminal_payload = _torch_load_v41_immutable_metadata(terminal_path)
    try:
        _validate_v41_immutable_archived_terminal_metadata(
            terminal_payload,
            checkpoint_path=terminal_path,
        )
    finally:
        del terminal_payload

    return {
        "schema": "pivot.stageb.v41_immutable_archived_diagnostic/v1",
        "optimizer_updates": optimizer_updates,
        "checkpoint_reason": str(snapshot_spec["checkpoint_reason"]),
        "snapshot_before_validation": snapshot_before,
        "terminal_before_validation": terminal_before,
    }


def _verify_v41_immutable_archived_diagnostic_files(
    checkpoint_path: Path,
) -> Dict[str, Any]:
    resolved = Path(checkpoint_path).expanduser().resolve(strict=True)
    optimizer_updates, snapshot_spec = _v41_immutable_archived_snapshot_spec(
        resolved
    )
    snapshot = _v41_immutable_archived_file_record(resolved)
    terminal_path = Path(_V41_IMMUTABLE_ARCHIVED_TERMINAL["path"]).resolve(
        strict=True
    )
    terminal = _v41_immutable_archived_file_record(terminal_path)
    if snapshot["sha256"] != snapshot_spec["sha256"]:
        raise RuntimeError("v41 immutable archived snapshot changed during evaluation")
    if terminal["sha256"] != _V41_IMMUTABLE_ARCHIVED_TERMINAL["sha256"]:
        raise RuntimeError("v41 immutable archived terminal changed during evaluation")
    return {
        "optimizer_updates": optimizer_updates,
        "snapshot": snapshot,
        "terminal": terminal,
    }


def _v42_immutable_archived_file_record(path: Path) -> Dict[str, Any]:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise RuntimeError(
            "v42 immutable archived diagnostic files must not be symlinks"
        )
    resolved = candidate.resolve(strict=True)
    return {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": _sha256_file(resolved),
    }


def _v42_immutable_archived_snapshot_spec(
    checkpoint_path: Path,
) -> Tuple[int, Dict[str, Any]]:
    resolved = Path(checkpoint_path).expanduser().resolve(strict=True)
    for optimizer_updates, raw_spec in _V42_IMMUTABLE_ARCHIVED_SNAPSHOTS.items():
        spec = dict(raw_spec)
        if resolved == Path(spec["path"]).resolve(strict=True):
            spec["optimizer_updates"] = optimizer_updates
            return optimizer_updates, spec
    raise RuntimeError(
        "v42 immutable archived diagnostic requires the exact U100, U200, "
        "U300, or U400 checkpoint path"
    )


def _validate_v42_immutable_archived_metadata(
    payload: Mapping[str, Any],
    *,
    checkpoint_path: Path,
    optimizer_updates: int,
    checkpoint_reason: str,
) -> None:
    if not isinstance(payload, Mapping):
        raise RuntimeError("v42 immutable archived checkpoint payload is invalid")
    saved_args = payload.get("args")
    if not isinstance(saved_args, Mapping):
        raise RuntimeError("v42 immutable archived checkpoint lacks saved args")
    contract = saved_args.get("stage_b_dense_duty_training_contract")
    if (
        not isinstance(contract, Mapping)
        or contract.get("schema")
        != "pivot.stageb.dense_duty_training_contract/v24"
        or not isinstance(contract.get("values"), Mapping)
    ):
        raise RuntimeError("v42 immutable archived checkpoint lacks its v24 contract")
    contract_values = contract["values"]
    expected_contract_values = {
        "stage_b_dense_duty_confidence_carrier_selector_contract": (
            "final_layer_reference_argmax_exact_eligible_v1"
        ),
        "stage_b_dense_duty_raw_veto_tn_carrier_balance": 0.25,
        "stage_b_dense_duty_raw_veto_carrier_pair_weight": 0.25,
        "stage_b_dense_duty_raw_veto_carrier_pair_margin": 0.25,
        "stage_b_dense_duty_raw_veto_tail_quantile": 0.95,
        "stage_b_dense_duty_raw_veto_tail_temperature": 0.1,
        "stage_b_dense_duty_raw_veto_tail_min_count": 256,
        "stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract": (
            "tn_only_positive_detached_v2"
        ),
    }
    if any(
        contract_values.get(name) != expected
        or saved_args.get(name) != expected
        for name, expected in expected_contract_values.items()
    ):
        raise RuntimeError("v42 immutable archived checkpoint contract drifted")
    expected_output_dir = Path(
        _V42_IMMUTABLE_ARCHIVED_TERMINAL["path"]
    ).parent.resolve(strict=True)
    try:
        saved_output_dir = Path(str(saved_args.get("output_dir", ""))).expanduser()
        saved_output_dir = saved_output_dir.resolve(strict=True)
    except (OSError, RuntimeError, ValueError):
        saved_output_dir = None
    if (
        payload.get("optimizer_updates") != optimizer_updates
        or payload.get("checkpoint_reason") != checkpoint_reason
        or saved_args.get("max_train_iters") != 400
        or saved_args.get(
            "stage_b_dense_duty_confidence_expected_optimizer_updates"
        )
        != 400
        or saved_args.get("stage_b_dense_duty_execution_scope") != "probe"
        or saved_args.get("stage_b_dense_duty_evaluation_scope") != "probe"
        or saved_args.get("stage_b_dense_duty_phase") != "confidence"
        or saved_args.get("stage_b_dense_duty_confidence_revision")
        != "word_veto_candidate_asymmetric_confidence_v32"
        or saved_args.get(
            "stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract"
        )
        != "tn_only_positive_detached_v2"
        or saved_output_dir != expected_output_dir
        or checkpoint_path.name != "checkpoint_iter.pth"
    ):
        raise RuntimeError(
            "v42 immutable archived checkpoint metadata does not match its fixed "
            "TN-only carrier-pair contract"
        )
    if optimizer_updates == 400:
        if (
            checkpoint_path
            != Path(_V42_IMMUTABLE_ARCHIVED_TERMINAL["path"]).resolve(strict=True)
            or payload.get("epoch") != 1
            or payload.get("iteration") != 356
            or payload.get("epoch_finished") is not False
        ):
            raise RuntimeError(
                "v42 immutable archived terminal must be the exact epoch-1, "
                "iteration-356 U400 checkpoint"
            )
    elif (
        checkpoint_path.parent.name != f"u{optimizer_updates:06d}"
        or checkpoint_path.parent.parent.name != "intermediate_snapshots"
    ):
        raise RuntimeError(
            "v42 immutable archived partial snapshot directory does not match "
            "its optimizer update"
        )


def _torch_load_v42_immutable_metadata(path: Path) -> Mapping[str, Any]:
    try:
        return torch.load(
            str(path),
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except TypeError:
        return _torch_load_compat(str(path), map_location="cpu")


def _prepare_v42_immutable_archived_diagnostic(
    payload: Mapping[str, Any], cfg, *, checkpoint_path: Path
) -> Dict[str, Any]:
    config_path = Path(str(getattr(cfg, "_filename", ""))).expanduser()
    if config_path.resolve(strict=True) != _V42_IMMUTABLE_ARCHIVED_CONFIG.resolve(
        strict=True
    ):
        raise RuntimeError(
            "v42 immutable archived diagnostic requires its fixed probe config"
        )
    resolved = Path(checkpoint_path).expanduser().resolve(strict=True)
    optimizer_updates, snapshot_spec = _v42_immutable_archived_snapshot_spec(
        resolved
    )
    snapshot_before = _v42_immutable_archived_file_record(resolved)
    if snapshot_before["sha256"] != snapshot_spec["sha256"]:
        raise RuntimeError("v42 immutable archived snapshot SHA256 mismatch")
    _validate_v42_immutable_archived_metadata(
        payload,
        checkpoint_path=resolved,
        optimizer_updates=optimizer_updates,
        checkpoint_reason=str(snapshot_spec["checkpoint_reason"]),
    )

    terminal_path = Path(_V42_IMMUTABLE_ARCHIVED_TERMINAL["path"]).resolve(
        strict=True
    )
    terminal_before = _v42_immutable_archived_file_record(terminal_path)
    if terminal_before["sha256"] != _V42_IMMUTABLE_ARCHIVED_TERMINAL["sha256"]:
        raise RuntimeError("v42 immutable archived terminal U400 SHA256 mismatch")
    terminal_payload = _torch_load_v42_immutable_metadata(terminal_path)
    try:
        _validate_v42_immutable_archived_metadata(
            terminal_payload,
            checkpoint_path=terminal_path,
            optimizer_updates=400,
            checkpoint_reason="max_train_iters",
        )
    finally:
        del terminal_payload

    return {
        "schema": "pivot.stageb.v42_immutable_archived_diagnostic/v1",
        "optimizer_updates": optimizer_updates,
        "checkpoint_reason": str(snapshot_spec["checkpoint_reason"]),
        "snapshot_before_validation": snapshot_before,
        "terminal_before_validation": terminal_before,
    }


def _verify_v42_immutable_archived_diagnostic_files(
    checkpoint_path: Path,
) -> Dict[str, Any]:
    resolved = Path(checkpoint_path).expanduser().resolve(strict=True)
    optimizer_updates, snapshot_spec = _v42_immutable_archived_snapshot_spec(
        resolved
    )
    snapshot = _v42_immutable_archived_file_record(resolved)
    terminal_path = Path(_V42_IMMUTABLE_ARCHIVED_TERMINAL["path"]).resolve(
        strict=True
    )
    terminal = _v42_immutable_archived_file_record(terminal_path)
    if snapshot["sha256"] != snapshot_spec["sha256"]:
        raise RuntimeError("v42 immutable archived snapshot changed during evaluation")
    if terminal["sha256"] != _V42_IMMUTABLE_ARCHIVED_TERMINAL["sha256"]:
        raise RuntimeError("v42 immutable archived terminal changed during evaluation")
    return {
        "optimizer_updates": optimizer_updates,
        "snapshot": snapshot,
        "terminal": terminal,
    }


def _validate_v43_deployed_routing_config(cfg) -> bool:
    """Validate the hash-independent v43 architecture contract."""
    revision = str(
        getattr(cfg, "stage_b_dense_duty_confidence_revision", "")
    ).strip()
    if revision != _V43_DEPLOYED_ROUTING_REVISION:
        return False

    observed = {
        "gate_gradient_contract": str(
            getattr(
                cfg,
                "stage_b_dense_duty_confidence_gate_gradient_contract",
                "",
            )
        ).strip(),
        "routing_weight": float(
            getattr(cfg, "stage_b_dense_duty_deployed_veto_routing_weight", -1.0)
        ),
        "positive_max": float(
            getattr(cfg, "stage_b_dense_duty_deployed_veto_positive_max", -1.0)
        ),
        "tn_min": float(
            getattr(cfg, "stage_b_dense_duty_deployed_veto_tn_min", -1.0)
        ),
        "carrier_pair_gradient_contract": str(
            getattr(
                cfg,
                "stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract",
                "bidirectional_v1",
            )
        ).strip(),
        "token_edit_query_scope": str(
            getattr(cfg, "stage_b_v21_token_edit_query_scope", "target_iou_v1")
        ).strip().lower(),
        "positive_tail_gradient_contract": str(
            getattr(
                cfg,
                "stage_b_v15_tail_queue_positive_gradient_contract",
                "",
            )
        ).strip(),
        "veto_gate_offset": float(
            getattr(cfg, "stage_b_dense_duty_confidence_veto_gate_offset", -1.0)
        ),
    }
    expected = {
        "gate_gradient_contract": _V43_DEPLOYED_ROUTING_GATE_CONTRACT,
        "routing_weight": _V43_DEPLOYED_ROUTING_WEIGHT,
        "positive_max": _V43_DEPLOYED_ROUTING_POSITIVE_MAX,
        "tn_min": _V43_DEPLOYED_ROUTING_TN_MIN,
        "carrier_pair_gradient_contract": "bidirectional_v1",
        "token_edit_query_scope": "target_iou_v1",
        "positive_tail_gradient_contract": (
            "elementwise_bounded_mean_plus_sixteenth_exact_lower_tail_st_v6"
        ),
        "veto_gate_offset": 0.0,
    }
    drift = {
        key: (observed[key], expected_value)
        for key, expected_value in expected.items()
        if observed[key] != expected_value
    }
    if drift:
        raise RuntimeError(
            "v43 deployed-routing confidence config drifted: "
            + json.dumps(drift, sort_keys=True)
        )
    return True


def _validate_v45_split_tail_aligned_config(cfg) -> bool:
    """Validate the hash-independent v45 split/tail-aligned contract."""
    revision = str(
        getattr(cfg, "stage_b_dense_duty_confidence_revision", "")
    ).strip()
    if revision != _V45_SPLIT_TAIL_ALIGNED_REVISION:
        return False

    observed = {
        "gate_gradient_contract": str(
            getattr(
                cfg,
                "stage_b_dense_duty_confidence_gate_gradient_contract",
                "",
            )
        ).strip(),
        "routing_weight": float(
            getattr(cfg, "stage_b_dense_duty_deployed_veto_routing_weight", -1.0)
        ),
        "positive_max": float(
            getattr(cfg, "stage_b_dense_duty_deployed_veto_positive_max", -1.0)
        ),
        "tn_min": float(
            getattr(cfg, "stage_b_dense_duty_deployed_veto_tn_min", -1.0)
        ),
        "carrier_pair_gradient_contract": str(
            getattr(
                cfg,
                "stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract",
                "bidirectional_v1",
            )
        ).strip(),
        "token_edit_query_scope": str(
            getattr(cfg, "stage_b_v21_token_edit_query_scope", "target_iou_v1")
        ).strip().lower(),
        "positive_tail_gradient_contract": str(
            getattr(
                cfg,
                "stage_b_v15_tail_queue_positive_gradient_contract",
                "",
            )
        ).strip(),
        "veto_gate_offset": float(
            getattr(cfg, "stage_b_dense_duty_confidence_veto_gate_offset", -1.0)
        ),
        "head_gradient_contract": str(
            getattr(
                cfg,
                "stage_b_dense_duty_confidence_head_gradient_contract",
                "",
            )
        ).strip(),
        "routing_reduction_contract": str(
            getattr(
                cfg,
                "stage_b_dense_duty_deployed_veto_routing_reduction_contract",
                "",
            )
        ).strip(),
        "positive_trust_reduction_contract": str(
            getattr(
                cfg,
                "stage_b_v15_tail_queue_positive_trust_reduction_contract",
                "",
            )
        ).strip(),
        "positive_trust_contract": str(
            getattr(cfg, "stage_b_dense_duty_positive_trust_contract", "")
        ).strip(),
    }
    expected = {
        "gate_gradient_contract": _V43_DEPLOYED_ROUTING_GATE_CONTRACT,
        "routing_weight": 1.0,
        "positive_max": _V43_DEPLOYED_ROUTING_POSITIVE_MAX,
        "tn_min": _V43_DEPLOYED_ROUTING_TN_MIN,
        "carrier_pair_gradient_contract": "bidirectional_v1",
        "token_edit_query_scope": "target_iou_v1",
        "positive_tail_gradient_contract": (
            "elementwise_bounded_mean_plus_sixteenth_exact_lower_tail_st_v6"
        ),
        "veto_gate_offset": 0.0,
        "head_gradient_contract": _V45_SPLIT_TAIL_ALIGNED_HEAD_CONTRACT,
        "routing_reduction_contract": _V45_SPLIT_TAIL_ALIGNED_ROUTING_REDUCTION,
        "positive_trust_reduction_contract": (
            _V45_SPLIT_TAIL_ALIGNED_TRUST_REDUCTION
        ),
        "positive_trust_contract": _V45_SPLIT_TAIL_ALIGNED_POSITIVE_TRUST,
    }
    drift = {
        key: (observed[key], expected_value)
        for key, expected_value in expected.items()
        if observed[key] != expected_value
    }
    if drift:
        raise RuntimeError(
            "v45 split-tail-aligned confidence config drifted: "
            + json.dumps(drift, sort_keys=True)
        )
    return True


def _validate_v46_split_positive_tail_config(cfg) -> bool:
    """Validate the hash-independent V44 plus positive-tail V46 contract."""
    revision = str(
        getattr(cfg, "stage_b_dense_duty_confidence_revision", "")
    ).strip()
    if revision != _V46_SPLIT_POSITIVE_TAIL_REVISION:
        return False

    observed = {
        "gate_gradient_contract": str(
            getattr(
                cfg,
                "stage_b_dense_duty_confidence_gate_gradient_contract",
                "",
            )
        ).strip(),
        "routing_weight": float(
            getattr(cfg, "stage_b_dense_duty_deployed_veto_routing_weight", -1.0)
        ),
        "positive_max": float(
            getattr(cfg, "stage_b_dense_duty_deployed_veto_positive_max", -1.0)
        ),
        "tn_min": float(
            getattr(cfg, "stage_b_dense_duty_deployed_veto_tn_min", -1.0)
        ),
        "carrier_pair_gradient_contract": str(
            getattr(
                cfg,
                "stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract",
                "bidirectional_v1",
            )
        ).strip(),
        "token_edit_query_scope": str(
            getattr(cfg, "stage_b_v21_token_edit_query_scope", "target_iou_v1")
        ).strip().lower(),
        "positive_tail_gradient_contract": str(
            getattr(
                cfg,
                "stage_b_v15_tail_queue_positive_gradient_contract",
                "",
            )
        ).strip(),
        "veto_gate_offset": float(
            getattr(cfg, "stage_b_dense_duty_confidence_veto_gate_offset", -1.0)
        ),
        "head_gradient_contract": str(
            getattr(
                cfg,
                "stage_b_dense_duty_confidence_head_gradient_contract",
                "",
            )
        ).strip(),
        "routing_reduction_contract": str(
            getattr(
                cfg,
                "stage_b_dense_duty_deployed_veto_routing_reduction_contract",
                "",
            )
        ).strip(),
        "positive_trust_reduction_contract": str(
            getattr(
                cfg,
                "stage_b_v15_tail_queue_positive_trust_reduction_contract",
                "",
            )
        ).strip(),
        "positive_trust_contract": str(
            getattr(cfg, "stage_b_dense_duty_positive_trust_contract", "")
        ).strip(),
    }
    expected = {
        "gate_gradient_contract": _V43_DEPLOYED_ROUTING_GATE_CONTRACT,
        "routing_weight": _V43_DEPLOYED_ROUTING_WEIGHT,
        "positive_max": _V43_DEPLOYED_ROUTING_POSITIVE_MAX,
        "tn_min": _V43_DEPLOYED_ROUTING_TN_MIN,
        "carrier_pair_gradient_contract": "bidirectional_v1",
        "token_edit_query_scope": "target_iou_v1",
        "positive_tail_gradient_contract": (
            "elementwise_bounded_mean_plus_sixteenth_exact_lower_tail_st_v6"
        ),
        "veto_gate_offset": 0.0,
        "head_gradient_contract": _V46_SPLIT_POSITIVE_TAIL_HEAD_CONTRACT,
        "routing_reduction_contract": _V46_SPLIT_POSITIVE_TAIL_ROUTING_REDUCTION,
        "positive_trust_reduction_contract": (
            _V46_SPLIT_POSITIVE_TAIL_TRUST_REDUCTION
        ),
        "positive_trust_contract": _V46_SPLIT_POSITIVE_TAIL_POSITIVE_TRUST,
    }
    drift = {
        key: (observed[key], expected_value)
        for key, expected_value in expected.items()
        if observed[key] != expected_value
    }
    if drift:
        raise RuntimeError(
            "v46 split-positive-tail confidence config drifted: "
            + json.dumps(drift, sort_keys=True)
        )
    return True


def _validate_v47_split_boundary_routing_config(cfg) -> bool:
    """Validate the hash-independent V46 plus boundary-routing V47 contract."""
    revision = str(
        getattr(cfg, "stage_b_dense_duty_confidence_revision", "")
    ).strip()
    if revision != _V47_SPLIT_BOUNDARY_ROUTING_REVISION:
        return False

    observed = {
        "gate_gradient_contract": str(
            getattr(
                cfg,
                "stage_b_dense_duty_confidence_gate_gradient_contract",
                "",
            )
        ).strip(),
        "routing_weight": float(
            getattr(cfg, "stage_b_dense_duty_deployed_veto_routing_weight", -1.0)
        ),
        "positive_max": float(
            getattr(cfg, "stage_b_dense_duty_deployed_veto_positive_max", -1.0)
        ),
        "tn_min": float(
            getattr(cfg, "stage_b_dense_duty_deployed_veto_tn_min", -1.0)
        ),
        "carrier_pair_gradient_contract": str(
            getattr(
                cfg,
                "stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract",
                "bidirectional_v1",
            )
        ).strip(),
        "token_edit_query_scope": str(
            getattr(cfg, "stage_b_v21_token_edit_query_scope", "target_iou_v1")
        ).strip().lower(),
        "positive_tail_gradient_contract": str(
            getattr(
                cfg,
                "stage_b_v15_tail_queue_positive_gradient_contract",
                "",
            )
        ).strip(),
        "veto_gate_offset": float(
            getattr(cfg, "stage_b_dense_duty_confidence_veto_gate_offset", -1.0)
        ),
        "head_gradient_contract": str(
            getattr(
                cfg,
                "stage_b_dense_duty_confidence_head_gradient_contract",
                "",
            )
        ).strip(),
        "routing_reduction_contract": str(
            getattr(
                cfg,
                "stage_b_dense_duty_deployed_veto_routing_reduction_contract",
                "",
            )
        ).strip(),
        "positive_trust_reduction_contract": str(
            getattr(
                cfg,
                "stage_b_v15_tail_queue_positive_trust_reduction_contract",
                "",
            )
        ).strip(),
        "positive_trust_contract": str(
            getattr(cfg, "stage_b_dense_duty_positive_trust_contract", "")
        ).strip(),
    }
    expected = {
        "gate_gradient_contract": _V43_DEPLOYED_ROUTING_GATE_CONTRACT,
        "routing_weight": _V43_DEPLOYED_ROUTING_WEIGHT,
        "positive_max": _V43_DEPLOYED_ROUTING_POSITIVE_MAX,
        "tn_min": _V43_DEPLOYED_ROUTING_TN_MIN,
        "carrier_pair_gradient_contract": "bidirectional_v1",
        "token_edit_query_scope": "target_iou_v1",
        "positive_tail_gradient_contract": (
            "elementwise_bounded_mean_plus_sixteenth_exact_lower_tail_st_v6"
        ),
        "veto_gate_offset": 0.0,
        "head_gradient_contract": _V47_SPLIT_BOUNDARY_ROUTING_HEAD_CONTRACT,
        "routing_reduction_contract": (
            _V47_SPLIT_BOUNDARY_ROUTING_ROUTING_REDUCTION
        ),
        "positive_trust_reduction_contract": (
            _V47_SPLIT_BOUNDARY_ROUTING_TRUST_REDUCTION
        ),
        "positive_trust_contract": _V47_SPLIT_BOUNDARY_ROUTING_POSITIVE_TRUST,
    }
    drift = {
        key: (observed[key], expected_value)
        for key, expected_value in expected.items()
        if observed[key] != expected_value
    }
    if drift:
        raise RuntimeError(
            "v47 split-boundary-routing confidence config drifted: "
            + json.dumps(drift, sort_keys=True)
        )
    return True


def _validate_v48_split_fpr_active_set_config(cfg) -> bool:
    """Validate the hash-independent V47 plus exact FPR active-set contract."""
    revision = str(
        getattr(cfg, "stage_b_dense_duty_confidence_revision", "")
    ).strip()
    if revision != _V48_SPLIT_FPR_ACTIVE_SET_REVISION:
        return False

    observed = {
        "gate_gradient_contract": str(
            getattr(
                cfg,
                "stage_b_dense_duty_confidence_gate_gradient_contract",
                "",
            )
        ).strip(),
        "routing_weight": float(
            getattr(cfg, "stage_b_dense_duty_deployed_veto_routing_weight", -1.0)
        ),
        "positive_max": float(
            getattr(cfg, "stage_b_dense_duty_deployed_veto_positive_max", -1.0)
        ),
        "tn_min": float(
            getattr(cfg, "stage_b_dense_duty_deployed_veto_tn_min", -1.0)
        ),
        "carrier_pair_gradient_contract": str(
            getattr(
                cfg,
                "stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract",
                "bidirectional_v1",
            )
        ).strip(),
        "token_edit_query_scope": str(
            getattr(cfg, "stage_b_v21_token_edit_query_scope", "target_iou_v1")
        ).strip().lower(),
        "positive_tail_gradient_contract": str(
            getattr(
                cfg,
                "stage_b_v15_tail_queue_positive_gradient_contract",
                "",
            )
        ).strip(),
        "veto_gate_offset": float(
            getattr(cfg, "stage_b_dense_duty_confidence_veto_gate_offset", -1.0)
        ),
        "head_gradient_contract": str(
            getattr(
                cfg,
                "stage_b_dense_duty_confidence_head_gradient_contract",
                "",
            )
        ).strip(),
        "routing_reduction_contract": str(
            getattr(
                cfg,
                "stage_b_dense_duty_deployed_veto_routing_reduction_contract",
                "",
            )
        ).strip(),
        "positive_trust_reduction_contract": str(
            getattr(
                cfg,
                "stage_b_v15_tail_queue_positive_trust_reduction_contract",
                "",
            )
        ).strip(),
        "positive_trust_contract": str(
            getattr(cfg, "stage_b_dense_duty_positive_trust_contract", "")
        ).strip(),
        "negative_reduction_contract": str(
            getattr(
                cfg,
                "stage_b_v15_tail_queue_negative_reduction_contract",
                "",
            )
        ).strip(),
    }
    expected = {
        "gate_gradient_contract": _V43_DEPLOYED_ROUTING_GATE_CONTRACT,
        "routing_weight": _V43_DEPLOYED_ROUTING_WEIGHT,
        "positive_max": _V43_DEPLOYED_ROUTING_POSITIVE_MAX,
        "tn_min": _V43_DEPLOYED_ROUTING_TN_MIN,
        "carrier_pair_gradient_contract": "bidirectional_v1",
        "token_edit_query_scope": "target_iou_v1",
        "positive_tail_gradient_contract": (
            "elementwise_bounded_mean_plus_sixteenth_exact_lower_tail_st_v6"
        ),
        "veto_gate_offset": 0.0,
        "head_gradient_contract": _V47_SPLIT_BOUNDARY_ROUTING_HEAD_CONTRACT,
        "routing_reduction_contract": (
            _V47_SPLIT_BOUNDARY_ROUTING_ROUTING_REDUCTION
        ),
        "positive_trust_reduction_contract": (
            _V47_SPLIT_BOUNDARY_ROUTING_TRUST_REDUCTION
        ),
        "positive_trust_contract": _V47_SPLIT_BOUNDARY_ROUTING_POSITIVE_TRUST,
        "negative_reduction_contract": (
            _V48_SPLIT_FPR_ACTIVE_SET_NEGATIVE_REDUCTION
        ),
    }
    drift = {
        key: (observed[key], expected_value)
        for key, expected_value in expected.items()
        if observed[key] != expected_value
    }
    if drift:
        raise RuntimeError(
            "v48 split-FPR-active-set confidence config drifted: "
            + json.dumps(drift, sort_keys=True)
        )
    return True


def _validate_v49_split_global_trust_veto_config(cfg) -> bool:
    """Validate the hash-independent V47 plus split global-head contract."""
    revision = str(
        getattr(cfg, "stage_b_dense_duty_confidence_revision", "")
    ).strip()
    if revision != _V49_SPLIT_GLOBAL_TRUST_VETO_REVISION:
        return False

    observed = {
        "gate_gradient_contract": str(
            getattr(
                cfg,
                "stage_b_dense_duty_confidence_gate_gradient_contract",
                "",
            )
        ).strip(),
        "routing_weight": float(
            getattr(cfg, "stage_b_dense_duty_deployed_veto_routing_weight", -1.0)
        ),
        "positive_max": float(
            getattr(cfg, "stage_b_dense_duty_deployed_veto_positive_max", -1.0)
        ),
        "tn_min": float(
            getattr(cfg, "stage_b_dense_duty_deployed_veto_tn_min", -1.0)
        ),
        "carrier_pair_gradient_contract": str(
            getattr(
                cfg,
                "stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract",
                "bidirectional_v1",
            )
        ).strip(),
        "token_edit_query_scope": str(
            getattr(cfg, "stage_b_v21_token_edit_query_scope", "target_iou_v1")
        ).strip().lower(),
        "positive_tail_gradient_contract": str(
            getattr(
                cfg,
                "stage_b_v15_tail_queue_positive_gradient_contract",
                "",
            )
        ).strip(),
        "veto_gate_offset": float(
            getattr(cfg, "stage_b_dense_duty_confidence_veto_gate_offset", -1.0)
        ),
        "head_gradient_contract": str(
            getattr(
                cfg,
                "stage_b_dense_duty_confidence_head_gradient_contract",
                "",
            )
        ).strip(),
        "routing_reduction_contract": str(
            getattr(
                cfg,
                "stage_b_dense_duty_deployed_veto_routing_reduction_contract",
                "",
            )
        ).strip(),
        "positive_trust_reduction_contract": str(
            getattr(
                cfg,
                "stage_b_v15_tail_queue_positive_trust_reduction_contract",
                "",
            )
        ).strip(),
        "positive_trust_contract": str(
            getattr(cfg, "stage_b_dense_duty_positive_trust_contract", "")
        ).strip(),
        "negative_reduction_contract": str(
            getattr(
                cfg,
                "stage_b_v15_tail_queue_negative_reduction_contract",
                _V49_SPLIT_GLOBAL_TRUST_VETO_NEGATIVE_REDUCTION,
            )
        ).strip(),
    }
    expected = {
        "gate_gradient_contract": _V43_DEPLOYED_ROUTING_GATE_CONTRACT,
        "routing_weight": _V43_DEPLOYED_ROUTING_WEIGHT,
        "positive_max": _V43_DEPLOYED_ROUTING_POSITIVE_MAX,
        "tn_min": _V43_DEPLOYED_ROUTING_TN_MIN,
        "carrier_pair_gradient_contract": "bidirectional_v1",
        "token_edit_query_scope": "target_iou_v1",
        "positive_tail_gradient_contract": (
            "elementwise_bounded_mean_plus_sixteenth_exact_lower_tail_st_v6"
        ),
        "veto_gate_offset": 0.0,
        "head_gradient_contract": _V49_SPLIT_GLOBAL_TRUST_VETO_HEAD_CONTRACT,
        "routing_reduction_contract": (
            _V47_SPLIT_BOUNDARY_ROUTING_ROUTING_REDUCTION
        ),
        "positive_trust_reduction_contract": (
            _V47_SPLIT_BOUNDARY_ROUTING_TRUST_REDUCTION
        ),
        "positive_trust_contract": _V47_SPLIT_BOUNDARY_ROUTING_POSITIVE_TRUST,
        "negative_reduction_contract": (
            _V49_SPLIT_GLOBAL_TRUST_VETO_NEGATIVE_REDUCTION
        ),
    }
    drift = {
        key: (observed[key], expected_value)
        for key, expected_value in expected.items()
        if observed[key] != expected_value
    }
    if drift:
        raise RuntimeError(
            "v49 split-global-trust-veto confidence config drifted: "
            + json.dumps(drift, sort_keys=True)
        )
    return True


def _validate_v50_split_strong_boundary_routing_config(cfg) -> bool:
    """Validate the exact V47 surface with strong boundary routing."""
    revision = str(
        getattr(cfg, "stage_b_dense_duty_confidence_revision", "")
    ).strip()
    if revision != _V50_SPLIT_STRONG_BOUNDARY_ROUTING_REVISION:
        return False

    observed = {
        "gate_gradient_contract": str(
            getattr(
                cfg,
                "stage_b_dense_duty_confidence_gate_gradient_contract",
                "",
            )
        ).strip(),
        "routing_weight": float(
            getattr(cfg, "stage_b_dense_duty_deployed_veto_routing_weight", -1.0)
        ),
        "positive_max": float(
            getattr(cfg, "stage_b_dense_duty_deployed_veto_positive_max", -1.0)
        ),
        "tn_min": float(
            getattr(cfg, "stage_b_dense_duty_deployed_veto_tn_min", -1.0)
        ),
        "carrier_pair_gradient_contract": str(
            getattr(
                cfg,
                "stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract",
                "bidirectional_v1",
            )
        ).strip(),
        "token_edit_query_scope": str(
            getattr(cfg, "stage_b_v21_token_edit_query_scope", "target_iou_v1")
        ).strip().lower(),
        "positive_tail_gradient_contract": str(
            getattr(
                cfg,
                "stage_b_v15_tail_queue_positive_gradient_contract",
                "",
            )
        ).strip(),
        "veto_gate_offset": float(
            getattr(cfg, "stage_b_dense_duty_confidence_veto_gate_offset", -1.0)
        ),
        "head_gradient_contract": str(
            getattr(
                cfg,
                "stage_b_dense_duty_confidence_head_gradient_contract",
                "",
            )
        ).strip(),
        "routing_reduction_contract": str(
            getattr(
                cfg,
                "stage_b_dense_duty_deployed_veto_routing_reduction_contract",
                "",
            )
        ).strip(),
        "positive_trust_reduction_contract": str(
            getattr(
                cfg,
                "stage_b_v15_tail_queue_positive_trust_reduction_contract",
                "",
            )
        ).strip(),
        "positive_trust_contract": str(
            getattr(cfg, "stage_b_dense_duty_positive_trust_contract", "")
        ).strip(),
        "negative_reduction_contract": str(
            getattr(
                cfg,
                "stage_b_v15_tail_queue_negative_reduction_contract",
                _V50_SPLIT_STRONG_BOUNDARY_ROUTING_NEGATIVE_REDUCTION,
            )
        ).strip(),
    }
    expected = {
        "gate_gradient_contract": _V43_DEPLOYED_ROUTING_GATE_CONTRACT,
        "routing_weight": _V50_SPLIT_STRONG_BOUNDARY_ROUTING_WEIGHT,
        "positive_max": _V43_DEPLOYED_ROUTING_POSITIVE_MAX,
        "tn_min": _V43_DEPLOYED_ROUTING_TN_MIN,
        "carrier_pair_gradient_contract": "bidirectional_v1",
        "token_edit_query_scope": "target_iou_v1",
        "positive_tail_gradient_contract": (
            "elementwise_bounded_mean_plus_sixteenth_exact_lower_tail_st_v6"
        ),
        "veto_gate_offset": 0.0,
        "head_gradient_contract": (
            _V50_SPLIT_STRONG_BOUNDARY_ROUTING_HEAD_CONTRACT
        ),
        "routing_reduction_contract": (
            _V50_SPLIT_STRONG_BOUNDARY_ROUTING_ROUTING_REDUCTION
        ),
        "positive_trust_reduction_contract": (
            _V50_SPLIT_STRONG_BOUNDARY_ROUTING_TRUST_REDUCTION
        ),
        "positive_trust_contract": (
            _V50_SPLIT_STRONG_BOUNDARY_ROUTING_POSITIVE_TRUST
        ),
        "negative_reduction_contract": (
            _V50_SPLIT_STRONG_BOUNDARY_ROUTING_NEGATIVE_REDUCTION
        ),
    }
    drift = {
        key: (observed[key], expected_value)
        for key, expected_value in expected.items()
        if observed[key] != expected_value
    }
    if drift:
        raise RuntimeError(
            "v50 split-strong-boundary-routing confidence config drifted: "
            + json.dumps(drift, sort_keys=True)
        )
    return True


def _validate_v51_split_independent_deployed_router_config(cfg) -> bool:
    """Validate the exact V47 surface with an independent deployed router."""
    revision = str(
        getattr(cfg, "stage_b_dense_duty_confidence_revision", "")
    ).strip()
    if revision != _V51_SPLIT_INDEPENDENT_DEPLOYED_ROUTER_REVISION:
        return False

    observed = {
        "gate_gradient_contract": str(
            getattr(
                cfg,
                "stage_b_dense_duty_confidence_gate_gradient_contract",
                "",
            )
        ).strip(),
        "routing_weight": float(
            getattr(cfg, "stage_b_dense_duty_deployed_veto_routing_weight", -1.0)
        ),
        "positive_max": float(
            getattr(cfg, "stage_b_dense_duty_deployed_veto_positive_max", -1.0)
        ),
        "tn_min": float(
            getattr(cfg, "stage_b_dense_duty_deployed_veto_tn_min", -1.0)
        ),
        "carrier_pair_gradient_contract": str(
            getattr(
                cfg,
                "stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract",
                "bidirectional_v1",
            )
        ).strip(),
        "token_edit_query_scope": str(
            getattr(cfg, "stage_b_v21_token_edit_query_scope", "target_iou_v1")
        ).strip().lower(),
        "positive_tail_gradient_contract": str(
            getattr(
                cfg,
                "stage_b_v15_tail_queue_positive_gradient_contract",
                "",
            )
        ).strip(),
        "veto_gate_offset": float(
            getattr(cfg, "stage_b_dense_duty_confidence_veto_gate_offset", -1.0)
        ),
        "head_gradient_contract": str(
            getattr(
                cfg,
                "stage_b_dense_duty_confidence_head_gradient_contract",
                "",
            )
        ).strip(),
        "routing_reduction_contract": str(
            getattr(
                cfg,
                "stage_b_dense_duty_deployed_veto_routing_reduction_contract",
                "",
            )
        ).strip(),
        "positive_trust_reduction_contract": str(
            getattr(
                cfg,
                "stage_b_v15_tail_queue_positive_trust_reduction_contract",
                "",
            )
        ).strip(),
        "positive_trust_contract": str(
            getattr(cfg, "stage_b_dense_duty_positive_trust_contract", "")
        ).strip(),
        "negative_reduction_contract": str(
            getattr(
                cfg,
                "stage_b_v15_tail_queue_negative_reduction_contract",
                _V51_SPLIT_INDEPENDENT_DEPLOYED_ROUTER_NEGATIVE_REDUCTION,
            )
        ).strip(),
    }
    expected = {
        "gate_gradient_contract": _V43_DEPLOYED_ROUTING_GATE_CONTRACT,
        "routing_weight": _V51_SPLIT_INDEPENDENT_DEPLOYED_ROUTER_WEIGHT,
        "positive_max": _V43_DEPLOYED_ROUTING_POSITIVE_MAX,
        "tn_min": _V43_DEPLOYED_ROUTING_TN_MIN,
        "carrier_pair_gradient_contract": "bidirectional_v1",
        "token_edit_query_scope": "target_iou_v1",
        "positive_tail_gradient_contract": (
            "elementwise_bounded_mean_plus_sixteenth_exact_lower_tail_st_v6"
        ),
        "veto_gate_offset": 0.0,
        "head_gradient_contract": (
            _V51_SPLIT_INDEPENDENT_DEPLOYED_ROUTER_HEAD_CONTRACT
        ),
        "routing_reduction_contract": (
            _V51_SPLIT_INDEPENDENT_DEPLOYED_ROUTER_ROUTING_REDUCTION
        ),
        "positive_trust_reduction_contract": (
            _V51_SPLIT_INDEPENDENT_DEPLOYED_ROUTER_TRUST_REDUCTION
        ),
        "positive_trust_contract": (
            _V51_SPLIT_INDEPENDENT_DEPLOYED_ROUTER_POSITIVE_TRUST
        ),
        "negative_reduction_contract": (
            _V51_SPLIT_INDEPENDENT_DEPLOYED_ROUTER_NEGATIVE_REDUCTION
        ),
    }
    drift = {
        key: (observed[key], expected_value)
        for key, expected_value in expected.items()
        if observed[key] != expected_value
    }
    if drift:
        raise RuntimeError(
            "v51 split-independent-deployed-router confidence config drifted: "
            + json.dumps(drift, sort_keys=True)
        )
    return True


def _validate_v52_candidate_sample_calibrator_config(cfg) -> bool:
    """Validate the exact fresh-U6551 V52 surface without a deployed router."""
    revision = str(
        getattr(cfg, "stage_b_dense_duty_confidence_revision", "")
    ).strip()
    if revision != _V52_CANDIDATE_SAMPLE_CALIBRATOR_REVISION:
        return False

    observed = {
        "gate_gradient_contract": str(
            getattr(
                cfg,
                "stage_b_dense_duty_confidence_gate_gradient_contract",
                "",
            )
        ).strip(),
        "routing_weight": float(
            getattr(cfg, "stage_b_dense_duty_deployed_veto_routing_weight", -1.0)
        ),
        "positive_max": float(
            getattr(cfg, "stage_b_dense_duty_deployed_veto_positive_max", -1.0)
        ),
        "tn_min": float(
            getattr(cfg, "stage_b_dense_duty_deployed_veto_tn_min", -1.0)
        ),
        "carrier_pair_gradient_contract": str(
            getattr(
                cfg,
                "stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract",
                "",
            )
        ).strip(),
        "token_edit_query_scope": str(
            getattr(cfg, "stage_b_v21_token_edit_query_scope", "")
        ).strip().lower(),
        "positive_tail_gradient_contract": str(
            getattr(
                cfg,
                "stage_b_v15_tail_queue_positive_gradient_contract",
                "",
            )
        ).strip(),
        "veto_gate_offset": float(
            getattr(cfg, "stage_b_dense_duty_confidence_veto_gate_offset", -1.0)
        ),
        "head_gradient_contract": str(
            getattr(
                cfg,
                "stage_b_dense_duty_confidence_head_gradient_contract",
                "",
            )
        ).strip(),
        "routing_reduction_contract": str(
            getattr(
                cfg,
                "stage_b_dense_duty_deployed_veto_routing_reduction_contract",
                "",
            )
        ).strip(),
        "positive_trust_reduction_contract": str(
            getattr(
                cfg,
                "stage_b_v15_tail_queue_positive_trust_reduction_contract",
                "",
            )
        ).strip(),
        "positive_trust_contract": str(
            getattr(cfg, "stage_b_dense_duty_positive_trust_contract", "")
        ).strip(),
        "negative_reduction_contract": str(
            getattr(
                cfg,
                "stage_b_v15_tail_queue_negative_reduction_contract",
                "",
            )
        ).strip(),
        "pool_feature_contract": str(
            getattr(
                cfg,
                "stage_b_dense_duty_confidence_pool_feature_contract",
                "",
            )
        ).strip(),
        "rank_source_optimizer_updates": int(
            getattr(cfg, "stage_b_dense_duty_rank_source_optimizer_updates", -1)
        ),
        "trainable_params_min": int(
            getattr(cfg, "stage_b_v11_trainable_params_min", -1)
        ),
        "trainable_params_max": int(
            getattr(cfg, "stage_b_v11_trainable_params_max", -1)
        ),
    }
    expected = {
        "gate_gradient_contract": _V43_DEPLOYED_ROUTING_GATE_CONTRACT,
        "routing_weight": _V52_CANDIDATE_SAMPLE_CALIBRATOR_ROUTING_WEIGHT,
        "positive_max": _V43_DEPLOYED_ROUTING_POSITIVE_MAX,
        "tn_min": _V43_DEPLOYED_ROUTING_TN_MIN,
        "carrier_pair_gradient_contract": "bidirectional_v1",
        "token_edit_query_scope": "target_iou_v1",
        "positive_tail_gradient_contract": (
            "elementwise_bounded_mean_plus_sixteenth_exact_lower_tail_st_v6"
        ),
        "veto_gate_offset": 0.0,
        "head_gradient_contract": _V52_CANDIDATE_SAMPLE_CALIBRATOR_HEAD_CONTRACT,
        "routing_reduction_contract": (
            _V52_CANDIDATE_SAMPLE_CALIBRATOR_ROUTING_REDUCTION
        ),
        "positive_trust_reduction_contract": (
            _V52_CANDIDATE_SAMPLE_CALIBRATOR_TRUST_REDUCTION
        ),
        "positive_trust_contract": (
            _V52_CANDIDATE_SAMPLE_CALIBRATOR_POSITIVE_TRUST
        ),
        "negative_reduction_contract": (
            _V52_CANDIDATE_SAMPLE_CALIBRATOR_NEGATIVE_REDUCTION
        ),
        "pool_feature_contract": (
            _V52_CANDIDATE_SAMPLE_CALIBRATOR_POOL_FEATURE_CONTRACT
        ),
        "rank_source_optimizer_updates": (
            _V52_CANDIDATE_SAMPLE_CALIBRATOR_SOURCE_UPDATES
        ),
        "trainable_params_min": _V52_CANDIDATE_SAMPLE_CALIBRATOR_TRAINABLE_PARAMS,
        "trainable_params_max": _V52_CANDIDATE_SAMPLE_CALIBRATOR_TRAINABLE_PARAMS,
    }
    drift = {
        key: (observed[key], expected_value)
        for key, expected_value in expected.items()
        if observed[key] != expected_value
    }
    if drift:
        raise RuntimeError(
            "v52 candidate/sample-calibrator confidence config drifted: "
            + json.dumps(drift, sort_keys=True)
        )
    return True


def _validate_v52_three_owner_runtime_audit(
    runtime: Mapping[str, Any],
    *,
    optimizer_updates: int,
) -> None:
    """Require complete, violation-free V52 clip evidence for every update."""
    if not isinstance(runtime, Mapping) or type(optimizer_updates) is not int:
        raise RuntimeError(
            "v52 confidence checkpoint lacks valid no-router three-owner "
            "gradient/clip runtime evidence"
        )

    violation_fields = (
        "owner_clip_violation_steps",
        "active_pre_decomposition_violation_steps",
        "active_post_decomposition_violation_steps",
        "live_tensor_count_violation_steps",
        "active_monotonic_violation_steps",
    )
    owner_labels = (
        "token_veto",
        "candidate_absolute",
        "sample_calibrator",
    )

    def _zero_counter(key: str) -> bool:
        value = runtime.get(key, 0)
        return type(value) is int and value == 0

    def _positive_finite(key: str) -> bool:
        value = runtime.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        return math.isfinite(float(value)) and float(value) > 0.0

    invalid_owner_evidence = any(
        not _positive_finite(f"max_{owner}_grad_norm_preclip")
        or not _zero_counter(f"nonfinite_{owner}_gradient_boundaries")
        or not _zero_counter(f"zero_{owner}_gradient_successful_steps")
        for owner in owner_labels
    )
    if (
        runtime.get("clip_contract_schema")
        != _V52_THREE_OWNER_CLIP_CONTRACT_SCHEMA
        or type(runtime.get("clip_contract_checked_steps")) is not int
        or runtime.get("clip_contract_checked_steps") != optimizer_updates
        or any(not _zero_counter(field) for field in violation_fields)
        or invalid_owner_evidence
        or any("deployed_router" in str(key) for key in runtime)
    ):
        raise RuntimeError(
            "v52 confidence checkpoint lacks valid no-router three-owner "
            "gradient/clip runtime evidence"
        )


def _validate_fulltext_global_absolute_config(
    cfg,
    *,
    revision_contract: str,
    head_contract: str,
    gate_contract: str,
    pool_feature_contract: str,
    routing_weight: float,
    routing_reduction: str,
    trust_reduction: str,
    positive_trust: str,
    negative_reduction: str,
    carrier_selector: str,
    source_updates: int,
    trainable_params: int,
    revision_label: str,
) -> bool:
    """Validate one exact fresh-U6551 full-text two-owner surface."""
    revision = str(
        getattr(cfg, "stage_b_dense_duty_confidence_revision", "")
    ).strip()
    if revision != revision_contract:
        return False

    observed = {
        "phrase_aggregation": str(
            getattr(cfg, "stage_b_dense_duty_confidence_phrase_aggregation", "")
        ).strip(),
        "gate_gradient_contract": str(
            getattr(
                cfg,
                "stage_b_dense_duty_confidence_gate_gradient_contract",
                "",
            )
        ).strip(),
        "head_gradient_contract": str(
            getattr(
                cfg,
                "stage_b_dense_duty_confidence_head_gradient_contract",
                "",
            )
        ).strip(),
        "pool_feature_contract": str(
            getattr(
                cfg,
                "stage_b_dense_duty_confidence_pool_feature_contract",
                "",
            )
        ).strip(),
        "token_contract": str(
            getattr(cfg, "stage_b_dense_duty_confidence_token_contract", "")
        ).strip(),
        "rank_evidence_contract": str(
            getattr(
                cfg,
                "stage_b_dense_duty_confidence_rank_evidence_contract",
                "",
            )
        ).strip(),
        "residual_parameterization_gain": float(
            getattr(
                cfg,
                "stage_b_dense_duty_confidence_residual_parameterization_gain",
                -1.0,
            )
        ),
        "routing_weight": float(
            getattr(cfg, "stage_b_dense_duty_deployed_veto_routing_weight", -1.0)
        ),
        "positive_max": float(
            getattr(cfg, "stage_b_dense_duty_deployed_veto_positive_max", -1.0)
        ),
        "tn_min": float(
            getattr(cfg, "stage_b_dense_duty_deployed_veto_tn_min", -1.0)
        ),
        "carrier_pair_gradient_contract": str(
            getattr(
                cfg,
                "stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract",
                "",
            )
        ).strip(),
        "carrier_selector_contract": str(
            getattr(
                cfg,
                "stage_b_dense_duty_confidence_carrier_selector_contract",
                "",
            )
        ).strip(),
        "token_edit_query_scope": str(
            getattr(cfg, "stage_b_v21_token_edit_query_scope", "")
        ).strip().lower(),
        "positive_tail_gradient_contract": str(
            getattr(
                cfg,
                "stage_b_v15_tail_queue_positive_gradient_contract",
                "",
            )
        ).strip(),
        "veto_gate_offset": float(
            getattr(cfg, "stage_b_dense_duty_confidence_veto_gate_offset", -1.0)
        ),
        "routing_reduction_contract": str(
            getattr(
                cfg,
                "stage_b_dense_duty_deployed_veto_routing_reduction_contract",
                "",
            )
        ).strip(),
        "positive_trust_reduction_contract": str(
            getattr(
                cfg,
                "stage_b_v15_tail_queue_positive_trust_reduction_contract",
                "",
            )
        ).strip(),
        "positive_trust_contract": str(
            getattr(cfg, "stage_b_dense_duty_positive_trust_contract", "")
        ).strip(),
        "negative_reduction_contract": str(
            getattr(
                cfg,
                "stage_b_v15_tail_queue_negative_reduction_contract",
                "",
            )
        ).strip(),
        "raw_veto_query_scope": str(
            getattr(cfg, "stage_b_dense_duty_raw_veto_query_scope", "")
        ).strip(),
        "raw_veto_carrier_pair_weight": float(
            getattr(cfg, "stage_b_dense_duty_raw_veto_carrier_pair_weight", -1.0)
        ),
        "raw_veto_carrier_pair_margin": float(
            getattr(cfg, "stage_b_dense_duty_raw_veto_carrier_pair_margin", -1.0)
        ),
        "raw_veto_tail_quantile": float(
            getattr(cfg, "stage_b_dense_duty_raw_veto_tail_quantile", -1.0)
        ),
        "raw_veto_tail_temperature": float(
            getattr(cfg, "stage_b_dense_duty_raw_veto_tail_temperature", -1.0)
        ),
        "raw_veto_tail_min_count": int(
            getattr(cfg, "stage_b_dense_duty_raw_veto_tail_min_count", -1)
        ),
        "rank_source_optimizer_updates": int(
            getattr(cfg, "stage_b_dense_duty_rank_source_optimizer_updates", -1)
        ),
        "trainable_params_min": int(
            getattr(cfg, "stage_b_v11_trainable_params_min", -1)
        ),
        "trainable_params_max": int(
            getattr(cfg, "stage_b_v11_trainable_params_max", -1)
        ),
    }
    expected = {
        "phrase_aggregation": (
            "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        ),
        "gate_gradient_contract": gate_contract,
        "head_gradient_contract": head_contract,
        "pool_feature_contract": pool_feature_contract,
        "token_contract": "detached_rank_token_minus_zero_init_residual_v1",
        "rank_evidence_contract": (
            "zero_init_carrier_token_rank_affine_sparse_rank_channel_v6"
        ),
        "residual_parameterization_gain": 25.0 / 3.0,
        "routing_weight": routing_weight,
        "positive_max": _V43_DEPLOYED_ROUTING_POSITIVE_MAX,
        "tn_min": _V43_DEPLOYED_ROUTING_TN_MIN,
        "carrier_pair_gradient_contract": "bidirectional_v1",
        "carrier_selector_contract": carrier_selector,
        "token_edit_query_scope": "target_iou_v1",
        "positive_tail_gradient_contract": (
            "elementwise_bounded_mean_plus_sixteenth_exact_lower_tail_st_v6"
        ),
        "veto_gate_offset": 0.0,
        "routing_reduction_contract": routing_reduction,
        "positive_trust_reduction_contract": trust_reduction,
        "positive_trust_contract": positive_trust,
        "negative_reduction_contract": negative_reduction,
        "raw_veto_query_scope": (
            "tn_all_admitted_tail_weighted_carrier_tail_paired_v7"
        ),
        "raw_veto_carrier_pair_weight": 0.25,
        "raw_veto_carrier_pair_margin": 0.25,
        "raw_veto_tail_quantile": 0.95,
        "raw_veto_tail_temperature": 0.1,
        "raw_veto_tail_min_count": 256,
        "rank_source_optimizer_updates": source_updates,
        "trainable_params_min": trainable_params,
        "trainable_params_max": trainable_params,
    }
    drift = {
        key: (observed[key], expected_value)
        for key, expected_value in expected.items()
        if observed[key] != expected_value
    }
    if drift:
        raise RuntimeError(
            f"{revision_label} fulltext/global-absolute confidence config drifted: "
            + json.dumps(drift, sort_keys=True)
        )
    return True


def _validate_v53_fulltext_global_absolute_config(cfg) -> bool:
    """Validate the exact fresh-U6551 V53 two-owner confidence surface."""
    return _validate_fulltext_global_absolute_config(
        cfg,
        revision_contract=_V53_FULLTEXT_GLOBAL_ABSOLUTE_REVISION,
        head_contract=_V53_FULLTEXT_GLOBAL_ABSOLUTE_HEAD_CONTRACT,
        gate_contract=_V53_FULLTEXT_GLOBAL_ABSOLUTE_GATE_CONTRACT,
        pool_feature_contract=_V53_FULLTEXT_GLOBAL_ABSOLUTE_POOL_FEATURE_CONTRACT,
        routing_weight=_V53_FULLTEXT_GLOBAL_ABSOLUTE_ROUTING_WEIGHT,
        routing_reduction=_V53_FULLTEXT_GLOBAL_ABSOLUTE_ROUTING_REDUCTION,
        trust_reduction=_V53_FULLTEXT_GLOBAL_ABSOLUTE_TRUST_REDUCTION,
        positive_trust=_V53_FULLTEXT_GLOBAL_ABSOLUTE_POSITIVE_TRUST,
        negative_reduction=_V53_FULLTEXT_GLOBAL_ABSOLUTE_NEGATIVE_REDUCTION,
        carrier_selector=_V53_FULLTEXT_GLOBAL_ABSOLUTE_CARRIER_SELECTOR,
        source_updates=_V53_FULLTEXT_GLOBAL_ABSOLUTE_SOURCE_UPDATES,
        trainable_params=_V53_FULLTEXT_GLOBAL_ABSOLUTE_TRAINABLE_PARAMS,
        revision_label="v53",
    )


def _validate_v54_fulltext_global_absolute_exact_residual_config(cfg) -> bool:
    """Validate the exact-reference V54 two-owner confidence surface."""
    return _validate_fulltext_global_absolute_config(
        cfg,
        revision_contract=(
            _V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_REVISION
        ),
        head_contract=(
            _V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_HEAD_CONTRACT
        ),
        gate_contract=(
            _V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_GATE_CONTRACT
        ),
        pool_feature_contract=(
            _V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_POOL_FEATURE_CONTRACT
        ),
        routing_weight=(
            _V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_ROUTING_WEIGHT
        ),
        routing_reduction=(
            _V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_ROUTING_REDUCTION
        ),
        trust_reduction=(
            _V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_TRUST_REDUCTION
        ),
        positive_trust=(
            _V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_POSITIVE_TRUST
        ),
        negative_reduction=(
            _V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_NEGATIVE_REDUCTION
        ),
        carrier_selector=(
            _V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_CARRIER_SELECTOR
        ),
        source_updates=(
            _V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_SOURCE_UPDATES
        ),
        trainable_params=(
            _V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_TRAINABLE_PARAMS
        ),
        revision_label="v54",
    )


def _validate_v55_fulltext_global_independent_absolute_config(cfg) -> bool:
    """Validate the independent-global V55 two-owner confidence surface."""
    return _validate_fulltext_global_absolute_config(
        cfg,
        revision_contract=(
            _V55_FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_REVISION
        ),
        head_contract=(
            _V55_FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_HEAD_CONTRACT
        ),
        gate_contract=(
            _V55_FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_GATE_CONTRACT
        ),
        pool_feature_contract=(
            _V55_FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_POOL_FEATURE_CONTRACT
        ),
        routing_weight=(
            _V55_FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_ROUTING_WEIGHT
        ),
        routing_reduction=(
            _V55_FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_ROUTING_REDUCTION
        ),
        trust_reduction=(
            _V55_FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_TRUST_REDUCTION
        ),
        positive_trust=(
            _V55_FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_POSITIVE_TRUST
        ),
        negative_reduction=(
            _V55_FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_NEGATIVE_REDUCTION
        ),
        carrier_selector=(
            _V55_FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_CARRIER_SELECTOR
        ),
        source_updates=(
            _V55_FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_SOURCE_UPDATES
        ),
        trainable_params=(
            _V55_FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_TRAINABLE_PARAMS
        ),
        revision_label="v55",
    )


def _validate_fulltext_two_owner_runtime_audit(
    runtime: Mapping[str, Any],
    *,
    optimizer_updates: int,
    clip_contract_schema: str,
    token_veto_tensor_count: int,
    global_absolute_tensor_count: int,
    revision_label: str,
) -> None:
    """Require complete, independently clipped two-owner evidence."""
    error = (
        f"{revision_label} confidence checkpoint lacks valid two-owner gradient/clip "
        "runtime evidence"
    )
    if not isinstance(runtime, Mapping) or type(optimizer_updates) is not int:
        raise RuntimeError(error)

    owners = {
        "token_veto": token_veto_tensor_count,
        "global_absolute": global_absolute_tensor_count,
    }
    violation_fields = (
        "owner_clip_violation_steps",
        "active_pre_decomposition_violation_steps",
        "active_post_decomposition_violation_steps",
        "live_tensor_count_violation_steps",
        "active_monotonic_violation_steps",
    )
    residual_fields = (
        "max_active_pre_decomposition_residual",
        "max_active_post_decomposition_residual",
        "max_owner_clip_residual",
        "max_active_monotonic_residual",
    )

    def _zero_counter(key: str) -> bool:
        value = runtime.get(key, 0)
        return type(value) is int and value == 0

    def _positive_finite(key: str) -> bool:
        value = runtime.get(key)
        return (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            and float(value) > 0.0
        )

    tolerance = runtime.get("clip_contract_tolerance")
    max_norm = runtime.get("clip_contract_max_norm")
    valid_tolerance = (
        not isinstance(tolerance, bool)
        and isinstance(tolerance, (int, float))
        and math.isfinite(float(tolerance))
        and float(tolerance) >= 0.0
    )
    valid_residuals = valid_tolerance and all(
        not isinstance(runtime.get(field), bool)
        and isinstance(runtime.get(field), (int, float))
        and math.isfinite(float(runtime[field]))
        and 0.0 <= float(runtime[field]) <= float(tolerance)
        for field in residual_fields
    )
    invalid_owner_evidence = any(
        not _positive_finite(f"last_{owner}_grad_norm_preclip")
        or not _positive_finite(f"max_{owner}_grad_norm_preclip")
        or not _zero_counter(f"nonfinite_{owner}_gradient_boundaries")
        or not _zero_counter(f"zero_{owner}_gradient_successful_steps")
        or type(runtime.get(f"expected_{owner}_tensor_count")) is not int
        or runtime.get(f"expected_{owner}_tensor_count") != count
        or type(runtime.get(f"last_observed_{owner}_tensor_count")) is not int
        or runtime.get(f"last_observed_{owner}_tensor_count") != count
        for owner, count in owners.items()
    )
    forbidden_owner_evidence = any(
        any(label in str(key) for label in (
            "candidate_absolute",
            "sample_calibrator",
            "deployed_router",
        ))
        for key in runtime
    )
    if (
        runtime.get("clip_contract_schema")
        != clip_contract_schema
        or type(runtime.get("clip_contract_checked_steps")) is not int
        or runtime.get("clip_contract_checked_steps") != optimizer_updates
        or any(not _zero_counter(field) for field in violation_fields)
        or isinstance(max_norm, bool)
        or not isinstance(max_norm, (int, float))
        or not math.isfinite(float(max_norm))
        or not math.isclose(float(max_norm), 0.1, rel_tol=0.0, abs_tol=1e-12)
        or not valid_residuals
        or invalid_owner_evidence
        or forbidden_owner_evidence
    ):
        raise RuntimeError(error)


def _validate_v53_two_owner_runtime_audit(
    runtime: Mapping[str, Any],
    *,
    optimizer_updates: int,
) -> None:
    """Require complete, independently clipped V53 owner evidence."""
    _validate_fulltext_two_owner_runtime_audit(
        runtime,
        optimizer_updates=optimizer_updates,
        clip_contract_schema=_V53_TWO_OWNER_CLIP_CONTRACT_SCHEMA,
        token_veto_tensor_count=_V53_TOKEN_VETO_TENSOR_COUNT,
        global_absolute_tensor_count=_V53_GLOBAL_ABSOLUTE_TENSOR_COUNT,
        revision_label="v53",
    )


def _validate_v54_two_owner_runtime_audit(
    runtime: Mapping[str, Any],
    *,
    optimizer_updates: int,
) -> None:
    """Require complete, independently clipped V54 owner evidence."""
    _validate_fulltext_two_owner_runtime_audit(
        runtime,
        optimizer_updates=optimizer_updates,
        clip_contract_schema=_V54_TWO_OWNER_CLIP_CONTRACT_SCHEMA,
        token_veto_tensor_count=_V54_TOKEN_VETO_TENSOR_COUNT,
        global_absolute_tensor_count=_V54_GLOBAL_ABSOLUTE_TENSOR_COUNT,
        revision_label="v54",
    )


def _validate_v55_two_owner_runtime_audit(
    runtime: Mapping[str, Any],
    *,
    optimizer_updates: int,
) -> None:
    """Require complete, independently clipped V55 owner evidence."""
    _validate_fulltext_two_owner_runtime_audit(
        runtime,
        optimizer_updates=optimizer_updates,
        clip_contract_schema=_V55_TWO_OWNER_CLIP_CONTRACT_SCHEMA,
        token_veto_tensor_count=_V55_TOKEN_VETO_TENSOR_COUNT,
        global_absolute_tensor_count=_V55_GLOBAL_ABSOLUTE_TENSOR_COUNT,
        revision_label="v55",
    )


def _validate_dense_duty_partial_confidence_diagnostic_checkpoint(
    payload: Mapping[str, Any],
    cfg,
    *,
    checkpoint_path: Path,
) -> Dict[str, Any]:
    """Admit an immutable partial snapshot or the fixed terminal U300 probe."""
    from util.stage_b_dense_duty_audit import (
        SOURCE_CLOSURE_ARG,
        audit_checkpoint_payload,
        build_code_source_closure,
        build_source_closure,
        validate_code_source_closure,
        validate_evaluation_checkpoint_payload,
        validate_source_closure,
        validate_strict_resume_checkpoint_payload,
    )

    resolved = checkpoint_path.expanduser().resolve(strict=True)
    immutable_v39_archived_diagnostic = bool(
        getattr(
            cfg,
            "stage_b_dense_duty_immutable_v39_archived_snapshot_diagnostic",
            False,
        )
    )
    immutable_v40_archived_diagnostic = bool(
        getattr(
            cfg,
            "stage_b_dense_duty_immutable_v40_archived_snapshot_diagnostic",
            False,
        )
    )
    immutable_v41_archived_diagnostic = bool(
        getattr(
            cfg,
            "stage_b_dense_duty_immutable_v41_archived_snapshot_diagnostic",
            False,
        )
    )
    immutable_v42_archived_diagnostic = bool(
        getattr(
            cfg,
            "stage_b_dense_duty_immutable_v42_archived_snapshot_diagnostic",
            False,
        )
    )
    if sum(
        (
            immutable_v39_archived_diagnostic,
            immutable_v40_archived_diagnostic,
            immutable_v41_archived_diagnostic,
            immutable_v42_archived_diagnostic,
        )
    ) > 1:
        raise RuntimeError(
            "v39, v40, v41, and v42 immutable archived diagnostics are mutually "
            "exclusive"
        )
    immutable_v39_context = (
        _prepare_v39_immutable_archived_diagnostic(
            payload,
            cfg,
            checkpoint_path=resolved,
        )
        if immutable_v39_archived_diagnostic
        else None
    )
    immutable_v40_context = (
        _prepare_v40_immutable_archived_diagnostic(
            payload,
            cfg,
            checkpoint_path=resolved,
        )
        if immutable_v40_archived_diagnostic
        else None
    )
    immutable_v41_context = (
        _prepare_v41_immutable_archived_diagnostic(
            payload,
            cfg,
            checkpoint_path=resolved,
        )
        if immutable_v41_archived_diagnostic
        else None
    )
    immutable_v42_context = (
        _prepare_v42_immutable_archived_diagnostic(
            payload,
            cfg,
            checkpoint_path=resolved,
        )
        if immutable_v42_archived_diagnostic
        else None
    )
    saved_args = payload.get("args")
    if not isinstance(saved_args, Mapping):
        raise RuntimeError("dense-duty confidence diagnostic lacks saved args")
    output_dir = saved_args.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise RuntimeError(
            "dense-duty confidence diagnostic lacks its training output_dir"
        )
    canonical = Path(output_dir).expanduser().resolve(strict=True) / "checkpoint_iter.pth"
    canonical = canonical.resolve(strict=True)
    snapshot_sha256 = _sha256_file(resolved)
    canonical_sha256 = _sha256_file(canonical)
    if immutable_v39_context is not None:
        expected_canonical = Path(
            _V39_IMMUTABLE_ARCHIVED_TERMINAL["path"]
        ).resolve(strict=True)
        if canonical != expected_canonical:
            raise RuntimeError(
                "v39 immutable archived diagnostic training output_dir does not "
                "point at the fixed terminal U400 checkpoint"
            )
    elif immutable_v40_context is not None:
        expected_canonical = Path(
            _V40_IMMUTABLE_ARCHIVED_TERMINAL["path"]
        ).resolve(strict=True)
        if canonical != expected_canonical:
            raise RuntimeError(
                "v40 immutable archived diagnostic training output_dir does not "
                "point at the fixed terminal U400 checkpoint"
            )
    elif immutable_v41_context is not None:
        expected_canonical = Path(
            _V41_IMMUTABLE_ARCHIVED_TERMINAL["path"]
        ).resolve(strict=True)
        if canonical != expected_canonical:
            raise RuntimeError(
                "v41 immutable archived diagnostic training output_dir does not "
                "point at the fixed terminal U400 checkpoint"
            )
    elif immutable_v42_context is not None:
        expected_canonical = Path(
            _V42_IMMUTABLE_ARCHIVED_TERMINAL["path"]
        ).resolve(strict=True)
        if canonical != expected_canonical:
            raise RuntimeError(
                "v42 immutable archived diagnostic training output_dir does not "
                "point at the fixed terminal U400 checkpoint"
            )
    elif snapshot_sha256 != canonical_sha256:
        raise RuntimeError(
            "dense-duty confidence diagnostic snapshot differs from the canonical "
            "strict-resume checkpoint"
        )

    audit = audit_checkpoint_payload(payload, checkpoint_path=resolved)
    expected_updates = int(
        getattr(cfg, "stage_b_dense_duty_confidence_expected_optimizer_updates", 0)
    )
    observed_updates = payload.get("optimizer_updates")
    evaluation_scope = str(
        getattr(cfg, "stage_b_dense_duty_evaluation_scope", "formal") or ""
    ).strip().lower()
    confidence_revision = str(
        getattr(cfg, "stage_b_dense_duty_confidence_revision", "")
    ).strip()
    phrase_aggregation = str(
        getattr(
            cfg,
            "stage_b_dense_duty_confidence_phrase_aggregation",
            "",
        )
    ).strip()
    word_veto_v4_revision_contract = (
        confidence_revision == "word_veto_coverage_absolute_cap_v4"
        and phrase_aggregation == "trace_activated_word_veto_absolute_cap_v4"
        and float(
            getattr(cfg, "stage_b_dense_duty_raw_veto_gate_weight", 0.0)
            or 0.0
        )
        > 0.0
    )
    word_veto_v5_revision_contract = (
        confidence_revision == "word_veto_gated_pool_absolute_cap_v5"
        and phrase_aggregation
        == "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        and float(
            getattr(cfg, "stage_b_dense_duty_raw_veto_gate_weight", 0.0)
            or 0.0
        )
        > 0.0
    )
    word_veto_v6_revision_contract = (
        confidence_revision == "word_veto_gated_pool_calibrated_v6"
        and phrase_aggregation
        == "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        and float(
            getattr(cfg, "stage_b_dense_duty_raw_veto_gate_weight", 0.0)
            or 0.0
        )
        > 0.0
    )
    word_veto_v7_revision_contract = (
        confidence_revision == "word_veto_gated_pool_carrier_balanced_v7"
        and phrase_aggregation
        == "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        and float(
            getattr(cfg, "stage_b_dense_duty_raw_veto_gate_weight", 0.0)
            or 0.0
        )
        > 0.0
    )
    word_veto_v8_revision_contract = (
        confidence_revision == "word_veto_gated_pool_carrier_quarter_v8"
        and phrase_aggregation
        == "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        and float(
            getattr(cfg, "stage_b_dense_duty_raw_veto_gate_weight", 0.0)
            or 0.0
        )
        > 0.0
    )
    word_veto_v9_revision_contract = (
        confidence_revision == "word_veto_gated_pool_carrier_pair_v9"
        and phrase_aggregation
        == "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        and float(
            getattr(cfg, "stage_b_dense_duty_raw_veto_gate_weight", 0.0)
            or 0.0
        )
        > 0.0
    )
    word_veto_v15_revision_contract = (
        confidence_revision == "word_veto_gated_pool_carrier_affine_v15"
        and phrase_aggregation
        == "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        and float(
            getattr(cfg, "stage_b_dense_duty_raw_veto_gate_weight", 0.0)
            or 0.0
        )
        > 0.0
    )
    word_veto_v17_revision_contract = (
        confidence_revision == "word_veto_gated_pool_tail_carrier_v17"
        and phrase_aggregation
        == "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        and float(
            getattr(cfg, "stage_b_dense_duty_raw_veto_gate_weight", 0.0)
            or 0.0
        )
        > 0.0
    )
    word_veto_v18_revision_contract = (
        confidence_revision == "word_veto_gated_pool_tail_paired_v18"
        and phrase_aggregation
        == "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        and float(
            getattr(cfg, "stage_b_dense_duty_raw_veto_gate_weight", 0.0)
            or 0.0
        )
        > 0.0
    )
    word_veto_v19_revision_contract = (
        confidence_revision
        == "word_veto_gated_pool_tail_paired_rank_channel_v19"
        and phrase_aggregation
        == "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        and float(
            getattr(cfg, "stage_b_dense_duty_raw_veto_gate_weight", 0.0)
            or 0.0
        )
        > 0.0
    )
    word_veto_v20_revision_contract = (
        confidence_revision
        == "word_veto_gated_pool_tail_paired_signed_rank_pool_v20"
        and phrase_aggregation
        == "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        and float(
            getattr(cfg, "stage_b_dense_duty_raw_veto_gate_weight", 0.0)
            or 0.0
        )
        > 0.0
    )
    word_veto_v21_revision_contract = (
        confidence_revision
        in {
            "word_veto_continuous_conditional_residual_v21",
            "word_veto_continuous_monotone_depth_v22",
            "word_veto_token_conditioned_monotone_depth_v23",
            "word_veto_complementary_trust_veto_v24",
            "word_veto_ungated_monotone_tail_veto_v25",
            "word_veto_floor_gated_monotone_tail_veto_v26",
            "word_veto_independent_absolute_confidence_v27",
            "word_veto_cross_attention_absolute_confidence_v28",
            "word_veto_candidate_absolute_confidence_v29",
            "word_veto_candidate_patch_invariant_confidence_v30",
            "word_veto_candidate_normalized_confidence_v31",
            "word_veto_candidate_asymmetric_confidence_v32",
            "word_veto_candidate_set_attention_confidence_v33",
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
        }
        and phrase_aggregation
        == "trace_activated_word_veto_gated_pool_absolute_cap_v5"
        and float(
            getattr(cfg, "stage_b_dense_duty_raw_veto_gate_weight", 0.0)
            or 0.0
        )
        > 0.0
    )
    word_veto_v22_revision_contract = (
        confidence_revision
        in {
            "word_veto_continuous_monotone_depth_v22",
            "word_veto_token_conditioned_monotone_depth_v23",
        }
    )
    word_veto_v23_revision_contract = (
        confidence_revision
        == "word_veto_token_conditioned_monotone_depth_v23"
    )
    word_veto_v24_revision_contract = (
        confidence_revision == "word_veto_complementary_trust_veto_v24"
    )
    word_veto_v25_revision_contract = (
        confidence_revision == "word_veto_ungated_monotone_tail_veto_v25"
    )
    word_veto_v26_revision_contract = (
        confidence_revision
        == "word_veto_floor_gated_monotone_tail_veto_v26"
    )
    word_veto_v27_revision_contract = (
        confidence_revision
        == "word_veto_independent_absolute_confidence_v27"
    )
    word_veto_v28_revision_contract = (
        confidence_revision
        == "word_veto_cross_attention_absolute_confidence_v28"
    )
    word_veto_v29_revision_contract = (
        confidence_revision
        == "word_veto_candidate_absolute_confidence_v29"
    )
    word_veto_v30_revision_contract = (
        confidence_revision
        == "word_veto_candidate_patch_invariant_confidence_v30"
    )
    word_veto_v31_revision_contract = (
        confidence_revision == "word_veto_candidate_normalized_confidence_v31"
    )
    word_veto_v32_revision_contract = (
        confidence_revision == "word_veto_candidate_asymmetric_confidence_v32"
    )
    word_veto_v43_revision_contract = _validate_v43_deployed_routing_config(cfg)
    word_veto_v45_revision_contract = _validate_v45_split_tail_aligned_config(cfg)
    word_veto_v46_revision_contract = _validate_v46_split_positive_tail_config(cfg)
    word_veto_v47_revision_contract = _validate_v47_split_boundary_routing_config(cfg)
    word_veto_v48_revision_contract = _validate_v48_split_fpr_active_set_config(cfg)
    word_veto_v49_revision_contract = _validate_v49_split_global_trust_veto_config(cfg)
    word_veto_v50_revision_contract = (
        _validate_v50_split_strong_boundary_routing_config(cfg)
    )
    word_veto_v51_revision_contract = (
        _validate_v51_split_independent_deployed_router_config(cfg)
    )
    word_veto_v52_revision_contract = (
        _validate_v52_candidate_sample_calibrator_config(cfg)
    )
    word_veto_v53_revision_contract = (
        _validate_v53_fulltext_global_absolute_config(cfg)
    )
    word_veto_v54_revision_contract = (
        _validate_v54_fulltext_global_absolute_exact_residual_config(cfg)
    )
    word_veto_v55_revision_contract = (
        _validate_v55_fulltext_global_independent_absolute_config(cfg)
    )
    word_veto_deployed_routing_revision_contract = (
        word_veto_v43_revision_contract
        or word_veto_v45_revision_contract
        or word_veto_v46_revision_contract
        or word_veto_v47_revision_contract
        or word_veto_v48_revision_contract
        or word_veto_v49_revision_contract
        or word_veto_v50_revision_contract
        or word_veto_v51_revision_contract
    )
    if word_veto_v50_revision_contract:
        saved_training_contract = saved_args.get(
            "stage_b_dense_duty_training_contract"
        )
        if (
            not isinstance(saved_training_contract, Mapping)
            or saved_training_contract.get("schema")
            != _V50_SPLIT_STRONG_BOUNDARY_ROUTING_TRAINING_CONTRACT_SCHEMA
            or not isinstance(saved_training_contract.get("values"), Mapping)
        ):
            raise RuntimeError(
                "v50 confidence checkpoint requires its exact v32 training contract"
            )
    if word_veto_v51_revision_contract:
        saved_training_contract = saved_args.get(
            "stage_b_dense_duty_training_contract"
        )
        if (
            not isinstance(saved_training_contract, Mapping)
            or saved_training_contract.get("schema")
            != _V51_SPLIT_INDEPENDENT_DEPLOYED_ROUTER_TRAINING_CONTRACT_SCHEMA
            or not isinstance(saved_training_contract.get("values"), Mapping)
        ):
            raise RuntimeError(
                "v51 confidence checkpoint requires its exact v33 training contract"
            )
    if word_veto_v52_revision_contract:
        saved_training_contract = saved_args.get(
            "stage_b_dense_duty_training_contract"
        )
        if (
            not isinstance(saved_training_contract, Mapping)
            or saved_training_contract.get("schema")
            != _V52_CANDIDATE_SAMPLE_CALIBRATOR_TRAINING_CONTRACT_SCHEMA
            or not isinstance(saved_training_contract.get("values"), Mapping)
        ):
            raise RuntimeError(
                "v52 confidence checkpoint requires its exact v34 training contract"
            )
        migration_audit = saved_args.get(
            "stage_b_dense_duty_confidence_adapter_migration_audit"
        )
        if (
            not isinstance(migration_audit, Mapping)
            or migration_audit.get("schema")
            != _V52_CANDIDATE_SAMPLE_CALIBRATOR_MIGRATION_SCHEMA
            or migration_audit.get("source_optimizer_updates")
            != _V52_CANDIDATE_SAMPLE_CALIBRATOR_SOURCE_UPDATES
            or migration_audit.get("head_gradient_contract")
            != _V52_CANDIDATE_SAMPLE_CALIBRATOR_HEAD_CONTRACT
        ):
            raise RuntimeError(
                "v52 confidence checkpoint requires the exact fresh-U6551 "
                "candidate/sample-calibrator migration audit"
            )
        model_state = payload.get("model")
        if not isinstance(model_state, Mapping) or any(
            "deployed_router" in str(name) for name in model_state
        ):
            raise RuntimeError(
                "v52 confidence checkpoint must contain a model state without "
                "deployed-router parameters"
            )
    if word_veto_v53_revision_contract:
        saved_training_contract = saved_args.get(
            "stage_b_dense_duty_training_contract"
        )
        if (
            not isinstance(saved_training_contract, Mapping)
            or saved_training_contract.get("schema")
            != _V53_FULLTEXT_GLOBAL_ABSOLUTE_TRAINING_CONTRACT_SCHEMA
            or not isinstance(saved_training_contract.get("values"), Mapping)
        ):
            raise RuntimeError(
                "v53 confidence checkpoint requires its exact v35 training contract"
            )
        migration_audit = saved_args.get(
            "stage_b_dense_duty_confidence_adapter_migration_audit"
        )
        if (
            not isinstance(migration_audit, Mapping)
            or migration_audit.get("schema")
            != _V53_FULLTEXT_GLOBAL_ABSOLUTE_MIGRATION_SCHEMA
            or migration_audit.get("source_optimizer_updates")
            != _V53_FULLTEXT_GLOBAL_ABSOLUTE_SOURCE_UPDATES
            or migration_audit.get("fresh_confidence_contract")
            != _V53_FULLTEXT_GLOBAL_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT
            or migration_audit.get("head_gradient_contract")
            != _V53_FULLTEXT_GLOBAL_ABSOLUTE_HEAD_CONTRACT
            or migration_audit.get("pool_feature_contract")
            != _V53_FULLTEXT_GLOBAL_ABSOLUTE_POOL_FEATURE_CONTRACT
        ):
            raise RuntimeError(
                "v53 confidence checkpoint requires the exact fresh-U6551 "
                "fulltext/global-absolute migration audit"
            )
        forbidden_parameter_fragments = (
            "deployed_router",
            "patch_residual",
            "global_query_norm",
            "veto_cap_raw_ceiling",
            "candidate_patch_scale_raw",
            "candidate_veto_depth_raw",
            "candidate_coverage_depth_raw",
        )
        model_state = payload.get("model")
        if not isinstance(model_state, Mapping) or any(
            any(fragment in str(name) for fragment in forbidden_parameter_fragments)
            for name in model_state
        ):
            raise RuntimeError(
                "v53 confidence checkpoint must contain only its complete "
                "two-owner parameter surface"
            )
    if word_veto_v54_revision_contract:
        saved_training_contract = saved_args.get(
            "stage_b_dense_duty_training_contract"
        )
        if (
            not isinstance(saved_training_contract, Mapping)
            or saved_training_contract.get("schema")
            != _V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_TRAINING_CONTRACT_SCHEMA
            or not isinstance(saved_training_contract.get("values"), Mapping)
        ):
            raise RuntimeError(
                "v54 confidence checkpoint requires its exact v36 training contract"
            )
        migration_audit = saved_args.get(
            "stage_b_dense_duty_confidence_adapter_migration_audit"
        )
        if (
            not isinstance(migration_audit, Mapping)
            or migration_audit.get("schema")
            != _V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_MIGRATION_SCHEMA
            or migration_audit.get("source_optimizer_updates")
            != _V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_SOURCE_UPDATES
            or migration_audit.get("fresh_confidence_contract")
            != _V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_FRESH_CONFIDENCE_CONTRACT
            or migration_audit.get("head_gradient_contract")
            != _V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_HEAD_CONTRACT
            or migration_audit.get("pool_feature_contract")
            != _V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_POOL_FEATURE_CONTRACT
        ):
            raise RuntimeError(
                "v54 confidence checkpoint requires the exact fresh-U6551 "
                "fulltext/global-absolute exact-residual migration audit"
            )
        forbidden_parameter_fragments = (
            "deployed_router",
            "patch_residual",
            "global_query_norm",
            "veto_cap_raw_ceiling",
            "candidate_patch_scale_raw",
            "candidate_veto_depth_raw",
            "candidate_coverage_depth_raw",
        )
        model_state = payload.get("model")
        if not isinstance(model_state, Mapping) or any(
            any(fragment in str(name) for fragment in forbidden_parameter_fragments)
            for name in model_state
        ):
            raise RuntimeError(
                "v54 confidence checkpoint must contain only its complete "
                "two-owner parameter surface"
            )
    if word_veto_v55_revision_contract:
        saved_training_contract = saved_args.get(
            "stage_b_dense_duty_training_contract"
        )
        if (
            not isinstance(saved_training_contract, Mapping)
            or saved_training_contract.get("schema")
            != _V55_FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_TRAINING_CONTRACT_SCHEMA
            or not isinstance(saved_training_contract.get("values"), Mapping)
        ):
            raise RuntimeError(
                "v55 confidence checkpoint requires its exact v37 training contract"
            )
        migration_audit = saved_args.get(
            "stage_b_dense_duty_confidence_adapter_migration_audit"
        )
        if (
            not isinstance(migration_audit, Mapping)
            or migration_audit.get("schema")
            != _V55_FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_MIGRATION_SCHEMA
            or migration_audit.get("source_optimizer_updates")
            != _V55_FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_SOURCE_UPDATES
            or migration_audit.get("fresh_confidence_contract")
            != _V55_FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_FRESH_CONFIDENCE_CONTRACT
            or migration_audit.get("head_gradient_contract")
            != _V55_FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_HEAD_CONTRACT
            or migration_audit.get("pool_feature_contract")
            != _V55_FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_POOL_FEATURE_CONTRACT
        ):
            raise RuntimeError(
                "v55 confidence checkpoint requires the exact fresh-U6551 "
                "fulltext/global-independent-absolute migration audit"
            )
        forbidden_parameter_fragments = (
            "deployed_router",
            "patch_residual",
            "global_query_norm",
            "veto_cap_raw_ceiling",
            "candidate_patch_scale_raw",
            "candidate_veto_depth_raw",
            "candidate_coverage_depth_raw",
        )
        model_state = payload.get("model")
        if not isinstance(model_state, Mapping) or any(
            any(fragment in str(name) for fragment in forbidden_parameter_fragments)
            for name in model_state
        ):
            raise RuntimeError(
                "v55 confidence checkpoint must contain only its complete "
                "two-owner parameter surface"
            )
    token_edit_query_scope = str(
        getattr(cfg, "stage_b_v21_token_edit_query_scope", "target_iou_v1")
    ).strip().lower()
    if token_edit_query_scope not in {
        "target_iou_v1",
        "target_iou_union_detached_final_confidence_base_argmax_v2",
        "target_iou_union_detached_role_complete_confidence_base_argmax_v3",
    }:
        raise RuntimeError(
            "dense-duty confidence diagnostic has an unknown token edit-query scope"
        )
    if token_edit_query_scope in {
        "target_iou_union_detached_final_confidence_base_argmax_v2",
        "target_iou_union_detached_role_complete_confidence_base_argmax_v3",
    } and (
        not word_veto_v32_revision_contract
        or str(
            getattr(
                cfg,
                "stage_b_v15_tail_queue_positive_gradient_contract",
                "",
            )
        ).strip()
        != "elementwise_bounded_mean_plus_sixteenth_exact_lower_tail_st_v6"
        or float(
            getattr(
                cfg,
                "stage_b_dense_duty_confidence_veto_gate_offset",
                -1.0,
            )
        )
        != 0.0
    ):
        raise RuntimeError(
            "carrier token diagnostics require the exact v40/v41 confidence "
            "surface"
        )
    word_veto_v33_revision_contract = (
        confidence_revision
        == "word_veto_candidate_set_attention_confidence_v33"
    )
    terminal_only_revision_contract = (
        word_veto_v15_revision_contract
        or word_veto_v17_revision_contract
        or word_veto_v18_revision_contract
        or word_veto_v19_revision_contract
        or word_veto_v20_revision_contract
        or word_veto_v21_revision_contract
    )
    fixed_u50_revision_contract = (
        terminal_only_revision_contract and expected_updates == 50
    )
    fixed_u100_revision_contract = (
        word_veto_v18_revision_contract and expected_updates == 100
    )
    fixed_u300_revision_contract = (
        (
            word_veto_v23_revision_contract
            or word_veto_v24_revision_contract
            or word_veto_v25_revision_contract
            or word_veto_v26_revision_contract
            or word_veto_v27_revision_contract
            or word_veto_v28_revision_contract
            or word_veto_v29_revision_contract
            or word_veto_v30_revision_contract
            or word_veto_v31_revision_contract
            or word_veto_v32_revision_contract
            or word_veto_v33_revision_contract
        )
        and expected_updates == 300
    )
    fixed_u600_revision_contract = (
        word_veto_v29_revision_contract and expected_updates == 600
    )
    fixed_u400_revision_contract = (
        (
            word_veto_v30_revision_contract
            or word_veto_v31_revision_contract
            or word_veto_v32_revision_contract
            or word_veto_v33_revision_contract
            or word_veto_deployed_routing_revision_contract
            or word_veto_v52_revision_contract
            or word_veto_v53_revision_contract
            or word_veto_v54_revision_contract
            or word_veto_v55_revision_contract
        )
        and expected_updates == 400
    )
    fixed_terminal_revision_contract = (
        fixed_u50_revision_contract
        or fixed_u100_revision_contract
        or fixed_u300_revision_contract
        or fixed_u400_revision_contract
        or fixed_u600_revision_contract
    )
    word_veto_absolute_cap_revision_contract = (
        word_veto_v4_revision_contract
        or word_veto_v5_revision_contract
        or word_veto_v6_revision_contract
        or word_veto_v7_revision_contract
        or word_veto_v8_revision_contract
        or word_veto_v9_revision_contract
        or word_veto_v15_revision_contract
        or word_veto_v17_revision_contract
        or word_veto_v18_revision_contract
        or word_veto_v19_revision_contract
        or word_veto_v20_revision_contract
        or word_veto_v21_revision_contract
    )
    word_veto_revision_contract = (
        (
            confidence_revision == "word_veto_net_trust_v1"
            and phrase_aggregation == "trace_activated_word_veto_product_v1"
        )
        or (
            confidence_revision == "word_veto_raw_gate_margin_v3"
            and phrase_aggregation == "trace_activated_word_veto_penalty_v2"
            and float(
                getattr(cfg, "stage_b_dense_duty_raw_veto_gate_weight", 0.0)
                or 0.0
            )
            > 0.0
        )
        or word_veto_absolute_cap_revision_contract
    )
    expected_absolute_cap_scope = (
        "tn_all_admitted_tail_weighted_carrier_tail_paired_v7"
        if (
            word_veto_v18_revision_contract
            or word_veto_v19_revision_contract
            or word_veto_v20_revision_contract
            or word_veto_v21_revision_contract
        )
        else (
            "tn_all_admitted_tail_weighted_carrier_positive_carrier_paired_v6"
            if word_veto_v17_revision_contract
            else (
                "tn_all_admitted_carrier_balanced_positive_carrier_paired_v4"
                if word_veto_v9_revision_contract or word_veto_v15_revision_contract
                else (
                    "tn_all_admitted_carrier_balanced_positive_carrier_v3"
                    if word_veto_v7_revision_contract or word_veto_v8_revision_contract
                    else "tn_all_admitted_positive_carrier_v2"
                )
            )
        )
    )
    calibrated_carrier_gate = (
        word_veto_v6_revision_contract
        or word_veto_v7_revision_contract
        or word_veto_v8_revision_contract
        or word_veto_v9_revision_contract
        or word_veto_v15_revision_contract
        or word_veto_v17_revision_contract
        or word_veto_v18_revision_contract
        or word_veto_v19_revision_contract
        or word_veto_v20_revision_contract
        or word_veto_v21_revision_contract
    )
    word_veto_absolute_cap_probe_contract = (
        word_veto_absolute_cap_revision_contract
        and float(
            getattr(cfg, "stage_b_dense_duty_raw_veto_gate_weight", 0.0)
            or 0.0
        )
        == 1.0
        and float(
            getattr(cfg, "stage_b_dense_duty_raw_veto_positive_margin", -1.0)
        )
        == 0.1
        and float(
            getattr(cfg, "stage_b_dense_duty_raw_veto_tn_margin", -1.0)
        )
        == 0.15
        and str(
            getattr(cfg, "stage_b_dense_duty_raw_veto_query_scope", "")
        ).strip()
        == expected_absolute_cap_scope
        and float(
            getattr(cfg, "stage_b_dense_duty_confidence_veto_gate_offset", -1.0)
        )
        in ({0.0, 0.02} if calibrated_carrier_gate else {0.05})
        and float(
            getattr(cfg, "stage_b_dense_duty_confidence_veto_gate_scale", -1.0)
        )
        == (0.03 if calibrated_carrier_gate else 0.1)
        and float(
            getattr(
                cfg,
                "stage_b_dense_duty_confidence_veto_coverage_offset",
                -1.0,
            )
        )
        == 0.1
        and float(
            getattr(
                cfg,
                "stage_b_dense_duty_confidence_veto_coverage_ramp",
                -1.0,
            )
        )
        == 0.8
        and float(
            getattr(
                cfg,
                "stage_b_dense_duty_confidence_veto_cap_temperature",
                -1.0,
            )
        )
        == 0.1
        and float(
            getattr(
                cfg,
                "stage_b_dense_duty_confidence_veto_cap_initial_ceiling",
                0.0,
            )
        )
        == -0.1
        and (
            not (
                word_veto_v7_revision_contract
                or word_veto_v8_revision_contract
                or word_veto_v9_revision_contract
                or word_veto_v15_revision_contract
                or word_veto_v17_revision_contract
                or word_veto_v18_revision_contract
                or word_veto_v19_revision_contract
                or word_veto_v20_revision_contract
                or word_veto_v21_revision_contract
            )
            or (
                float(
                    getattr(
                        cfg,
                        "stage_b_dense_duty_raw_veto_tn_carrier_balance",
                        -1.0,
                    )
                )
                == (
                    0.25
                    if word_veto_v8_revision_contract
                    or word_veto_v9_revision_contract
                    or word_veto_v15_revision_contract
                    or word_veto_v17_revision_contract
                    or word_veto_v18_revision_contract
                    or word_veto_v19_revision_contract
                    or word_veto_v20_revision_contract
                    or word_veto_v21_revision_contract
                    else 0.5
                )
                and str(
                    getattr(
                        cfg,
                        "stage_b_dense_duty_confidence_carrier_selector_contract",
                        "",
                    )
                ).strip()
                == "final_layer_reference_argmax_exact_eligible_v1"
                and (
                    not (
                        word_veto_v9_revision_contract
                        or word_veto_v15_revision_contract
                        or word_veto_v17_revision_contract
                        or word_veto_v18_revision_contract
                        or word_veto_v19_revision_contract
                        or word_veto_v20_revision_contract
                        or word_veto_v21_revision_contract
                    )
                    or (
                        float(
                            getattr(
                                cfg,
                                "stage_b_dense_duty_raw_veto_carrier_pair_weight",
                                -1.0,
                            )
                        )
                        == 0.25
                        and float(
                            getattr(
                                cfg,
                                "stage_b_dense_duty_raw_veto_carrier_pair_margin",
                                -1.0,
                            )
                        )
                        == 0.25
                    )
                )
            )
        )
        and (
            not (
                word_veto_v15_revision_contract
                or word_veto_v17_revision_contract
                or word_veto_v18_revision_contract
                or word_veto_v19_revision_contract
                or word_veto_v20_revision_contract
                or word_veto_v21_revision_contract
            )
            or (
                str(
                    getattr(
                        cfg,
                        "stage_b_dense_duty_confidence_rank_evidence_contract",
                        "",
                    )
                ).strip()
                == (
                    "zero_init_carrier_token_rank_affine_sparse_rank_channel_v6"
                    if (
                        word_veto_v19_revision_contract
                        or word_veto_v20_revision_contract
                        or word_veto_v21_revision_contract
                    )
                    else "zero_init_carrier_token_rank_affine_v5"
                )
                and float(
                    getattr(
                        cfg,
                        "stage_b_dense_duty_confidence_residual_parameterization_gain",
                        -1.0,
                    )
                )
                == (25.0 / 3.0)
            )
        )
        and (
            not (
                word_veto_v17_revision_contract
                or word_veto_v18_revision_contract
                or word_veto_v19_revision_contract
                or word_veto_v20_revision_contract
                or word_veto_v21_revision_contract
            )
            or (
                str(
                    getattr(
                        cfg,
                        "stage_b_dense_duty_confidence_gate_gradient_contract",
                        "",
                    )
                ).strip()
                == (
                    (
                        "continuous_sigmoid_complementary_trust_veto_v5"
                        if word_veto_v24_revision_contract
                        else (
                            "token_conditioned_floor_gated_monotone_depth_v7"
                            if word_veto_v26_revision_contract
                            else (
                                (
                                    _V43_DEPLOYED_ROUTING_GATE_CONTRACT
                                    if (
                                        word_veto_deployed_routing_revision_contract
                                        or word_veto_v52_revision_contract
                                    )
                                    else (
                                        "candidate_set_attention_asymmetric_monotone_veto_absolute_logit_v14"
                                        if word_veto_v33_revision_contract
                                        else (
                                            "candidate_raw_patch_asymmetric_monotone_veto_absolute_logit_v13"
                                            if (
                                                word_veto_v32_revision_contract
                                                or word_veto_v53_revision_contract
                                                or word_veto_v54_revision_contract
                                                or word_veto_v55_revision_contract
                                            )
                                            else (
                                                "candidate_normalized_patch_amplified_monotone_veto_absolute_logit_v12"
                                                if word_veto_v31_revision_contract
                                                else (
                                                    "candidate_patch_invariant_monotone_veto_absolute_logit_v11"
                                                    if word_veto_v30_revision_contract
                                                    else (
                                                        "candidate_cross_attention_independent_absolute_logit_v10"
                                                        if word_veto_v29_revision_contract
                                                        else (
                                                            "cross_attention_independent_absolute_logit_v9"
                                                            if word_veto_v28_revision_contract
                                                            else "token_conditioned_independent_absolute_logit_v8"
                                                        )
                                                    )
                                                )
                                            )
                                        )
                                    )
                                )
                                if (
                                    word_veto_v27_revision_contract
                                    or word_veto_v28_revision_contract
                                    or word_veto_v29_revision_contract
                                    or word_veto_v30_revision_contract
                                    or word_veto_v31_revision_contract
                                    or word_veto_v32_revision_contract
                                    or word_veto_v33_revision_contract
                                    or word_veto_deployed_routing_revision_contract
                                    or word_veto_v52_revision_contract
                                    or word_veto_v53_revision_contract
                                    or word_veto_v54_revision_contract
                                    or word_veto_v55_revision_contract
                                )
                                else "token_conditioned_ungated_monotone_depth_v6"
                            )
                        )
                    )
                    if (
                        word_veto_v24_revision_contract
                        or word_veto_v25_revision_contract
                        or word_veto_v26_revision_contract
                        or word_veto_v27_revision_contract
                        or word_veto_v28_revision_contract
                        or word_veto_v29_revision_contract
                        or word_veto_v30_revision_contract
                        or word_veto_v31_revision_contract
                        or word_veto_v32_revision_contract
                        or word_veto_v33_revision_contract
                        or word_veto_deployed_routing_revision_contract
                        or word_veto_v52_revision_contract
                        or word_veto_v53_revision_contract
                        or word_veto_v54_revision_contract
                        or word_veto_v55_revision_contract
                    )
                    else (
                        "continuous_sigmoid_monotone_depth_v4"
                        if word_veto_v22_revision_contract
                        else (
                            "continuous_sigmoid_v3"
                            if word_veto_v21_revision_contract
                            else "hard_detached_v1"
                        )
                    )
                )
                and float(
                    getattr(
                        cfg,
                        "stage_b_dense_duty_raw_veto_tail_quantile",
                        -1.0,
                    )
                )
                == 0.95
                and float(
                    getattr(
                        cfg,
                        "stage_b_dense_duty_raw_veto_tail_temperature",
                        -1.0,
                    )
                )
                == 0.1
                and int(
                    getattr(
                        cfg,
                        "stage_b_dense_duty_raw_veto_tail_min_count",
                        -1,
                    )
                )
                == 256
            )
        )
    )
    word_veto_probe_contract = (
        evaluation_scope == "probe"
        and (
            fixed_terminal_revision_contract
            or (
                not terminal_only_revision_contract
                and expected_updates == 300
            )
        )
        and word_veto_revision_contract
        and str(
        getattr(cfg, "stage_b_dense_duty_positive_trust_contract", "")
    ).strip()
    == (
        "absolute_global_pool_logit_v4"
        if word_veto_v55_revision_contract
        else (
            "exact_frozen_rank_max_confidence_delta_v3"
            if word_veto_v54_revision_contract
            else (
                "absolute_global_confidence_logit_v2"
                if (
                    word_veto_v27_revision_contract
                    or word_veto_v28_revision_contract
                    or word_veto_v29_revision_contract
                    or word_veto_v30_revision_contract
                    or word_veto_v31_revision_contract
                    or word_veto_v32_revision_contract
                    or word_veto_v33_revision_contract
                    or word_veto_deployed_routing_revision_contract
                    or word_veto_v52_revision_contract
                    or word_veto_v53_revision_contract
                )
                else "net_total_confidence_delta_v1"
            )
        )
    )
        and str(
            getattr(cfg, "stage_b_dense_duty_confidence_tn_scope", "")
        ).strip()
        == "direct_trace_valid_v1"
        and float(
            getattr(
                cfg,
                "stage_b_dense_duty_confidence_word_softmin_temperature",
                -1.0,
            )
        )
        == 0.1
        and (
            word_veto_absolute_cap_probe_contract
            or (
                not word_veto_absolute_cap_revision_contract
                and float(
                    getattr(
                        cfg,
                        "stage_b_dense_duty_confidence_veto_gate_scale",
                        -1.0,
                    )
                )
                == 1.0
            )
        )
        and bool(getattr(cfg, "stage_b_v15_exclude_canonical_from_score", False))
    )
    terminal_probe = (
        word_veto_probe_contract
        and observed_updates == expected_updates
        and saved_args.get("max_train_iters") == expected_updates
        and payload.get("checkpoint_reason") == "max_train_iters"
    )
    partial_checkpoint = (
        isinstance(observed_updates, int)
        and not isinstance(observed_updates, bool)
        and (
            not terminal_only_revision_contract
            or (
                (
                    word_veto_v29_revision_contract
                    and expected_updates == 600
                )
                or (
                    word_veto_v30_revision_contract
                    and expected_updates == 400
                )
                or (
                    word_veto_v31_revision_contract
                    and expected_updates == 400
                )
                or (
                    word_veto_v32_revision_contract
                    and expected_updates == 400
                )
                or (
                    word_veto_v33_revision_contract
                    and expected_updates == 400
                )
                or (
                    (
                        word_veto_deployed_routing_revision_contract
                        or word_veto_v52_revision_contract
                        or word_veto_v53_revision_contract
                        or word_veto_v54_revision_contract
                        or word_veto_v55_revision_contract
                    )
                    and expected_updates == 400
                )
            )
        )
        and 0 < observed_updates < expected_updates
        and saved_args.get("max_train_iters") == expected_updates
        and payload.get("checkpoint_reason") in {"signal", "interval"}
    )
    immutable_archived_context = any(
        context is not None
        for context in (
            immutable_v39_context,
            immutable_v40_context,
            immutable_v41_context,
            immutable_v42_context,
        )
    )
    if terminal_probe:
        from types import SimpleNamespace

        cfg_values = getattr(cfg, "_cfg_dict", None)
        terminal_cfg = (
            SimpleNamespace(
                **dict(cfg_values),
                _filename=getattr(cfg, "_filename", ""),
            )
            if isinstance(cfg_values, Mapping)
            else cfg
        )
        resume = validate_evaluation_checkpoint_payload(
            payload,
            terminal_cfg,
            checkpoint_path=canonical,
        )
        resume.update(
            {
                "phase": audit.get("phase"),
                "optimizer_updates": observed_updates,
                "checkpoint_reason": payload.get("checkpoint_reason"),
            }
        )
    else:
        resume_args = saved_args
        resume_checkpoint = canonical
        if immutable_archived_context:
            resume_args = dict(saved_args)
            resume_args["output_dir"] = str(resolved.parent)
            resume_checkpoint = resolved
        resume = validate_strict_resume_checkpoint_payload(
            payload,
            resume_args,
            checkpoint_path=resume_checkpoint,
        )
    if (
        audit.get("status") != "passed"
        or audit.get("phase") != "confidence"
        or resume.get("phase") != "confidence"
        or isinstance(observed_updates, bool)
        or not isinstance(observed_updates, int)
        or expected_updates <= 0
        or not (partial_checkpoint or terminal_probe)
        or resume.get("optimizer_updates") != observed_updates
        or resume.get("checkpoint_reason") != payload.get("checkpoint_reason")
    ):
        terminal_revision_label = {
            "word_veto_gated_pool_carrier_affine_v15": "v15",
            "word_veto_gated_pool_tail_carrier_v17": "v17",
            "word_veto_gated_pool_tail_paired_v18": "v18",
            "word_veto_gated_pool_tail_paired_rank_channel_v19": "v19",
            "word_veto_gated_pool_tail_paired_signed_rank_pool_v20": "v20",
            "word_veto_continuous_conditional_residual_v21": "v21",
            "word_veto_continuous_monotone_depth_v22": "v22",
            "word_veto_token_conditioned_monotone_depth_v23": "v23",
            "word_veto_complementary_trust_veto_v24": "v24",
            "word_veto_ungated_monotone_tail_veto_v25": "v25",
            "word_veto_floor_gated_monotone_tail_veto_v26": "v26",
            "word_veto_independent_absolute_confidence_v27": "v27",
            "word_veto_cross_attention_absolute_confidence_v28": "v28",
            "word_veto_candidate_absolute_confidence_v29": "v29",
            "word_veto_candidate_patch_invariant_confidence_v30": "v30",
            "word_veto_candidate_normalized_confidence_v31": "v31",
            "word_veto_candidate_asymmetric_confidence_v32": "v32",
            "word_veto_candidate_set_attention_confidence_v33": "v33",
            _V43_DEPLOYED_ROUTING_REVISION: "v43",
            _V45_SPLIT_TAIL_ALIGNED_REVISION: "v45",
            _V46_SPLIT_POSITIVE_TAIL_REVISION: "v46",
            _V47_SPLIT_BOUNDARY_ROUTING_REVISION: "v47",
            _V48_SPLIT_FPR_ACTIVE_SET_REVISION: "v48",
            _V49_SPLIT_GLOBAL_TRUST_VETO_REVISION: "v49",
            _V50_SPLIT_STRONG_BOUNDARY_ROUTING_REVISION: "v50",
            _V51_SPLIT_INDEPENDENT_DEPLOYED_ROUTER_REVISION: "v51",
            _V52_CANDIDATE_SAMPLE_CALIBRATOR_REVISION: "v52",
            _V53_FULLTEXT_GLOBAL_ABSOLUTE_REVISION: "v53",
            _V54_FULLTEXT_GLOBAL_ABSOLUTE_EXACT_RESIDUAL_REVISION: "v54",
            _V55_FULLTEXT_GLOBAL_INDEPENDENT_ABSOLUTE_REVISION: "v55",
        }.get(confidence_revision)
        terminal_probe_label = (
            f"terminal U{expected_updates} {terminal_revision_label} probe"
            if terminal_revision_label is not None
            else "terminal U300 probe"
        )
        raise RuntimeError(
            "dense-duty partial confidence diagnostic requires a valid "
            f"non-terminal confidence checkpoint or {terminal_probe_label}"
        )

    if evaluation_scope not in {"formal", "probe"} or saved_args.get(
        "stage_b_dense_duty_execution_scope"
    ) != evaluation_scope:
        raise RuntimeError(
            "dense-duty confidence diagnostic execution/evaluation scope mismatch"
        )
    if evaluation_scope == "formal":
        current_code = validate_code_source_closure(build_code_source_closure())
        saved_source = validate_source_closure(saved_args.get(SOURCE_CLOSURE_ARG))
        if saved_source["code"]["sha256"] != current_code["sha256"]:
            raise RuntimeError(
                "dense-duty confidence diagnostic training code source closure drifted"
            )
        audit["source_closure"] = saved_source
    elif terminal_probe:
        config_path = Path(str(getattr(cfg, "_filename", ""))).expanduser()
        if not config_path.is_file():
            raise RuntimeError(
                "dense-duty terminal confidence probe lacks its current config path"
            )
        config_path = config_path.resolve(strict=True)
        saved_source = validate_source_closure(saved_args.get(SOURCE_CLOSURE_ARG))
        if immutable_archived_context:
            audit["immutable_archived_source_closure"] = saved_source
        elif confidence_revision in _HISTORICAL_DENSE_DUTY_U50_SOURCES:
            audit["historical_source_archive"] = (
                _validate_historical_dense_duty_u50_source_archive(
                    revision=confidence_revision,
                    config_path=config_path,
                    checkpoint_path=resolved,
                    checkpoint_sha256=snapshot_sha256,
                    source_closure=saved_source,
                )
            )
        elif (
            fixed_u100_revision_contract
            and confidence_revision in _HISTORICAL_DENSE_DUTY_U100_SOURCES
        ):
            audit["historical_source_archive"] = (
                _validate_historical_dense_duty_u100_source_archive(
                    revision=confidence_revision,
                    config_path=config_path,
                    checkpoint_path=resolved,
                    checkpoint_sha256=snapshot_sha256,
                    source_closure=saved_source,
                )
            )
        else:
            current_source = validate_source_closure(
                build_source_closure(config_path)
            )
            if saved_source["sha256"] != current_source["sha256"]:
                raise RuntimeError(
                    "dense-duty terminal confidence probe source closure drifted"
                )
        audit["source_closure"] = saved_source

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
        "stage_b_dense_duty_forward_pack_factor",
        "stage_b_dense_duty_logical_loss_batch_size",
        "stage_b_dense_duty_expected_forward_batch_size",
        "stage_b_dense_duty_expected_logical_batches_per_epoch",
        "stage_b_dense_duty_expected_physical_forwards_per_epoch",
    )
    if token_edit_query_scope in {
        "target_iou_union_detached_final_confidence_base_argmax_v2",
        "target_iou_union_detached_role_complete_confidence_base_argmax_v3",
    }:
        required_equal_args += ("stage_b_v21_token_edit_query_scope",)
    if phrase_aggregation in {
        "trace_activated_word_veto_product_v1",
        "trace_activated_word_veto_penalty_v2",
        "trace_activated_word_veto_absolute_cap_v4",
        "trace_activated_word_veto_gated_pool_absolute_cap_v5",
    }:
        required_equal_args += (
            "stage_b_dense_duty_confidence_revision",
            "stage_b_dense_duty_confidence_phrase_aggregation",
            "stage_b_dense_duty_confidence_word_softmin_temperature",
            "stage_b_dense_duty_confidence_veto_gate_scale",
            "stage_b_dense_duty_positive_trust_contract",
            "stage_b_dense_duty_confidence_tn_scope",
            "stage_b_dense_duty_confidence_probe_admission_contract",
            "stage_b_dense_duty_confidence_probe_admission_report",
        )
        if float(
            getattr(cfg, "stage_b_dense_duty_raw_veto_gate_weight", 0.0)
            or 0.0
        ) > 0.0:
            required_equal_args += (
                "stage_b_dense_duty_raw_veto_gate_weight",
                "stage_b_dense_duty_raw_veto_positive_margin",
                "stage_b_dense_duty_raw_veto_tn_margin",
            )
        if phrase_aggregation in {
            "trace_activated_word_veto_absolute_cap_v4",
            "trace_activated_word_veto_gated_pool_absolute_cap_v5",
        }:
            required_equal_args += (
                "stage_b_dense_duty_raw_veto_query_scope",
                "stage_b_dense_duty_confidence_veto_gate_offset",
                "stage_b_dense_duty_confidence_veto_coverage_offset",
                "stage_b_dense_duty_confidence_veto_coverage_ramp",
                "stage_b_dense_duty_confidence_veto_cap_temperature",
                "stage_b_dense_duty_confidence_veto_cap_initial_ceiling",
            )
        if confidence_revision in {
            "word_veto_gated_pool_carrier_balanced_v7",
            "word_veto_gated_pool_carrier_quarter_v8",
            "word_veto_gated_pool_carrier_pair_v9",
            "word_veto_gated_pool_carrier_affine_v15",
            "word_veto_gated_pool_tail_carrier_v17",
            "word_veto_gated_pool_tail_paired_v18",
            "word_veto_gated_pool_tail_paired_rank_channel_v19",
            "word_veto_gated_pool_tail_paired_signed_rank_pool_v20",
            "word_veto_continuous_conditional_residual_v21",
            "word_veto_continuous_monotone_depth_v22",
            "word_veto_token_conditioned_monotone_depth_v23",
            "word_veto_complementary_trust_veto_v24",
            "word_veto_ungated_monotone_tail_veto_v25",
            "word_veto_floor_gated_monotone_tail_veto_v26",
            "word_veto_independent_absolute_confidence_v27",
            "word_veto_cross_attention_absolute_confidence_v28",
            "word_veto_candidate_absolute_confidence_v29",
            "word_veto_candidate_patch_invariant_confidence_v30",
            "word_veto_candidate_normalized_confidence_v31",
            "word_veto_candidate_asymmetric_confidence_v32",
            "word_veto_candidate_set_attention_confidence_v33",
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
        }:
            required_equal_args += (
                "stage_b_dense_duty_raw_veto_tn_carrier_balance",
                "stage_b_dense_duty_confidence_carrier_selector_contract",
            )
            if confidence_revision in {
                "word_veto_gated_pool_carrier_pair_v9",
                "word_veto_gated_pool_carrier_affine_v15",
                "word_veto_gated_pool_tail_carrier_v17",
                "word_veto_gated_pool_tail_paired_v18",
                "word_veto_gated_pool_tail_paired_rank_channel_v19",
                "word_veto_gated_pool_tail_paired_signed_rank_pool_v20",
                "word_veto_continuous_conditional_residual_v21",
                "word_veto_continuous_monotone_depth_v22",
                "word_veto_token_conditioned_monotone_depth_v23",
                "word_veto_complementary_trust_veto_v24",
                "word_veto_ungated_monotone_tail_veto_v25",
                "word_veto_floor_gated_monotone_tail_veto_v26",
                "word_veto_independent_absolute_confidence_v27",
                "word_veto_cross_attention_absolute_confidence_v28",
                "word_veto_candidate_absolute_confidence_v29",
                "word_veto_candidate_patch_invariant_confidence_v30",
                "word_veto_candidate_normalized_confidence_v31",
                "word_veto_candidate_asymmetric_confidence_v32",
                "word_veto_candidate_set_attention_confidence_v33",
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
            }:
                required_equal_args += (
                    "stage_b_dense_duty_raw_veto_carrier_pair_weight",
                    "stage_b_dense_duty_raw_veto_carrier_pair_margin",
                )
            if confidence_revision in {
                "word_veto_gated_pool_carrier_affine_v15",
                "word_veto_gated_pool_tail_carrier_v17",
                "word_veto_gated_pool_tail_paired_v18",
                "word_veto_gated_pool_tail_paired_rank_channel_v19",
                "word_veto_gated_pool_tail_paired_signed_rank_pool_v20",
                "word_veto_continuous_conditional_residual_v21",
                "word_veto_continuous_monotone_depth_v22",
                "word_veto_token_conditioned_monotone_depth_v23",
                "word_veto_complementary_trust_veto_v24",
                "word_veto_ungated_monotone_tail_veto_v25",
                "word_veto_floor_gated_monotone_tail_veto_v26",
                "word_veto_independent_absolute_confidence_v27",
                "word_veto_cross_attention_absolute_confidence_v28",
                "word_veto_candidate_absolute_confidence_v29",
                "word_veto_candidate_patch_invariant_confidence_v30",
                "word_veto_candidate_normalized_confidence_v31",
                "word_veto_candidate_asymmetric_confidence_v32",
                "word_veto_candidate_set_attention_confidence_v33",
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
            }:
                required_equal_args += (
                    "stage_b_dense_duty_confidence_rank_evidence_contract",
                    "stage_b_dense_duty_confidence_residual_parameterization_gain",
                )
            if confidence_revision in {
                "word_veto_gated_pool_tail_carrier_v17",
                "word_veto_gated_pool_tail_paired_v18",
                "word_veto_gated_pool_tail_paired_rank_channel_v19",
                "word_veto_gated_pool_tail_paired_signed_rank_pool_v20",
                "word_veto_continuous_conditional_residual_v21",
                "word_veto_continuous_monotone_depth_v22",
                "word_veto_token_conditioned_monotone_depth_v23",
                "word_veto_complementary_trust_veto_v24",
                "word_veto_ungated_monotone_tail_veto_v25",
                "word_veto_floor_gated_monotone_tail_veto_v26",
                "word_veto_independent_absolute_confidence_v27",
                "word_veto_cross_attention_absolute_confidence_v28",
                "word_veto_candidate_absolute_confidence_v29",
                "word_veto_candidate_patch_invariant_confidence_v30",
                "word_veto_candidate_normalized_confidence_v31",
                "word_veto_candidate_asymmetric_confidence_v32",
                "word_veto_candidate_set_attention_confidence_v33",
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
            }:
                required_equal_args += (
                    "stage_b_dense_duty_confidence_gate_gradient_contract",
                    "stage_b_dense_duty_raw_veto_tail_quantile",
                    "stage_b_dense_duty_raw_veto_tail_temperature",
                    "stage_b_dense_duty_raw_veto_tail_min_count",
                )
            if (
                word_veto_deployed_routing_revision_contract
                or word_veto_v52_revision_contract
                or word_veto_v53_revision_contract
                or word_veto_v54_revision_contract
                or word_veto_v55_revision_contract
            ):
                required_equal_args += (
                    "stage_b_v21_token_edit_query_scope",
                    "stage_b_dense_duty_raw_veto_carrier_pair_gradient_contract",
                    "stage_b_dense_duty_deployed_veto_routing_weight",
                    "stage_b_dense_duty_deployed_veto_positive_max",
                    "stage_b_dense_duty_deployed_veto_tn_min",
                )
            if (
                word_veto_v45_revision_contract
                or word_veto_v46_revision_contract
                or word_veto_v47_revision_contract
                or word_veto_v48_revision_contract
                or word_veto_v49_revision_contract
                or word_veto_v50_revision_contract
                or word_veto_v51_revision_contract
                or word_veto_v52_revision_contract
                or word_veto_v53_revision_contract
                or word_veto_v54_revision_contract
                or word_veto_v55_revision_contract
            ):
                required_equal_args += (
                    "stage_b_dense_duty_confidence_head_gradient_contract",
                    "stage_b_dense_duty_deployed_veto_routing_reduction_contract",
                    "stage_b_v15_tail_queue_positive_trust_reduction_contract",
                )
            if (
                word_veto_v48_revision_contract
                or word_veto_v50_revision_contract
                or word_veto_v51_revision_contract
                or word_veto_v52_revision_contract
                or word_veto_v53_revision_contract
                or word_veto_v54_revision_contract
                or word_veto_v55_revision_contract
            ):
                required_equal_args += (
                    "stage_b_v15_tail_queue_negative_reduction_contract",
                )
    drift = {
        key: (saved_args.get(key), getattr(cfg, key, None))
        for key in required_equal_args
        if saved_args.get(key) != getattr(cfg, key, None)
    }
    if word_veto_v49_revision_contract:
        key = "stage_b_v15_tail_queue_negative_reduction_contract"
        saved_negative_reduction = str(
            saved_args.get(key, _V49_SPLIT_GLOBAL_TRUST_VETO_NEGATIVE_REDUCTION)
        ).strip()
        current_negative_reduction = str(
            getattr(
                cfg,
                key,
                _V49_SPLIT_GLOBAL_TRUST_VETO_NEGATIVE_REDUCTION,
            )
        ).strip()
        if saved_negative_reduction != current_negative_reduction:
            drift[key] = (saved_negative_reduction, current_negative_reduction)
    runtime = saved_args.get("stage_b_dense_duty_runtime_audit")
    lineage = audit.get("lineage")
    rank_handoff = resume.get("rank_handoff")
    if drift:
        raise RuntimeError(
            "dense-duty confidence diagnostic configuration drifted from "
            f"training: {drift}"
        )
    if (
        saved_args.get("stage_b_dense_duty_no_stageb_teacher") is not True
        or saved_args.get("stage_b_v22_score_ownership")
        != "rank_tower_stopgrad_token_adapter_two_phase"
        or not isinstance(lineage, Mapping)
        or lineage.get("no_stage_b_teacher") is not True
        or lineage.get("execution_scope") != evaluation_scope
        or not isinstance(rank_handoff, Mapping)
        or not isinstance(runtime, Mapping)
        or runtime.get("successful_optimizer_steps") != observed_updates
        or runtime.get("optimizer_step_boundaries") != observed_updates
        or int(runtime.get("amp_skipped_optimizer_steps", -1)) != 0
        or int(runtime.get("nonfinite_gradient_boundaries", -1)) != 0
        or int(runtime.get("zero_gradient_successful_steps", -1)) != 0
        or float(runtime.get("max_active_grad_norm_preclip", 0.0)) <= 0.0
        or int(runtime.get("peak_reserved_bytes", 0)) <= 0
    ):
        raise RuntimeError(
            "dense-duty confidence diagnostic lacks valid adapter lineage/runtime "
            "evidence"
        )
    if word_veto_v52_revision_contract:
        _validate_v52_three_owner_runtime_audit(
            runtime,
            optimizer_updates=observed_updates,
        )
    if word_veto_v53_revision_contract:
        _validate_v53_two_owner_runtime_audit(
            runtime,
            optimizer_updates=observed_updates,
        )
    if word_veto_v54_revision_contract:
        _validate_v54_two_owner_runtime_audit(
            runtime,
            optimizer_updates=observed_updates,
        )
    if word_veto_v55_revision_contract:
        _validate_v55_two_owner_runtime_audit(
            runtime,
            optimizer_updates=observed_updates,
        )

    audit.update(
        {
            "checkpoint_reason": payload["checkpoint_reason"],
            "evaluation_scope": evaluation_scope,
            "diagnostic_only": True,
            "formal_gate_eligible": False,
            "confidence_evaluated": True,
            "training_phase": "confidence",
            "terminal_checkpoint": False,
            "expected_optimizer_updates": expected_updates,
            "remaining_optimizer_updates": expected_updates - observed_updates,
            "canonical_checkpoint": str(canonical),
            "canonical_checkpoint_sha256": canonical_sha256,
            "evaluation_checkpoint_sha256": snapshot_sha256,
            "strict_resume": resume,
            "rank_handoff": dict(rank_handoff),
        }
    )
    audit["terminal_checkpoint"] = terminal_probe
    if immutable_v39_context is not None:
        post_validation = _verify_v39_immutable_archived_diagnostic_files(resolved)
        audit.update(
            {
                "immutable_archived_snapshot_diagnostic": True,
                "terminal_checkpoint": False,
                "immutable_archived_provenance": {
                    **immutable_v39_context,
                    "snapshot_after_validation": post_validation["snapshot"],
                    "terminal_after_validation": post_validation["terminal"],
                },
            }
        )
    elif immutable_v40_context is not None:
        post_validation = _verify_v40_immutable_archived_diagnostic_files(resolved)
        audit.update(
            {
                "immutable_archived_snapshot_diagnostic": True,
                "immutable_archived_snapshot_version": "v40",
                "terminal_checkpoint": False,
                "immutable_archived_provenance": {
                    **immutable_v40_context,
                    "snapshot_after_validation": post_validation["snapshot"],
                    "terminal_after_validation": post_validation["terminal"],
                },
            }
        )
    elif immutable_v41_context is not None:
        post_validation = _verify_v41_immutable_archived_diagnostic_files(resolved)
        audit.update(
            {
                "immutable_archived_snapshot_diagnostic": True,
                "immutable_archived_snapshot_version": "v41",
                "terminal_checkpoint": False,
                "immutable_archived_provenance": {
                    **immutable_v41_context,
                    "snapshot_after_validation": post_validation["snapshot"],
                    "terminal_after_validation": post_validation["terminal"],
                },
            }
        )
    elif immutable_v42_context is not None:
        post_validation = _verify_v42_immutable_archived_diagnostic_files(resolved)
        audit.update(
            {
                "immutable_archived_snapshot_diagnostic": True,
                "immutable_archived_snapshot_version": "v42",
                "terminal_checkpoint": observed_updates == expected_updates,
                "immutable_archived_provenance": {
                    **immutable_v42_context,
                    "snapshot_after_validation": post_validation["snapshot"],
                    "terminal_after_validation": post_validation["terminal"],
                },
            }
        )
    return audit


def _bind_dense_duty_formal_probe_admission(cfg) -> None:
    contract = str(
        getattr(
            cfg,
            "stage_b_dense_duty_confidence_probe_admission_contract",
            "",
        )
    ).strip()
    if contract == "u300_word_veto_strict1607_v1":
        from tools import (
            run_stageb_confidence_adapter_veto_probe_evaluation as promotion,
        )
    elif contract == "u300_word_veto_gate_strict1607_v3":
        from tools import (
            run_stageb_confidence_adapter_veto_gate_probe_evaluation as promotion,
        )
    elif contract == "u300_word_veto_absolute_cap_strict1607_v4":
        from tools import (
            run_stageb_confidence_adapter_veto_cap_probe_evaluation as promotion,
        )
    elif contract == "u300_word_veto_gated_pool_absolute_cap_strict1607_v5":
        from tools import (
            run_stageb_confidence_adapter_veto_gated_pool_probe_evaluation as promotion,
        )
    elif contract == "u300_word_veto_gated_pool_calibrated_strict1607_v6":
        from tools import (
            run_stageb_confidence_adapter_veto_gated_pool_calibrated_probe_evaluation as promotion,
        )
    elif contract == "u300_word_veto_gated_pool_carrier_balanced_strict1607_v7":
        from tools import (
            run_stageb_confidence_adapter_veto_gated_pool_carrier_balanced_probe_evaluation as promotion,
        )
    elif contract == "u300_word_veto_gated_pool_carrier_quarter_strict1607_v8":
        from tools import (
            run_stageb_confidence_adapter_veto_gated_pool_carrier_quarter_probe_evaluation as promotion,
        )
    elif contract == "u300_word_veto_gated_pool_carrier_pair_strict1607_v9":
        from tools import (
            run_stageb_confidence_adapter_veto_gated_pool_carrier_pair_probe_evaluation as promotion,
        )
    elif contract == (
        "u400_word_veto_candidate_hardest_edit_confidence_strict1607_v40"
    ):
        from tools import (
            run_stageb_confidence_adapter_candidate_hardest_edit_probe_evaluation as promotion,
        )
    elif contract == (
        "u400_word_veto_candidate_role_complete_carrier_confidence_strict1607_v41"
    ):
        from tools import (
            run_stageb_confidence_adapter_candidate_role_complete_carrier_probe_evaluation as promotion,
        )
    elif contract == (
        "u400_word_veto_candidate_split_strong_boundary_routing_confidence_"
        "strict1607_v50"
    ):
        from tools import (
            run_stageb_confidence_adapter_candidate_split_strong_boundary_routing_probe_evaluation as promotion,
        )
    elif contract == (
        "u400_word_veto_candidate_split_independent_deployed_router_confidence_"
        "strict1607_v51"
    ):
        from tools import (
            run_stageb_confidence_adapter_candidate_split_independent_deployed_router_probe_evaluation as promotion,
        )
    elif contract == (
        "u400_word_veto_rank_full_expression_global_absolute_"
        "confidence_strict1607_v53"
    ):
        from tools import (
            run_stageb_confidence_adapter_fulltext_global_absolute_probe_evaluation as promotion,
        )
    elif contract == (
        "u400_word_veto_rank_full_expression_global_absolute_exact_residual_"
        "confidence_strict1607_v54"
    ):
        from tools import (
            run_stageb_confidence_adapter_fulltext_global_absolute_exact_residual_probe_evaluation as promotion,
        )
    elif contract == (
        "u400_word_veto_rank_full_expression_global_independent_absolute_"
        "confidence_strict1607_v55"
    ):
        from tools import (
            run_stageb_confidence_adapter_fulltext_global_independent_absolute_probe_evaluation as promotion,
        )
    else:
        return

    configured_report = Path(
        str(
            getattr(
                cfg,
                "stage_b_dense_duty_confidence_probe_admission_report",
                "",
            )
        )
    ).expanduser().resolve(strict=True)
    if configured_report != promotion.REPORT.resolve(strict=True):
        raise RuntimeError(
            "formal confidence evaluation points at a noncanonical promotion "
            "report"
        )
    cfg.stage_b_dense_duty_confidence_probe_admission_audit = (
        promotion.verify_admission_report(configured_report)
    )


def _load_model(cfg, ckpt_path: str, device: torch.device):
    _bind_dense_duty_formal_probe_admission(cfg)
    build_func = MODULE_BUILD_FUNCS.get(cfg.modelname)
    if build_func is None:
        raise KeyError(f"Unknown modelname={cfg.modelname}")
    if not hasattr(cfg, "label_list"):
        # Ref/TN evaluators consume model-native Stage-B scores and never call
        # the generic detection postprocessor, but the builder still creates it.
        cfg.label_list = ["object"]
    model, _criterion, _postprocessors = build_func(cfg)
    ckpt = _torch_load_compat(ckpt_path, map_location="cpu")
    state = clean_state_dict(_extract_state_dict(ckpt))
    dense_duty_checkpoint_audit = None
    stage_b_dense_duty = bool(getattr(cfg, "stage_b_dense_duty", False))
    if stage_b_dense_duty:
        partial_rank_diagnostic = bool(
            getattr(
                cfg,
                "stage_b_dense_duty_partial_rank_diagnostic",
                False,
            )
        )
        partial_confidence_diagnostic = bool(
            getattr(
                cfg,
                "stage_b_dense_duty_partial_confidence_diagnostic",
                False,
            )
        )
        if partial_rank_diagnostic and partial_confidence_diagnostic:
            raise RuntimeError(
                "dense-duty rank and confidence partial diagnostics are mutually "
                "exclusive"
            )
        if partial_rank_diagnostic:
            dense_duty_checkpoint_audit = (
                _validate_dense_duty_partial_rank_diagnostic_checkpoint(
                    ckpt,
                    cfg,
                    checkpoint_path=Path(ckpt_path),
                )
            )
            cfg.stage_b_dense_duty_partial_rank_diagnostic_optimizer_updates = (
                dense_duty_checkpoint_audit["optimizer_updates"]
            )
            cfg.stage_b_dense_duty_partial_rank_diagnostic_checkpoint_reason = (
                dense_duty_checkpoint_audit["checkpoint_reason"]
            )
        elif partial_confidence_diagnostic:
            dense_duty_checkpoint_audit = (
                _validate_dense_duty_partial_confidence_diagnostic_checkpoint(
                    ckpt,
                    cfg,
                    checkpoint_path=Path(ckpt_path),
                )
            )
            cfg.stage_b_dense_duty_partial_confidence_diagnostic_optimizer_updates = (
                dense_duty_checkpoint_audit["optimizer_updates"]
            )
            cfg.stage_b_dense_duty_partial_confidence_diagnostic_checkpoint_reason = (
                dense_duty_checkpoint_audit["checkpoint_reason"]
            )
            cfg.stage_b_dense_duty_partial_confidence_diagnostic_expected_optimizer_updates = (
                dense_duty_checkpoint_audit["expected_optimizer_updates"]
            )
            cfg.stage_b_dense_duty_partial_confidence_diagnostic_terminal_checkpoint = (
                bool(dense_duty_checkpoint_audit.get("terminal_checkpoint", False))
            )
            if bool(
                dense_duty_checkpoint_audit.get(
                    "immutable_archived_snapshot_diagnostic", False
                )
            ):
                if bool(
                    getattr(
                        cfg,
                        "stage_b_dense_duty_immutable_v42_archived_snapshot_diagnostic",
                        False,
                    )
                ):
                    cfg.stage_b_dense_duty_immutable_v42_archived_snapshot_audit = (
                        dense_duty_checkpoint_audit
                    )
                elif bool(
                    getattr(
                        cfg,
                        "stage_b_dense_duty_immutable_v41_archived_snapshot_diagnostic",
                        False,
                    )
                ):
                    cfg.stage_b_dense_duty_immutable_v41_archived_snapshot_audit = (
                        dense_duty_checkpoint_audit
                    )
                elif bool(
                    getattr(
                        cfg,
                        "stage_b_dense_duty_immutable_v40_archived_snapshot_diagnostic",
                        False,
                    )
                ):
                    cfg.stage_b_dense_duty_immutable_v40_archived_snapshot_audit = (
                        dense_duty_checkpoint_audit
                    )
                else:
                    cfg.stage_b_dense_duty_immutable_v39_archived_snapshot_audit = (
                        dense_duty_checkpoint_audit
                    )
        else:
            from util.stage_b_dense_duty_audit import (
                validate_evaluation_checkpoint_payload,
            )

            dense_duty_checkpoint_audit = validate_evaluation_checkpoint_payload(
                ckpt,
                cfg,
                checkpoint_path=Path(ckpt_path),
            )
    if bool(getattr(cfg, "stage_b_v11_fixed_text", False)):
        validate_stage_b_fixed_text_scorer_checkpoint(
            model,
            state,
            checkpoint_label=f"Stage B v11 evaluation checkpoint {ckpt_path}",
        )
    if bool(getattr(cfg, "stage_b_gdino_score_adapter", False)):
        from models.GroundingDINO.stage_b_gdino_score_adapter import (
            validate_stage_b_gdino_score_adapter_checkpoint,
        )

        validate_stage_b_gdino_score_adapter_checkpoint(
            model,
            state,
            checkpoint_label=f"Stage-B GDINO adapter evaluation checkpoint {ckpt_path}",
        )
    stage_b_u0_patch_rank = bool(
        getattr(cfg, "stage_b_u0_patch_rank", False)
    )
    stage_b_data_only_composite = bool(
        getattr(cfg, "stage_b_data_only_composite", False)
    )
    if stage_b_data_only_composite:
        if not stage_b_u0_patch_rank:
            raise ValueError(
                "stage_b_data_only_composite requires stage_b_u0_patch_rank"
            )
        from tools.build_stageb_data_only_composite import (
            validate_data_only_composite_payload,
            validate_data_only_composite_runtime_config,
        )

        validate_data_only_composite_payload(
            model,
            ckpt,
            checkpoint_label=(
                f"Stage-B data-only composite Ref checkpoint {ckpt_path}"
            ),
        )
        validate_data_only_composite_runtime_config(
            cfg,
            ckpt,
            checkpoint_path=ckpt_path,
            checkpoint_label=(
                f"Stage-B data-only composite Ref checkpoint {ckpt_path}"
            ),
        )
    stage_b_data_driven_score = bool(
        getattr(cfg, "stage_b_data_driven_score", False)
    )
    stage_b_native_patch_category = _validate_native_patch_category_config(cfg)
    if stage_b_data_driven_score:
        from models.GroundingDINO.stage_b_data_driven_score import (
            validate_data_driven_trained_checkpoint_payload,
        )

        expected_eval_updates = getattr(
            cfg, "stage_b_data_driven_eval_expected_optimizer_updates", 0
        )
        if (
            isinstance(expected_eval_updates, bool)
            or not isinstance(expected_eval_updates, int)
            or expected_eval_updates < 0
        ):
            raise ValueError(
                "stage_b_data_driven_eval_expected_optimizer_updates must be "
                "a non-negative exact integer"
            )

        validate_data_driven_trained_checkpoint_payload(
            model,
            ckpt,
            checkpoint_label=(
                f"Stage-B data-driven evaluation checkpoint {ckpt_path}"
            ),
            expected_experiment_id=str(
                getattr(cfg, "stage_b_data_driven_experiment_id", "")
            ),
            expected_confidence_trained=bool(
                getattr(cfg, "stage_b_data_driven_confidence_trained", False)
            ),
            expected_variant_id=(
                str(
                    getattr(cfg, "stage_b_data_driven_variant_id", "")
                ).strip()
                or None
            ),
            expected_rank_supervision=str(
                getattr(
                    cfg,
                    "stage_b_data_driven_rank_supervision",
                    "all_nonpositive_negative_v1",
                )
            ),
            expected_rank_negative_iou_threshold=float(
                getattr(
                    cfg,
                    "stage_b_data_driven_rank_negative_iou_threshold",
                    0.3,
                )
            ),
            expected_assignment_weight=getattr(
                cfg, "stage_b_data_driven_assignment_weight", None
            ),
            expected_deployment_weight=getattr(
                cfg, "stage_b_data_driven_deployment_weight", None
            ),
            expected_token_weight=(
                float(getattr(cfg, "stage_b_data_driven_token_weight", 0.0))
                if str(
                    getattr(cfg, "stage_b_data_driven_experiment_id", "")
                )
                in {"DD2", "DD3"}
                else None
            ),
            expected_confidence_initializer_sha256=(
                str(
                    getattr(
                        cfg,
                        "stage_b_data_driven_confidence_initializer_sha256",
                        "",
                    )
                )
                if str(
                    getattr(cfg, "stage_b_data_driven_experiment_id", "")
                )
                in {"DD2", "DD3"}
                else None
            ),
            expected_optimizer_updates=expected_eval_updates or None,
        )
    if stage_b_u0_patch_rank:
        from models.GroundingDINO.stage_b_u0_patch_rank import (
            validate_stage_b_u0_patch_rank_checkpoint,
        )

        validate_stage_b_u0_patch_rank_checkpoint(
            model,
            state,
            checkpoint_label=f"Stage-B U0 Ref evaluation checkpoint {ckpt_path}",
        )
    missing, unexpected = model.load_state_dict(
        state,
        strict=(
            stage_b_u0_patch_rank
            or stage_b_data_driven_score
            or stage_b_native_patch_category
            or stage_b_dense_duty
        ),
    )
    if missing:
        print(f"[WARN] {ckpt_path}: missing keys={len(missing)}", file=sys.stderr)
    if unexpected:
        print(f"[WARN] {ckpt_path}: unexpected keys={len(unexpected)}", file=sys.stderr)
    if stage_b_native_patch_category:
        model.stage_b_native_patch_confidence_trained = bool(
            getattr(cfg, "stage_b_native_patch_confidence_trained", False)
        )
    if dense_duty_checkpoint_audit is not None:
        if bool(
            dense_duty_checkpoint_audit.get(
                "immutable_archived_snapshot_diagnostic", False
            )
        ):
            immutable_v42 = bool(
                getattr(
                    cfg,
                    "stage_b_dense_duty_immutable_v42_archived_snapshot_diagnostic",
                    False,
                )
            )
            immutable_v41 = bool(
                getattr(
                    cfg,
                    "stage_b_dense_duty_immutable_v41_archived_snapshot_diagnostic",
                    False,
                )
            )
            immutable_v40 = bool(
                getattr(
                    cfg,
                    "stage_b_dense_duty_immutable_v40_archived_snapshot_diagnostic",
                    False,
                )
            )
            post_load = (
                _verify_v42_immutable_archived_diagnostic_files(Path(ckpt_path))
                if immutable_v42
                else (
                    _verify_v41_immutable_archived_diagnostic_files(Path(ckpt_path))
                    if immutable_v41
                    else (
                        _verify_v40_immutable_archived_diagnostic_files(
                            Path(ckpt_path)
                        )
                        if immutable_v40
                        else _verify_v39_immutable_archived_diagnostic_files(
                            Path(ckpt_path)
                        )
                    )
                )
            )
            dense_duty_checkpoint_audit[
                "immutable_archived_provenance"
            ]["snapshot_after_model_load"] = post_load["snapshot"]
            dense_duty_checkpoint_audit[
                "immutable_archived_provenance"
            ]["terminal_after_model_load"] = post_load["terminal"]
            # SLConfig materializes assigned mappings, so publish the updated
            # audit again after adding the post-load identities.
            if immutable_v42:
                cfg.stage_b_dense_duty_immutable_v42_archived_snapshot_audit = (
                    dense_duty_checkpoint_audit
                )
            elif immutable_v41:
                cfg.stage_b_dense_duty_immutable_v41_archived_snapshot_audit = (
                    dense_duty_checkpoint_audit
                )
            elif immutable_v40:
                cfg.stage_b_dense_duty_immutable_v40_archived_snapshot_audit = (
                    dense_duty_checkpoint_audit
                )
            else:
                cfg.stage_b_dense_duty_immutable_v39_archived_snapshot_audit = (
                    dense_duty_checkpoint_audit
                )
        model.stage_b_dense_duty_checkpoint_audit = dense_duty_checkpoint_audit
    model.to(device).eval()
    return model


def _seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _norm_text(value: Any) -> str:
    return _WS_RE.sub(" ", str(value or "").replace("_", " ").replace(".", " ").strip().lower())


def _clean_phrase(value: Any) -> str:
    text = _WS_RE.sub(" ", str(value or "").replace("_", " ").replace(".", " ").strip())
    return text or "object"


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "run"


def _ckpt_run_prefix(ckpt_path: str) -> str:
    path = Path(ckpt_path)
    parent = path.parent.name
    stem = path.stem
    return _safe_name(f"{parent}_{stem}") if parent else _safe_name(stem)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _file_identity(value: str, *, label: str) -> Dict[str, Any]:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file: {path}")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def _require_bound_file(record: Dict[str, Any], *, label: str) -> Dict[str, Any]:
    if set(record) != {"path", "sha256", "size_bytes"}:
        raise ValueError(f"{label} binding must contain path/sha256/size_bytes")
    observed = _file_identity(str(record["path"]), label=label)
    expected = {
        "path": str(Path(str(record["path"])).expanduser().resolve()),
        "sha256": record["sha256"],
        "size_bytes": record["size_bytes"],
    }
    if observed != expected:
        raise ValueError(f"{label} file identity drifted")
    return observed


def _load_bound_config(record: Dict[str, Any], *, label: str):
    _require_bound_file(record, label=label)
    cfg = SLConfig.fromfile(str(record["path"]))
    _require_bound_file(record, label=label)
    return cfg


def _load_bound_model(
    cfg,
    checkpoint_record: Dict[str, Any],
    config_record: Dict[str, Any],
    device: torch.device,
    *,
    label: str,
):
    _require_bound_file(config_record, label=f"{label} config")
    _require_bound_file(checkpoint_record, label=f"{label} checkpoint")
    model = _load_model(cfg, str(checkpoint_record["path"]), device)
    _require_bound_file(checkpoint_record, label=f"{label} checkpoint")
    _require_bound_file(config_record, label=f"{label} config")
    return model


def _validate_formal_cli_contract(args) -> None:
    splits = list(args.splits or [])
    if splits not in (["all"], list(REF_SPLIT_ORDER)):
        raise ValueError(
            "formal external rank transfer requires --splits all or the exact "
            "canonical Ref8 order"
        )
    if int(args.max_batches) != 0 or int(args.max_images) != 0:
        raise ValueError(
            "formal external rank transfer requires full manifests "
            "(--max_batches=0 and --max_images=0)"
        )
    if str(args.holdout_level) != "none" or list(args.exclude_train_jsonl or []):
        raise ValueError(
            "formal external rank transfer forbids holdout/exclusion filtering"
        )
    if bool(args.no_per_example_records):
        raise ValueError(
            "formal external rank transfer requires canonical per-example records"
        )


def _reverify_formal_runtime_settings(settings: Dict[str, Any]) -> None:
    refreshed = evaluator_settings_from_artifact(settings["artifact"]["path"])
    if refreshed != settings:
        raise ValueError(
            "formal external artifact or a bound runtime component changed during evaluation"
        )


def _validate_formal_routed_runtime(
    settings: Dict[str, Any],
    *,
    batch_size: int,
    num_workers: int,
    amp: bool,
) -> None:
    version = int(settings.get("formal_artifact_version", 1))
    if version not in (2, 3):
        return
    if version == 2:
        contract = settings["route_selection"]["selection_contract"]
        expected_batch_size = int(contract["batch_size"])
        expected_num_workers = int(contract["num_workers"])
        amp_contract = str(contract["amp"])
        if amp_contract.startswith("enabled"):
            expected_amp = True
        elif amp_contract.startswith("disabled"):
            expected_amp = False
        else:
            raise ValueError(
                "formal routed selection has an unsupported AMP contract: "
                f"{amp_contract!r}"
            )
    else:
        protocol = settings["evaluation_protocol"]
        expected_batch_size = int(protocol["batch_size"])
        expected_num_workers = int(protocol["num_workers"])
        expected_amp = bool(protocol["amp"])
    observed = (int(batch_size), int(num_workers), bool(amp))
    expected = (expected_batch_size, expected_num_workers, expected_amp)
    if observed != expected:
        raise ValueError(
            "formal routed evaluation runtime must match frozen selection/protocol: "
            f"batch_size/num_workers/amp expected={expected}, observed={observed}"
        )


def _load_canonical_name_maps(path: Path) -> Tuple[Dict[str, int], Dict[int, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    name_to_id: Dict[str, int] = {}
    id_to_name: Dict[int, str] = {}
    alias_values: List[Tuple[str, int]] = []
    for row in data:
        if not isinstance(row, dict) or row.get("id") is None:
            continue
        cid = int(row["id"])
        preferred = row.get("base_name") or row.get("norm_name") or row.get("raw_name")
        if isinstance(preferred, str) and preferred.strip():
            id_to_name.setdefault(cid, _clean_phrase(preferred))
        values = [row.get("raw_name"), row.get("norm_name"), row.get("base_name")]
        values.extend(row.get("synonyms") or [])
        for value in values:
            if isinstance(value, str) and value.strip():
                name_to_id.setdefault(_norm_text(value), cid)
        for alias in row.get("aliases") or []:
            if not isinstance(alias, dict):
                continue
            for key in ("name", "norm_name"):
                value = alias.get(key)
                if isinstance(value, str) and value.strip():
                    alias_values.append((value, cid))
    for value, cid in alias_values:
        name_to_id.setdefault(_norm_text(value), cid)
    return name_to_id, id_to_name


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _phrase_key(source: str, ref_id: Any, ann_id: Any, image_id: Any, phrase: Any) -> Tuple[str, int, int, int, str]:
    return (str(source), int(ref_id), int(ann_id), int(image_id), _norm_text(phrase))


def _load_phrase_maps(paths: Iterable[Path]) -> Dict[Tuple[str, int, int, int, str], Dict[str, Any]]:
    out: Dict[Tuple[str, int, int, int, str], Dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        for row in _iter_jsonl(path):
            instances = row.get("instances")
            if isinstance(instances, list) and instances:
                for inst in instances:
                    phrase = inst.get("raw_phrase") or inst.get("phrase") or inst.get("positive_phrase")
                    source = inst.get("pair_source") or row.get("pair_source") or row.get("source")
                    ref_id = row.get("ref_id")
                    ann_id = row.get("ann_id")
                    image_id = row.get("image_id")
                    if source is None or ref_id is None or ann_id is None or image_id is None or not phrase:
                        continue
                    rec = dict(inst)
                    rec.update({"source": source, "ref_id": ref_id, "ann_id": ann_id, "image_id": image_id})
                    out.setdefault(_phrase_key(source, ref_id, ann_id, image_id, phrase), rec)
                continue
            phrase = row.get("raw_phrase") or row.get("phrase") or row.get("head_phrase")
            source = row.get("source") or row.get("pair_source")
            ref_id = row.get("ref_id")
            ann_id = row.get("ann_id")
            image_id = row.get("image_id")
            if source is None or ref_id is None or ann_id is None or image_id is None or not phrase:
                continue
            out.setdefault(_phrase_key(source, ref_id, ann_id, image_id, phrase), dict(row))
    return out


def _load_instances(path: Path) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]], Dict[int, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    anns = {int(row["id"]): row for row in data.get("annotations", [])}
    images = {int(row["id"]): row for row in data.get("images", [])}
    cats = {int(row["id"]): str(row.get("name", "")) for row in data.get("categories", [])}
    return anns, images, cats


def _image_path(data_root: Path, image: Dict[str, Any]) -> str:
    filename = str(image.get("file_name", ""))
    candidates = [
        data_root / "COCO" / "coco2014" / "train2014" / filename,
        data_root / "COCO" / "coco2014" / "val2014" / filename,
        data_root / "COCO" / "coco2017" / "train2017" / "train2017" / filename.replace("COCO_train2014_", ""),
        data_root / "COCO" / "coco2017" / "val2017" / filename.replace("COCO_val2014_", ""),
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    return str(candidates[0])


def _resolve_class_record(
    *,
    phrase_maps: Dict[Tuple[str, int, int, int, str], Dict[str, Any]],
    phrase_sources: List[str],
    ref: Dict[str, Any],
    ann: Dict[str, Any],
    image_id: int,
    phrase: str,
    category_name: str,
    name_to_id: Dict[str, int],
    id_to_name: Dict[int, str],
) -> Dict[str, Any]:
    hit: Optional[Dict[str, Any]] = None
    for source in phrase_sources:
        hit = phrase_maps.get(_phrase_key(source, ref["ref_id"], ann["id"], image_id, phrase))
        if hit:
            break
    if hit is None:
        cid = name_to_id.get(_norm_text(category_name), int(ann.get("category_id", -1)))
        canon = id_to_name.get(int(cid), _clean_phrase(category_name))
        return {
            "class_id": int(cid),
            "head": canon,
            "head_phrase": canon,
            "canonical_name": canon,
            "class_id_source": "category_fallback",
        }
    cid = int(hit.get("class_id", hit.get("head_classifier_class_id", ann.get("category_id", -1))))
    canon = (
        hit.get("canonical_name")
        or hit.get("class_norm_name")
        or hit.get("class_raw_name")
        or id_to_name.get(cid)
        or category_name
    )
    return {
        "class_id": cid,
        "head": hit.get("head") or hit.get("try_tn_head") or canon,
        "head_phrase": hit.get("head_phrase") or hit.get("try_tn_head_phrase") or canon,
        "canonical_name": _clean_phrase(canon),
        "class_id_source": hit.get("class_id_source") or hit.get("label_match_type") or "phrase_map",
    }


def _build_split_jsonl(
    *,
    data_root: Path,
    output_dir: Path,
    dataset: str,
    splitby: str,
    split: str,
    phrase_sources: List[str],
    phrase_maps: Dict[Tuple[str, int, int, int, str], Dict[str, Any]],
    name_to_id: Dict[str, int],
    id_to_name: Dict[int, str],
    holdout_level: str = "none",
    holdout_ann_keys=None,
    holdout_image_ids=None,
) -> Tuple[Path, int]:
    ref_root = data_root / "COCO" / dataset
    refs_path = ref_root / f"refs({splitby}).p"
    instances_path = ref_root / "instances.json"
    refs = pickle.load(refs_path.open("rb"))
    anns, images, cats = _load_instances(instances_path)

    out_dir = output_dir / "refcoco_eval_inputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset.replace('+', 'plus')}_{splitby}_{split}.jsonl"
    count = 0
    holdout_ann_keys = holdout_ann_keys or set()
    holdout_image_ids = holdout_image_ids or set()
    with out_path.open("w", encoding="utf-8") as f:
        for ref in refs:
            if str(ref.get("split")) != str(split):
                continue
            if is_excluded(
                image_id=int(ref["image_id"]),
                ann_id=int(ref["ann_id"]),
                level=holdout_level,
                ann_keys=holdout_ann_keys,
                image_ids=holdout_image_ids,
            ):
                continue
            ann = anns.get(int(ref["ann_id"]))
            image = images.get(int(ref["image_id"]))
            if ann is None or image is None:
                continue
            category_name = cats.get(int(ann.get("category_id", -1)), "")
            for sent in ref.get("sentences", []) or []:
                phrase = _clean_phrase(sent.get("sent") or sent.get("raw"))
                class_rec = _resolve_class_record(
                    phrase_maps=phrase_maps,
                    phrase_sources=phrase_sources,
                    ref=ref,
                    ann=ann,
                    image_id=int(ref["image_id"]),
                    phrase=phrase,
                    category_name=category_name,
                    name_to_id=name_to_id,
                    id_to_name=id_to_name,
                )
                row = {
                    "filename": _image_path(data_root, image),
                    "source": f"{dataset}_{splitby}_{split}",
                    "image_id": int(ref["image_id"]),
                    "ann_id": int(ref["ann_id"]),
                    "ref_id": int(ref["ref_id"]),
                    "sent_id": int(sent.get("sent_id", count)),
                    "split": split,
                    "instances": [
                        {
                            "bbox": ann["bbox"],
                            "class_id": int(class_rec["class_id"]),
                            "raw_phrase": phrase,
                            "head_phrase": _clean_phrase(class_rec["head_phrase"]),
                            "head": _clean_phrase(class_rec["head"]),
                            "canonical_name": _clean_phrase(class_rec["canonical_name"]),
                            "positive_phrase": phrase,
                            "text_is_negative": False,
                            "pair_source": phrase_sources[0] if phrase_sources else dataset,
                            "category_name": category_name,
                            "class_id_source": class_rec["class_id_source"],
                            "refcoco_category_id": int(ann.get("category_id", -1)),
                        }
                    ],
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    return out_path, count


def _default_splits() -> List[Dict[str, Any]]:
    return [
        {"name": "refcoco_val", "dataset": "refcoco", "splitby": "unc", "split": "val", "sources": ["refcoco_unc"]},
        {"name": "refcoco_testA", "dataset": "refcoco", "splitby": "unc", "split": "testA", "sources": ["refcoco_unc"]},
        {"name": "refcoco_testB", "dataset": "refcoco", "splitby": "unc", "split": "testB", "sources": ["refcoco_unc"]},
        {"name": "refcocop_val", "dataset": "refcoco+", "splitby": "unc", "split": "val", "sources": ["refcoco+_unc"]},
        {"name": "refcocop_testA", "dataset": "refcoco+", "splitby": "unc", "split": "testA", "sources": ["refcoco+_unc"]},
        {"name": "refcocop_testB", "dataset": "refcoco+", "splitby": "unc", "split": "testB", "sources": ["refcoco+_unc"]},
        {"name": "refcocog_val", "dataset": "refcocog", "splitby": "umd", "split": "val", "sources": ["refcocog_google"]},
        {"name": "refcocog_test", "dataset": "refcocog", "splitby": "umd", "split": "test", "sources": ["refcocog_google"]},
    ]


def _canonical_ref_split_seed_map(base_seed: int) -> Dict[str, int]:
    specs = _default_splits()
    names = [str(spec.get("name", "")) for spec in specs]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("canonical Ref split definitions must have unique names")
    if tuple(names) != REF_SPLIT_ORDER:
        raise ValueError(
            "canonical Ref split definitions/order drifted from the stable seed protocol"
        )
    return stable_ref_split_seed_map(base_seed)


def _requested_ref_split_specs(
    requested: Iterable[str], base_seed: int
) -> List[Tuple[str, Dict[str, Any], int]]:
    specs = _default_splits()
    by_name = {str(spec["name"]): spec for spec in specs}
    if len(by_name) != len(specs):
        raise ValueError("canonical Ref split definitions contain duplicate names")
    wanted = [str(name) for name in requested]
    if wanted == ["all"]:
        wanted = list(REF_SPLIT_ORDER)
    if len(set(wanted)) != len(wanted):
        raise ValueError(f"duplicate Ref split names are not allowed: {wanted}")
    unknown = [name for name in wanted if name not in by_name]
    if unknown:
        raise KeyError(
            f"Unknown split names: {unknown}; available={list(REF_SPLIT_ORDER)}"
        )
    seeds = _canonical_ref_split_seed_map(base_seed)
    return [(name, by_name[name], seeds[name]) for name in wanted]


def _make_datasetinfo(
    data_root: Path,
    name: str,
    anno: Path,
    *,
    adapter_no_support: bool = False,
) -> Dict[str, Any]:
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
        "text_mask_warn_limit": 0,
        "text_mask_skip_invalid_canonical": False,
        "tn_balance_sampling": False,
    }
    if adapter_no_support:
        datasetinfo.update(
            stage_b_gdino_adapter_no_support=True,
            stage_b_gdino_adapter_ref_eval=True,
        )
    else:
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


def _adapter_ref_eval_uses_no_support(cfg) -> bool:
    return bool(
        getattr(cfg, "stage_b_gdino_score_adapter", False)
        and not getattr(cfg, "stage_b_u0_patch_rank", False)
    )


def _build_loader(cfg, datasetinfo: Dict[str, Any], batch_size: int, num_workers: int, device: torch.device, seed: int):
    _set_seed(seed)
    dataset = build_dataset(image_set="val", args=cfg, datasetinfo=datasetinfo)
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


def _pad_target_mask(targets: List[Dict[str, Any]], key: str, kmax: int, device: torch.device) -> Optional[torch.Tensor]:
    if not any(key in t for t in targets):
        return None
    out = torch.zeros((len(targets), kmax, 256), dtype=torch.bool, device=device)
    for i, target in enumerate(targets):
        mask = target.get(key)
        if not torch.is_tensor(mask):
            continue
        rows = min(kmax, int(mask.shape[0]))
        cols = min(256, int(mask.shape[-1]))
        if rows > 0 and cols > 0:
            out[i, :rows, :cols] = mask[:rows, :cols].to(device=device, dtype=torch.bool)
    return out


def _native_patch_category_phrase_mask(
    raw_targets: List[Dict[str, Any]], device: torch.device
) -> torch.Tensor:
    masks = [target.get("phrase_to_token_mask") for target in raw_targets]
    if not masks or any(
        not torch.is_tensor(mask)
        or mask.dtype != torch.bool
        or mask.dim() != 2
        or int(mask.shape[0]) != 1
        or not bool(mask.any().item())
        for mask in masks
    ):
        raise ValueError(
            "every native patch-category Ref row requires one non-empty boolean "
            "phrase_to_token_mask"
        )
    shapes = {tuple(mask.shape) for mask in masks}
    if len(shapes) != 1:
        raise ValueError(
            "native patch-category Ref phrase masks must share one token geometry"
        )
    return torch.stack(masks, dim=0).to(device=device)


def _set_native_patch_category_derived_output(
    outputs: Dict[str, torch.Tensor], key: str, value: torch.Tensor
) -> None:
    existing = outputs.get(key)
    if existing is not None:
        if not torch.is_tensor(existing) or not torch.equal(existing, value):
            raise RuntimeError(
                f"native patch-category model output {key!r} conflicts with the "
                "authoritative evaluator derivation"
            )
        return
    outputs[key] = value


def _derive_native_patch_category_ref_scores(
    outputs: Dict[str, torch.Tensor], expected_phrase_mask: torch.Tensor
) -> None:
    required = ("pred_logits_text", "pred_logits_patch", "pred_boxes")
    missing = [key for key in required if not torch.is_tensor(outputs.get(key))]
    if missing:
        raise KeyError(
            "native patch-category Ref forward is missing required tensor keys: "
            f"{missing}"
        )
    text_logits = outputs["pred_logits_text"]
    patch_logits = outputs["pred_logits_patch"]
    pred_boxes = outputs["pred_boxes"]
    if (
        text_logits.dim() != 3
        or not text_logits.is_floating_point()
        or pred_boxes.dim() != 3
        or not pred_boxes.is_floating_point()
        or int(pred_boxes.shape[-1]) != 4
        or tuple(text_logits.shape[:2]) != tuple(pred_boxes.shape[:2])
        or int(text_logits.shape[1]) != _NATIVE_PATCH_QUERY_COUNT
    ):
        raise ValueError(
            "native patch-category Ref requires aligned floating "
            f"(B,{_NATIVE_PATCH_QUERY_COUNT},T) text logits and (B,Q,4) boxes"
        )
    if not bool(torch.isfinite(text_logits).all().item()) or not bool(
        torch.isfinite(pred_boxes).all().item()
    ):
        raise ValueError(
            "native patch-category Ref text logits and boxes must be finite"
        )
    if patch_logits.dim() == 3:
        if int(patch_logits.shape[-1]) != 1:
            raise ValueError(
                "native patch-category Ref requires exactly one patch support slot"
            )
        flat_patch_logits = patch_logits[..., 0]
    elif patch_logits.dim() == 2:
        flat_patch_logits = patch_logits
    else:
        raise ValueError(
            "native patch-category Ref pred_logits_patch must be (B,Q) or (B,Q,1)"
        )
    if tuple(flat_patch_logits.shape) != tuple(text_logits.shape[:2]):
        raise ValueError(
            "native patch-category Ref patch logits do not align with text queries"
        )

    observed_phrase_mask = outputs.get("phrase_to_token_mask")
    if not torch.is_tensor(observed_phrase_mask):
        raise KeyError(
            "native patch-category Ref forward is missing phrase_to_token_mask"
        )
    observed_phrase_mask = observed_phrase_mask.to(
        device=expected_phrase_mask.device, dtype=torch.bool
    )
    if tuple(observed_phrase_mask.shape) != tuple(expected_phrase_mask.shape) or not torch.equal(
        observed_phrase_mask, expected_phrase_mask
    ):
        raise RuntimeError(
            "native patch-category Ref forward changed the exact full-expression mask"
        )
    if int(observed_phrase_mask.shape[-1]) != int(text_logits.shape[-1]):
        raise ValueError(
            "native patch-category Ref expression mask does not align with text logits"
        )
    expression_mask = observed_phrase_mask.any(dim=1)
    native_score = aggregate_gdino_full_expression_score(
        text_logits, expression_mask
    )
    candidate_mask = torch.ones_like(native_score, dtype=torch.bool)
    rank_score, eligible, standardized_patch = apply_native_patch_category_gate(
        native_score,
        flat_patch_logits,
        candidate_mask,
        max_gap=_NATIVE_PATCH_GATE_MAX_GAP,
        clip=_NATIVE_PATCH_GATE_CLIP,
    )
    for key, value in (
        (_NATIVE_PATCH_BASE_SCORE_KEY, native_score),
        (_NATIVE_PATCH_RANK_SCORE_KEY, rank_score),
        (_NATIVE_PATCH_ELIGIBLE_MASK_KEY, eligible),
        (_NATIVE_PATCH_STANDARDIZED_SCORE_KEY, standardized_patch),
        (_NATIVE_PATCH_EXPRESSION_MASK_KEY, expression_mask),
    ):
        _set_native_patch_category_derived_output(outputs, key, value)


def _target_texts(raw_targets: List[Dict[str, Any]], key: str, fallback_key: str = "caption") -> List[str]:
    texts: List[str] = []
    for target in raw_targets:
        value = target.get(key, None)
        if not isinstance(value, str) or not value.strip():
            value = target.get(fallback_key, "object .")
        texts.append(str(value or "object ."))
    return texts


def _uses_stage_b_post_candidate_scorer(cfg) -> bool:
    return bool(
        getattr(cfg, "stage_b_v11_fixed_text", False)
        or getattr(cfg, "stage_b_v7", False)
    )


def _slot_scores(outputs: Dict[str, torch.Tensor], cfg, beta: float) -> torch.Tensor:
    if bool(getattr(cfg, "stage_b_native_patch_category", False)):
        _validate_native_patch_category_config(cfg)
        score = outputs.get(_NATIVE_PATCH_RANK_SCORE_KEY)
        if score is None:
            raise KeyError(
                "native patch-category Ref evaluation is missing "
                f"{_NATIVE_PATCH_RANK_SCORE_KEY}; fallback to generic slot "
                "logits is forbidden"
            )
        if not torch.is_tensor(score) or score.dim() != 2:
            shape = (
                tuple(score.shape)
                if torch.is_tensor(score)
                else type(score).__name__
            )
            raise ValueError(
                f"{_NATIVE_PATCH_RANK_SCORE_KEY} must be a (B,Q) tensor, got {shape}"
            )
        boxes = outputs.get("pred_boxes")
        if (
            not torch.is_tensor(boxes)
            or boxes.dim() != 3
            or int(boxes.shape[-1]) != 4
            or tuple(score.shape) != tuple(boxes.shape[:2])
        ):
            shape = (
                tuple(boxes.shape)
                if torch.is_tensor(boxes)
                else type(boxes).__name__
            )
            raise ValueError(
                f"{_NATIVE_PATCH_RANK_SCORE_KEY} must align with pred_boxes "
                f"(B,Q,4), got {shape}"
            )
        score = score.detach().float()
        if not bool(torch.isfinite(score).all().item()):
            raise ValueError(
                f"{_NATIVE_PATCH_RANK_SCORE_KEY} must contain only finite values"
            )
        return score.unsqueeze(-1)

    if bool(getattr(cfg, "stage_b_data_driven_score", False)):
        score = outputs.get(_DATA_DRIVEN_RANK_SCORE_KEY)
        if not torch.is_tensor(score) or score.dim() != 2:
            shape = tuple(score.shape) if torch.is_tensor(score) else type(score).__name__
            raise ValueError(
                f"{_DATA_DRIVEN_RANK_SCORE_KEY} must be a (B,Q) tensor, got {shape}"
            )
        score = score.detach().float()
        if not bool(torch.isfinite(score).all().item()):
            raise ValueError(
                f"{_DATA_DRIVEN_RANK_SCORE_KEY} must contain only finite values"
            )
        return score.unsqueeze(-1)

    if bool(getattr(cfg, "stage_b_u0_patch_rank", False)):
        score = outputs.get(_U0_RANK_SCORE_KEY, None)
        if score is None:
            raise KeyError(
                "Stage-B U0 Ref evaluation is missing "
                f"{_U0_RANK_SCORE_KEY}; fallback to the sealed GDINO rank "
                "score or generic slot logits is forbidden"
            )
        if not torch.is_tensor(score) or score.dim() != 2:
            shape = tuple(score.shape) if torch.is_tensor(score) else type(score).__name__
            raise ValueError(
                f"{_U0_RANK_SCORE_KEY} must be a (B,Q) tensor, got {shape}"
            )
        score = score.detach().float()
        if not bool(torch.isfinite(score).all().item()):
            raise ValueError(f"{_U0_RANK_SCORE_KEY} must contain only finite values")
        return score.unsqueeze(-1)

    gdino_rank_score = outputs.get("stage_b_gdino_rank_score", None)
    if (
        bool(getattr(cfg, "stage_b_gdino_score_adapter", False))
        and gdino_rank_score is None
    ):
        raise KeyError(
            "Stage-B GDINO adapter evaluation is missing stage_b_gdino_rank_score"
        )
    if gdino_rank_score is not None:
        score = gdino_rank_score.detach().float()
        if score.dim() == 2:
            score = score.unsqueeze(-1)
        if score.dim() != 3:
            raise ValueError(
                "stage_b_gdino_rank_score must be (B,Q) or (B,Q,K), "
                f"got {tuple(score.shape)}"
            )
        return score
    legacy_gate_score = outputs.get("stage_b_legacy_global_confidence", None)
    if legacy_gate_score is not None:
        score = legacy_gate_score.detach().float()
        if score.dim() == 2:
            score = score.unsqueeze(-1)
        if score.dim() != 3:
            raise ValueError(
                "stage_b_legacy_global_confidence must be (B,Q) or (B,Q,K), "
                f"got {tuple(score.shape)}"
            )
        return score
    if _uses_stage_b_post_candidate_scorer(cfg):
        score = None
        explicit_ownership = str(
            getattr(cfg, "stage_b_v22_score_ownership", "") or ""
        ).strip().lower().replace("-", "_")
        use_explicit_rank_score = explicit_ownership in {
            "shared_trunk_two_heads",
            "independent_decoders_joint",
            "independent_decoders_two_phase",
            "rank_tower_stopgrad_token_adapter_two_phase",
        }
        if bool(getattr(cfg, "stage_b_v15_decoupled_confidence", False)) or use_explicit_rank_score:
            score = outputs.get("stage_b_v15_dense_rank_score", None)
            if score is None:
                raise KeyError(
                    "Stage-B separate-rank Ref evaluation requires "
                    "stage_b_v15_dense_rank_score."
                )
        else:
            score = outputs.get("stage_b_v7_final_score", None)
        if score is None:
            score = outputs.get("stage_b_v7_predicate_score", None)
        if score is None:
            raise KeyError(
                "Stage-B post-candidate eval requires stage_b_v7_final_score "
                "or stage_b_v7_predicate_score."
            )
        score = score.detach().float()
        if score.dim() == 2:
            score = score.unsqueeze(-1)
        if score.dim() != 3:
            raise ValueError(f"stage_b_v7 score must be (B,Q) or (B,Q,K), got {tuple(score.shape)}")
        return score

    return compute_stage_b_slot_logits(
        outputs,
        beta=float(beta),
        canonical_weight=float(getattr(cfg, "stage_b_infer_canonical_weight", 1.0)),
        text_agg=str(getattr(cfg, "stage_b_infer_text_agg", "mean")),
        softmin_tau=float(getattr(cfg, "stage_b_infer_softmin_tau", 0.7)),
        mean_softmin_alpha=float(getattr(cfg, "stage_b_infer_mean_softmin_alpha", 0.5)),
        normalize_fused_score=bool(getattr(cfg, "stage_b_infer_normalize_fused_score", True)),
        score_mode=str(getattr(cfg, "stage_b_score_mode", "patch_text")),
    )


def _diagnostic_patch_rank_settings(
    values: Optional[Iterable[float]],
    cfg,
) -> Optional[Dict[str, Any]]:
    if values is None:
        return None
    if not bool(getattr(cfg, "stage_b_v11_fixed_text", False)):
        raise ValueError(
            "--diagnostic_patch_rank_weights requires stage_b_v11_fixed_text=true"
        )
    if bool(getattr(cfg, "stage_b_gdino_score_adapter", False)):
        raise ValueError(
            "--diagnostic_patch_rank_weights is unavailable for the GDINO score adapter"
        )
    if not bool(getattr(cfg, "stage_b_v15_patch_rank_fusion", False)):
        raise ValueError(
            "--diagnostic_patch_rank_weights requires the fixed scorer patch-rank "
            "fusion contract to be enabled"
        )
    candidate_topk = int(getattr(cfg, "stage_b_v11_candidate_topk", 0))
    if candidate_topk != 50:
        raise ValueError(
            "--diagnostic_patch_rank_weights preserves the fixed Top50 admission "
            f"contract and therefore requires stage_b_v11_candidate_topk=50, got {candidate_topk}"
        )

    weights = [float(value) for value in values]
    if not weights:
        raise ValueError("--diagnostic_patch_rank_weights requires at least one weight")
    if any(not math.isfinite(value) or value < 0.0 for value in weights):
        raise ValueError("diagnostic patch-rank weights must be finite and non-negative")
    if len(set(weights)) != len(weights):
        raise ValueError("diagnostic patch-rank weights must be unique")

    contract_weight = float(getattr(cfg, "stage_b_v15_patch_rank_weight", float("nan")))
    if not math.isfinite(contract_weight) or contract_weight < 0.0:
        raise ValueError(
            "stage_b_v15_patch_rank_weight must be finite and non-negative"
        )
    contract_matches = [
        value
        for value in weights
        if math.isclose(value, contract_weight, rel_tol=0.0, abs_tol=1e-12)
    ]
    if len(contract_matches) != 1:
        raise ValueError(
            "diagnostic patch-rank grid must contain the scorer contract weight "
            f"{contract_weight:g} exactly once"
        )
    return {
        "weights": weights,
        "contract_weight": contract_weight,
        "candidate_topk": candidate_topk,
    }


def _diagnostic_external_rank_transfer_requested(*values: Any) -> bool:
    return any(value is not None for value in values)


def _unique_finite_grid(
    values: Optional[Iterable[float]],
    *,
    label: str,
    strictly_positive: bool = False,
) -> List[float]:
    if values is None:
        raise ValueError(f"{label} must be provided explicitly")
    grid = [float(value) for value in values]
    if not grid:
        raise ValueError(f"{label} requires at least one value")
    if any(not math.isfinite(value) for value in grid):
        raise ValueError(f"{label} values must be finite")
    if strictly_positive:
        if any(value <= 0.0 for value in grid):
            raise ValueError(f"{label} values must be strictly positive")
    elif any(value < 0.0 for value in grid):
        raise ValueError(f"{label} values must be non-negative")
    if len(set(grid)) != len(grid):
        raise ValueError(f"{label} values must be unique")
    return grid


def _diagnostic_external_rank_transfer_grid(
    settings: Dict[str, Any],
) -> List[Dict[str, Any]]:
    transfer_variants: List[Tuple[str, Optional[float]]] = []
    for mode in settings["transfer_modes"]:
        if mode in {
            "nearest_iou",
            "top_query_nearest_candidate",
            "top_query_nearest_candidate_external_box",
        }:
            transfer_variants.append((mode, None))
        elif mode in {
            "max_score_iou_power",
            "max_score_iou_power_external_box",
        }:
            transfer_variants.extend(
                (mode, float(power)) for power in settings["iou_powers"]
            )
        else:  # Settings are validated before this helper is called.
            raise ValueError(f"unsupported external rank transfer mode: {mode!r}")
    return [
        {
            "transfer_mode": mode,
            "iou_power": power,
            "patch_weight": float(patch_weight),
            "text_weight": float(text_weight),
        }
        for mode, power in transfer_variants
        for patch_weight in settings["patch_weights"]
        for text_weight in settings["text_weights"]
    ]


def _diagnostic_external_rank_transfer_settings(
    *,
    external_config_path: Optional[str],
    external_checkpoint_path: Optional[str],
    transfer_modes: Optional[Iterable[str]],
    iou_powers: Optional[Iterable[float]],
    patch_weights: Optional[Iterable[float]],
    text_weights: Optional[Iterable[float]],
    patch_cfg,
    external_cfg,
    patch_config_path: str,
    patch_checkpoint_paths: Iterable[str],
    include_patch_internal_rank_identity: bool = False,
    include_external_gdino_base_identity: bool = False,
) -> Dict[str, Any]:
    required = {
        "--diagnostic_external_gdino_config": external_config_path,
        "--diagnostic_external_gdino_checkpoint": external_checkpoint_path,
        "--diagnostic_external_transfer_modes": transfer_modes,
        "--diagnostic_external_patch_weights": patch_weights,
        "--diagnostic_external_text_weights": text_weights,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "external-GDINO rank transfer is all-or-nothing; missing explicit "
            f"arguments: {missing}"
        )
    if not bool(getattr(patch_cfg, "stage_b_v11_fixed_text", False)):
        raise ValueError(
            "external-GDINO rank transfer requires a fixed-text patch model "
            "with stage_b_v11_fixed_text=true"
        )
    if bool(getattr(patch_cfg, "stage_b_gdino_score_adapter", False)):
        raise ValueError(
            "the patch component cannot reuse the external GDINO score adapter"
        )
    contract_weight = float(
        getattr(patch_cfg, "stage_b_v15_patch_rank_weight", float("nan"))
    )
    patch_contract = _diagnostic_patch_rank_settings(
        [contract_weight], patch_cfg
    )

    if external_cfg is None or not bool(
        getattr(external_cfg, "stage_b_gdino_score_adapter", False)
    ):
        raise ValueError(
            "the external config must enable stage_b_gdino_score_adapter=true"
        )
    if bool(getattr(external_cfg, "stage_b_v11_fixed_text", False)) or bool(
        getattr(external_cfg, "stage_b_v7", False)
    ):
        raise ValueError(
            "the external component must be ordinary non-patch GroundingDINO, "
            "not a fixed/post-candidate scorer"
        )
    configured_query_count = int(getattr(external_cfg, "num_queries", 0))
    if configured_query_count != _EXTERNAL_GDINO_QUERY_COUNT:
        raise ValueError(
            "external-GDINO rank transfer requires num_queries=900, got "
            f"{configured_query_count}"
        )

    modes = [str(value).strip() for value in transfer_modes or []]
    if not modes:
        raise ValueError(
            "--diagnostic_external_transfer_modes requires at least one mode"
        )
    unknown_modes = [
        value for value in modes if value not in _EXTERNAL_RANK_TRANSFER_MODES
    ]
    if unknown_modes:
        raise ValueError(
            "unsupported external rank transfer modes "
            f"{unknown_modes}; available={list(_EXTERNAL_RANK_TRANSFER_MODES)}"
        )
    if len(set(modes)) != len(modes):
        raise ValueError("external rank transfer modes must be unique")
    power_modes = {
        "max_score_iou_power",
        "max_score_iou_power_external_box",
    }
    if any(mode in power_modes for mode in modes):
        powers = _unique_finite_grid(
            iou_powers,
            label="--diagnostic_external_iou_powers",
            strictly_positive=True,
        )
    else:
        if iou_powers is not None:
            raise ValueError(
                "--diagnostic_external_iou_powers is only valid when "
                "an IoU-power transfer mode is selected"
            )
        powers = []
    patch_grid = _unique_finite_grid(
        patch_weights,
        label="--diagnostic_external_patch_weights",
    )
    text_grid = _unique_finite_grid(
        text_weights,
        label="--diagnostic_external_text_weights",
    )
    if not any(value > 0.0 for value in text_grid):
        raise ValueError(
            "the external text-weight grid must contain a positive value"
        )
    if 0.0 in patch_grid and 0.0 in text_grid:
        raise ValueError(
            "the Cartesian fusion grid cannot contain the undefined (patch=0, text=0) point"
        )

    patch_config = _file_identity(patch_config_path, label="patch config")
    external_config = _file_identity(
        str(external_config_path), label="external GDINO config"
    )
    if (
        patch_config["path"] == external_config["path"]
        or patch_config["sha256"] == external_config["sha256"]
    ):
        raise ValueError(
            "patch and external GDINO configs must be independent, non-identical files"
        )
    external_checkpoint = _file_identity(
        str(external_checkpoint_path), label="external GDINO checkpoint"
    )
    patch_checkpoints = [
        _file_identity(str(path), label="patch checkpoint")
        for path in patch_checkpoint_paths
    ]
    if not patch_checkpoints:
        raise ValueError("external-GDINO rank transfer requires a patch checkpoint")
    for component in patch_checkpoints:
        if (
            component["path"] == external_checkpoint["path"]
            or component["sha256"] == external_checkpoint["sha256"]
        ):
            raise ValueError(
                "patch and external GDINO checkpoints must be independent, "
                "non-identical files"
            )

    settings = {
        "transfer_modes": modes,
        "iou_powers": powers,
        "patch_weights": patch_grid,
        "text_weights": text_grid,
        "candidate_topk": int(patch_contract["candidate_topk"]),
        "contract_patch_rank_weight": float(patch_contract["contract_weight"]),
        "external_query_count": _EXTERNAL_GDINO_QUERY_COUNT,
        "include_patch_internal_rank_identity": bool(
            include_patch_internal_rank_identity
        ),
        "include_external_gdino_base_identity": bool(
            include_external_gdino_base_identity
        ),
        "patch_config": patch_config,
        "patch_checkpoints": patch_checkpoints,
        "external_config": external_config,
        "external_checkpoint": external_checkpoint,
        "transfer_contract": {
            "version": _EXTERNAL_RANK_TRANSFER_CONTRACT_VERSION,
            "candidate_source": "detached_patch_stage_b_v11_candidate_idx",
            "candidate_admission": "unchanged_exact_stage_a_top50",
            "candidate_box_source": "gather_patch_pred_boxes_at_candidate_idx",
            "external_caption_source": "target.caption_full_expression",
            "external_query_count": _EXTERNAL_GDINO_QUERY_COUNT,
            "external_rank_score_key": _EXTERNAL_GDINO_RANK_SCORE_KEY,
            "nearest_iou": "raw rank score at first argmax-IoU external query",
            "max_score_iou_power": (
                "max(rank_score * IoU**p) over strictly-positive-IoU queries"
            ),
            "max_score_iou_power_external_box": (
                "use the same fixed-candidate rank transfer and winner as "
                "max_score_iou_power, but emit the winning matched full-text "
                "GDINO query box instead of the patch candidate box"
            ),
            "top_query_nearest_candidate": (
                "assign each external query to its first argmax-IoU fixed candidate; "
                "candidate score is max assigned raw rank score"
            ),
            "top_query_nearest_candidate_external_box": (
                "use top_query_nearest_candidate scoring, but emit the winning "
                "assigned full-text GDINO query box instead of the patch candidate box"
            ),
            "top_query_unassigned_candidate_policy": (
                "fallback_to_sample_min_external_rank_score"
            ),
            "top_query_global_rank_tie_policy": (
                "first external argmax query's nearest candidate wins at patch_weight=0"
            ),
            "signed_score_policy": (
                "preserve signed rank scores without sigmoid/clamp/abs; zero-IoU "
                "queries are excluded so they cannot synthesize a zero winner"
            ),
            "all_zero_iou_policy": "fallback_to_nearest_iou_raw_rank_score",
            "box_format": "normalized_cxcywh; clamp converted xyxy to [0,1]",
            "fusion_formula": (
                "patch_weight * stage_b_v15_candidate_patch_logits + "
                "text_weight * transferred_stage_b_gdino_rank_score"
            ),
        },
        "patch_internal_rank_identity_contract": {
            "version": _PATCH_INTERNAL_RANK_IDENTITY_CONTRACT_VERSION,
            "descriptor_kind": _PATCH_INTERNAL_RANK_IDENTITY_KIND,
            "diagnostic_only": True,
            "formal_gate_eligible": False,
            "standard_ref_beta": 0.0,
            "score_source": "authoritative_patch_model_slot_scores",
            "winner": "bitwise_standard_ref_beta0_flat_argmax",
            "candidate_admission": "unchanged_exact_stage_a_top50",
            "prediction_box_source": "patch_model_pred_boxes_at_winning_query",
            "uses_external_rank_score": False,
            "uses_external_box": False,
            "uses_fusion_weights": False,
        },
        "external_gdino_base_identity_contract": {
            "version": _EXTERNAL_GDINO_BASE_IDENTITY_CONTRACT_VERSION,
            "descriptor_kind": _EXTERNAL_GDINO_BASE_IDENTITY_KIND,
            "identity_kind": _EXTERNAL_GDINO_BASE_IDENTITY_ID,
            "diagnostic_only": True,
            "formal_gate_eligible": False,
            "standard_ref_beta": 0.0,
            "score_source": _EXTERNAL_GDINO_BASE_SCORE_KEY,
            "winner": "first_argmax_over_full_external_query_axis",
            "query_domain": "all_900_external_gdino_queries",
            "prediction_box_source": "external_gdino_pred_boxes_at_winning_query",
            "uses_adapter_rank_residual": False,
            "uses_patch_top50_admission": False,
            "uses_top_query_mapping": False,
            "uses_fusion_weights": False,
        },
    }
    settings["fixed_grid"] = _diagnostic_external_rank_transfer_grid(settings)
    return settings


def _require_diagnostic_tensor(
    outputs: Dict[str, torch.Tensor],
    key: str,
) -> torch.Tensor:
    if key not in outputs:
        raise KeyError(f"patch-rank diagnostic requires output {key!r}")
    value = outputs[key]
    if not torch.is_tensor(value):
        raise TypeError(f"patch-rank diagnostic output {key!r} must be a tensor")
    return value.detach()


def _diagnostic_patch_rank_candidate_logits(
    outputs: Dict[str, torch.Tensor],
    *,
    weights: Iterable[float],
    contract_weight: float,
    require_patch_topk_order: bool = True,
) -> Tuple[Dict[float, torch.Tensor], torch.Tensor, torch.Tensor]:
    pred_boxes = _require_diagnostic_tensor(outputs, "pred_boxes")
    fused_logits = _require_diagnostic_tensor(
        outputs, "stage_b_v11_final_phrase_logits"
    )
    patch_logits = _require_diagnostic_tensor(
        outputs, "stage_b_v15_candidate_patch_logits"
    )
    candidate_idx = _require_diagnostic_tensor(outputs, "stage_b_v11_candidate_idx")
    expression_valid = _require_diagnostic_tensor(
        outputs, "stage_b_v11_expression_valid_mask"
    )
    dense_candidate_mask = _require_diagnostic_tensor(
        outputs, "stage_b_v11_candidate_mask"
    )
    dense_fused_logits = _require_diagnostic_tensor(
        outputs, "stage_b_v15_dense_rank_logits"
    )
    dense_fused_score = _require_diagnostic_tensor(
        outputs, "stage_b_v15_dense_rank_score"
    )

    if pred_boxes.dim() != 3 or int(pred_boxes.shape[-1]) != 4:
        raise ValueError(
            "patch-rank diagnostic pred_boxes must be (B,Q,4), got "
            f"{tuple(pred_boxes.shape)}"
        )
    if fused_logits.dim() != 3 or not fused_logits.is_floating_point():
        raise ValueError(
            "stage_b_v11_final_phrase_logits must be floating (B,N,K), got "
            f"shape={tuple(fused_logits.shape)} dtype={fused_logits.dtype}"
        )
    batch_size, candidate_count, slot_count = fused_logits.shape
    query_count = int(pred_boxes.shape[1])
    if batch_size <= 0 or candidate_count <= 0 or slot_count <= 0 or query_count <= 0:
        raise ValueError("patch-rank diagnostic dimensions must all be positive")
    if int(pred_boxes.shape[0]) != batch_size:
        raise ValueError("pred_boxes batch dimension must align with candidate logits")
    if tuple(patch_logits.shape) != (batch_size, candidate_count):
        raise ValueError(
            "stage_b_v15_candidate_patch_logits must be (B,N), expected "
            f"{(batch_size, candidate_count)}, got {tuple(patch_logits.shape)}"
        )
    if not patch_logits.is_floating_point():
        raise TypeError("stage_b_v15_candidate_patch_logits must be floating point")
    if tuple(candidate_idx.shape) != (batch_size, candidate_count):
        raise ValueError(
            "stage_b_v11_candidate_idx must be (B,N), expected "
            f"{(batch_size, candidate_count)}, got {tuple(candidate_idx.shape)}"
        )
    if candidate_idx.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError("stage_b_v11_candidate_idx must have an integer dtype")
    if tuple(expression_valid.shape) != (batch_size, slot_count):
        raise ValueError(
            "stage_b_v11_expression_valid_mask must be (B,K), expected "
            f"{(batch_size, slot_count)}, got {tuple(expression_valid.shape)}"
        )
    if expression_valid.dtype != torch.bool:
        raise TypeError("stage_b_v11_expression_valid_mask must have dtype bool")
    dense_shape = (batch_size, query_count, slot_count)
    for key, value in (
        ("stage_b_v11_candidate_mask", dense_candidate_mask),
        ("stage_b_v15_dense_rank_logits", dense_fused_logits),
        ("stage_b_v15_dense_rank_score", dense_fused_score),
    ):
        if tuple(value.shape) != dense_shape:
            raise ValueError(
                f"{key} must be {dense_shape}, got {tuple(value.shape)}"
            )
    if dense_candidate_mask.dtype != torch.bool:
        raise TypeError("stage_b_v11_candidate_mask must have dtype bool")
    if not dense_fused_logits.is_floating_point() or not dense_fused_score.is_floating_point():
        raise TypeError("dense rank logits and scores must be floating point")

    tensors = (
        fused_logits,
        patch_logits,
        candidate_idx,
        expression_valid,
        dense_candidate_mask,
        dense_fused_logits,
        dense_fused_score,
    )
    if any(value.device != pred_boxes.device for value in tensors):
        raise ValueError("all patch-rank diagnostic tensors must share one device")
    if not bool(expression_valid.any(dim=1).all().item()):
        raise ValueError("each diagnostic row must have at least one valid expression slot")
    if not bool(torch.isfinite(patch_logits).all().item()):
        raise ValueError("candidate patch logits must all be finite")
    valid_candidate_slots = expression_valid[:, None, :].expand_as(fused_logits)
    if not bool(torch.isfinite(fused_logits[valid_candidate_slots]).all().item()):
        raise ValueError("raw fused candidate logits must be finite on valid slots")

    candidate_idx_long = candidate_idx.to(dtype=torch.int64)
    if bool(((candidate_idx_long < 0) | (candidate_idx_long >= query_count)).any().item()):
        raise ValueError("stage_b_v11_candidate_idx contains an out-of-range query index")
    for row in candidate_idx_long:
        if int(torch.unique(row).numel()) != candidate_count:
            raise ValueError("stage_b_v11_candidate_idx must be unique within each row")
    if (
        bool(require_patch_topk_order)
        and candidate_count > 1
        and bool((patch_logits[:, 1:] > patch_logits[:, :-1]).any().item())
    ):
        raise ValueError(
            "candidate patch logits must preserve descending Stage-A Top-K order"
        )

    scatter_idx = candidate_idx_long.unsqueeze(-1).expand(-1, -1, slot_count)
    expected_dense_mask = torch.zeros_like(dense_candidate_mask)
    expected_dense_mask.scatter_(1, scatter_idx, True)
    expected_dense_mask &= expression_valid[:, None, :]
    if not torch.equal(dense_candidate_mask, expected_dense_mask):
        raise ValueError(
            "stage_b_v11_candidate_mask does not match candidate indices and valid slots"
        )

    gathered_dense_logits = torch.gather(dense_fused_logits, 1, scatter_idx)
    if not torch.equal(gathered_dense_logits, fused_logits):
        raise ValueError(
            "dense raw fused rank logits do not exactly match candidate fused logits"
        )
    expected_candidate_score = fused_logits.sigmoid().masked_fill(
        ~valid_candidate_slots, 0.0
    )
    gathered_dense_score = torch.gather(dense_fused_score, 1, scatter_idx)
    if not torch.equal(gathered_dense_score, expected_candidate_score):
        raise ValueError(
            "dense rank scores do not exactly match sigmoid(raw fused candidate logits)"
        )
    if bool((dense_fused_score.masked_select(~dense_candidate_mask) != 0.0).any().item()):
        raise ValueError("dense rank scores must be zero outside the admitted candidates")
    if not bool((expected_candidate_score[valid_candidate_slots] > 0.0).all().item()):
        raise ValueError(
            "contract rank scores must be positive on every admitted valid slot"
        )
    contract_weight = float(contract_weight)
    if not math.isfinite(contract_weight) or contract_weight < 0.0:
        raise ValueError("contract patch-rank weight must be finite and non-negative")
    normalized_weights = [float(value) for value in weights]
    text_logits = fused_logits - contract_weight * patch_logits.unsqueeze(-1)
    logits_by_weight: Dict[float, torch.Tensor] = {}
    for weight in normalized_weights:
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("diagnostic patch-rank weights must be finite and non-negative")
        if math.isclose(weight, contract_weight, rel_tol=0.0, abs_tol=1e-12):
            # Reuse the authoritative tensor so the contract point is bitwise identical.
            diagnostic_logits = fused_logits
        else:
            diagnostic_logits = text_logits + weight * patch_logits.unsqueeze(-1)
        diagnostic_logits = diagnostic_logits.masked_fill(
            ~valid_candidate_slots, torch.finfo(diagnostic_logits.dtype).min
        )
        logits_by_weight[weight] = diagnostic_logits
    return logits_by_weight, candidate_idx_long, expression_valid


def _normalized_cxcywh_to_xyxy(
    boxes: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    if not torch.is_tensor(boxes) or boxes.dim() != 3 or int(boxes.shape[-1]) != 4:
        shape = tuple(boxes.shape) if torch.is_tensor(boxes) else type(boxes).__name__
        raise ValueError(f"{name} must have shape (B,Q,4), got {shape}")
    if not boxes.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    detached = boxes.detach().float()
    if not bool(torch.isfinite(detached).all().item()):
        raise ValueError(f"{name} must be finite")
    if bool(((detached < 0.0) | (detached > 1.0)).any().item()):
        raise ValueError(f"{name} must contain normalized cxcywh values in [0,1]")
    if bool((detached[..., 2:] <= 0.0).any().item()):
        raise ValueError(f"{name} widths and heights must be strictly positive")
    return box_ops.box_cxcywh_to_xyxy(detached).clamp(0.0, 1.0)


def _transfer_external_rank_scores_from_iou(
    pair_iou: torch.Tensor,
    external_rank_score: torch.Tensor,
    *,
    transfer_modes: Iterable[str],
    iou_powers: Iterable[float],
    expected_query_count: Optional[int] = _EXTERNAL_GDINO_QUERY_COUNT,
) -> Dict[Tuple[str, Optional[float]], torch.Tensor]:
    """Transfer external query scores onto fixed candidates without admission.

    ``max_score_iou_power`` excludes exactly-zero IoU queries before its max.
    This matters for signed adapter outputs: otherwise ``negative * 0 == 0``
    would let an unrelated query beat every genuinely overlapping negative
    score.  A candidate with no positive overlap falls back to the deterministic
    nearest-IoU query (the first query wins an IoU tie).
    """

    if (
        not torch.is_tensor(pair_iou)
        or pair_iou.dim() != 3
        or not pair_iou.is_floating_point()
    ):
        shape = (
            tuple(pair_iou.shape)
            if torch.is_tensor(pair_iou)
            else type(pair_iou).__name__
        )
        raise ValueError(f"pair_iou must be floating (B,N,Q), got {shape}")
    if (
        not torch.is_tensor(external_rank_score)
        or external_rank_score.dim() != 2
        or not external_rank_score.is_floating_point()
    ):
        shape = (
            tuple(external_rank_score.shape)
            if torch.is_tensor(external_rank_score)
            else type(external_rank_score).__name__
        )
        raise ValueError(
            f"external rank score must be floating (B,Q), got {shape}"
        )
    batch_size, candidate_count, query_count = pair_iou.shape
    if batch_size <= 0 or candidate_count <= 0 or query_count <= 0:
        raise ValueError("pair_iou dimensions must all be positive")
    if tuple(external_rank_score.shape) != (batch_size, query_count):
        raise ValueError(
            "external rank score must align with pair_iou (B,Q), expected "
            f"{(batch_size, query_count)}, got {tuple(external_rank_score.shape)}"
        )
    if expected_query_count is not None and query_count != int(expected_query_count):
        raise ValueError(
            f"external query count must be {int(expected_query_count)}, got {query_count}"
        )
    if pair_iou.device != external_rank_score.device:
        raise ValueError("pair_iou and external rank score must share one device")
    iou = pair_iou.detach().float()
    score = external_rank_score.detach().float()
    if not bool(torch.isfinite(iou).all().item()) or bool(
        ((iou < 0.0) | (iou > 1.0)).any().item()
    ):
        raise ValueError("pair_iou must contain finite values in [0,1]")
    if not bool(torch.isfinite(score).all().item()):
        raise ValueError("external rank scores must all be finite")

    modes = [str(value) for value in transfer_modes]
    if not modes or len(set(modes)) != len(modes):
        raise ValueError("transfer modes must be a non-empty unique list")
    unknown = [mode for mode in modes if mode not in _EXTERNAL_RANK_TRANSFER_MODES]
    if unknown:
        raise ValueError(f"unsupported transfer modes: {unknown}")
    powers = [float(value) for value in iou_powers]
    if any(not math.isfinite(value) or value <= 0.0 for value in powers):
        raise ValueError("IoU powers must be finite and strictly positive")
    if len(set(powers)) != len(powers):
        raise ValueError("IoU powers must be unique")
    power_modes = {
        "max_score_iou_power",
        "max_score_iou_power_external_box",
    }
    has_power_mode = any(mode in power_modes for mode in modes)
    if has_power_mode and not powers:
        raise ValueError("IoU-power transfer modes require at least one IoU power")
    if not has_power_mode and powers:
        raise ValueError("IoU powers require an IoU-power transfer mode")

    nearest_index = iou.argmax(dim=-1)
    expanded_score = score[:, None, :].expand(-1, candidate_count, -1)
    nearest_score = torch.gather(
        expanded_score, -1, nearest_index.unsqueeze(-1)
    ).squeeze(-1)
    transferred: Dict[Tuple[str, Optional[float]], torch.Tensor] = {}
    for mode in modes:
        if mode == "nearest_iou":
            transferred[(mode, None)] = nearest_score
            continue
        if mode in {
            "top_query_nearest_candidate",
            "top_query_nearest_candidate_external_box",
        }:
            nearest_candidate = iou.argmax(dim=1)
            candidate_score = score.amin(dim=1, keepdim=True).expand(
                -1, candidate_count
            ).clone()
            candidate_score.scatter_reduce_(
                1,
                nearest_candidate,
                score,
                reduce="amax",
                include_self=True,
            )
            global_query = score.argmax(dim=1, keepdim=True)
            preferred_candidate = torch.gather(
                nearest_candidate, 1, global_query
            )
            global_score = torch.gather(score, 1, global_query)
            candidate_ids = torch.arange(
                candidate_count, device=score.device
            ).view(1, -1)
            nonpreferred_global_tie = (
                candidate_score == global_score
            ) & (candidate_ids != preferred_candidate)
            lower = torch.nextafter(
                global_score,
                torch.full_like(global_score, -torch.inf),
            )
            if bool(torch.isfinite(lower).all().item()):
                candidate_score = torch.where(
                    nonpreferred_global_tie, lower, candidate_score
                )
            else:
                higher = torch.nextafter(
                    global_score,
                    torch.full_like(global_score, torch.inf),
                )
                if not bool(torch.isfinite(higher).all().item()):
                    raise ValueError(
                        "cannot represent deterministic top-query rank tie policy"
                    )
                candidate_score.scatter_(1, preferred_candidate, higher)
            if not bool(torch.isfinite(candidate_score).all().item()):
                raise ValueError("transferred external rank scores must be finite")
            transferred[(mode, None)] = candidate_score
            continue
        positive_overlap = iou > 0.0
        has_positive_overlap = positive_overlap.any(dim=-1)
        for power in powers:
            weighted = expanded_score * iou.pow(power)
            weighted = weighted.masked_fill(~positive_overlap, -torch.inf)
            best = weighted.max(dim=-1).values
            best = torch.where(has_positive_overlap, best, nearest_score)
            if not bool(torch.isfinite(best).all().item()):
                raise ValueError("transferred external rank scores must be finite")
            transferred[(mode, power)] = best
    return transferred


def _diagnostic_external_rank_candidate_scores(
    patch_outputs: Dict[str, torch.Tensor],
    external_outputs: Dict[str, torch.Tensor],
    *,
    settings: Dict[str, Any],
) -> Tuple[
    Dict[Tuple[str, Optional[float]], torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    contract_weight = float(settings["contract_patch_rank_weight"])
    _logits, candidate_idx, expression_valid = (
        _diagnostic_patch_rank_candidate_logits(
            patch_outputs,
            weights=[contract_weight],
            contract_weight=contract_weight,
        )
    )
    batch_size, candidate_count = candidate_idx.shape
    expected_topk = int(settings["candidate_topk"])
    if candidate_count != expected_topk:
        raise ValueError(
            "external rank transfer must preserve exact configured candidate Top-K: "
            f"expected {expected_topk}, got {candidate_count}"
        )
    if tuple(expression_valid.shape) != (batch_size, 1) or not bool(
        expression_valid.all().item()
    ):
        raise ValueError(
            "external full-expression rank transfer requires exactly one valid "
            "expression slot per RefCOCO row"
        )

    patch_boxes_xyxy = _normalized_cxcywh_to_xyxy(
        _require_diagnostic_tensor(patch_outputs, "pred_boxes"),
        name="patch pred_boxes",
    )
    query_count = int(patch_boxes_xyxy.shape[1])
    if bool(((candidate_idx < 0) | (candidate_idx >= query_count)).any().item()):
        raise ValueError("fixed patch candidate indices are out of range")
    candidate_boxes = torch.gather(
        patch_boxes_xyxy,
        1,
        candidate_idx.unsqueeze(-1).expand(-1, -1, 4),
    )
    # Re-read the authoritative key after contract validation.  This is the
    # Stage-A patch score already used by the fixed scorer, not a recomputed proxy.
    patch_score = _require_diagnostic_tensor(
        patch_outputs, "stage_b_v15_candidate_patch_logits"
    ).float()
    if tuple(patch_score.shape) != (batch_size, candidate_count):
        raise ValueError("patch candidate score shape drifted after contract validation")

    if _EXTERNAL_GDINO_RANK_SCORE_KEY not in external_outputs:
        raise KeyError(
            "external-GDINO rank transfer requires adapter output "
            f"{_EXTERNAL_GDINO_RANK_SCORE_KEY!r}"
        )
    external_rank_score = external_outputs[_EXTERNAL_GDINO_RANK_SCORE_KEY]
    if not torch.is_tensor(external_rank_score):
        raise TypeError(
            f"external output {_EXTERNAL_GDINO_RANK_SCORE_KEY!r} must be a tensor"
        )
    external_rank_score = external_rank_score.detach()
    external_boxes = external_outputs.get("pred_boxes")
    if not torch.is_tensor(external_boxes):
        raise KeyError("external-GDINO rank transfer requires tensor output 'pred_boxes'")
    external_boxes_xyxy = _normalized_cxcywh_to_xyxy(
        external_boxes,
        name="external GDINO pred_boxes",
    )
    external_query_count = int(settings["external_query_count"])
    if tuple(external_boxes_xyxy.shape) != (
        batch_size,
        external_query_count,
        4,
    ):
        raise ValueError(
            "external GDINO pred_boxes must be exactly "
            f"{(batch_size, external_query_count, 4)}, got "
            f"{tuple(external_boxes_xyxy.shape)}"
        )
    if candidate_boxes.device != external_boxes_xyxy.device:
        raise ValueError("patch and external boxes must share one device")

    pair_iou = torch.stack(
        [
            box_ops.box_iou(candidate_boxes[row], external_boxes_xyxy[row])[0]
            for row in range(batch_size)
        ],
        dim=0,
    )
    transferred = _transfer_external_rank_scores_from_iou(
        pair_iou,
        external_rank_score,
        transfer_modes=settings["transfer_modes"],
        iou_powers=settings["iou_powers"],
        expected_query_count=external_query_count,
    )
    return (
        transferred,
        patch_score,
        candidate_idx.clone(),
        candidate_boxes.clone(),
        pair_iou.clone(),
        external_boxes_xyxy.clone(),
    )


@torch.no_grad()
def _forward(model, batch, device: torch.device, *, amp: bool, cfg):
    raw_targets = list(batch[1])
    root = model.module if hasattr(model, "module") else model
    stage_b_native_patch_category = _validate_native_patch_category_config(cfg)
    if stage_b_native_patch_category and not bool(
        getattr(root, "stage_b_native_patch_category", False)
    ):
        raise ValueError(
            "native patch-category Ref config requires a native patch-category model"
        )
    stage_b_u0_patch_rank = bool(
        getattr(cfg, "stage_b_u0_patch_rank", False)
    )
    u0_adapter = getattr(root, "stage_b_u0_patch_rank_adapter", None)
    if stage_b_u0_patch_rank and u0_adapter is None:
        raise ValueError(
            "Stage-B U0 Ref evaluation config requires a model U0 patch-rank adapter"
        )
    stage_b_data_driven_score = bool(
        getattr(cfg, "stage_b_data_driven_score", False)
    )
    data_driven_heads = getattr(root, "stage_b_data_driven_score_heads", None)
    if stage_b_data_driven_score and data_driven_heads is None:
        raise ValueError(
            "Stage-B data-driven Ref config requires its absolute score heads"
        )
    if stage_b_native_patch_category:
        support_keys = ("patch", "patches", "patch_global")
        routes = [
            tuple(key for key in support_keys if key in target)
            for target in raw_targets
        ]
        if not raw_targets or any(len(route) != 1 for route in routes) or len(
            set(routes)
        ) != 1:
            raise ValueError(
                "native patch-category Ref evaluation requires exactly one "
                "consistent support-patch representation per row"
            )
        captions = []
        for index, target in enumerate(raw_targets):
            caption = target.get("caption")
            if not isinstance(caption, str) or not caption.strip():
                raise KeyError(
                    f"native patch-category Ref target {index} requires a full caption"
                )
            captions.append(caption)
        phrase_mask = _native_patch_category_phrase_mask(raw_targets, device)
        (
            samples,
            targets,
            _captions,
            patches,
            patch_global,
            patch_mask,
        ) = _prepare_patch_batch(*batch, device)
        supports = [value for value in (patches, patch_global) if value is not None]
        if len(supports) != 1 or int(supports[0].shape[0]) != len(raw_targets):
            raise RuntimeError(
                "native patch-category Ref patch preparation did not yield one "
                "aligned support tensor"
            )
        if patch_mask is not None and (
            tuple(patch_mask.shape) != (len(raw_targets), 1)
            or not bool(patch_mask.all().item())
        ):
            raise RuntimeError(
                "native patch-category Ref requires one valid support slot per row"
            )
        with torch.cuda.amp.autocast(
            enabled=bool(amp) and device.type == "cuda"
        ):
            outputs = model(
                samples,
                targets=targets,
                captions=captions,
                patches=patches,
                patch_global=patch_global,
                patch_mask=patch_mask,
                patch_only=False,
                patch_only_compute_text_logits=True,
                disable_patch_dn=True,
                phrase_to_token_mask=phrase_mask,
            )
        if not isinstance(outputs, dict):
            raise TypeError(
                "native patch-category Ref forward must return a dictionary"
            )
        _derive_native_patch_category_ref_scores(outputs, phrase_mask)
        if _NATIVE_PATCH_RANK_SCORE_KEY not in outputs:
            raise KeyError(
                "native patch-category Ref forward did not produce its required rank key"
            )
        return outputs, targets
    if stage_b_data_driven_score:
        (
            samples,
            targets,
            _captions,
            patches,
            patch_global,
            patch_mask,
        ) = _prepare_patch_batch(*batch, device)
        canonical_captions = []
        expression_captions = []
        for index, target in enumerate(raw_targets):
            canonical = target.get("stage_a_caption")
            expression = target.get("caption")
            if not isinstance(canonical, str) or not canonical.strip():
                raise KeyError(
                    f"data-driven Ref target {index} requires stage_a_caption"
                )
            if not isinstance(expression, str) or not expression.strip():
                raise KeyError(
                    f"data-driven Ref target {index} requires a full caption"
                )
            canonical_captions.append(canonical)
            expression_captions.append(expression)
        with torch.cuda.amp.autocast(
            enabled=bool(amp) and device.type == "cuda"
        ):
            outputs = model(
                samples,
                captions=canonical_captions,
                patches=patches,
                patch_global=patch_global,
                patch_mask=patch_mask,
                patch_only=False,
                disable_patch_dn=True,
                stage_b_data_driven_expression_captions=expression_captions,
            )
        if not isinstance(outputs, dict) or _DATA_DRIVEN_RANK_SCORE_KEY not in outputs:
            raise KeyError(
                "data-driven Ref forward did not produce its required rank score"
            )
        return outputs, targets
    if (
        getattr(root, "stage_b_gdino_score_adapter", None) is not None
        and not stage_b_u0_patch_rank
    ):
        samples = batch[0].to(device)
        targets = [
            {
                key: value.to(device)
                for key, value in target.items()
                if torch.is_tensor(value)
                and key not in {"patch", "patches", "patch_global"}
            }
            for target in raw_targets
        ]
        captions = _target_texts(raw_targets, "caption")
        with torch.cuda.amp.autocast(
            enabled=bool(amp) and device.type == "cuda"
        ):
            outputs = model(samples, captions=captions)
        return outputs, targets

    if stage_b_u0_patch_rank:
        support_keys = ("patch", "patches", "patch_global")
        routes = [
            tuple(key for key in support_keys if key in target)
            for target in raw_targets
        ]
        if any(len(route) != 1 for route in routes) or len(set(routes)) != 1:
            raise ValueError(
                "Stage-B U0 Ref evaluation requires exactly one consistent "
                "support-patch representation per row"
            )

    samples, targets, captions, patches, patch_global, patch_mask = _prepare_patch_batch(*batch, device)
    if _uses_stage_b_post_candidate_scorer(cfg):
        stage_a_captions = _target_texts(raw_targets, "stage_a_caption")
        verifier_captions = _target_texts(raw_targets, "verifier_caption")
        kmax = int(patch_mask.shape[1]) if patch_mask is not None else 1
        phrase_mask = _pad_target_mask(raw_targets, "phrase_to_token_mask", kmax, device)
        canonical_mask = _pad_target_mask(raw_targets, "canonical_to_token_mask", kmax, device)
        with torch.cuda.amp.autocast(enabled=bool(amp) and device.type == "cuda"):
            outputs = model(
                samples,
                targets=targets,
                captions=stage_a_captions,
                patches=patches,
                patch_global=patch_global,
                patch_mask=patch_mask,
                patch_only=True,
                patch_only_compute_text_logits=False,
                disable_patch_dn=True,
                return_stage_b_v7_features=True,
                stage_b_v7_verifier_captions=verifier_captions,
                phrase_to_token_mask=phrase_mask,
                canonical_to_token_mask=canonical_mask,
            )
        if phrase_mask is not None:
            outputs["phrase_to_token_mask"] = phrase_mask
        if canonical_mask is not None:
            outputs["canonical_to_token_mask"] = canonical_mask
        outputs["stage_a_captions"] = stage_a_captions
        outputs["verifier_captions"] = verifier_captions
        return outputs, targets

    with torch.cuda.amp.autocast(enabled=bool(amp) and device.type == "cuda"):
        outputs = model(
            samples,
            targets=targets,
            captions=captions,
            patches=patches,
            patch_global=patch_global,
            patch_mask=patch_mask,
            patch_only=not stage_b_u0_patch_rank,
            patch_only_compute_text_logits=True,
        )
    if stage_b_u0_patch_rank and (
        not isinstance(outputs, dict) or _U0_RANK_SCORE_KEY not in outputs
    ):
        raise KeyError(
            "Stage-B U0 Ref forward did not produce the required rank key "
            f"{_U0_RANK_SCORE_KEY!r}"
        )
    if outputs["pred_logits_patch"].dim() == 2:
        kmax = 1
    else:
        kmax = int(outputs["pred_logits_patch"].shape[-1])
    phrase_mask = _pad_target_mask(raw_targets, "phrase_to_token_mask", kmax, device)
    canonical_mask = _pad_target_mask(raw_targets, "canonical_to_token_mask", kmax, device)
    if phrase_mask is not None:
        outputs["phrase_to_token_mask"] = phrase_mask
    if canonical_mask is not None:
        outputs["canonical_to_token_mask"] = canonical_mask
    return outputs, targets


@torch.no_grad()
def _forward_external_gdino_rank_adapter(
    model,
    batch,
    device: torch.device,
    *,
    amp: bool,
    cfg,
) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
    if not bool(getattr(cfg, "stage_b_gdino_score_adapter", False)):
        raise ValueError(
            "external rank forward requires a GDINO score-adapter config"
        )
    root = model.module if hasattr(model, "module") else model
    if getattr(root, "stage_b_gdino_score_adapter", None) is None:
        raise ValueError(
            "external rank forward model has no stage_b_gdino_score_adapter"
        )
    raw_targets = list(batch[1])
    captions: List[str] = []
    for row, target in enumerate(raw_targets):
        caption = target.get("caption")
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError(
                "external full-expression forward requires a non-empty target.caption "
                f"for row {row}"
            )
        captions.append(caption)
    samples = batch[0].to(device)
    targets = [
        {
            key: value.to(device)
            for key, value in target.items()
            if torch.is_tensor(value)
            and key not in {"patch", "patches", "patch_global"}
        }
        for target in raw_targets
    ]
    with torch.cuda.amp.autocast(enabled=bool(amp) and device.type == "cuda"):
        outputs = model(samples, captions=captions)
    if not isinstance(outputs, dict):
        raise TypeError("external GDINO adapter forward must return a dict")
    if _EXTERNAL_GDINO_RANK_SCORE_KEY not in outputs:
        raise KeyError(
            "external GDINO adapter forward did not produce the required rank key "
            f"{_EXTERNAL_GDINO_RANK_SCORE_KEY!r}"
        )
    return outputs, targets


def _validate_dual_forward_targets(
    patch_targets: List[Dict[str, torch.Tensor]],
    external_targets: List[Dict[str, torch.Tensor]],
) -> None:
    if len(patch_targets) != len(external_targets):
        raise ValueError("patch and external forward target batches do not align")
    for row, (patch_target, external_target) in enumerate(
        zip(patch_targets, external_targets)
    ):
        patch_boxes = patch_target.get("boxes")
        external_boxes = external_target.get("boxes")
        if not torch.is_tensor(patch_boxes) or not torch.is_tensor(external_boxes):
            raise ValueError(f"dual forward row {row} is missing target boxes")
        if not torch.equal(patch_boxes.detach(), external_boxes.detach()):
            raise ValueError(
                f"patch and external forward target boxes drifted for row {row}"
            )


def _validate_formal_external_caption_provenance(
    raw_targets: List[Dict[str, Any]],
    manifest: EvalManifest,
    start_index: int,
    *,
    settings: Dict[str, Any],
    routed_canonical_id_to_caption: Optional[Dict[int, str]] = None,
) -> List[str]:
    if settings.get("caption_provenance") != CAPTION_PROVENANCE_CONTRACT:
        raise ValueError("formal external caption provenance contract drifted")
    validated_full_expressions: List[str] = []
    for local_index, target in enumerate(raw_targets):
        manifest_index = int(start_index) + local_index
        if manifest_index >= manifest.size:
            raise IndexError("formal caption provenance exceeds the Ref manifest")
        source = manifest.rows[manifest_index]
        instances = source.get("instances")
        if (
            not isinstance(instances, list)
            or len(instances) != 1
            or not isinstance(instances[0], dict)
        ):
            raise ValueError(
                f"formal Ref manifest row {manifest_index} must contain one instance"
            )
        raw_phrase = instances[0].get("raw_phrase")
        if not isinstance(raw_phrase, str) or not raw_phrase.strip():
            raise ValueError(
                f"formal Ref manifest row {manifest_index} lacks instances[0].raw_phrase"
            )
        caption = target.get("caption")
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError(
                f"formal Ref target row {manifest_index} lacks a full-expression caption"
            )
        expected_caption = f"{_clean_phrase(raw_phrase)} ."
        observed_caption = _WS_RE.sub(" ", caption.strip())
        if observed_caption != expected_caption:
            raise ValueError(
                "formal external caption/manifest provenance drift at row "
                f"{manifest_index}: {observed_caption!r} != {expected_caption!r}"
            )
        validated_full_expressions.append(observed_caption)
        if routed_canonical_id_to_caption is not None:
            class_id = instances[0].get("class_id")
            if isinstance(class_id, bool) or not isinstance(class_id, int):
                raise ValueError(
                    f"formal routed Ref manifest row {manifest_index} lacks "
                    "an exact integer instances[0].class_id"
                )
            expected_stage_a_caption = routed_canonical_id_to_caption.get(
                class_id, "object"
            )
            stage_a_caption = target.get("stage_a_caption")
            if not isinstance(stage_a_caption, str) or not stage_a_caption.strip():
                raise ValueError(
                    f"formal routed Ref target row {manifest_index} lacks a "
                    "non-empty stage_a_caption"
                )
            observed_stage_a_caption = _norm_text(stage_a_caption)
            if observed_stage_a_caption != expected_stage_a_caption:
                raise ValueError(
                    "formal routed stage_a_caption/canonical-class provenance "
                    f"drift at row {manifest_index}: "
                    f"{observed_stage_a_caption!r} != {expected_stage_a_caption!r}"
                )
    return validated_full_expressions


def _box_iou_one(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.detach().float().view(4)
    b = b.detach().float().view(4)
    x1 = torch.maximum(a[0], b[0])
    y1 = torch.maximum(a[1], b[1])
    x2 = torch.minimum(a[2], b[2])
    y2 = torch.minimum(a[3], b[3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_a = (a[2] - a[0]).clamp(min=0) * (a[3] - a[1]).clamp(min=0)
    area_b = (b[2] - b[0]).clamp(min=0) * (b[3] - b[1]).clamp(min=0)
    return float((inter / (area_a + area_b - inter).clamp(min=1e-6)).item())


def _data_driven_query_diagnostic_values(
    outputs: Dict[str, torch.Tensor],
    pred_boxes_xyxy: torch.Tensor,
    gt_box_xyxy: torch.Tensor,
    *,
    batch_index: int,
    ranked_query_indices: torch.Tensor,
) -> Dict[str, Any]:
    required = {
        "final_score": _DATA_DRIVEN_RANK_SCORE_KEY,
        "raw_text_score": "stage_b_data_driven_text_rank_score",
        "candidate_mask": "stage_b_data_driven_candidate_mask",
        "gap3_eligible_mask": (
            "stage_b_data_driven_category_gate_eligible_mask"
        ),
        "patch_z_score": "stage_b_data_driven_category_gate_patch_score",
        "raw_patch_score": "pred_logits_patch",
    }
    values: Dict[str, torch.Tensor] = {}
    for label, key in required.items():
        value = outputs.get(key)
        if not torch.is_tensor(value):
            raise KeyError(
                f"data-driven query diagnostics require tensor output {key!r}"
            )
        if label == "raw_patch_score" and value.dim() == 3:
            if int(value.shape[-1]) != 1:
                raise ValueError(
                    "data-driven raw patch score must have a singleton slot"
                )
            value = value[..., 0]
        if value.dim() != 2:
            raise ValueError(
                f"data-driven diagnostic {key!r} must have shape (B,Q)"
            )
        values[label] = value.detach()

    reference_shape = tuple(values["final_score"].shape)
    if any(tuple(value.shape) != reference_shape for value in values.values()):
        raise ValueError("data-driven diagnostic query tensors are not aligned")
    if tuple(pred_boxes_xyxy.shape[:2]) != reference_shape:
        raise ValueError("data-driven diagnostic boxes are not query-aligned")
    if batch_index < 0 or batch_index >= reference_shape[0]:
        raise IndexError("data-driven diagnostic batch index is out of range")

    candidate = values["candidate_mask"][batch_index].to(dtype=torch.bool)
    eligible = values["gap3_eligible_mask"][batch_index].to(dtype=torch.bool)
    if bool((eligible & ~candidate).any().item()):
        raise ValueError("Gap3 eligibility escaped the candidate mask")
    if not bool(candidate.any().item()) or not bool(eligible.any().item()):
        raise ValueError("data-driven diagnostics require non-empty candidate rows")

    boxes = pred_boxes_xyxy[batch_index]
    iou = box_ops.box_iou(boxes, gt_box_xyxy.view(1, 4))[0][:, 0]
    if not bool(torch.isfinite(iou).all().item()):
        raise ValueError("data-driven diagnostic IoUs must be finite")
    final_score = values["final_score"][batch_index].float()
    raw_text_score = values["raw_text_score"][batch_index].float()
    patch_z_score = values["patch_z_score"][batch_index].float()
    raw_patch_score = values["raw_patch_score"][batch_index].float()
    for label, value in (
        ("final", final_score),
        ("raw text", raw_text_score),
        ("patch z", patch_z_score),
        ("raw patch", raw_patch_score),
    ):
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"data-driven diagnostic {label} scores must be finite")

    masked_final_score = final_score.masked_fill(~candidate, -torch.inf)
    final_argmax = int(masked_final_score.argmax().item())
    final_max = masked_final_score.amax()
    ranked_query_indices = ranked_query_indices.detach().reshape(-1).to(
        device=final_score.device, dtype=torch.int64
    )
    if int(ranked_query_indices.numel()) <= 0:
        raise ValueError("data-driven diagnostics require at least one ranked query")
    final_winner = int(ranked_query_indices[0].item())
    if not bool(candidate[final_winner].item()) or not bool(
        final_score[final_winner] == final_max
    ):
        raise RuntimeError(
            "data-driven evaluated winner does not attain the maximum score"
        )
    raw_text_winner = int(
        raw_text_score.masked_fill(~candidate, -torch.inf).argmax().item()
    )
    patch_winner = int(
        patch_z_score.masked_fill(~candidate, -torch.inf).argmax().item()
    )
    gt_best = int(iou.masked_fill(~candidate, -torch.inf).argmax().item())
    gate_oracle = int(iou.masked_fill(~eligible, -torch.inf).argmax().item())

    positive = candidate & (iou >= 0.5)
    ambiguous = candidate & (iou >= 0.3) & (iou < 0.5)

    def role(index: int) -> str:
        value = float(iou[index].item())
        if value >= 0.5:
            return "primary_positive"
        if value >= 0.3:
            return "primary_ambiguous"
        return "primary_negative"

    def snapshot(prefix: str, index: int) -> Dict[str, Any]:
        return {
            f"{prefix}_query": int(index),
            f"{prefix}_iou": float(iou[index].item()),
            f"{prefix}_role": role(index),
            f"{prefix}_gap3_eligible": bool(eligible[index].item()),
            f"{prefix}_raw_text_score": float(raw_text_score[index].item()),
            f"{prefix}_final_score": float(final_score[index].item()),
            f"{prefix}_patch_z_score": float(patch_z_score[index].item()),
            f"{prefix}_raw_patch_score": float(raw_patch_score[index].item()),
        }

    top_count = min(10, int(ranked_query_indices.numel()))
    top_queries = [
        int(index) for index in ranked_query_indices[:top_count].tolist()
    ]
    result: Dict[str, Any] = {
        "data_driven_query_diagnostic_contract": (
            _DATA_DRIVEN_QUERY_DIAGNOSTIC_CONTRACT
        ),
        "data_driven_candidate_queries": int(candidate.sum().item()),
        "data_driven_gap3_eligible_queries": int(eligible.sum().item()),
        "data_driven_primary_positive_queries": int(positive.sum().item()),
        "data_driven_primary_ambiguous_queries": int(ambiguous.sum().item()),
        "data_driven_gap3_positive_queries": int((positive & eligible).sum().item()),
        "data_driven_gap3_oracle_correct50": bool(iou[gate_oracle].item() >= 0.5),
        "data_driven_final_score_argmax_query": int(final_argmax),
        "data_driven_final_score_tied_max_queries": int(
            (candidate & (final_score == final_max)).sum().item()
        ),
        "data_driven_top_queries": top_queries,
        "data_driven_top_query_ious": [
            float(iou[index].item()) for index in top_queries
        ],
    }
    result.update(snapshot("data_driven_final_winner", final_winner))
    result.update(snapshot("data_driven_raw_text_winner", raw_text_winner))
    result.update(snapshot("data_driven_patch_winner", patch_winner))
    result.update(snapshot("data_driven_gt_best", gt_best))
    result.update(snapshot("data_driven_gap3_oracle", gate_oracle))
    return result


class RefExpAccumulator:
    def __init__(
        self,
        betas: Iterable[float],
        topks: Iterable[int],
        *,
        manifest: EvalManifest,
        run_prefix: str,
        data_driven_query_diagnostics: bool = False,
    ) -> None:
        self.betas = [float(b) for b in betas]
        self.topks = sorted({max(1, int(k)) for k in topks})
        self.manifest = manifest
        self.run_prefix = str(run_prefix)
        self.data_driven_query_diagnostics = bool(
            data_driven_query_diagnostics
        )
        self.total = 0
        self.iou_sum = {b: 0.0 for b in self.betas}
        self.correct50 = {b: 0 for b in self.betas}
        self.correct25 = {b: 0 for b in self.betas}
        self.recall = {(b, k): 0 for b in self.betas for k in self.topks}
        self.eval_records: Dict[float, List[Dict[str, Any]]] = {b: [] for b in self.betas}

    def update(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        *,
        cfg,
    ) -> None:
        if self.data_driven_query_diagnostics and not (
            bool(getattr(cfg, "stage_b_data_driven_score", False))
            and bool(
                getattr(cfg, "stage_b_data_driven_category_gate", False)
            )
        ):
            raise ValueError(
                "data-driven query diagnostics require the active Gap3 route"
            )
        start_index = int(self.total)
        pred_boxes = box_ops.box_cxcywh_to_xyxy(outputs["pred_boxes"].detach().float()).clamp(0.0, 1.0)
        diagnostic_cache: Dict[int, Dict[str, Any]] = {}
        for beta in self.betas:
            slot_logits = _slot_scores(outputs, cfg, beta)
            bsz, q, k = slot_logits.shape
            flat = slot_logits.reshape(bsz, q * k)
            max_topk = min(max(self.topks), int(q * k))
            top_vals, top_idx = torch.topk(flat, k=max_topk, dim=1, largest=True)
            del top_vals
            query_idx = torch.div(top_idx, k, rounding_mode="floor")
            for b, target in enumerate(targets):
                if beta == self.betas[0]:
                    self.total += 1
                gt_boxes = target["boxes"].detach().float()
                if gt_boxes.numel() == 0:
                    self.eval_records[beta].append(
                        make_eval_record(
                            self.manifest,
                            index=start_index + b,
                            run_id=f"{self.run_prefix}:b{beta:g}",
                            valid=False,
                            values={"beta": float(beta), "top1_iou": None, "correct50": None},
                        )
                    )
                    continue
                gt = box_ops.box_cxcywh_to_xyxy(gt_boxes[:1]).clamp(0.0, 1.0)[0]
                top_queries = query_idx[b]
                ious = torch.stack([box_ops.box_iou(pred_boxes[b, qi : qi + 1], gt.view(1, 4))[0].view(-1)[0] for qi in top_queries])
                best_iou = float(ious[0].item()) if ious.numel() else 0.0
                valid = math.isfinite(best_iou)
                record_values: Dict[str, Any] = {
                    "beta": float(beta),
                    "top1_iou": best_iou,
                    "correct50": bool(best_iou >= 0.5) if valid else None,
                }
                if self.data_driven_query_diagnostics:
                    if b not in diagnostic_cache:
                        diagnostic_cache[b] = _data_driven_query_diagnostic_values(
                            outputs,
                            pred_boxes,
                            gt,
                            batch_index=b,
                            ranked_query_indices=top_queries,
                        )
                    record_values.update(diagnostic_cache[b])
                self.eval_records[beta].append(
                    make_eval_record(
                        self.manifest,
                        index=start_index + b,
                        run_id=f"{self.run_prefix}:b{beta:g}",
                        valid=valid,
                        values=record_values,
                    )
                )
                self.iou_sum[beta] += best_iou
                if best_iou >= 0.5:
                    self.correct50[beta] += 1
                if best_iou >= 0.25:
                    self.correct25[beta] += 1
                for topk in self.topks:
                    if bool((ious[: min(topk, ious.numel())] >= 0.5).any().item()):
                        self.recall[(beta, topk)] += 1

    def results(self) -> List[Dict[str, Any]]:
        out = []
        denom = max(1, int(self.total))
        for beta in self.betas:
            row = {
                "beta": float(beta),
                "num_expressions": int(self.total),
                "acc50": float(self.correct50[beta] / denom),
                "acc25": float(self.correct25[beta] / denom),
                "mean_iou_top1": float(self.iou_sum[beta] / denom),
            }
            for topk in self.topks:
                row[f"recall50@{topk}"] = float(self.recall[(beta, topk)] / denom)
            out.append(row)
        return out


class DiagnosticPatchRankAccumulator:
    def __init__(
        self,
        weights: Iterable[float],
        contract_weight: float,
        expected_candidate_topk: int,
        contract_selection_topk: int,
    ) -> None:
        self.weights = [float(value) for value in weights]
        self.contract_weight = float(contract_weight)
        self.expected_candidate_topk = int(expected_candidate_topk)
        self.contract_selection_topk = int(contract_selection_topk)
        if self.contract_selection_topk <= 0:
            raise ValueError("contract selection Top-K must be positive")
        self.total = 0
        self.candidate_topk: Optional[int] = None
        self.iou_sum = {weight: 0.0 for weight in self.weights}
        self.correct50 = {weight: 0 for weight in self.weights}
        self.correct25 = {weight: 0 for weight in self.weights}
        self.candidate_oracle_correct50 = 0

    def update(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> None:
        logits_by_weight, candidate_idx, expression_valid = (
            _diagnostic_patch_rank_candidate_logits(
                outputs,
                weights=self.weights,
                contract_weight=self.contract_weight,
            )
        )
        batch_size, candidate_count = candidate_idx.shape
        slot_count = int(expression_valid.shape[1])
        if len(targets) != batch_size:
            raise ValueError(
                "patch-rank diagnostic targets must align with output batch size"
            )
        if candidate_count != self.expected_candidate_topk:
            raise ValueError(
                "patch-rank diagnostic did not preserve the configured candidate "
                f"Top-K: expected {self.expected_candidate_topk}, got {candidate_count}"
            )
        if self.candidate_topk is None:
            self.candidate_topk = int(candidate_count)
        elif self.candidate_topk != int(candidate_count):
            raise ValueError(
                "patch-rank diagnostic candidate count changed across batches: "
                f"{self.candidate_topk} vs {candidate_count}"
            )

        pred_boxes = box_ops.box_cxcywh_to_xyxy(
            outputs["pred_boxes"].detach().float()
        ).clamp(0.0, 1.0)
        candidate_boxes = torch.gather(
            pred_boxes,
            1,
            candidate_idx.unsqueeze(-1).expand(-1, -1, 4),
        )
        valid_candidate_slots = expression_valid[:, None, :].expand(
            -1, candidate_count, -1
        )
        dense_contract_score = outputs["stage_b_v15_dense_rank_score"].detach().float()
        flat_contract_score = dense_contract_score.reshape(batch_size, -1)
        contract_top_idx = torch.topk(
            flat_contract_score,
            k=min(self.contract_selection_topk, int(flat_contract_score.shape[1])),
            dim=1,
            largest=True,
        ).indices
        contract_winner_query = torch.div(
            contract_top_idx[:, 0], slot_count, rounding_mode="floor"
        )
        for batch_idx, target in enumerate(targets):
            self.total += 1
            gt_boxes = target["boxes"].detach().float()
            if gt_boxes.numel() == 0:
                continue
            gt = box_ops.box_cxcywh_to_xyxy(gt_boxes[:1]).clamp(0.0, 1.0)
            candidate_ious = box_ops.box_iou(candidate_boxes[batch_idx], gt)[0].view(-1)
            if not bool(torch.isfinite(candidate_ious).all().item()):
                raise ValueError("patch-rank diagnostic candidate IoUs must be finite")
            if bool((candidate_ious >= 0.5).any().item()):
                self.candidate_oracle_correct50 += 1

            for weight in self.weights:
                if math.isclose(
                    weight,
                    self.contract_weight,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    winner_query = int(contract_winner_query[batch_idx].item())
                    top1_iou = float(
                        box_ops.box_iou(
                            pred_boxes[batch_idx, winner_query : winner_query + 1],
                            gt,
                        )[0]
                        .view(-1)[0]
                        .item()
                    )
                else:
                    flat_logits = logits_by_weight[weight][batch_idx].masked_fill(
                        ~valid_candidate_slots[batch_idx],
                        torch.finfo(logits_by_weight[weight].dtype).min,
                    ).reshape(-1)
                    winner_flat_idx = int(torch.argmax(flat_logits).item())
                    winner_candidate_idx = winner_flat_idx // slot_count
                    top1_iou = float(candidate_ious[winner_candidate_idx].item())
                self.iou_sum[weight] += top1_iou
                if top1_iou >= 0.5:
                    self.correct50[weight] += 1
                if top1_iou >= 0.25:
                    self.correct25[weight] += 1

    def results(self) -> List[Dict[str, Any]]:
        denom = max(1, int(self.total))
        candidate_topk = int(self.candidate_topk or 0)
        oracle = float(self.candidate_oracle_correct50 / denom)
        return [
            {
                "diagnostic_only": True,
                "formal_gate_eligible": False,
                "diagnostic_patch_rank_weight": float(weight),
                "contract_patch_rank_weight": float(self.contract_weight),
                "num_expressions": int(self.total),
                "acc50": float(self.correct50[weight] / denom),
                "acc25": float(self.correct25[weight] / denom),
                "mean_iou_top1": float(self.iou_sum[weight] / denom),
                "recall50@1": float(self.correct50[weight] / denom),
                "candidate_oracle_recall50": oracle,
                "candidate_topk": candidate_topk,
            }
            for weight in self.weights
        ]


def _external_grid_key(row: Dict[str, Any]) -> Tuple[str, Optional[float], float, float]:
    power = row.get("iou_power")
    return (
        str(row["transfer_mode"]),
        None if power is None else float(power),
        float(row["patch_weight"]),
        float(row["text_weight"]),
    )


def _external_rank_transfer_fused_candidates(
    patch_outputs: Dict[str, torch.Tensor],
    external_outputs: Dict[str, torch.Tensor],
    *,
    settings: Dict[str, Any],
    include_matched_queries: bool = False,
) -> Any:
    (
        transferred,
        patch_score,
        candidate_idx,
        candidate_boxes,
        pair_iou,
        external_boxes,
    ) = (
        _diagnostic_external_rank_candidate_scores(
            patch_outputs,
            external_outputs,
            settings=settings,
        )
    )
    batch_size, candidate_count = candidate_idx.shape
    fused_by_grid: Dict[
        Tuple[str, Optional[float], float, float], torch.Tensor
    ] = {}
    prediction_boxes_by_transfer: Dict[
        Tuple[str, Optional[float]], torch.Tensor
    ] = {}
    matched_queries_by_transfer: Dict[
        Tuple[str, Optional[float]], torch.Tensor
    ] = {}
    external_rank_score = _require_diagnostic_tensor(
        external_outputs, _EXTERNAL_GDINO_RANK_SCORE_KEY
    ).detach().float()
    for descriptor in settings["fixed_grid"]:
        key = _external_grid_key(descriptor)
        transfer_key = (key[0], key[1])
        if transfer_key not in transferred:
            raise KeyError(f"missing transferred score grid point {transfer_key}")
        fused = key[2] * patch_score + key[3] * transferred[transfer_key]
        if tuple(fused.shape) != (batch_size, candidate_count) or not bool(
            torch.isfinite(fused).all().item()
        ):
            raise ValueError("external rank-transfer fused candidate scores are invalid")
        fused_by_grid[key] = fused
        if transfer_key in prediction_boxes_by_transfer:
            continue
        if key[0] in {
            "max_score_iou_power",
            "max_score_iou_power_external_box",
        }:
            if key[1] is None:
                raise ValueError("max-score transfer requires an IoU power")
            positive_overlap = pair_iou > 0.0
            has_positive_overlap = positive_overlap.any(dim=-1)
            weighted = external_rank_score[:, None, :] * pair_iou.pow(key[1])
            weighted = weighted.masked_fill(~positive_overlap, -torch.inf)
            selected_query = weighted.argmax(dim=-1)
            nearest_query = pair_iou.argmax(dim=-1)
            selected_query = torch.where(
                has_positive_overlap, selected_query, nearest_query
            )
        elif key[0] in {
            "top_query_nearest_candidate",
            "top_query_nearest_candidate_external_box",
        }:
            nearest_candidate = pair_iou.argmax(dim=1)
            candidate_ids = torch.arange(
                candidate_count, device=pair_iou.device
            ).view(1, -1, 1)
            assigned = nearest_candidate[:, None, :] == candidate_ids
            assigned_score = external_rank_score[:, None, :].masked_fill(
                ~assigned, -torch.inf
            )
            has_assigned = assigned.any(dim=-1)
            selected_query = assigned_score.argmax(dim=-1)
            nearest_query = pair_iou.argmax(dim=-1)
            selected_query = torch.where(
                has_assigned, selected_query, nearest_query
            )
        elif key[0] == "nearest_iou":
            selected_query = pair_iou.argmax(dim=-1)
        else:
            raise ValueError(
                f"unsupported external rank-transfer evidence mode {key[0]!r}"
            )
        matched_queries_by_transfer[transfer_key] = selected_query
        if key[0] in {
            "max_score_iou_power_external_box",
            "top_query_nearest_candidate_external_box",
        }:
            mapped_boxes = torch.gather(
                external_boxes,
                1,
                selected_query.unsqueeze(-1).expand(-1, -1, 4),
            )
            if not bool(torch.isfinite(mapped_boxes).all().item()):
                raise ValueError("mapped external prediction boxes must be finite")
            prediction_boxes_by_transfer[transfer_key] = mapped_boxes
        else:
            prediction_boxes_by_transfer[transfer_key] = candidate_boxes
    result = (
        fused_by_grid,
        candidate_idx,
        prediction_boxes_by_transfer,
        candidate_boxes,
    )
    if include_matched_queries:
        return (*result, matched_queries_by_transfer)
    return result


def _diagnostic_patch_internal_rank_identity_winner(
    patch_outputs: Dict[str, torch.Tensor],
    *,
    patch_cfg,
    candidate_idx: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the exact standard-Ref beta=0 winner without recomputing scores."""

    if patch_cfg is None:
        raise ValueError(
            "patch internal-rank identity requires the active patch config"
        )
    standard_slot_scores = _slot_scores(patch_outputs, patch_cfg, 0.0)
    if (
        not torch.is_tensor(standard_slot_scores)
        or standard_slot_scores.dim() != 3
        or not standard_slot_scores.is_floating_point()
    ):
        shape = (
            tuple(standard_slot_scores.shape)
            if torch.is_tensor(standard_slot_scores)
            else type(standard_slot_scores).__name__
        )
        raise ValueError(
            "standard Ref beta=0 slot scores must be floating (B,Q,K), "
            f"got {shape}"
        )
    batch_size, query_count, slot_count = standard_slot_scores.shape
    if (
        slot_count <= 0
        or not torch.is_tensor(candidate_idx)
        or candidate_idx.dim() != 2
        or int(candidate_idx.shape[0]) != batch_size
    ):
        raise ValueError(
            "patch internal-rank identity scores and fixed candidates are misaligned"
        )
    if candidate_idx.device != standard_slot_scores.device:
        raise ValueError(
            "patch internal-rank identity scores and candidates must share one device"
        )
    if bool(((candidate_idx < 0) | (candidate_idx >= query_count)).any().item()):
        raise ValueError(
            "patch internal-rank identity fixed candidate index is out of range"
        )
    if not bool(torch.isfinite(standard_slot_scores).all().item()):
        raise ValueError("standard Ref beta=0 slot scores must all be finite")

    flat_scores = standard_slot_scores.reshape(batch_size, query_count * slot_count)
    winner_flat_idx = torch.argmax(flat_scores, dim=1)
    winner_query_idx = torch.div(
        winner_flat_idx, slot_count, rounding_mode="floor"
    )
    admitted = candidate_idx == winner_query_idx.unsqueeze(1)
    if not bool(admitted.any(dim=1).all().item()):
        raise ValueError(
            "standard Ref beta=0 winner escaped the unchanged fixed Top-K admission"
        )

    patch_boxes_xyxy = _normalized_cxcywh_to_xyxy(
        _require_diagnostic_tensor(patch_outputs, "pred_boxes"),
        name="patch pred_boxes",
    )
    if tuple(patch_boxes_xyxy.shape[:2]) != (batch_size, query_count):
        raise ValueError(
            "standard Ref beta=0 scores and patch prediction boxes are misaligned"
        )
    winner_boxes = torch.gather(
        patch_boxes_xyxy,
        1,
        winner_query_idx.view(batch_size, 1, 1).expand(-1, -1, 4),
    ).squeeze(1)
    return (
        standard_slot_scores,
        winner_flat_idx,
        winner_query_idx,
        winner_boxes,
    )


def _diagnostic_external_gdino_base_identity_winner(
    external_outputs: Dict[str, torch.Tensor],
    *,
    expected_query_count: int = _EXTERNAL_GDINO_QUERY_COUNT,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select the direct ordinary-GDINO top query from its authoritative base."""

    if _EXTERNAL_GDINO_BASE_SCORE_KEY not in external_outputs:
        raise KeyError(
            "external GDINO base identity requires output "
            f"{_EXTERNAL_GDINO_BASE_SCORE_KEY!r}"
        )
    base_score = external_outputs[_EXTERNAL_GDINO_BASE_SCORE_KEY]
    if not torch.is_tensor(base_score):
        raise TypeError(
            f"{_EXTERNAL_GDINO_BASE_SCORE_KEY} must be a tensor"
        )
    if base_score.dim() != 2 or not base_score.is_floating_point():
        raise ValueError(
            f"{_EXTERNAL_GDINO_BASE_SCORE_KEY} must be floating (B,Q), "
            f"got shape={tuple(base_score.shape)} dtype={base_score.dtype}"
        )
    base_score = base_score.detach().float()
    batch_size, query_count = base_score.shape
    expected_query_count = int(expected_query_count)
    if query_count != expected_query_count:
        raise ValueError(
            "external GDINO base identity requires exactly "
            f"{expected_query_count} queries, got {query_count}"
        )
    if batch_size <= 0 or not bool(torch.isfinite(base_score).all().item()):
        raise ValueError(
            f"{_EXTERNAL_GDINO_BASE_SCORE_KEY} must be non-empty and finite"
        )

    external_boxes = _normalized_cxcywh_to_xyxy(
        external_outputs.get("pred_boxes"),
        name="external GDINO pred_boxes",
    )
    if tuple(external_boxes.shape) != (
        batch_size,
        expected_query_count,
        4,
    ):
        raise ValueError(
            "external GDINO base identity pred_boxes must be exactly "
            f"{(batch_size, expected_query_count, 4)}, got "
            f"{tuple(external_boxes.shape)}"
        )
    if external_boxes.device != base_score.device:
        raise ValueError(
            "external GDINO base identity scores and boxes must share one device"
        )

    # torch.argmax deterministically returns the first index for equal maxima.
    winner_query_idx = torch.argmax(base_score, dim=1)
    winner_boxes = torch.gather(
        external_boxes,
        1,
        winner_query_idx.view(batch_size, 1, 1).expand(-1, -1, 4),
    ).squeeze(1)
    return base_score, winner_query_idx, winner_boxes


def _diagnostic_top1_iou_breakdown(
    labels: Iterable[str],
    values: List[Optional[float]],
    *,
    fixed_labels: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    normalized_labels = list(labels)
    if len(normalized_labels) != len(values):
        raise ValueError("diagnostic caption rows lost example alignment")
    keys = (
        list(fixed_labels)
        if fixed_labels is not None
        else sorted(set(normalized_labels))
    )
    out: Dict[str, Dict[str, Any]] = {}
    for label in keys:
        indices = [
            index
            for index, observed in enumerate(normalized_labels)
            if observed == label
        ]
        count = len(indices)
        selected = [values[index] for index in indices]
        correct = sum(value is not None and value >= 0.5 for value in selected)
        iou_sum = sum(
            0.0 if value is None else float(value) for value in selected
        )
        out[label] = {
            "num_expressions": count,
            "acc50": float(correct / count) if count > 0 else None,
            "mean_iou_top1": float(iou_sum / count) if count > 0 else None,
        }
    return out


class DiagnosticExternalGDINORankTransferAccumulator:
    def __init__(self, settings: Dict[str, Any], *, patch_cfg=None) -> None:
        self.settings = settings
        self.grid = list(settings["fixed_grid"])
        if not self.grid:
            raise ValueError("external rank-transfer diagnostic grid is empty")
        self.patch_internal_rank_identity_enabled = bool(
            settings.get("include_patch_internal_rank_identity", False)
        )
        self.external_gdino_base_identity_enabled = bool(
            settings.get("include_external_gdino_base_identity", False)
        )
        self.patch_cfg = patch_cfg
        if self.patch_internal_rank_identity_enabled and patch_cfg is None:
            raise ValueError(
                "patch internal-rank identity requires the active patch config"
            )
        self.total = 0
        self.candidate_topk: Optional[int] = None
        self.iou_sum = {_external_grid_key(row): 0.0 for row in self.grid}
        self.correct50 = {_external_grid_key(row): 0 for row in self.grid}
        self.correct25 = {_external_grid_key(row): 0 for row in self.grid}
        self.per_example_top1_iou: Dict[
            Tuple[str, Optional[float], float, float], List[Optional[float]]
        ] = {_external_grid_key(row): [] for row in self.grid}
        self.canonical_stage_a_groups: List[str] = []
        self.canonical_stage_a_captions: List[str] = []
        self.canonical_stage_a_groups_complete = True
        if len(self.iou_sum) != len(self.grid):
            raise ValueError("external rank-transfer diagnostic grid contains duplicates")
        self.candidate_oracle_correct50 = 0
        self.identity_iou_sum = 0.0
        self.identity_correct50 = 0
        self.identity_correct25 = 0
        self.identity_per_example_top1_iou: List[Optional[float]] = []
        self.external_base_identity_iou_sum = 0.0
        self.external_base_identity_correct50 = 0
        self.external_base_identity_correct25 = 0
        self.external_base_identity_per_example_top1_iou: List[
            Optional[float]
        ] = []

    def update(
        self,
        patch_outputs: Dict[str, torch.Tensor],
        external_outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> None:
        (
            fused_by_grid,
            candidate_idx,
            prediction_boxes_by_transfer,
            candidate_boxes,
        ) = (
            _external_rank_transfer_fused_candidates(
                patch_outputs,
                external_outputs,
                settings=self.settings,
            )
        )
        batch_size, candidate_count = candidate_idx.shape
        if len(targets) != batch_size:
            raise ValueError(
                "external rank-transfer targets must align with output batch size"
            )
        expected_topk = int(self.settings["candidate_topk"])
        if candidate_count != expected_topk:
            raise ValueError(
                "external rank transfer changed fixed candidate admission: "
                f"expected Top{expected_topk}, got Top{candidate_count}"
            )
        if self.candidate_topk is None:
            self.candidate_topk = candidate_count
        elif self.candidate_topk != candidate_count:
            raise ValueError(
                "external rank-transfer candidate count changed across batches"
            )

        canonical_captions = patch_outputs.get("stage_a_captions")
        if canonical_captions is None:
            batch_canonical_groups = ["unavailable"] * batch_size
            batch_normalized_captions = ["unavailable"] * batch_size
            self.canonical_stage_a_groups_complete = False
        else:
            if (
                not isinstance(canonical_captions, (list, tuple))
                or len(canonical_captions) != batch_size
                or any(
                    not isinstance(value, str) or not value.strip()
                    for value in canonical_captions
                )
            ):
                raise ValueError(
                    "diagnostic canonical grouping requires one non-empty "
                    "patch_outputs.stage_a_captions string per row"
                )
            batch_normalized_captions = [
                _norm_text(value) for value in canonical_captions
            ]
            if any(not value for value in batch_normalized_captions):
                raise ValueError(
                    "diagnostic canonical grouping requires every "
                    "patch_outputs.stage_a_captions value to remain non-empty "
                    "after _norm_text normalization"
                )
            batch_canonical_groups = [
                "person" if value == "person" else "other"
                for value in batch_normalized_captions
            ]

        identity_winner_boxes: Optional[torch.Tensor] = None
        if self.patch_internal_rank_identity_enabled:
            (
                _identity_scores,
                _identity_winner_flat_idx,
                _identity_winner_query_idx,
                identity_winner_boxes,
            ) = _diagnostic_patch_internal_rank_identity_winner(
                patch_outputs,
                patch_cfg=self.patch_cfg,
                candidate_idx=candidate_idx,
            )
        external_base_identity_winner_boxes: Optional[torch.Tensor] = None
        if self.external_gdino_base_identity_enabled:
            (
                _external_base_score,
                _external_base_winner_query_idx,
                external_base_identity_winner_boxes,
            ) = _diagnostic_external_gdino_base_identity_winner(
                external_outputs,
                expected_query_count=int(self.settings["external_query_count"]),
            )

        for batch_idx, target in enumerate(targets):
            self.total += 1
            self.canonical_stage_a_groups.append(
                batch_canonical_groups[batch_idx]
            )
            self.canonical_stage_a_captions.append(
                batch_normalized_captions[batch_idx]
            )
            gt_boxes = target["boxes"].detach().float()
            if gt_boxes.numel() == 0:
                for descriptor in self.grid:
                    self.per_example_top1_iou[_external_grid_key(descriptor)].append(
                        None
                    )
                if self.patch_internal_rank_identity_enabled:
                    self.identity_per_example_top1_iou.append(None)
                if self.external_gdino_base_identity_enabled:
                    self.external_base_identity_per_example_top1_iou.append(None)
                continue
            gt = box_ops.box_cxcywh_to_xyxy(gt_boxes[:1]).clamp(0.0, 1.0)
            admission_candidate_ious = box_ops.box_iou(
                candidate_boxes[batch_idx], gt
            )[0].view(-1)
            if not bool(torch.isfinite(admission_candidate_ious).all().item()):
                raise ValueError(
                    "external rank-transfer candidate/target IoUs must be finite"
                )
            if bool((admission_candidate_ious >= 0.5).any().item()):
                self.candidate_oracle_correct50 += 1
            prediction_ious_by_transfer = {
                transfer_key: box_ops.box_iou(
                    boxes[batch_idx], gt
                )[0].view(-1)
                for transfer_key, boxes in prediction_boxes_by_transfer.items()
            }
            if any(
                not bool(values.isfinite().all().item())
                for values in prediction_ious_by_transfer.values()
            ):
                raise ValueError(
                    "external rank-transfer prediction/target IoUs must be finite"
                )
            for descriptor in self.grid:
                key = _external_grid_key(descriptor)
                winner = int(torch.argmax(fused_by_grid[key][batch_idx]).item())
                top1_iou = float(
                    prediction_ious_by_transfer[(key[0], key[1])][winner].item()
                )
                self.per_example_top1_iou[key].append(top1_iou)
                self.iou_sum[key] += top1_iou
                if top1_iou >= 0.5:
                    self.correct50[key] += 1
                if top1_iou >= 0.25:
                    self.correct25[key] += 1
            if identity_winner_boxes is not None:
                identity_top1_iou = float(
                    box_ops.box_iou(
                        identity_winner_boxes[batch_idx].view(1, 4), gt
                    )[0]
                    .view(-1)[0]
                    .item()
                )
                if not math.isfinite(identity_top1_iou):
                    raise ValueError(
                        "patch internal-rank identity target IoU must be finite"
                    )
                self.identity_per_example_top1_iou.append(identity_top1_iou)
                self.identity_iou_sum += identity_top1_iou
                if identity_top1_iou >= 0.5:
                    self.identity_correct50 += 1
                if identity_top1_iou >= 0.25:
                    self.identity_correct25 += 1
            if external_base_identity_winner_boxes is not None:
                external_base_top1_iou = float(
                    box_ops.box_iou(
                        external_base_identity_winner_boxes[batch_idx].view(1, 4),
                        gt,
                    )[0]
                    .view(-1)[0]
                    .item()
                )
                if not math.isfinite(external_base_top1_iou):
                    raise ValueError(
                        "external GDINO base identity target IoU must be finite"
                    )
                self.external_base_identity_per_example_top1_iou.append(
                    external_base_top1_iou
                )
                self.external_base_identity_iou_sum += external_base_top1_iou
                if external_base_top1_iou >= 0.5:
                    self.external_base_identity_correct50 += 1
                if external_base_top1_iou >= 0.25:
                    self.external_base_identity_correct25 += 1

    def _attach_canonical_caption_breakdowns(
        self,
        row: Dict[str, Any],
        values: List[Optional[float]],
    ) -> None:
        if not self.canonical_stage_a_groups_complete:
            return
        if (
            len(self.canonical_stage_a_groups) != self.total
            or len(self.canonical_stage_a_captions) != self.total
            or len(values) != self.total
        ):
            raise ValueError(
                "diagnostic canonical caption rows lost example alignment"
            )
        row["canonical_stage_a_group_contract"] = {
            "source": "patch_outputs.stage_a_captions",
            "normalization": "_norm_text",
            "person_rule": "normalized_caption == 'person'",
            "uses_target_category_or_box_for_routing": False,
        }
        row["by_canonical_stage_a_group"] = _diagnostic_top1_iou_breakdown(
            self.canonical_stage_a_groups,
            values,
            fixed_labels=("person", "other"),
        )
        row["canonical_stage_a_caption_contract"] = {
            "source": "patch_outputs.stage_a_captions",
            "normalization": "_norm_text",
            "key": "exact_normalized_caption",
            "uses_target_category_or_box_for_grouping": False,
        }
        row["by_canonical_stage_a_caption"] = _diagnostic_top1_iou_breakdown(
            self.canonical_stage_a_captions,
            values,
        )

    def results(self) -> List[Dict[str, Any]]:
        denom = max(1, int(self.total))
        candidate_topk = int(self.candidate_topk or 0)
        oracle = float(self.candidate_oracle_correct50 / denom)
        rows: List[Dict[str, Any]] = []
        for descriptor in self.grid:
            key = _external_grid_key(descriptor)
            row = {
                "diagnostic_only": True,
                "formal_gate_eligible": False,
                "diagnostic_descriptor_kind": "external_rank_transfer",
                "diagnostic_transfer_mode": key[0],
                "diagnostic_iou_power": key[1],
                "diagnostic_patch_weight": key[2],
                "diagnostic_text_weight": key[3],
                "num_expressions": int(self.total),
                "acc50": float(self.correct50[key] / denom),
                "acc25": float(self.correct25[key] / denom),
                "mean_iou_top1": float(self.iou_sum[key] / denom),
                "recall50@1": float(self.correct50[key] / denom),
                "candidate_oracle_recall50": oracle,
                "candidate_topk": candidate_topk,
            }
            self._attach_canonical_caption_breakdowns(
                row, self.per_example_top1_iou[key]
            )
            rows.append(row)
        if self.patch_internal_rank_identity_enabled:
            identity_row = {
                "diagnostic_only": True,
                "formal_gate_eligible": False,
                "diagnostic_descriptor_kind": _PATCH_INTERNAL_RANK_IDENTITY_KIND,
                "diagnostic_standard_ref_beta": 0.0,
                "diagnostic_score_source": "authoritative_patch_model_slot_scores",
                "diagnostic_winner_rule": "bitwise_standard_ref_beta0_flat_argmax",
                "diagnostic_prediction_box_source": (
                    "patch_model_pred_boxes_at_winning_query"
                ),
                "uses_external_rank_score": False,
                "uses_external_box": False,
                "uses_fusion_weights": False,
                "patch_internal_rank_identity_contract_version": (
                    _PATCH_INTERNAL_RANK_IDENTITY_CONTRACT_VERSION
                ),
                "num_expressions": int(self.total),
                "acc50": float(self.identity_correct50 / denom),
                "acc25": float(self.identity_correct25 / denom),
                "mean_iou_top1": float(self.identity_iou_sum / denom),
                "recall50@1": float(self.identity_correct50 / denom),
                "candidate_oracle_recall50": oracle,
                "candidate_topk": candidate_topk,
            }
            self._attach_canonical_caption_breakdowns(
                identity_row, self.identity_per_example_top1_iou
            )
            rows.append(identity_row)
        if self.external_gdino_base_identity_enabled:
            external_base_identity_row = {
                "diagnostic_only": True,
                "formal_gate_eligible": False,
                "diagnostic_descriptor_kind": _EXTERNAL_GDINO_BASE_IDENTITY_KIND,
                "diagnostic_identity_kind": _EXTERNAL_GDINO_BASE_IDENTITY_ID,
                "diagnostic_score_key": _EXTERNAL_GDINO_BASE_SCORE_KEY,
                "diagnostic_query_count": int(
                    self.settings["external_query_count"]
                ),
                "diagnostic_output_box_source": (
                    "external_outputs.pred_boxes_at_direct_global_argmax"
                ),
                "diagnostic_standard_ref_beta": 0.0,
                "diagnostic_score_source": _EXTERNAL_GDINO_BASE_SCORE_KEY,
                "diagnostic_winner_rule": (
                    "first_argmax_over_full_external_query_axis"
                ),
                "diagnostic_query_domain": "all_900_external_gdino_queries",
                "diagnostic_prediction_box_source": (
                    "external_gdino_pred_boxes_at_winning_query"
                ),
                "uses_external_base_score": True,
                "uses_external_rank_score": False,
                "uses_external_box": True,
                "uses_adapter_rank_residual": False,
                "uses_patch_top50_admission": False,
                "uses_top_query_mapping": False,
                "uses_fusion_weights": False,
                "external_gdino_base_identity_contract_version": (
                    _EXTERNAL_GDINO_BASE_IDENTITY_CONTRACT_VERSION
                ),
                "external_query_count": int(self.settings["external_query_count"]),
                "candidate_topk": candidate_topk,
                "candidate_topk_scope": (
                    "run_context_only_not_used_by_this_descriptor"
                ),
                "candidate_oracle_scope": (
                    "run_context_only_not_used_by_this_descriptor"
                ),
                "num_expressions": int(self.total),
                "acc50": float(self.external_base_identity_correct50 / denom),
                "acc25": float(self.external_base_identity_correct25 / denom),
                "mean_iou_top1": float(
                    self.external_base_identity_iou_sum / denom
                ),
                "recall50@1": float(
                    self.external_base_identity_correct50 / denom
                ),
                "candidate_oracle_recall50": oracle,
            }
            self._attach_canonical_caption_breakdowns(
                external_base_identity_row,
                self.external_base_identity_per_example_top1_iou,
            )
            rows.append(external_base_identity_row)
        return rows


class FormalExternalGDINORankTransferAccumulator:
    def __init__(
        self,
        settings: Dict[str, Any],
        *,
        manifest: EvalManifest,
        run_prefix: str,
    ) -> None:
        self.settings = settings
        self.grid = list(settings["fixed_grid"])
        if len(self.grid) != 1:
            raise ValueError("formal external rank transfer requires exactly one grid point")
        self.key = _external_grid_key(self.grid[0])
        identity = str(settings["artifact_identity"]["sha256"])
        self.run_id = f"{run_prefix}:formal_external_gdino_rank_transfer={identity[:16]}"
        self.manifest = manifest
        self.total = 0
        self.correct50 = 0
        self.correct25 = 0
        self.iou_sum = 0.0
        self.candidate_oracle_correct50 = 0
        self.candidate_topk: Optional[int] = None
        self.eval_records: List[Dict[str, Any]] = []

    def update(
        self,
        patch_outputs: Dict[str, torch.Tensor],
        external_outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> None:
        start_index = int(self.total)
        canonical_captions = patch_outputs.get("stage_a_captions")
        if (
            not isinstance(canonical_captions, (list, tuple))
            or len(canonical_captions) != len(targets)
            or any(
                not isinstance(value, str) or not value.strip()
                for value in canonical_captions
            )
        ):
            raise ValueError(
                "formal external transfer requires patch_outputs.stage_a_captions"
            )
        (
            fused_by_grid,
            candidate_idx,
            prediction_boxes_by_transfer,
            candidate_boxes,
        ) = (
            _external_rank_transfer_fused_candidates(
                patch_outputs,
                external_outputs,
                settings=self.settings,
            )
        )
        batch_size, candidate_count = candidate_idx.shape
        if len(targets) != batch_size:
            raise ValueError("formal external rank-transfer targets do not align")
        expected_topk = int(self.settings["candidate_topk"])
        if candidate_count != expected_topk:
            raise ValueError(
                "formal external rank transfer changed fixed candidate admission: "
                f"expected Top{expected_topk}, got Top{candidate_count}"
            )
        if self.candidate_topk is None:
            self.candidate_topk = candidate_count
        elif self.candidate_topk != candidate_count:
            raise ValueError(
                "formal external rank-transfer candidate count changed across batches"
            )

        fused = fused_by_grid[self.key]
        artifact_sha256 = str(self.settings["artifact_identity"]["sha256"])
        for batch_idx, target in enumerate(targets):
            self.total += 1
            gt_boxes = target["boxes"].detach().float()
            valid = gt_boxes.numel() > 0
            top1_iou: Optional[float] = None
            oracle_correct50 = False
            if valid:
                gt = box_ops.box_cxcywh_to_xyxy(gt_boxes[:1]).clamp(0.0, 1.0)
                admission_candidate_ious = box_ops.box_iou(
                    candidate_boxes[batch_idx], gt
                )[0].view(-1)
                if not bool(torch.isfinite(admission_candidate_ious).all().item()):
                    raise ValueError(
                        "formal external rank-transfer candidate/target IoUs must be finite"
                    )
                oracle_correct50 = bool(
                    (admission_candidate_ious >= 0.5).any().item()
                )
                winner = int(torch.argmax(fused[batch_idx]).item())
                prediction_ious = box_ops.box_iou(
                    prediction_boxes_by_transfer[
                        (self.key[0], self.key[1])
                    ][batch_idx],
                    gt,
                )[0].view(-1)
                if not bool(torch.isfinite(prediction_ious).all().item()):
                    raise ValueError(
                        "formal external rank-transfer prediction/target IoUs must be finite"
                    )
                top1_iou = float(prediction_ious[winner].item())
                if not math.isfinite(top1_iou):
                    valid = False
                    top1_iou = None

            if oracle_correct50:
                self.candidate_oracle_correct50 += 1
            if top1_iou is not None:
                self.iou_sum += top1_iou
                if top1_iou >= 0.5:
                    self.correct50 += 1
                if top1_iou >= 0.25:
                    self.correct25 += 1
            self.eval_records.append(
                make_eval_record(
                    self.manifest,
                    index=start_index + batch_idx,
                    run_id=self.run_id,
                    valid=valid,
                    values={
                        "top1_iou": top1_iou,
                        "correct50": (
                            bool(top1_iou >= 0.5) if top1_iou is not None else None
                        ),
                        "correct25": (
                            bool(top1_iou >= 0.25) if top1_iou is not None else None
                        ),
                        "candidate_oracle_correct50": (
                            oracle_correct50 if valid else None
                        ),
                        "candidate_topk": expected_topk,
                        "external_transfer_artifact_sha256": artifact_sha256,
                        "external_transfer_mode": self.key[0],
                        "external_transfer_iou_power": self.key[1],
                        "external_transfer_patch_weight": self.key[2],
                        "external_transfer_text_weight": self.key[3],
                        "canonical_caption": str(canonical_captions[batch_idx]),
                        "canonical_class_norm": _norm_text(
                            canonical_captions[batch_idx]
                        ),
                    },
                )
            )

    def results(self) -> List[Dict[str, Any]]:
        denom = max(1, int(self.total))
        return [
            {
                "diagnostic_only": False,
                "formal_gate_eligible": True,
                "formal_transfer_mode": self.key[0],
                "formal_iou_power": self.key[1],
                "formal_patch_weight": self.key[2],
                "formal_text_weight": self.key[3],
                "external_transfer_artifact_sha256": str(
                    self.settings["artifact_identity"]["sha256"]
                ),
                "num_expressions": int(self.total),
                "acc50": float(self.correct50 / denom),
                "acc25": float(self.correct25 / denom),
                "mean_iou_top1": float(self.iou_sum / denom),
                "recall50@1": float(self.correct50 / denom),
                "candidate_oracle_recall50": float(
                    self.candidate_oracle_correct50 / denom
                ),
                "candidate_topk": int(self.candidate_topk or 0),
            }
        ]


class FormalRoutedExternalGDINORankTransferAccumulator:
    def __init__(
        self,
        settings: Dict[str, Any],
        *,
        manifest: EvalManifest,
        run_prefix: str,
    ) -> None:
        self.formal_route_version = int(settings.get("formal_artifact_version", 0))
        if self.formal_route_version not in (2, 3):
            raise ValueError(
                "formal routed accumulator requires artifact version 2 or 3"
            )
        self.settings = settings
        self.manifest = manifest
        identity = str(settings["artifact_identity"]["sha256"])
        route_label = (
            "formal_external_gdino_caption_route"
            if self.formal_route_version == 2
            else "formal_external_gdino_fulltext_gated_caption_route"
        )
        self.run_id = f"{run_prefix}:{route_label}={identity[:16]}"
        self.default_descriptor_id = str(settings["route_default_descriptor_id"])
        if self.default_descriptor_id != ROUTE_DEFAULT_DESCRIPTOR_ID:
            raise ValueError("formal caption route default descriptor drifted")
        if settings.get("descriptor_registry") != ROUTE_DESCRIPTOR_REGISTRY:
            raise ValueError("formal caption route descriptor registry drifted")
        if self.formal_route_version == 2:
            self.unconditional_overrides = dict(settings["route_overrides"])
            self.conditional_overrides: Dict[str, Dict[str, Any]] = {}
        else:
            self.unconditional_overrides = dict(
                settings["route_unconditional_overrides"]
            )
            if dict(settings.get("route_overrides", {})) != (
                self.unconditional_overrides
            ):
                raise ValueError(
                    "formal full-text route compatibility overrides drifted"
                )
            self.conditional_overrides = {
                str(key): dict(value)
                for key, value in settings["route_conditional_overrides"].items()
            }
            self._validate_fulltext_conditional_route()
        # v2 callers and existing provenance checks use this attribute.
        self.overrides = self.unconditional_overrides
        allowed = set(ROUTE_CANDIDATE_DESCRIPTOR_IDS)
        if any(
            not isinstance(caption, str)
            or not caption
            or _norm_text(caption) != caption
            or descriptor_id not in allowed
            for caption, descriptor_id in self.unconditional_overrides.items()
        ):
            raise ValueError("formal caption route override mapping is invalid")
        self.descriptor_grid_by_id = {
            str(key): dict(value)
            for key, value in settings["descriptor_grid_by_id"].items()
        }
        if set(self.descriptor_grid_by_id) != set(ROUTE_CANDIDATE_DESCRIPTOR_IDS):
            raise ValueError(
                "formal caption route must carry the exact four-descriptor registry"
            )
        self.grid_key_by_descriptor = {
            descriptor_id: _external_grid_key(descriptor)
            for descriptor_id, descriptor in self.descriptor_grid_by_id.items()
        }
        if list(settings["fixed_grid"]) != [
            self.descriptor_grid_by_id[key]
            for key in self.descriptor_grid_by_id
        ]:
            raise ValueError("formal caption route fixed grid drifted")
        self.total = 0
        self.correct50 = 0
        self.correct25 = 0
        self.iou_sum = 0.0
        self.candidate_oracle_correct50 = 0
        self.all_query_oracle_correct50 = 0
        self.candidate_topk: Optional[int] = None
        self.eval_records: List[Dict[str, Any]] = []
        self.route_counts_by_descriptor: Dict[str, int] = {
            descriptor_id: 0 for descriptor_id in ROUTE_DESCRIPTOR_REGISTRY
        }
        self.route_counts_by_caption: Dict[str, Dict[str, Any]] = {}
        self.fulltext_route_gate_counts: Optional[Dict[str, int]] = (
            {
                "conditional_predicate_matched": 0,
                "conditional_fallback_matched": 0,
                "unconditional_override": 0,
                "default": 0,
            }
            if self.formal_route_version == 3
            else None
        )

    def _validate_fulltext_conditional_route(self) -> None:
        if set(self.conditional_overrides) != {FULLTEXT_GATED_CAPTION}:
            raise ValueError(
                "formal full-text route requires exactly the frozen person gate"
            )
        if FULLTEXT_GATED_CAPTION in self.unconditional_overrides:
            raise ValueError("formal full-text person route cannot be unconditional")
        rule = self.conditional_overrides[FULLTEXT_GATED_CAPTION]
        if set(rule) != {"descriptor_id", "fallback_descriptor_id", "predicate"}:
            raise ValueError("formal full-text person route fields drifted")
        predicate = rule.get("predicate")
        if not isinstance(predicate, dict) or set(predicate) != {
            "kind",
            "max_tokens",
            "token_count_contract",
        }:
            raise ValueError("formal full-text person predicate fields drifted")
        max_tokens = predicate.get("max_tokens")
        if (
            rule.get("descriptor_id") != FULLTEXT_GATED_DESCRIPTOR_ID
            or rule.get("fallback_descriptor_id") != self.default_descriptor_id
            or predicate.get("kind")
            != "full_expression_lexical_token_count_lte"
            or isinstance(max_tokens, bool)
            or not isinstance(max_tokens, int)
            or max_tokens <= 0
            or predicate.get("token_count_contract")
            != FULLTEXT_TOKEN_COUNT_CONTRACT
        ):
            raise ValueError("formal full-text person predicate contract drifted")
        self.fulltext_max_tokens = int(max_tokens)

    def _route_descriptors(
        self,
        normalized_captions: List[str],
        patch_outputs: Dict[str, torch.Tensor],
    ) -> Tuple[List[str], List[Dict[str, Any]]]:
        if self.formal_route_version == 2:
            return (
                [
                    self.unconditional_overrides.get(
                        caption, self.default_descriptor_id
                    )
                    for caption in normalized_captions
                ],
                [{} for _ in normalized_captions],
            )

        full_expressions = patch_outputs.get(_FULLTEXT_CAPTIONS_OUTPUT_KEY)
        if (
            not isinstance(full_expressions, (list, tuple))
            or len(full_expressions) != len(normalized_captions)
            or any(
                not isinstance(value, str) or not value.strip()
                for value in full_expressions
            )
        ):
            raise ValueError(
                "formal full-text route requires one manifest-validated full "
                f"expression in patch_outputs.{_FULLTEXT_CAPTIONS_OUTPUT_KEY} per row "
                "before target access"
            )
        descriptors: List[str] = []
        evidence: List[Dict[str, Any]] = []
        rule = self.conditional_overrides[FULLTEXT_GATED_CAPTION]
        for caption, full_expression in zip(normalized_captions, full_expressions):
            normalized_expression = _WS_RE.sub(" ", full_expression.strip())
            token_count = len(
                _FULLTEXT_LEXICAL_TOKEN_RE.findall(normalized_expression.lower())
            )
            if token_count <= 0:
                raise ValueError(
                    "formal full-text route expression has no [a-z0-9]+ lexical tokens"
                )
            predicate_matched = False
            fallback_matched = False
            if caption in self.unconditional_overrides:
                descriptor_id = self.unconditional_overrides[caption]
                route_source = "unconditional_override"
            elif caption == FULLTEXT_GATED_CAPTION:
                predicate_matched = token_count <= self.fulltext_max_tokens
                fallback_matched = not predicate_matched
                descriptor_id = str(
                    rule[
                        "descriptor_id"
                        if predicate_matched
                        else "fallback_descriptor_id"
                    ]
                )
                route_source = (
                    "conditional_predicate_matched"
                    if predicate_matched
                    else "conditional_fallback_matched"
                )
            else:
                descriptor_id = self.default_descriptor_id
                route_source = "default"
            descriptors.append(descriptor_id)
            evidence.append(
                {
                    "fulltext_route_gate_contract_version": (
                        _FULLTEXT_ROUTE_GATE_CONTRACT_VERSION
                    ),
                    "fulltext_route_gate_full_expression": normalized_expression,
                    "fulltext_route_gate_token_count": token_count,
                    "fulltext_route_gate_max_tokens": self.fulltext_max_tokens,
                    "fulltext_route_gate_predicate_matched": predicate_matched,
                    "fulltext_route_gate_fallback_matched": fallback_matched,
                    "fulltext_route_gate_source": (
                        "external_base_direct"
                        if fallback_matched
                        else "formal_routed_v2"
                    ),
                    "fulltext_route_gate_route_kind": route_source,
                    "fulltext_route_gate_selected_descriptor_id": descriptor_id,
                    "fulltext_route_gate_predicate_kind": rule["predicate"]["kind"],
                    "fulltext_route_gate_token_count_contract_sha256": (
                        _canonical_json_sha256(FULLTEXT_TOKEN_COUNT_CONTRACT)
                    ),
                }
            )
        return descriptors, evidence

    def _route_batch(
        self,
        patch_outputs: Dict[str, torch.Tensor],
        external_outputs: Dict[str, torch.Tensor],
    ) -> Tuple[
        List[str],
        List[str],
        List[Dict[str, Any]],
        torch.Tensor,
        torch.Tensor,
    ]:
        patch_pred_boxes = patch_outputs.get("pred_boxes")
        if (
            not torch.is_tensor(patch_pred_boxes)
            or patch_pred_boxes.dim() != 3
            or int(patch_pred_boxes.shape[-1]) != 4
        ):
            raise ValueError(
                "formal caption route requires patch pred_boxes before target access"
            )
        batch_size = int(patch_pred_boxes.shape[0])
        canonical_captions = patch_outputs.get("stage_a_captions")
        if (
            not isinstance(canonical_captions, (list, tuple))
            or len(canonical_captions) != batch_size
            or any(
                not isinstance(value, str) or not value.strip()
                for value in canonical_captions
            )
        ):
            raise ValueError(
                "formal caption route requires one non-empty "
                "patch_outputs.stage_a_captions string per row before target access"
            )
        normalized_captions = [_norm_text(value) for value in canonical_captions]
        if any(not value for value in normalized_captions):
            raise ValueError(
                "formal caption route canonical caption is empty after _norm_text"
            )
        descriptor_ids, fulltext_evidence = self._route_descriptors(
            normalized_captions, patch_outputs
        )
        if any(
            descriptor_id not in ROUTE_DESCRIPTOR_REGISTRY
            for descriptor_id in descriptor_ids
        ):
            raise ValueError("formal caption route selected an unknown descriptor")

        (
            _base_score,
            base_winner_query,
            base_winner_boxes,
        ) = _diagnostic_external_gdino_base_identity_winner(
            external_outputs,
            expected_query_count=int(self.settings["external_query_count"]),
        )
        external_boxes = _normalized_cxcywh_to_xyxy(
            external_outputs.get("pred_boxes"),
            name="external GDINO pred_boxes",
        )
        (
            fused_by_grid,
            candidate_idx,
            prediction_boxes_by_transfer,
            candidate_boxes,
            matched_queries_by_transfer,
        ) = _external_rank_transfer_fused_candidates(
            patch_outputs,
            external_outputs,
            settings=self.settings,
            include_matched_queries=True,
        )
        if int(candidate_idx.shape[0]) != batch_size:
            raise ValueError("formal caption route model batches do not align")
        candidate_count = int(candidate_idx.shape[1])
        expected_topk = int(self.settings["candidate_topk"])
        if candidate_count != expected_topk:
            raise ValueError(
                "formal caption route changed fixed candidate admission: "
                f"expected Top{expected_topk}, got Top{candidate_count}"
            )
        if self.candidate_topk is None:
            self.candidate_topk = candidate_count
        elif self.candidate_topk != candidate_count:
            raise ValueError("formal caption route candidate count changed")

        decisions: List[Dict[str, Any]] = []
        for batch_idx, descriptor_id in enumerate(descriptor_ids):
            descriptor = ROUTE_DESCRIPTOR_REGISTRY[descriptor_id]
            if descriptor_id == self.default_descriptor_id:
                winner_candidate_index = None
                winner_patch_query_index = None
                matched_external_query_index = int(
                    base_winner_query[batch_idx].item()
                )
                selected_box = base_winner_boxes[batch_idx]
            else:
                key = self.grid_key_by_descriptor[descriptor_id]
                winner_candidate_index = int(
                    torch.argmax(fused_by_grid[key][batch_idx]).item()
                )
                winner_patch_query_index = int(
                    candidate_idx[batch_idx, winner_candidate_index].item()
                )
                transfer_key = (key[0], key[1])
                matched_external_query_index = int(
                    matched_queries_by_transfer[transfer_key][
                        batch_idx, winner_candidate_index
                    ].item()
                )
                selected_box = prediction_boxes_by_transfer[transfer_key][
                    batch_idx, winner_candidate_index
                ]
            if not bool(torch.isfinite(selected_box).all().item()):
                raise ValueError("formal caption route selected box must be finite")
            decision = {
                "caption_route_caption": normalized_captions[batch_idx],
                "caption_route_descriptor_id": descriptor_id,
                "caption_route_descriptor_sha256": _canonical_json_sha256(
                    descriptor
                ),
                "caption_route_used_default": (
                    descriptor_id == self.default_descriptor_id
                ),
                "caption_route_output_box_source": descriptor[
                    "output_box_source"
                ],
                "winner_candidate_index": winner_candidate_index,
                "winner_patch_query_index": winner_patch_query_index,
                "matched_external_query_index": matched_external_query_index,
                "selected_box": [
                    float(value) for value in selected_box.detach().cpu().tolist()
                ],
                "selected_box_format": "normalized_xyxy",
                "_selected_box_tensor": selected_box,
            }
            decision.update(fulltext_evidence[batch_idx])
            decisions.append(decision)
        return (
            list(canonical_captions),
            descriptor_ids,
            decisions,
            candidate_boxes,
            external_boxes,
        )

    def update(
        self,
        patch_outputs: Dict[str, torch.Tensor],
        external_outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> None:
        start_index = int(self.total)
        (
            canonical_captions,
            descriptor_ids,
            decisions,
            candidate_boxes,
            external_boxes,
        ) = self._route_batch(patch_outputs, external_outputs)
        batch_size = len(decisions)
        if len(targets) != batch_size:
            raise ValueError("formal caption route targets do not align")

        routing = (
            self.settings["route_selection"]
            if self.formal_route_version == 2
            else self.settings["canonical_route_selection"]
        )
        selection_artifact = routing["artifact"]
        selection_identity = routing["artifact_identity"]
        policy_sha256 = str(self.settings["route_policy_sha256"])
        formal_artifact_sha256 = str(self.settings["artifact_identity"]["sha256"])
        for batch_idx in range(batch_size):
            target = targets[batch_idx]
            decision = decisions[batch_idx]
            descriptor_id = descriptor_ids[batch_idx]
            self.total += 1
            self.route_counts_by_descriptor[descriptor_id] += 1
            caption = decision["caption_route_caption"]
            if self.formal_route_version == 2:
                caption_row = self.route_counts_by_caption.setdefault(
                    caption,
                    {"descriptor_id": descriptor_id, "num_expressions": 0},
                )
                if caption_row["descriptor_id"] != descriptor_id:
                    raise ValueError("formal caption route changed within one caption")
            else:
                caption_row = self.route_counts_by_caption.setdefault(
                    caption,
                    {"descriptor_counts": {}, "num_expressions": 0},
                )
                descriptor_counts = caption_row["descriptor_counts"]
                descriptor_counts[descriptor_id] = (
                    int(descriptor_counts.get(descriptor_id, 0)) + 1
                )
                assert self.fulltext_route_gate_counts is not None
                route_kind = str(decision["fulltext_route_gate_route_kind"])
                self.fulltext_route_gate_counts[route_kind] += 1
            caption_row["num_expressions"] += 1

            gt_boxes = target["boxes"].detach().float()
            valid = gt_boxes.numel() > 0
            top1_iou: Optional[float] = None
            oracle_correct50 = False
            all_query_best_iou: Optional[float] = None
            patch_candidate_oracle_iou: Optional[float] = None
            if valid:
                gt = box_ops.box_cxcywh_to_xyxy(gt_boxes[:1]).clamp(0.0, 1.0)
                admission_candidate_ious = box_ops.box_iou(
                    candidate_boxes[batch_idx], gt
                )[0].view(-1)
                if not bool(torch.isfinite(admission_candidate_ious).all().item()):
                    raise ValueError(
                        "formal caption route candidate/target IoUs must be finite"
                    )
                oracle_correct50 = bool(
                    (admission_candidate_ious >= 0.5).any().item()
                )
                patch_candidate_oracle_iou = float(
                    admission_candidate_ious.max().item()
                )
                all_query_ious = box_ops.box_iou(
                    external_boxes[batch_idx], gt
                )[0].view(-1)
                if not bool(torch.isfinite(all_query_ious).all().item()):
                    raise ValueError(
                        "formal caption route all-query/target IoUs must be finite"
                    )
                all_query_best_iou = float(all_query_ious.max().item())
                selected_box = decision["_selected_box_tensor"].view(1, 4)
                selected_iou = box_ops.box_iou(selected_box, gt)[0].view(-1)
                if not bool(torch.isfinite(selected_iou).all().item()):
                    raise ValueError(
                        "formal caption route prediction/target IoU must be finite"
                    )
                top1_iou = float(selected_iou[0].item())
                if not math.isfinite(top1_iou):
                    valid = False
                    top1_iou = None
            if oracle_correct50:
                self.candidate_oracle_correct50 += 1
            if all_query_best_iou is not None and all_query_best_iou >= 0.5:
                self.all_query_oracle_correct50 += 1
            if top1_iou is not None:
                self.iou_sum += top1_iou
                if top1_iou >= 0.5:
                    self.correct50 += 1
                if top1_iou >= 0.25:
                    self.correct25 += 1
            record_decision = {
                key: value
                for key, value in decision.items()
                if not key.startswith("_")
            }
            record_identity = {
                "external_transfer_artifact_sha256": formal_artifact_sha256,
                "caption_route_selection_artifact_path": selection_artifact[
                    "path"
                ],
                "caption_route_selection_artifact_file_sha256": (
                    selection_artifact["sha256"]
                ),
                "caption_route_selection_artifact_size_bytes": (
                    selection_artifact["size_bytes"]
                ),
                "caption_route_selection_artifact_identity_sha256": (
                    selection_identity["sha256"]
                ),
                "caption_route_policy_sha256": policy_sha256,
                "caption_route_contract_version": self.formal_route_version,
            }
            if self.formal_route_version == 3:
                gate_artifact = self.settings["fulltext_route_gate_artifact"]
                routed_v2 = self.settings["routed_v2_artifact"]
                record_identity.update(
                    {
                        "fulltext_route_gate_artifact_path": gate_artifact["path"],
                        "fulltext_route_gate_artifact_file_sha256": gate_artifact[
                            "sha256"
                        ],
                        "fulltext_route_gate_artifact_size_bytes": gate_artifact[
                            "size_bytes"
                        ],
                        "fulltext_route_gate_artifact_identity_sha256": (
                            self.settings["fulltext_route_gate_artifact_identity"][
                                "sha256"
                            ]
                        ),
                        "routed_v2_artifact_path": routed_v2["artifact"]["path"],
                        "routed_v2_artifact_file_sha256": routed_v2["artifact"][
                            "sha256"
                        ],
                        "routed_v2_artifact_size_bytes": routed_v2["artifact"][
                            "size_bytes"
                        ],
                        "routed_v2_artifact_identity_sha256": routed_v2[
                            "artifact_identity"
                        ]["sha256"],
                        "fulltext_route_policy_sha256": self.settings[
                            "route_policy_sha256"
                        ],
                        "canonical_route_policy_sha256": self.settings[
                            "canonical_route_policy_sha256"
                        ],
                    }
                )
            self.eval_records.append(
                make_eval_record(
                    self.manifest,
                    index=start_index + batch_idx,
                    run_id=self.run_id,
                    valid=valid,
                    values={
                        "top1_iou": top1_iou,
                        "all_query_best_iou": all_query_best_iou,
                        "patch_candidate_oracle_iou": patch_candidate_oracle_iou,
                        "correct50": (
                            bool(top1_iou >= 0.5) if top1_iou is not None else None
                        ),
                        "correct25": (
                            bool(top1_iou >= 0.25) if top1_iou is not None else None
                        ),
                        "candidate_oracle_correct50": (
                            oracle_correct50 if valid else None
                        ),
                        "candidate_topk": int(self.settings["candidate_topk"]),
                        **record_identity,
                        "canonical_caption": str(canonical_captions[batch_idx]),
                        "canonical_class_norm": decision["caption_route_caption"],
                        **record_decision,
                    },
                )
            )

    def results(self) -> List[Dict[str, Any]]:
        denom = max(1, int(self.total))
        route_selection = (
            self.settings["route_selection"]
            if self.formal_route_version == 2
            else self.settings["canonical_route_selection"]
        )
        result = {
            "diagnostic_only": False,
            "formal_gate_eligible": True,
            "formal_route_version": self.formal_route_version,
            "formal_route_default_descriptor_id": self.default_descriptor_id,
            "external_transfer_artifact_sha256": str(
                self.settings["artifact_identity"]["sha256"]
            ),
            "caption_route_selection_artifact_identity_sha256": str(
                route_selection["artifact_identity"]["sha256"]
            ),
            "caption_route_policy_sha256": str(
                self.settings["route_policy_sha256"]
            ),
            "route_counts_by_descriptor": dict(
                sorted(self.route_counts_by_descriptor.items())
            ),
            "route_counts_by_caption": dict(
                sorted(self.route_counts_by_caption.items())
            ),
            "num_expressions": int(self.total),
            "acc50": float(self.correct50 / denom),
            "acc25": float(self.correct25 / denom),
            "mean_iou_top1": float(self.iou_sum / denom),
            "recall50@1": float(self.correct50 / denom),
            "candidate_oracle_recall50": float(
                self.candidate_oracle_correct50 / denom
            ),
            "all_query_oracle_recall50": float(
                self.all_query_oracle_correct50 / denom
            ),
            "candidate_topk": int(self.candidate_topk or 0),
        }
        if self.formal_route_version == 3:
            result.update(
                {
                    "fulltext_route_gate_artifact_identity_sha256": str(
                        self.settings["fulltext_route_gate_artifact_identity"][
                            "sha256"
                        ]
                    ),
                    "routed_v2_artifact_identity_sha256": str(
                        self.settings["routed_v2_artifact_identity"]["sha256"]
                    ),
                    "fulltext_route_policy_sha256": str(
                        self.settings["route_policy_sha256"]
                    ),
                    "canonical_route_policy_sha256": str(
                        self.settings["canonical_route_policy_sha256"]
                    ),
                    "fulltext_route_gate_counts": dict(
                        sorted((self.fulltext_route_gate_counts or {}).items())
                    ),
                }
            )
        return [result]


@torch.no_grad()
def evaluate_dataset(
    *,
    cfg,
    model,
    ckpt_path: str,
    datasetinfo: Dict[str, Any],
    dataset_name: str,
    device: torch.device,
    betas: List[float],
    topks: List[int],
    batch_size: int,
    num_workers: int,
    seed: int,
    amp: bool,
    max_batches: int,
    max_images: int,
    log_every: int,
    records_output_dir: Optional[Path] = None,
    data_driven_query_diagnostics: bool = False,
    diagnostic_patch_rank_weights: Optional[List[float]] = None,
    diagnostic_patch_rank_contract_weight: Optional[float] = None,
    diagnostic_patch_rank_candidate_topk: Optional[int] = None,
    diagnostic_external_rank_transfer_settings: Optional[Dict[str, Any]] = None,
    formal_external_rank_transfer_settings: Optional[Dict[str, Any]] = None,
    external_gdino_model=None,
    external_gdino_cfg=None,
) -> List[Dict[str, Any]]:
    routed_canonical_id_to_caption: Optional[Dict[int, str]] = None
    if formal_external_rank_transfer_settings is not None:
        _validate_formal_routed_runtime(
            formal_external_rank_transfer_settings,
            batch_size=batch_size,
            num_workers=num_workers,
            amp=amp,
        )
        formal_version = int(
            formal_external_rank_transfer_settings.get(
                "formal_artifact_version", 1
            )
        )
        if formal_version in (2, 3):
            route_selection_key = (
                "route_selection"
                if formal_version == 2
                else "canonical_route_selection"
            )
            canonical_binding = formal_external_rank_transfer_settings[
                route_selection_key
            ]["canonical_classes"]
            _require_bound_file(
                canonical_binding,
                label="formal caption route canonical classes",
            )
            _name_to_id, canonical_id_to_name = _load_canonical_name_maps(
                Path(canonical_binding["path"])
            )
            routed_canonical_id_to_caption = {
                int(class_id): _norm_text(name)
                for class_id, name in canonical_id_to_name.items()
                if _norm_text(name)
            }
            if not routed_canonical_id_to_caption:
                raise ValueError(
                    "formal caption route canonical classes contain no usable names"
                )
    loader = _build_loader(cfg, datasetinfo, batch_size, num_workers, device, seed)
    manifest = load_eval_manifest(
        Path(datasetinfo["anno"]),
        task="ref",
        split=dataset_name,
    )
    run_prefix = _ckpt_run_prefix(ckpt_path)
    diagnostic_enabled = diagnostic_patch_rank_weights is not None
    external_diagnostic_enabled = (
        diagnostic_external_rank_transfer_settings is not None
    )
    formal_external_enabled = formal_external_rank_transfer_settings is not None
    if sum(
        int(value)
        for value in (
            diagnostic_enabled,
            external_diagnostic_enabled,
            formal_external_enabled,
        )
    ) > 1:
        raise ValueError(
            "patch-rank diagnostic, external diagnostic, and formal external "
            "transfer modes are mutually exclusive"
        )
    if data_driven_query_diagnostics and (
        diagnostic_enabled
        or external_diagnostic_enabled
        or formal_external_enabled
        or records_output_dir is None
    ):
        raise ValueError(
            "data-driven query diagnostics require the standard record-emitting "
            "evaluation path"
        )
    active_patch_checkpoint: Optional[Dict[str, Any]] = None
    active_external_settings = (
        formal_external_rank_transfer_settings
        if formal_external_enabled
        else diagnostic_external_rank_transfer_settings
    )
    if external_diagnostic_enabled or formal_external_enabled:
        if external_gdino_model is None or external_gdino_cfg is None:
            raise ValueError(
                "external rank-transfer evaluation requires its independent model and config"
            )
        patch_root = model.module if hasattr(model, "module") else model
        external_root = (
            external_gdino_model.module
            if hasattr(external_gdino_model, "module")
            else external_gdino_model
        )
        if model is external_gdino_model or patch_root is external_root:
            raise ValueError(
                "patch and external GDINO diagnostics must use independent model instances"
            )
        if external_diagnostic_enabled and records_output_dir is not None:
            raise ValueError(
                "diagnostic-only external rank transfer cannot emit formal eval records"
            )
        if formal_external_enabled and records_output_dir is None:
            raise ValueError(
                "formal external rank transfer requires canonical per-example records"
            )
        resolved_checkpoint = str(Path(ckpt_path).expanduser().resolve())
        matches = [
            component
            for component in active_external_settings["patch_checkpoints"]
            if component["path"] == resolved_checkpoint
        ]
        if len(matches) != 1:
            raise ValueError(
                "active patch checkpoint is not uniquely bound to the diagnostic contract"
            )
        active_patch_checkpoint = matches[0]
        if formal_external_enabled:
            if int(
                formal_external_rank_transfer_settings.get(
                    "formal_artifact_version", 1
                )
            ) in (2, 3):
                acc = FormalRoutedExternalGDINORankTransferAccumulator(
                    formal_external_rank_transfer_settings,
                    manifest=manifest,
                    run_prefix=run_prefix,
                )
            else:
                acc = FormalExternalGDINORankTransferAccumulator(
                    formal_external_rank_transfer_settings,
                    manifest=manifest,
                    run_prefix=run_prefix,
                )
        else:
            acc = DiagnosticExternalGDINORankTransferAccumulator(
                diagnostic_external_rank_transfer_settings,
                patch_cfg=cfg,
            )
    elif diagnostic_enabled:
        if (
            diagnostic_patch_rank_contract_weight is None
            or diagnostic_patch_rank_candidate_topk is None
        ):
            raise ValueError(
                "diagnostic patch-rank evaluation requires the contract weight and Top-K"
            )
        if records_output_dir is not None:
            raise ValueError(
                "diagnostic-only patch-rank sweeps cannot emit formal eval records"
            )
        acc = DiagnosticPatchRankAccumulator(
            diagnostic_patch_rank_weights,
            diagnostic_patch_rank_contract_weight,
            diagnostic_patch_rank_candidate_topk,
            max(topks),
        )
    else:
        acc = RefExpAccumulator(
            betas,
            topks,
            manifest=manifest,
            run_prefix=run_prefix,
            data_driven_query_diagnostics=data_driven_query_diagnostics,
        )
    start = time.time()
    total_batches = len(loader)
    if formal_external_enabled:
        print(
            f"[INFO] formal external-GDINO rank transfer "
            f"patch_ckpt={Path(ckpt_path).name} dataset={dataset_name} "
            f"expressions={len(loader.dataset)} batches={total_batches} "
            f"batch_size={batch_size} artifact="
            f"{formal_external_rank_transfer_settings['artifact_identity']['sha256']}"
        )
    elif external_diagnostic_enabled:
        print(
            f"[INFO] diagnostic external-GDINO rank transfer "
            f"patch_ckpt={Path(ckpt_path).name} dataset={dataset_name} "
            f"expressions={len(loader.dataset)} batches={total_batches} "
            f"batch_size={batch_size} grid_points={len(acc.grid)}"
        )
    elif diagnostic_enabled:
        print(
            f"[INFO] diagnostic patch-rank eval ckpt={Path(ckpt_path).name} "
            f"dataset={dataset_name} expressions={len(loader.dataset)} "
            f"batches={total_batches} batch_size={batch_size} "
            f"weights={diagnostic_patch_rank_weights}"
        )
    else:
        print(
            f"[INFO] refexp eval ckpt={Path(ckpt_path).name} dataset={dataset_name} "
            f"expressions={len(loader.dataset)} batches={total_batches} batch_size={batch_size} betas={betas}"
        )
    for batch_i, batch in enumerate(loader):
        if max_batches > 0 and batch_i >= max_batches:
            break
        if max_images > 0 and acc.total >= max_images:
            break
        validate_eval_manifest_batch_alignment(
            list(batch[1]), manifest, int(acc.total)
        )
        validated_full_expressions: Optional[List[str]] = None
        if formal_external_enabled:
            validated_full_expressions = _validate_formal_external_caption_provenance(
                list(batch[1]),
                manifest,
                int(acc.total),
                settings=formal_external_rank_transfer_settings,
                routed_canonical_id_to_caption=(
                    routed_canonical_id_to_caption
                ),
            )
        outputs, targets = _forward(model, batch, device, amp=amp, cfg=cfg)
        if (
            formal_external_enabled
            and int(
                formal_external_rank_transfer_settings.get(
                    "formal_artifact_version", 1
                )
            )
            == 3
        ):
            if validated_full_expressions is None:
                raise AssertionError("formal full-text provenance was not validated")
            outputs[_FULLTEXT_CAPTIONS_OUTPUT_KEY] = list(
                validated_full_expressions
            )
        if external_diagnostic_enabled or formal_external_enabled:
            external_outputs, external_targets = _forward_external_gdino_rank_adapter(
                external_gdino_model,
                batch,
                device,
                amp=amp,
                cfg=external_gdino_cfg,
            )
            _validate_dual_forward_targets(targets, external_targets)
            acc.update(outputs, external_outputs, targets)
        elif diagnostic_enabled:
            acc.update(outputs, targets)
        else:
            acc.update(outputs, targets, cfg=cfg)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if log_every > 0 and (batch_i == 0 or (batch_i + 1) % log_every == 0):
            elapsed = time.time() - start
            done = batch_i + 1
            target_batches = min(total_batches, max_batches) if max_batches > 0 else total_batches
            if max_images > 0:
                target_batches = min(target_batches, math.ceil(max_images / max(1, batch_size)))
            eta = elapsed / max(1, done) * max(0, target_batches - done)
            print(
                f"[INFO] {dataset_name} {Path(ckpt_path).name}: batch {done}/{target_batches}, "
                f"expressions={acc.total}, elapsed={elapsed/60:.1f}m, eta={eta/60:.1f}m"
            )
    rows = acc.results()
    elapsed = time.time() - start
    for row in rows:
        if formal_external_enabled:
            formal_version = int(
                formal_external_rank_transfer_settings.get(
                    "formal_artifact_version", 1
                )
            )
            if formal_version in (2, 3):
                route_label = (
                    "formal_external_gdino_caption_route"
                    if formal_version == 2
                    else "formal_external_gdino_fulltext_gated_caption_route"
                )
                run_suffix = (
                    f"{route_label}="
                    f"{formal_external_rank_transfer_settings['artifact_identity']['sha256'][:16]}"
                )
            else:
                run_suffix = (
                    "formal_external_gdino_rank_transfer="
                    f"{formal_external_rank_transfer_settings['artifact_identity']['sha256'][:16]}"
                )
        elif external_diagnostic_enabled:
            if (
                row.get("diagnostic_descriptor_kind")
                == _EXTERNAL_GDINO_BASE_IDENTITY_KIND
            ):
                run_suffix = (
                    "diagnostic_external_gdino_base_identity=direct_global_argmax"
                )
            elif (
                row.get("diagnostic_descriptor_kind")
                == _PATCH_INTERNAL_RANK_IDENTITY_KIND
            ):
                run_suffix = (
                    "diagnostic_patch_internal_rank_identity=standard_ref_beta0"
                )
            else:
                power = row["diagnostic_iou_power"]
                power_suffix = (
                    "none" if power is None else f"{float(power):g}"
                )
                run_suffix = (
                    "diagnostic_external_gdino_rank_transfer="
                    f"{row['diagnostic_transfer_mode']},p={power_suffix},"
                    f"patch={float(row['diagnostic_patch_weight']):g},"
                    f"text={float(row['diagnostic_text_weight']):g}"
                )
        elif diagnostic_enabled:
            run_suffix = (
                "diagnostic_patch_rank_weight="
                f"{float(row['diagnostic_patch_rank_weight']):g}"
            )
        else:
            run_suffix = f"b{row['beta']:g}"
        row.update(
            {
                "run_id": f"{run_prefix}:{run_suffix}",
                "checkpoint": str(ckpt_path),
                "checkpoint_name": Path(ckpt_path).name,
                "checkpoint_run_prefix": run_prefix,
                "dataset": dataset_name,
                "seconds": float(elapsed),
                "batch_size": int(batch_size),
                "num_workers": int(num_workers),
                "seed": int(seed),
                "seed_protocol": SPLIT_SEED_PROTOCOL,
                "max_batches": int(max_batches),
                "max_images": int(max_images),
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
        if data_driven_query_diagnostics:
            row.update(
                {
                    "data_driven_query_diagnostics": True,
                    "data_driven_query_diagnostic_contract": (
                        _DATA_DRIVEN_QUERY_DIAGNOSTIC_CONTRACT
                    ),
                }
            )
        if external_diagnostic_enabled or formal_external_enabled:
            assert active_patch_checkpoint is not None
            settings = active_external_settings
            row.update(
                {
                    "patch_checkpoint_sha256": active_patch_checkpoint["sha256"],
                    "patch_config": settings["patch_config"]["path"],
                    "patch_config_sha256": settings["patch_config"]["sha256"],
                    "external_gdino_checkpoint": settings["external_checkpoint"]["path"],
                    "external_gdino_checkpoint_sha256": settings["external_checkpoint"]["sha256"],
                    "external_gdino_config": settings["external_config"]["path"],
                    "external_gdino_config_sha256": settings["external_config"]["sha256"],
                    "external_gdino_rank_score_key": _EXTERNAL_GDINO_RANK_SCORE_KEY,
                    "external_gdino_query_count": _EXTERNAL_GDINO_QUERY_COUNT,
                    "transfer_contract_version": (
                        int(settings.get("formal_artifact_version", 1))
                        if formal_external_enabled
                        and int(settings.get("formal_artifact_version", 1)) in (2, 3)
                        else _EXTERNAL_RANK_TRANSFER_CONTRACT_VERSION
                    ),
                }
            )
            if formal_external_enabled:
                row.update(
                    {
                        "external_transfer_artifact": settings["artifact"]["path"],
                        "external_transfer_artifact_file_sha256": settings["artifact"][
                            "sha256"
                        ],
                        "external_transfer_artifact_sha256": settings[
                            "artifact_identity"
                        ]["sha256"],
                    }
                )
                if int(settings.get("formal_artifact_version", 1)) in (2, 3):
                    row["amp"] = bool(amp)
        if (
            records_output_dir is not None
            and not diagnostic_enabled
            and not external_diagnostic_enabled
        ):
            if formal_external_enabled:
                records_path = Path(records_output_dir) / (
                    f"{run_prefix}__{_safe_name(dataset_name)}__formal_external_"
                    f"{settings['artifact_identity']['sha256'][:16]}.records.jsonl"
                )
                eval_records = acc.eval_records
            else:
                beta = float(row["beta"])
                records_path = Path(records_output_dir) / (
                    f"{run_prefix}__{_safe_name(dataset_name)}__"
                    f"{_safe_name(f'b{beta:g}')}.records.jsonl"
                )
                eval_records = acc.eval_records[beta]
            write_eval_records(records_path, eval_records)
            row.update(
                {
                    "records_jsonl": str(records_path),
                    "manifest_sha256": manifest.sha256,
                    "manifest_n": manifest.size,
                    "invalid_records": int(
                        sum(not bool(record.get("valid")) for record in eval_records)
                    ),
                }
            )
            if (
                formal_external_enabled
                and int(settings.get("formal_artifact_version", 1)) in (2, 3)
            ):
                row.update(
                    {
                        "records_sha256": _sha256_file(records_path),
                        "records_size_bytes": int(records_path.stat().st_size),
                    }
                )
    return rows


def _mean_metric(results: List[Dict[str, Any]], run_id: str, metric: str) -> float:
    vals = [float(r.get(metric, 0.0)) for r in results if r["run_id"] == run_id]
    return sum(vals) / max(1, len(vals))


def _diagnostic_summary_metadata(
    settings: Dict[str, Any],
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    fixed_grid = [float(value) for value in settings["weights"]]
    fixed_rows: Dict[str, int] = {}
    fixed_seeds: Dict[str, int] = {}
    observed_weights: Dict[str, set] = {}
    for row in results:
        if not bool(row.get("diagnostic_only", False)) or bool(
            row.get("formal_gate_eligible", True)
        ):
            raise ValueError("diagnostic summary cannot mix formal evaluation rows")
        dataset = str(row["dataset"])
        row_count = int(row["num_expressions"])
        seed = int(row["seed"])
        if dataset in fixed_rows and fixed_rows[dataset] != row_count:
            raise ValueError(
                f"diagnostic row count drifted for {dataset}: "
                f"{fixed_rows[dataset]} vs {row_count}"
            )
        if dataset in fixed_seeds and fixed_seeds[dataset] != seed:
            raise ValueError(
                f"diagnostic seed drifted for {dataset}: "
                f"{fixed_seeds[dataset]} vs {seed}"
            )
        fixed_rows[dataset] = row_count
        fixed_seeds[dataset] = seed
        observed_weights.setdefault(dataset, set()).add(
            float(row["diagnostic_patch_rank_weight"])
        )
    expected_weights = set(fixed_grid)
    for dataset, dataset_weights in observed_weights.items():
        if dataset_weights != expected_weights:
            raise ValueError(
                f"diagnostic grid drifted for {dataset}: "
                f"expected={fixed_grid}, observed={sorted(dataset_weights)}"
            )
    return {
        "diagnostic_only": True,
        "formal_gate_eligible": False,
        "diagnostic_kind": "patch_rank_weight_sweep",
        "evaluation_seed_protocol": SPLIT_SEED_PROTOCOL,
        "single_forward_per_batch": True,
        "candidate_admission": "unchanged_stage_b_v11_candidate_idx",
        "contract_patch_rank_weight": float(settings["contract_weight"]),
        "candidate_topk": int(settings["candidate_topk"]),
        "fixed_grid": fixed_grid,
        "fixed_rows": fixed_rows,
        "fixed_seeds": fixed_seeds,
    }


def _external_result_grid_key(
    row: Dict[str, Any],
) -> Tuple[str, Optional[float], float, float]:
    power = row.get("diagnostic_iou_power")
    return (
        str(row["diagnostic_transfer_mode"]),
        None if power is None else float(power),
        float(row["diagnostic_patch_weight"]),
        float(row["diagnostic_text_weight"]),
    )


def _diagnostic_external_rank_transfer_summary_metadata(
    settings: Dict[str, Any],
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    expected_grid = {_external_grid_key(row) for row in settings["fixed_grid"]}
    observed_grid: Dict[Tuple[str, str], set] = {}
    observed_identity: Dict[Tuple[str, str], int] = {}
    observed_external_base_identity: Dict[Tuple[str, str], int] = {}
    fixed_rows: Dict[str, int] = {}
    fixed_seeds: Dict[str, int] = {}
    observed_patch_checkpoints: Dict[str, Dict[str, Any]] = {}
    expected_patch_by_path = {
        component["path"]: component for component in settings["patch_checkpoints"]
    }
    for row in results:
        if not bool(row.get("diagnostic_only", False)) or bool(
            row.get("formal_gate_eligible", True)
        ):
            raise ValueError(
                "external rank-transfer summary cannot mix formal evaluation rows"
            )
        if "records_jsonl" in row:
            raise ValueError(
                "external rank-transfer diagnostics must not contain formal records"
            )
        if int(row.get("candidate_topk", -1)) != int(settings["candidate_topk"]):
            raise ValueError("external rank-transfer candidate Top-K drifted")
        identity_checks = {
            "external_gdino_checkpoint_sha256": settings["external_checkpoint"][
                "sha256"
            ],
            "external_gdino_config_sha256": settings["external_config"]["sha256"],
            "patch_config_sha256": settings["patch_config"]["sha256"],
            "external_gdino_rank_score_key": _EXTERNAL_GDINO_RANK_SCORE_KEY,
            "external_gdino_query_count": _EXTERNAL_GDINO_QUERY_COUNT,
            "transfer_contract_version": _EXTERNAL_RANK_TRANSFER_CONTRACT_VERSION,
        }
        for key, expected in identity_checks.items():
            if row.get(key) != expected:
                raise ValueError(
                    f"external rank-transfer identity drift for {key}: "
                    f"expected={expected!r}, observed={row.get(key)!r}"
                )
        checkpoint_path = str(Path(row["checkpoint"]).expanduser().resolve())
        component = expected_patch_by_path.get(checkpoint_path)
        if component is None or row.get("patch_checkpoint_sha256") != component["sha256"]:
            raise ValueError(
                "external rank-transfer patch checkpoint binding is invalid"
            )
        observed_patch_checkpoints[checkpoint_path] = component

        dataset = str(row["dataset"])
        row_count = int(row["num_expressions"])
        seed = int(row["seed"])
        if dataset in fixed_rows and fixed_rows[dataset] != row_count:
            raise ValueError(
                f"external rank-transfer row count drifted for {dataset}"
            )
        if dataset in fixed_seeds and fixed_seeds[dataset] != seed:
            raise ValueError(f"external rank-transfer seed drifted for {dataset}")
        fixed_rows[dataset] = row_count
        fixed_seeds[dataset] = seed
        group = (checkpoint_path, dataset)
        descriptor_kind = row.get(
            "diagnostic_descriptor_kind", "external_rank_transfer"
        )
        if descriptor_kind == _PATCH_INTERNAL_RANK_IDENTITY_KIND:
            identity_checks = {
                "diagnostic_standard_ref_beta": 0.0,
                "uses_external_rank_score": False,
                "uses_external_box": False,
                "uses_fusion_weights": False,
                "patch_internal_rank_identity_contract_version": (
                    _PATCH_INTERNAL_RANK_IDENTITY_CONTRACT_VERSION
                ),
            }
            for key, expected in identity_checks.items():
                if row.get(key) != expected:
                    raise ValueError(
                        "patch internal-rank identity contract drift for "
                        f"{key}: expected={expected!r}, observed={row.get(key)!r}"
                    )
            forbidden = {
                "diagnostic_transfer_mode",
                "diagnostic_iou_power",
                "diagnostic_patch_weight",
                "diagnostic_text_weight",
            }
            present = sorted(forbidden.intersection(row))
            if present:
                raise ValueError(
                    "patch internal-rank identity must not be represented as "
                    f"external transfer/fusion fields: {present}"
                )
            observed_identity[group] = observed_identity.get(group, 0) + 1
        elif descriptor_kind == _EXTERNAL_GDINO_BASE_IDENTITY_KIND:
            base_identity_checks = {
                "diagnostic_identity_kind": _EXTERNAL_GDINO_BASE_IDENTITY_ID,
                "diagnostic_score_key": _EXTERNAL_GDINO_BASE_SCORE_KEY,
                "diagnostic_query_count": _EXTERNAL_GDINO_QUERY_COUNT,
                "diagnostic_output_box_source": (
                    "external_outputs.pred_boxes_at_direct_global_argmax"
                ),
                "diagnostic_standard_ref_beta": 0.0,
                "diagnostic_score_source": _EXTERNAL_GDINO_BASE_SCORE_KEY,
                "diagnostic_winner_rule": (
                    "first_argmax_over_full_external_query_axis"
                ),
                "diagnostic_query_domain": "all_900_external_gdino_queries",
                "uses_external_base_score": True,
                "uses_external_rank_score": False,
                "uses_external_box": True,
                "uses_adapter_rank_residual": False,
                "uses_patch_top50_admission": False,
                "uses_top_query_mapping": False,
                "uses_fusion_weights": False,
                "external_gdino_base_identity_contract_version": (
                    _EXTERNAL_GDINO_BASE_IDENTITY_CONTRACT_VERSION
                ),
                "external_query_count": _EXTERNAL_GDINO_QUERY_COUNT,
                "candidate_topk_scope": (
                    "run_context_only_not_used_by_this_descriptor"
                ),
            }
            for key, expected in base_identity_checks.items():
                if row.get(key) != expected:
                    raise ValueError(
                        "external GDINO base identity contract drift for "
                        f"{key}: expected={expected!r}, observed={row.get(key)!r}"
                    )
            forbidden = {
                "diagnostic_transfer_mode",
                "diagnostic_iou_power",
                "diagnostic_patch_weight",
                "diagnostic_text_weight",
            }
            present = sorted(forbidden.intersection(row))
            if present:
                raise ValueError(
                    "external GDINO base identity must not be represented as "
                    f"a transfer/fusion grid point: {present}"
                )
            observed_external_base_identity[group] = (
                observed_external_base_identity.get(group, 0) + 1
            )
        elif descriptor_kind == "external_rank_transfer":
            observed_grid.setdefault(group, set()).add(
                _external_result_grid_key(row)
            )
        else:
            raise ValueError(
                "unknown external diagnostic descriptor kind: "
                f"{descriptor_kind!r}"
            )
    for group, grid in observed_grid.items():
        if grid != expected_grid:
            raise ValueError(
                "external rank-transfer fixed grid drifted for "
                f"{group}: expected={sorted(map(str, expected_grid))}, "
                f"observed={sorted(map(str, grid))}"
            )
    expected_identity = bool(
        settings.get("include_patch_internal_rank_identity", False)
    )
    expected_external_base_identity = bool(
        settings.get("include_external_gdino_base_identity", False)
    )
    for group in observed_grid:
        identity_count = observed_identity.get(group, 0)
        expected_count = 1 if expected_identity else 0
        if identity_count != expected_count:
            raise ValueError(
                "patch internal-rank identity descriptor count drifted for "
                f"{group}: expected={expected_count}, observed={identity_count}"
            )
        external_base_count = observed_external_base_identity.get(group, 0)
        expected_external_base_count = 1 if expected_external_base_identity else 0
        if external_base_count != expected_external_base_count:
            raise ValueError(
                "external GDINO base identity descriptor count drifted for "
                f"{group}: expected={expected_external_base_count}, "
                f"observed={external_base_count}"
            )
    unexpected_identity_groups = set(observed_identity).difference(observed_grid)
    if unexpected_identity_groups:
        raise ValueError(
            "patch internal-rank identity appeared without its external transfer "
            f"grid: {sorted(unexpected_identity_groups)}"
        )
    unexpected_external_base_groups = set(
        observed_external_base_identity
    ).difference(observed_grid)
    if unexpected_external_base_groups:
        raise ValueError(
            "external GDINO base identity appeared without its diagnostic "
            f"transfer grid: {sorted(unexpected_external_base_groups)}"
        )
    return {
        "diagnostic_only": True,
        "formal_gate_eligible": False,
        "diagnostic_kind": "external_gdino_rank_transfer",
        "evaluation_seed_protocol": SPLIT_SEED_PROTOCOL,
        "single_forward_per_model_per_batch": True,
        "model_forwards_per_batch": {"patch_model": 1, "external_gdino_model": 1},
        "candidate_admission": "unchanged_exact_stage_a_top50",
        "candidate_topk": int(settings["candidate_topk"]),
        "fixed_grid": list(settings["fixed_grid"]),
        "fixed_transfer_modes": list(settings["transfer_modes"]),
        "fixed_iou_powers": list(settings["iou_powers"]),
        "fixed_patch_weights": list(settings["patch_weights"]),
        "fixed_text_weights": list(settings["text_weights"]),
        "include_patch_internal_rank_identity": expected_identity,
        "patch_internal_rank_identity_contract": dict(
            settings["patch_internal_rank_identity_contract"]
        ),
        "include_external_gdino_base_identity": expected_external_base_identity,
        "external_gdino_base_identity_contract": dict(
            settings["external_gdino_base_identity_contract"]
        ),
        "fixed_rows": fixed_rows,
        "fixed_seeds": fixed_seeds,
        "patch_config": dict(settings["patch_config"]),
        "patch_checkpoints": [
            observed_patch_checkpoints[path]
            for path in sorted(observed_patch_checkpoints)
        ],
        "external_gdino_config": dict(settings["external_config"]),
        "external_gdino_checkpoint": dict(settings["external_checkpoint"]),
        "transfer_contract": dict(settings["transfer_contract"]),
    }


def _formal_external_rank_transfer_summary_metadata(
    settings: Dict[str, Any],
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if len(settings["fixed_grid"]) != 1:
        raise ValueError("formal external summary requires exactly one transfer point")
    if len(results) != len(REF_SPLIT_ORDER):
        raise ValueError(
            "formal external summary requires exactly one row for every canonical "
            f"Ref8 split, got {len(results)} rows"
        )
    observed_order = [str(row.get("dataset", "")) for row in results]
    if observed_order != list(REF_SPLIT_ORDER):
        raise ValueError(
            "formal external summary rows must follow the exact canonical Ref8 order"
        )
    descriptor = settings["fixed_grid"][0]
    expected_key = _external_grid_key(descriptor)
    fixed_rows: Dict[str, int] = {}
    fixed_seeds: Dict[str, int] = {}
    observed_datasets = set()
    artifact_sha256 = str(settings["artifact_identity"]["sha256"])
    patch_checkpoint = settings["patch_checkpoints"][0]
    for row in results:
        if bool(row.get("diagnostic_only", True)) or not bool(
            row.get("formal_gate_eligible", False)
        ):
            raise ValueError("formal external summary cannot contain diagnostic rows")
        dataset = str(row["dataset"])
        if dataset in observed_datasets:
            raise ValueError(
                f"formal external summary has duplicate rows for {dataset}"
            )
        observed_datasets.add(dataset)
        if "records_jsonl" not in row:
            raise ValueError(
                "formal external summary requires canonical per-example records"
            )
        records_path = Path(str(row["records_jsonl"])).expanduser().resolve()
        if not records_path.is_file():
            raise ValueError(f"formal external records are missing: {records_path}")
        records = list(_iter_jsonl(records_path))
        num_expressions = int(row["num_expressions"])
        manifest_n = int(row.get("manifest_n", -1))
        if manifest_n <= 0 or num_expressions != manifest_n:
            raise ValueError(
                f"formal external evaluation is partial for {dataset}: "
                f"num_expressions={num_expressions}, manifest_n={manifest_n}"
            )
        if len(records) != num_expressions:
            raise ValueError(
                f"formal external record count drifted for {dataset}: "
                f"{len(records)} != {row['num_expressions']}"
            )
        for index, record in enumerate(records):
            record_checks = {
                "schema": RECORD_SCHEMA,
                "task": "ref",
                "manifest_index": index,
                "manifest_sha256": row.get("manifest_sha256"),
                "manifest_n": row.get("manifest_n"),
                "run_id": row.get("run_id"),
                "external_transfer_artifact_sha256": artifact_sha256,
            }
            for key, expected in record_checks.items():
                if record.get(key) != expected:
                    raise ValueError(
                        f"formal external record {index} drifted for {key}"
                    )
            if (
                not isinstance(record.get("canonical_caption"), str)
                or not record["canonical_caption"].strip()
                or not isinstance(record.get("canonical_class_norm"), str)
                or not record["canonical_class_norm"].strip()
            ):
                raise ValueError(
                    f"formal external record {index} lacks canonical caption provenance"
                )
        if int(row.get("candidate_topk", -1)) != int(settings["candidate_topk"]):
            raise ValueError("formal external candidate Top-K drifted")
        observed_key = (
            str(row.get("formal_transfer_mode")),
            (
                None
                if row.get("formal_iou_power") is None
                else float(row["formal_iou_power"])
            ),
            float(row.get("formal_patch_weight", float("nan"))),
            float(row.get("formal_text_weight", float("nan"))),
        )
        if observed_key != expected_key:
            raise ValueError("formal external transfer point drifted")
        identity_checks = {
            "external_transfer_artifact_sha256": artifact_sha256,
            "external_transfer_artifact_file_sha256": settings["artifact"][
                "sha256"
            ],
            "patch_checkpoint_sha256": patch_checkpoint["sha256"],
            "external_gdino_checkpoint_sha256": settings["external_checkpoint"][
                "sha256"
            ],
            "external_gdino_config_sha256": settings["external_config"]["sha256"],
            "patch_config_sha256": settings["patch_config"]["sha256"],
            "external_gdino_rank_score_key": _EXTERNAL_GDINO_RANK_SCORE_KEY,
            "external_gdino_query_count": _EXTERNAL_GDINO_QUERY_COUNT,
            "transfer_contract_version": _EXTERNAL_RANK_TRANSFER_CONTRACT_VERSION,
        }
        for key, expected in identity_checks.items():
            if row.get(key) != expected:
                raise ValueError(
                    f"formal external identity drift for {key}: "
                    f"{row.get(key)!r} != {expected!r}"
                )
        checkpoint_path = str(Path(row["checkpoint"]).expanduser().resolve())
        if checkpoint_path != patch_checkpoint["path"]:
            raise ValueError("formal external patch checkpoint path drifted")
        fixed_rows[dataset] = int(row["num_expressions"])
        fixed_seeds[dataset] = int(row["seed"])

    expected_seed_map = settings["evaluation_protocol"]["split_seeds"]
    if set(fixed_seeds) != set(REF_SPLIT_ORDER):
        raise ValueError("formal external summary lost canonical Ref8 seed coverage")
    for dataset, seed in fixed_seeds.items():
        if dataset not in expected_seed_map or int(expected_seed_map[dataset]) != seed:
            raise ValueError(f"formal external stable seed drifted for {dataset}")
    return {
        "diagnostic_only": False,
        "formal_gate_eligible": True,
        "evaluation_kind": "formal_external_gdino_rank_transfer",
        "model_forwards_per_batch": {"patch_model": 1, "external_gdino_model": 1},
        "candidate_admission": "unchanged_exact_stage_a_top50",
        "candidate_topk": int(settings["candidate_topk"]),
        "selected_transfer": dict(descriptor),
        "caption_provenance": dict(settings["caption_provenance"]),
        "evaluation_protocol": dict(settings["evaluation_protocol"]),
        "fixed_rows": fixed_rows,
        "fixed_seeds": fixed_seeds,
        "patch_config": dict(settings["patch_config"]),
        "patch_checkpoint": dict(patch_checkpoint),
        "external_gdino_config": dict(settings["external_config"]),
        "external_gdino_checkpoint": dict(settings["external_checkpoint"]),
        "external_transfer_artifact": dict(settings["artifact"]),
        "external_transfer_artifact_identity": dict(settings["artifact_identity"]),
        "transfer_contract": dict(settings["transfer_contract"]),
    }


def _formal_routed_external_rank_transfer_summary_metadata(
    settings: Dict[str, Any],
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    formal_version = int(settings.get("formal_artifact_version", 0))
    if formal_version not in (2, 3):
        raise ValueError("formal routed summary requires artifact version 2 or 3")
    if len(results) != len(REF_SPLIT_ORDER):
        raise ValueError(
            "formal routed summary requires exactly one row for every canonical "
            f"Ref8 split, got {len(results)} rows"
        )
    observed_order = [str(row.get("dataset", "")) for row in results]
    if observed_order != list(REF_SPLIT_ORDER):
        raise ValueError(
            "formal routed summary rows must follow the exact canonical Ref8 order"
        )
    artifact_sha256 = str(settings["artifact_identity"]["sha256"])
    selection = (
        settings["route_selection"]
        if formal_version == 2
        else settings["canonical_route_selection"]
    )
    selection_contract = selection["selection_contract"]
    selection_artifact = selection["artifact"]
    selection_identity_sha256 = str(selection["artifact_identity"]["sha256"])
    policy_sha256 = str(settings["route_policy_sha256"])
    default_descriptor_id = str(settings["route_default_descriptor_id"])
    overrides = dict(
        settings["route_overrides"]
        if formal_version == 2
        else settings["route_unconditional_overrides"]
    )
    conditional_overrides = (
        {}
        if formal_version == 2
        else dict(settings["route_conditional_overrides"])
    )
    fulltext_gate_artifact = (
        None
        if formal_version == 2
        else dict(settings["fulltext_route_gate_artifact"])
    )
    routed_v2_component = (
        None if formal_version == 2 else dict(settings["routed_v2_artifact"])
    )
    registry = dict(settings["descriptor_registry"])
    patch_checkpoint = settings["patch_checkpoints"][0]
    fixed_rows: Dict[str, int] = {}
    fixed_seeds: Dict[str, int] = {}
    descriptor_counts_by_split: Dict[str, Dict[str, int]] = {}
    caption_counts_by_split: Dict[str, Dict[str, Dict[str, Any]]] = {}
    aggregate_descriptor_counts = {key: 0 for key in registry}
    aggregate_caption_counts: Dict[str, Dict[str, Any]] = {}
    aggregate_fulltext_gate_counts: Optional[Dict[str, int]] = (
        None
        if formal_version == 2
        else {
            "conditional_predicate_matched": 0,
            "conditional_fallback_matched": 0,
            "unconditional_override": 0,
            "default": 0,
        }
    )
    fulltext_gate_counts_by_split: Optional[Dict[str, Dict[str, int]]] = (
        None if formal_version == 2 else {}
    )

    def require_close(observed: Any, expected: float, *, label: str) -> None:
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(float(observed))
            or not math.isclose(
                float(observed), float(expected), rel_tol=0.0, abs_tol=1e-12
            )
        ):
            raise ValueError(f"formal routed summary metric drift for {label}")

    for row in results:
        if bool(row.get("diagnostic_only", True)) or not bool(
            row.get("formal_gate_eligible", False)
        ):
            raise ValueError("formal routed summary cannot contain diagnostic rows")
        dataset = str(row["dataset"])
        records_path = Path(str(row.get("records_jsonl", ""))).expanduser().resolve()
        if not records_path.is_file():
            raise ValueError(f"formal routed records are missing: {records_path}")
        records_identity = _file_identity(
            str(records_path), label=f"formal routed records {dataset}"
        )
        if (
            row.get("records_sha256") != records_identity["sha256"]
            or int(row.get("records_size_bytes", -1))
            != records_identity["size_bytes"]
        ):
            raise ValueError(f"formal routed records identity drifted for {dataset}")
        records = list(_iter_jsonl(records_path))
        num_expressions = int(row["num_expressions"])
        manifest_n = int(row.get("manifest_n", -1))
        if manifest_n <= 0 or num_expressions != manifest_n:
            raise ValueError(
                f"formal routed evaluation is partial for {dataset}: "
                f"num_expressions={num_expressions}, manifest_n={manifest_n}"
            )
        if len(records) != num_expressions:
            raise ValueError(
                f"formal routed record count drifted for {dataset}: "
                f"{len(records)} != {num_expressions}"
            )

        descriptor_counts = {key: 0 for key in registry}
        caption_counts: Dict[str, Dict[str, Any]] = {}
        fulltext_gate_counts: Optional[Dict[str, int]] = (
            None
            if formal_version == 2
            else {
                "conditional_predicate_matched": 0,
                "conditional_fallback_matched": 0,
                "unconditional_override": 0,
                "default": 0,
            }
        )
        correct50 = 0
        correct25 = 0
        iou_sum = 0.0
        patch_oracle50 = 0
        all_query_oracle50 = 0
        invalid = 0
        for index, record in enumerate(records):
            record_checks = {
                "schema": RECORD_SCHEMA,
                "task": "ref",
                "manifest_index": index,
                "manifest_sha256": row.get("manifest_sha256"),
                "manifest_n": row.get("manifest_n"),
                "run_id": row.get("run_id"),
                "external_transfer_artifact_sha256": artifact_sha256,
                "caption_route_selection_artifact_path": selection_artifact["path"],
                "caption_route_selection_artifact_file_sha256": selection_artifact[
                    "sha256"
                ],
                "caption_route_selection_artifact_size_bytes": selection_artifact[
                    "size_bytes"
                ],
                "caption_route_selection_artifact_identity_sha256": (
                    selection_identity_sha256
                ),
                "caption_route_policy_sha256": policy_sha256,
                "caption_route_contract_version": formal_version,
                "selected_box_format": "normalized_xyxy",
            }
            if formal_version == 3:
                assert fulltext_gate_artifact is not None
                assert routed_v2_component is not None
                record_checks.update(
                    {
                        "fulltext_route_gate_contract_version": (
                            _FULLTEXT_ROUTE_GATE_CONTRACT_VERSION
                        ),
                        "fulltext_route_gate_artifact_path": (
                            fulltext_gate_artifact["path"]
                        ),
                        "fulltext_route_gate_artifact_file_sha256": (
                            fulltext_gate_artifact["sha256"]
                        ),
                        "fulltext_route_gate_artifact_size_bytes": (
                            fulltext_gate_artifact["size_bytes"]
                        ),
                        "fulltext_route_gate_artifact_identity_sha256": (
                            settings["fulltext_route_gate_artifact_identity"][
                                "sha256"
                            ]
                        ),
                        "routed_v2_artifact_path": routed_v2_component[
                            "artifact"
                        ]["path"],
                        "routed_v2_artifact_file_sha256": routed_v2_component[
                            "artifact"
                        ]["sha256"],
                        "routed_v2_artifact_size_bytes": routed_v2_component[
                            "artifact"
                        ]["size_bytes"],
                        "routed_v2_artifact_identity_sha256": (
                            routed_v2_component["artifact_identity"]["sha256"]
                        ),
                        "fulltext_route_policy_sha256": policy_sha256,
                        "canonical_route_policy_sha256": settings[
                            "canonical_route_policy_sha256"
                        ],
                        "fulltext_route_gate_token_count_contract_sha256": (
                            _canonical_json_sha256(FULLTEXT_TOKEN_COUNT_CONTRACT)
                        ),
                    }
                )
            for key, expected in record_checks.items():
                if record.get(key) != expected:
                    raise ValueError(
                        f"formal routed record {index} drifted for {key}"
                    )
            raw_caption = record.get("canonical_caption")
            route_caption = record.get("caption_route_caption")
            if (
                not isinstance(raw_caption, str)
                or not raw_caption.strip()
                or not isinstance(route_caption, str)
                or not route_caption
                or _norm_text(raw_caption) != route_caption
                or record.get("canonical_class_norm") != route_caption
            ):
                raise ValueError(
                    f"formal routed record {index} caption provenance drifted"
                )
            if formal_version == 2:
                expected_descriptor = overrides.get(
                    route_caption, default_descriptor_id
                )
            else:
                full_expression = record.get(
                    "fulltext_route_gate_full_expression"
                )
                if (
                    not isinstance(full_expression, str)
                    or not full_expression.strip()
                    or _WS_RE.sub(" ", full_expression.strip())
                    != full_expression
                ):
                    raise ValueError(
                        f"formal full-text record {index} expression drifted"
                    )
                token_count = len(
                    _FULLTEXT_LEXICAL_TOKEN_RE.findall(full_expression.lower())
                )
                if token_count <= 0:
                    raise ValueError(
                        f"formal full-text record {index} has no lexical tokens"
                    )
                conditional = conditional_overrides[FULLTEXT_GATED_CAPTION]
                max_tokens = int(conditional["predicate"]["max_tokens"])
                predicate_matched = False
                fallback_matched = False
                if route_caption in overrides:
                    expected_descriptor = overrides[route_caption]
                    expected_source = "unconditional_override"
                elif route_caption == FULLTEXT_GATED_CAPTION:
                    predicate_matched = token_count <= max_tokens
                    fallback_matched = not predicate_matched
                    expected_descriptor = conditional[
                        "descriptor_id"
                        if predicate_matched
                        else "fallback_descriptor_id"
                    ]
                    expected_source = (
                        "conditional_predicate_matched"
                        if predicate_matched
                        else "conditional_fallback_matched"
                    )
                else:
                    expected_descriptor = default_descriptor_id
                    expected_source = "default"
                gate_checks = {
                    "fulltext_route_gate_token_count": token_count,
                    "fulltext_route_gate_max_tokens": max_tokens,
                    "fulltext_route_gate_predicate_matched": predicate_matched,
                    "fulltext_route_gate_fallback_matched": fallback_matched,
                    "fulltext_route_gate_source": (
                        "external_base_direct"
                        if fallback_matched
                        else "formal_routed_v2"
                    ),
                    "fulltext_route_gate_route_kind": expected_source,
                    "fulltext_route_gate_selected_descriptor_id": (
                        expected_descriptor
                    ),
                    "fulltext_route_gate_predicate_kind": conditional[
                        "predicate"
                    ]["kind"],
                }
                for key, expected in gate_checks.items():
                    if record.get(key) != expected:
                        raise ValueError(
                            f"formal full-text record {index} gate drifted for {key}"
                        )
                assert fulltext_gate_counts is not None
                fulltext_gate_counts[expected_source] += 1
            descriptor_id = record.get("caption_route_descriptor_id")
            if descriptor_id != expected_descriptor or descriptor_id not in registry:
                raise ValueError(
                    f"formal routed record {index} descriptor routing drifted"
                )
            descriptor = registry[descriptor_id]
            descriptor_checks = {
                "caption_route_descriptor_sha256": _canonical_json_sha256(
                    descriptor
                ),
                "caption_route_used_default": (
                    descriptor_id == default_descriptor_id
                ),
                "caption_route_output_box_source": descriptor[
                    "output_box_source"
                ],
            }
            for key, expected in descriptor_checks.items():
                if record.get(key) != expected:
                    raise ValueError(
                        f"formal routed record {index} drifted for {key}"
                    )
            winner_candidate = record.get("winner_candidate_index")
            winner_patch_query = record.get("winner_patch_query_index")
            matched_external = record.get("matched_external_query_index")
            if descriptor_id == default_descriptor_id:
                if winner_candidate is not None or winner_patch_query is not None:
                    raise ValueError(
                        f"formal routed base record {index} must bypass patch Top50"
                    )
            elif (
                type(winner_candidate) is not int
                or not 0 <= winner_candidate < int(settings["candidate_topk"])
                or type(winner_patch_query) is not int
                or winner_patch_query < 0
            ):
                raise ValueError(
                    f"formal routed transfer record {index} winner evidence drifted"
                )
            if (
                type(matched_external) is not int
                or not 0 <= matched_external < _EXTERNAL_GDINO_QUERY_COUNT
            ):
                raise ValueError(
                    f"formal routed record {index} external query drifted"
                )
            selected_box = record.get("selected_box")
            if (
                not isinstance(selected_box, list)
                or len(selected_box) != 4
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0.0
                    or float(value) > 1.0
                    for value in selected_box
                )
            ):
                raise ValueError(
                    f"formal routed record {index} selected box drifted"
                )
            descriptor_counts[descriptor_id] += 1
            if formal_version == 2:
                caption_row = caption_counts.setdefault(
                    route_caption,
                    {"descriptor_id": descriptor_id, "num_expressions": 0},
                )
                if caption_row["descriptor_id"] != descriptor_id:
                    raise ValueError(
                        f"formal routed caption {route_caption!r} changed descriptor"
                    )
            else:
                caption_row = caption_counts.setdefault(
                    route_caption,
                    {"descriptor_counts": {}, "num_expressions": 0},
                )
                per_caption_descriptors = caption_row["descriptor_counts"]
                per_caption_descriptors[descriptor_id] = int(
                    per_caption_descriptors.get(descriptor_id, 0)
                ) + 1
            caption_row["num_expressions"] += 1

            valid = bool(record.get("valid", False))
            if not valid:
                invalid += 1
                continue
            top1_iou = record.get("top1_iou")
            all_query_best_iou = record.get("all_query_best_iou")
            patch_oracle_iou = record.get("patch_candidate_oracle_iou")
            for key, value in (
                ("top1_iou", top1_iou),
                ("all_query_best_iou", all_query_best_iou),
                ("patch_candidate_oracle_iou", patch_oracle_iou),
            ):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or not 0.0 <= float(value) <= 1.0
                ):
                    raise ValueError(
                        f"formal routed record {index} lacks finite {key}"
                    )
            top1_iou = float(top1_iou)
            correct50 += int(top1_iou >= 0.5)
            correct25 += int(top1_iou >= 0.25)
            iou_sum += top1_iou
            patch_oracle50 += int(float(patch_oracle_iou) >= 0.5)
            all_query_oracle50 += int(float(all_query_best_iou) >= 0.5)
            if record.get("correct50") is not bool(top1_iou >= 0.5):
                raise ValueError(
                    f"formal routed record {index} correct50 drifted"
                )
            if record.get("correct25") is not bool(top1_iou >= 0.25):
                raise ValueError(
                    f"formal routed record {index} correct25 drifted"
                )
            if record.get("candidate_oracle_correct50") is not bool(
                float(patch_oracle_iou) >= 0.5
            ):
                raise ValueError(
                    f"formal routed record {index} patch oracle flag drifted"
                )
        if descriptor_counts != row.get("route_counts_by_descriptor"):
            raise ValueError(f"formal routed descriptor counts drifted for {dataset}")
        if caption_counts != row.get("route_counts_by_caption"):
            raise ValueError(f"formal routed caption counts drifted for {dataset}")
        if formal_version == 3:
            if fulltext_gate_counts != row.get("fulltext_route_gate_counts"):
                raise ValueError(
                    f"formal full-text gate counts drifted for {dataset}"
                )
        if invalid != int(row.get("invalid_records", -1)):
            raise ValueError(f"formal routed invalid-record count drifted for {dataset}")
        require_close(row.get("acc50"), correct50 / num_expressions, label=f"{dataset}.acc50")
        require_close(row.get("acc25"), correct25 / num_expressions, label=f"{dataset}.acc25")
        require_close(
            row.get("mean_iou_top1"),
            iou_sum / num_expressions,
            label=f"{dataset}.mean_iou_top1",
        )
        require_close(
            row.get("candidate_oracle_recall50"),
            patch_oracle50 / num_expressions,
            label=f"{dataset}.candidate_oracle_recall50",
        )
        require_close(
            row.get("all_query_oracle_recall50"),
            all_query_oracle50 / num_expressions,
            label=f"{dataset}.all_query_oracle_recall50",
        )
        runtime_contract = (
            {
                "batch_size": int(selection_contract["batch_size"]),
                "num_workers": int(selection_contract["num_workers"]),
                "amp": str(selection_contract["amp"]).startswith("enabled"),
            }
            if formal_version == 2
            else {
                "batch_size": int(settings["evaluation_protocol"]["batch_size"]),
                "num_workers": int(settings["evaluation_protocol"]["num_workers"]),
                "amp": bool(settings["evaluation_protocol"]["amp"]),
            }
        )
        identity_checks = {
            "formal_route_version": formal_version,
            "formal_route_default_descriptor_id": default_descriptor_id,
            "external_transfer_artifact_sha256": artifact_sha256,
            "external_transfer_artifact_file_sha256": settings["artifact"]["sha256"],
            "caption_route_selection_artifact_identity_sha256": (
                selection_identity_sha256
            ),
            "caption_route_policy_sha256": policy_sha256,
            "patch_checkpoint_sha256": patch_checkpoint["sha256"],
            "external_gdino_checkpoint_sha256": settings["external_checkpoint"][
                "sha256"
            ],
            "external_gdino_config_sha256": settings["external_config"]["sha256"],
            "patch_config_sha256": settings["patch_config"]["sha256"],
            "external_gdino_rank_score_key": _EXTERNAL_GDINO_RANK_SCORE_KEY,
            "external_gdino_query_count": _EXTERNAL_GDINO_QUERY_COUNT,
            "transfer_contract_version": formal_version,
            "candidate_topk": int(settings["candidate_topk"]),
            **runtime_contract,
        }
        if formal_version == 3:
            identity_checks.update(
                {
                    "fulltext_route_gate_artifact_identity_sha256": settings[
                        "fulltext_route_gate_artifact_identity"
                    ]["sha256"],
                    "routed_v2_artifact_identity_sha256": settings[
                        "routed_v2_artifact_identity"
                    ]["sha256"],
                    "fulltext_route_policy_sha256": policy_sha256,
                    "canonical_route_policy_sha256": settings[
                        "canonical_route_policy_sha256"
                    ],
                }
            )
        for key, expected in identity_checks.items():
            if row.get(key) != expected:
                raise ValueError(
                    f"formal routed identity drift for {dataset}/{key}"
                )
        checkpoint_path = str(Path(row["checkpoint"]).expanduser().resolve())
        if checkpoint_path != patch_checkpoint["path"]:
            raise ValueError("formal routed patch checkpoint path drifted")
        fixed_rows[dataset] = num_expressions
        fixed_seeds[dataset] = int(row["seed"])
        descriptor_counts_by_split[dataset] = descriptor_counts
        caption_counts_by_split[dataset] = caption_counts
        for descriptor_id, count in descriptor_counts.items():
            aggregate_descriptor_counts[descriptor_id] += count
        for caption, caption_row in caption_counts.items():
            if formal_version == 2:
                aggregate = aggregate_caption_counts.setdefault(
                    caption,
                    {
                        "descriptor_id": caption_row["descriptor_id"],
                        "num_expressions": 0,
                    },
                )
                if aggregate["descriptor_id"] != caption_row["descriptor_id"]:
                    raise ValueError(
                        f"formal routed caption {caption!r} changed across splits"
                    )
            else:
                aggregate = aggregate_caption_counts.setdefault(
                    caption,
                    {"descriptor_counts": {}, "num_expressions": 0},
                )
                for descriptor_id, count in caption_row[
                    "descriptor_counts"
                ].items():
                    aggregate["descriptor_counts"][descriptor_id] = int(
                        aggregate["descriptor_counts"].get(descriptor_id, 0)
                    ) + int(count)
            aggregate["num_expressions"] += caption_row["num_expressions"]
        if formal_version == 3:
            assert aggregate_fulltext_gate_counts is not None
            assert fulltext_gate_counts is not None
            assert fulltext_gate_counts_by_split is not None
            fulltext_gate_counts_by_split[dataset] = dict(fulltext_gate_counts)
            for source, count in fulltext_gate_counts.items():
                aggregate_fulltext_gate_counts[source] += count

    expected_seed_map = settings["evaluation_protocol"]["split_seeds"]
    for dataset, seed in fixed_seeds.items():
        if int(expected_seed_map.get(dataset, -1)) != seed:
            raise ValueError(f"formal routed stable seed drifted for {dataset}")
    metadata = {
        "diagnostic_only": False,
        "formal_gate_eligible": True,
        "evaluation_kind": (
            "formal_external_gdino_canonical_caption_route_v2"
            if formal_version == 2
            else "formal_external_gdino_fulltext_gated_caption_route_v3"
        ),
        "model_forwards_per_batch": {"patch_model": 1, "external_gdino_model": 1},
        "candidate_admission": (
            "caption_routed_mixed_base_full900_or_transfer_exact_stage_a_top50"
            if formal_version == 2
            else "fulltext_gated_caption_routed_mixed_base_full900_or_transfer_exact_stage_a_top50"
        ),
        "candidate_topk": int(settings["candidate_topk"]),
        "caption_provenance": dict(settings["caption_provenance"]),
        "evaluation_protocol": dict(settings["evaluation_protocol"]),
        "fixed_rows": fixed_rows,
        "fixed_seeds": fixed_seeds,
        "route_counts_by_descriptor_by_split": descriptor_counts_by_split,
        "route_counts_by_caption_by_split": caption_counts_by_split,
        "route_counts_by_descriptor": aggregate_descriptor_counts,
        "route_counts_by_caption": dict(sorted(aggregate_caption_counts.items())),
        "route_policy": dict(settings["route_policy"]),
        "route_policy_sha256": policy_sha256,
        "caption_route_selection_artifact": dict(selection_artifact),
        "caption_route_selection_artifact_identity": dict(
            selection["artifact_identity"]
        ),
        "descriptor_registry": registry,
        "patch_config": dict(settings["patch_config"]),
        "patch_checkpoint": dict(patch_checkpoint),
        "external_gdino_config": dict(settings["external_config"]),
        "external_gdino_checkpoint": dict(settings["external_checkpoint"]),
        "external_transfer_artifact": dict(settings["artifact"]),
        "external_transfer_artifact_identity": dict(settings["artifact_identity"]),
    }
    if formal_version == 3:
        metadata.update(
            {
                "fulltext_route_gate_counts": aggregate_fulltext_gate_counts,
                "fulltext_route_gate_counts_by_split": (
                    fulltext_gate_counts_by_split
                ),
                "fulltext_route_gate": dict(settings["fulltext_route_gate"]),
                "fulltext_route_gate_artifact": dict(
                    settings["fulltext_route_gate_artifact"]
                ),
                "fulltext_route_gate_artifact_identity": dict(
                    settings["fulltext_route_gate_artifact_identity"]
                ),
                "routed_v2_artifact": dict(settings["routed_v2_artifact"]),
                "routed_v2_artifact_identity": dict(
                    settings["routed_v2_artifact_identity"]
                ),
                "canonical_route_policy": dict(
                    settings["canonical_route_policy"]
                ),
                "canonical_route_policy_sha256": str(
                    settings["canonical_route_policy_sha256"]
                ),
                "fulltext_route_policy_sha256": policy_sha256,
            }
        )
    return metadata


def _write_summary(
    output_dir: Path,
    results: List[Dict[str, Any]],
    primary_metric: str,
    *,
    diagnostic_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_ids: List[str] = []
    seen = set()
    for row in results:
        run_id = row["run_id"]
        if run_id not in seen:
            seen.add(run_id)
            run_ids.append(run_id)
    datasets: List[str] = []
    seen_ds = set()
    for row in results:
        ds = row["dataset"]
        if ds not in seen_ds:
            seen_ds.add(ds)
            datasets.append(ds)
    ranking = []
    for i, run_id in enumerate(
        sorted(run_ids, key=lambda r: _mean_metric(results, r, primary_metric), reverse=True)
    ):
        rank_row = {
            "rank": i + 1,
            "run_id": run_id,
            f"mean_{primary_metric}": _mean_metric(results, run_id, primary_metric),
            "mean_mean_iou_top1": _mean_metric(results, run_id, "mean_iou_top1"),
        }
        if diagnostic_metadata is not None:
            rank_row["mean_candidate_oracle_recall50"] = _mean_metric(
                results, run_id, "candidate_oracle_recall50"
            )
        ranking.append(rank_row)
    payload = {"primary_metric": primary_metric, "ranking": ranking, "results": results}
    is_diagnostic_summary = bool(
        diagnostic_metadata is not None
        and diagnostic_metadata.get("diagnostic_only", False)
    )
    if diagnostic_metadata is not None:
        payload.update(diagnostic_metadata)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    by_run_ds = {(r["run_id"], r["dataset"]): r for r in results}
    lines = ["# RefCOCO Stage-B Evaluation", ""]
    if is_diagnostic_summary:
        lines.extend(
            [
                "**Diagnostic only; `formal_gate_eligible=false`.**",
                "",
            ]
        )
    lines.extend(
        [
            f"Primary metric: mean `{primary_metric}` across evaluated splits.",
            "",
        ]
    )
    if diagnostic_metadata is not None:
        lines.extend(
            [
                "| rank | run | mean acc50 | mean IoU@1 | mean candidate oracle@50 | "
                + " | ".join(f"{ds} acc50" for ds in datasets)
                + " |",
                "|---:|---|---:|---:|---:|"
                + "|".join("---:" for _ in datasets)
                + "|",
            ]
        )
    else:
        lines.extend(
            [
                "| rank | run | mean acc50 | mean IoU@1 | "
                + " | ".join(f"{ds} acc50" for ds in datasets)
                + " |",
                "|---:|---|---:|---:|"
                + "|".join("---:" for _ in datasets)
                + "|",
            ]
        )
    for row in ranking:
        run_id = row["run_id"]
        ds_vals = [f"{float(by_run_ds.get((run_id, ds), {}).get('acc50', 0.0)):.6f}" for ds in datasets]
        oracle_column = ""
        if diagnostic_metadata is not None:
            oracle_column = (
                f"{float(row['mean_candidate_oracle_recall50']):.6f} | "
            )
        lines.append(
            f"| {row['rank']} | `{run_id}` | "
            f"{float(row[f'mean_{primary_metric}']):.6f} | "
            f"{float(row['mean_mean_iou_top1']):.6f} | "
            + oracle_column
            + " | ".join(ds_vals)
            + " |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Stage-B checkpoints on standard RefCOCO splits.")
    parser.add_argument("--config", default="config/cfg_patch_stage_b.py")
    parser.add_argument("--ckpts", nargs="+", required=True)
    parser.add_argument("--output_dir", default="outputs/refcoco_stageb_eval")
    parser.add_argument("--data_root", default=os.environ.get("DATA_ROOT", "/media/haoyi/T9/data"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--betas", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument("--topk", nargs="+", type=int, default=[1, 5, 10])
    parser.add_argument("--splits", nargs="*", default=["refcoco_val", "refcocop_val", "refcocog_val"])
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--max_images", type=int, default=0, help="Maximum expression rows per split; 0 means full split.")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--primary_metric", default="acc50")
    parser.add_argument("--stage_b_v7_candidate_topk", type=int, default=None)
    parser.add_argument("--stage_b_v11_candidate_topk", type=int, default=None)
    parser.add_argument("--stage_b_v7_patch_prior_weight", type=float, default=None)
    parser.add_argument("--stage_b_v7_phrase_agg", default=None)
    parser.add_argument("--stage_b_v7_phrase_mean_weight", type=float, default=None)
    parser.add_argument("--stage_b_v7_phrase_softmin_tau", type=float, default=None)
    parser.add_argument(
        "--diagnostic_patch_rank_weights",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Diagnostic-only fixed-scorer patch-rank weight grid. Reuses each "
            "forward and the unchanged Stage-A Top50 candidates; never produces "
            "formal-gate-eligible output."
        ),
    )
    parser.add_argument(
        "--diagnostic_external_gdino_config",
        default=None,
        help=(
            "Diagnostic-only independent pure-GDINO adapter config used to "
            "transfer full-expression rank scores onto fixed patch candidates."
        ),
    )
    parser.add_argument(
        "--diagnostic_external_gdino_checkpoint",
        default=None,
        help="Independent pure-GDINO adapter checkpoint for rank transfer.",
    )
    parser.add_argument(
        "--diagnostic_external_transfer_modes",
        nargs="+",
        choices=list(_EXTERNAL_RANK_TRANSFER_MODES),
        default=None,
        help="Explicit fixed rank-transfer mode grid.",
    )
    parser.add_argument(
        "--diagnostic_external_iou_powers",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Explicit positive p grid for max_score_iou_power. Omit only when "
            "that transfer mode is not selected."
        ),
    )
    parser.add_argument(
        "--diagnostic_external_patch_weights",
        nargs="+",
        type=float,
        default=None,
        help="Explicit patch-logit fusion-weight grid.",
    )
    parser.add_argument(
        "--diagnostic_external_text_weights",
        nargs="+",
        type=float,
        default=None,
        help="Explicit transferred full-text rank-score fusion-weight grid.",
    )
    parser.add_argument(
        "--diagnostic_external_include_patch_internal_rank_identity",
        action="store_true",
        help=(
            "Append one diagnostic-only patch-model identity descriptor beside "
            "the external transfer grid. It is the exact standard Ref beta=0 "
            "winner on the unchanged fixed Top50, uses the patch box, has no "
            "external fusion weights, and is never formal-gate eligible."
        ),
    )
    parser.add_argument(
        "--diagnostic_external_include_gdino_base_identity",
        action="store_true",
        help=(
            "Append one diagnostic-only direct ordinary-GDINO identity "
            "descriptor beside the transfer grid. It takes the first global "
            "argmax of stage_b_gdino_base_score across all 900 external "
            "queries and emits that external query box; patch Top50, adapter "
            "residuals, fusion weights, and formal gates are not used."
        ),
    )
    parser.add_argument(
        "--formal_external_rank_transfer_artifact",
        default=None,
        help=(
            "Single versioned formal transfer artifact. Formal mode takes no "
            "independent external config/checkpoint or sweep arguments."
        ),
    )
    parser.add_argument("--exclude_train_jsonl", nargs="*", default=[])
    parser.add_argument("--holdout_level", choices=["none", "ann", "image"], default="none")
    parser.add_argument(
        "--no_per_example_records",
        action="store_true",
        help="Disable canonical *.records.jsonl output used by the paired final gate.",
    )
    parser.add_argument(
        "--data_driven_query_diagnostics",
        action="store_true",
        help=(
            "Add query indices, Gap3 eligibility, component scores, and gate "
            "oracle fields to data-driven per-example records. Diagnostic "
            "metadata only; scores and model routing are unchanged."
        ),
    )
    args = parser.parse_args()

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    device = torch.device(args.device)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)

    cfg = SLConfig.fromfile(args.config)
    native_patch_category = _validate_native_patch_category_ref_request(
        cfg, args.ckpts
    )
    if args.data_driven_query_diagnostics:
        if args.no_per_example_records:
            raise ValueError(
                "data-driven query diagnostics require per-example records"
            )
        if not (
            bool(getattr(cfg, "stage_b_data_driven_score", False))
            and bool(
                getattr(cfg, "stage_b_data_driven_category_gate", False)
            )
        ):
            raise ValueError(
                "data-driven query diagnostics require a data-driven Gap3 config"
            )
    cfg.device = str(device)
    cfg.patch_only = not bool(
        getattr(cfg, "stage_b_gdino_score_adapter", False)
        or getattr(cfg, "stage_b_u0_patch_rank", False)
        or getattr(cfg, "stage_b_data_driven_score", False)
        or native_patch_category
    )
    cfg.build_text_token_masks = True
    cfg.use_coco_eval = False
    # Evaluation runs without gradients, so checkpointing only increases
    # allocator pressure and can trigger large transient CUDA allocations.
    cfg.use_checkpoint = False
    cfg.use_transformer_ckpt = False
    cfg.batch_size = int(args.batch_size)
    cfg.text_mask_warn_limit = 0
    for key in (
        "stage_b_v7_candidate_topk",
        "stage_b_v11_candidate_topk",
        "stage_b_v7_patch_prior_weight",
        "stage_b_v7_phrase_agg",
        "stage_b_v7_phrase_mean_weight",
        "stage_b_v7_phrase_softmin_tau",
    ):
        value = getattr(args, key)
        if value is not None:
            setattr(cfg, key, value)
    diagnostic_settings = _diagnostic_patch_rank_settings(
        args.diagnostic_patch_rank_weights,
        cfg,
    )
    external_option_values = (
        args.diagnostic_external_gdino_config,
        args.diagnostic_external_gdino_checkpoint,
        args.diagnostic_external_transfer_modes,
        args.diagnostic_external_iou_powers,
        args.diagnostic_external_patch_weights,
        args.diagnostic_external_text_weights,
        (
            True
            if args.diagnostic_external_include_patch_internal_rank_identity
            else None
        ),
        (
            True
            if args.diagnostic_external_include_gdino_base_identity
            else None
        ),
    )
    external_diagnostic_requested = _diagnostic_external_rank_transfer_requested(
        *external_option_values
    )
    formal_external_requested = args.formal_external_rank_transfer_artifact is not None
    if native_patch_category:
        _validate_native_patch_category_ref_request(
            cfg,
            args.ckpts,
            extra_score_source_requested=bool(
                diagnostic_settings is not None
                or external_diagnostic_requested
                or formal_external_requested
            ),
        )
    if formal_external_requested:
        _validate_formal_cli_contract(args)
    if diagnostic_settings is not None and external_diagnostic_requested:
        raise ValueError(
            "--diagnostic_patch_rank_weights cannot be combined with external-GDINO rank transfer"
        )
    if formal_external_requested and (
        diagnostic_settings is not None or external_diagnostic_requested
    ):
        raise ValueError(
            "formal external rank transfer accepts only its artifact and cannot be "
            "combined with diagnostic grids"
        )
    runtime_overrides = {
        key: getattr(args, key)
        for key in (
            "stage_b_v7_candidate_topk",
            "stage_b_v11_candidate_topk",
            "stage_b_v7_patch_prior_weight",
            "stage_b_v7_phrase_agg",
            "stage_b_v7_phrase_mean_weight",
            "stage_b_v7_phrase_softmin_tau",
        )
        if getattr(args, key) is not None
    }
    if formal_external_requested and runtime_overrides:
        raise ValueError(
            "formal external rank transfer forbids runtime scorer overrides: "
            f"{sorted(runtime_overrides)}"
        )
    external_cfg = None
    external_diagnostic_settings = None
    formal_external_settings = None
    if external_diagnostic_requested:
        if args.diagnostic_external_gdino_config is None:
            raise ValueError(
                "external-GDINO rank transfer requires --diagnostic_external_gdino_config"
            )
        external_cfg = SLConfig.fromfile(args.diagnostic_external_gdino_config)
        external_cfg.device = str(device)
        external_cfg.patch_only = False
        external_cfg.build_text_token_masks = True
        external_cfg.use_coco_eval = False
        external_cfg.use_checkpoint = False
        external_cfg.use_transformer_ckpt = False
        external_cfg.batch_size = int(args.batch_size)
        external_cfg.text_mask_warn_limit = 0
        external_diagnostic_settings = (
            _diagnostic_external_rank_transfer_settings(
                external_config_path=args.diagnostic_external_gdino_config,
                external_checkpoint_path=args.diagnostic_external_gdino_checkpoint,
                transfer_modes=args.diagnostic_external_transfer_modes,
                iou_powers=args.diagnostic_external_iou_powers,
                patch_weights=args.diagnostic_external_patch_weights,
                text_weights=args.diagnostic_external_text_weights,
                patch_cfg=cfg,
                external_cfg=external_cfg,
                patch_config_path=args.config,
                patch_checkpoint_paths=args.ckpts,
                include_patch_internal_rank_identity=(
                    args.diagnostic_external_include_patch_internal_rank_identity
                ),
                include_external_gdino_base_identity=(
                    args.diagnostic_external_include_gdino_base_identity
                ),
            )
        )
    elif formal_external_requested:
        formal_external_settings = evaluator_settings_from_artifact(
            args.formal_external_rank_transfer_artifact
        )
        if len(args.ckpts) != 1:
            raise ValueError(
                "formal external rank transfer requires exactly one patch checkpoint"
            )
        artifact_patch_config = formal_external_settings["patch_config"]
        artifact_patch_checkpoint = formal_external_settings["patch_checkpoints"][0]
        if str(Path(args.config).expanduser().resolve()) != artifact_patch_config["path"]:
            raise ValueError("--config does not match the formal transfer artifact")
        if (
            str(Path(args.ckpts[0]).expanduser().resolve())
            != artifact_patch_checkpoint["path"]
        ):
            raise ValueError("--ckpts does not match the formal transfer artifact")
        protocol = formal_external_settings["evaluation_protocol"]
        if (
            int(protocol["base_seed"]) != int(args.seed)
            or protocol["seed_protocol"] != SPLIT_SEED_PROTOCOL
            or protocol["split_seeds"] != stable_ref_split_seed_map(int(args.seed))
        ):
            raise ValueError("formal transfer artifact stable seed protocol drifted")
        cfg = _load_bound_config(
            artifact_patch_config, label="formal patch config"
        )
        cfg.device = str(device)
        cfg.patch_only = not bool(
            getattr(cfg, "stage_b_gdino_score_adapter", False)
            or getattr(cfg, "stage_b_u0_patch_rank", False)
            or getattr(cfg, "stage_b_data_driven_score", False)
        )
        cfg.build_text_token_masks = True
        cfg.use_coco_eval = False
        cfg.use_checkpoint = False
        cfg.use_transformer_ckpt = False
        cfg.batch_size = int(args.batch_size)
        cfg.text_mask_warn_limit = 0
        external_cfg = _load_bound_config(
            formal_external_settings["external_config"],
            label="formal external config",
        )
        if (
            not bool(getattr(external_cfg, "stage_b_gdino_score_adapter", False))
            or not bool(
                getattr(external_cfg, "stage_b_gdino_adapter_merged_eval_only", False)
            )
            or int(
                getattr(
                    external_cfg,
                    "stage_b_gdino_adapter_merged_eval_contract_version",
                    0,
                )
            )
            != 1
            or int(getattr(external_cfg, "num_queries", 0)) != 900
            or bool(getattr(external_cfg, "stage_b_v11_fixed_text", False))
            or bool(getattr(external_cfg, "stage_b_v7", False))
        ):
            raise ValueError("formal artifact external config lost its merged-eval contract")
        external_cfg.device = str(device)
        external_cfg.patch_only = False
        external_cfg.build_text_token_masks = True
        external_cfg.use_coco_eval = False
        external_cfg.use_checkpoint = False
        external_cfg.use_transformer_ckpt = False
        external_cfg.batch_size = int(args.batch_size)
        external_cfg.text_mask_warn_limit = 0

    canonical_json = data_root / "canonical_classes_with_aliases.json"
    if (
        formal_external_settings is not None
        and int(formal_external_settings.get("formal_artifact_version", 1)) in (2, 3)
    ):
        formal_version = int(
            formal_external_settings.get("formal_artifact_version", 1)
        )
        route_selection_key = (
            "route_selection"
            if formal_version == 2
            else "canonical_route_selection"
        )
        canonical_binding = formal_external_settings[route_selection_key][
            "canonical_classes"
        ]
        _require_bound_file(
            canonical_binding,
            label="formal caption route canonical classes",
        )
        if canonical_json.expanduser().resolve() != Path(
            canonical_binding["path"]
        ).expanduser().resolve():
            raise ValueError(
                "formal caption route data_root canonical classes do not match "
                "the frozen selection artifact"
            )
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
            f"ann_keys={len(holdout_ann_keys)} image_ids={len(holdout_image_ids)}"
        )
    requested_splits = _requested_ref_split_specs(args.splits or [], int(args.seed))

    datasetinfos = []
    for name, spec, split_seed in requested_splits:
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
        print(f"[INFO] built {name}: {count} expressions -> {jsonl_path}")
        datasetinfos.append(
            (
                name,
                _make_datasetinfo(
                    data_root,
                    name,
                    jsonl_path,
                    adapter_no_support=_adapter_ref_eval_uses_no_support(cfg),
                ),
                split_seed,
            )
        )

    results: List[Dict[str, Any]] = []
    external_gdino_model = None
    active_external_settings = (
        formal_external_settings
        if formal_external_settings is not None
        else external_diagnostic_settings
    )
    if active_external_settings is not None:
        print(
            "[INFO] loading independent external GDINO adapter checkpoint: "
            f"{active_external_settings['external_checkpoint']['path']}"
        )
        _set_seed(int(args.seed))
        if formal_external_settings is not None:
            external_gdino_model = _load_bound_model(
                external_cfg,
                active_external_settings["external_checkpoint"],
                active_external_settings["external_config"],
                device,
                label="formal external GDINO",
            )
        else:
            external_gdino_model = _load_model(
                external_cfg,
                active_external_settings["external_checkpoint"]["path"],
                device,
            )
    for ckpt_i, ckpt_path in enumerate(args.ckpts):
        ckpt_path = str(ckpt_path)
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(ckpt_path)
        print(f"[INFO] loading checkpoint {ckpt_i + 1}/{len(args.ckpts)}: {ckpt_path}")
        _set_seed(int(args.seed))
        if formal_external_settings is not None:
            model = _load_bound_model(
                cfg,
                formal_external_settings["patch_checkpoints"][0],
                formal_external_settings["patch_config"],
                device,
                label="formal patch model",
            )
        else:
            model = _load_model(cfg, ckpt_path, device)
        for name, datasetinfo, split_seed in datasetinfos:
            rows = evaluate_dataset(
                cfg=cfg,
                model=model,
                ckpt_path=ckpt_path,
                datasetinfo=datasetinfo,
                dataset_name=name,
                device=device,
                betas=list(args.betas),
                topks=list(args.topk),
                batch_size=int(args.batch_size),
                num_workers=int(args.num_workers),
                seed=int(split_seed),
                amp=bool(args.amp),
                max_batches=int(args.max_batches),
                max_images=int(args.max_images),
                log_every=int(args.log_every),
                records_output_dir=(
                    None
                    if (
                        args.no_per_example_records
                        or diagnostic_settings is not None
                        or external_diagnostic_settings is not None
                    )
                    else output_dir / "per_example_records"
                ),
                data_driven_query_diagnostics=bool(
                    args.data_driven_query_diagnostics
                ),
                diagnostic_patch_rank_weights=(
                    None
                    if diagnostic_settings is None
                    else list(diagnostic_settings["weights"])
                ),
                diagnostic_patch_rank_contract_weight=(
                    None
                    if diagnostic_settings is None
                    else float(diagnostic_settings["contract_weight"])
                ),
                diagnostic_patch_rank_candidate_topk=(
                    None
                    if diagnostic_settings is None
                    else int(diagnostic_settings["candidate_topk"])
                ),
                diagnostic_external_rank_transfer_settings=(
                    external_diagnostic_settings
                ),
                formal_external_rank_transfer_settings=formal_external_settings,
                external_gdino_model=external_gdino_model,
                external_gdino_cfg=external_cfg,
            )
            results.extend(rows)
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / f"{_ckpt_run_prefix(ckpt_path)}__{name}.json").write_text(
                json.dumps(rows, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            if external_diagnostic_settings is not None:
                diagnostic_metadata = (
                    _diagnostic_external_rank_transfer_summary_metadata(
                        external_diagnostic_settings,
                        results,
                    )
                )
            elif diagnostic_settings is not None:
                diagnostic_metadata = _diagnostic_summary_metadata(
                    diagnostic_settings, results
                )
            else:
                diagnostic_metadata = None
            if formal_external_settings is None:
                _write_summary(
                    output_dir,
                    results,
                    str(args.primary_metric),
                    diagnostic_metadata=diagnostic_metadata,
                )
            for row in rows:
                if formal_external_settings is not None:
                    print(
                        f"[FORMAL] {row['run_id']} {name}: "
                        f"acc50={row['acc50']:.6f} "
                        f"mean_iou@1={row['mean_iou_top1']:.6f} "
                        f"candidate_oracle_recall50="
                        f"{row['candidate_oracle_recall50']:.6f}"
                    )
                elif external_diagnostic_settings is not None:
                    print(
                        f"[DIAGNOSTIC] {row['run_id']} {name}: "
                        f"acc50={row['acc50']:.6f} "
                        f"mean_iou@1={row['mean_iou_top1']:.6f} "
                        f"candidate_oracle_recall50="
                        f"{row['candidate_oracle_recall50']:.6f}"
                    )
                elif diagnostic_settings is not None:
                    print(
                        f"[DIAGNOSTIC] {row['run_id']} {name}: "
                        f"acc50={row['acc50']:.6f} "
                        f"mean_iou@1={row['mean_iou_top1']:.6f} "
                        f"candidate_oracle_recall50="
                        f"{row['candidate_oracle_recall50']:.6f}"
                    )
                else:
                    print(
                        f"[RESULT] {row['run_id']} {name}: "
                        f"acc50={row['acc50']:.6f} mean_iou@1={row['mean_iou_top1']:.6f} "
                        f"recall50@5={row.get('recall50@5', 0.0):.6f}"
                    )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if formal_external_settings is not None:
        _reverify_formal_runtime_settings(formal_external_settings)
        formal_version = int(
            formal_external_settings.get("formal_artifact_version", 1)
        )
        if formal_version in (2, 3):
            diagnostic_metadata = (
                _formal_routed_external_rank_transfer_summary_metadata(
                    formal_external_settings,
                    results,
                )
            )
        else:
            diagnostic_metadata = _formal_external_rank_transfer_summary_metadata(
                formal_external_settings,
                results,
            )
    elif external_diagnostic_settings is not None:
        diagnostic_metadata = _diagnostic_external_rank_transfer_summary_metadata(
            external_diagnostic_settings,
            results,
        )
    elif diagnostic_settings is not None:
        diagnostic_metadata = _diagnostic_summary_metadata(
            diagnostic_settings,
            results,
        )
    else:
        diagnostic_metadata = None
    _write_summary(
        output_dir,
        results,
        str(args.primary_metric),
        diagnostic_metadata=diagnostic_metadata,
    )
    print(f"[INFO] wrote {output_dir / 'summary.json'}")
    print(f"[INFO] wrote {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
