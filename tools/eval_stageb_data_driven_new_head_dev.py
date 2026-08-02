#!/usr/bin/env python3
"""Evaluate teacher-free DD0/DD1 rank heads on the sealed internal dev split.

This evaluator deliberately has a narrow contract: one DD0 or DD1 checkpoint,
one of the two sealed dev partitions, three Ref training sources, all 900
queries, and the text-rank score with the category gate disabled.  It also
provides a paired, global-image-cluster bootstrap for two completed artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from groundingdino.util import box_ops  # noqa: E402
from models.GroundingDINO.stage_b_data_driven_score import (  # noqa: E402
    data_driven_tensor_state_sha256,
)
from tools.eval_refcoco_stageb import (  # noqa: E402
    _build_loader,
    _forward,
    _load_model,
    _make_datasetinfo,
    _torch_load_compat,
)
from util.slconfig import SLConfig  # noqa: E402


PARTITION_RECEIPT_SCHEMA = (
    "pivot.stageb.data_driven.new_head_partition_receipt/v1"
)
EVALUATION_SUMMARY_SCHEMA = (
    "pivot.stageb.data_driven.new_head_dev_evaluation/v1"
)
EXPRESSION_RECORD_SCHEMA = (
    "pivot.stageb.data_driven.new_head_dev_expression/v1"
)
BOOTSTRAP_SCHEMA = (
    "pivot.stageb.data_driven.new_head_dev_paired_bootstrap/v1"
)
EVALUATION_SCOPE = "new_head_only_common_frozen_b58_not_end_to_end_unseen_v1"
FORMAL_NEW_HEAD_EXECUTION_SCOPE = "formal_fresh_a0_new_head_3epoch_v1"
FORMAL_NEW_HEAD_CONTRACT = "sealed_new_head_d0_d1_3epoch_v1"
FORMAL_NEW_HEAD_OPTIMIZER_UPDATES = 12_357
FORMAL_NEW_HEAD_BINDING_SCHEMA = (
    "pivot.stageb.data_driven.new_head_formal_runtime_binding/v1"
)
FORMAL_NEW_HEAD_TRAIN_ROWS = 263_661
FORMAL_NEW_HEAD_LR_SELECTION_SCHEMA = (
    "pivot.stageb.data_driven.new_head_lr_selection_receipt/v1"
)
FORMAL_NEW_HEAD_LR_CANDIDATES = [3e-5, 1e-4, 3e-4]
TRAINING_PARTITION_SCHEMA = PARTITION_RECEIPT_SCHEMA
SUPPORT_RECEIPT_SCHEMA = "pivot.stageb.data_driven.support_partition_receipt/v1"
DEFAULT_PARTITION_RECEIPT = (
    REPO_ROOT
    / "data/ablations/stageb_data_driven_new_head_partition_20260723/receipt.json"
)
EXPECTED_PARTITION_RECEIPT_SHA256 = (
    "56de31d883ed137f3f9332c34de846839d82c0724120f42a49c5c1c302f38506"
)
DEFAULT_SUPPORT_RECEIPT = (
    REPO_ROOT
    / "data/ablations/stageb_data_driven_support_partition_20260723/receipt.json"
)
EXPECTED_SUPPORT_RECEIPT_SHA256 = (
    "a0e6632182bc7c01ac6e6997b15f1f96e0fbb0bf6dd9d1e3fd8485ad39a6da62"
)
SOURCE_MANIFESTS = (
    ("refcoco_stageb_phrase_v1.jsonl", "refcoco"),
    ("refcocoplus_stageb_phrase_v1.jsonl", "refcocoplus"),
    ("refcocog_stageb_phrase_v1.jsonl", "refcocog"),
)
SOURCE_NAMES = tuple(source for _manifest, source in SOURCE_MANIFESTS)
MANIFEST_NAMES = tuple(manifest for manifest, _source in SOURCE_MANIFESTS)
VARIANTS = ("d0_ordinary_primary", "d1_category_complete")
EVALUATION_DATA_VARIANT = VARIANTS[0]
PARTITIONS = ("train", "dev_full", "dev_screen", "quarantine")
DEV_PARTITIONS = ("dev_screen", "dev_full")
IDENTITY_KEYS = (
    "source",
    "image_id",
    "ann_id",
    "ref_id",
    "sent_id",
    "split",
    "filename",
)
TEACHER_ROUTE_FLAGS = (
    "stage_b",
    "stage_b_gdino_score_adapter",
    "stage_b_u0_patch_rank",
    "stage_b_v7",
    "stage_b_v11_fixed_text",
    "stage_b_legacy_global_gate",
)
INITIALIZER_PAIR_REQUIRED_INVARIANTS = (
    "b58_is_only_tensor_checkpoint_source",
    "all_common_non_rank_non_contract_tensors_bitwise_equal",
    "patch_and_confidence_initialization_bitwise_equal",
    "rank_subtree_is_the_only_parameterized_architecture_intervention",
    "no_teacher_u1000_u5020_or_old_initializer_tensor_source",
)
PARTITION_RECEIPT_REQUIRED_INVARIANTS = (
    "D0_and_D1_partition_identity_streams_and_counts_match",
    "D0_and_D1_partition_row_counts_match",
    "D0_and_D1_primary_instances_match",
    "D0_and_D1_rows_share_identity_filename_image_id_and_order",
    "all_eight_official_ref_manifests_match_contract",
    "all_rows_for_one_global_image_share_the_same_main_partition",
    "all_six_source_manifests_match_preregistered_sha256",
    "category_complete_receipt_matches_preregistered_sha256",
    "dev_full_has_exactly_requested_images",
    "dev_screen_has_exactly_requested_images",
    "dev_screen_is_nested_in_dev_full",
    "dev_sets_are_disjoint_from_official_ref8",
    "official_ref8_images_are_quarantined_before_selection",
    "outputs_are_raw_byte_for_byte_input_subsequences",
    "pair_receipt_matches_preregistered_sha256",
    "paired_streams_replay_the_upstream_receipt",
    "selection_is_deterministic_model_score_free_and_hash_bound",
    "train_dev_full_and_quarantine_images_are_pairwise_disjoint",
)
SUPPORT_RECEIPT_REQUIRED_INVARIANTS = (
    "D0_and_D1_use_the_identical_runtime_support_tsv",
    "alias_bridges_are_unique_canonical_metadata_matches",
    "alias_bridges_reuse_only_filtered_base_paths",
    "all_d0_d1_training_classes_have_runtime_support",
    "all_preregistered_inputs_match_sha256",
    "audit_output_is_raw_header_plus_original_row_subsequence",
    "audit_rows_exactly_cover_filtered_base_cache",
    "cache_reads_and_writes_must_be_disabled_for_formal_training",
    "d1_train_manifests_replay_partition_receipt",
    "each_runtime_class_has_at_most_200_candidates",
    "every_cache_candidate_has_one_raw_clean_row",
    "every_cached_vg_image_id_has_official_metadata",
    "no_runtime_candidate_has_an_excluded_coco_id",
    "official_ref8_manifests_replay_partition_receipt",
    "partition_receipt_canonical_payload_replays",
    "runtime_base_stream_is_sealed_cache_order_delete_only",
    "runtime_paths_are_absolute_existing_clean_mirror_jpegs",
    "runtime_row_delta_equals_explicit_alias_bridge_rows",
    "runtime_rows_have_integer_canonical_class_ids",
    "vg_null_coco_ids_are_retained",
    "vg_zip_member_equals_standalone_json_byte_for_byte",
)
PAIRED_TRAINING_EQUAL_KEYS = (
    "stage_b_data_driven_train_mode",
    "stage_b_data_driven_rank_architecture",
    "stage_b_data_driven_rank_dim",
    "stage_b_data_driven_rank_num_heads",
    "stage_b_data_driven_rank_image_level_policy",
    "stage_b_data_driven_rank_image_levels",
    "stage_b_data_driven_rank_image_pool_size",
    "stage_b_data_driven_rank_image_pool_policy",
    "stage_b_data_driven_rank_box_fourier_bands",
    "stage_b_data_driven_rank_ffn_dim",
    "stage_b_data_driven_rank_dropout",
    "stage_b_data_driven_head_init_seed",
    "stage_b_data_driven_rank_supervision",
    "stage_b_data_driven_rank_negative_iou_threshold",
    "stage_b_data_driven_rank_weight",
    "stage_b_data_driven_patch_weight",
    "stage_b_data_driven_positive_iou_threshold",
    "stage_b_data_driven_patch_negative_iou_threshold",
    "stage_b_data_driven_temperature",
    "stage_b_data_driven_category_margin",
    "stage_b_data_driven_rank_lr",
    "stage_b_data_driven_patch_lr",
    "stage_b_data_driven_sampling_contract",
    "stage_b_data_driven_sampler_seed",
    "stage_b_data_driven_loader_seed",
    "stage_b_data_driven_grad_clip_contract",
    "batch_size",
    "epochs",
    "max_train_iters",
    "gradient_accumulation_steps",
    "weight_decay",
    "clip_max_norm",
    "lr_drop",
    "onecyclelr",
    "multi_step_lr",
    "lr_drop_list",
    "num_workers",
    "prefetch_factor",
    "pin_memory",
    "persistent_workers",
    "stage_b_data_driven_required_allocator_env",
    "stage_b_data_driven_required_allocator_conf",
    "stage_b_data_driven_execution_scope",
    "stage_b_data_driven_formal_fresh_start",
    "stage_b_data_driven_formal_expected_optimizer_updates",
    "stage_b_data_driven_new_head_formal_contract",
    "seed",
    "amp",
    "fix_size",
    "strong_aug",
    "data_aug_hflip_prob",
    "aux_loss",
    "use_checkpoint",
    "use_transformer_ckpt",
)
EVALUATION_CONFIG_CHECKPOINT_EQUAL_KEYS = tuple(
    key
    for key in PAIRED_TRAINING_EQUAL_KEYS
    if key
    not in {
        "max_train_iters",
        "gradient_accumulation_steps",
        "num_workers",
        "prefetch_factor",
        "pin_memory",
        "seed",
        "amp",
    }
)
SHARED_TRAINING_PROVENANCE_KEYS = (
    "schema",
    "code_files",
    "software",
    "required_allocator",
    "allocator_environment",
    "support_patch_pool_content",
)
ABSOLUTE_INITIALIZER_ROLE_NAMES = (
    "b58_base",
    "shared_backbone_alias",
    "random_patch_projection",
    "random_absolute_heads",
)
PATCH_PROJECTION_KEYS = (
    "patch_logit_scale",
    "patch_encoder.input_proj.0.weight",
    "patch_encoder.input_proj.0.bias",
    "patch_encoder.input_proj.1.weight",
    "patch_encoder.input_proj.1.bias",
    "patch_encoder.norm.weight",
    "patch_encoder.norm.bias",
    "query_proj_for_patch.weight",
    "query_proj_for_patch.bias",
)
SCORE_CONTRACT_BUFFER_KEY = (
    "stage_b_data_driven_score_heads._contract_version"
)
QUERY_COUNT = 900
DEFAULT_BOOTSTRAP_SEED = 20260723
DEFAULT_BOOTSTRAP_ITERATIONS = 10_000
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_COCO_FILENAME_RE = re.compile(
    r"^COCO_(?P<split>train2014|val2014)_(?P<image_id>[0-9]{12})"
    r"\.(?:jpg|jpeg)$",
    re.IGNORECASE,
)


class NewHeadDevEvalError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ManifestRow:
    identity: tuple[Any, ...]
    identity_object: dict[str, Any]
    coco_split: str
    image_id: int


@dataclass(frozen=True, slots=True)
class BoundManifest:
    name: str
    source: str
    path: Path
    record: dict[str, Any]
    rows: tuple[ManifestRow, ...]


@dataclass(frozen=True, slots=True)
class PartitionBinding:
    receipt_path: Path
    receipt_file: dict[str, Any]
    canonical_payload_sha256: str
    variant: str
    partition: str
    manifests: tuple[BoundManifest, ...]


@dataclass(frozen=True, slots=True)
class SupportBinding:
    receipt_path: Path
    receipt_file: dict[str, Any]
    canonical_payload_sha256: str
    runtime_tsv: dict[str, Any]
    canonical_classes: dict[str, Any]
    support_patch_pool_content: dict[str, Any]


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeEncodeError) as error:
        raise NewHeadDevEvalError("value is not canonical JSON") from error


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    if path.is_symlink() or not path.is_file():
        raise NewHeadDevEvalError(f"bound path must be one regular file: {path}")
    before = path.stat()
    digest = _sha256_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (
        after.st_size,
        after.st_mtime_ns,
    ):
        raise NewHeadDevEvalError(f"file changed while hashing: {path}")
    return {
        "path": str(path),
        "size_bytes": int(before.st_size),
        "sha256": digest,
    }


def _validate_required_invariants(
    value: Any, *, required: Sequence[str], label: str
) -> dict[str, bool]:
    if not isinstance(value, Mapping) or not value:
        raise NewHeadDevEvalError(f"{label} invariants are missing")
    missing = [key for key in required if key not in value]
    false_or_invalid = [key for key, item in value.items() if item is not True]
    if missing or false_or_invalid:
        raise NewHeadDevEvalError(
            f"{label} invariants drifted: "
            f"missing={missing}, false_or_invalid={false_or_invalid}"
        )
    return {key: True for key in required}


def _direct_support_pool_content_record(tsv_path: Path) -> dict[str, Any]:
    """Replay main.py's direct-TSV support content provenance exactly."""
    try:
        tsv_path = tsv_path.expanduser().resolve(strict=True)
    except OSError as error:
        raise NewHeadDevEvalError(
            f"could not resolve direct support TSV: {tsv_path}: {error}"
        ) from error
    bank: dict[int, list[str]] = {}
    try:
        with tsv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if not reader.fieldnames or not {"class_id", "path"}.issubset(
                reader.fieldnames
            ):
                raise NewHeadDevEvalError(
                    f"direct support TSV has no class_id/path columns: {tsv_path}"
                )
            for row_index, row in enumerate(reader, start=2):
                try:
                    class_id = int(row["class_id"])
                except (KeyError, TypeError, ValueError) as error:
                    raise NewHeadDevEvalError(
                        f"direct support TSV row {row_index} has invalid class_id"
                    ) from error
                raw_path = str(row.get("path", "") or "").strip()
                if not raw_path:
                    raise NewHeadDevEvalError(
                        f"direct support TSV row {row_index} has no image path"
                    )
                image_path = Path(raw_path)
                if not image_path.is_absolute():
                    image_path = tsv_path.parent / image_path
                try:
                    resolved = image_path.expanduser().resolve(strict=True)
                except OSError as error:
                    raise NewHeadDevEvalError(
                        "direct support TSV row "
                        f"{row_index} image could not be resolved: {image_path}"
                    ) from error
                bank.setdefault(class_id, []).append(str(resolved))
    except (OSError, UnicodeError, csv.Error) as error:
        raise NewHeadDevEvalError(
            f"could not read direct support TSV {tsv_path}: {error}"
        ) from error

    digest = hashlib.sha256()
    file_count = 0
    total_size = 0
    for class_id, candidates in sorted(bank.items()):
        if not candidates:
            raise NewHeadDevEvalError(f"direct support class {class_id} is empty")
        for candidate_index, raw_path in enumerate(candidates):
            image_path = Path(raw_path).resolve(strict=True)
            try:
                image_sha = _sha256_file(image_path)
                stat = image_path.stat()
            except OSError as error:
                raise NewHeadDevEvalError(
                    f"could not hash direct support image: {image_path}: {error}"
                ) from error
            header = json.dumps(
                [
                    class_id,
                    candidate_index,
                    str(image_path),
                    int(stat.st_size),
                    image_sha,
                ],
                ensure_ascii=True,
                separators=(",", ":"),
            ).encode("ascii")
            digest.update(len(header).to_bytes(8, "little"))
            digest.update(header)
            file_count += 1
            total_size += int(stat.st_size)
    if not bank or file_count == 0:
        raise NewHeadDevEvalError("direct support TSV has no candidate images")
    return {
        "support_tsv_path": str(tsv_path),
        "class_count": len(bank),
        "file_count": file_count,
        "total_size_bytes": total_size,
        "ordered_content_sha256": digest.hexdigest(),
    }


def _stable_image_record(path: Path, *, label: str) -> dict[str, Any]:
    try:
        path = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise NewHeadDevEvalError(f"could not resolve {label}: {path}") from error
    if not path.is_file():
        raise NewHeadDevEvalError(f"{label} is not a regular file: {path}")
    before = path.stat()
    digest = _sha256_file(path)
    after = path.stat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if before_identity != after_identity:
        raise NewHeadDevEvalError(f"{label} changed while hashing: {path}")
    return {
        "path": str(path),
        "size_bytes": int(after.st_size),
        "sha256": digest,
    }


def query_image_content_record(
    binding: PartitionBinding, *, data_root: Path
) -> dict[str, Any]:
    coco_root = (data_root / "COCO/coco2014").expanduser().resolve(strict=True)
    image_keys = sorted(
        {
            (row.coco_split, row.image_id)
            for manifest in binding.manifests
            for row in manifest.rows
        }
    )
    if not image_keys:
        raise NewHeadDevEvalError("selected dev manifests have no query images")
    digest = hashlib.sha256()
    total_size = 0
    for coco_split, image_id in image_keys:
        image_path = (
            coco_root
            / coco_split
            / f"COCO_{coco_split}_{int(image_id):012d}.jpg"
        )
        record = _stable_image_record(
            image_path,
            label=f"query image {coco_split}/{int(image_id):012d}",
        )
        header = json.dumps(
            [
                coco_split,
                int(image_id),
                record["path"],
                record["size_bytes"],
                record["sha256"],
            ],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        total_size += int(record["size_bytes"])
    return {
        "coco_root": str(coco_root),
        "iteration_policy": "sorted_unique_coco_split_then_image_id_v1",
        "image_count": len(image_keys),
        "total_size_bytes": total_size,
        "ordered_content_sha256": digest.hexdigest(),
    }


def effective_eval_contract(cfg: Any, device: torch.device) -> dict[str, Any]:
    if os.environ.get("GFLOPS_DEBUG_SHILONG") == "INFO":
        raise NewHeadDevEvalError(
            "GFLOPS_DEBUG_SHILONG=INFO changes evaluation resize and is forbidden"
        )
    raw_scales = getattr(
        cfg,
        "data_aug_scales",
        [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800],
    )
    if (
        not isinstance(raw_scales, Sequence)
        or isinstance(raw_scales, (str, bytes))
        or not raw_scales
    ):
        raise NewHeadDevEvalError("data_aug_scales must be a non-empty sequence")
    configured_scales = []
    for index, value in enumerate(raw_scales):
        if type(value) is not int or value <= 0:
            raise NewHeadDevEvalError(
                f"data_aug_scales[{index}] must be an exact positive integer"
            )
        configured_scales.append(int(value))
    configured_max_size = getattr(cfg, "data_aug_max_size", 1333)
    if type(configured_max_size) is not int or configured_max_size <= 0:
        raise NewHeadDevEvalError(
            "data_aug_max_size must be an exact positive integer"
        )
    raw_overlap = getattr(cfg, "data_aug_scale_overlap", None)
    if raw_overlap is not None:
        if isinstance(raw_overlap, bool) or not isinstance(raw_overlap, (int, float)):
            raise NewHeadDevEvalError(
                "data_aug_scale_overlap must be None or a finite number"
            )
        overlap = float(raw_overlap)
        if not math.isfinite(overlap):
            raise NewHeadDevEvalError(
                "data_aug_scale_overlap must be None or a finite number"
            )
    else:
        overlap = None
    effective_scales = list(configured_scales)
    effective_max_size = int(configured_max_size)
    if overlap is not None and overlap > 0:
        effective_scales = [int(value * overlap) for value in configured_scales]
        effective_max_size = int(configured_max_size * overlap)
    if any(value <= 0 for value in effective_scales) or effective_max_size <= 0:
        raise NewHeadDevEvalError("effective evaluation resize must stay positive")
    max_text_len = getattr(cfg, "max_text_len", 256)
    if type(max_text_len) is not int or max_text_len <= 0:
        raise NewHeadDevEvalError("max_text_len must be an exact positive integer")
    text_encoder_type = getattr(cfg, "text_encoder_type", "bert-base-uncased")
    if not isinstance(text_encoder_type, str) or not text_encoder_type.strip():
        raise NewHeadDevEvalError("text_encoder_type must be non-empty text")
    return {
        "configured_resize_scales": configured_scales,
        "configured_resize_max_size": int(configured_max_size),
        "data_aug_scale_overlap": overlap,
        "effective_resize_scales": effective_scales,
        "effective_resize_max_size": effective_max_size,
        "effective_eval_short_side": max(effective_scales),
        "max_text_len": int(max_text_len),
        "text_encoder_type": text_encoder_type.strip(),
        "device": str(device),
        "device_type": device.type,
        "device_index": device.index,
        "gflops_debug_shilong": False,
    }


def _paired_training_contract(saved_args: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in PAIRED_TRAINING_EQUAL_KEYS if key not in saved_args]
    if missing:
        raise NewHeadDevEvalError(
            f"checkpoint paired training contract is missing keys: {missing}"
        )
    contract = {key: saved_args[key] for key in PAIRED_TRAINING_EQUAL_KEYS}
    try:
        _canonical_bytes(contract)
    except NewHeadDevEvalError as error:
        raise NewHeadDevEvalError(
            "checkpoint paired training contract is not canonical JSON"
        ) from error
    return contract


def _validate_eval_config_training_contract(
    cfg: Any, saved_args: Mapping[str, Any]
) -> dict[str, Any]:
    missing_cfg = [
        key
        for key in EVALUATION_CONFIG_CHECKPOINT_EQUAL_KEYS
        if not hasattr(cfg, key)
    ]
    missing_checkpoint = [
        key
        for key in EVALUATION_CONFIG_CHECKPOINT_EQUAL_KEYS
        if key not in saved_args
    ]
    if missing_cfg or missing_checkpoint:
        raise NewHeadDevEvalError(
            "evaluation config/checkpoint training contract is incomplete: "
            f"config_missing={missing_cfg}, checkpoint_missing={missing_checkpoint}"
        )
    drift = {
        key: (getattr(cfg, key), saved_args[key])
        for key in EVALUATION_CONFIG_CHECKPOINT_EQUAL_KEYS
        if getattr(cfg, key) != saved_args[key]
    }
    if drift:
        raise NewHeadDevEvalError(
            f"evaluation config differs from checkpoint training contract: {drift}"
        )
    matched = {
        key: saved_args[key] for key in EVALUATION_CONFIG_CHECKPOINT_EQUAL_KEYS
    }
    try:
        _canonical_bytes(matched)
    except NewHeadDevEvalError as error:
        raise NewHeadDevEvalError(
            "evaluation config/checkpoint training contract is not canonical JSON"
        ) from error
    return matched


def _new_head_execution_status(
    saved_args: Mapping[str, Any], *, optimizer_updates: int
) -> dict[str, Any]:
    scope = saved_args["stage_b_data_driven_execution_scope"]
    fresh_start = saved_args["stage_b_data_driven_formal_fresh_start"]
    expected_updates = saved_args[
        "stage_b_data_driven_formal_expected_optimizer_updates"
    ]
    contract = saved_args["stage_b_data_driven_new_head_formal_contract"]
    if not isinstance(scope, str) or not scope.strip():
        raise NewHeadDevEvalError("checkpoint new-head execution scope is invalid")
    if type(fresh_start) is not bool:
        raise NewHeadDevEvalError(
            "checkpoint formal_fresh_start must be an exact boolean"
        )
    _require_exact_int(
        expected_updates,
        label="checkpoint formal expected optimizer updates",
        minimum=1,
    )
    if not isinstance(contract, str) or not contract.strip():
        raise NewHeadDevEvalError("checkpoint new-head formal contract is invalid")

    scope = scope.strip()
    contract = contract.strip()
    if scope == FORMAL_NEW_HEAD_EXECUTION_SCOPE:
        declaration_drift = {}
        if fresh_start is not True:
            declaration_drift["formal_fresh_start"] = fresh_start
        if expected_updates != FORMAL_NEW_HEAD_OPTIMIZER_UPDATES:
            declaration_drift["formal_expected_optimizer_updates"] = (
                expected_updates
            )
        if contract != FORMAL_NEW_HEAD_CONTRACT:
            declaration_drift["new_head_formal_contract"] = contract
        if declaration_drift:
            raise NewHeadDevEvalError(
                "formal new-head checkpoint declaration drifted: "
                f"{declaration_drift}"
            )
        complete = optimizer_updates == FORMAL_NEW_HEAD_OPTIMIZER_UPDATES
        return {
            "formal": complete,
            "reason": (
                "sealed_formal_fresh_run_complete"
                if complete
                else "sealed_formal_fresh_run_intermediate_checkpoint"
            ),
            "execution_scope": scope,
            "formal_fresh_start": True,
            "declared_optimizer_updates": expected_updates,
            "observed_optimizer_updates": optimizer_updates,
            "formal_contract": contract,
        }
    if (
        contract == FORMAL_NEW_HEAD_CONTRACT
        or fresh_start is True
        or expected_updates == FORMAL_NEW_HEAD_OPTIMIZER_UPDATES
    ):
        raise NewHeadDevEvalError(
            "diagnostic checkpoint partially claims the sealed formal contract"
        )
    return {
        "formal": False,
        "reason": "diagnostic_new_head_execution_scope",
        "execution_scope": scope,
        "formal_fresh_start": fresh_start,
        "declared_optimizer_updates": expected_updates,
        "observed_optimizer_updates": optimizer_updates,
        "formal_contract": contract,
    }


def _shared_training_provenance(saved_args: Mapping[str, Any]) -> dict[str, Any]:
    provenance = saved_args.get("stage_b_data_driven_training_provenance")
    if not isinstance(provenance, Mapping):
        raise NewHeadDevEvalError(
            "checkpoint has no data-driven training provenance"
        )
    missing = [key for key in SHARED_TRAINING_PROVENANCE_KEYS if key not in provenance]
    if missing:
        raise NewHeadDevEvalError(
            f"checkpoint shared training provenance is missing keys: {missing}"
        )
    shared = {key: provenance[key] for key in SHARED_TRAINING_PROVENANCE_KEYS}
    try:
        _canonical_bytes(shared)
    except NewHeadDevEvalError as error:
        raise NewHeadDevEvalError(
            "checkpoint shared training provenance is not canonical JSON"
        ) from error
    required_allocator = shared["required_allocator"]
    if required_allocator != {
        "environment_variable": saved_args[
            "stage_b_data_driven_required_allocator_env"
        ],
        "value": saved_args["stage_b_data_driven_required_allocator_conf"],
    }:
        raise NewHeadDevEvalError(
            "checkpoint required allocator provenance differs from saved args"
        )
    return shared


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise NewHeadDevEvalError(f"{label} must be a lowercase SHA-256")
    return value


def _require_exact_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise NewHeadDevEvalError(
            f"{label} must be an exact integer >= {minimum}"
        )
    return int(value)


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NewHeadDevEvalError(f"could not load {label}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise NewHeadDevEvalError(f"{label} must be a JSON object: {path}")
    return value


def _identity_from_row(
    row: Mapping[str, Any], *, path: Path, line_number: int
) -> ManifestRow:
    context = f"{path}:{line_number}"
    values = tuple(row.get(key) for key in IDENTITY_KEYS)
    if any(value is None for value in values):
        raise NewHeadDevEvalError(f"{context}: identity field is missing")
    for key in ("image_id", "ann_id", "ref_id", "sent_id"):
        if type(row.get(key)) is not int:
            raise NewHeadDevEvalError(f"{context}: {key} must be an exact integer")
    for key in ("source", "split", "filename"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            raise NewHeadDevEvalError(f"{context}: {key} must be non-empty text")
    basename = Path(str(row["filename"])).name
    match = _COCO_FILENAME_RE.fullmatch(basename)
    if match is None:
        raise NewHeadDevEvalError(
            f"{context}: filename is not a canonical COCO 2014 image"
        )
    image_id = int(row["image_id"])
    if int(match.group("image_id")) != image_id:
        raise NewHeadDevEvalError(f"{context}: filename/image_id drifted")
    identity_object = {key: row[key] for key in IDENTITY_KEYS}
    return ManifestRow(
        identity=values,
        identity_object=identity_object,
        coco_split=match.group("split").lower(),
        image_id=image_id,
    )


def _scan_bound_manifest(
    *, name: str, source: str, record: Mapping[str, Any], variant: str, partition: str
) -> BoundManifest:
    expected_keys = {
        "path",
        "rows",
        "unique_identities",
        "unique_image_keys",
        "ordered_identity_stream_sha256",
        "size_bytes",
        "sha256",
    }
    if not isinstance(record, Mapping) or set(record) != expected_keys:
        raise NewHeadDevEvalError(
            f"partition output record keys drifted: {variant}/{partition}/{name}"
        )
    path_value = record.get("path")
    if not isinstance(path_value, str) or not path_value.strip():
        raise NewHeadDevEvalError(f"partition output path is invalid: {name}")
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        raise NewHeadDevEvalError(f"partition output path must be absolute: {path}")
    path = path.resolve(strict=True)
    if (
        path.name != name
        or path.parent.name != partition
        or path.parent.parent.name != variant
    ):
        raise NewHeadDevEvalError(
            f"partition output layout drifted: {variant}/{partition}/{name}"
        )
    rows_expected = _require_exact_int(record.get("rows"), label=f"{name}.rows")
    unique_expected = _require_exact_int(
        record.get("unique_identities"), label=f"{name}.unique_identities"
    )
    images_expected = _require_exact_int(
        record.get("unique_image_keys"), label=f"{name}.unique_image_keys"
    )
    size_expected = _require_exact_int(
        record.get("size_bytes"), label=f"{name}.size_bytes"
    )
    expected_sha = _require_sha256(record.get("sha256"), label=f"{name}.sha256")
    expected_identity_sha = _require_sha256(
        record.get("ordered_identity_stream_sha256"),
        label=f"{name}.ordered_identity_stream_sha256",
    )
    observed = _file_record(path)
    if observed["size_bytes"] != size_expected or observed["sha256"] != expected_sha:
        raise NewHeadDevEvalError(f"partition output file identity drifted: {path}")

    rows: list[ManifestRow] = []
    identity_digest = hashlib.sha256()
    identities: set[bytes] = set()
    images: set[tuple[str, int]] = set()
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.endswith(b"\n") or not raw.rstrip(b"\r\n"):
                raise NewHeadDevEvalError(
                    f"manifest row must be non-empty and LF terminated: {path}:{line_number}"
                )
            try:
                row = json.loads(raw)
            except (UnicodeError, json.JSONDecodeError) as error:
                raise NewHeadDevEvalError(
                    f"invalid manifest JSON: {path}:{line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise NewHeadDevEvalError(
                    f"manifest row is not an object: {path}:{line_number}"
                )
            parsed = _identity_from_row(row, path=path, line_number=line_number)
            identity_bytes = _canonical_bytes(parsed.identity)
            identity_member = hashlib.sha256(identity_bytes).digest()
            if identity_member in identities:
                raise NewHeadDevEvalError(f"duplicate manifest identity: {path}:{line_number}")
            identities.add(identity_member)
            identity_digest.update(identity_bytes + b"\n")
            images.add((parsed.coco_split, parsed.image_id))
            rows.append(parsed)
    if (
        len(rows) != rows_expected
        or len(identities) != unique_expected
        or len(images) != images_expected
        or unique_expected != rows_expected
        or identity_digest.hexdigest() != expected_identity_sha
    ):
        raise NewHeadDevEvalError(f"partition output stream contract drifted: {path}")
    return BoundManifest(
        name=name,
        source=source,
        path=path,
        record=dict(record),
        rows=tuple(rows),
    )


def load_partition_binding(
    receipt_path: Path,
    *,
    variant: str,
    partition: str,
    expected_receipt_sha256: str | None = None,
) -> PartitionBinding:
    if variant not in VARIANTS:
        raise NewHeadDevEvalError(f"unsupported variant: {variant!r}")
    if partition not in DEV_PARTITIONS:
        raise NewHeadDevEvalError(f"unsupported dev partition: {partition!r}")
    receipt_path = receipt_path.expanduser().resolve(strict=True)
    receipt_file = _file_record(receipt_path)
    if expected_receipt_sha256 is not None:
        expected_receipt_sha256 = _require_sha256(
            expected_receipt_sha256, label="expected partition receipt SHA-256"
        )
        if receipt_file["sha256"] != expected_receipt_sha256:
            raise NewHeadDevEvalError(
                "partition receipt does not match the preregistered SHA-256"
            )
    receipt = _load_json_object(receipt_path, label="new-head partition receipt")
    if receipt.get("schema") != PARTITION_RECEIPT_SCHEMA:
        raise NewHeadDevEvalError("new-head partition receipt schema drifted")
    canonical_sha = _require_sha256(
        receipt.get("canonical_payload_sha256"),
        label="partition canonical payload hash",
    )
    canonical_payload = dict(receipt)
    del canonical_payload["canonical_payload_sha256"]
    if _sha256_bytes(_canonical_bytes(canonical_payload)) != canonical_sha:
        raise NewHeadDevEvalError("new-head partition canonical payload hash drifted")
    _validate_required_invariants(
        receipt.get("invariants"),
        required=PARTITION_RECEIPT_REQUIRED_INVARIANTS,
        label="new-head partition receipt",
    )
    if receipt.get("source_manifest_order") != list(MANIFEST_NAMES):
        raise NewHeadDevEvalError("new-head partition source order drifted")
    if receipt.get("output_layout") != "<variant>/<partition>/<source_manifest>":
        raise NewHeadDevEvalError("new-head partition output layout drifted")
    if (
        receipt.get("output_stream_encoding")
        != "raw_input_record_including_original_line_ending_v1"
    ):
        raise NewHeadDevEvalError("new-head partition stream encoding drifted")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(VARIANTS):
        raise NewHeadDevEvalError("new-head partition output variants drifted")
    for output_variant in VARIANTS:
        value = outputs.get(output_variant)
        if not isinstance(value, dict) or set(value) != set(PARTITIONS):
            raise NewHeadDevEvalError(
                f"new-head partition set drifted for {output_variant}"
            )
        for output_partition in PARTITIONS:
            records = value.get(output_partition)
            if not isinstance(records, dict) or set(records) != set(MANIFEST_NAMES):
                raise NewHeadDevEvalError(
                    "new-head partition manifest set drifted: "
                    f"{output_variant}/{output_partition}"
                )
    selected = outputs[variant][partition]
    manifests = tuple(
        _scan_bound_manifest(
            name=name,
            source=source,
            record=selected[name],
            variant=variant,
            partition=partition,
        )
        for name, source in SOURCE_MANIFESTS
    )
    summary = receipt.get("partition_summary")
    selected_summary = summary.get(partition) if isinstance(summary, dict) else None
    if not isinstance(selected_summary, dict):
        raise NewHeadDevEvalError("new-head partition summary is missing")
    if selected_summary.get("rows_by_manifest") != {
        manifest.name: len(manifest.rows) for manifest in manifests
    } or selected_summary.get("rows") != sum(len(manifest.rows) for manifest in manifests):
        raise NewHeadDevEvalError("new-head partition selected summary drifted")
    return PartitionBinding(
        receipt_path=receipt_path,
        receipt_file=receipt_file,
        canonical_payload_sha256=canonical_sha,
        variant=variant,
        partition=partition,
        manifests=manifests,
    )


def load_shared_evaluation_binding(
    receipt_path: Path,
    *,
    experiment_variant: str,
    partition: str,
    expected_receipt_sha256: str | None = None,
) -> PartitionBinding:
    if experiment_variant not in VARIANTS:
        raise NewHeadDevEvalError(
            f"unsupported experiment variant: {experiment_variant!r}"
        )
    return load_partition_binding(
        receipt_path,
        variant=EVALUATION_DATA_VARIANT,
        partition=partition,
        expected_receipt_sha256=expected_receipt_sha256,
    )


def load_support_binding(
    receipt_path: Path,
    *,
    expected_receipt_sha256: str,
    partition_binding: PartitionBinding,
    canonical_classes_path: Path,
) -> SupportBinding:
    receipt_path = receipt_path.expanduser().resolve(strict=True)
    receipt_file = _file_record(receipt_path)
    expected_receipt_sha256 = _require_sha256(
        expected_receipt_sha256, label="expected support receipt SHA-256"
    )
    if receipt_file["sha256"] != expected_receipt_sha256:
        raise NewHeadDevEvalError("support receipt does not match its sealed SHA-256")
    receipt = _load_json_object(receipt_path, label="support partition receipt")
    if receipt.get("schema") != SUPPORT_RECEIPT_SCHEMA:
        raise NewHeadDevEvalError("support partition receipt schema drifted")
    canonical_sha = _require_sha256(
        receipt.get("canonical_payload_sha256"),
        label="support receipt canonical payload hash",
    )
    canonical_payload = dict(receipt)
    del canonical_payload["canonical_payload_sha256"]
    if _sha256_bytes(_canonical_bytes(canonical_payload)) != canonical_sha:
        raise NewHeadDevEvalError("support receipt canonical payload hash drifted")
    _validate_required_invariants(
        receipt.get("invariants"),
        required=SUPPORT_RECEIPT_REQUIRED_INVARIANTS,
        label="support receipt",
    )

    inputs = receipt.get("inputs")
    if not isinstance(inputs, Mapping):
        raise NewHeadDevEvalError("support receipt input bindings are missing")
    partition_record = inputs.get("partition_receipt")
    if not isinstance(partition_record, Mapping) or dict(partition_record) != dict(
        partition_binding.receipt_file
    ):
        raise NewHeadDevEvalError(
            "support receipt was not built from the selected new-head partition"
        )
    canonical_classes = _file_record(
        canonical_classes_path.expanduser().resolve(strict=True)
    )
    canonical_record = inputs.get("canonical_classes")
    if not isinstance(canonical_record, Mapping) or dict(canonical_record) != dict(
        canonical_classes
    ):
        raise NewHeadDevEvalError(
            "evaluation canonical classes differ from the support receipt"
        )

    contract = receipt.get("filter_contract")
    required_settings = (
        contract.get("required_dataset_settings")
        if isinstance(contract, Mapping)
        else None
    )
    if (
        not isinstance(contract, Mapping)
        or contract.get("candidate_stream_policy")
        != "sealed_cache_order_delete_only_v1"
        or contract.get("exclusion_policy")
        != "dev_full_union_official_ref8_by_numeric_coco_id_v1"
        or contract.get("D0_and_D1_share_identical_runtime_bank") is not True
        or contract.get("bank_consumers") != ["D0", "D1"]
        or required_settings
        != {
            "patch_bank_cache": False,
            "patch_bank_cache_write": False,
            "support_patch_max_per_class": 200,
            "support_patch_use_embedding": False,
        }
    ):
        raise NewHeadDevEvalError("support filter/runtime contract drifted")
    coverage = receipt.get("training_class_coverage")
    if (
        not isinstance(coverage, Mapping)
        or coverage.get("required_class_count") != 78
        or coverage.get("covered_class_count") != 78
        or coverage.get("missing_class_ids") != []
    ):
        raise NewHeadDevEvalError("support receipt training class coverage drifted")
    outputs = receipt.get("outputs")
    sealed_runtime = (
        outputs.get("runtime_support_tsv") if isinstance(outputs, Mapping) else None
    )
    if not isinstance(sealed_runtime, Mapping) or set(sealed_runtime) != {
        "path",
        "rows",
        "size_bytes",
        "sha256",
    }:
        raise NewHeadDevEvalError("support runtime TSV binding drifted")
    rows = _require_exact_int(
        sealed_runtime.get("rows"), label="support runtime TSV rows", minimum=1
    )
    runtime_path = Path(str(sealed_runtime.get("path", ""))).expanduser().resolve(
        strict=True
    )
    runtime_file = _file_record(runtime_path)
    if (
        runtime_file["sha256"] != sealed_runtime.get("sha256")
        or runtime_file["size_bytes"] != sealed_runtime.get("size_bytes")
    ):
        raise NewHeadDevEvalError("support runtime TSV file identity drifted")
    support_content = _direct_support_pool_content_record(runtime_path)
    if support_content["file_count"] != rows:
        raise NewHeadDevEvalError(
            "support runtime TSV row/content counts drifted: "
            f"receipt={rows}, content={support_content['file_count']}"
        )
    return SupportBinding(
        receipt_path=receipt_path,
        receipt_file=receipt_file,
        canonical_payload_sha256=canonical_sha,
        runtime_tsv={**runtime_file, "rows": rows},
        canonical_classes=canonical_classes,
        support_patch_pool_content=support_content,
    )


def _validate_bound_file_record(record: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise NewHeadDevEvalError(f"{label} must contain exact path/sha256 fields")
    path_value = record.get("path")
    if not isinstance(path_value, str) or not path_value.strip():
        raise NewHeadDevEvalError(f"{label} path is invalid")
    expected_sha = _require_sha256(record.get("sha256"), label=f"{label}.sha256")
    observed = _file_record(Path(path_value))
    if observed["sha256"] != expected_sha:
        raise NewHeadDevEvalError(f"{label} SHA-256 drifted")
    return observed


def _validate_initializer_provenance(saved_args: Mapping[str, Any]) -> dict[str, Any]:
    initializer = saved_args.get("stage_b_data_driven_base_initializer")
    pair = saved_args.get("stage_b_data_driven_initializer_pair_receipt")
    if not isinstance(initializer, Mapping) or not isinstance(pair, Mapping):
        raise NewHeadDevEvalError(
            "checkpoint lacks bound common-initializer provenance"
        )
    initializer_file = _validate_bound_file_record(
        initializer, label="checkpoint base initializer"
    )
    if (
        saved_args.get("stage_b_data_driven_base_initializer_sha256")
        != initializer_file["sha256"]
        or Path(str(saved_args.get("stage_b_data_driven_base_initializer_path", "")))
        .expanduser()
        .resolve(strict=True)
        != Path(initializer_file["path"])
    ):
        raise NewHeadDevEvalError("checkpoint base initializer attribution drifted")
    pair_path = Path(str(pair.get("path", ""))).expanduser().resolve(strict=True)
    pair_sha = _require_sha256(pair.get("sha256"), label="initializer pair receipt")
    pair_file = _file_record(pair_path)
    if pair_file["sha256"] != pair_sha:
        raise NewHeadDevEvalError("initializer pair receipt SHA-256 drifted")
    pair_payload = _load_json_object(pair_path, label="initializer pair receipt")
    if (
        pair_payload.get("schema") != "pivot.stageb.data_driven_initializer_pair/v1"
        or pair_payload.get("status") != "passed"
    ):
        raise NewHeadDevEvalError("initializer pair receipt is not passed")
    members = {
        value.get("sha256")
        for value in (
            pair_payload.get("absolute_initializer"),
            pair_payload.get("relational_initializer"),
        )
        if isinstance(value, Mapping)
    }
    if initializer_file["sha256"] not in members:
        raise NewHeadDevEvalError("base initializer is absent from its pair receipt")
    invariants = pair_payload.get("invariants")
    if not isinstance(invariants, Mapping) or any(
        invariants.get(key) is not True
        for key in INITIALIZER_PAIR_REQUIRED_INVARIANTS
    ):
        raise NewHeadDevEvalError("initializer pair receipt invariants drifted")
    common_tensor_sha = _require_sha256(
        (pair_payload.get("common_non_rank_non_contract") or {}).get("tensor_sha256"),
        label="initializer common tensor hash",
    )
    if pair.get("common_tensor_sha256") != common_tensor_sha:
        raise NewHeadDevEvalError("initializer common tensor attribution drifted")
    return {
        "base_initializer": initializer_file,
        "pair_receipt": pair_file,
        "common_tensor_sha256": common_tensor_sha,
        "required_invariants": {
            key: True for key in INITIALIZER_PAIR_REQUIRED_INVARIANTS
        },
    }


def _tensor_bitwise_equal(left: Any, right: Any) -> bool:
    if not torch.is_tensor(left) or not torch.is_tensor(right):
        return False
    if (
        left.dtype != right.dtype
        or left.layout != right.layout
        or left.layout != torch.strided
        or tuple(left.shape) != tuple(right.shape)
        or tuple(left.stride()) != tuple(right.stride())
    ):
        return False
    left_bytes = left.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    right_bytes = right.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return bool(torch.equal(left_bytes, right_bytes))


def _audit_frozen_initializer_tensors(
    initializer_payload: Any, trained_state: Any
) -> dict[str, Any]:
    if not isinstance(initializer_payload, Mapping) or set(initializer_payload) != {
        "model",
        "data_driven_initializer",
    }:
        raise NewHeadDevEvalError("A0 initializer payload schema drifted")
    initializer_state = initializer_payload.get("model")
    metadata = initializer_payload.get("data_driven_initializer")
    if not isinstance(initializer_state, Mapping) or not isinstance(
        trained_state, Mapping
    ):
        raise NewHeadDevEvalError("A0/trained model state is missing")
    if not isinstance(metadata, Mapping) or metadata.get("schema") != (
        "pivot.stageb.data_driven.initializer/v1"
    ):
        raise NewHeadDevEvalError("A0 initializer metadata schema drifted")
    roles = metadata.get("role_keys")
    if not isinstance(roles, Mapping) or set(roles) != set(
        ABSOLUTE_INITIALIZER_ROLE_NAMES
    ):
        raise NewHeadDevEvalError("A0 initializer role schema drifted")
    normalized: dict[str, list[str]] = {}
    flattened: list[str] = []
    for role in ABSOLUTE_INITIALIZER_ROLE_NAMES:
        keys = roles.get(role)
        if (
            not isinstance(keys, list)
            or not keys
            or any(not isinstance(key, str) or not key for key in keys)
            or len(set(keys)) != len(keys)
        ):
            raise NewHeadDevEvalError(f"A0 initializer role {role!r} drifted")
        normalized[role] = list(keys)
        flattened.extend(keys)
    if len(flattened) != len(set(flattened)):
        raise NewHeadDevEvalError("A0 initializer roles overlap")
    if set(flattened) != set(initializer_state):
        raise NewHeadDevEvalError(
            "A0 initializer roles do not partition the model state"
        )
    if set(initializer_state) != set(trained_state):
        raise NewHeadDevEvalError(
            "trained model key coverage differs from the A0 initializer"
        )
    non_tensors = [
        key
        for key in initializer_state
        if not torch.is_tensor(initializer_state[key])
        or not torch.is_tensor(trained_state[key])
    ]
    if non_tensors:
        raise NewHeadDevEvalError(
            f"A0/trained model state contains non-tensors: {non_tensors[:8]}"
        )
    required_invariants = (
        "b58_is_only_checkpoint_source",
        "no_r100_p50_u0_or_stagea_tensor_source",
        "canonical_query_and_full_text_heads_are_separate",
        "rank_and_confidence_parameters_are_disjoint",
        "patch_backbone_aliases_b58",
    )
    _validate_required_invariants(
        metadata.get("invariants"),
        required=required_invariants,
        label="A0 initializer",
    )
    for role, keys in normalized.items():
        try:
            observed = data_driven_tensor_state_sha256(initializer_state, keys)
        except ValueError as error:
            raise NewHeadDevEvalError(
                f"could not hash A0 initializer role {role!r}"
            ) from error
        if metadata.get(f"{role}_tensor_sha256") != observed:
            raise NewHeadDevEvalError(
                f"A0 initializer metadata hash drifted for role {role!r}"
            )
    if set(normalized["random_patch_projection"]) != set(PATCH_PROJECTION_KEYS):
        raise NewHeadDevEvalError("A0 patch projection role drifted")
    if not all(
        key.startswith("patch_encoder.backbone.")
        for key in normalized["shared_backbone_alias"]
    ):
        raise NewHeadDevEvalError("A0 shared-backbone alias role drifted")

    absolute = normalized["random_absolute_heads"]
    rank = [
        key
        for key in absolute
        if key.startswith("stage_b_data_driven_score_heads.rank_branch.")
    ]
    confidence = [
        key
        for key in absolute
        if key.startswith(
            "stage_b_data_driven_score_heads.confidence_branch."
        )
        or key.startswith("stage_b_data_driven_score_heads.confidence_gate.")
    ]
    score_contract = [key for key in absolute if key == SCORE_CONTRACT_BUFFER_KEY]
    if (
        not rank
        or not confidence
        or score_contract != [SCORE_CONTRACT_BUFFER_KEY]
        or set(rank).intersection(confidence)
        or set(rank) | set(confidence) | set(score_contract) != set(absolute)
    ):
        raise NewHeadDevEvalError(
            "A0 absolute rank/confidence/contract role split drifted"
        )

    frozen_roles = {
        "b58_base": normalized["b58_base"],
        "shared_backbone_alias": normalized["shared_backbone_alias"],
        "random_absolute_confidence": confidence,
        "score_contract_buffer": score_contract,
    }
    frozen_keys = sorted(
        key for keys in frozen_roles.values() for key in keys
    )
    changed_frozen = [
        key
        for key in frozen_keys
        if not _tensor_bitwise_equal(initializer_state[key], trained_state[key])
    ]
    if changed_frozen:
        raise NewHeadDevEvalError(
            "formal checkpoint changed frozen A0 tensors: "
            f"{changed_frozen[:8]}"
        )

    role_audit = {}
    for role, keys in frozen_roles.items():
        initializer_hash = data_driven_tensor_state_sha256(initializer_state, keys)
        checkpoint_hash = data_driven_tensor_state_sha256(trained_state, keys)
        if initializer_hash != checkpoint_hash:
            raise NewHeadDevEvalError(
                f"formal checkpoint frozen role hash drifted: {role}"
            )
        role_audit[role] = {
            "tensor_count": len(keys),
            "initializer_tensor_sha256": initializer_hash,
            "checkpoint_tensor_sha256": checkpoint_hash,
            "all_tensors_bitwise_equal": True,
        }
    initializer_frozen_hash = data_driven_tensor_state_sha256(
        initializer_state, frozen_keys
    )
    checkpoint_frozen_hash = data_driven_tensor_state_sha256(
        trained_state, frozen_keys
    )
    mutable_roles = {
        "absolute_rank": rank,
        "random_patch_projection": normalized["random_patch_projection"],
    }
    mutable_audit = {}
    for role, keys in mutable_roles.items():
        changed_count = sum(
            not _tensor_bitwise_equal(initializer_state[key], trained_state[key])
            for key in keys
        )
        initializer_hash = data_driven_tensor_state_sha256(
            initializer_state, keys
        )
        checkpoint_hash = data_driven_tensor_state_sha256(trained_state, keys)
        if changed_count <= 0 or checkpoint_hash == initializer_hash:
            raise NewHeadDevEvalError(
                f"formal checkpoint did not update mutable A0 role: {role}"
            )
        mutable_audit[role] = {
            "tensor_count": len(keys),
            "changed_tensor_count": changed_count,
            "initializer_tensor_sha256": initializer_hash,
            "checkpoint_tensor_sha256": checkpoint_hash,
        }
    return {
        "schema": "pivot.stageb.data_driven.a0_frozen_tensor_audit/v1",
        "passed": True,
        "initializer_metadata_schema": metadata["schema"],
        "model_tensor_count": len(initializer_state),
        "frozen_tensor_count": len(frozen_keys),
        "initializer_frozen_tensor_sha256": initializer_frozen_hash,
        "checkpoint_frozen_tensor_sha256": checkpoint_frozen_hash,
        "frozen_tensor_sha256_equal": (
            initializer_frozen_hash == checkpoint_frozen_hash
        ),
        "frozen_roles": role_audit,
        "mutable_roles": mutable_audit,
    }


def _load_initializer_payload_mmap(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(
            path, map_location="cpu", weights_only=False, mmap=True
        )
    except TypeError:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise NewHeadDevEvalError("A0 initializer payload is not a mapping")
    return payload


def _training_partition_status(
    saved_args: Mapping[str, Any],
    binding: PartitionBinding,
    *,
    experiment_variant: str,
    support_binding: SupportBinding | None = None,
) -> dict[str, Any]:
    record = saved_args.get("stage_b_data_driven_dataset_config")
    if not isinstance(record, Mapping):
        return {
            "formal": False,
            "reason": "checkpoint_missing_bound_dataset_config",
        }
    dataset_file = _validate_bound_file_record(
        record, label="checkpoint dataset config"
    )
    datasets_value = saved_args.get("datasets")
    if not isinstance(datasets_value, str) or (
        Path(datasets_value).expanduser().resolve(strict=True)
        != Path(dataset_file["path"])
    ):
        raise NewHeadDevEvalError("checkpoint dataset config path attribution drifted")
    payload = _load_json_object(Path(dataset_file["path"]), label="checkpoint dataset config")
    train = payload.get("train")
    if not isinstance(train, list) or len(train) != len(MANIFEST_NAMES):
        raise NewHeadDevEvalError("checkpoint dataset config must contain three train rows")
    receipt_schemas: set[str] = set()
    loaded_receipts: dict[Path, dict[str, Any]] = {}
    loaded_receipt_files: dict[Path, dict[str, Any]] = {}
    for row in train:
        if not isinstance(row, dict):
            raise NewHeadDevEvalError("checkpoint dataset train row is invalid")
        receipt_value = row.get("stage_b_data_driven_receipt")
        receipt_sha = row.get("stage_b_data_driven_receipt_sha256")
        if not isinstance(receipt_value, str) or not receipt_value.strip():
            return {
                "formal": False,
                "reason": "training_rows_do_not_bind_a_partition_receipt",
                "dataset_config": dataset_file,
            }
        receipt_path = Path(receipt_value).expanduser().resolve(strict=True)
        _require_sha256(receipt_sha, label="training partition receipt SHA-256")
        if receipt_path not in loaded_receipts:
            observed = _file_record(receipt_path)
            if observed["sha256"] != receipt_sha:
                raise NewHeadDevEvalError("training partition receipt SHA-256 drifted")
            loaded_receipt_files[receipt_path] = observed
            loaded_receipts[receipt_path] = _load_json_object(
                receipt_path, label="training partition receipt"
            )
        elif loaded_receipt_files[receipt_path]["sha256"] != receipt_sha:
            raise NewHeadDevEvalError(
                "training rows disagree on partition receipt SHA-256"
            )
        receipt_schemas.add(str(loaded_receipts[receipt_path].get("schema", "")))
    if receipt_schemas != {TRAINING_PARTITION_SCHEMA}:
        return {
            "formal": False,
            "reason": "legacy_training_data_not_new_head_partition",
            "dataset_config": dataset_file,
            "observed_receipt_schemas": sorted(receipt_schemas),
        }
    if set(loaded_receipts) != {binding.receipt_path}:
        raise NewHeadDevEvalError(
            "checkpoint training partition differs from evaluation partition receipt"
        )
    training_receipt = loaded_receipts[binding.receipt_path]
    canonical_sha = training_receipt.get("canonical_payload_sha256")
    if canonical_sha != binding.canonical_payload_sha256:
        raise NewHeadDevEvalError("checkpoint training partition canonical hash drifted")
    if support_binding is None:
        raise NewHeadDevEvalError(
            "formal new-head checkpoint validation requires a support binding"
        )
    expected_dataset_variant = (
        "dd0_ordinary_primary"
        if experiment_variant == "d0_ordinary_primary"
        else "dd1_category_complete"
    )
    train_records = training_receipt["outputs"][experiment_variant]["train"]
    seen: set[str] = set()
    for row in train:
        anno_value = row.get("anno")
        if not isinstance(anno_value, str) or not anno_value.strip():
            raise NewHeadDevEvalError("checkpoint training annotation path is invalid")
        anno_path = Path(anno_value).expanduser().resolve(strict=True)
        name = anno_path.name
        if name not in MANIFEST_NAMES or name in seen:
            raise NewHeadDevEvalError("checkpoint training manifest set drifted")
        seen.add(name)
        expected = train_records[name]
        if (
            row.get("stage_b_data_driven_variant") != expected_dataset_variant
            or row.get("stage_b_data_driven_partition") != "train"
            or row.get("stage_b_data_driven_manifest_sha256") != expected["sha256"]
            or anno_path != Path(expected["path"]).expanduser().resolve(strict=True)
            or _file_record(anno_path)["sha256"] != expected["sha256"]
        ):
            raise NewHeadDevEvalError(
                f"checkpoint training partition binding drifted for {name}"
            )
        support_receipt_value = row.get("stage_b_data_driven_support_receipt")
        support_receipt_sha = row.get(
            "stage_b_data_driven_support_receipt_sha256"
        )
        support_tsv_value = row.get("support_patch_tsv")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (
                support_receipt_value,
                support_receipt_sha,
                support_tsv_value,
            )
        ):
            raise NewHeadDevEvalError(
                f"checkpoint training support binding is incomplete for {name}"
            )
        _require_sha256(
            support_receipt_sha,
            label=f"checkpoint training support receipt SHA-256 for {name}",
        )
        try:
            row_support_receipt = (
                Path(support_receipt_value).expanduser().resolve(strict=True)
            )
            row_support_tsv = Path(support_tsv_value).expanduser().resolve(
                strict=True
            )
        except OSError as error:
            raise NewHeadDevEvalError(
                f"checkpoint training support path could not be resolved for {name}"
            ) from error
        if (
            row_support_receipt != support_binding.receipt_path
            or support_receipt_sha
            != support_binding.receipt_file["sha256"]
            or row_support_tsv
            != Path(support_binding.runtime_tsv["path"])
        ):
            raise NewHeadDevEvalError(
                f"checkpoint training support receipt/TSV drifted for {name}"
            )
        expected_support_settings = {
            "patch_bank_cache": False,
            "patch_bank_cache_write": False,
            "support_patch_max_per_class": 200,
            "support_patch_use_embedding": False,
        }
        support_setting_drift = {
            key: (row.get(key), expected)
            for key, expected in expected_support_settings.items()
            if type(row.get(key)) is not type(expected)
            or row.get(key) != expected
        }
        if support_setting_drift:
            raise NewHeadDevEvalError(
                f"checkpoint training support settings drifted for {name}: "
                f"{support_setting_drift}"
            )
        if str(row.get("patch_bank_cache_path", "") or "").strip():
            raise NewHeadDevEvalError(
                f"checkpoint training retained a support cache path for {name}"
            )
    if seen != set(MANIFEST_NAMES):
        raise NewHeadDevEvalError("checkpoint training manifest coverage drifted")
    provenance = saved_args.get("stage_b_data_driven_training_provenance")
    observed_support_content = (
        provenance.get("support_patch_pool_content")
        if isinstance(provenance, Mapping)
        else None
    )
    expected_support_content = [support_binding.support_patch_pool_content]
    if observed_support_content != expected_support_content:
        raise NewHeadDevEvalError(
            "checkpoint training support image content provenance drifted"
        )
    return {
        "formal": True,
        "reason": "new_head_train_partition_bound",
        "dataset_config": dataset_file,
        "training_partition_receipt": binding.receipt_file,
        "training_partition_canonical_payload_sha256": canonical_sha,
        "training_support_receipt": support_binding.receipt_file,
        "training_support_receipt_canonical_payload_sha256": (
            support_binding.canonical_payload_sha256
        ),
        "training_support_patch_pool_content": (
            support_binding.support_patch_pool_content
        ),
    }


def _formal_runtime_binding_status(
    saved_args: Mapping[str, Any],
    *,
    experiment_variant: str,
    config_file: Mapping[str, Any],
    training_partition: Mapping[str, Any],
    partition_binding: PartitionBinding,
    support_binding: SupportBinding,
    initializer: Mapping[str, Any],
) -> dict[str, Any]:
    scope = str(saved_args.get("stage_b_data_driven_execution_scope", "") or "").strip()
    raw = saved_args.get("stage_b_data_driven_new_head_formal_binding")
    if scope != FORMAL_NEW_HEAD_EXECUTION_SCOPE:
        if raw not in (None, {}):
            raise NewHeadDevEvalError(
                "diagnostic checkpoint unexpectedly carries a formal runtime binding"
            )
        return {
            "formal": False,
            "reason": "diagnostic_checkpoint_has_no_formal_runtime_binding",
        }
    if not isinstance(raw, Mapping):
        raise NewHeadDevEvalError(
            "formal checkpoint has no main-validated new-head runtime binding"
        )
    try:
        _canonical_bytes(raw)
    except NewHeadDevEvalError as error:
        raise NewHeadDevEvalError(
            "formal checkpoint runtime binding is not canonical JSON"
        ) from error
    expected_keys = {
        "schema",
        "scope",
        "contract",
        "experiment_id",
        "variant_id",
        "category_complete",
        "config_file",
        "dataset_config",
        "output_dir",
        "initializer",
        "initializer_pair_receipt",
        "lr_selection_receipt",
        "partition_receipt",
        "support_receipt",
        "runtime_support_tsv",
        "manifests",
        "training_budget",
        "optimizer_contract",
        "runtime",
    }
    if set(raw) != expected_keys:
        raise NewHeadDevEvalError("formal checkpoint runtime binding keys drifted")
    expected_experiment = "DD0" if experiment_variant == VARIANTS[0] else "DD1"
    expected_variant_id = (
        "DD0-NEWHEAD-DATA"
        if experiment_variant == VARIANTS[0]
        else "DD1-NEWHEAD-DATA"
    )
    exact_header = {
        "schema": FORMAL_NEW_HEAD_BINDING_SCHEMA,
        "scope": FORMAL_NEW_HEAD_EXECUTION_SCOPE,
        "contract": FORMAL_NEW_HEAD_CONTRACT,
        "experiment_id": expected_experiment,
        "variant_id": expected_variant_id,
        "category_complete": experiment_variant == VARIANTS[1],
    }
    header_drift = {
        key: (raw.get(key), expected)
        for key, expected in exact_header.items()
        if type(raw.get(key)) is not type(expected) or raw.get(key) != expected
    }
    if header_drift:
        raise NewHeadDevEvalError(
            f"formal checkpoint runtime binding header drifted: {header_drift}"
        )

    expected_config = {
        "path": config_file["path"],
        "sha256": config_file["sha256"],
    }
    dataset_file = training_partition.get("dataset_config")
    if not isinstance(dataset_file, Mapping):
        raise NewHeadDevEvalError("formal training dataset file binding is missing")
    expected_dataset = {
        "path": dataset_file["path"],
        "sha256": dataset_file["sha256"],
    }
    expected_initializer = {
        "path": initializer["base_initializer"]["path"],
        "sha256": initializer["base_initializer"]["sha256"],
    }
    expected_pair = {
        "path": initializer["pair_receipt"]["path"],
        "sha256": initializer["pair_receipt"]["sha256"],
    }
    expected_partition = {
        "path": partition_binding.receipt_file["path"],
        "sha256": partition_binding.receipt_file["sha256"],
        "canonical_payload_sha256": (
            partition_binding.canonical_payload_sha256
        ),
    }
    expected_support = {
        "path": support_binding.receipt_file["path"],
        "sha256": support_binding.receipt_file["sha256"],
        "canonical_payload_sha256": support_binding.canonical_payload_sha256,
    }
    expected_runtime_support = {
        key: support_binding.runtime_tsv[key]
        for key in ("path", "sha256", "size_bytes", "rows")
    }
    bound_records = {
        "config_file": expected_config,
        "dataset_config": expected_dataset,
        "initializer": expected_initializer,
        "initializer_pair_receipt": expected_pair,
        "partition_receipt": expected_partition,
        "support_receipt": expected_support,
        "runtime_support_tsv": expected_runtime_support,
    }
    record_drift = {
        key: (raw.get(key), expected)
        for key, expected in bound_records.items()
        if raw.get(key) != expected
    }
    if record_drift:
        raise NewHeadDevEvalError(
            "formal checkpoint runtime asset binding drifted: "
            f"{sorted(record_drift)}"
        )

    lr_binding = raw.get("lr_selection_receipt")
    if not isinstance(lr_binding, Mapping) or set(lr_binding) != {
        "path",
        "sha256",
        "schema",
        "selected_rank_lr",
    }:
        raise NewHeadDevEvalError("formal LR selection binding drifted")
    lr_path_value = saved_args.get(
        "stage_b_data_driven_new_head_lr_selection_receipt_path"
    )
    lr_sha_value = saved_args.get(
        "stage_b_data_driven_new_head_lr_selection_receipt_sha256"
    )
    if not isinstance(lr_path_value, str) or not lr_path_value.strip():
        raise NewHeadDevEvalError("formal LR selection receipt path is missing")
    lr_sha = _require_sha256(
        lr_sha_value, label="formal LR selection receipt SHA-256"
    )
    lr_path = Path(lr_path_value).expanduser().resolve(strict=True)
    lr_file = _file_record(lr_path)
    if (
        lr_file["sha256"] != lr_sha
        or lr_binding.get("path") != lr_file["path"]
        or lr_binding.get("sha256") != lr_sha
        or lr_binding.get("schema") != FORMAL_NEW_HEAD_LR_SELECTION_SCHEMA
    ):
        raise NewHeadDevEvalError("formal LR selection receipt file drifted")
    lr_receipt = _load_json_object(lr_path, label="formal LR selection receipt")
    selected_rank_lr = lr_receipt.get("selected_rank_lr")
    saved_rank_lr = saved_args.get("stage_b_data_driven_rank_lr")
    if not (
        lr_receipt.get("schema") == FORMAL_NEW_HEAD_LR_SELECTION_SCHEMA
        and lr_receipt.get("status") == "passed"
        and lr_receipt.get("candidate_rank_lrs")
        == FORMAL_NEW_HEAD_LR_CANDIDATES
        and lr_receipt.get("optimizer_updates_per_candidate") == 1000
        and lr_receipt.get("selection_partition") == "dev_screen"
        and lr_receipt.get("selection_metric") == "macro_ref3_acc50"
        and lr_receipt.get("secondary_selection_metric")
        == "macro_ref3_mean_listwise_nll"
        and type(selected_rank_lr) is float
        and selected_rank_lr in FORMAL_NEW_HEAD_LR_CANDIDATES
        and type(saved_rank_lr) is float
        and selected_rank_lr == saved_rank_lr
        and lr_binding.get("selected_rank_lr") == selected_rank_lr
    ):
        raise NewHeadDevEvalError("formal LR selection receipt semantics drifted")
    expected_optimizer = {
        "rank_lr": saved_rank_lr,
        "selected_rank_lr": selected_rank_lr,
        "patch_lr": saved_args.get("stage_b_data_driven_patch_lr"),
        "weight_decay": saved_args.get("weight_decay"),
        "clip_max_norm": saved_args.get("clip_max_norm"),
        "lr_drop": saved_args.get("lr_drop"),
        "onecyclelr": saved_args.get("onecyclelr"),
        "multi_step_lr": saved_args.get("multi_step_lr"),
    }
    if raw.get("optimizer_contract") != expected_optimizer:
        raise NewHeadDevEvalError("formal optimizer/LR selection contract drifted")
    output_dir = saved_args.get("output_dir")
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise NewHeadDevEvalError("formal checkpoint output_dir is missing")
    if Path(str(raw.get("output_dir", ""))).expanduser().resolve() != (
        Path(output_dir).expanduser().resolve()
    ):
        raise NewHeadDevEvalError("formal checkpoint output binding drifted")

    receipt = _load_json_object(
        partition_binding.receipt_path, label="formal partition receipt"
    )
    train_records = receipt["outputs"][experiment_variant]["train"]
    expected_manifests = [
        {
            "path": train_records[name]["path"],
            "sha256": train_records[name]["sha256"],
            "rows": train_records[name]["rows"],
        }
        for name in MANIFEST_NAMES
    ]
    if raw.get("manifests") != expected_manifests:
        raise NewHeadDevEvalError("formal checkpoint manifest binding drifted")
    expected_budget = {
        "train_rows_per_epoch": FORMAL_NEW_HEAD_TRAIN_ROWS,
        "batch_size": 64,
        "drop_last": True,
        "steps_per_epoch": 4119,
        "dropped_rows_per_epoch": 45,
        "epochs": 3,
        "expected_optimizer_updates": FORMAL_NEW_HEAD_OPTIMIZER_UPDATES,
        "max_train_iters": 0,
    }
    if raw.get("training_budget") != expected_budget:
        raise NewHeadDevEvalError("formal checkpoint training budget drifted")
    expected_runtime = {
        "seed": 42,
        "sampler_seed": 42,
        "loader_seed": 1042,
        "gradient_accumulation_steps": 1,
        "amp": True,
        "num_workers": 4,
        "prefetch_factor": 1,
        "pin_memory": True,
        "persistent_workers": False,
        "allocator": {
            "environment_variable": "PYTORCH_CUDA_ALLOC_CONF",
            "value": "expandable_segments:True",
        },
    }
    if raw.get("runtime") != expected_runtime:
        raise NewHeadDevEvalError("formal checkpoint runtime execution binding drifted")
    return {
        "formal": True,
        "reason": "main_validated_formal_runtime_binding",
        "binding": dict(raw),
    }


def inspect_checkpoint_contract(
    checkpoint_path: Path,
    *,
    cfg: Any,
    config_path: Path,
    binding: PartitionBinding,
    support_binding: SupportBinding,
    experiment_variant: str,
) -> tuple[dict[str, Any], int]:
    checkpoint_path = checkpoint_path.expanduser().resolve(strict=True)
    checkpoint_file = _file_record(checkpoint_path)
    config_file = _file_record(config_path.expanduser().resolve(strict=True))
    payload = _torch_load_compat(str(checkpoint_path), map_location="cpu")
    if not isinstance(payload, Mapping):
        raise NewHeadDevEvalError("checkpoint payload must be a mapping")
    saved_args = payload.get("args")
    if not isinstance(saved_args, Mapping):
        raise NewHeadDevEvalError("checkpoint has no saved args mapping")
    if experiment_variant not in VARIANTS:
        raise NewHeadDevEvalError(
            f"unsupported experiment variant: {experiment_variant!r}"
        )
    if binding.variant != EVALUATION_DATA_VARIANT:
        raise NewHeadDevEvalError(
            "DD0/DD1 comparison must use the shared ordinary-primary dev manifests"
        )
    expected_experiment = "DD0" if experiment_variant == VARIANTS[0] else "DD1"
    required = {
        "stage_b_data_driven_score": True,
        "stage_b_data_driven_experiment_id": expected_experiment,
        "stage_b_data_driven_train_mode": "rank_patch_only",
        "stage_b_data_driven_category_complete": experiment_variant == VARIANTS[1],
        "stage_b_data_driven_confidence_trained": False,
    }
    drift = {
        key: (saved_args.get(key), expected)
        for key, expected in required.items()
        if saved_args.get(key) != expected
    }
    if drift:
        raise NewHeadDevEvalError(f"checkpoint DD0/DD1 metadata drifted: {drift}")
    enabled = {
        key: saved_args.get(key)
        for key in TEACHER_ROUTE_FLAGS
        if saved_args.get(key) is not False
    }
    if enabled:
        raise NewHeadDevEvalError(
            f"checkpoint teacher/legacy routes are not exactly disabled: {enabled}"
        )
    updates = _require_exact_int(
        payload.get("optimizer_updates"),
        label="checkpoint optimizer_updates",
        minimum=1,
    )
    initializer = _validate_initializer_provenance(saved_args)
    training_partition = _training_partition_status(
        saved_args,
        binding,
        experiment_variant=experiment_variant,
        support_binding=support_binding,
    )
    paired_training_contract = _paired_training_contract(saved_args)
    shared_training_provenance = _shared_training_provenance(saved_args)
    evaluation_config_training_contract = (
        _validate_eval_config_training_contract(cfg, saved_args)
    )
    execution_status = _new_head_execution_status(
        saved_args, optimizer_updates=updates
    )
    formal_runtime_binding = _formal_runtime_binding_status(
        saved_args,
        experiment_variant=experiment_variant,
        config_file=config_file,
        training_partition=training_partition,
        partition_binding=binding,
        support_binding=support_binding,
        initializer=initializer,
    )
    if execution_status["formal"]:
        if formal_runtime_binding["formal"] is not True:
            raise NewHeadDevEvalError(
                "complete formal checkpoint lacks a validated runtime binding"
            )
        initializer_payload = _load_initializer_payload_mmap(
            Path(initializer["base_initializer"]["path"])
        )
        frozen_tensor_audit = _audit_frozen_initializer_tensors(
            initializer_payload, payload.get("model")
        )
    else:
        frozen_tensor_audit = {
            "schema": "pivot.stageb.data_driven.a0_frozen_tensor_audit/v1",
            "passed": False,
            "performed": False,
            "reason": "only_complete_formal_checkpoint_requires_tensor_audit",
        }

    cfg_required = {
        "stage_b_data_driven_score": True,
        "stage_b_data_driven_experiment_id": expected_experiment,
        "stage_b_data_driven_train_mode": "rank_patch_only",
        "stage_b_data_driven_category_complete": experiment_variant == VARIANTS[1],
        "stage_b_data_driven_confidence_trained": False,
    }
    cfg_drift = {
        key: (getattr(cfg, key, None), expected)
        for key, expected in cfg_required.items()
        if getattr(cfg, key, None) != expected
    }
    if cfg_drift:
        raise NewHeadDevEvalError(f"evaluation config DD0/DD1 metadata drifted: {cfg_drift}")
    cfg_enabled = {
        key: getattr(cfg, key, None)
        for key in TEACHER_ROUTE_FLAGS
        if getattr(cfg, key, None) is not False
    }
    if cfg_enabled:
        raise NewHeadDevEvalError(
            f"evaluation config teacher/legacy routes are not disabled: {cfg_enabled}"
        )
    return (
        {
            "checkpoint": checkpoint_file,
            "config": config_file,
            "optimizer_updates": updates,
            "experiment_id": expected_experiment,
            "rank_architecture": saved_args.get(
                "stage_b_data_driven_rank_architecture", "absolute_token"
            ),
            "rank_lr": saved_args.get("stage_b_data_driven_rank_lr"),
            "paired_training_contract": paired_training_contract,
            "shared_training_provenance": shared_training_provenance,
            "evaluation_config_training_contract": (
                evaluation_config_training_contract
            ),
            "formal_execution_status": execution_status,
            "formal_runtime_binding_status": formal_runtime_binding,
            "frozen_initializer_tensor_audit": frozen_tensor_audit,
            "initializer_provenance": initializer,
            "training_partition_status": training_partition,
            "training_support_patch_pool_content": (
                training_partition.get("training_support_patch_pool_content")
            ),
            "formal_new_head_partition_evaluation": bool(
                training_partition["formal"]
                and execution_status["formal"]
                and formal_runtime_binding["formal"]
                and frozen_tensor_audit["passed"]
            ),
        },
        updates,
    )


def _model_root(model: Any) -> Any:
    return model.module if hasattr(model, "module") else model


def validate_rank_only_runtime(cfg: Any, model: Any) -> None:
    if getattr(cfg, "stage_b_data_driven_category_gate", None) is not False:
        raise NewHeadDevEvalError("rank-only cfg category gate must be exactly false")
    if getattr(cfg, "num_queries", None) != QUERY_COUNT:
        raise NewHeadDevEvalError(
            f"rank-only cfg must declare num_queries={QUERY_COUNT}"
        )
    root = _model_root(model)
    if getattr(root, "num_queries", None) != QUERY_COUNT:
        raise NewHeadDevEvalError(
            f"rank-only model must expose num_queries={QUERY_COUNT}"
        )
    heads = getattr(root, "stage_b_data_driven_score_heads", None)
    if heads is None or getattr(heads, "category_gate", None) is not False:
        raise NewHeadDevEvalError(
            "rank-only model data-driven category gate must be exactly false"
        )


def _target_scalar_int(target: Mapping[str, Any], key: str) -> int:
    value = target.get(key)
    if not torch.is_tensor(value) or value.numel() != 1:
        raise NewHeadDevEvalError(f"target {key} must be one exact tensor scalar")
    scalar = value.detach().cpu().reshape(-1)[0].item()
    if isinstance(scalar, bool) or not isinstance(scalar, int):
        raise NewHeadDevEvalError(f"target {key} must be an exact integer")
    return int(scalar)


def _validate_target_identity(target: Mapping[str, Any], row: ManifestRow) -> None:
    identity = row.identity_object
    for key in ("image_id", "ann_id", "ref_id", "sent_id"):
        if _target_scalar_int(target, key) != int(identity[key]):
            raise NewHeadDevEvalError(f"runtime target identity drifted at {key}")
    dataset_name = target.get("dataset_name")
    if dataset_name != identity["source"]:
        raise NewHeadDevEvalError("runtime target source identity drifted")


def _listwise_nll(
    score: torch.Tensor, positive: torch.Tensor, *, temperature: float
) -> float | None:
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise NewHeadDevEvalError("rank temperature must be finite and positive")
    if score.dim() != 1 or positive.dtype != torch.bool or positive.shape != score.shape:
        raise NewHeadDevEvalError("listwise score/positive tensors are invalid")
    negative = ~positive
    if not bool(positive.any().item()) or not bool(negative.any().item()):
        return None
    scaled = score.float() / float(temperature)
    value = torch.logsumexp(scaled, dim=0) - torch.logsumexp(
        scaled[positive], dim=0
    )
    result = float(value.item())
    if not math.isfinite(result):
        raise NewHeadDevEvalError("listwise NLL is not finite")
    return result


def evaluate_rank_only_batch(
    outputs: Mapping[str, Any],
    targets: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[ManifestRow],
    *,
    source: str,
    variant: str,
    partition: str,
    temperature: float,
) -> list[dict[str, Any]]:
    batch_size = len(targets)
    if batch_size <= 0 or len(manifest_rows) != batch_size:
        raise NewHeadDevEvalError("batch targets and manifest rows are misaligned")
    rank = outputs.get("stage_b_data_driven_rank_score")
    text_rank = outputs.get("stage_b_data_driven_text_rank_score")
    candidate_mask = outputs.get("stage_b_data_driven_candidate_mask")
    pred_boxes = outputs.get("pred_boxes")
    for name, value in (("rank_score", rank), ("text_rank_score", text_rank)):
        if (
            not torch.is_tensor(value)
            or not value.is_floating_point()
            or tuple(value.shape) != (batch_size, QUERY_COUNT)
            or not bool(torch.isfinite(value).all().item())
        ):
            shape = tuple(value.shape) if torch.is_tensor(value) else type(value).__name__
            raise NewHeadDevEvalError(
                f"{name} must be finite floating {(batch_size, QUERY_COUNT)}, got {shape}"
            )
    if not torch.equal(rank, text_rank):
        raise NewHeadDevEvalError(
            "deployed rank_score is not elementwise equal to text_rank_score"
        )
    if (
        not torch.is_tensor(candidate_mask)
        or candidate_mask.dtype != torch.bool
        or tuple(candidate_mask.shape) != (batch_size, QUERY_COUNT)
        or not bool(candidate_mask.all().item())
    ):
        raise NewHeadDevEvalError("candidate mask must be exact all-true (B,900)")
    if (
        not torch.is_tensor(pred_boxes)
        or not pred_boxes.is_floating_point()
        or tuple(pred_boxes.shape) != (batch_size, QUERY_COUNT, 4)
        or not bool(torch.isfinite(pred_boxes).all().item())
    ):
        raise NewHeadDevEvalError("pred_boxes must be finite floating (B,900,4)")

    records: list[dict[str, Any]] = []
    for batch_index, (target, manifest_row) in enumerate(
        zip(targets, manifest_rows, strict=True)
    ):
        _validate_target_identity(target, manifest_row)
        boxes = target.get("boxes")
        primary = target.get("primary_instance_mask")
        if (
            not torch.is_tensor(boxes)
            or not boxes.is_floating_point()
            or boxes.dim() != 2
            or int(boxes.shape[1]) != 4
            or not bool(torch.isfinite(boxes).all().item())
            or not torch.is_tensor(primary)
            or primary.dtype != torch.bool
            or tuple(primary.reshape(-1).shape) != (int(boxes.shape[0]),)
            or int(primary.sum().item()) != 1
        ):
            raise NewHeadDevEvalError(
                "each target requires finite boxes and one exact primary_instance_mask"
            )
        score = rank[batch_index].detach().float()
        raw_pred_xyxy = box_ops.box_cxcywh_to_xyxy(
            pred_boxes[batch_index].detach().float()
        )
        raw_gt_xyxy = box_ops.box_cxcywh_to_xyxy(
            boxes.detach().to(device=raw_pred_xyxy.device, dtype=torch.float32)[
                primary.reshape(-1)
            ]
        )
        raw_iou = box_ops.box_iou(raw_pred_xyxy, raw_gt_xyxy)[0].reshape(-1)
        if tuple(raw_iou.shape) != (QUERY_COUNT,) or not bool(
            torch.isfinite(raw_iou).all().item()
        ):
            raise NewHeadDevEvalError("unclamped candidate/primary IoU is invalid")
        positive = raw_iou >= 0.5
        positive_count = int(positive.sum().item())
        negative_count = QUERY_COUNT - positive_count
        nll = _listwise_nll(score, positive, temperature=temperature)

        # torch.argmax returns the first index for a tied maximum.
        winner = int(torch.argmax(score).item())
        clamped_iou = box_ops.box_iou(
            raw_pred_xyxy.clamp(0.0, 1.0), raw_gt_xyxy.clamp(0.0, 1.0)
        )[0].reshape(-1)
        top1_iou = float(clamped_iou[winner].item())
        if not math.isfinite(top1_iou):
            raise NewHeadDevEvalError("clamped top-1 IoU is not finite")
        records.append(
            {
                "schema": EXPRESSION_RECORD_SCHEMA,
                "variant": variant,
                "partition": partition,
                "dataset_source": source,
                "identity": manifest_row.identity_object,
                "image_key": {
                    "coco_split": manifest_row.coco_split,
                    "image_id": manifest_row.image_id,
                },
                "winner_query_index": winner,
                "top1_iou_clamped": top1_iou,
                "acc50": bool(top1_iou >= 0.5),
                "listwise_nll": nll,
                "positive_query_count": positive_count,
                "negative_query_count": negative_count,
                "no_positive_query": positive_count == 0,
                "no_negative_query": negative_count == 0,
            }
        )
    return records


def summarize_records(
    records_by_source: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, Any]:
    if set(records_by_source) != set(SOURCE_NAMES):
        raise NewHeadDevEvalError("summary requires exactly the three Ref sources")
    source_summaries: dict[str, Any] = {}
    total_rows = 0
    total_correct = 0
    total_nll: list[float] = []
    for source in SOURCE_NAMES:
        records = list(records_by_source[source])
        if not records:
            raise NewHeadDevEvalError(f"source has no evaluation rows: {source}")
        correct = sum(record.get("acc50") is True for record in records)
        nll_values = [
            float(record["listwise_nll"])
            for record in records
            if record.get("listwise_nll") is not None
        ]
        if any(not math.isfinite(value) for value in nll_values):
            raise NewHeadDevEvalError(f"source has non-finite NLL: {source}")
        summary = {
            "expressions": len(records),
            "correct50": correct,
            "acc50": float(correct / len(records)),
            "listwise_nll_valid_rows": len(nll_values),
            "mean_listwise_nll": (
                float(sum(nll_values) / len(nll_values)) if nll_values else None
            ),
            "no_positive_rows": sum(
                record.get("no_positive_query") is True for record in records
            ),
            "no_negative_rows": sum(
                record.get("no_negative_query") is True for record in records
            ),
        }
        source_summaries[source] = summary
        total_rows += len(records)
        total_correct += correct
        total_nll.extend(nll_values)
    source_acc = [source_summaries[source]["acc50"] for source in SOURCE_NAMES]
    source_nll = [
        source_summaries[source]["mean_listwise_nll"] for source in SOURCE_NAMES
    ]
    valid_source_nll = [value for value in source_nll if value is not None]
    return {
        "sources": source_summaries,
        "macro_ref3_acc50": float(sum(source_acc) / len(source_acc)),
        "macro_ref3_mean_listwise_nll": (
            float(sum(valid_source_nll) / len(valid_source_nll))
            if len(valid_source_nll) == len(SOURCE_NAMES)
            else None
        ),
        "micro": {
            "expressions": total_rows,
            "correct50": total_correct,
            "acc50": float(total_correct / total_rows),
            "listwise_nll_valid_rows": len(total_nll),
            "mean_listwise_nll": (
                float(sum(total_nll) / len(total_nll)) if total_nll else None
            ),
        },
    }


def _jsonl_bytes(records: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical_bytes(record) + b"\n" for record in records)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_evaluation_artifact(
    output_dir: Path,
    *,
    records_by_source: Mapping[str, Sequence[Mapping[str, Any]]],
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise NewHeadDevEvalError(f"refusing to replace output directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    committed = False
    try:
        output_records: dict[str, Any] = {}
        for source in SOURCE_NAMES:
            records = list(records_by_source[source])
            data = _jsonl_bytes(records)
            name = f"{source}.records.jsonl"
            path = temporary / name
            with path.open("xb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            output_records[source] = {
                "path": str(output_dir / name),
                "rows": len(records),
                "size_bytes": len(data),
                "sha256": _sha256_bytes(data),
            }
        payload = dict(summary)
        payload["record_files"] = output_records
        payload["canonical_payload_sha256"] = _sha256_bytes(_canonical_bytes(payload))
        summary_data = json.dumps(
            payload, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True
        ).encode("ascii") + b"\n"
        with (temporary / "summary.json").open("xb") as handle:
            handle.write(summary_data)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(temporary)
        if output_dir.exists():
            raise NewHeadDevEvalError(
                f"refusing concurrent replacement of output directory: {output_dir}"
            )
        os.rename(temporary, output_dir)
        committed = True
        _fsync_directory(output_dir.parent)
        return payload
    finally:
        if not committed and temporary.exists():
            shutil.rmtree(temporary)


def _record_identity_key(record: Mapping[str, Any]) -> bytes:
    identity = record.get("identity")
    if not isinstance(identity, Mapping) or set(identity) != set(IDENTITY_KEYS):
        raise NewHeadDevEvalError("expression record identity drifted")
    return _canonical_bytes([identity[key] for key in IDENTITY_KEYS])


def _validate_paired_records(
    d0_records: Mapping[str, Sequence[Mapping[str, Any]]],
    d1_records: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    if set(d0_records) != set(SOURCE_NAMES) or set(d1_records) != set(SOURCE_NAMES):
        raise NewHeadDevEvalError("paired records require exactly three Ref sources")
    for source in SOURCE_NAMES:
        left = list(d0_records[source])
        right = list(d1_records[source])
        if len(left) != len(right):
            raise NewHeadDevEvalError(f"paired record row count drifted: {source}")
        for index, (d0, d1) in enumerate(zip(left, right, strict=True)):
            if _record_identity_key(d0) != _record_identity_key(d1):
                raise NewHeadDevEvalError(
                    f"paired record identity/order drifted: {source}:{index + 1}"
                )
            for record, variant in ((d0, VARIANTS[0]), (d1, VARIANTS[1])):
                if (
                    record.get("schema") != EXPRESSION_RECORD_SCHEMA
                    or record.get("dataset_source") != source
                    or record.get("variant") != variant
                    or type(record.get("acc50")) is not bool
                ):
                    raise NewHeadDevEvalError(
                        f"paired expression record contract drifted: {source}:{index + 1}"
                    )
                image_key = record.get("image_key")
                if not (
                    isinstance(image_key, Mapping)
                    and isinstance(image_key.get("coco_split"), str)
                    and type(image_key.get("image_id")) is int
                ):
                    raise NewHeadDevEvalError("paired expression image key drifted")
            if d0["image_key"] != d1["image_key"]:
                raise NewHeadDevEvalError("paired expression image key mismatch")


def paired_cluster_bootstrap(
    d0_records: Mapping[str, Sequence[Mapping[str, Any]]],
    d1_records: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    _validate_paired_records(d0_records, d1_records)
    iterations = _require_exact_int(iterations, label="bootstrap iterations", minimum=1)
    if type(seed) is not int:
        raise NewHeadDevEvalError("bootstrap seed must be an exact integer")
    clusters: dict[tuple[str, int], dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(lambda: [0, 0, 0])
    )
    source_totals = {source: [0, 0, 0] for source in SOURCE_NAMES}
    for source in SOURCE_NAMES:
        for d0, d1 in zip(d0_records[source], d1_records[source], strict=True):
            image_key = d0["image_key"]
            key = (str(image_key["coco_split"]), int(image_key["image_id"]))
            values = clusters[key][source]
            values[0] += int(d0["acc50"])
            values[1] += int(d1["acc50"])
            values[2] += 1
            totals = source_totals[source]
            totals[0] += int(d0["acc50"])
            totals[1] += int(d1["acc50"])
            totals[2] += 1
    cluster_keys = sorted(clusters)
    if not cluster_keys or any(source_totals[source][2] <= 0 for source in SOURCE_NAMES):
        raise NewHeadDevEvalError("paired bootstrap has an empty source or cluster set")
    point_by_source = {
        source: float((values[1] - values[0]) / values[2])
        for source, values in source_totals.items()
    }
    point_macro = float(
        sum(point_by_source[source] for source in SOURCE_NAMES) / len(SOURCE_NAMES)
    )
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    samples_by_source = {source: [] for source in SOURCE_NAMES}
    rejected = 0
    attempts = 0
    maximum_attempts = max(iterations * 100, iterations + 1000)
    cluster_count = len(cluster_keys)
    while len(samples) < iterations and attempts < maximum_attempts:
        attempts += 1
        selected = rng.integers(0, cluster_count, size=cluster_count)
        totals = {source: [0, 0, 0] for source in SOURCE_NAMES}
        for cluster_index in selected.tolist():
            cluster = clusters[cluster_keys[cluster_index]]
            for source, values in cluster.items():
                destination = totals[source]
                destination[0] += values[0]
                destination[1] += values[1]
                destination[2] += values[2]
        if any(totals[source][2] == 0 for source in SOURCE_NAMES):
            rejected += 1
            continue
        deltas = {
            source: float(
                (totals[source][1] - totals[source][0]) / totals[source][2]
            )
            for source in SOURCE_NAMES
        }
        for source in SOURCE_NAMES:
            samples_by_source[source].append(deltas[source])
        samples.append(float(sum(deltas.values()) / len(SOURCE_NAMES)))
    if len(samples) != iterations:
        raise NewHeadDevEvalError(
            "could not obtain the requested valid global-image bootstrap replicates"
        )

    def interval(values: Sequence[float]) -> list[float]:
        array = np.asarray(values, dtype=np.float64)
        return [
            float(np.quantile(array, 0.025, method="linear")),
            float(np.quantile(array, 0.975, method="linear")),
        ]

    return {
        "metric": "D1_minus_D0_macro_ref3_acc50",
        "cluster_identity": ["coco_split", "image_id"],
        "global_cluster_resampling": True,
        "seed": seed,
        "iterations": iterations,
        "attempts": attempts,
        "rejected_empty_source_replicates": rejected,
        "unique_image_clusters": cluster_count,
        "point_estimate": point_macro,
        "ci95": interval(samples),
        "by_source": {
            source: {
                "point_estimate": point_by_source[source],
                "ci95": interval(samples_by_source[source]),
            }
            for source in SOURCE_NAMES
        },
    }


def _load_completed_artifact(
    value: Path, *, expected_variant: str
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    summary_path = value.expanduser()
    if summary_path.is_dir():
        summary_path = summary_path / "summary.json"
    summary_path = summary_path.resolve(strict=True)
    summary = _load_json_object(summary_path, label="new-head dev summary")
    if summary.get("schema") != EVALUATION_SUMMARY_SCHEMA:
        raise NewHeadDevEvalError("evaluation summary schema drifted")
    canonical_sha = _require_sha256(
        summary.get("canonical_payload_sha256"), label="evaluation summary canonical hash"
    )
    canonical = dict(summary)
    del canonical["canonical_payload_sha256"]
    if _sha256_bytes(_canonical_bytes(canonical)) != canonical_sha:
        raise NewHeadDevEvalError("evaluation summary canonical hash drifted")
    if summary.get("variant") != expected_variant:
        raise NewHeadDevEvalError("paired evaluation variant drifted")
    record_files = summary.get("record_files")
    if not isinstance(record_files, dict) or set(record_files) != set(SOURCE_NAMES):
        raise NewHeadDevEvalError("evaluation record-file set drifted")
    records_by_source: dict[str, list[dict[str, Any]]] = {}
    for source in SOURCE_NAMES:
        record = record_files[source]
        if not isinstance(record, Mapping):
            raise NewHeadDevEvalError("evaluation record-file binding is invalid")
        path = Path(str(record.get("path", ""))).expanduser().resolve(strict=True)
        observed = _file_record(path)
        if (
            observed["sha256"] != record.get("sha256")
            or observed["size_bytes"] != record.get("size_bytes")
        ):
            raise NewHeadDevEvalError("evaluation record file identity drifted")
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    value_row = json.loads(line)
                except json.JSONDecodeError as error:
                    raise NewHeadDevEvalError(
                        f"invalid expression record JSON: {path}:{line_number}"
                    ) from error
                if not isinstance(value_row, dict):
                    raise NewHeadDevEvalError("expression record is not an object")
                rows.append(value_row)
        if len(rows) != record.get("rows"):
            raise NewHeadDevEvalError("evaluation record row count drifted")
        records_by_source[source] = rows
    return summary, records_by_source


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        raise NewHeadDevEvalError(f"refusing to replace output file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(
        payload, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True
    ).encode("ascii") + b"\n"
    descriptor, temporary_value = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_value)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    except FileExistsError as error:
        raise NewHeadDevEvalError(f"refusing concurrent replacement: {path}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _is_sealed_formal_checkpoint_contract(
    value: Any, *, label: str
) -> bool:
    if not isinstance(value, Mapping):
        return False
    if value.get("formal_new_head_partition_evaluation") is not True:
        return False
    execution = value.get("formal_execution_status")
    runtime_binding = value.get("formal_runtime_binding_status")
    tensor_audit = value.get("frozen_initializer_tensor_audit")
    training = value.get("training_partition_status")
    paired = value.get("paired_training_contract")
    required_execution = {
        "formal": True,
        "reason": "sealed_formal_fresh_run_complete",
        "execution_scope": FORMAL_NEW_HEAD_EXECUTION_SCOPE,
        "formal_fresh_start": True,
        "declared_optimizer_updates": FORMAL_NEW_HEAD_OPTIMIZER_UPDATES,
        "observed_optimizer_updates": FORMAL_NEW_HEAD_OPTIMIZER_UPDATES,
        "formal_contract": FORMAL_NEW_HEAD_CONTRACT,
    }
    if (
        value.get("optimizer_updates") != FORMAL_NEW_HEAD_OPTIMIZER_UPDATES
        or not isinstance(training, Mapping)
        or training.get("formal") is not True
        or not isinstance(execution, Mapping)
        or dict(execution) != required_execution
        or not isinstance(runtime_binding, Mapping)
        or runtime_binding.get("formal") is not True
        or runtime_binding.get("reason")
        != "main_validated_formal_runtime_binding"
        or not isinstance(runtime_binding.get("binding"), Mapping)
        or runtime_binding["binding"].get("schema")
        != FORMAL_NEW_HEAD_BINDING_SCHEMA
        or not isinstance(tensor_audit, Mapping)
        or tensor_audit.get("schema")
        != "pivot.stageb.data_driven.a0_frozen_tensor_audit/v1"
        or tensor_audit.get("passed") is not True
        or tensor_audit.get("frozen_tensor_sha256_equal") is not True
        or not isinstance(paired, Mapping)
        or paired.get("stage_b_data_driven_execution_scope")
        != FORMAL_NEW_HEAD_EXECUTION_SCOPE
        or paired.get("stage_b_data_driven_formal_fresh_start") is not True
        or paired.get(
            "stage_b_data_driven_formal_expected_optimizer_updates"
        )
        != FORMAL_NEW_HEAD_OPTIMIZER_UPDATES
        or paired.get("stage_b_data_driven_new_head_formal_contract")
        != FORMAL_NEW_HEAD_CONTRACT
    ):
        raise NewHeadDevEvalError(
            f"{label} falsely claims a sealed formal new-head evaluation"
        )
    mutable = tensor_audit.get("mutable_roles")
    if not isinstance(mutable, Mapping) or set(mutable) != {
        "absolute_rank",
        "random_patch_projection",
    }:
        raise NewHeadDevEvalError(
            f"{label} formal mutable-role tensor audit is missing"
        )
    for role, record in mutable.items():
        if (
            not isinstance(record, Mapping)
            or type(record.get("changed_tensor_count")) is not int
            or record["changed_tensor_count"] <= 0
            or record.get("checkpoint_tensor_sha256")
            == record.get("initializer_tensor_sha256")
        ):
            raise NewHeadDevEvalError(
                f"{label} formal mutable role did not train: {role}"
            )
    return True


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    binding = load_shared_evaluation_binding(
        args.partition_receipt,
        experiment_variant=args.variant,
        partition=args.partition,
        expected_receipt_sha256=args.partition_receipt_sha256,
    )
    config_path = args.config.expanduser().resolve(strict=True)
    data_root = args.data_root.expanduser().resolve(strict=True)
    support_binding = load_support_binding(
        args.support_receipt,
        expected_receipt_sha256=args.support_receipt_sha256,
        partition_binding=binding,
        canonical_classes_path=data_root / "canonical_classes_with_aliases.json",
    )
    cfg = SLConfig.fromfile(str(config_path))
    device = torch.device(args.device)
    eval_contract = effective_eval_contract(cfg, device)
    checkpoint_contract, updates = inspect_checkpoint_contract(
        args.checkpoint,
        cfg=cfg,
        config_path=config_path,
        binding=binding,
        support_binding=support_binding,
        experiment_variant=args.variant,
    )
    cfg.stage_b_data_driven_category_gate = False
    cfg.stage_b_data_driven_eval_expected_optimizer_updates = updates
    temperature = float(getattr(cfg, "stage_b_data_driven_temperature", 0.0))
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise NewHeadDevEvalError("config rank temperature must be finite and positive")
    query_images = query_image_content_record(binding, data_root=data_root)
    model = _load_model(cfg, str(args.checkpoint), device)
    validate_rank_only_runtime(cfg, model)

    records_by_source: dict[str, list[dict[str, Any]]] = {}
    for source_index, manifest in enumerate(binding.manifests):
        datasetinfo = _make_datasetinfo(
            data_root,
            f"new_head_{manifest.source}_{binding.partition}",
            manifest.path,
        )
        datasetinfo["strict_sample_identity"] = True
        datasetinfo.update(
            support_patch_tsv=support_binding.runtime_tsv["path"],
            support_patch_bucket="clean",
            support_patch_use_embedding=False,
            support_patch_image_root=str(data_root / "patches_quality"),
            support_patch_max_per_class=200,
            patch_bank_cache=False,
            patch_bank_cache_write=False,
            support_min_count=1,
        )
        loader = _build_loader(
            cfg,
            datasetinfo,
            args.batch_size,
            args.num_workers,
            device,
            args.seed + source_index,
        )
        if len(loader.dataset) != len(manifest.rows):
            raise NewHeadDevEvalError(
                f"runtime dataset row count drifted for {manifest.source}"
            )
        source_records: list[dict[str, Any]] = []
        offset = 0
        for batch in loader:
            outputs, targets = _forward(
                model, batch, device, amp=args.amp, cfg=cfg
            )
            batch_size = len(targets)
            rows = manifest.rows[offset : offset + batch_size]
            source_records.extend(
                evaluate_rank_only_batch(
                    outputs,
                    targets,
                    rows,
                    source=manifest.source,
                    variant=args.variant,
                    partition=binding.partition,
                    temperature=temperature,
                )
            )
            offset += batch_size
        if offset != len(manifest.rows):
            raise NewHeadDevEvalError(
                f"runtime iteration lost rows for {manifest.source}"
            )
        records_by_source[manifest.source] = source_records
    metrics = summarize_records(records_by_source)
    summary = {
        "schema": EVALUATION_SUMMARY_SCHEMA,
        "evaluation_scope": EVALUATION_SCOPE,
        "scope_limits": {
            "common_base": "frozen_b58",
            "frozen_b58_may_have_seen_dev_images": True,
            "authorized_selection_uses": [
                "random_new_head_learning_rate",
                "random_new_head_early_stopping",
            ],
            "whole_model_image_disjoint_claim_eligible": False,
        },
        "variant": args.variant,
        "evaluation_manifest_variant": binding.variant,
        "partition": binding.partition,
        "partition_receipt": binding.receipt_file,
        "partition_canonical_payload_sha256": binding.canonical_payload_sha256,
        "evaluation_inputs": {
            "evaluator_code": _file_record(Path(__file__)),
            "data_root": str(data_root),
            "canonical_classes": support_binding.canonical_classes,
            "support_receipt": support_binding.receipt_file,
            "support_receipt_canonical_payload_sha256": (
                support_binding.canonical_payload_sha256
            ),
            "runtime_support_tsv": support_binding.runtime_tsv,
            "support_patch_pool_content": (
                support_binding.support_patch_pool_content
            ),
            "query_image_content": query_images,
        },
        "checkpoint_contract": checkpoint_contract,
        "protocol": {
            "evaluation_scope": EVALUATION_SCOPE,
            "evaluation_manifest_variant": binding.variant,
            "rank_only": True,
            "category_gate": False,
            "query_count": QUERY_COUNT,
            "candidate_mask": "exact_all_true",
            "winner_tie_break": "torch_argmax_first_index",
            "acc50_iou": "clamped_xyxy_primary_instance",
            "listwise_positive_mask": "unclamped_xyxy_iou_ge_0.5",
            "listwise_eligible_mask": "all_900_queries",
            "listwise_temperature": temperature,
            "macro": "equal_weight_refcoco_refcocoplus_refcocog",
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "amp": bool(args.amp),
            "seed": int(args.seed),
            "effective_eval_contract": eval_contract,
        },
        "metrics": metrics,
    }
    return write_evaluation_artifact(
        args.output_dir, records_by_source=records_by_source, summary=summary
    )


def run_bootstrap(args: argparse.Namespace) -> dict[str, Any]:
    d0_summary, d0_records = _load_completed_artifact(
        args.d0, expected_variant=VARIANTS[0]
    )
    d1_summary, d1_records = _load_completed_artifact(
        args.d1, expected_variant=VARIANTS[1]
    )
    for key in (
        "partition",
        "partition_canonical_payload_sha256",
        "evaluation_manifest_variant",
    ):
        if d0_summary.get(key) != d1_summary.get(key):
            raise NewHeadDevEvalError(f"paired evaluation {key} drifted")
    if d0_summary.get("evaluation_manifest_variant") != EVALUATION_DATA_VARIANT:
        raise NewHeadDevEvalError(
            "paired evaluation did not use the shared ordinary-primary dev manifests"
        )
    if (
        d0_summary.get("evaluation_scope") != EVALUATION_SCOPE
        or d1_summary.get("evaluation_scope") != EVALUATION_SCOPE
    ):
        raise NewHeadDevEvalError("paired evaluation scope drifted")
    d0_protocol = d0_summary.get("protocol")
    d1_protocol = d1_summary.get("protocol")
    if (
        not isinstance(d0_protocol, Mapping)
        or not isinstance(d1_protocol, Mapping)
        or dict(d0_protocol) != dict(d1_protocol)
    ):
        raise NewHeadDevEvalError("paired evaluation metric protocol drifted")
    if d0_summary.get("evaluation_inputs") != d1_summary.get("evaluation_inputs"):
        raise NewHeadDevEvalError("paired evaluation input bindings drifted")
    d0_initializer = (
        (d0_summary.get("checkpoint_contract") or {}).get("initializer_provenance")
        or {}
    )
    d1_initializer = (
        (d1_summary.get("checkpoint_contract") or {}).get("initializer_provenance")
        or {}
    )
    for key in ("common_tensor_sha256", "base_initializer", "pair_receipt"):
        left = d0_initializer.get(key)
        right = d1_initializer.get(key)
        if key in {"base_initializer", "pair_receipt"} and isinstance(
            left, Mapping
        ) and isinstance(right, Mapping):
            left = left.get("sha256")
            right = right.get("sha256")
        if left != right:
            raise NewHeadDevEvalError("D0/D1 do not share one common initializer")
    d0_contract = d0_summary.get("checkpoint_contract") or {}
    d1_contract = d1_summary.get("checkpoint_contract") or {}
    if d0_contract.get("optimizer_updates") != d1_contract.get("optimizer_updates"):
        raise NewHeadDevEvalError("D0/D1 optimizer update counts differ")
    if d0_contract.get("paired_training_contract") != d1_contract.get(
        "paired_training_contract"
    ):
        raise NewHeadDevEvalError("D0/D1 paired training contracts differ")
    if d0_contract.get("shared_training_provenance") != d1_contract.get(
        "shared_training_provenance"
    ):
        raise NewHeadDevEvalError(
            "D0/D1 shared training code/runtime provenance differs"
        )
    d0_formal = _is_sealed_formal_checkpoint_contract(d0_contract, label="D0")
    d1_formal = _is_sealed_formal_checkpoint_contract(d1_contract, label="D1")
    both_formal = d0_formal and d1_formal
    allow_nonformal = bool(getattr(args, "allow_nonformal", False))
    if not both_formal and not allow_nonformal:
        raise NewHeadDevEvalError(
            "paired comparison requires two formal new-head partition evaluations"
        )
    result = paired_cluster_bootstrap(
        d0_records,
        d1_records,
        iterations=args.iterations,
        seed=args.seed,
    )
    payload = {
        "schema": BOOTSTRAP_SCHEMA,
        "evaluation_scope": EVALUATION_SCOPE,
        "d0_summary": _file_record(
            (args.d0 / "summary.json") if args.d0.is_dir() else args.d0
        ),
        "d1_summary": _file_record(
            (args.d1 / "summary.json") if args.d1.is_dir() else args.d1
        ),
        "partition": d0_summary["partition"],
        "partition_canonical_payload_sha256": d0_summary[
            "partition_canonical_payload_sha256"
        ],
        "both_formal_new_head_partition_evaluations": both_formal,
        "nonformal_override": allow_nonformal,
        "bootstrap": result,
    }
    payload["canonical_payload_sha256"] = _sha256_bytes(_canonical_bytes(payload))
    _atomic_write_json(args.output, payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    evaluate = subparsers.add_parser("evaluate", help="run one DD0/DD1 dev evaluation")
    evaluate.add_argument(
        "--partition-receipt", type=Path, default=DEFAULT_PARTITION_RECEIPT
    )
    evaluate.add_argument(
        "--partition-receipt-sha256",
        default=EXPECTED_PARTITION_RECEIPT_SHA256,
    )
    evaluate.add_argument("--variant", choices=VARIANTS, required=True)
    evaluate.add_argument("--partition", choices=DEV_PARTITIONS, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--data-root", type=Path, required=True)
    evaluate.add_argument(
        "--support-receipt", type=Path, default=DEFAULT_SUPPORT_RECEIPT
    )
    evaluate.add_argument(
        "--support-receipt-sha256",
        default=EXPECTED_SUPPORT_RECEIPT_SHA256,
    )
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--batch-size", type=int, default=16)
    evaluate.add_argument("--num-workers", type=int, default=4)
    evaluate.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    evaluate.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)

    bootstrap = subparsers.add_parser(
        "paired-bootstrap", help="compare completed paired DD0/DD1 artifacts"
    )
    bootstrap.add_argument("--d0", type=Path, required=True)
    bootstrap.add_argument("--d1", type=Path, required=True)
    bootstrap.add_argument("--output", type=Path, required=True)
    bootstrap.add_argument("--iterations", type=int, default=DEFAULT_BOOTSTRAP_ITERATIONS)
    bootstrap.add_argument("--seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    bootstrap.add_argument(
        "--allow-nonformal",
        action="store_true",
        help="diagnostic only: allow artifacts not trained on the sealed partition",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "evaluate":
        result = run_evaluation(args)
    else:
        result = run_bootstrap(args)
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
