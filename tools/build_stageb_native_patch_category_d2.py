#!/usr/bin/env python3
"""Build deterministic full-expression Stage-B native-patch D2 data.

D2 consumes only the sealed category-complete Stage-B GT partition and the
sealed train-filtered support bank.  It accounts for every input expression,
quarantines train images whose bytes duplicate dev_full, writes one manifest
per RefCOCO source, and records the exact sampling weights needed for a 2:2:1
source mix with capped square-root class balancing.

No model, teacher, checkpoint, embedding, or score output is an input.
"""

from __future__ import annotations

import argparse
import copy
import ctypes
import errno
import hashlib
import json
import math
import os
import shutil
import struct
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools import build_stageb_native_patch_category_d1 as _d1


REPO_ROOT = Path(__file__).resolve().parents[1]
PARTITION_RECEIPT = _d1.PARTITION_RECEIPT
SUPPORT_RECEIPT = _d1.SUPPORT_RECEIPT
SUPPORT_TSV = _d1.SUPPORT_TSV
COCO_IMAGE_ROOT = _d1.COCO_IMAGE_ROOT
OUTPUT_ROOT = (
    REPO_ROOT / "data/ablations/stageb_native_patch_category_d2_20260724"
)

RECEIPT_SCHEMA = "pivot.stageb.native_patch_category_d2_receipt/v1"
ROW_SCHEMA = "pivot.stageb.native_patch_category_d2_row/v1"
QUARANTINE_ROW_SCHEMA = (
    "pivot.stageb.native_patch_category_d2_content_overlap_quarantine_row/v1"
)
UPSTREAM_PARTITION_SCHEMA = _d1.UPSTREAM_PARTITION_SCHEMA
UPSTREAM_SUPPORT_SCHEMA = _d1.UPSTREAM_SUPPORT_SCHEMA
UPSTREAM_ROW_SCHEMA = _d1.UPSTREAM_ROW_SCHEMA
STREAM_ENCODING = _d1.STREAM_ENCODING

SPLITS = _d1.SPLITS
SOURCE_MANIFESTS = _d1.SOURCE_MANIFESTS
SUPPORT_COLUMNS = _d1.SUPPORT_COLUMNS
SOURCES = tuple(source for _manifest, source in SOURCE_MANIFESTS)
SOURCE_BY_MANIFEST = dict(SOURCE_MANIFESTS)
SOURCE_MIX_WEIGHTS = {
    "refcoco": 2,
    "refcocoplus": 2,
    "refcocog": 1,
}
SOURCE_MIX_DENOMINATOR = sum(SOURCE_MIX_WEIGHTS.values())
OUTPUT_FILES = {
    split: {
        source: f"{split}_{source}.jsonl"
        for source in SOURCES
    }
    for split in SPLITS
}
QUARANTINE_FILE = "train_content_overlap_quarantine.jsonl"

CLASS_BALANCE_CAP = 4.0
SAMPLING_WEIGHT_FIELD = "native_patch_category_sampling_weight"
SAMPLING_CONTRACT = (
    "source_mix_2_2_1_group_dedup_capped_sqrt_class_v1"
)
GROUP_NAMESPACE = "pivot.stageb.native_patch_category_d2.group/v1"
SUPPORT_ROTATION_NAMESPACE = (
    "pivot.stageb.native_patch_category_d2.support_rotation/v1"
)

EXPECTED_PARTITION_RECEIPT_SHA256 = _d1.EXPECTED_PARTITION_RECEIPT_SHA256
EXPECTED_SUPPORT_RECEIPT_SHA256 = _d1.EXPECTED_SUPPORT_RECEIPT_SHA256
EXPECTED_SUPPORT_TSV_SHA256 = _d1.EXPECTED_SUPPORT_TSV_SHA256

LEGACY_OUTPUT_KEYS = frozenset(
    {
        "stage_b_u2_category_complete",
        "stage_b_u2_category_complete_schema",
        "stage_b_native_patch_category_d1",
        "stage_b_native_patch_category_d1_schema",
    }
)

GroupKey = _d1.GroupKey
ImageKey = _d1.ImageKey
SupportCandidate = _d1.SupportCandidate


class NativePatchCategoryD2Error(RuntimeError):
    """The requested D2 artifact violates its sealed data contract."""


@dataclass(frozen=True, slots=True)
class ExpressionRef:
    partition: str
    manifest: str
    source_dataset: str
    line_number: int
    raw_sha256: str
    identity_sha256: str
    full_text: str
    ann_id: int
    ref_id: int
    sent_id: int

    @property
    def key(self) -> tuple[str, str, int]:
        return (self.partition, self.manifest, self.line_number)


@dataclass(slots=True)
class GroupState:
    key: GroupKey
    group_id: str
    filename: str
    query_path: Path
    class_id: int
    instance_set_sha256: str
    instance_count: int
    expressions: dict[str, list[ExpressionRef]] = field(
        default_factory=lambda: defaultdict(list)
    )


@dataclass(frozen=True, slots=True)
class SamplingRow:
    weight: float
    raw_weight: float
    group_expression_count: int
    class_group_count: int
    source_max_class_group_count: int
    class_balance_multiplier: float


@dataclass(frozen=True, slots=True)
class BuildPlan:
    receipt: dict[str, Any]
    outputs: Mapping[str, bytes]
    sealed_files: Mapping[str, Mapping[str, Any]]
    content_identities: Mapping[str, tuple[int, int, int, int]]


_canonical_bytes = _d1._canonical_bytes
_sha256_bytes = _d1._sha256_bytes
_sha256_file = _d1._sha256_file
_file_record = _d1._file_record
_file_record_with_identity = _d1._file_record_with_identity
_record_stream_sha256 = _d1._record_stream_sha256
_image_key_text = _d1._image_key_text
_group_key_text = _d1._group_key_text


def _validate_exact_cap(value: Any) -> float:
    if type(value) not in (int, float):
        raise NativePatchCategoryD2Error(
            "class balance cap must be an exact finite number"
        )
    cap = float(value)
    if not math.isfinite(cap) or cap != CLASS_BALANCE_CAP:
        raise NativePatchCategoryD2Error(
            f"D2 requires class_balance_cap={CLASS_BALANCE_CAP}"
        )
    return cap


def _source_for_manifest(manifest: str) -> str:
    try:
        return SOURCE_BY_MANIFEST[manifest]
    except KeyError as error:
        raise NativePatchCategoryD2Error(
            f"unsupported D2 source manifest: {manifest}"
        ) from error


def _group_id(group_key: GroupKey) -> str:
    return _sha256_bytes(
        _canonical_bytes(
            {
                "namespace": GROUP_NAMESPACE,
                "group": _group_key_text(group_key),
            }
        )
    )


def _expression_signature(group: GroupState) -> tuple[Any, ...]:
    return (
        group.class_id,
        group.instance_set_sha256,
        group.instance_count,
        tuple(
            (
                source,
                tuple(
                    sorted(
                        (
                            ref.identity_sha256,
                            ref.raw_sha256,
                            ref.full_text,
                            ref.ann_id,
                            ref.ref_id,
                            ref.sent_id,
                        )
                        for ref in group.expressions.get(source, ())
                    )
                ),
            )
            for source in SOURCES
        ),
    )


def _scan_manifest(
    *,
    partition: str,
    manifest: str,
    source_dataset: str,
    path: Path,
    expected_rows: int,
    coco_image_root: Path,
    groups: dict[GroupKey, GroupState],
) -> tuple[int, set[ImageKey]]:
    rows = 0
    images: set[ImageKey] = set()
    identities: set[str] = set()
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            row = _d1._load_jsonl_row(raw, path=path, line_number=line_number)
            context = f"{partition}/{manifest}:{line_number}"
            (
                group_key,
                _basename,
                query_path,
                class_id,
                instance_set_sha,
                instance_count,
                full_text,
            ) = _d1._validate_category_row(
                row, context=context, coco_image_root=coco_image_root
            )
            row_source = _d1._required_text(
                row.get("source"), field="source", context=context
            ).casefold()
            if not row_source.startswith(_d1.SOURCE_ROW_PREFIXES[source_dataset]):
                raise NativePatchCategoryD2Error(
                    f"{context}: source row does not match manifest dataset"
                )

            identity_sha = _d1._source_identity(row, context=context)
            if identity_sha in identities:
                raise NativePatchCategoryD2Error(
                    f"{context}: duplicate source identity in one manifest"
                )
            identities.add(identity_sha)

            group = groups.get(group_key)
            if group is None:
                group = GroupState(
                    key=group_key,
                    group_id=_group_id(group_key),
                    filename=str(row["filename"]),
                    query_path=query_path,
                    class_id=class_id,
                    instance_set_sha256=instance_set_sha,
                    instance_count=instance_count,
                )
                groups[group_key] = group
            elif (
                group.class_id != class_id
                or group.instance_set_sha256 != instance_set_sha
                or group.instance_count != instance_count
                or group.query_path != query_path
            ):
                raise NativePatchCategoryD2Error(
                    f"{context}: physical group has inconsistent instance coverage"
                )

            group.expressions[source_dataset].append(
                ExpressionRef(
                    partition=partition,
                    manifest=manifest,
                    source_dataset=source_dataset,
                    line_number=line_number,
                    raw_sha256=_sha256_bytes(raw),
                    identity_sha256=identity_sha,
                    full_text=full_text,
                    ann_id=_d1._required_int(
                        row.get("ann_id"), field="ann_id", context=context
                    ),
                    ref_id=_d1._required_int(
                        row.get("ref_id"), field="ref_id", context=context
                    ),
                    sent_id=_d1._required_int(
                        row.get("sent_id"), field="sent_id", context=context
                    ),
                )
            )
            images.add((group_key[0], group_key[1]))
            rows += 1
    if rows != expected_rows:
        raise NativePatchCategoryD2Error(
            f"{partition}/{manifest}: row count drifted: {rows} != {expected_rows}"
        )
    return rows, images


def _sampling_weights_for_groups(
    groups: Mapping[GroupKey, GroupState],
    *,
    cap: float = CLASS_BALANCE_CAP,
) -> tuple[dict[tuple[str, str, int], SamplingRow], dict[str, Any]]:
    """Return mean-one per-source weights for the training split."""
    cap = _validate_exact_cap(cap)
    weights: dict[tuple[str, str, int], SamplingRow] = {}
    summary: dict[str, Any] = {}
    for source in SOURCES:
        source_groups = [
            group for group in groups.values() if group.expressions.get(source)
        ]
        if not source_groups:
            raise NativePatchCategoryD2Error(
                f"D2 train source {source} has no physical groups"
            )
        class_group_counts = Counter(group.class_id for group in source_groups)
        max_class_groups = max(class_group_counts.values())
        raw_by_key: dict[tuple[str, str, int], tuple[float, int, int, float]] = {}
        for group in source_groups:
            refs = group.expressions[source]
            expression_count = len(refs)
            if expression_count <= 0:
                raise NativePatchCategoryD2Error(
                    "sampling group unexpectedly has no expressions"
                )
            class_count = class_group_counts[group.class_id]
            multiplier = min(
                cap, math.sqrt(float(max_class_groups) / float(class_count))
            )
            raw_weight = multiplier / float(expression_count)
            for ref in refs:
                if ref.key in raw_by_key:
                    raise NativePatchCategoryD2Error(
                        "sampling key is duplicated within one source"
                    )
                raw_by_key[ref.key] = (
                    raw_weight,
                    expression_count,
                    class_count,
                    multiplier,
                )
        raw_sum = math.fsum(value[0] for value in raw_by_key.values())
        row_count = len(raw_by_key)
        if row_count <= 0 or not math.isfinite(raw_sum) or raw_sum <= 0.0:
            raise NativePatchCategoryD2Error(
                f"D2 train source {source} has invalid sampling mass"
            )
        normalization = float(row_count) / raw_sum
        normalized_values: list[float] = []
        for key, (raw_weight, expression_count, class_count, multiplier) in (
            raw_by_key.items()
        ):
            normalized = raw_weight * normalization
            if not math.isfinite(normalized) or normalized <= 0.0:
                raise NativePatchCategoryD2Error(
                    "D2 sampling produced a non-positive weight"
                )
            weights[key] = SamplingRow(
                weight=normalized,
                raw_weight=raw_weight,
                group_expression_count=expression_count,
                class_group_count=class_count,
                source_max_class_group_count=max_class_groups,
                class_balance_multiplier=multiplier,
            )
            normalized_values.append(normalized)
        normalized_sum = math.fsum(normalized_values)
        summary[source] = {
            "rows": row_count,
            "groups": len(source_groups),
            "classes": len(class_group_counts),
            "class_group_counts": {
                str(class_id): class_group_counts[class_id]
                for class_id in sorted(class_group_counts)
            },
            "max_class_group_count": max_class_groups,
            "class_balance_cap": cap,
            "raw_weight_sum": raw_sum,
            "normalization": normalization,
            "normalized_weight_sum": normalized_sum,
            "normalized_weight_mean": normalized_sum / float(row_count),
            "normalized_weight_min": min(normalized_values),
            "normalized_weight_max": max(normalized_values),
            "mix_weight": SOURCE_MIX_WEIGHTS[source],
            "expected_mix_fraction": (
                SOURCE_MIX_WEIGHTS[source] / SOURCE_MIX_DENOMINATOR
            ),
        }
    return weights, summary


def _support_rotation_key(group_id: str, source_identity_sha256: str) -> str:
    return _sha256_bytes(
        _canonical_bytes(
            {
                "namespace": SUPPORT_ROTATION_NAMESPACE,
                "group_id": group_id,
                "source_identity_sha256": source_identity_sha256,
            }
        )
    )


def _select_support_witness(
    *,
    group: GroupState,
    source_ref: ExpressionRef,
    candidates: Sequence[SupportCandidate],
    query_record: Mapping[str, Any],
    forbidden_query_content_sha256: frozenset[str],
    support_receipt_sha256: str,
    content_cache: dict[Path, dict[str, Any]],
    content_identities: dict[str, tuple[int, int, int, int]],
) -> dict[str, Any]:
    if not candidates:
        raise NativePatchCategoryD2Error(
            f"support pool is empty for class {group.class_id}"
        )
    rotation_key = _support_rotation_key(
        group.group_id, source_ref.identity_sha256
    )
    start = int(rotation_key[:16], 16) % len(candidates)
    for offset in range(len(candidates)):
        index = (start + offset) % len(candidates)
        candidate = candidates[index]
        if candidate.class_id != group.class_id:
            raise NativePatchCategoryD2Error("support class grouping drifted")
        if candidate.coco_id == group.key[1]:
            continue
        content = _d1._content_record(
            candidate.path,
            cache=content_cache,
            identities=content_identities,
        )
        if content["sha256"] == query_record["sha256"]:
            continue
        if content["sha256"] in forbidden_query_content_sha256:
            continue
        return {
            "candidate_id": candidate.candidate_id,
            "class_assignment": candidate.class_assignment,
            "class_id": candidate.class_id,
            "coco_id": candidate.coco_id,
            "content_sha256": content["sha256"],
            "path": content["path"],
            "rotation_key_sha256": rotation_key,
            "rotation_start_index": start,
            "rotation_selected_index": index,
            "rotation_offset": offset,
            "rotation_pool_size": len(candidates),
            "selection_contract": SUPPORT_ROTATION_NAMESPACE,
            "size_bytes": content["size_bytes"],
            "source": candidate.source,
            "source_cache_class_id": candidate.source_cache_class_id,
            "source_class": candidate.source_class,
            "source_image_id": candidate.source_image_id,
            "source_image_identity": candidate.image_identity,
            "source_row_number": candidate.source_row_number,
            "source_row_sha256": candidate.source_row_sha256,
            "support_partition_receipt_sha256": support_receipt_sha256,
            "train_filtered": True,
        }
    raise NativePatchCategoryD2Error(
        "no train-filtered, same-class, different-image/content support "
        f"for group {_group_key_text(group.key)}"
    )


def _make_plan_impl(
    *,
    partition_receipt: Path = PARTITION_RECEIPT,
    support_receipt: Path = SUPPORT_RECEIPT,
    support_tsv: Path = SUPPORT_TSV,
    coco_image_root: Path = COCO_IMAGE_ROOT,
    output_root: Path = OUTPUT_ROOT,
    expected_partition_receipt_sha256: str | None = (
        EXPECTED_PARTITION_RECEIPT_SHA256
    ),
    expected_support_receipt_sha256: str | None = (
        EXPECTED_SUPPORT_RECEIPT_SHA256
    ),
    expected_support_tsv_sha256: str | None = EXPECTED_SUPPORT_TSV_SHA256,
    class_balance_cap: float = CLASS_BALANCE_CAP,
    splits: Sequence[str] = SPLITS,
) -> BuildPlan:
    if tuple(splits) != SPLITS:
        raise NativePatchCategoryD2Error(
            "D2 split order must be train/dev_screen/dev_full"
        )
    cap = _validate_exact_cap(class_balance_cap)
    output_root = output_root.expanduser().resolve()
    coco_image_root = coco_image_root.expanduser().resolve(strict=True)

    partition, partition_record = _d1._validate_receipt(
        partition_receipt,
        label="new-head partition receipt",
        expected_schema=UPSTREAM_PARTITION_SCHEMA,
        expected_sha256=expected_partition_receipt_sha256,
    )
    support, support_record = _d1._validate_receipt(
        support_receipt,
        label="support partition receipt",
        expected_schema=UPSTREAM_SUPPORT_SCHEMA,
        expected_sha256=expected_support_receipt_sha256,
    )
    support_partition_binding = support.get("inputs", {}).get(
        "partition_receipt"
    )
    if (
        not isinstance(support_partition_binding, Mapping)
        or support_partition_binding.get("sha256") != partition_record["sha256"]
        or support_partition_binding.get("size_bytes")
        != partition_record["size_bytes"]
    ):
        raise NativePatchCategoryD2Error(
            "support bank is not bound to the selected new-head partition"
        )
    if expected_support_tsv_sha256 is not None:
        _d1._validate_sha256(
            expected_support_tsv_sha256,
            label="expected support TSV hash",
        )
        if _sha256_file(support_tsv) != expected_support_tsv_sha256:
            raise NativePatchCategoryD2Error(
                "support TSV preregistered hash drifted"
            )

    official_keys, official_records = _d1._official_image_keys(partition)
    partition_outputs = partition.get("outputs", {}).get("d1_category_complete")
    manifest_order = partition.get("source_manifest_order")
    expected_manifest_order = [name for name, _source in SOURCE_MANIFESTS]
    if (
        not isinstance(partition_outputs, Mapping)
        or manifest_order != expected_manifest_order
    ):
        raise NativePatchCategoryD2Error(
            "D2 category-complete partition manifest order drifted"
        )

    builder_record = _file_record(Path(__file__))
    dependency_record = _file_record(Path(_d1.__file__))
    sealed_files: dict[str, dict[str, Any]] = {
        "d2_builder": builder_record,
        "d1_validation_dependency": dependency_record,
        "partition_receipt": partition_record,
        "support_receipt": support_record,
    }
    for name, record in official_records.items():
        sealed_files[f"official_ref8:{name}"] = record

    manifest_paths: dict[tuple[str, str], Path] = {}
    manifest_records: dict[str, dict[str, dict[str, Any]]] = {}
    input_row_counts: dict[str, dict[str, int]] = {}
    groups_by_split: dict[str, dict[GroupKey, GroupState]] = {}
    image_sets: dict[str, set[ImageKey]] = {}
    for split in SPLITS:
        split_outputs = partition_outputs.get(split)
        if not isinstance(split_outputs, Mapping):
            raise NativePatchCategoryD2Error(
                f"partition receipt lost category-complete {split} outputs"
            )
        groups: dict[GroupKey, GroupState] = {}
        images: set[ImageKey] = set()
        manifest_records[split] = {}
        input_row_counts[split] = {}
        for manifest in manifest_order:
            source = _source_for_manifest(manifest)
            entry = split_outputs.get(manifest)
            path, record = _d1._bound_record(
                entry, label=f"D2 {split}/{manifest}"
            )
            expected_rows = entry.get("rows") if isinstance(entry, Mapping) else None
            if type(expected_rows) is not int or expected_rows < 0:
                raise NativePatchCategoryD2Error(
                    f"D2 {split}/{manifest} row count is invalid"
                )
            observed_rows, observed_images = _scan_manifest(
                partition=split,
                manifest=manifest,
                source_dataset=source,
                path=path,
                expected_rows=expected_rows,
                coco_image_root=coco_image_root,
                groups=groups,
            )
            manifest_paths[(split, manifest)] = path
            manifest_records[split][manifest] = {
                **record,
                "rows": observed_rows,
            }
            input_row_counts[split][manifest] = observed_rows
            images.update(observed_images)
            sealed_files[f"category_complete:{split}:{manifest}"] = record

        if images & set(official_keys):
            raise NativePatchCategoryD2Error(
                f"D2 {split} contains an official Ref8 image"
            )
        summary = partition.get("partition_summary", {}).get(split)
        image_stream = _record_stream_sha256(
            [_image_key_text(key) for key in sorted(images)]
        )
        if (
            not isinstance(summary, Mapping)
            or summary.get("rows") != sum(input_row_counts[split].values())
            or summary.get("rows_by_manifest") != input_row_counts[split]
            or summary.get("unique_image_keys") != len(images)
            or summary.get("ordered_image_key_stream_sha256") != image_stream
        ):
            raise NativePatchCategoryD2Error(
                f"D2 {split} does not replay partition summary"
            )
        groups_by_split[split] = groups
        image_sets[split] = images

    dev_full_members = _d1._partition_member_keys(
        partition, field_name="dev_full_members"
    )
    dev_screen_members = _d1._partition_member_keys(
        partition, field_name="dev_screen_members"
    )
    if dev_full_members != image_sets["dev_full"]:
        raise NativePatchCategoryD2Error(
            "dev_full outputs do not match explicit partition members"
        )
    if dev_screen_members != image_sets["dev_screen"]:
        raise NativePatchCategoryD2Error(
            "dev_screen outputs do not match explicit partition members"
        )
    if image_sets["train"] & image_sets["dev_full"]:
        raise NativePatchCategoryD2Error("D2 train and dev_full images overlap")
    if not image_sets["dev_screen"].issubset(image_sets["dev_full"]):
        raise NativePatchCategoryD2Error(
            "D2 dev_screen is not nested in dev_full"
        )
    if not set(groups_by_split["dev_screen"]).issubset(
        groups_by_split["dev_full"]
    ):
        raise NativePatchCategoryD2Error(
            "D2 dev_screen groups are not nested in dev_full"
        )
    for group_key, screen_group in groups_by_split["dev_screen"].items():
        full_group = groups_by_split["dev_full"][group_key]
        if _expression_signature(screen_group) != _expression_signature(full_group):
            raise NativePatchCategoryD2Error(
                "dev_screen group content/source rows differ from dev_full: "
                f"{_group_key_text(group_key)}"
            )

    required_classes = frozenset(
        group.class_id
        for groups in groups_by_split.values()
        for group in groups.values()
    )
    support_by_class, support_summary = _d1._load_support_candidates(
        support_tsv=support_tsv,
        support_receipt=support,
        required_classes=required_classes,
    )
    expected_exclusion_ids = {
        image_id
        for _split, image_id in (
            set(image_sets["dev_full"]) | set(official_keys)
        )
    }
    observed_exclusion_ids = set(
        support_summary["excluded_numeric_coco_ids"]
    )
    if observed_exclusion_ids != expected_exclusion_ids:
        raise NativePatchCategoryD2Error(
            "support exclusion union does not equal dev_full plus official Ref8"
        )
    support_tsv_record = support_summary["record"]
    sealed_files["support_tsv"] = support_tsv_record

    ref_lookup: dict[tuple[str, str, int], tuple[GroupState, ExpressionRef]] = {}
    for split, groups in groups_by_split.items():
        for group in groups.values():
            for source in SOURCES:
                for ref in group.expressions.get(source, ()):
                    if ref.key in ref_lookup:
                        raise NativePatchCategoryD2Error(
                            "one source row belongs to multiple physical groups"
                        )
                    ref_lookup[ref.key] = (group, ref)
    if len(ref_lookup) != sum(
        sum(counts.values()) for counts in input_row_counts.values()
    ):
        raise NativePatchCategoryD2Error(
            "D2 did not index every input expression exactly once"
        )

    content_cache: dict[Path, dict[str, Any]] = {}
    content_identities: dict[str, tuple[int, int, int, int]] = {}
    query_record_by_group: dict[tuple[str, GroupKey], dict[str, Any]] = {}
    query_hashes_by_split: dict[str, set[str]] = {
        split: set() for split in SPLITS
    }
    all_query_paths: set[str] = set()
    for split, groups in groups_by_split.items():
        for group_key, group in groups.items():
            record = _d1._content_record(
                group.query_path,
                cache=content_cache,
                identities=content_identities,
            )
            query_record_by_group[(split, group_key)] = record
            query_hashes_by_split[split].add(record["sha256"])
            all_query_paths.add(record["path"])
    content_overlap_hashes = frozenset(
        query_hashes_by_split["train"] & query_hashes_by_split["dev_full"]
    )
    quarantined_train_group_keys = frozenset(
        group_key
        for group_key in groups_by_split["train"]
        if query_record_by_group[("train", group_key)]["sha256"]
        in content_overlap_hashes
    )
    eligible_train_groups = {
        group_key: group
        for group_key, group in groups_by_split["train"].items()
        if group_key not in quarantined_train_group_keys
    }
    sampling_rows, sampling_summary = _sampling_weights_for_groups(
        eligible_train_groups, cap=cap
    )
    dev_image_keys_by_content: dict[str, list[str]] = defaultdict(list)
    for group_key in groups_by_split["dev_full"]:
        content_sha = query_record_by_group[("dev_full", group_key)]["sha256"]
        if content_sha in content_overlap_hashes:
            dev_image_keys_by_content[content_sha].append(
                _image_key_text((group_key[0], group_key[1]))
            )
    for values in dev_image_keys_by_content.values():
        values.sort()
    all_query_hashes = frozenset(
        value
        for hashes in query_hashes_by_split.values()
        for value in hashes
    )

    output_parts: dict[str, list[bytes]] = {
        filename: []
        for by_source in OUTPUT_FILES.values()
        for filename in by_source.values()
    }
    output_parts[QUARANTINE_FILE] = []
    source_receipts: dict[str, dict[str, Any]] = {
        split: {} for split in SPLITS
    }
    selected_support_paths: set[str] = set()
    selected_support_hashes: set[str] = set()
    quarantine_row_hashes: list[str] = []
    quarantine_source_hashes: list[str] = []
    quarantine_identity_hashes: list[str] = []
    quarantine_group_ids: list[str] = []
    quarantine_source_counts: Counter[str] = Counter()

    for split in SPLITS:
        for manifest in manifest_order:
            source = _source_for_manifest(manifest)
            filename = OUTPUT_FILES[split][source]
            row_hashes: list[str] = []
            source_hashes: list[str] = []
            identity_hashes: list[str] = []
            support_ids: list[str] = []
            support_hashes: list[str] = []
            query_hashes: list[str] = []
            group_ids: list[str] = []
            group_keys_seen: set[GroupKey] = set()
            class_ids: set[int] = set()
            expression_histogram: Counter[int] = Counter()
            weight_values: list[float] = []
            path = manifest_paths[(split, manifest)]
            with path.open("rb") as handle:
                for line_number, raw in enumerate(handle, start=1):
                    lookup_key = (split, manifest, line_number)
                    indexed = ref_lookup.get(lookup_key)
                    if indexed is None:
                        raise NativePatchCategoryD2Error(
                            f"D2 source row was not indexed: {lookup_key}"
                        )
                    group, source_ref = indexed
                    if _sha256_bytes(raw) != source_ref.raw_sha256:
                        raise NativePatchCategoryD2Error(
                            f"source row changed during D2 replay: {lookup_key}"
                        )
                    row = _d1._load_jsonl_row(
                        raw, path=path, line_number=line_number
                    )
                    if _d1._source_identity(
                        row, context=f"D2 replay {lookup_key}"
                    ) != source_ref.identity_sha256:
                        raise NativePatchCategoryD2Error(
                            f"source identity changed during D2 replay: {lookup_key}"
                        )
                    query_record = query_record_by_group[(split, group.key)]
                    if (
                        split == "train"
                        and group.key in quarantined_train_group_keys
                    ):
                        quarantined = copy.deepcopy(row)
                        for key in LEGACY_OUTPUT_KEYS:
                            quarantined.pop(key, None)
                        quarantined.update(
                            {
                                "native_patch_category_class_id": group.class_id,
                                "native_patch_category_group_id": group.group_id,
                                "native_patch_category_source_dataset": source,
                                "native_patch_category_source_identity_sha256": (
                                    source_ref.identity_sha256
                                ),
                                "native_patch_category_source_line_number": (
                                    line_number
                                ),
                                "native_patch_category_source_manifest": manifest,
                                "native_patch_category_source_row_sha256": (
                                    source_ref.raw_sha256
                                ),
                                "native_patch_category_quarantine_reason": (
                                    "train_query_content_sha256_overlaps_dev_full"
                                ),
                                "native_patch_category_conflicting_dev_image_keys": (
                                    dev_image_keys_by_content[
                                        query_record["sha256"]
                                    ]
                                ),
                                "query_image_witness": {
                                    "content_sha256": query_record["sha256"],
                                    "path": query_record["path"],
                                    "size_bytes": query_record["size_bytes"],
                                    "source_filename": group.filename,
                                },
                                "stage_b_native_patch_category_d2_content_overlap_quarantine": True,
                                "stage_b_native_patch_category_d2_content_overlap_quarantine_schema": (
                                    QUARANTINE_ROW_SCHEMA
                                ),
                            }
                        )
                        _d1._reject_forbidden_keys(
                            quarantined, context="emitted D2 quarantine row"
                        )
                        if any(key in quarantined for key in LEGACY_OUTPUT_KEYS):
                            raise NativePatchCategoryD2Error(
                                "D2 quarantine retained a legacy U2/D1 marker"
                            )
                        encoded_quarantine = (
                            _canonical_bytes(quarantined) + b"\n"
                        )
                        output_parts[QUARANTINE_FILE].append(
                            encoded_quarantine
                        )
                        quarantine_row_hashes.append(
                            _sha256_bytes(encoded_quarantine)
                        )
                        quarantine_source_hashes.append(
                            source_ref.raw_sha256
                        )
                        quarantine_identity_hashes.append(
                            source_ref.identity_sha256
                        )
                        quarantine_group_ids.append(group.group_id)
                        quarantine_source_counts[source] += 1
                        continue
                    support_witness = _select_support_witness(
                        group=group,
                        source_ref=source_ref,
                        candidates=support_by_class[group.class_id],
                        query_record=query_record,
                        forbidden_query_content_sha256=all_query_hashes,
                        support_receipt_sha256=support_record["sha256"],
                        content_cache=content_cache,
                        content_identities=content_identities,
                    )

                    emitted = copy.deepcopy(row)
                    for key in LEGACY_OUTPUT_KEYS:
                        emitted.pop(key, None)
                    expression_count = len(group.expressions[source])
                    emitted.update(
                        {
                            "native_patch_category_class_id": group.class_id,
                            "native_patch_category_group_id": group.group_id,
                            "native_patch_category_source_dataset": source,
                            "native_patch_category_source_identity_sha256": (
                                source_ref.identity_sha256
                            ),
                            "native_patch_category_source_line_number": line_number,
                            "native_patch_category_source_manifest": manifest,
                            "native_patch_category_source_row_sha256": (
                                source_ref.raw_sha256
                            ),
                            "native_patch_category_source_group_expression_count": (
                                expression_count
                            ),
                            "native_patch_category_source_mix_weight": (
                                SOURCE_MIX_WEIGHTS[source]
                            ),
                            "native_patch_category_sampling_contract": (
                                SAMPLING_CONTRACT
                            ),
                            "query_image_witness": {
                                "content_sha256": query_record["sha256"],
                                "path": query_record["path"],
                                "size_bytes": query_record["size_bytes"],
                                "source_filename": group.filename,
                            },
                            "stage_b_native_patch_category_d2": True,
                            "stage_b_native_patch_category_d2_schema": ROW_SCHEMA,
                            "support_patch_witness": support_witness,
                        }
                    )
                    if split == "train":
                        sampling = sampling_rows.get(source_ref.key)
                        if sampling is None:
                            raise NativePatchCategoryD2Error(
                                "train expression has no D2 sampling weight"
                            )
                        emitted.update(
                            {
                                SAMPLING_WEIGHT_FIELD: sampling.weight,
                                "native_patch_category_sampling_raw_weight": (
                                    sampling.raw_weight
                                ),
                                "native_patch_category_class_group_count": (
                                    sampling.class_group_count
                                ),
                                "native_patch_category_source_max_class_group_count": (
                                    sampling.source_max_class_group_count
                                ),
                                "native_patch_category_class_balance_multiplier": (
                                    sampling.class_balance_multiplier
                                ),
                            }
                        )
                        weight_values.append(sampling.weight)
                    _d1._reject_forbidden_keys(
                        emitted, context="emitted D2 row"
                    )
                    if any(key in emitted for key in LEGACY_OUTPUT_KEYS):
                        raise NativePatchCategoryD2Error(
                            "emitted D2 row retained a legacy U2/D1 marker"
                        )
                    encoded = _canonical_bytes(emitted) + b"\n"
                    output_parts[filename].append(encoded)
                    row_hashes.append(_sha256_bytes(encoded))
                    source_hashes.append(source_ref.raw_sha256)
                    identity_hashes.append(source_ref.identity_sha256)
                    support_ids.append(support_witness["candidate_id"])
                    support_hashes.append(support_witness["content_sha256"])
                    query_hashes.append(query_record["sha256"])
                    group_ids.append(group.group_id)
                    group_keys_seen.add(group.key)
                    class_ids.add(group.class_id)
                    selected_support_paths.add(support_witness["path"])
                    selected_support_hashes.add(
                        support_witness["content_sha256"]
                    )

            for group_key in group_keys_seen:
                expression_histogram[
                    len(groups_by_split[split][group_key].expressions[source])
                ] += 1
            expected_rows = input_row_counts[split][manifest]
            if split == "train":
                expected_rows -= quarantine_source_counts[source]
            if len(row_hashes) != expected_rows:
                raise NativePatchCategoryD2Error(
                    f"D2 {split}/{source} did not preserve every eligible expression"
                )
            payload = b"".join(output_parts[filename])
            output_parts[filename] = [payload]
            output_record = _d1._predicted_output(
                output_root / filename, payload, rows=len(row_hashes)
            )
            source_receipts[split][source] = {
                "input_rows": input_row_counts[split][manifest],
                "quarantined_rows": (
                    quarantine_source_counts[source] if split == "train" else 0
                ),
                "rows": len(row_hashes),
                "groups": len(group_keys_seen),
                "classes": len(class_ids),
                "mix_weight": SOURCE_MIX_WEIGHTS[source],
                "expected_mix_fraction": (
                    SOURCE_MIX_WEIGHTS[source] / SOURCE_MIX_DENOMINATOR
                ),
                "group_expression_count_histogram": {
                    str(count): expression_histogram[count]
                    for count in sorted(expression_histogram)
                },
                "sampling_weight_sum": (
                    math.fsum(weight_values) if split == "train" else None
                ),
                "sampling_weight_mean": (
                    math.fsum(weight_values) / len(weight_values)
                    if weight_values
                    else None
                ),
                "sampling_weight_float64_le_stream_sha256": (
                    hashlib.sha256(
                        b"".join(
                            struct.pack("<d", value) for value in weight_values
                        )
                    ).hexdigest()
                    if split == "train"
                    else None
                ),
                "ordered_group_id_stream_sha256": _record_stream_sha256(
                    group_ids
                ),
                "ordered_output_row_sha256_stream_sha256": (
                    _record_stream_sha256(row_hashes)
                ),
                "ordered_source_row_sha256_stream_sha256": (
                    _record_stream_sha256(source_hashes)
                ),
                "ordered_source_identity_sha256_stream_sha256": (
                    _record_stream_sha256(identity_hashes)
                ),
                "ordered_support_candidate_id_stream_sha256": (
                    _record_stream_sha256(support_ids)
                ),
                "ordered_support_content_sha256_stream_sha256": (
                    _record_stream_sha256(support_hashes)
                ),
                "ordered_query_content_sha256_stream_sha256": (
                    _record_stream_sha256(query_hashes)
                ),
                "output": output_record,
            }

    quarantine_payload = b"".join(output_parts[QUARANTINE_FILE])
    output_parts[QUARANTINE_FILE] = [quarantine_payload]
    quarantine_output_record = _d1._predicted_output(
        output_root / QUARANTINE_FILE,
        quarantine_payload,
        rows=len(quarantine_row_hashes),
    )
    quarantine_receipt = {
        "schema": QUARANTINE_ROW_SCHEMA,
        "reason": "train_query_content_sha256_overlaps_dev_full",
        "rows": len(quarantine_row_hashes),
        "groups": len(set(quarantine_group_ids)),
        "source_rows": {
            source: quarantine_source_counts[source] for source in SOURCES
        },
        "overlap_content_sha256": sorted(content_overlap_hashes),
        "overlap_content_sha256_count": len(content_overlap_hashes),
        "train_image_keys": sorted(
            _image_key_text((group_key[0], group_key[1]))
            for group_key in quarantined_train_group_keys
        ),
        "dev_image_keys_by_content_sha256": {
            content_sha: dev_image_keys_by_content[content_sha]
            for content_sha in sorted(dev_image_keys_by_content)
        },
        "ordered_group_id_stream_sha256": _record_stream_sha256(
            quarantine_group_ids
        ),
        "ordered_output_row_sha256_stream_sha256": _record_stream_sha256(
            quarantine_row_hashes
        ),
        "ordered_source_row_sha256_stream_sha256": _record_stream_sha256(
            quarantine_source_hashes
        ),
        "ordered_source_identity_sha256_stream_sha256": _record_stream_sha256(
            quarantine_identity_hashes
        ),
        "output": quarantine_output_record,
    }

    outputs = {
        filename: parts[0]
        for filename, parts in output_parts.items()
    }
    dev_screen_group_ids = {
        group.group_id for group in groups_by_split["dev_screen"].values()
    }
    dev_full_group_ids = {
        group.group_id for group in groups_by_split["dev_full"].values()
    }
    all_input_rows_accounted = all(
        source_receipts[split][source]["rows"]
        + (
            quarantine_source_counts[source]
            if split == "train"
            else 0
        )
        == input_row_counts[split][manifest]
        for split in SPLITS
        for manifest, source in SOURCE_MANIFESTS
    )
    eligible_train_image_keys = {
        (group_key[0], group_key[1]) for group_key in eligible_train_groups
    }
    quarantined_train_image_keys = {
        (group_key[0], group_key[1])
        for group_key in quarantined_train_group_keys
    }
    eligible_train_query_hashes = {
        query_record_by_group[("train", group_key)]["sha256"]
        for group_key in eligible_train_groups
    }
    expected_quarantine_group_ids = {
        groups_by_split["train"][group_key].group_id
        for group_key in quarantined_train_group_keys
    }
    train_weight_means_are_one = all(
        math.isclose(
            float(source_receipts["train"][source]["sampling_weight_mean"]),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for source in SOURCES
    )

    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "row_schema": ROW_SCHEMA,
        "quarantine_row_schema": QUARANTINE_ROW_SCHEMA,
        "builder": builder_record,
        "builder_dependencies": {
            "d1_gt_receipt_validation_helpers": dependency_record,
        },
        "inputs": {
            "new_head_partition_receipt": partition_record,
            "support_partition_receipt": support_record,
            "runtime_support_tsv": support_tsv_record,
            "category_complete_manifests": manifest_records,
            "official_ref8_manifests": official_records,
        },
        "full_expression_contract": {
            "input_partition": "d1_category_complete_gt_only",
            "all_input_source_rows_accounted_exactly_once": True,
            "eligible_rows_preserved_in_split_outputs": True,
            "ineligible_train_rows_preserved_in_content_overlap_quarantine": True,
            "source_output_order": list(SOURCES),
            "row_order": "upstream_manifest_line_order_v1",
            "group_key": [
                "category_complete_coco_split",
                "image_id",
                "category_complete_coco_category_id",
            ],
            "group_namespace": GROUP_NAMESPACE,
            "legacy_u2_and_d1_runtime_markers_removed": sorted(
                LEGACY_OUTPUT_KEYS
            ),
            "d2_runtime_marker": "stage_b_native_patch_category_d2",
            "content_overlap_quarantine_file": QUARANTINE_FILE,
        },
        "sampling_contract": {
            "name": SAMPLING_CONTRACT,
            "training_source_manifests_are_separate": True,
            "source_mix_weights": dict(SOURCE_MIX_WEIGHTS),
            "source_mix_denominator": SOURCE_MIX_DENOMINATOR,
            "expected_source_fractions": {
                source: SOURCE_MIX_WEIGHTS[source] / SOURCE_MIX_DENOMINATOR
                for source in SOURCES
            },
            "group_expression_deduplication": (
                "divide_each_row_by_source_group_expression_count"
            ),
            "class_group_count_unit": (
                "distinct_coco_split_image_id_category_id_group"
            ),
            "class_multiplier_formula": (
                "min(cap,sqrt(source_max_class_group_count/class_group_count))"
            ),
            "row_raw_weight_formula": (
                "class_multiplier/source_group_expression_count"
            ),
            "per_source_normalization": "arithmetic_row_mean_equals_one",
            "class_balance_cap": cap,
            "sampler_requirement": (
                "weighted_replacement_with_deterministic_epoch_ledger_v1"
            ),
            "weight_field": SAMPLING_WEIGHT_FIELD,
            "train_sources": sampling_summary,
        },
        "support_contract": {
            "candidate_source": "sealed_train_filtered_runtime_support_tsv",
            "selection_namespace": SUPPORT_ROTATION_NAMESPACE,
            "selection_key": ["group_id", "source_identity_sha256"],
            "selection_method": (
                "sha256_start_then_circular_first_nonleaking_candidate_v1"
            ),
            "same_canonical_class_id": True,
            "different_query_image_identity": True,
            "different_all_output_query_content_sha256": True,
            "known_coco_leakage_exclusion": (
                "dev_full_union_official_ref8_numeric_coco_id"
            ),
            "runtime_embeddings_consumed": False,
        },
        "provenance_contract": {
            "model_score_free": True,
            "teacher_score_free": True,
            "checkpoint_output_free": True,
            "embedding_free": True,
            "stage_a_tensor_free": True,
            "u2_p50_r100_tensor_free": True,
            "forbidden_source_keys": sorted(_d1.FORBIDDEN_SOURCE_KEYS),
            "forbidden_source_key_prefixes": list(
                _d1.FORBIDDEN_OWNER_PREFIXES
            ),
        },
        "split_relationships": {
            "train_images": len(image_sets["train"]),
            "eligible_train_images": len(eligible_train_image_keys),
            "quarantined_train_images": len(quarantined_train_image_keys),
            "dev_screen_images": len(image_sets["dev_screen"]),
            "dev_full_images": len(image_sets["dev_full"]),
            "official_ref8_images": len(official_keys),
            "train_dev_full_image_overlap": 0,
            "detected_raw_train_dev_full_content_sha256_overlap": len(
                content_overlap_hashes
            ),
            "eligible_train_dev_full_content_sha256_overlap": len(
                eligible_train_query_hashes
                & query_hashes_by_split["dev_full"]
            ),
            "train_official_ref8_image_overlap": 0,
            "dev_full_official_ref8_image_overlap": 0,
            "dev_screen_is_nested_in_dev_full": True,
            "dev_screen_groups_are_nested_in_dev_full": True,
            "dev_screen_group_content_and_all_expressions_equal_dev_full": True,
            "dev_screen_group_count": len(dev_screen_group_ids),
            "dev_full_group_count": len(dev_full_group_ids),
        },
        "splits": source_receipts,
        "content_overlap_quarantine": quarantine_receipt,
        "content_binding": {
            "unique_query_image_paths": len(all_query_paths),
            "unique_query_content_sha256": len(all_query_hashes),
            "unique_selected_support_paths": len(selected_support_paths),
            "unique_selected_support_content_sha256": len(
                selected_support_hashes
            ),
            "sorted_query_path_stream_sha256": _record_stream_sha256(
                sorted(all_query_paths)
            ),
            "sorted_query_content_sha256_stream_sha256": (
                _record_stream_sha256(sorted(all_query_hashes))
            ),
            "sorted_support_path_stream_sha256": _record_stream_sha256(
                sorted(selected_support_paths)
            ),
            "sorted_support_content_sha256_stream_sha256": (
                _record_stream_sha256(sorted(selected_support_hashes))
            ),
        },
        "invariants": {
            "partition_receipt_hash_and_payload_replay": True,
            "support_receipt_hash_and_payload_replay": True,
            "all_category_complete_manifests_are_content_hash_bound": True,
            "partition_rows_and_image_streams_replay": True,
            "official_ref8_manifests_are_content_hash_bound": True,
            "runtime_support_tsv_is_content_hash_bound": True,
            "support_exclusion_replays_dev_full_plus_official_ref8": True,
            "all_input_expressions_are_accounted_exactly_once": (
                all_input_rows_accounted
            ),
            "quarantine_groups_equal_detected_content_overlap_groups": (
                set(quarantine_group_ids) == expected_quarantine_group_ids
            ),
            "three_training_source_manifests_enable_exact_2_2_1_mix": True,
            "train_sampling_weight_mean_is_one_per_source": (
                train_weight_means_are_one
            ),
            "sampling_uses_distinct_physical_group_class_counts": True,
            "support_selection_is_bound_to_group_and_source_identity": True,
            "every_support_matches_query_canonical_class": True,
            "every_support_is_train_filtered": True,
            "every_support_differs_from_all_output_query_content": not (
                selected_support_hashes & set(all_query_hashes)
            ),
            "source_rows_contain_no_model_derived_fields": True,
            "outputs_contain_no_legacy_u2_or_d1_runtime_marker": True,
            "official_ref8_images_are_excluded_from_all_outputs": True,
            "train_and_dev_full_images_are_disjoint": not (
                image_sets["train"] & image_sets["dev_full"]
            ),
            "eligible_train_and_dev_full_content_is_disjoint": not (
                eligible_train_query_hashes
                & query_hashes_by_split["dev_full"]
            ),
            "dev_screen_images_are_nested_in_dev_full": image_sets[
                "dev_screen"
            ].issubset(image_sets["dev_full"]),
            "dev_screen_groups_are_nested_in_dev_full": (
                dev_screen_group_ids.issubset(dev_full_group_ids)
            ),
            "dev_screen_group_content_and_all_expressions_equal_dev_full": True,
        },
    }
    if any(value is not True for value in receipt["invariants"].values()):
        failed = sorted(
            key for key, value in receipt["invariants"].items() if value is not True
        )
        raise NativePatchCategoryD2Error(
            f"one or more D2 invariants failed: {failed}"
        )
    receipt["canonical_payload_sha256"] = _sha256_bytes(
        _canonical_bytes(receipt)
    )
    return BuildPlan(
        receipt=receipt,
        outputs=outputs,
        sealed_files=sealed_files,
        content_identities=content_identities,
    )


def make_plan(**kwargs: Any) -> BuildPlan:
    try:
        return _make_plan_impl(**kwargs)
    except NativePatchCategoryD2Error:
        raise
    except _d1.NativePatchCategoryD1Error as error:
        raise NativePatchCategoryD2Error(str(error)) from error


def _receipt_bytes(receipt: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            receipt,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise NativePatchCategoryD2Error(
            "atomic create-new publish requires renameat2(RENAME_NOREPLACE)"
        )
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise NativePatchCategoryD2Error(
            f"refusing concurrent overwrite of output root: {destination}"
        )
    raise OSError(error_number, os.strerror(error_number), str(destination))


def _assert_inputs_unchanged(plan: BuildPlan) -> None:
    for label, sealed in plan.sealed_files.items():
        observed = _file_record(Path(str(sealed["path"])))
        if observed != dict(sealed):
            raise NativePatchCategoryD2Error(
                f"{label} changed before atomic commit"
            )
    for raw_path, identity in plan.content_identities.items():
        path = Path(raw_path)
        if _d1._stat_identity(path.stat()) != identity:
            raise NativePatchCategoryD2Error(
                f"content-bound image changed before atomic commit: {path}"
            )


def build(**kwargs: Any) -> dict[str, Any]:
    output_root = Path(kwargs.get("output_root", OUTPUT_ROOT)).expanduser().resolve()
    if output_root.exists():
        raise NativePatchCategoryD2Error(
            f"refusing to replace existing output root: {output_root}"
        )
    plan = make_plan(**kwargs)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.tmp-", dir=str(output_root.parent)
        )
    )
    committed = False
    try:
        for filename, payload in sorted(plan.outputs.items()):
            path = temporary_root / filename
            with path.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        receipt_path = temporary_root / "receipt.json"
        with receipt_path.open("xb") as handle:
            handle.write(_receipt_bytes(plan.receipt))
            handle.flush()
            os.fsync(handle.fileno())
        _assert_inputs_unchanged(plan)
        _fsync_directory(temporary_root)
        if output_root.exists():
            raise NativePatchCategoryD2Error(
                f"refusing concurrent overwrite of output root: {output_root}"
            )
        _rename_noreplace(temporary_root, output_root)
        committed = True
        _fsync_directory(output_root.parent)
        return plan.receipt
    finally:
        if not committed and temporary_root.exists():
            shutil.rmtree(temporary_root)


def verify(**kwargs: Any) -> dict[str, Any]:
    output_root = Path(kwargs.get("output_root", OUTPUT_ROOT)).expanduser().resolve(
        strict=True
    )
    if not output_root.is_dir():
        raise NativePatchCategoryD2Error(
            f"output root is not a directory: {output_root}"
        )
    expected_entries = {"receipt.json", QUARANTINE_FILE} | {
        filename
        for by_source in OUTPUT_FILES.values()
        for filename in by_source.values()
    }
    observed_entries = {entry.name for entry in output_root.iterdir()}
    if observed_entries != expected_entries:
        raise NativePatchCategoryD2Error(
            "D2 artifact directory entry set does not replay exactly: "
            f"expected={sorted(expected_entries)}, "
            f"observed={sorted(observed_entries)}"
        )
    plan = make_plan(**kwargs)
    for filename, expected in plan.outputs.items():
        try:
            observed = (output_root / filename).read_bytes()
        except OSError as error:
            raise NativePatchCategoryD2Error(
                f"could not read D2 output {filename}: {error}"
            ) from error
        if observed != expected:
            raise NativePatchCategoryD2Error(
                f"D2 output {filename} does not replay byte-for-byte"
            )
    if (output_root / "receipt.json").read_bytes() != _receipt_bytes(plan.receipt):
        raise NativePatchCategoryD2Error(
            "D2 receipt does not replay byte-for-byte"
        )
    return plan.receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition-receipt", type=Path, default=PARTITION_RECEIPT)
    parser.add_argument("--support-receipt", type=Path, default=SUPPORT_RECEIPT)
    parser.add_argument("--support-tsv", type=Path, default=SUPPORT_TSV)
    parser.add_argument("--coco-image-root", type=Path, default=COCO_IMAGE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--expected-partition-receipt-sha256",
        default=EXPECTED_PARTITION_RECEIPT_SHA256,
    )
    parser.add_argument(
        "--expected-support-receipt-sha256",
        default=EXPECTED_SUPPORT_RECEIPT_SHA256,
    )
    parser.add_argument(
        "--expected-support-tsv-sha256",
        default=EXPECTED_SUPPORT_TSV_SHA256,
    )
    parser.add_argument(
        "--class-balance-cap", type=float, default=CLASS_BALANCE_CAP
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kwargs = {
        "partition_receipt": args.partition_receipt,
        "support_receipt": args.support_receipt,
        "support_tsv": args.support_tsv,
        "coco_image_root": args.coco_image_root,
        "output_root": args.output_root,
        "expected_partition_receipt_sha256": (
            args.expected_partition_receipt_sha256
        ),
        "expected_support_receipt_sha256": (
            args.expected_support_receipt_sha256
        ),
        "expected_support_tsv_sha256": args.expected_support_tsv_sha256,
        "class_balance_cap": args.class_balance_cap,
    }
    if args.verify:
        receipt = verify(**kwargs)
    elif args.dry_run:
        receipt = make_plan(**kwargs).receipt
    else:
        receipt = build(**kwargs)
    print(json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
