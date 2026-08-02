#!/usr/bin/env python3
"""Build deterministic category-grouped Stage-B native-patch D1 data.

The builder reuses the sealed new-head image partition and its category-complete
RefCOCO rows.  It emits exactly three full-text variants for every physical
COCO ``(split, image_id, category_id)`` group and binds an explicit support
patch witness to every row.  Selection depends only on immutable data identity
and SHA-256 priorities; model, teacher, checkpoint, and embedding outputs are
not inputs.
"""

from __future__ import annotations

import argparse
import copy
import csv
import ctypes
import errno
import hashlib
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


PARTITION_RECEIPT = (
    REPO_ROOT
    / "data/ablations/stageb_data_driven_new_head_partition_20260723/receipt.json"
)
SUPPORT_RECEIPT = (
    REPO_ROOT
    / "data/ablations/stageb_data_driven_support_partition_20260723/receipt.json"
)
SUPPORT_TSV = (
    REPO_ROOT
    / "data/ablations/stageb_data_driven_support_partition_20260723/filtered_support.tsv"
)
COCO_IMAGE_ROOT = Path("/media/haoyi/T9/data/COCO/coco2014")
OUTPUT_ROOT = (
    REPO_ROOT / "data/ablations/stageb_native_patch_category_d1_20260724"
)

RECEIPT_SCHEMA = "pivot.stageb.native_patch_category_d1_receipt/v1"
ROW_SCHEMA = "pivot.stageb.native_patch_category_d1_row/v1"
UPSTREAM_PARTITION_SCHEMA = "pivot.stageb.data_driven.new_head_partition_receipt/v1"
UPSTREAM_SUPPORT_SCHEMA = "pivot.stageb.data_driven.support_partition_receipt/v1"
UPSTREAM_ROW_SCHEMA = "pivot.stageb.u2_category_complete_ref/v1"
SELECTION_NAMESPACE = "pivot.stageb.native_patch_category_d1.selection/v1"
SUPPORT_SELECTION_NAMESPACE = (
    "pivot.stageb.native_patch_category_d1.support_selection/v1"
)
STREAM_ENCODING = "utf8_canonical_json_record_plus_lf_v1"
K_VARIANTS = 3
SPLITS = ("train", "dev_screen", "dev_full")
OUTPUT_FILES = {
    "train": "train.jsonl",
    "dev_screen": "dev_screen.jsonl",
    "dev_full": "dev_full.jsonl",
}
SOURCE_MANIFESTS = (
    ("refcoco_stageb_phrase_v1.jsonl", "refcoco"),
    ("refcocoplus_stageb_phrase_v1.jsonl", "refcocoplus"),
    ("refcocog_stageb_phrase_v1.jsonl", "refcocog"),
)
PREFERRED_SOURCES = tuple(source for _name, source in SOURCE_MANIFESTS)
SOURCE_ROW_PREFIXES = {
    "refcoco": ("refcoco_",),
    "refcocoplus": ("refcoco+_", "refcocoplus_"),
    "refcocog": ("refcocog_",),
}
EXPECTED_PARTITION_RECEIPT_SHA256 = (
    "56de31d883ed137f3f9332c34de846839d82c0724120f42a49c5c1c302f38506"
)
EXPECTED_SUPPORT_RECEIPT_SHA256 = (
    "a0e6632182bc7c01ac6e6997b15f1f96e0fbb0bf6dd9d1e3fd8485ad39a6da62"
)
EXPECTED_SUPPORT_TSV_SHA256 = (
    "a3c7dc02e1159ebac5196ccb2c53da1e1bd7e2c2b0322159efcf4178a53a1d37"
)

SUPPORT_COLUMNS = (
    "path",
    "class_id",
    "source_cache_class_id",
    "class_assignment",
    "source",
    "source_image_id",
    "coco_id",
    "source_class",
    "source_row_number",
    "source_row_sha256",
)
FORBIDDEN_SOURCE_KEYS = frozenset(
    {
        "teacher_score",
        "teacher_scores",
        "teacher_logit",
        "teacher_logits",
        "teacher_probability",
        "teacher_probabilities",
        "teacher_label",
        "teacher_labels",
        "teacher_query",
        "teacher_queries",
        "model_score",
        "model_scores",
        "model_logit",
        "model_logits",
        "model_probability",
        "model_probabilities",
        "model_label",
        "model_labels",
        "model_output",
        "model_outputs",
        "checkpoint_score",
        "checkpoint_scores",
        "checkpoint_logit",
        "checkpoint_logits",
        "checkpoint_label",
        "checkpoint_labels",
        "checkpoint_output",
        "checkpoint_outputs",
    }
)
FORBIDDEN_OWNER_PREFIXES = ("teacher", "model", "checkpoint")

_HEX64_RE = re.compile(r"[0-9a-f]{64}")
_COCO_FILENAME_RE = re.compile(
    r"^COCO_(?P<split>train2014|val2014)_(?P<image_id>[0-9]{12})"
    r"\.(?:jpg|jpeg)$",
    re.IGNORECASE,
)

GroupKey = tuple[str, int, int]
ImageKey = tuple[str, int]


class NativePatchCategoryD1Error(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SourceRef:
    partition: str
    source_dataset: str
    manifest: str
    line_number: int
    raw_sha256: str
    identity_sha256: str
    priority_sha256: str
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
    source_row_counts: Counter[str] = field(default_factory=Counter)
    candidates: dict[str, list[SourceRef]] = field(default_factory=dict)
    source_stream_hashers: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SupportCandidate:
    path: Path
    class_id: int
    source_cache_class_id: int
    class_assignment: str
    source: str
    source_image_id: int
    coco_id: int | None
    source_class: str
    source_row_number: int
    source_row_sha256: str
    candidate_id: str

    @property
    def image_identity(self) -> str:
        if self.coco_id is not None:
            return f"coco_numeric_id:{self.coco_id}"
        return f"{self.source}_external_id:{self.source_image_id}"


@dataclass(frozen=True, slots=True)
class SelectedRef:
    source: SourceRef
    mode: str


@dataclass(frozen=True, slots=True)
class BuildPlan:
    receipt: dict[str, Any]
    outputs: Mapping[str, bytes]
    sealed_files: Mapping[str, Mapping[str, Any]]
    content_identities: Mapping[str, tuple[int, int, int, int]]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stat_identity(stat: os.stat_result) -> tuple[int, int, int, int]:
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)


def _file_record_with_identity(
    path: Path,
) -> tuple[dict[str, Any], tuple[int, int, int, int]]:
    path = path.expanduser().resolve(strict=True)
    if not path.is_file():
        raise NativePatchCategoryD1Error(f"not a file: {path}")
    before = path.stat()
    digest = _sha256_file(path)
    after = path.stat()
    if _stat_identity(before) != _stat_identity(after):
        raise NativePatchCategoryD1Error(f"file changed while hashing: {path}")
    return (
        {
            "path": str(path),
            "size_bytes": int(after.st_size),
            "sha256": digest,
        },
        _stat_identity(after),
    )


def _file_record(path: Path) -> dict[str, Any]:
    return _file_record_with_identity(path)[0]


def _validate_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _HEX64_RE.fullmatch(value) is None:
        raise NativePatchCategoryD1Error(f"{label} is not a lowercase SHA-256")
    return value


def _required_int(value: Any, *, field: str, context: str) -> int:
    if type(value) is not int:
        raise NativePatchCategoryD1Error(
            f"{context}: {field} must be an exact integer"
        )
    return int(value)


def _required_text(value: Any, *, field: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NativePatchCategoryD1Error(f"{context}: {field} must be non-empty text")
    return value.strip()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NativePatchCategoryD1Error(
            f"could not load {label}: {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise NativePatchCategoryD1Error(f"{label} must be a JSON object")
    return value


def _validate_receipt(
    path: Path,
    *,
    label: str,
    expected_schema: str,
    expected_sha256: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _file_record(path)
    if expected_sha256 is not None:
        _validate_sha256(expected_sha256, label=f"expected {label} hash")
        if record["sha256"] != expected_sha256:
            raise NativePatchCategoryD1Error(
                f"{label} SHA-256 mismatch: expected={expected_sha256}, "
                f"observed={record['sha256']}"
            )
    receipt = _load_json(path, label=label)
    if receipt.get("schema") != expected_schema:
        raise NativePatchCategoryD1Error(f"{label} schema drifted")
    stored_payload = _validate_sha256(
        receipt.get("canonical_payload_sha256"),
        label=f"{label} canonical payload",
    )
    payload = dict(receipt)
    del payload["canonical_payload_sha256"]
    if _sha256_bytes(_canonical_bytes(payload)) != stored_payload:
        raise NativePatchCategoryD1Error(f"{label} canonical payload does not replay")
    invariants = receipt.get("invariants")
    if (
        not isinstance(invariants, dict)
        or not invariants
        or any(value is not True for value in invariants.values())
    ):
        raise NativePatchCategoryD1Error(f"{label} invariants are not all true")
    return receipt, record


def _load_jsonl_row(raw: bytes, *, path: Path, line_number: int) -> dict[str, Any]:
    if not raw.endswith(b"\n"):
        raise NativePatchCategoryD1Error(
            f"JSONL row lacks terminating LF at {path}:{line_number}"
        )
    if not raw.rstrip(b"\r\n"):
        raise NativePatchCategoryD1Error(f"blank JSONL row at {path}:{line_number}")
    try:
        row = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise NativePatchCategoryD1Error(
            f"invalid JSON at {path}:{line_number}: {error}"
        ) from error
    if not isinstance(row, dict):
        raise NativePatchCategoryD1Error(
            f"JSONL row is not an object at {path}:{line_number}"
        )
    return row


def _reject_forbidden_keys(value: Any, *, context: str, path: str = "$") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise NativePatchCategoryD1Error(
                    f"{context}: non-text object key at {path}"
                )
            folded_key = key.casefold()
            if folded_key in FORBIDDEN_SOURCE_KEYS or folded_key.startswith(
                FORBIDDEN_OWNER_PREFIXES
            ):
                raise NativePatchCategoryD1Error(
                    f"{context}: forbidden model-derived field at {path}.{key}"
                )
            _reject_forbidden_keys(nested, context=context, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_forbidden_keys(
                nested, context=context, path=f"{path}[{index}]"
            )


def _image_key(row: Mapping[str, Any], *, context: str) -> tuple[ImageKey, str]:
    image_id = _required_int(row.get("image_id"), field="image_id", context=context)
    filename = _required_text(row.get("filename"), field="filename", context=context)
    basename = Path(filename).name
    match = _COCO_FILENAME_RE.fullmatch(basename)
    if match is None:
        raise NativePatchCategoryD1Error(
            f"{context}: filename is not canonical COCO 2014: {filename}"
        )
    split = match.group("split").lower()
    filename_id = int(match.group("image_id"))
    if filename_id != image_id:
        raise NativePatchCategoryD1Error(
            f"{context}: filename/image_id mismatch: {filename_id} != {image_id}"
        )
    return (split, image_id), basename


def _image_key_text(key: ImageKey) -> str:
    return f"{key[0]}:{key[1]:012d}"


def _group_key_text(key: GroupKey) -> str:
    return f"{key[0]}:{key[1]:012d}:{key[2]}"


def _record_stream_sha256(records: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.encode("utf-8") + b"\n")
    return digest.hexdigest()


def _source_for_manifest(name: str) -> str:
    mapping = dict(SOURCE_MANIFESTS)
    source = mapping.get(name)
    if source is None:
        raise NativePatchCategoryD1Error(f"unsupported source manifest: {name}")
    return source


def _bound_record(record: Any, *, label: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(record, Mapping):
        raise NativePatchCategoryD1Error(f"{label} binding is missing")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise NativePatchCategoryD1Error(f"{label} binding has no path")
    observed = _file_record(Path(raw_path))
    if (
        record.get("sha256") != observed["sha256"]
        or record.get("size_bytes") != observed["size_bytes"]
    ):
        raise NativePatchCategoryD1Error(f"{label} file binding drifted")
    return Path(observed["path"]), observed


def _official_image_keys(
    partition: Mapping[str, Any],
) -> tuple[frozenset[ImageKey], dict[str, dict[str, Any]]]:
    official = partition.get("official_ref8")
    if not isinstance(official, Mapping):
        raise NativePatchCategoryD1Error("partition receipt lost official Ref8")
    split_order = official.get("split_order")
    entries = official.get("splits")
    if not isinstance(split_order, list) or not isinstance(entries, Mapping):
        raise NativePatchCategoryD1Error("official Ref8 split contract drifted")
    keys: set[ImageKey] = set()
    records: dict[str, dict[str, Any]] = {}
    total_rows = 0
    for split in split_order:
        if not isinstance(split, str) or not isinstance(entries.get(split), Mapping):
            raise NativePatchCategoryD1Error("official Ref8 split entry drifted")
        entry = entries[split]
        path, record = _bound_record(
            entry.get("manifest"), label=f"official Ref8 {split}"
        )
        rows = 0
        with path.open("rb") as handle:
            for line_number, raw in enumerate(handle, start=1):
                row = _load_jsonl_row(raw, path=path, line_number=line_number)
                key, _basename = _image_key(
                    row, context=f"official Ref8 {split}:{line_number}"
                )
                keys.add(key)
                rows += 1
        if rows != entry.get("rows"):
            raise NativePatchCategoryD1Error(
                f"official Ref8 row count drifted for {split}"
            )
        total_rows += rows
        records[split] = record
    if (
        total_rows != official.get("rows")
        or len(keys) != official.get("unique_image_keys")
    ):
        raise NativePatchCategoryD1Error("official Ref8 union count drifted")
    expected_stream = official.get("ordered_image_key_stream_sha256")
    if expected_stream is not None:
        observed_stream = _record_stream_sha256(
            [_image_key_text(key) for key in sorted(keys)]
        )
        if observed_stream != expected_stream:
            raise NativePatchCategoryD1Error("official Ref8 image stream drifted")
    return frozenset(keys), records


def _bbox_value(value: Any, *, context: str) -> tuple[float, float, float, float]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(type(number) not in (int, float) for number in value)
    ):
        raise NativePatchCategoryD1Error(f"{context}: bbox must contain four numbers")
    box = tuple(float(number) for number in value)
    if any(not math.isfinite(number) for number in box) or box[2] <= 0 or box[3] <= 0:
        raise NativePatchCategoryD1Error(f"{context}: bbox is invalid")
    return box


def _validate_category_row(
    row: Mapping[str, Any],
    *,
    context: str,
    coco_image_root: Path,
) -> tuple[GroupKey, str, Path, int, str, int, str]:
    _reject_forbidden_keys(row, context=context)
    primary_index = row.get("primary_support_instance_index")
    if (
        type(primary_index) is not int
        or primary_index != 0
        or row.get("stage_b_u2_category_complete") is not True
        or row.get("stage_b_u2_category_complete_schema") != UPSTREAM_ROW_SCHEMA
    ):
        raise NativePatchCategoryD1Error(
            f"{context}: category-complete marker contract drifted"
        )
    image_key, basename = _image_key(row, context=context)
    declared_split = _required_text(
        row.get("category_complete_coco_split"),
        field="category_complete_coco_split",
        context=context,
    ).lower()
    if declared_split != image_key[0]:
        raise NativePatchCategoryD1Error(
            f"{context}: category-complete COCO split drifted"
        )
    category_id = _required_int(
        row.get("category_complete_coco_category_id"),
        field="category_complete_coco_category_id",
        context=context,
    )
    if category_id <= 0:
        raise NativePatchCategoryD1Error(f"{context}: category id must be positive")
    instances = row.get("instances")
    if (
        not isinstance(instances, list)
        or not instances
        or any(not isinstance(instance, Mapping) for instance in instances)
        or row.get("category_complete_instance_count") != len(instances)
    ):
        raise NativePatchCategoryD1Error(
            f"{context}: category-complete instance list drifted"
        )
    primary = instances[0]
    ann_id = _required_int(row.get("ann_id"), field="ann_id", context=context)
    if (
        primary.get("category_complete_primary") is not True
        or primary.get("coco_ann_id") != ann_id
        or primary.get("text_is_negative") is not False
    ):
        raise NativePatchCategoryD1Error(f"{context}: primary instance drifted")
    full_text = _required_text(
        primary.get("positive_phrase"), field="positive_phrase", context=context
    )
    _required_text(primary.get("raw_phrase"), field="raw_phrase", context=context)
    class_id = _required_int(
        primary.get("class_id"), field="instances[0].class_id", context=context
    )
    normalized_instances: list[dict[str, Any]] = []
    seen_ann_ids: set[int] = set()
    for index, instance in enumerate(instances):
        member_context = f"{context} instances[{index}]"
        member_class = _required_int(
            instance.get("class_id"), field="class_id", context=member_context
        )
        member_category = _required_int(
            instance.get("refcoco_category_id"),
            field="refcoco_category_id",
            context=member_context,
        )
        member_ann = _required_int(
            instance.get("coco_ann_id"), field="coco_ann_id", context=member_context
        )
        if member_class != class_id or member_category != category_id:
            raise NativePatchCategoryD1Error(
                f"{member_context}: instance category/class differs from primary"
            )
        if member_ann in seen_ann_ids:
            raise NativePatchCategoryD1Error(
                f"{member_context}: duplicate COCO annotation id"
            )
        seen_ann_ids.add(member_ann)
        if index > 0 and instance.get("category_complete_auxiliary") is not True:
            raise NativePatchCategoryD1Error(
                f"{member_context}: auxiliary marker is missing"
            )
        normalized_instances.append(
            {
                "bbox": list(_bbox_value(instance.get("bbox"), context=member_context)),
                "class_id": member_class,
                "coco_ann_id": member_ann,
                "refcoco_category_id": member_category,
            }
        )
    normalized_instances.sort(key=lambda member: member["coco_ann_id"])
    instance_set_sha = _sha256_bytes(_canonical_bytes(normalized_instances))
    query_path = coco_image_root / image_key[0] / basename
    group_key = (image_key[0], image_key[1], category_id)
    return (
        group_key,
        basename,
        query_path,
        class_id,
        instance_set_sha,
        len(instances),
        full_text,
    )


def _source_identity(row: Mapping[str, Any], *, context: str) -> str:
    identity: dict[str, Any] = {}
    for field_name in (
        "source",
        "image_id",
        "ann_id",
        "ref_id",
        "sent_id",
        "split",
        "filename",
    ):
        value = row.get(field_name)
        if value is None:
            raise NativePatchCategoryD1Error(
                f"{context}: source identity field {field_name} is missing"
            )
        identity[field_name] = value
    return _sha256_bytes(_canonical_bytes(identity))


def _insert_candidate(group: GroupState, source: SourceRef, *, k: int) -> None:
    hasher = group.source_stream_hashers.setdefault(
        source.source_dataset, hashlib.sha256()
    )
    hasher.update(
        _canonical_bytes(
            {
                "full_text": source.full_text,
                "identity_sha256": source.identity_sha256,
                "priority_sha256": source.priority_sha256,
                "raw_sha256": source.raw_sha256,
            }
        )
        + b"\n"
    )
    values = group.candidates.setdefault(source.source_dataset, [])
    values.append(source)
    values.sort(key=lambda member: (member.priority_sha256, member.identity_sha256))
    del values[k:]
    group.source_row_counts[source.source_dataset] += 1


def _scan_partition_manifest(
    *,
    partition: str,
    manifest: str,
    source_dataset: str,
    path: Path,
    expected_rows: int,
    coco_image_root: Path,
    groups: dict[GroupKey, GroupState],
    k: int,
) -> tuple[int, set[ImageKey]]:
    rows = 0
    images: set[ImageKey] = set()
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            row = _load_jsonl_row(raw, path=path, line_number=line_number)
            context = f"{partition}/{manifest}:{line_number}"
            (
                group_key,
                _basename,
                query_path,
                class_id,
                instance_set_sha,
                instance_count,
                full_text,
            ) = _validate_category_row(
                row, context=context, coco_image_root=coco_image_root
            )
            row_source = _required_text(
                row.get("source"), field="source", context=context
            ).casefold()
            if not row_source.startswith(SOURCE_ROW_PREFIXES[source_dataset]):
                raise NativePatchCategoryD1Error(
                    f"{context}: source row does not match manifest dataset"
                )
            group_id = _sha256_bytes(
                _canonical_bytes(
                    {
                        "namespace": SELECTION_NAMESPACE,
                        "group": _group_key_text(group_key),
                    }
                )
            )
            group = groups.get(group_key)
            if group is None:
                group = GroupState(
                    key=group_key,
                    group_id=group_id,
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
                raise NativePatchCategoryD1Error(
                    f"{context}: one physical group has inconsistent instance coverage"
                )
            identity_sha = _source_identity(row, context=context)
            raw_sha = _sha256_bytes(raw)
            priority = _sha256_bytes(
                _canonical_bytes(
                    {
                        "namespace": SELECTION_NAMESPACE,
                        "group_id": group_id,
                        "source_dataset": source_dataset,
                        "source_identity_sha256": identity_sha,
                        "full_text_sha256": _sha256_bytes(
                            full_text.encode("utf-8")
                        ),
                    }
                )
            )
            source_ref = SourceRef(
                partition=partition,
                source_dataset=source_dataset,
                manifest=manifest,
                line_number=line_number,
                raw_sha256=raw_sha,
                identity_sha256=identity_sha,
                priority_sha256=priority,
                full_text=full_text,
                ann_id=_required_int(row.get("ann_id"), field="ann_id", context=context),
                ref_id=_required_int(row.get("ref_id"), field="ref_id", context=context),
                sent_id=_required_int(row.get("sent_id"), field="sent_id", context=context),
            )
            _insert_candidate(group, source_ref, k=k)
            images.add((group_key[0], group_key[1]))
            rows += 1
    if rows != expected_rows:
        raise NativePatchCategoryD1Error(
            f"{partition}/{manifest}: row count drifted: {rows} != {expected_rows}"
        )
    return rows, images


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _select_source_variants(group: GroupState, *, k: int) -> list[SelectedRef]:
    all_candidates = sorted(
        (candidate for values in group.candidates.values() for candidate in values),
        key=lambda member: (member.priority_sha256, member.identity_sha256),
    )
    if not all_candidates:
        raise NativePatchCategoryD1Error(f"group {group.group_id} has no source rows")
    selected: list[SelectedRef] = []
    selected_ids: set[str] = set()
    selected_texts: set[str] = set()
    for source_dataset in PREFERRED_SOURCES:
        values = group.candidates.get(source_dataset, [])
        if not values:
            continue
        distinct = [
            value
            for value in values
            if _normalized_text(value.full_text) not in selected_texts
        ]
        chosen = (distinct or values)[0]
        selected.append(SelectedRef(chosen, "preferred_source"))
        selected_ids.add(chosen.identity_sha256)
        selected_texts.add(_normalized_text(chosen.full_text))
    if len(selected) > k:
        selected = selected[:k]
        selected_ids = {member.source.identity_sha256 for member in selected}
        selected_texts = {
            _normalized_text(member.source.full_text) for member in selected
        }
    for require_distinct_text in (True, False):
        for candidate in all_candidates:
            if len(selected) == k:
                break
            if candidate.identity_sha256 in selected_ids:
                continue
            normalized = _normalized_text(candidate.full_text)
            if require_distinct_text and normalized in selected_texts:
                continue
            selected.append(SelectedRef(candidate, "hash_fill_unique_candidate"))
            selected_ids.add(candidate.identity_sha256)
            selected_texts.add(normalized)
        if len(selected) == k:
            break
    if len(selected) < k:
        cycle = sorted(
            all_candidates,
            key=lambda member: _sha256_bytes(
                _canonical_bytes(
                    {
                        "namespace": SELECTION_NAMESPACE,
                        "mode": "hash_cycle_repeat",
                        "group_id": group.group_id,
                        "source_identity_sha256": member.identity_sha256,
                    }
                )
            ),
        )
        rotation = int(group.group_id[:16], 16) % len(cycle)
        while len(selected) < k:
            chosen = cycle[(rotation + len(selected)) % len(cycle)]
            selected.append(SelectedRef(chosen, "hash_cycle_repeat"))
    if len(selected) != k:
        raise NativePatchCategoryD1Error("variant selection did not produce exact K")
    return selected


def _group_source_signature(group: GroupState) -> tuple[Any, ...]:
    return (
        group.class_id,
        group.instance_set_sha256,
        group.instance_count,
        tuple(
            (
                source,
                group.source_row_counts.get(source, 0),
                (
                    group.source_stream_hashers[source].copy().hexdigest()
                    if source in group.source_stream_hashers
                    else None
                ),
                tuple(
                    (
                        candidate.identity_sha256,
                        candidate.raw_sha256,
                        candidate.priority_sha256,
                        candidate.full_text,
                        candidate.ann_id,
                        candidate.ref_id,
                        candidate.sent_id,
                    )
                    for candidate in group.candidates.get(source, [])
                ),
            )
            for source in PREFERRED_SOURCES
        ),
    )


def _partition_member_keys(
    partition: Mapping[str, Any], *, field_name: str
) -> frozenset[ImageKey]:
    members = partition.get(field_name)
    if not isinstance(members, list):
        raise NativePatchCategoryD1Error(
            f"partition receipt lost {field_name}"
        )
    keys: set[ImageKey] = set()
    for index, member in enumerate(members):
        context = f"partition {field_name}[{index}]"
        if not isinstance(member, Mapping):
            raise NativePatchCategoryD1Error(f"{context}: member is not an object")
        split = _required_text(
            member.get("coco_split"), field="coco_split", context=context
        ).lower()
        image_id = _required_int(
            member.get("image_id"), field="image_id", context=context
        )
        key = (split, image_id)
        if member.get("image_key") != _image_key_text(key):
            raise NativePatchCategoryD1Error(f"{context}: image key text drifted")
        if key in keys:
            raise NativePatchCategoryD1Error(f"{context}: duplicate image key")
        keys.add(key)
    return frozenset(keys)


def _selected_source_signature(
    selected: Sequence[SelectedRef],
) -> tuple[tuple[str, str, str, str], ...]:
    return tuple(
        (
            member.source.source_dataset,
            member.source.identity_sha256,
            member.source.raw_sha256,
            member.mode,
        )
        for member in selected
    )


def _support_candidate_from_row(
    row: Mapping[str, str], *, context: str
) -> SupportCandidate:
    path = Path(_required_text(row.get("path"), field="path", context=context))
    path = path.expanduser().resolve(strict=True)
    if not path.is_file():
        raise NativePatchCategoryD1Error(f"{context}: support path is not a file")
    class_id = _required_int(int(row["class_id"]), field="class_id", context=context)
    source_cache_class_id = _required_int(
        int(row["source_cache_class_id"]),
        field="source_cache_class_id",
        context=context,
    )
    source_image_id = _required_int(
        int(row["source_image_id"]), field="source_image_id", context=context
    )
    source_row_number = _required_int(
        int(row["source_row_number"]), field="source_row_number", context=context
    )
    raw_coco_id = row.get("coco_id", "")
    coco_id = None if raw_coco_id == "" else int(raw_coco_id)
    source_row_sha = _validate_sha256(
        row.get("source_row_sha256"), label=f"{context} source-row hash"
    )
    values = {
        key: row.get(key, "")
        for key in SUPPORT_COLUMNS
    }
    candidate_id = _sha256_bytes(_canonical_bytes(values))
    return SupportCandidate(
        path=path,
        class_id=class_id,
        source_cache_class_id=source_cache_class_id,
        class_assignment=_required_text(
            row.get("class_assignment"), field="class_assignment", context=context
        ),
        source=_required_text(row.get("source"), field="source", context=context),
        source_image_id=source_image_id,
        coco_id=coco_id,
        source_class=_required_text(
            row.get("source_class"), field="source_class", context=context
        ),
        source_row_number=source_row_number,
        source_row_sha256=source_row_sha,
        candidate_id=candidate_id,
    )


def _load_support_candidates(
    *,
    support_tsv: Path,
    support_receipt: Mapping[str, Any],
    required_classes: frozenset[int],
) -> tuple[dict[int, list[SupportCandidate]], dict[str, Any]]:
    record, _identity = _file_record_with_identity(support_tsv)
    sealed = support_receipt.get("outputs", {}).get("runtime_support_tsv")
    if (
        not isinstance(sealed, Mapping)
        or sealed.get("sha256") != record["sha256"]
        or sealed.get("size_bytes") != record["size_bytes"]
        or Path(str(sealed.get("path", ""))).expanduser().resolve()
        != Path(record["path"])
    ):
        raise NativePatchCategoryD1Error("support TSV binding drifted")
    try:
        payload = support_tsv.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise NativePatchCategoryD1Error(f"could not read support TSV: {error}") from error
    reader = csv.DictReader(io.StringIO(payload), delimiter="\t", strict=True)
    if tuple(reader.fieldnames or ()) != SUPPORT_COLUMNS:
        raise NativePatchCategoryD1Error("support TSV columns drifted")
    excluded = support_receipt.get("exclusion", {}).get("union_numeric_coco_ids")
    if not isinstance(excluded, list) or any(type(value) is not int for value in excluded):
        raise NativePatchCategoryD1Error("support exclusion union drifted")
    excluded_ids = frozenset(excluded)
    by_class: dict[int, list[SupportCandidate]] = defaultdict(list)
    rows = 0
    all_candidate_ids: set[str] = set()
    for line_number, row in enumerate(reader, start=2):
        context = f"support TSV:{line_number}"
        candidate = _support_candidate_from_row(row, context=context)
        if candidate.candidate_id in all_candidate_ids:
            raise NativePatchCategoryD1Error(f"{context}: duplicate support candidate")
        all_candidate_ids.add(candidate.candidate_id)
        if candidate.coco_id is not None and candidate.coco_id in excluded_ids:
            raise NativePatchCategoryD1Error(
                f"{context}: candidate leaks a dev/official COCO image"
            )
        if candidate.class_id in required_classes:
            by_class[candidate.class_id].append(candidate)
        rows += 1
    if rows != sealed.get("rows"):
        raise NativePatchCategoryD1Error("support TSV row count drifted")
    missing = sorted(required_classes - set(by_class))
    if missing:
        raise NativePatchCategoryD1Error(
            f"train-filtered support lacks D1 classes: {missing}"
        )
    for values in by_class.values():
        values.sort(key=lambda candidate: candidate.candidate_id)
    return dict(by_class), {
        "record": record,
        "rows": rows,
        "required_classes": sorted(required_classes),
        "covered_classes": sorted(by_class),
        "excluded_numeric_coco_ids": sorted(excluded_ids),
    }


def _content_record(
    path: Path,
    *,
    cache: dict[Path, dict[str, Any]],
    identities: dict[str, tuple[int, int, int, int]],
) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    record = cache.get(path)
    if record is None:
        record, identity = _file_record_with_identity(path)
        cache[path] = record
        identities[str(path)] = identity
    return record


def _support_priority(group_id: str, variant_index: int, candidate_id: str) -> str:
    return _sha256_bytes(
        _canonical_bytes(
            {
                "namespace": SUPPORT_SELECTION_NAMESPACE,
                "group_id": group_id,
                "variant_index": variant_index,
                "candidate_id": candidate_id,
            }
        )
    )


def _select_support_witnesses(
    *,
    group: GroupState,
    candidates: Sequence[SupportCandidate],
    query_record: Mapping[str, Any],
    support_receipt_sha256: str,
    content_cache: dict[Path, dict[str, Any]],
    content_identities: dict[str, tuple[int, int, int, int]],
    k: int,
) -> tuple[list[dict[str, Any]], int]:
    selected: list[dict[str, Any]] = []
    used_paths: set[Path] = set()
    used_content: set[str] = set()
    used_identities: set[str] = set()
    reuse_fallbacks = 0
    for variant_index in range(k):
        ordered = sorted(
            candidates,
            key=lambda candidate: (
                _support_priority(group.group_id, variant_index, candidate.candidate_id),
                candidate.candidate_id,
            ),
        )
        chosen: tuple[SupportCandidate, dict[str, Any], str] | None = None
        for require_unused in (True, False):
            for candidate in ordered:
                if candidate.class_id != group.class_id:
                    raise NativePatchCategoryD1Error("support class grouping drifted")
                if candidate.coco_id == group.key[1]:
                    continue
                content = _content_record(
                    candidate.path,
                    cache=content_cache,
                    identities=content_identities,
                )
                if content["sha256"] == query_record["sha256"]:
                    continue
                if require_unused and (
                    candidate.path in used_paths
                    or content["sha256"] in used_content
                    or candidate.image_identity in used_identities
                ):
                    continue
                priority = _support_priority(
                    group.group_id, variant_index, candidate.candidate_id
                )
                chosen = (candidate, content, priority)
                if not require_unused:
                    reuse_fallbacks += 1
                break
            if chosen is not None:
                break
        if chosen is None:
            raise NativePatchCategoryD1Error(
                "no train-filtered, same-class, different-image/content support "
                f"candidate for group {_group_key_text(group.key)}"
            )
        candidate, content, priority = chosen
        used_paths.add(candidate.path)
        used_content.add(content["sha256"])
        used_identities.add(candidate.image_identity)
        selected.append(
            {
                "candidate_id": candidate.candidate_id,
                "class_assignment": candidate.class_assignment,
                "class_id": candidate.class_id,
                "coco_id": candidate.coco_id,
                "content_sha256": content["sha256"],
                "path": content["path"],
                "selection_priority_sha256": priority,
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
        )
    return selected, reuse_fallbacks


def _load_selected_rows(
    *,
    manifest_paths: Mapping[tuple[str, str], Path],
    selected: Mapping[tuple[str, str, int], SourceRef],
) -> dict[tuple[str, str, int], dict[str, Any]]:
    by_file: dict[tuple[str, str], dict[int, SourceRef]] = defaultdict(dict)
    for key, source_ref in selected.items():
        by_file[(key[0], key[1])][key[2]] = source_ref
    loaded: dict[tuple[str, str, int], dict[str, Any]] = {}
    for file_key, line_refs in by_file.items():
        path = manifest_paths[file_key]
        with path.open("rb") as handle:
            for line_number, raw in enumerate(handle, start=1):
                source_ref = line_refs.get(line_number)
                if source_ref is None:
                    continue
                if _sha256_bytes(raw) != source_ref.raw_sha256:
                    raise NativePatchCategoryD1Error(
                        f"selected source row changed at {path}:{line_number}"
                    )
                row = _load_jsonl_row(raw, path=path, line_number=line_number)
                if _source_identity(row, context=f"{path}:{line_number}") != (
                    source_ref.identity_sha256
                ):
                    raise NativePatchCategoryD1Error(
                        f"selected source identity changed at {path}:{line_number}"
                    )
                loaded[source_ref.key] = row
    if len(loaded) != len(selected):
        missing = sorted(set(selected) - set(loaded))
        raise NativePatchCategoryD1Error(f"selected source rows were not found: {missing}")
    return loaded


def _predicted_output(path: Path, payload: bytes, *, rows: int) -> dict[str, Any]:
    return {
        "path": str(path.expanduser().resolve()),
        "rows": rows,
        "size_bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }


def make_plan(
    *,
    partition_receipt: Path = PARTITION_RECEIPT,
    support_receipt: Path = SUPPORT_RECEIPT,
    support_tsv: Path = SUPPORT_TSV,
    coco_image_root: Path = COCO_IMAGE_ROOT,
    output_root: Path = OUTPUT_ROOT,
    expected_partition_receipt_sha256: str | None = (
        EXPECTED_PARTITION_RECEIPT_SHA256
    ),
    expected_support_receipt_sha256: str | None = EXPECTED_SUPPORT_RECEIPT_SHA256,
    expected_support_tsv_sha256: str | None = EXPECTED_SUPPORT_TSV_SHA256,
    k: int = K_VARIANTS,
    splits: Sequence[str] = SPLITS,
) -> BuildPlan:
    if type(k) is not int or k != K_VARIANTS:
        raise NativePatchCategoryD1Error(f"D1 requires K={K_VARIANTS}")
    if tuple(splits) != SPLITS:
        raise NativePatchCategoryD1Error("D1 split order must be train/dev_screen/dev_full")
    output_root = output_root.expanduser().resolve()
    coco_image_root = coco_image_root.expanduser().resolve(strict=True)
    partition, partition_record = _validate_receipt(
        partition_receipt,
        label="new-head partition receipt",
        expected_schema=UPSTREAM_PARTITION_SCHEMA,
        expected_sha256=expected_partition_receipt_sha256,
    )
    support, support_record = _validate_receipt(
        support_receipt,
        label="support partition receipt",
        expected_schema=UPSTREAM_SUPPORT_SCHEMA,
        expected_sha256=expected_support_receipt_sha256,
    )
    support_partition_binding = support.get("inputs", {}).get("partition_receipt")
    if (
        not isinstance(support_partition_binding, Mapping)
        or support_partition_binding.get("sha256") != partition_record["sha256"]
        or support_partition_binding.get("size_bytes") != partition_record["size_bytes"]
    ):
        raise NativePatchCategoryD1Error(
            "support bank is not bound to the selected new-head partition"
        )
    if expected_support_tsv_sha256 is not None:
        _validate_sha256(expected_support_tsv_sha256, label="expected support TSV hash")
        if _sha256_file(support_tsv) != expected_support_tsv_sha256:
            raise NativePatchCategoryD1Error("support TSV preregistered hash drifted")

    official_keys, official_records = _official_image_keys(partition)
    outputs = partition.get("outputs", {}).get("d1_category_complete")
    manifest_order = partition.get("source_manifest_order")
    if (
        not isinstance(outputs, Mapping)
        or not isinstance(manifest_order, list)
        or tuple(manifest_order) != tuple(name for name, _source in SOURCE_MANIFESTS)
    ):
        raise NativePatchCategoryD1Error("D1 partition manifest order drifted")

    manifest_paths: dict[tuple[str, str], Path] = {}
    manifest_records: dict[str, dict[str, dict[str, Any]]] = {}
    sealed_files: dict[str, dict[str, Any]] = {
        "partition_receipt": partition_record,
        "support_receipt": support_record,
    }
    for split, record in official_records.items():
        sealed_files[f"official_ref8:{split}"] = record

    groups_by_split: dict[str, dict[GroupKey, GroupState]] = {}
    image_sets: dict[str, set[ImageKey]] = {}
    input_row_counts: dict[str, dict[str, int]] = {}
    for partition_name in splits:
        partition_outputs = outputs.get(partition_name)
        if not isinstance(partition_outputs, Mapping):
            raise NativePatchCategoryD1Error(
                f"partition receipt lost D1 {partition_name} outputs"
            )
        groups: dict[GroupKey, GroupState] = {}
        images: set[ImageKey] = set()
        manifest_records[partition_name] = {}
        input_row_counts[partition_name] = {}
        for manifest in manifest_order:
            source_dataset = _source_for_manifest(manifest)
            entry = partition_outputs.get(manifest)
            path, record = _bound_record(
                entry, label=f"D1 {partition_name}/{manifest}"
            )
            expected_rows = entry.get("rows")
            if type(expected_rows) is not int or expected_rows < 0:
                raise NativePatchCategoryD1Error(
                    f"D1 {partition_name}/{manifest} row count is invalid"
                )
            observed_rows, manifest_images = _scan_partition_manifest(
                partition=partition_name,
                manifest=manifest,
                source_dataset=source_dataset,
                path=path,
                expected_rows=expected_rows,
                coco_image_root=coco_image_root,
                groups=groups,
                k=k,
            )
            manifest_paths[(partition_name, manifest)] = path
            manifest_records[partition_name][manifest] = {
                **record,
                "rows": observed_rows,
            }
            input_row_counts[partition_name][manifest] = observed_rows
            images.update(manifest_images)
            sealed_files[f"D1:{partition_name}:{manifest}"] = record
        if any(image in official_keys for image in images):
            raise NativePatchCategoryD1Error(
                f"D1 {partition_name} contains an official Ref8 image"
            )
        summary = partition.get("partition_summary", {}).get(partition_name)
        observed_rows_by_manifest = input_row_counts[partition_name]
        observed_image_stream = _record_stream_sha256(
            [_image_key_text(key) for key in sorted(images)]
        )
        if (
            not isinstance(summary, Mapping)
            or summary.get("rows") != sum(observed_rows_by_manifest.values())
            or summary.get("rows_by_manifest") != observed_rows_by_manifest
            or summary.get("unique_image_keys") != len(images)
            or summary.get("ordered_image_key_stream_sha256")
            != observed_image_stream
        ):
            raise NativePatchCategoryD1Error(
                f"D1 {partition_name} does not replay partition summary"
            )
        groups_by_split[partition_name] = groups
        image_sets[partition_name] = images

    dev_full_member_keys = _partition_member_keys(
        partition, field_name="dev_full_members"
    )
    dev_screen_member_keys = _partition_member_keys(
        partition, field_name="dev_screen_members"
    )
    if dev_full_member_keys != image_sets["dev_full"]:
        raise NativePatchCategoryD1Error(
            "dev_full outputs do not match explicit partition members"
        )
    if dev_screen_member_keys != image_sets["dev_screen"]:
        raise NativePatchCategoryD1Error(
            "dev_screen outputs do not match explicit partition members"
        )

    if image_sets["train"] & image_sets["dev_full"]:
        raise NativePatchCategoryD1Error("train and dev_full images overlap")
    if not image_sets["dev_screen"].issubset(image_sets["dev_full"]):
        raise NativePatchCategoryD1Error("dev_screen is not nested in dev_full")
    if not set(groups_by_split["dev_screen"]).issubset(groups_by_split["dev_full"]):
        raise NativePatchCategoryD1Error("dev_screen groups are not nested in dev_full")
    for group_key, screen_group in groups_by_split["dev_screen"].items():
        full_group = groups_by_split["dev_full"][group_key]
        if _group_source_signature(screen_group) != _group_source_signature(full_group):
            raise NativePatchCategoryD1Error(
                "dev_screen group content/source rows differ from dev_full: "
                f"{_group_key_text(group_key)}"
            )

    required_classes = frozenset(
        group.class_id
        for groups in groups_by_split.values()
        for group in groups.values()
    )
    support_by_class, support_summary = _load_support_candidates(
        support_tsv=support_tsv,
        support_receipt=support,
        required_classes=required_classes,
    )
    expected_support_exclusion_ids = {
        image_id for _split, image_id in (set(image_sets["dev_full"]) | set(official_keys))
    }
    observed_support_exclusion_ids = set(
        support_summary["excluded_numeric_coco_ids"]
    )
    if observed_support_exclusion_ids != expected_support_exclusion_ids:
        raise NativePatchCategoryD1Error(
            "support exclusion union does not equal dev_full plus official Ref8"
        )
    support_tsv_record = support_summary["record"]
    sealed_files["support_tsv"] = support_tsv_record

    selected_by_split: dict[str, dict[GroupKey, list[SelectedRef]]] = {}
    selected_rows: dict[tuple[str, str, int], SourceRef] = {}
    for partition_name in splits:
        selected_by_split[partition_name] = {}
        for group_key in sorted(groups_by_split[partition_name]):
            group = groups_by_split[partition_name][group_key]
            selected = _select_source_variants(group, k=k)
            selected_by_split[partition_name][group_key] = selected
            for member in selected:
                selected_rows[member.source.key] = member.source
    for group_key, screen_selected in selected_by_split["dev_screen"].items():
        full_selected = selected_by_split["dev_full"][group_key]
        if _selected_source_signature(screen_selected) != _selected_source_signature(
            full_selected
        ):
            raise NativePatchCategoryD1Error(
                "dev_screen selected variants differ from dev_full: "
                f"{_group_key_text(group_key)}"
            )
    loaded_rows = _load_selected_rows(
        manifest_paths=manifest_paths,
        selected=selected_rows,
    )

    content_cache: dict[Path, dict[str, Any]] = {}
    content_identities: dict[str, tuple[int, int, int, int]] = {}
    output_payloads: dict[str, bytes] = {}
    split_receipts: dict[str, Any] = {}
    all_selected_support_paths: set[str] = set()
    all_selected_support_hashes: set[str] = set()
    all_query_paths: set[str] = set()
    all_query_hashes: set[str] = set()
    global_support_reuse_fallbacks = 0

    for partition_name in splits:
        payload_parts: list[bytes] = []
        group_ids: list[str] = []
        row_hashes: list[str] = []
        source_row_hashes: list[str] = []
        source_identity_hashes: list[str] = []
        support_content_hashes: list[str] = []
        support_candidate_ids: list[str] = []
        query_content_hashes: list[str] = []
        instance_set_hashes: list[str] = []
        availability_counts: Counter[str] = Counter()
        selected_source_counts: Counter[str] = Counter()
        selection_mode_counts: Counter[str] = Counter()
        selected_unique_text_histogram: Counter[str] = Counter()
        support_reuse_fallbacks = 0
        instance_counts: Counter[int] = Counter()

        for group_key in sorted(groups_by_split[partition_name]):
            group = groups_by_split[partition_name][group_key]
            selected = selected_by_split[partition_name][group_key]
            query_record = _content_record(
                group.query_path,
                cache=content_cache,
                identities=content_identities,
            )
            all_query_paths.add(query_record["path"])
            all_query_hashes.add(query_record["sha256"])
            support_witnesses, reuse_count = _select_support_witnesses(
                group=group,
                candidates=support_by_class[group.class_id],
                query_record=query_record,
                support_receipt_sha256=support_record["sha256"],
                content_cache=content_cache,
                content_identities=content_identities,
                k=k,
            )
            support_reuse_fallbacks += reuse_count
            available = [
                source for source in PREFERRED_SOURCES if group.candidates.get(source)
            ]
            availability_counts["+".join(available)] += 1
            selected_unique_text_histogram[
                str(len({_normalized_text(member.source.full_text) for member in selected}))
            ] += 1
            instance_counts[group.instance_count] += 1
            group_ids.append(group.group_id)
            instance_set_hashes.append(group.instance_set_sha256)

            for variant_index, (selected_ref, support_witness) in enumerate(
                zip(selected, support_witnesses, strict=True)
            ):
                source_ref = selected_ref.source
                row = copy.deepcopy(loaded_rows[source_ref.key])
                row.update(
                    {
                        "native_patch_category_group_id": group.group_id,
                        "native_patch_category_source_dataset": (
                            source_ref.source_dataset
                        ),
                        "native_patch_category_source_identity_sha256": (
                            source_ref.identity_sha256
                        ),
                        "native_patch_category_source_line_number": (
                            source_ref.line_number
                        ),
                        "native_patch_category_source_manifest": source_ref.manifest,
                        "native_patch_category_source_row_sha256": (
                            source_ref.raw_sha256
                        ),
                        "native_patch_category_variant_index": variant_index,
                        "native_patch_category_variant_selection": selected_ref.mode,
                        "query_image_witness": {
                            "content_sha256": query_record["sha256"],
                            "path": query_record["path"],
                            "size_bytes": query_record["size_bytes"],
                            "source_filename": group.filename,
                        },
                        "stage_b_native_patch_category_d1": True,
                        "stage_b_native_patch_category_d1_schema": ROW_SCHEMA,
                        "support_patch_witness": support_witness,
                    }
                )
                _reject_forbidden_keys(row, context="emitted D1 row")
                raw = _canonical_bytes(row) + b"\n"
                payload_parts.append(raw)
                row_hashes.append(_sha256_bytes(raw))
                source_row_hashes.append(source_ref.raw_sha256)
                source_identity_hashes.append(source_ref.identity_sha256)
                support_content_hashes.append(support_witness["content_sha256"])
                support_candidate_ids.append(support_witness["candidate_id"])
                query_content_hashes.append(query_record["sha256"])
                selected_source_counts[source_ref.source_dataset] += 1
                selection_mode_counts[selected_ref.mode] += 1
                all_selected_support_paths.add(support_witness["path"])
                all_selected_support_hashes.add(support_witness["content_sha256"])

        payload = b"".join(payload_parts)
        output_payloads[partition_name] = payload
        output_record = _predicted_output(
            output_root / OUTPUT_FILES[partition_name],
            payload,
            rows=len(row_hashes),
        )
        split_receipts[partition_name] = {
            "groups": len(group_ids),
            "rows": len(row_hashes),
            "unique_images": len(image_sets[partition_name]),
            "unique_physical_categories": len(
                {group_key[2] for group_key in groups_by_split[partition_name]}
            ),
            "input_rows_by_manifest": input_row_counts[partition_name],
            "source_availability_group_counts": dict(sorted(availability_counts.items())),
            "selected_source_counts": dict(sorted(selected_source_counts.items())),
            "selection_mode_counts": dict(sorted(selection_mode_counts.items())),
            "selected_unique_full_texts_per_group_histogram": dict(
                sorted(selected_unique_text_histogram.items())
            ),
            "support_reuse_fallback_rows": support_reuse_fallbacks,
            "instance_count_group_histogram": {
                str(count): instance_counts[count] for count in sorted(instance_counts)
            },
            "ordered_group_id_stream_sha256": _record_stream_sha256(group_ids),
            "ordered_instance_set_sha256_stream_sha256": _record_stream_sha256(
                instance_set_hashes
            ),
            "ordered_output_row_sha256_stream_sha256": _record_stream_sha256(
                row_hashes
            ),
            "ordered_source_row_sha256_stream_sha256": _record_stream_sha256(
                source_row_hashes
            ),
            "ordered_source_identity_sha256_stream_sha256": _record_stream_sha256(
                source_identity_hashes
            ),
            "ordered_support_candidate_id_stream_sha256": _record_stream_sha256(
                support_candidate_ids
            ),
            "ordered_support_content_sha256_stream_sha256": _record_stream_sha256(
                support_content_hashes
            ),
            "ordered_query_content_sha256_stream_sha256": _record_stream_sha256(
                query_content_hashes
            ),
            "output": output_record,
        }
        global_support_reuse_fallbacks += support_reuse_fallbacks

    dev_screen_group_ids = {
        groups_by_split["dev_screen"][key].group_id
        for key in groups_by_split["dev_screen"]
    }
    dev_full_group_ids = {
        groups_by_split["dev_full"][key].group_id
        for key in groups_by_split["dev_full"]
    }
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "row_schema": ROW_SCHEMA,
        "builder": _file_record(Path(__file__)),
        "inputs": {
            "new_head_partition_receipt": partition_record,
            "support_partition_receipt": support_record,
            "runtime_support_tsv": support_tsv_record,
            "category_complete_manifests": manifest_records,
            "official_ref8_manifests": official_records,
        },
        "selection_contract": {
            "group_key": [
                "category_complete_coco_split",
                "image_id",
                "category_complete_coco_category_id",
            ],
            "variants_per_group": k,
            "preferred_source_order": list(PREFERRED_SOURCES),
            "preferred_source_policy": (
                "one_min_sha256_priority_distinct_text_if_available_per_source_v1"
            ),
            "missing_source_policy": "hash_fill_unique_then_hash_cycle_repeat_v1",
            "group_order": "coco_split_image_id_category_id_ascending",
            "variant_order": "preferred_sources_then_hash_fill_v1",
            "namespace": SELECTION_NAMESPACE,
            "model_score_free": True,
            "teacher_score_free": True,
            "checkpoint_output_free": True,
            "embedding_free": True,
            "forbidden_source_keys": sorted(FORBIDDEN_SOURCE_KEYS),
            "forbidden_source_key_prefixes": list(FORBIDDEN_OWNER_PREFIXES),
        },
        "category_complete_contract": {
            "primary_support_instance_index": 0,
            "primary_marker": "category_complete_primary",
            "auxiliary_marker": "category_complete_auxiliary",
            "instance_set_identity": (
                "sorted_coco_ann_id_bbox_class_id_refcoco_category_id_v1"
            ),
            "all_same_physical_coco_category": True,
            "legacy_runtime_marker_preserved": "stage_b_u2_category_complete",
            "native_provenance_marker": "stage_b_native_patch_category_d1",
        },
        "support_contract": {
            "candidate_source": "sealed_train_filtered_runtime_support_tsv",
            "selection_namespace": SUPPORT_SELECTION_NAMESPACE,
            "same_canonical_class_id": True,
            "different_query_image_identity": True,
            "different_query_image_content_sha256": True,
            "prefer_distinct_support_path_content_and_image_within_group": True,
            "allow_within_group_reuse_only_when_class_pool_requires_it": True,
            "known_coco_leakage_exclusion": (
                "dev_full_union_official_ref8_numeric_coco_id"
            ),
            "vg_null_coco_id_scope": (
                "external_image_retained_under_upstream_support_partition_policy"
            ),
            "runtime_embeddings_consumed": False,
        },
        "split_relationships": {
            "train_images": len(image_sets["train"]),
            "dev_screen_images": len(image_sets["dev_screen"]),
            "dev_full_images": len(image_sets["dev_full"]),
            "official_ref8_images": len(official_keys),
            "train_dev_full_image_overlap": 0,
            "train_official_ref8_image_overlap": 0,
            "dev_full_official_ref8_image_overlap": 0,
            "dev_screen_is_nested_in_dev_full": True,
            "dev_screen_groups_are_nested_in_dev_full": True,
            "dev_screen_group_content_and_selected_sources_equal_dev_full": True,
            "dev_screen_group_count": len(dev_screen_group_ids),
            "dev_full_group_count": len(dev_full_group_ids),
        },
        "splits": split_receipts,
        "content_binding": {
            "unique_query_image_paths": len(all_query_paths),
            "unique_query_content_sha256": len(all_query_hashes),
            "unique_selected_support_paths": len(all_selected_support_paths),
            "unique_selected_support_content_sha256": len(
                all_selected_support_hashes
            ),
            "selected_support_reuse_fallback_rows": (
                global_support_reuse_fallbacks
            ),
            "sorted_query_path_stream_sha256": _record_stream_sha256(
                sorted(all_query_paths)
            ),
            "sorted_query_content_sha256_stream_sha256": _record_stream_sha256(
                sorted(all_query_hashes)
            ),
            "sorted_support_path_stream_sha256": _record_stream_sha256(
                sorted(all_selected_support_paths)
            ),
            "sorted_support_content_sha256_stream_sha256": _record_stream_sha256(
                sorted(all_selected_support_hashes)
            ),
        },
        "invariants": {
            "new_head_partition_receipt_hash_and_payload_replay": True,
            "support_partition_receipt_hash_and_payload_replay": True,
            "all_category_complete_manifests_are_content_hash_bound": True,
            "partition_rows_by_manifest_and_image_streams_replay": True,
            "dev_full_and_dev_screen_explicit_members_replay": True,
            "all_official_ref8_manifests_are_content_hash_bound": True,
            "runtime_support_tsv_is_content_hash_bound": True,
            "support_exclusion_union_replays_dev_full_plus_official_ref8": True,
            "source_rows_contain_no_forbidden_model_derived_fields": True,
            "selection_consumes_no_model_teacher_checkpoint_or_embedding_outputs": True,
            "every_group_emits_exactly_three_full_text_rows": all(
                value["rows"] == value["groups"] * k
                for value in split_receipts.values()
            ),
            "every_row_preserves_primary_index_zero_and_all_same_category_instances": True,
            "all_variants_in_a_group_share_the_same_complete_instance_set": True,
            "every_support_witness_matches_the_query_canonical_class": True,
            "every_support_witness_is_from_the_train_filtered_bank": True,
            "every_support_witness_differs_from_query_image_identity": True,
            "every_support_witness_differs_from_query_content_sha256": True,
            "all_query_and_selected_support_files_are_content_hash_bound": True,
            "official_ref8_images_are_excluded_from_all_outputs": True,
            "train_and_dev_full_images_are_disjoint": not (
                image_sets["train"] & image_sets["dev_full"]
            ),
            "dev_screen_images_are_nested_in_dev_full": image_sets[
                "dev_screen"
            ].issubset(image_sets["dev_full"]),
            "dev_screen_groups_are_nested_in_dev_full": dev_screen_group_ids.issubset(
                dev_full_group_ids
            ),
            "dev_screen_group_content_and_selected_sources_equal_dev_full": True,
        },
    }
    if any(value is not True for value in receipt["invariants"].values()):
        raise NativePatchCategoryD1Error("one or more D1 invariants failed")
    receipt["canonical_payload_sha256"] = _sha256_bytes(_canonical_bytes(receipt))
    return BuildPlan(
        receipt=receipt,
        outputs=output_payloads,
        sealed_files=sealed_files,
        content_identities=content_identities,
    )


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
    """Atomically publish a directory without replacing an existing name."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise NativePatchCategoryD1Error(
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
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(destination),
        rename_noreplace,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise NativePatchCategoryD1Error(
            f"refusing concurrent overwrite of output root: {destination}"
        )
    raise OSError(error_number, os.strerror(error_number), str(destination))


def _assert_inputs_unchanged(plan: BuildPlan) -> None:
    for label, sealed in plan.sealed_files.items():
        observed = _file_record(Path(str(sealed["path"])))
        if observed != dict(sealed):
            raise NativePatchCategoryD1Error(f"{label} changed before atomic commit")
    for raw_path, identity in plan.content_identities.items():
        path = Path(raw_path)
        if _stat_identity(path.stat()) != identity:
            raise NativePatchCategoryD1Error(
                f"content-bound image changed before atomic commit: {path}"
            )


def build(**kwargs: Any) -> dict[str, Any]:
    output_root = Path(kwargs.get("output_root", OUTPUT_ROOT)).expanduser().resolve()
    if output_root.exists():
        raise NativePatchCategoryD1Error(
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
        for split in SPLITS:
            path = temporary_root / OUTPUT_FILES[split]
            with path.open("xb") as handle:
                handle.write(plan.outputs[split])
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
            raise NativePatchCategoryD1Error(
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
        raise NativePatchCategoryD1Error(
            f"output root is not a directory: {output_root}"
        )
    expected_entries = {"receipt.json", *OUTPUT_FILES.values()}
    observed_entries = {entry.name for entry in output_root.iterdir()}
    if observed_entries != expected_entries:
        raise NativePatchCategoryD1Error(
            "D1 artifact directory entry set does not replay exactly: "
            f"expected={sorted(expected_entries)}, observed={sorted(observed_entries)}"
        )
    plan = make_plan(**kwargs)
    for split in SPLITS:
        try:
            observed = (output_root / OUTPUT_FILES[split]).read_bytes()
        except OSError as error:
            raise NativePatchCategoryD1Error(
                f"could not read {split} output: {error}"
            ) from error
        if observed != plan.outputs[split]:
            raise NativePatchCategoryD1Error(
                f"{split} output does not replay byte-for-byte"
            )
    try:
        observed_receipt = (output_root / "receipt.json").read_bytes()
    except OSError as error:
        raise NativePatchCategoryD1Error(
            f"could not read D1 receipt: {error}"
        ) from error
    if observed_receipt != _receipt_bytes(plan.receipt):
        raise NativePatchCategoryD1Error("D1 receipt does not replay byte-for-byte")
    return plan.receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition-receipt", type=Path, default=PARTITION_RECEIPT)
    parser.add_argument("--support-receipt", type=Path, default=SUPPORT_RECEIPT)
    parser.add_argument("--support-tsv", type=Path, default=SUPPORT_TSV)
    parser.add_argument("--coco-image-root", type=Path, default=COCO_IMAGE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
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
