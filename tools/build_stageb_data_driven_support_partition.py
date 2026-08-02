#!/usr/bin/env python3
"""Build a leakage-filtered, runtime-exact Stage-B support bank.

The sealed image cache is the candidate source of truth.  Filtering therefore
only removes candidates from the exact bank used by the existing runtime; it
never repeats the stochastic reservoir sampler that originally built that
cache.  A second, audit-only TSV preserves the corresponding upstream TSV rows
byte-for-byte in their original order.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import re
import shutil
import tempfile
import zipfile
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SUPPORT_TSV = Path(
    "/media/haoyi/T9/data/patches_quality_emb/emb_index_from_quality.tsv"
)
SUPPORT_CACHE = Path(str(SUPPORT_TSV) + ".bank.clean.img.pkl")
SUPPORT_IMAGE_ROOT = Path("/media/haoyi/T9/data/patches_quality")
CANONICAL_CLASSES = Path("/media/haoyi/T9/data/canonical_classes_with_aliases.json")
VG_METADATA_ROOT = Path("/media/haoyi/T9/data/visual_genome_metadata")
VG_METADATA_ZIP = VG_METADATA_ROOT / "image_data_v1_2_official.zip"
VG_METADATA_JSON = VG_METADATA_ROOT / "image_data.json"
PARTITION_RECEIPT = (
    REPO_ROOT
    / "data/ablations/stageb_data_driven_new_head_partition_20260723/receipt.json"
)
OUTPUT_ROOT = (
    REPO_ROOT
    / "data/ablations/stageb_data_driven_support_partition_20260723"
)
OUTPUT_RUNTIME_TSV = "filtered_support.tsv"
OUTPUT_AUDIT_TSV = "raw_filtered_clean.tsv"

RECEIPT_SCHEMA = "pivot.stageb.data_driven.support_partition_receipt/v1"
PARTITION_SCHEMA = "pivot.stageb.data_driven.new_head_partition_receipt/v1"
STREAM_POLICY = "sealed_cache_order_delete_only_v1"
EXCLUSION_POLICY = "dev_full_union_official_ref8_by_numeric_coco_id_v1"
RUNTIME_COLUMNS = (
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

EXPECTED_SHA256 = {
    "support_tsv": (
        "93b1de99dd611577470960c5194faee15909af52b066713f956bbb6f25f78d47"
    ),
    "support_cache": (
        "01ff270cef70bcf93e9884e89fb58d9a64ce749a1db5c2f7ca7a1ba3fde799c1"
    ),
    "canonical_classes": (
        "9074350284a759a8de8dc619cae86a0c76bfe6fa5db597f279507443089fd853"
    ),
    "vg_metadata_zip": (
        "b87a94918cb2ff4d952cf1dfeca0b9cf6cd6fd204c2f8704645653be1163681a"
    ),
    "vg_metadata_json": (
        "93d0053d7e8f451497646fa513e8fb9b1a889df6e757010b88e51e2e21803487"
    ),
}
EXPECTED_RAW_ROWS = 1_450_870
EXPECTED_RAW_CLEAN_ROWS = 576_958
EXPECTED_CACHE_CLASSES = 2_020
EXPECTED_CACHE_CANDIDATES = 168_326
EXPECTED_TRAINING_CLASSES = 78
EXPECTED_PARTITION_RECEIPT_SHA256 = (
    "56de31d883ed137f3f9332c34de846839d82c0724120f42a49c5c1c302f38506"
)

_HEX64_RE = re.compile(r"[0-9a-f]{64}")
_SUPPORT_BASENAME_RE = re.compile(
    r"^(?P<class_name>.+)_(?P<source_image_id>[0-9]+)_"
    r"(?P<annotation_id>[0-9]+)_(?P<index>[0-9]+)\.jpg$"
)
_COCO_FILENAME_RE = re.compile(
    r"^COCO_(?P<split>train2014|val2014)_(?P<image_id>[0-9]{12})"
    r"\.(?:jpg|jpeg)$",
    re.IGNORECASE,
)
_WS_RE = re.compile(r"\s+")
_PUNC_RE = re.compile(r"[^a-z0-9 _-]+")


class SupportPartitionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Candidate:
    class_id: int
    source_cache_class_id: int
    class_assignment: str
    candidate_index: int
    path: Path
    source: str
    source_dir: str
    source_class: str
    source_image_id: int
    coco_id: int | None


@dataclass(frozen=True, slots=True)
class RawLink:
    line_number: int
    row_sha256: str
    source_class: str


@dataclass(frozen=True, slots=True)
class PartitionInfo:
    dev_full_ids: frozenset[int]
    official_ref8_ids: frozenset[int]
    excluded_ids: frozenset[int]
    training_class_ids: frozenset[int]
    receipt_summary: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class CacheInfo:
    candidates: tuple[Candidate, ...]
    class_ids: frozenset[int]
    receipt_summary: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class BuildPlan:
    receipt: dict[str, Any]
    candidates: tuple[Candidate, ...]
    runtime_candidates: tuple[Candidate, ...]
    retained_paths: frozenset[Path]
    raw_links: Mapping[Path, RawLink]
    name_to_canonical_id: Mapping[str, int]
    input_records: Mapping[str, Mapping[str, Any]]


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


def _validate_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _HEX64_RE.fullmatch(value) is None:
        raise SupportPartitionError(f"{label} is not a lowercase SHA-256")
    return value


def _file_record(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    if not path.is_file():
        raise SupportPartitionError(f"not a file: {path}")
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
        raise SupportPartitionError(f"file changed while hashing: {path}")
    return {
        "path": str(path),
        "size_bytes": int(after.st_size),
        "sha256": digest,
    }


def _predicted_record(path: Path, *, digest: Any, size: int, rows: int) -> dict[str, Any]:
    return {
        "path": str(path.expanduser().resolve()),
        "size_bytes": int(size),
        "sha256": digest.hexdigest(),
        "rows": int(rows),
    }


def _load_json(path: Path, *, label: str) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SupportPartitionError(f"could not load {label}: {path}: {error}") from error


def _required_int(value: Any, *, field: str, context: str) -> int:
    if type(value) is not int:
        raise SupportPartitionError(f"{context}: {field} must be an exact integer")
    return int(value)


def _required_text(value: Any, *, field: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SupportPartitionError(f"{context}: {field} must be non-empty text")
    return value.strip()


def _norm_text(value: str) -> str:
    value = value.strip().lower().replace("_", " ").replace("-", " ")
    value = _PUNC_RE.sub(" ", value)
    return _WS_RE.sub(" ", value).strip()


def _parse_tsv_record(raw: bytes, *, context: str) -> list[str]:
    if not raw.endswith(b"\n"):
        raise SupportPartitionError(f"{context}: TSV record lacks terminating LF")
    try:
        text = raw.rstrip(b"\r\n").decode("utf-8")
        records = list(csv.reader([text], delimiter="\t", strict=True))
    except (UnicodeError, csv.Error) as error:
        raise SupportPartitionError(f"{context}: invalid TSV record: {error}") from error
    if len(records) != 1:
        raise SupportPartitionError(f"{context}: invalid TSV record count")
    return records[0]


def _bound_file(record: Any, *, label: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(record, Mapping):
        raise SupportPartitionError(f"{label} file record is missing")
    raw_path = record.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise SupportPartitionError(f"{label} file record has no path")
    observed = _file_record(Path(raw_path))
    if (
        record.get("sha256") != observed["sha256"]
        or record.get("size_bytes") != observed["size_bytes"]
    ):
        raise SupportPartitionError(f"{label} file binding drifted")
    return Path(observed["path"]), observed


def _jsonl_rows(path: Path, *, label: str):
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.endswith(b"\n") or not raw.rstrip(b"\r\n"):
                raise SupportPartitionError(
                    f"{label}:{line_number}: invalid JSONL record framing"
                )
            try:
                row = json.loads(raw)
            except (UnicodeError, json.JSONDecodeError) as error:
                raise SupportPartitionError(
                    f"{label}:{line_number}: invalid JSON: {error}"
                ) from error
            if not isinstance(row, dict):
                raise SupportPartitionError(f"{label}:{line_number}: row is not an object")
            yield line_number, row


def _coco_key(row: Mapping[str, Any], *, context: str) -> tuple[str, int]:
    image_id = _required_int(row.get("image_id"), field="image_id", context=context)
    filename = _required_text(row.get("filename"), field="filename", context=context)
    match = _COCO_FILENAME_RE.fullmatch(Path(filename).name)
    if match is None or int(match.group("image_id")) != image_id:
        raise SupportPartitionError(f"{context}: non-canonical COCO filename/image_id")
    return match.group("split").lower(), image_id


def _validate_partition_receipt(
    path: Path,
    *,
    expected_sha256: str | None,
    expected_training_classes: int,
) -> PartitionInfo:
    record = _file_record(path)
    if expected_sha256 is not None:
        _validate_sha256(expected_sha256, label="expected partition receipt hash")
        if record["sha256"] != expected_sha256:
            raise SupportPartitionError("partition receipt SHA-256 mismatch")
    receipt = _load_json(path, label="new-head partition receipt")
    if not isinstance(receipt, dict) or receipt.get("schema") != PARTITION_SCHEMA:
        raise SupportPartitionError("new-head partition receipt schema drifted")
    stored_payload_sha = _validate_sha256(
        receipt.get("canonical_payload_sha256"),
        label="partition canonical payload hash",
    )
    payload = dict(receipt)
    del payload["canonical_payload_sha256"]
    if _sha256_bytes(_canonical_bytes(payload)) != stored_payload_sha:
        raise SupportPartitionError("partition canonical payload hash does not replay")
    invariants = receipt.get("invariants")
    if (
        not isinstance(invariants, dict)
        or not invariants
        or any(value is not True for value in invariants.values())
    ):
        raise SupportPartitionError("partition receipt has a failed invariant")

    dev_members = receipt.get("dev_full_members")
    if not isinstance(dev_members, list) or not dev_members:
        raise SupportPartitionError("partition receipt has no dev_full members")
    dev_keys: set[tuple[str, int]] = set()
    for index, member in enumerate(dev_members):
        context = f"dev_full_members[{index}]"
        if not isinstance(member, dict):
            raise SupportPartitionError(f"{context} is not an object")
        split = _required_text(member.get("coco_split"), field="coco_split", context=context)
        image_id = _required_int(member.get("image_id"), field="image_id", context=context)
        if split not in {"train2014", "val2014"} or member.get("image_key") != (
            f"{split}:{image_id:012d}"
        ):
            raise SupportPartitionError(f"{context}: image key drifted")
        if (split, image_id) in dev_keys:
            raise SupportPartitionError(f"{context}: duplicate image key")
        dev_keys.add((split, image_id))
    selection = receipt.get("selection_contract")
    summary = receipt.get("partition_summary")
    if (
        not isinstance(selection, dict)
        or selection.get("dev_full_target_images") != len(dev_keys)
        or not isinstance(summary, dict)
        or not isinstance(summary.get("dev_full"), dict)
        or summary["dev_full"].get("unique_image_keys") != len(dev_keys)
    ):
        raise SupportPartitionError("dev_full count does not replay partition receipt")

    official = receipt.get("official_ref8")
    if not isinstance(official, dict):
        raise SupportPartitionError("partition receipt has no official Ref8 binding")
    split_order = official.get("split_order")
    splits = official.get("splits")
    if (
        not isinstance(split_order, list)
        or not split_order
        or len(set(split_order)) != len(split_order)
        or not isinstance(splits, dict)
        or set(splits) != set(split_order)
    ):
        raise SupportPartitionError("official Ref8 split contract drifted")
    official_keys: set[tuple[str, int]] = set()
    official_rows = 0
    official_records: dict[str, Any] = {}
    for split_name in split_order:
        entry = splits[split_name]
        if not isinstance(entry, dict):
            raise SupportPartitionError(f"official split {split_name} is invalid")
        manifest_path, manifest_record = _bound_file(
            entry.get("manifest"), label=f"official split {split_name}"
        )
        rows = 0
        for line_number, row in _jsonl_rows(manifest_path, label=f"official {split_name}"):
            official_keys.add(
                _coco_key(row, context=f"official {split_name}:{line_number}")
            )
            rows += 1
        if entry.get("rows") != rows:
            raise SupportPartitionError(f"official split {split_name} row count drifted")
        official_rows += rows
        official_records[split_name] = {"rows": rows, "manifest": manifest_record}
    if (
        official.get("rows") != official_rows
        or official.get("unique_image_keys") != len(official_keys)
    ):
        raise SupportPartitionError("official Ref8 aggregate counts drifted")

    outputs = receipt.get("outputs")
    if not isinstance(outputs, dict):
        raise SupportPartitionError("partition receipt has no outputs")
    d1_outputs = outputs.get("d1_category_complete")
    train_outputs = d1_outputs.get("train") if isinstance(d1_outputs, dict) else None
    manifest_order = receipt.get("source_manifest_order")
    if (
        not isinstance(manifest_order, list)
        or not manifest_order
        or len(set(manifest_order)) != len(manifest_order)
        or not isinstance(train_outputs, dict)
        or set(train_outputs) != set(manifest_order)
    ):
        raise SupportPartitionError("partition D1 train output contract drifted")
    train_classes: set[int] = set()
    class_names: dict[int, set[str]] = {}
    train_rows = 0
    train_records: dict[str, Any] = {}
    for name in manifest_order:
        entry = train_outputs[name]
        manifest_path, manifest_record = _bound_file(entry, label=f"D1 train {name}")
        rows = 0
        for line_number, row in _jsonl_rows(manifest_path, label=f"D1 train {name}"):
            context = f"D1 train {name}:{line_number}"
            primary_index = _required_int(
                row.get("primary_support_instance_index"),
                field="primary_support_instance_index",
                context=context,
            )
            instances = row.get("instances")
            if (
                not isinstance(instances, list)
                or not instances
                or not 0 <= primary_index < len(instances)
                or not isinstance(instances[primary_index], dict)
            ):
                raise SupportPartitionError(f"{context}: invalid primary instance")
            primary = instances[primary_index]
            class_id = _required_int(
                primary.get("class_id"), field="class_id", context=context
            )
            if class_id < 0:
                raise SupportPartitionError(f"{context}: negative canonical class id")
            class_name = primary.get("canonical_name") or primary.get("category_name")
            if isinstance(class_name, str) and class_name.strip():
                class_names.setdefault(class_id, set()).add(class_name.strip())
            train_classes.add(class_id)
            rows += 1
        if entry.get("rows") != rows:
            raise SupportPartitionError(f"D1 train {name} row count drifted")
        train_rows += rows
        train_records[name] = {"rows": rows, "manifest": manifest_record}
    if type(expected_training_classes) is not int or expected_training_classes <= 0:
        raise SupportPartitionError("expected training class count is invalid")
    if len(train_classes) != expected_training_classes:
        raise SupportPartitionError(
            "D1 train class count drifted: "
            f"observed={len(train_classes)}, expected={expected_training_classes}"
        )
    if isinstance(summary.get("train"), dict) and summary["train"].get("rows") != train_rows:
        raise SupportPartitionError("D1 train rows do not match partition summary")

    dev_ids = frozenset(image_id for _split, image_id in dev_keys)
    official_ids = frozenset(image_id for _split, image_id in official_keys)
    return PartitionInfo(
        dev_full_ids=dev_ids,
        official_ref8_ids=official_ids,
        excluded_ids=frozenset(dev_ids | official_ids),
        training_class_ids=frozenset(train_classes),
        receipt_summary={
            "receipt": record,
            "schema": PARTITION_SCHEMA,
            "canonical_payload_sha256": stored_payload_sha,
            "dev_full": {
                "unique_image_keys": len(dev_keys),
                "unique_numeric_coco_ids": len(dev_ids),
                "numeric_coco_ids": sorted(dev_ids),
            },
            "official_ref8": {
                "rows": official_rows,
                "unique_image_keys": len(official_keys),
                "unique_numeric_coco_ids": len(official_ids),
                "numeric_coco_ids": sorted(official_ids),
                "splits": official_records,
            },
            "d1_train": {
                "rows": train_rows,
                "class_count": len(train_classes),
                "class_ids": sorted(train_classes),
                "class_names": {
                    str(class_id): sorted(class_names.get(class_id, set()))
                    for class_id in sorted(train_classes)
                },
                "manifests": train_records,
            },
        },
    )


def _load_vg_mapping(
    *,
    vg_json: Path,
    vg_zip: Path,
) -> tuple[dict[int, int | None], dict[str, Any]]:
    json_bytes = vg_json.read_bytes()
    try:
        with zipfile.ZipFile(vg_zip, "r") as archive:
            members = archive.infolist()
            if len(members) != 1 or members[0].filename != "image_data.json":
                raise SupportPartitionError("VG metadata ZIP member set drifted")
            member_bytes = archive.read(members[0])
            member_record = {
                "filename": members[0].filename,
                "size_bytes": members[0].file_size,
                "crc32": f"{members[0].CRC:08x}",
                "sha256": _sha256_bytes(member_bytes),
            }
    except (OSError, zipfile.BadZipFile, RuntimeError) as error:
        if isinstance(error, SupportPartitionError):
            raise
        raise SupportPartitionError(f"could not validate VG metadata ZIP: {error}") from error
    if member_bytes != json_bytes:
        raise SupportPartitionError("VG ZIP image_data.json differs from standalone JSON")
    try:
        payload = json.loads(json_bytes)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SupportPartitionError(f"invalid VG image_data.json: {error}") from error
    if not isinstance(payload, list) or not payload:
        raise SupportPartitionError("VG image_data.json must be a non-empty list")
    mapping: dict[int, int | None] = {}
    mapping_digest = hashlib.sha256()
    with_coco = 0
    null_coco = 0
    for index, row in enumerate(payload):
        context = f"VG image_data[{index}]"
        if not isinstance(row, dict):
            raise SupportPartitionError(f"{context} is not an object")
        image_id = _required_int(row.get("image_id"), field="image_id", context=context)
        if image_id <= 0 or image_id in mapping:
            raise SupportPartitionError(f"{context}: invalid or duplicate image_id")
        coco_value = row.get("coco_id")
        if coco_value is None:
            coco_id = None
            null_coco += 1
        else:
            coco_id = _required_int(coco_value, field="coco_id", context=context)
            if coco_id <= 0:
                raise SupportPartitionError(f"{context}: invalid coco_id")
            with_coco += 1
        mapping[image_id] = coco_id
    for image_id in sorted(mapping):
        mapping_digest.update(
            _canonical_bytes([image_id, mapping[image_id]]) + b"\n"
        )
    return mapping, {
        "entries": len(mapping),
        "with_coco_id": with_coco,
        "null_coco_id": null_coco,
        "ordered_mapping_sha256": mapping_digest.hexdigest(),
        "zip_member": member_record,
    }


def _load_canonical_classes(
    path: Path,
) -> tuple[
    dict[str, int],
    dict[int, str],
    dict[int, frozenset[str]],
    dict[str, Any],
]:
    payload = _load_json(path, label="canonical classes")
    if not isinstance(payload, list) or not payload:
        raise SupportPartitionError("canonical classes must be a non-empty list")
    name_to_id: dict[str, int] = {}
    id_to_name: dict[int, str] = {}
    names_by_id: dict[int, set[str]] = {}
    overwrite_count = 0
    for index, entry in enumerate(payload):
        context = f"canonical classes[{index}]"
        if not isinstance(entry, dict):
            raise SupportPartitionError(f"{context} is not an object")
        class_id = _required_int(entry.get("id"), field="id", context=context)
        if class_id < 0 or class_id in id_to_name:
            raise SupportPartitionError(f"{context}: duplicate or negative id")
        preferred = None
        class_names: set[str] = set()
        for field in ("base_name", "raw_name", "norm_name", "synset"):
            value = entry.get(field)
            if preferred is None and isinstance(value, str) and value.strip():
                preferred = value.strip()
            if isinstance(value, str) and value.strip():
                key = _norm_text(value)
                class_names.add(key)
                if key in name_to_id and name_to_id[key] != class_id:
                    overwrite_count += 1
                name_to_id[key] = class_id
        for field in ("synonyms",):
            values = entry.get(field)
            if isinstance(values, list):
                for value in values:
                    if isinstance(value, str) and value.strip():
                        key = _norm_text(value)
                        class_names.add(key)
                        if key in name_to_id and name_to_id[key] != class_id:
                            overwrite_count += 1
                        name_to_id[key] = class_id
        aliases = entry.get("aliases")
        if isinstance(aliases, list):
            for alias in aliases:
                if not isinstance(alias, dict):
                    continue
                for field in ("name", "norm_name"):
                    value = alias.get(field)
                    if isinstance(value, str) and value.strip():
                        key = _norm_text(value)
                        class_names.add(key)
                        if key in name_to_id and name_to_id[key] != class_id:
                            overwrite_count += 1
                        name_to_id[key] = class_id
        if preferred is None:
            raise SupportPartitionError(f"{context}: class has no usable name")
        id_to_name[class_id] = preferred
        names_by_id[class_id] = class_names
    return name_to_id, id_to_name, {
        class_id: frozenset(names) for class_id, names in names_by_id.items()
    }, {
        "class_count": len(id_to_name),
        "normalized_name_count": len(name_to_id),
        "loader_order_overwrite_count": overwrite_count,
        "mapping_policy": "datasets.patch_episode._build_name_to_canonical_id compatible",
    }


def _parse_support_path(
    path: Path,
    *,
    support_image_root: Path,
) -> tuple[str, str, str, int]:
    expanded = path.expanduser()
    if not expanded.is_absolute():
        raise SupportPartitionError(f"support candidate is not a canonical absolute file: {path}")
    resolved = Path(os.path.normpath(str(expanded)))
    if resolved != expanded or not resolved.is_file():
        raise SupportPartitionError(f"support candidate is not a canonical absolute file: {path}")
    try:
        relative = resolved.relative_to(support_image_root)
    except ValueError as error:
        raise SupportPartitionError(
            f"support candidate escapes support image root: {resolved}"
        ) from error
    if len(relative.parts) != 4 or relative.parts[0] != "clean":
        raise SupportPartitionError(f"support candidate is not in the clean mirror: {resolved}")
    source_dir, source_class, basename = relative.parts[1:]
    if source_dir not in {"lvis_patches", "vg_patches"}:
        raise SupportPartitionError(f"unsupported patch source parent: {source_dir}")
    match = _SUPPORT_BASENAME_RE.fullmatch(basename)
    if match is None or match.group("class_name") != source_class:
        raise SupportPartitionError(f"support basename/class directory drifted: {resolved}")
    source_image_id = int(match.group("source_image_id"))
    if source_image_id <= 0:
        raise SupportPartitionError(f"invalid source image id: {resolved}")
    source = "lvis" if source_dir == "lvis_patches" else "vg"
    return source, source_dir, source_class, source_image_id


def _load_cache(
    *,
    cache_path: Path,
    support_tsv: Path,
    canonical_classes: Path,
    support_image_root: Path,
    vg_mapping: Mapping[int, int | None],
    canonical_ids: frozenset[int],
    expected_classes: int,
    expected_candidates: int,
) -> CacheInfo:
    try:
        with cache_path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as error:
        raise SupportPartitionError(f"could not load sealed support cache: {error}") from error
    if not isinstance(payload, dict):
        raise SupportPartitionError("sealed support cache is not a mapping")
    meta = payload.get("meta")
    bank = payload.get("bank")
    if not isinstance(meta, dict) or not isinstance(bank, dict) or not bank:
        raise SupportPartitionError("sealed support cache contract drifted")
    required_meta = {
        "version": 3,
        "bucket": "clean",
        "use_embedding": False,
        "max_per_class": 200,
        "patch_class_map_json": None,
    }
    if any(meta.get(key) != value for key, value in required_meta.items()):
        raise SupportPartitionError("sealed support cache metadata drifted")
    for field, expected_path in (
        ("tsv_path", support_tsv),
        ("canonical_classes_json", canonical_classes),
        ("support_patch_image_root", support_image_root),
    ):
        raw = meta.get(field)
        if not isinstance(raw, str) or Path(raw).expanduser().resolve() != expected_path:
            raise SupportPartitionError(f"sealed support cache {field} binding drifted")

    normalized_bank: dict[int, Sequence[Any]] = {}
    for raw_class_id, values in bank.items():
        if type(raw_class_id) is not int or raw_class_id < 0:
            raise SupportPartitionError("support cache has a non-integer class key")
        if raw_class_id in normalized_bank:
            raise SupportPartitionError("support cache has duplicate class ids")
        if not isinstance(values, list) or not values or len(values) > 200:
            raise SupportPartitionError(f"support cache class {raw_class_id} is invalid")
        normalized_bank[raw_class_id] = values
    if set(normalized_bank) - set(canonical_ids):
        raise SupportPartitionError("support cache contains unknown canonical class ids")
    if len(normalized_bank) != expected_classes:
        raise SupportPartitionError(
            f"support cache class count drifted: {len(normalized_bank)} != {expected_classes}"
        )

    candidates: list[Candidate] = []
    seen_paths: set[Path] = set()
    source_counts: Counter[str] = Counter()
    class_counts: Counter[int] = Counter()
    stream_digest = hashlib.sha256()
    vg_ids: set[int] = set()
    for class_id in sorted(normalized_bank):
        for candidate_index, raw_path in enumerate(normalized_bank[class_id]):
            if not isinstance(raw_path, str) or not raw_path:
                raise SupportPartitionError(
                    f"support cache class {class_id} has an invalid candidate path"
                )
            path = Path(raw_path)
            source, source_dir, source_class, source_image_id = _parse_support_path(
                path, support_image_root=support_image_root
            )
            if path in seen_paths:
                raise SupportPartitionError(f"duplicate support cache path: {path}")
            seen_paths.add(path)
            if source == "lvis":
                coco_id: int | None = source_image_id
            else:
                vg_ids.add(source_image_id)
                if source_image_id not in vg_mapping:
                    raise SupportPartitionError(
                        f"VG support image id lacks official mapping: {source_image_id}"
                    )
                coco_id = vg_mapping[source_image_id]
            candidate = Candidate(
                class_id=class_id,
                source_cache_class_id=class_id,
                class_assignment="sealed_cache_identity_v1",
                candidate_index=candidate_index,
                path=path,
                source=source,
                source_dir=source_dir,
                source_class=source_class,
                source_image_id=source_image_id,
                coco_id=coco_id,
            )
            candidates.append(candidate)
            source_counts[source] += 1
            class_counts[class_id] += 1
            stream_digest.update(
                _canonical_bytes(
                    [class_id, candidate_index, str(path), source_image_id, coco_id]
                )
                + b"\n"
            )
    if len(candidates) != expected_candidates:
        raise SupportPartitionError(
            "support cache candidate count drifted: "
            f"{len(candidates)} != {expected_candidates}"
        )
    return CacheInfo(
        candidates=tuple(candidates),
        class_ids=frozenset(normalized_bank),
        receipt_summary={
            "classes": len(normalized_bank),
            "candidates": len(candidates),
            "max_candidates_per_class": max(class_counts.values()),
            "source_counts": dict(sorted(source_counts.items())),
            "class_counts": {
                str(class_id): class_counts[class_id] for class_id in sorted(class_counts)
            },
            "unique_vg_source_image_ids": len(vg_ids),
            "ordered_candidate_stream_sha256": stream_digest.hexdigest(),
            "candidate_order": "sorted_integer_class_id_then_sealed_candidate_list",
        },
    )


def _raw_mirror_path(
    *,
    values: Sequence[str],
    columns: Mapping[str, int],
    support_image_root: Path,
    context: str,
) -> tuple[Path, str, str, int]:
    raw_path = values[columns["path"]]
    emb_rel = values[columns["emb_rel_path"]]
    if not raw_path or not emb_rel:
        raise SupportPartitionError(f"{context}: cache-linked raw row has an empty path")
    source_path = Path(raw_path)
    if not source_path.is_absolute() or len(source_path.parts) < 4:
        raise SupportPartitionError(f"{context}: raw patch path is not absolute")
    source_dir = source_path.parts[-3]
    source_class = source_path.parts[-2]
    match = _SUPPORT_BASENAME_RE.fullmatch(source_path.name)
    if (
        source_dir not in {"lvis_patches", "vg_patches"}
        or match is None
        or match.group("class_name") != source_class
    ):
        raise SupportPartitionError(f"{context}: raw patch path contract drifted")
    relative = PurePosixPath(emb_rel)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or len(relative.parts) != 4
        or relative.parts[0] != "clean"
        or relative.parts[1] != source_dir
        or relative.parts[2] != source_class
        or relative.suffix != ".npy"
        or relative.stem != source_path.stem
    ):
        raise SupportPartitionError(f"{context}: emb_rel_path does not bind raw path")
    source = "lvis" if source_dir == "lvis_patches" else "vg"
    source_image_id = int(match.group("source_image_id"))
    mirror = (support_image_root / Path(*relative.parts)).with_suffix(".jpg")
    return mirror, source, source_class, source_image_id


def _scan_raw_tsv(
    *,
    support_tsv: Path,
    support_image_root: Path,
    candidates: Sequence[Candidate],
    retained_paths: frozenset[Path],
    name_to_canonical_id: Mapping[str, int],
    output_path: Path,
    write_handle: BinaryIO | None,
    expected_input_sha256: str,
    expected_rows: int,
    expected_clean_rows: int,
) -> tuple[dict[Path, RawLink], dict[str, Any], dict[str, Any]]:
    candidate_by_path = {candidate.path: candidate for candidate in candidates}
    if len(candidate_by_path) != len(candidates):
        raise SupportPartitionError("candidate paths are not globally unique")
    input_digest = hashlib.sha256()
    audit_digest = hashlib.sha256()
    audit_size = 0
    audit_rows = 0
    total_rows = 0
    clean_rows = 0
    bucket_counts: Counter[str] = Counter()
    links: dict[Path, RawLink] = {}
    with support_tsv.open("rb") as handle:
        header_raw = handle.readline()
        input_digest.update(header_raw)
        header = _parse_tsv_record(header_raw, context="support TSV header")
        if len(header) != len(set(header)):
            raise SupportPartitionError("support TSV has duplicate header columns")
        required = {"path", "class", "bucket", "emb_rel_path"}
        if not required.issubset(header):
            raise SupportPartitionError("support TSV is missing required columns")
        columns = {name: header.index(name) for name in required}
        audit_digest.update(header_raw)
        audit_size += len(header_raw)
        if write_handle is not None:
            write_handle.write(header_raw)
        for line_number, raw in enumerate(handle, start=2):
            input_digest.update(raw)
            values = _parse_tsv_record(raw, context=f"support TSV:{line_number}")
            if len(values) != len(header):
                raise SupportPartitionError(
                    f"support TSV:{line_number}: column count drifted"
                )
            total_rows += 1
            bucket = values[columns["bucket"]]
            bucket_counts[bucket] += 1
            if bucket != "clean":
                continue
            clean_rows += 1
            mirror, raw_source, raw_source_class, raw_source_image_id = _raw_mirror_path(
                values=values,
                columns=columns,
                support_image_root=support_image_root,
                context=f"support TSV:{line_number}",
            )
            candidate = candidate_by_path.get(mirror)
            if candidate is None:
                continue
            if mirror in links:
                raise SupportPartitionError(f"cache path has duplicate raw rows: {mirror}")
            source_class = values[columns["class"]]
            mapped = name_to_canonical_id.get(_norm_text(source_class))
            if mapped != candidate.source_cache_class_id:
                raise SupportPartitionError(
                    f"support TSV:{line_number}: raw class maps to {mapped}, "
                    f"cache binds {candidate.source_cache_class_id}"
                )
            if (
                raw_source != candidate.source
                or raw_source_class != candidate.source_class
                or raw_source_image_id != candidate.source_image_id
            ):
                raise SupportPartitionError(
                    f"support TSV:{line_number}: raw/cache source identity drifted"
                )
            link = RawLink(
                line_number=line_number,
                row_sha256=_sha256_bytes(raw),
                source_class=source_class,
            )
            links[mirror] = link
            if mirror in retained_paths:
                audit_digest.update(raw)
                audit_size += len(raw)
                audit_rows += 1
                if write_handle is not None:
                    write_handle.write(raw)
    if input_digest.hexdigest() != expected_input_sha256:
        raise SupportPartitionError("support TSV changed while scanning")
    if total_rows != expected_rows or clean_rows != expected_clean_rows:
        raise SupportPartitionError(
            "support TSV row counts drifted: "
            f"rows={total_rows}/{expected_rows}, clean={clean_rows}/{expected_clean_rows}"
        )
    missing = set(candidate_by_path) - set(links)
    if missing:
        sample = sorted(str(path) for path in missing)[:3]
        raise SupportPartitionError(
            f"sealed cache has {len(missing)} candidates without one raw clean row: {sample}"
        )
    if audit_rows != len(retained_paths):
        raise SupportPartitionError("audit TSV row count does not match retained cache")
    if write_handle is not None:
        write_handle.flush()
        os.fsync(write_handle.fileno())
    return links, {
        "rows": total_rows,
        "clean_rows": clean_rows,
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "cache_matched_rows": len(links),
        "input_stream_sha256": input_digest.hexdigest(),
    }, _predicted_record(
        output_path,
        digest=audit_digest,
        size=audit_size,
        rows=audit_rows,
    )


def _runtime_line(candidate: Candidate, link: RawLink) -> bytes:
    values = (
        str(candidate.path),
        str(candidate.class_id),
        str(candidate.source_cache_class_id),
        candidate.class_assignment,
        candidate.source,
        str(candidate.source_image_id),
        "" if candidate.coco_id is None else str(candidate.coco_id),
        link.source_class,
        str(link.line_number),
        link.row_sha256,
    )
    if any("\t" in value or "\r" in value or "\n" in value for value in values):
        raise SupportPartitionError("runtime support field contains TSV control bytes")
    return ("\t".join(values) + "\n").encode("utf-8")


def _stream_runtime(
    *,
    candidates: Sequence[Candidate],
    retained_paths: frozenset[Path],
    raw_links: Mapping[Path, RawLink],
    output_path: Path,
    write_handle: BinaryIO | None,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    header = ("\t".join(RUNTIME_COLUMNS) + "\n").encode("ascii")
    digest.update(header)
    size = len(header)
    rows = 0
    if write_handle is not None:
        write_handle.write(header)
    for candidate in candidates:
        if candidate.path not in retained_paths:
            raise SupportPartitionError(
                f"runtime candidate is not in the filtered base path set: {candidate.path}"
            )
        link = raw_links.get(candidate.path)
        if link is None:
            raise SupportPartitionError(f"retained candidate has no raw link: {candidate.path}")
        raw = _runtime_line(candidate, link)
        digest.update(raw)
        size += len(raw)
        rows += 1
        if write_handle is not None:
            write_handle.write(raw)
    if rows != len(candidates):
        raise SupportPartitionError("runtime TSV row count does not match runtime candidates")
    if write_handle is not None:
        write_handle.flush()
        os.fsync(write_handle.fileno())
    return _predicted_record(output_path, digest=digest, size=size, rows=rows)


def _counter_by_class(candidates: Sequence[Candidate]) -> dict[str, int]:
    counts = Counter(candidate.class_id for candidate in candidates)
    return {str(class_id): counts[class_id] for class_id in sorted(counts)}


def _counter_by_source(candidates: Sequence[Candidate]) -> dict[str, int]:
    counts = Counter(candidate.source for candidate in candidates)
    return dict(sorted(counts.items()))


def _compact_alias(value: str) -> str:
    return "".join(character for character in _norm_text(value) if character.isalnum())


def _build_runtime_candidates(
    *,
    retained_base: Sequence[Candidate],
    required_class_ids: frozenset[int],
    canonical_names_by_id: Mapping[int, frozenset[str]],
) -> tuple[tuple[Candidate, ...], tuple[dict[str, Any], ...]]:
    base_by_class: dict[int, list[Candidate]] = {}
    for candidate in retained_base:
        if (
            candidate.class_id != candidate.source_cache_class_id
            or candidate.class_assignment != "sealed_cache_identity_v1"
        ):
            raise SupportPartitionError("base candidate class provenance drifted")
        base_by_class.setdefault(candidate.class_id, []).append(candidate)
    missing = sorted(required_class_ids - set(base_by_class))
    bridges: list[dict[str, Any]] = []
    runtime_by_class = {class_id: list(values) for class_id, values in base_by_class.items()}
    available_ids = frozenset(base_by_class)
    for target_class_id in missing:
        target_names = canonical_names_by_id.get(target_class_id)
        if not target_names:
            raise SupportPartitionError(
                f"missing training class {target_class_id} has no canonical names"
            )
        target_keys = {
            _compact_alias(name)
            for name in target_names
            if len(_compact_alias(name)) >= 4
        }
        matches: list[tuple[int, tuple[str, ...]]] = []
        for source_class_id in sorted(available_ids):
            source_names = canonical_names_by_id.get(source_class_id, frozenset())
            source_keys = {
                _compact_alias(name)
                for name in source_names
                if len(_compact_alias(name)) >= 4
            }
            overlap = tuple(sorted(target_keys.intersection(source_keys)))
            if overlap:
                matches.append((source_class_id, overlap))
        if len(matches) != 1:
            raise SupportPartitionError(
                "missing training class does not have exactly one compact canonical "
                f"alias source: target={target_class_id}, matches={matches}"
            )
        source_class_id, compact_aliases = matches[0]
        source_candidates = base_by_class[source_class_id]
        bridged = [
            replace(
                candidate,
                class_id=target_class_id,
                class_assignment="canonical_compact_alias_bridge_v1",
            )
            for candidate in source_candidates
        ]
        runtime_by_class[target_class_id] = bridged
        path_digest = hashlib.sha256()
        for candidate in bridged:
            path_digest.update((str(candidate.path) + "\n").encode("utf-8"))
        bridges.append(
            {
                "target_class_id": target_class_id,
                "source_cache_class_id": source_class_id,
                "compact_aliases": list(compact_aliases),
                "target_canonical_names": sorted(target_names),
                "source_canonical_names": sorted(
                    canonical_names_by_id[source_class_id]
                ),
                "candidate_rows": len(bridged),
                "ordered_reused_path_stream_sha256": path_digest.hexdigest(),
                "new_unique_paths": 0,
            }
        )
    runtime = tuple(
        candidate
        for class_id in sorted(runtime_by_class)
        for candidate in runtime_by_class[class_id]
    )
    if not set(required_class_ids).issubset(
        {candidate.class_id for candidate in runtime}
    ):
        raise SupportPartitionError("canonical alias bridge did not cover training classes")
    if not set(candidate.path for candidate in runtime).issubset(
        {candidate.path for candidate in retained_base}
    ):
        raise SupportPartitionError("canonical alias bridge introduced a new support path")
    return runtime, tuple(bridges)


def make_plan(
    *,
    support_tsv: Path = SUPPORT_TSV,
    support_cache: Path = SUPPORT_CACHE,
    support_image_root: Path = SUPPORT_IMAGE_ROOT,
    canonical_classes: Path = CANONICAL_CLASSES,
    vg_metadata_zip: Path = VG_METADATA_ZIP,
    vg_metadata_json: Path = VG_METADATA_JSON,
    partition_receipt: Path = PARTITION_RECEIPT,
    output_root: Path = OUTPUT_ROOT,
    expected_sha256: Mapping[str, str] = EXPECTED_SHA256,
    expected_partition_receipt_sha256: str = EXPECTED_PARTITION_RECEIPT_SHA256,
    expected_raw_rows: int = EXPECTED_RAW_ROWS,
    expected_raw_clean_rows: int = EXPECTED_RAW_CLEAN_ROWS,
    expected_cache_classes: int = EXPECTED_CACHE_CLASSES,
    expected_cache_candidates: int = EXPECTED_CACHE_CANDIDATES,
    expected_training_classes: int = EXPECTED_TRAINING_CLASSES,
) -> BuildPlan:
    required_hashes = set(EXPECTED_SHA256)
    if set(expected_sha256) != required_hashes:
        raise SupportPartitionError("expected SHA-256 map keys drifted")
    paths = {
        "support_tsv": support_tsv,
        "support_cache": support_cache,
        "canonical_classes": canonical_classes,
        "vg_metadata_zip": vg_metadata_zip,
        "vg_metadata_json": vg_metadata_json,
    }
    input_records: dict[str, dict[str, Any]] = {}
    for name, raw_path in paths.items():
        expected = _validate_sha256(expected_sha256[name], label=f"expected {name} hash")
        record = _file_record(raw_path)
        if record["sha256"] != expected:
            raise SupportPartitionError(
                f"{name} SHA-256 mismatch: expected={expected}, observed={record['sha256']}"
            )
        input_records[name] = record
    support_tsv = Path(input_records["support_tsv"]["path"])
    support_cache = Path(input_records["support_cache"]["path"])
    canonical_classes = Path(input_records["canonical_classes"]["path"])
    vg_metadata_zip = Path(input_records["vg_metadata_zip"]["path"])
    vg_metadata_json = Path(input_records["vg_metadata_json"]["path"])
    support_image_root = support_image_root.expanduser().resolve(strict=True)
    output_root = output_root.expanduser().resolve()

    partition = _validate_partition_receipt(
        partition_receipt.expanduser().resolve(strict=True),
        expected_sha256=expected_partition_receipt_sha256,
        expected_training_classes=expected_training_classes,
    )
    input_records["partition_receipt"] = dict(
        partition.receipt_summary["receipt"]
    )
    vg_mapping, vg_summary = _load_vg_mapping(
        vg_json=vg_metadata_json, vg_zip=vg_metadata_zip
    )
    name_to_id, id_to_name, canonical_names_by_id, canonical_summary = _load_canonical_classes(
        canonical_classes
    )
    cache = _load_cache(
        cache_path=support_cache,
        support_tsv=support_tsv,
        canonical_classes=canonical_classes,
        support_image_root=support_image_root,
        vg_mapping=vg_mapping,
        canonical_ids=frozenset(id_to_name),
        expected_classes=expected_cache_classes,
        expected_candidates=expected_cache_candidates,
    )
    retained = tuple(
        candidate
        for candidate in cache.candidates
        if candidate.coco_id is None or candidate.coco_id not in partition.excluded_ids
    )
    retained_paths = frozenset(candidate.path for candidate in retained)
    excluded = tuple(
        candidate
        for candidate in cache.candidates
        if candidate.path not in retained_paths
    )
    if len(retained_paths) != len(retained):
        raise SupportPartitionError("retained candidate paths are not unique")
    runtime_candidates, alias_bridges = _build_runtime_candidates(
        retained_base=retained,
        required_class_ids=partition.training_class_ids,
        canonical_names_by_id=canonical_names_by_id,
    )

    raw_links, raw_summary, audit_record = _scan_raw_tsv(
        support_tsv=support_tsv,
        support_image_root=support_image_root,
        candidates=cache.candidates,
        retained_paths=retained_paths,
        name_to_canonical_id=name_to_id,
        output_path=output_root / OUTPUT_AUDIT_TSV,
        write_handle=None,
        expected_input_sha256=input_records["support_tsv"]["sha256"],
        expected_rows=expected_raw_rows,
        expected_clean_rows=expected_raw_clean_rows,
    )
    runtime_record = _stream_runtime(
        candidates=runtime_candidates,
        retained_paths=retained_paths,
        raw_links=raw_links,
        output_path=output_root / OUTPUT_RUNTIME_TSV,
        write_handle=None,
    )

    retained_classes = frozenset(candidate.class_id for candidate in retained)
    runtime_classes = frozenset(candidate.class_id for candidate in runtime_candidates)
    missing_training_classes = sorted(
        partition.training_class_ids - runtime_classes
    )
    if missing_training_classes:
        raise SupportPartitionError(
            f"filtered support loses D1 training classes: {missing_training_classes}"
        )
    excluded_coco_ids = sorted(
        {candidate.coco_id for candidate in excluded if candidate.coco_id is not None}
    )
    exclusion_union = sorted(partition.excluded_ids)
    exclusion_digest = hashlib.sha256()
    for image_id in exclusion_union:
        exclusion_digest.update(f"{image_id}\n".encode("ascii"))
    actual_excluded_digest = hashlib.sha256()
    for image_id in excluded_coco_ids:
        actual_excluded_digest.update(f"{image_id}\n".encode("ascii"))

    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "builder": _file_record(Path(__file__)),
        "inputs": input_records,
        "partition": partition.receipt_summary,
        "vg_mapping": vg_summary,
        "canonical_mapping": canonical_summary,
        "sealed_cache": cache.receipt_summary,
        "raw_tsv_scan": raw_summary,
        "filter_contract": {
            "candidate_source": "sealed image cache",
            "candidate_stream_policy": STREAM_POLICY,
            "exclusion_policy": EXCLUSION_POLICY,
            "cross_coco_release_identity": "numeric coco_id",
            "vg_null_coco_id_policy": "retain",
            "cache_class_iteration": "sorted integer class_id",
            "within_class_iteration": "sealed candidate list order",
            "missing_training_class_policy": (
                "unique_compact_canonical_alias_bridge_reusing_filtered_paths_v1"
            ),
            "bank_consumers": ["D0", "D1"],
            "D0_and_D1_share_identical_runtime_bank": True,
            "runtime_path_policy": "absolute existing clean-mirror JPEG",
            "runtime_columns": list(RUNTIME_COLUMNS),
            "required_dataset_settings": {
                "patch_bank_cache": False,
                "patch_bank_cache_write": False,
                "support_patch_use_embedding": False,
                "support_patch_max_per_class": 200,
            },
        },
        "exclusion": {
            "dev_full_numeric_coco_ids": sorted(partition.dev_full_ids),
            "official_ref8_numeric_coco_ids": sorted(partition.official_ref8_ids),
            "union_numeric_coco_ids": exclusion_union,
            "union_numeric_coco_id_count": len(exclusion_union),
            "union_numeric_coco_id_stream_sha256": exclusion_digest.hexdigest(),
            "actually_excluded_numeric_coco_ids": excluded_coco_ids,
            "actually_excluded_numeric_coco_id_count": len(excluded_coco_ids),
            "actually_excluded_numeric_coco_id_stream_sha256": (
                actual_excluded_digest.hexdigest()
            ),
            "excluded_candidate_rows": len(excluded),
            "excluded_source_counts": _counter_by_source(excluded),
            "excluded_class_counts": _counter_by_class(excluded),
            "excluded_lvis_source_image_ids": sorted(
                {c.source_image_id for c in excluded if c.source == "lvis"}
            ),
            "excluded_vg_source_image_ids": sorted(
                {c.source_image_id for c in excluded if c.source == "vg"}
            ),
        },
        "retained": {
            "role": "filtered_base_cache_before_alias_bridge",
            "candidate_rows": len(retained),
            "source_counts": _counter_by_source(retained),
            "class_count": len(retained_classes),
            "class_counts": _counter_by_class(retained),
            "null_coco_id_rows": sum(c.coco_id is None for c in retained),
        },
        "alias_bridges": list(alias_bridges),
        "runtime_bank": {
            "candidate_rows": len(runtime_candidates),
            "unique_paths": len({candidate.path for candidate in runtime_candidates}),
            "alias_duplicated_rows": len(runtime_candidates) - len(retained),
            "source_counts": _counter_by_source(runtime_candidates),
            "class_count": len(runtime_classes),
            "class_counts": _counter_by_class(runtime_candidates),
        },
        "training_class_coverage": {
            "required_class_count": len(partition.training_class_ids),
            "required_class_ids": sorted(partition.training_class_ids),
            "covered_class_count": len(partition.training_class_ids & runtime_classes),
            "covered_class_ids": sorted(partition.training_class_ids & runtime_classes),
            "missing_class_ids": missing_training_classes,
            "support_counts": {
                str(class_id): sum(c.class_id == class_id for c in runtime_candidates)
                for class_id in sorted(partition.training_class_ids)
            },
        },
        "outputs": {
            "runtime_support_tsv": runtime_record,
            "audit_raw_tsv": audit_record,
        },
        "invariants": {
            "all_preregistered_inputs_match_sha256": True,
            "partition_receipt_canonical_payload_replays": True,
            "official_ref8_manifests_replay_partition_receipt": True,
            "d1_train_manifests_replay_partition_receipt": True,
            "vg_zip_member_equals_standalone_json_byte_for_byte": True,
            "every_cached_vg_image_id_has_official_metadata": True,
            "vg_null_coco_ids_are_retained": all(
                candidate.path in retained_paths
                for candidate in cache.candidates
                if candidate.source == "vg" and candidate.coco_id is None
            ),
            "every_cache_candidate_has_one_raw_clean_row": (
                len(raw_links) == len(cache.candidates)
            ),
            "audit_output_is_raw_header_plus_original_row_subsequence": True,
            "runtime_base_stream_is_sealed_cache_order_delete_only": True,
            "alias_bridges_are_unique_canonical_metadata_matches": all(
                bridge["compact_aliases"]
                and bridge["new_unique_paths"] == 0
                for bridge in alias_bridges
            ),
            "alias_bridges_reuse_only_filtered_base_paths": {
                candidate.path for candidate in runtime_candidates
            }.issubset(retained_paths),
            "runtime_paths_are_absolute_existing_clean_mirror_jpegs": True,
            "runtime_rows_have_integer_canonical_class_ids": True,
            "no_runtime_candidate_has_an_excluded_coco_id": all(
                candidate.coco_id is None
                or candidate.coco_id not in partition.excluded_ids
                for candidate in runtime_candidates
            ),
            "all_d0_d1_training_classes_have_runtime_support": not missing_training_classes,
            "each_runtime_class_has_at_most_200_candidates": all(
                count <= 200
                for count in _counter_by_class(runtime_candidates).values()
            ),
            "audit_rows_exactly_cover_filtered_base_cache": (
                audit_record["rows"] == len(retained)
            ),
            "runtime_row_delta_equals_explicit_alias_bridge_rows": (
                runtime_record["rows"] - audit_record["rows"]
                == sum(bridge["candidate_rows"] for bridge in alias_bridges)
            ),
            "D0_and_D1_use_the_identical_runtime_support_tsv": True,
            "cache_reads_and_writes_must_be_disabled_for_formal_training": True,
        },
    }
    if any(value is not True for value in receipt["invariants"].values()):
        raise SupportPartitionError("one or more support partition invariants failed")
    receipt["canonical_payload_sha256"] = _sha256_bytes(_canonical_bytes(receipt))
    return BuildPlan(
        receipt=receipt,
        candidates=cache.candidates,
        runtime_candidates=runtime_candidates,
        retained_paths=retained_paths,
        raw_links=raw_links,
        name_to_canonical_id=name_to_id,
        input_records=input_records,
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


def _assert_inputs_unchanged(records: Mapping[str, Mapping[str, Any]]) -> None:
    for name, sealed in records.items():
        observed = _file_record(Path(str(sealed["path"])))
        if observed != dict(sealed):
            raise SupportPartitionError(f"{name} changed before atomic commit")


def build(**kwargs: Any) -> dict[str, Any]:
    output_root = Path(kwargs.get("output_root", OUTPUT_ROOT)).expanduser().resolve()
    if output_root.exists():
        raise SupportPartitionError(f"refusing to replace existing output root: {output_root}")
    plan = make_plan(**kwargs)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=str(output_root.parent))
    )
    committed = False
    try:
        audit_path = temporary_root / OUTPUT_AUDIT_TSV
        with audit_path.open("xb") as audit_handle:
            links, _raw_summary, audit_record = _scan_raw_tsv(
                support_tsv=Path(plan.input_records["support_tsv"]["path"]),
                support_image_root=Path(
                    kwargs.get("support_image_root", SUPPORT_IMAGE_ROOT)
                ).expanduser().resolve(strict=True),
                candidates=plan.candidates,
                retained_paths=plan.retained_paths,
                name_to_canonical_id=plan.name_to_canonical_id,
                output_path=output_root / OUTPUT_AUDIT_TSV,
                write_handle=audit_handle,
                expected_input_sha256=plan.input_records["support_tsv"]["sha256"],
                expected_rows=int(kwargs.get("expected_raw_rows", EXPECTED_RAW_ROWS)),
                expected_clean_rows=int(
                    kwargs.get("expected_raw_clean_rows", EXPECTED_RAW_CLEAN_ROWS)
                ),
            )
        if links != plan.raw_links or audit_record != plan.receipt["outputs"]["audit_raw_tsv"]:
            raise SupportPartitionError("audit output changed between plan and build")
        runtime_path = temporary_root / OUTPUT_RUNTIME_TSV
        with runtime_path.open("xb") as runtime_handle:
            runtime_record = _stream_runtime(
                candidates=plan.runtime_candidates,
                retained_paths=plan.retained_paths,
                raw_links=plan.raw_links,
                output_path=output_root / OUTPUT_RUNTIME_TSV,
                write_handle=runtime_handle,
            )
        if runtime_record != plan.receipt["outputs"]["runtime_support_tsv"]:
            raise SupportPartitionError("runtime output changed between plan and build")
        receipt_path = temporary_root / "receipt.json"
        with receipt_path.open("xb") as handle:
            handle.write(_receipt_bytes(plan.receipt))
            handle.flush()
            os.fsync(handle.fileno())
        _assert_inputs_unchanged(plan.input_records)
        _fsync_directory(temporary_root)
        os.replace(temporary_root, output_root)
        committed = True
        _fsync_directory(output_root.parent)
    finally:
        if not committed:
            shutil.rmtree(temporary_root, ignore_errors=True)
    return plan.receipt


def verify(**kwargs: Any) -> dict[str, Any]:
    output_root = Path(kwargs.get("output_root", OUTPUT_ROOT)).expanduser().resolve(strict=True)
    if not output_root.is_dir():
        raise SupportPartitionError(f"output root is not a directory: {output_root}")
    plan = make_plan(**kwargs)
    receipt_path = output_root / "receipt.json"
    observed_receipt = _load_json(receipt_path, label="support partition receipt")
    if observed_receipt != plan.receipt:
        raise SupportPartitionError("on-disk support partition receipt drifted")
    for key, filename in (
        ("runtime_support_tsv", OUTPUT_RUNTIME_TSV),
        ("audit_raw_tsv", OUTPUT_AUDIT_TSV),
    ):
        observed = _file_record(output_root / filename)
        sealed = plan.receipt["outputs"][key]
        if (
            observed["path"] != sealed["path"]
            or observed["size_bytes"] != sealed["size_bytes"]
            or observed["sha256"] != sealed["sha256"]
        ):
            raise SupportPartitionError(f"{key} output drifted")
    return plan.receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-tsv", type=Path, default=SUPPORT_TSV)
    parser.add_argument("--support-cache", type=Path, default=SUPPORT_CACHE)
    parser.add_argument("--support-image-root", type=Path, default=SUPPORT_IMAGE_ROOT)
    parser.add_argument("--canonical-classes", type=Path, default=CANONICAL_CLASSES)
    parser.add_argument("--vg-metadata-zip", type=Path, default=VG_METADATA_ZIP)
    parser.add_argument("--vg-metadata-json", type=Path, default=VG_METADATA_JSON)
    parser.add_argument("--partition-receipt", type=Path, default=PARTITION_RECEIPT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--expected-partition-receipt-sha256",
        default=EXPECTED_PARTITION_RECEIPT_SHA256,
    )
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kwargs = {
        "support_tsv": args.support_tsv,
        "support_cache": args.support_cache,
        "support_image_root": args.support_image_root,
        "canonical_classes": args.canonical_classes,
        "vg_metadata_zip": args.vg_metadata_zip,
        "vg_metadata_json": args.vg_metadata_json,
        "partition_receipt": args.partition_receipt,
        "output_root": args.output_root,
        "expected_partition_receipt_sha256": (
            args.expected_partition_receipt_sha256
        ),
    }
    receipt = verify(**kwargs) if args.verify else build(**kwargs)
    print(json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
