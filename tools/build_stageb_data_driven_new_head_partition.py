#!/usr/bin/env python3
"""Build the sealed, image-disjoint new-head train/dev partition.

The builder treats the paired DD0 ordinary-primary and DD1 category-complete
manifests as immutable byte streams.  Official Ref8 images are quarantined
before a deterministic, image-level stratified dev sample is drawn.  Every
output JSONL is a byte-for-byte subsequence of its corresponding input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from contextlib import ExitStack
from dataclasses import dataclass, field
from itertools import zip_longest
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.stageb_ref_split_contract import (  # noqa: E402
    REF_SPLIT_CONTRACT,
    REF_SPLIT_MANIFEST_FILES,
    REF_SPLITS,
)


PAIR_ROOT = REPO_ROOT / "data/ablations/stageb_data_driven_ref_pair_20260721"
INPUT_RECEIPT = PAIR_ROOT / "receipt.json"
D0_ROOT = PAIR_ROOT / "dd0_ordinary_primary"
D1_ROOT = (
    REPO_ROOT
    / "data/ablations/stageb_refexp_three_train_category_complete_20260720"
)
HELDOUT_ROOT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/baseline_b58_ref8_seed42/refcoco_eval_inputs"
)
OUTPUT_ROOT = (
    REPO_ROOT
    / "data/ablations/stageb_data_driven_new_head_partition_20260723"
)

RECEIPT_SCHEMA = "pivot.stageb.data_driven.new_head_partition_receipt/v1"
UPSTREAM_RECEIPT_SCHEMA = "pivot.stageb.data_driven_ref_pair_receipt/v1"
D0_ROW_SCHEMA = "pivot.stageb.data_driven_ordinary_primary/v1"
D1_ROW_SCHEMA = "pivot.stageb.u2_category_complete_ref/v1"
STREAM_ENCODING = "raw_input_record_including_original_line_ending_v1"
SELECTION_POLICY = "image_strata_largest_remainder_seeded_sha256_v1"
SELECTION_NAMESPACE = "pivot.stageb.data_driven.new_head_partition/v1"
SEED = 20260723
DEV_IMAGES = 2048
SCREEN_IMAGES = 512
EXPECTED_HELDOUT_UNION_IMAGES = 6549
EXPECTED_INPUT_RECEIPT_SHA256 = (
    "c8fb9144f6ce0457d90d6d0e1f90dfc55c10700817ace5e9d214e88b173326c7"
)
EXPECTED_CATEGORY_RECEIPT_SHA256 = (
    "fab09c61a8f53f05d75eedff25039a843ff27cb2d491d6c6576fe2b1e8aedd74"
)

MANIFEST_SOURCES = (
    ("refcoco_stageb_phrase_v1.jsonl", "refcoco"),
    ("refcocoplus_stageb_phrase_v1.jsonl", "refcocoplus"),
    ("refcocog_stageb_phrase_v1.jsonl", "refcocog"),
)
MANIFESTS = tuple(name for name, _source in MANIFEST_SOURCES)
SOURCE_LABELS = tuple(source for _name, source in MANIFEST_SOURCES)
VARIANTS = ("d0_ordinary_primary", "d1_category_complete")
PARTITIONS = ("train", "dev_full", "dev_screen", "quarantine")
IDENTITY_KEYS = (
    "source",
    "image_id",
    "ann_id",
    "ref_id",
    "sent_id",
    "split",
    "filename",
)

EXPECTED_ROWS = {
    "refcoco_stageb_phrase_v1.jsonl": 120624,
    "refcocoplus_stageb_phrase_v1.jsonl": 120191,
    "refcocog_stageb_phrase_v1.jsonl": 80512,
}
EXPECTED_D0_SHA256 = {
    "refcoco_stageb_phrase_v1.jsonl": (
        "a7bce1712f3256917d1c138e3ab484e4f3ece0858b329eeb10e7904ff0ae18a2"
    ),
    "refcocoplus_stageb_phrase_v1.jsonl": (
        "10e912351dcf4cb892af0b319de1df1a0a23e4f18fd446d6f860be199f170ce1"
    ),
    "refcocog_stageb_phrase_v1.jsonl": (
        "23ac971ee6729b6671b34e750db0a46775e416d3b242be0922f4669f4a45ecce"
    ),
}
EXPECTED_D1_SHA256 = {
    "refcoco_stageb_phrase_v1.jsonl": (
        "44bed1c327df66e91d1e8bc467092324da4a21c41a16ce7a7e7d86c5f977632c"
    ),
    "refcocoplus_stageb_phrase_v1.jsonl": (
        "eedca9b2e4118a09d4a2a8cca70794eec053314a0c83b8de25972202af26cdf2"
    ),
    "refcocog_stageb_phrase_v1.jsonl": (
        "60678c11d1a93f60bc0e5689798527b31d5195b78ce02698270f3598e00a7f69"
    ),
}

_COCO_FILENAME_RE = re.compile(
    r"^COCO_(?P<split>train2014|val2014)_(?P<image_id>[0-9]{12})"
    r"\.(?:jpg|jpeg)$",
    re.IGNORECASE,
)
_HEX64_RE = re.compile(r"[0-9a-f]{64}")

ImageKey = tuple[str, int]
Stratum = tuple[str, int, int]
OutputKey = tuple[str, str, str]


class NewHeadPartitionError(RuntimeError):
    pass


@dataclass(slots=True)
class MutableImageStats:
    source_rows: Counter[str] = field(default_factory=Counter)
    categories: Counter[int] = field(default_factory=Counter)
    expression_rows: int = 0
    filename_basename: str | None = None


@dataclass(frozen=True, slots=True)
class ImageRecord:
    key: ImageKey
    source_membership_mask: str
    dominant_primary_category: int
    expression_rows: int
    expression_log2_bucket: int
    stratum: Stratum


@dataclass(frozen=True, slots=True)
class Selection:
    official_keys: frozenset[ImageKey]
    quarantine_keys: frozenset[ImageKey]
    dev_full_keys: frozenset[ImageKey]
    dev_screen_keys: frozenset[ImageKey]
    image_records: Mapping[ImageKey, ImageRecord]
    strata: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class BuildPlan:
    receipt: dict[str, Any]
    selection: Selection
    source_records: Mapping[str, Mapping[str, Mapping[str, Any]]]


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


def _file_record(path: Path, *, reported_path: Path | None = None) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    if not path.is_file():
        raise NewHeadPartitionError(f"not a file: {path}")
    before = path.stat()
    digest = _sha256_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (
        after.st_size,
        after.st_mtime_ns,
    ):
        raise NewHeadPartitionError(f"file changed while hashing: {path}")
    return {
        "path": str((reported_path or path).expanduser().resolve()),
        "size_bytes": int(before.st_size),
        "sha256": digest,
    }


def _predicted_file_record(
    path: Path,
    *,
    rows: int,
    unique_identities: int,
    unique_image_keys: int,
    ordered_identity_stream_sha256: str,
    size_bytes: int,
    sha256: str,
) -> dict[str, Any]:
    return {
        "path": str(path.expanduser().resolve()),
        "rows": int(rows),
        "unique_identities": int(unique_identities),
        "unique_image_keys": int(unique_image_keys),
        "ordered_identity_stream_sha256": ordered_identity_stream_sha256,
        "size_bytes": int(size_bytes),
        "sha256": sha256,
    }


def _validate_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _HEX64_RE.fullmatch(value) is None:
        raise NewHeadPartitionError(f"{label} is not a lowercase SHA-256")
    return value


def _required_int(value: Any, *, field: str, context: str) -> int:
    if type(value) is not int:
        raise NewHeadPartitionError(f"{context}: {field} must be an exact integer")
    return int(value)


def _required_text(value: Any, *, field: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NewHeadPartitionError(f"{context}: {field} must be non-empty text")
    return value


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NewHeadPartitionError(f"could not load {label}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise NewHeadPartitionError(f"{label} must be a JSON object: {path}")
    return value


def _load_jsonl_row(raw: bytes, *, path: Path, line_number: int) -> dict[str, Any]:
    if not raw.endswith(b"\n"):
        raise NewHeadPartitionError(
            f"input row lacks a terminating LF at {path}:{line_number}"
        )
    if not raw.rstrip(b"\r\n"):
        raise NewHeadPartitionError(f"blank JSONL row at {path}:{line_number}")
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise NewHeadPartitionError(
            f"invalid JSON at {path}:{line_number}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise NewHeadPartitionError(f"row is not an object at {path}:{line_number}")
    return value


def _image_key(row: Mapping[str, Any], *, context: str) -> tuple[ImageKey, str]:
    image_id = _required_int(row.get("image_id"), field="image_id", context=context)
    filename = _required_text(row.get("filename"), field="filename", context=context)
    basename = Path(filename).name
    match = _COCO_FILENAME_RE.fullmatch(basename)
    if match is None:
        raise NewHeadPartitionError(
            f"{context}: filename is not a canonical COCO 2014 image: {filename}"
        )
    filename_image_id = int(match.group("image_id"))
    if filename_image_id != image_id:
        raise NewHeadPartitionError(
            f"{context}: filename/image_id drifted: {filename_image_id} != {image_id}"
        )
    split = match.group("split").lower()
    return (split, image_id), basename


def _identity(row: Mapping[str, Any], *, context: str) -> tuple[Any, ...]:
    values = tuple(row.get(key) for key in IDENTITY_KEYS)
    if any(value is None for value in values):
        raise NewHeadPartitionError(f"{context}: paired identity has a missing field")
    for field in ("image_id", "ann_id", "ref_id", "sent_id"):
        _required_int(row.get(field), field=field, context=context)
    for field in ("source", "split", "filename"):
        _required_text(row.get(field), field=field, context=context)
    return values


def _paired_primary(
    d0: Mapping[str, Any], d1: Mapping[str, Any], *, context: str
) -> tuple[dict[str, Any], int, int]:
    d0_instances = d0.get("instances")
    if (
        d0.get("primary_support_instance_index") != 0
        or d0.get("stage_b_data_driven_ordinary_primary") is not True
        or d0.get("stage_b_data_driven_ordinary_primary_schema") != D0_ROW_SCHEMA
        or not isinstance(d0_instances, list)
        or len(d0_instances) != 1
        or not isinstance(d0_instances[0], dict)
    ):
        raise NewHeadPartitionError(f"{context}: D0 ordinary-primary contract drifted")
    source_row_sha = d0.get("stage_b_data_driven_source_row_sha256")
    _validate_sha256(source_row_sha, label=f"{context} source-row hash")

    d1_instances = d1.get("instances")
    if (
        d1.get("primary_support_instance_index") != 0
        or d1.get("stage_b_u2_category_complete") is not True
        or d1.get("stage_b_u2_category_complete_schema") != D1_ROW_SCHEMA
        or not isinstance(d1_instances, list)
        or not d1_instances
        or any(not isinstance(instance, dict) for instance in d1_instances)
    ):
        raise NewHeadPartitionError(f"{context}: D1 category-complete contract drifted")
    d0_primary = dict(d0_instances[0])
    d1_primary = dict(d1_instances[0])
    if d1_primary.pop("category_complete_primary", None) is not True:
        raise NewHeadPartitionError(f"{context}: D1 primary marker drifted")
    d1_primary_ann_id = d1_primary.pop("coco_ann_id", None)
    if d1_primary_ann_id != d1.get("ann_id"):
        raise NewHeadPartitionError(f"{context}: D1 primary annotation id drifted")
    if d0_primary != d1_primary:
        raise NewHeadPartitionError(f"{context}: paired primary instance drifted")
    class_id = _required_int(
        d0_primary.get("class_id"), field="instances[0].class_id", context=context
    )
    if any(instance.get("class_id") != class_id for instance in d1_instances):
        raise NewHeadPartitionError(f"{context}: D1 contains another category")
    return d0_primary, class_id, len(d1_instances)


def _validate_expected_maps(
    *,
    expected_rows: Mapping[str, int],
    expected_d0_sha256: Mapping[str, str],
    expected_d1_sha256: Mapping[str, str],
) -> None:
    required = set(MANIFESTS)
    if (
        set(expected_rows) != required
        or set(expected_d0_sha256) != required
        or set(expected_d1_sha256) != required
    ):
        raise NewHeadPartitionError("expected input maps must exactly match manifests")
    for name in MANIFESTS:
        if type(expected_rows[name]) is not int or expected_rows[name] <= 0:
            raise NewHeadPartitionError(f"invalid expected row count for {name}")
        _validate_sha256(expected_d0_sha256[name], label=f"{name} D0 expected hash")
        _validate_sha256(expected_d1_sha256[name], label=f"{name} D1 expected hash")


def _validate_upstream_receipt(
    *,
    input_receipt: Path,
    d0_root: Path,
    d1_root: Path,
    expected_input_receipt_sha256: str,
    expected_category_receipt_sha256: str,
    expected_rows: Mapping[str, int],
    expected_d0_sha256: Mapping[str, str],
    expected_d1_sha256: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, Any]]]]:
    _validate_sha256(
        expected_input_receipt_sha256, label="expected pair receipt hash"
    )
    receipt_record = _file_record(input_receipt)
    if receipt_record["sha256"] != expected_input_receipt_sha256:
        raise NewHeadPartitionError(
            "pair receipt SHA-256 mismatch: "
            f"expected={expected_input_receipt_sha256}, "
            f"observed={receipt_record['sha256']}"
        )
    receipt = _load_json_object(input_receipt, label="pair receipt")
    expected_total = sum(expected_rows.values())
    invariants = receipt.get("invariants")
    if (
        receipt.get("schema") != UPSTREAM_RECEIPT_SCHEMA
        or receipt.get("ordinary_row_schema") != D0_ROW_SCHEMA
        or receipt.get("rows") != expected_total
        or receipt.get("unique_identities") != expected_total
        or not isinstance(receipt.get("category_complete_instances"), int)
        or not isinstance(invariants, dict)
        or not invariants
        or any(value is not True for value in invariants.values())
    ):
        raise NewHeadPartitionError("pair receipt contract drifted")
    for field in (
        "ordered_identity_stream_sha256",
        "source_primary_stream_sha256",
        "source_row_sha256_stream_sha256",
    ):
        _validate_sha256(receipt.get(field), label=f"pair receipt {field}")

    manifests = receipt.get("manifests")
    if not isinstance(manifests, dict) or set(manifests) != set(MANIFESTS):
        raise NewHeadPartitionError("pair receipt manifest set drifted")
    source_records: dict[str, dict[str, dict[str, Any]]] = {}
    for name in MANIFESTS:
        entry = manifests[name]
        if not isinstance(entry, dict) or entry.get("rows") != expected_rows[name]:
            raise NewHeadPartitionError(f"pair receipt row count drifted for {name}")
        if type(entry.get("complete_instances")) is not int or entry[
            "complete_instances"
        ] <= 0:
            raise NewHeadPartitionError(
                f"pair receipt complete-instance count drifted for {name}"
            )
        d0_record = _file_record(d0_root / name)
        d1_record = _file_record(d1_root / name)
        if d0_record["sha256"] != expected_d0_sha256[name]:
            raise NewHeadPartitionError(f"D0 SHA-256 drifted for {name}")
        if d1_record["sha256"] != expected_d1_sha256[name]:
            raise NewHeadPartitionError(f"D1 SHA-256 drifted for {name}")
        for role, observed in (
            ("ordinary_primary", d0_record),
            ("category_complete", d1_record),
        ):
            sealed = entry.get(role)
            if (
                not isinstance(sealed, dict)
                or sealed.get("sha256") != observed["sha256"]
                or sealed.get("size_bytes") != observed["size_bytes"]
                or not isinstance(sealed.get("path"), str)
                or Path(sealed["path"]).expanduser().resolve()
                != Path(observed["path"])
            ):
                raise NewHeadPartitionError(
                    f"pair receipt {role} binding drifted for {name}"
                )
        source_records[name] = {"d0": d0_record, "d1": d1_record}

    category_record = receipt.get("category_complete_receipt")
    if not isinstance(category_record, dict):
        raise NewHeadPartitionError("pair receipt lost category-complete receipt")
    category_path_value = category_record.get("path")
    if not isinstance(category_path_value, str) or not category_path_value.strip():
        raise NewHeadPartitionError("category-complete receipt path is invalid")
    category_path = Path(category_path_value).expanduser().resolve(strict=True)
    observed_category = _file_record(category_path)
    _validate_sha256(
        expected_category_receipt_sha256,
        label="expected category-complete receipt hash",
    )
    if (
        category_record.get("sha256") != expected_category_receipt_sha256
        or category_record.get("size_bytes") != observed_category["size_bytes"]
        or observed_category["sha256"] != expected_category_receipt_sha256
    ):
        raise NewHeadPartitionError("category-complete receipt binding drifted")
    return {
        "record": receipt_record,
        "category_complete_receipt": observed_category,
        "sealed_streams": {
            field: receipt[field]
            for field in (
                "ordered_identity_stream_sha256",
                "source_primary_stream_sha256",
                "source_row_sha256_stream_sha256",
            )
        },
        "rows": receipt["rows"],
        "unique_identities": receipt["unique_identities"],
        "category_complete_instances": receipt["category_complete_instances"],
    }, source_records


def _load_official_keys(
    *,
    heldout_root: Path,
    heldout_contract: Mapping[str, Mapping[str, Any]],
    heldout_manifest_files: Mapping[str, str],
    heldout_splits: Sequence[str],
    expected_heldout_union_images: int,
) -> tuple[frozenset[ImageKey], dict[str, Any]]:
    if (
        tuple(heldout_contract) != tuple(heldout_splits)
        or set(heldout_manifest_files) != set(heldout_splits)
    ):
        raise NewHeadPartitionError("official Ref8 contract is incomplete or reordered")
    keys: set[ImageKey] = set()
    splits: dict[str, Any] = {}
    total_rows = 0
    for split in heldout_splits:
        contract = heldout_contract[split]
        filename = heldout_manifest_files[split]
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise NewHeadPartitionError(f"official filename is invalid for {split}")
        expected_rows = contract.get("rows")
        expected_sha = contract.get("sha256")
        if type(expected_rows) is not int or expected_rows <= 0:
            raise NewHeadPartitionError(f"official row count is invalid for {split}")
        _validate_sha256(expected_sha, label=f"{split} official hash")
        path = heldout_root / filename
        record = _file_record(path)
        if record["sha256"] != expected_sha:
            raise NewHeadPartitionError(
                f"official Ref8 manifest SHA-256 drifted for {split}"
            )
        rows = 0
        with path.open("rb") as handle:
            for line_number, raw in enumerate(handle, start=1):
                row = _load_jsonl_row(raw, path=path, line_number=line_number)
                key, _basename = _image_key(
                    row, context=f"official {split}:{line_number}"
                )
                keys.add(key)
                rows += 1
        if rows != expected_rows:
            raise NewHeadPartitionError(
                f"official Ref8 row count drifted for {split}: "
                f"{rows} != {expected_rows}"
            )
        total_rows += rows
        splits[split] = {"rows": rows, "manifest": record}
    if len(keys) != expected_heldout_union_images:
        raise NewHeadPartitionError(
            "official Ref8 image-key union drifted: "
            f"observed={len(keys)}, expected={expected_heldout_union_images}"
        )
    ordered_keys = [_image_key_value(key) for key in sorted(keys)]
    return frozenset(keys), {
        "contract_schema": "pivot.stageb.official_ref_split_contract/v1",
        "split_order": list(heldout_splits),
        "splits": splits,
        "rows": total_rows,
        "unique_image_keys": len(keys),
        "ordered_image_key_stream_sha256": _record_stream_sha256(ordered_keys),
    }


def _scan_paired_inputs(
    *,
    d0_root: Path,
    d1_root: Path,
    expected_rows: Mapping[str, int],
    source_records: Mapping[str, Mapping[str, Mapping[str, Any]]],
    upstream_receipt: Mapping[str, Any],
) -> tuple[dict[ImageKey, MutableImageStats], dict[str, Any]]:
    images: dict[ImageKey, MutableImageStats] = {}
    seen_identity_digests: set[bytes] = set()
    identity_digest = hashlib.sha256()
    primary_digest = hashlib.sha256()
    source_row_digest = hashlib.sha256()
    total_rows = 0
    total_complete_instances = 0
    per_manifest: dict[str, Any] = {}
    source_by_manifest = dict(MANIFEST_SOURCES)

    for name in MANIFESTS:
        d0_path = d0_root / name
        d1_path = d1_root / name
        rows = 0
        complete_instances = 0
        d0_digest = hashlib.sha256()
        d1_digest = hashlib.sha256()
        with d0_path.open("rb") as d0_handle, d1_path.open("rb") as d1_handle:
            for line_number, pair in enumerate(
                zip_longest(d0_handle, d1_handle), start=1
            ):
                d0_raw, d1_raw = pair
                if d0_raw is None or d1_raw is None:
                    raise NewHeadPartitionError(
                        f"{name}: D0/D1 row counts differ at line {line_number}"
                    )
                d0_digest.update(d0_raw)
                d1_digest.update(d1_raw)
                d0 = _load_jsonl_row(d0_raw, path=d0_path, line_number=line_number)
                d1 = _load_jsonl_row(d1_raw, path=d1_path, line_number=line_number)
                context = f"{name}:{line_number}"
                d0_identity = _identity(d0, context=f"{context} D0")
                d1_identity = _identity(d1, context=f"{context} D1")
                if d0_identity != d1_identity:
                    raise NewHeadPartitionError(f"{context}: paired identity drifted")
                d0_key, d0_basename = _image_key(d0, context=f"{context} D0")
                d1_key, d1_basename = _image_key(d1, context=f"{context} D1")
                if d0_key != d1_key or d0_basename != d1_basename:
                    raise NewHeadPartitionError(
                        f"{context}: paired filename/image_id drifted"
                    )
                d0_primary, class_id, instance_count = _paired_primary(
                    d0, d1, context=context
                )
                identity_bytes = _canonical_bytes(d0_identity)
                identity_member_digest = hashlib.sha256(identity_bytes).digest()
                if identity_member_digest in seen_identity_digests:
                    raise NewHeadPartitionError(f"{context}: duplicate global identity")
                seen_identity_digests.add(identity_member_digest)
                identity_digest.update(identity_bytes + b"\n")
                primary_digest.update(_canonical_bytes(d0_primary) + b"\n")
                source_row_sha = d0["stage_b_data_driven_source_row_sha256"]
                source_row_digest.update(source_row_sha.encode("ascii") + b"\n")

                stats = images.setdefault(d0_key, MutableImageStats())
                if (
                    stats.filename_basename is not None
                    and stats.filename_basename != d0_basename
                ):
                    raise NewHeadPartitionError(
                        f"{context}: one image key has multiple filenames"
                    )
                stats.filename_basename = d0_basename
                stats.source_rows[source_by_manifest[name]] += 1
                stats.categories[class_id] += 1
                stats.expression_rows += 1
                rows += 1
                complete_instances += instance_count
        if rows != expected_rows[name]:
            raise NewHeadPartitionError(
                f"paired row count drifted for {name}: {rows} != {expected_rows[name]}"
            )
        for variant, digest in (("d0", d0_digest), ("d1", d1_digest)):
            observed = digest.hexdigest()
            expected = source_records[name][variant]["sha256"]
            if observed != expected:
                raise NewHeadPartitionError(
                    f"{variant.upper()} changed while scanning {name}"
                )
        per_manifest[name] = {
            "rows": rows,
            "complete_instances": complete_instances,
        }
        total_rows += rows
        total_complete_instances += complete_instances

    sealed = upstream_receipt["sealed_streams"]
    observed_streams = {
        "ordered_identity_stream_sha256": identity_digest.hexdigest(),
        "source_primary_stream_sha256": primary_digest.hexdigest(),
        "source_row_sha256_stream_sha256": source_row_digest.hexdigest(),
    }
    if observed_streams != sealed:
        raise NewHeadPartitionError(
            "paired input stream digests do not replay the sealed receipt"
        )
    if (
        total_rows != upstream_receipt["rows"]
        or len(seen_identity_digests) != upstream_receipt["unique_identities"]
        or total_complete_instances
        != upstream_receipt["category_complete_instances"]
    ):
        raise NewHeadPartitionError("paired input totals do not replay the sealed receipt")
    return images, {
        "rows": total_rows,
        "unique_identities": len(seen_identity_digests),
        "unique_image_keys": len(images),
        "category_complete_instances": total_complete_instances,
        "manifests": per_manifest,
        **observed_streams,
    }


def _image_key_value(key: ImageKey) -> str:
    return f"{key[0]}:{key[1]:012d}"


def _image_key_object(key: ImageKey) -> dict[str, Any]:
    return {"coco_split": key[0], "image_id": key[1]}


def _stratum_object(stratum: Stratum) -> dict[str, Any]:
    return {
        "source_membership_mask": stratum[0],
        "dominant_primary_category": stratum[1],
        "expression_log2_bucket": stratum[2],
    }


def _record_stream_sha256(records: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _priority(*, seed: int, phase: str, value: Any) -> str:
    payload = {
        "namespace": SELECTION_NAMESPACE,
        "seed": seed,
        "phase": phase,
        "value": value,
    }
    return _sha256_bytes(_canonical_bytes(payload))


def _largest_remainder(
    populations: Mapping[Stratum, int],
    *,
    target: int,
    seed: int,
    phase: str,
) -> dict[Stratum, int]:
    total = sum(populations.values())
    if type(target) is not int or target < 0 or target > total:
        raise NewHeadPartitionError(
            f"invalid {phase} target: target={target}, population={total}"
        )
    quotas = {
        stratum: population * target // total
        for stratum, population in populations.items()
    } if total else {}
    remaining = target - sum(quotas.values())
    order = sorted(
        populations,
        key=lambda stratum: (
            -(populations[stratum] * target % total),
            _priority(
                seed=seed,
                phase=f"{phase}_quota_tie",
                value=_stratum_object(stratum),
            ),
            stratum,
        ),
    ) if total else []
    for stratum in order[:remaining]:
        quotas[stratum] += 1
    if sum(quotas.values()) != target or any(
        quota < 0 or quota > populations[stratum]
        for stratum, quota in quotas.items()
    ):
        raise NewHeadPartitionError(f"{phase} largest-remainder allocation failed")
    return quotas


def _select_from_strata(
    members: Mapping[Stratum, Sequence[ImageKey]],
    quotas: Mapping[Stratum, int],
    *,
    seed: int,
    phase: str,
) -> frozenset[ImageKey]:
    selected: set[ImageKey] = set()
    for stratum in sorted(members):
        ordered = sorted(
            members[stratum],
            key=lambda key: (
                _priority(
                    seed=seed,
                    phase=f"{phase}_member",
                    value=_image_key_object(key),
                ),
                key,
            ),
        )
        selected.update(ordered[: quotas[stratum]])
    if len(selected) != sum(quotas.values()):
        raise NewHeadPartitionError(f"{phase} selected image count drifted")
    return frozenset(selected)


def _select_images(
    *,
    images: Mapping[ImageKey, MutableImageStats],
    official_keys: frozenset[ImageKey],
    seed: int,
    dev_images: int,
    screen_images: int,
) -> Selection:
    if type(seed) is not int:
        raise NewHeadPartitionError("selection seed must be an exact integer")
    if (
        type(dev_images) is not int
        or type(screen_images) is not int
        or dev_images <= 0
        or screen_images <= 0
        or screen_images > dev_images
    ):
        raise NewHeadPartitionError("invalid dev/screen image counts")
    quarantine = frozenset(set(images).intersection(official_keys))
    image_records: dict[ImageKey, ImageRecord] = {}
    candidate_members: dict[Stratum, list[ImageKey]] = defaultdict(list)
    for key, stats in images.items():
        if not stats.categories or stats.expression_rows <= 0:
            raise NewHeadPartitionError(f"image statistics are incomplete for {key}")
        mask = "".join(
            "1" if stats.source_rows[label] > 0 else "0"
            for label in SOURCE_LABELS
        )
        dominant = min(
            stats.categories,
            key=lambda class_id: (-stats.categories[class_id], class_id),
        )
        bucket = int(math.floor(math.log2(stats.expression_rows)))
        stratum = (mask, dominant, bucket)
        record = ImageRecord(
            key=key,
            source_membership_mask=mask,
            dominant_primary_category=dominant,
            expression_rows=stats.expression_rows,
            expression_log2_bucket=bucket,
            stratum=stratum,
        )
        image_records[key] = record
        if key not in official_keys:
            candidate_members[stratum].append(key)
    candidate_count = sum(len(values) for values in candidate_members.values())
    if candidate_count < dev_images:
        raise NewHeadPartitionError(
            f"not enough non-Ref8 images: {candidate_count} < {dev_images}"
        )
    populations = {stratum: len(values) for stratum, values in candidate_members.items()}
    dev_quotas = _largest_remainder(
        populations, target=dev_images, seed=seed, phase="dev_full"
    )
    dev_full = _select_from_strata(
        candidate_members, dev_quotas, seed=seed, phase="dev_full"
    )

    screen_members: dict[Stratum, list[ImageKey]] = defaultdict(list)
    for key in dev_full:
        screen_members[image_records[key].stratum].append(key)
    screen_populations = {
        stratum: len(values) for stratum, values in screen_members.items()
    }
    screen_quotas = _largest_remainder(
        screen_populations, target=screen_images, seed=seed, phase="dev_screen"
    )
    dev_screen = _select_from_strata(
        screen_members, screen_quotas, seed=seed, phase="dev_screen"
    )
    if not dev_screen.issubset(dev_full):
        raise NewHeadPartitionError("dev_screen is not nested in dev_full")
    if dev_full.intersection(official_keys):
        raise NewHeadPartitionError("dev_full intersects official Ref8")

    strata_records = []
    for stratum in sorted(populations):
        strata_records.append(
            {
                **_stratum_object(stratum),
                "stratum_id_sha256": _sha256_bytes(
                    _canonical_bytes(_stratum_object(stratum))
                ),
                "candidate_images": populations[stratum],
                "dev_full_images": dev_quotas[stratum],
                "dev_screen_images": screen_quotas.get(stratum, 0),
            }
        )
    return Selection(
        official_keys=official_keys,
        quarantine_keys=quarantine,
        dev_full_keys=dev_full,
        dev_screen_keys=dev_screen,
        image_records=image_records,
        strata=tuple(strata_records),
    )


def _main_partition(key: ImageKey, selection: Selection) -> str:
    if key in selection.official_keys:
        return "quarantine"
    if key in selection.dev_full_keys:
        return "dev_full"
    return "train"


def _output_path(output_root: Path, key: OutputKey) -> Path:
    variant, partition, manifest = key
    return output_root / variant / partition / manifest


def _stream_partition_outputs(
    *,
    d0_root: Path,
    d1_root: Path,
    output_root: Path,
    selection: Selection,
    source_records: Mapping[str, Mapping[str, Mapping[str, Any]]],
    write_root: Path | None,
) -> dict[str, dict[str, dict[str, Any]]]:
    digests = {
        (variant, partition, name): hashlib.sha256()
        for variant in VARIANTS
        for partition in PARTITIONS
        for name in MANIFESTS
    }
    sizes: Counter[OutputKey] = Counter()
    rows: Counter[OutputKey] = Counter()
    identity_stream_digests = {
        key: hashlib.sha256() for key in digests
    }
    identity_members: dict[OutputKey, set[bytes]] = {
        key: set() for key in digests
    }
    image_members: dict[OutputKey, set[ImageKey]] = {
        key: set() for key in digests
    }
    handles: dict[OutputKey, BinaryIO] = {}
    with ExitStack() as stack:
        if write_root is not None:
            for key in digests:
                path = _output_path(write_root, key)
                path.parent.mkdir(parents=True, exist_ok=True)
                handles[key] = stack.enter_context(path.open("xb"))
        for name in MANIFESTS:
            input_digests = {"d0": hashlib.sha256(), "d1": hashlib.sha256()}
            d0_path = d0_root / name
            d1_path = d1_root / name
            with d0_path.open("rb") as d0_handle, d1_path.open("rb") as d1_handle:
                for line_number, pair in enumerate(
                    zip_longest(d0_handle, d1_handle), start=1
                ):
                    d0_raw, d1_raw = pair
                    if d0_raw is None or d1_raw is None:
                        raise NewHeadPartitionError(
                            f"{name}: D0/D1 changed while replaying outputs"
                        )
                    input_digests["d0"].update(d0_raw)
                    input_digests["d1"].update(d1_raw)
                    d0 = _load_jsonl_row(
                        d0_raw, path=d0_path, line_number=line_number
                    )
                    d1 = _load_jsonl_row(
                        d1_raw, path=d1_path, line_number=line_number
                    )
                    d0_key, _ = _image_key(d0, context=f"{name}:{line_number} D0")
                    d1_key, _ = _image_key(d1, context=f"{name}:{line_number} D1")
                    if d0_key != d1_key or d0_key not in selection.image_records:
                        raise NewHeadPartitionError(
                            f"{name}:{line_number}: image key changed during replay"
                        )
                    d0_identity = _identity(d0, context=f"{name}:{line_number} D0")
                    d1_identity = _identity(d1, context=f"{name}:{line_number} D1")
                    if d0_identity != d1_identity:
                        raise NewHeadPartitionError(
                            f"{name}:{line_number}: paired identity changed during replay"
                        )
                    partitions = [_main_partition(d0_key, selection)]
                    if d0_key in selection.dev_screen_keys:
                        partitions.append("dev_screen")
                    for variant, raw, identity in zip(
                        VARIANTS,
                        (d0_raw, d1_raw),
                        (d0_identity, d1_identity),
                        strict=True,
                    ):
                        identity_bytes = _canonical_bytes(identity)
                        identity_digest = hashlib.sha256(identity_bytes).digest()
                        for partition in partitions:
                            output_key = (variant, partition, name)
                            digests[output_key].update(raw)
                            identity_stream_digests[output_key].update(
                                identity_bytes + b"\n"
                            )
                            identity_members[output_key].add(identity_digest)
                            image_members[output_key].add(d0_key)
                            sizes[output_key] += len(raw)
                            rows[output_key] += 1
                            if write_root is not None:
                                handles[output_key].write(raw)
            for variant in ("d0", "d1"):
                if (
                    input_digests[variant].hexdigest()
                    != source_records[name][variant]["sha256"]
                ):
                    raise NewHeadPartitionError(
                        f"{variant.upper()} changed while replaying {name}"
                    )
        if write_root is not None:
            for handle in handles.values():
                handle.flush()
                os.fsync(handle.fileno())

    records: dict[str, dict[str, dict[str, Any]]] = {
        variant: {partition: {} for partition in PARTITIONS}
        for variant in VARIANTS
    }
    for key, digest in digests.items():
        variant, partition, name = key
        records[variant][partition][name] = _predicted_file_record(
            _output_path(output_root, key),
            rows=rows[key],
            unique_identities=len(identity_members[key]),
            unique_image_keys=len(image_members[key]),
            ordered_identity_stream_sha256=(
                identity_stream_digests[key].hexdigest()
            ),
            size_bytes=sizes[key],
            sha256=digest.hexdigest(),
        )
        if records[variant][partition][name]["unique_identities"] != rows[key]:
            raise NewHeadPartitionError(
                "an output partition contains duplicate identities: "
                f"{variant}/{partition}/{name}"
            )
    return records


def _selection_member_records(
    keys: frozenset[ImageKey],
    *,
    selection: Selection,
    seed: int,
    phase: str,
) -> list[dict[str, Any]]:
    ordered = sorted(
        keys,
        key=lambda key: (
            _priority(
                seed=seed,
                phase=f"{phase}_member",
                value=_image_key_object(key),
            ),
            key,
        ),
    )
    return [
        {
            **_image_key_object(key),
            "image_key": _image_key_value(key),
            "selection_priority_sha256": _priority(
                seed=seed,
                phase=f"{phase}_member",
                value=_image_key_object(key),
            ),
            "source_membership_mask": selection.image_records[
                key
            ].source_membership_mask,
            "dominant_primary_category": selection.image_records[
                key
            ].dominant_primary_category,
            "expression_rows": selection.image_records[key].expression_rows,
            "expression_log2_bucket": selection.image_records[
                key
            ].expression_log2_bucket,
        }
        for key in ordered
    ]


def _partition_summary(
    output_records: Mapping[str, Mapping[str, Mapping[str, Any]]],
    selection: Selection,
) -> dict[str, Any]:
    image_sets = {
        "quarantine": selection.quarantine_keys,
        "dev_full": selection.dev_full_keys,
        "dev_screen": selection.dev_screen_keys,
        "train": frozenset(
            set(selection.image_records)
            - set(selection.quarantine_keys)
            - set(selection.dev_full_keys)
        ),
    }
    summary = {}
    for partition in PARTITIONS:
        records = output_records[VARIANTS[0]][partition]
        summary[partition] = {
            "unique_image_keys": len(image_sets[partition]),
            "rows": sum(record["rows"] for record in records.values()),
            "rows_by_manifest": {
                name: records[name]["rows"] for name in MANIFESTS
            },
            "ordered_image_key_stream_sha256": _record_stream_sha256(
                [_image_key_value(key) for key in sorted(image_sets[partition])]
            ),
        }
    return summary


def make_plan(
    *,
    input_receipt: Path = INPUT_RECEIPT,
    d0_root: Path = D0_ROOT,
    d1_root: Path = D1_ROOT,
    heldout_root: Path = HELDOUT_ROOT,
    output_root: Path = OUTPUT_ROOT,
    seed: int = SEED,
    dev_images: int = DEV_IMAGES,
    screen_images: int = SCREEN_IMAGES,
    expected_input_receipt_sha256: str = EXPECTED_INPUT_RECEIPT_SHA256,
    expected_category_receipt_sha256: str = EXPECTED_CATEGORY_RECEIPT_SHA256,
    expected_rows: Mapping[str, int] = EXPECTED_ROWS,
    expected_d0_sha256: Mapping[str, str] = EXPECTED_D0_SHA256,
    expected_d1_sha256: Mapping[str, str] = EXPECTED_D1_SHA256,
    heldout_contract: Mapping[str, Mapping[str, Any]] = REF_SPLIT_CONTRACT,
    heldout_manifest_files: Mapping[str, str] = REF_SPLIT_MANIFEST_FILES,
    heldout_splits: Sequence[str] = REF_SPLITS,
    expected_heldout_union_images: int = EXPECTED_HELDOUT_UNION_IMAGES,
) -> BuildPlan:
    _validate_expected_maps(
        expected_rows=expected_rows,
        expected_d0_sha256=expected_d0_sha256,
        expected_d1_sha256=expected_d1_sha256,
    )
    input_receipt = input_receipt.expanduser().resolve(strict=True)
    d0_root = d0_root.expanduser().resolve(strict=True)
    d1_root = d1_root.expanduser().resolve(strict=True)
    heldout_root = heldout_root.expanduser().resolve(strict=True)
    output_root = output_root.expanduser().resolve(strict=False)
    if not d0_root.is_dir() or not d1_root.is_dir() or not heldout_root.is_dir():
        raise NewHeadPartitionError("D0, D1, and heldout roots must be directories")

    upstream, source_records = _validate_upstream_receipt(
        input_receipt=input_receipt,
        d0_root=d0_root,
        d1_root=d1_root,
        expected_input_receipt_sha256=expected_input_receipt_sha256,
        expected_category_receipt_sha256=expected_category_receipt_sha256,
        expected_rows=expected_rows,
        expected_d0_sha256=expected_d0_sha256,
        expected_d1_sha256=expected_d1_sha256,
    )
    official_keys, official_receipt = _load_official_keys(
        heldout_root=heldout_root,
        heldout_contract=heldout_contract,
        heldout_manifest_files=heldout_manifest_files,
        heldout_splits=heldout_splits,
        expected_heldout_union_images=expected_heldout_union_images,
    )
    images, paired_scan = _scan_paired_inputs(
        d0_root=d0_root,
        d1_root=d1_root,
        expected_rows=expected_rows,
        source_records=source_records,
        upstream_receipt=upstream,
    )
    selection = _select_images(
        images=images,
        official_keys=official_keys,
        seed=seed,
        dev_images=dev_images,
        screen_images=screen_images,
    )
    output_records = _stream_partition_outputs(
        d0_root=d0_root,
        d1_root=d1_root,
        output_root=output_root,
        selection=selection,
        source_records=source_records,
        write_root=None,
    )
    partition_summary = _partition_summary(output_records, selection)
    dev_full_members = _selection_member_records(
        selection.dev_full_keys,
        selection=selection,
        seed=seed,
        phase="dev_full",
    )
    dev_screen_members = _selection_member_records(
        selection.dev_screen_keys,
        selection=selection,
        seed=seed,
        phase="dev_screen",
    )
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "builder": _file_record(Path(__file__)),
        "upstream_pair_receipt": upstream,
        "source_manifest_order": list(MANIFESTS),
        "source_membership_bit_order": list(SOURCE_LABELS),
        "source_manifests": {
            name: {
                "rows": expected_rows[name],
                "d0_ordinary_primary": source_records[name]["d0"],
                "d1_category_complete": source_records[name]["d1"],
            }
            for name in MANIFESTS
        },
        "paired_scan": paired_scan,
        "official_ref8": official_receipt,
        "selection_contract": {
            "policy": SELECTION_POLICY,
            "namespace": SELECTION_NAMESPACE,
            "seed": seed,
            "image_identity": ["COCO filename split", "image_id"],
            "quarantine_precedes_dev_selection": True,
            "strata": [
                "cross_source_membership_mask",
                "dominant_primary_class_id",
                "floor(log2(expression_rows))",
            ],
            "dominant_category_tie_break": "smallest_integer_class_id",
            "quota_method": "exact_integer_largest_remainder",
            "quota_remainder_tie_break": "seeded_sha256_then_stratum",
            "member_order": "seeded_sha256_then_image_key",
            "dev_full_target_images": dev_images,
            "dev_screen_target_images": screen_images,
            "dev_screen_is_nested_in_dev_full": True,
            "model_score_free": True,
            "forbidden_inputs": [
                "teacher_scores",
                "teacher_logits",
                "model_scores",
                "model_logits",
                "checkpoint_outputs",
            ],
        },
        "strata": list(selection.strata),
        "partition_summary": partition_summary,
        "dev_full_members": dev_full_members,
        "dev_screen_members": dev_screen_members,
        "quarantine": {
            "official_union_image_keys": len(selection.official_keys),
            "source_overlap_image_keys": len(selection.quarantine_keys),
            "ordered_source_overlap_image_key_stream_sha256": (
                _record_stream_sha256(
                    [
                        _image_key_value(key)
                        for key in sorted(selection.quarantine_keys)
                    ]
                )
            ),
        },
        "output_layout": "<variant>/<partition>/<source_manifest>",
        "output_stream_encoding": STREAM_ENCODING,
        "outputs": output_records,
        "invariants": {
            "pair_receipt_matches_preregistered_sha256": True,
            "category_complete_receipt_matches_preregistered_sha256": True,
            "all_six_source_manifests_match_preregistered_sha256": True,
            "D0_and_D1_rows_share_identity_filename_image_id_and_order": True,
            "D0_and_D1_primary_instances_match": True,
            "paired_streams_replay_the_upstream_receipt": True,
            "all_eight_official_ref_manifests_match_contract": True,
            "official_ref8_images_are_quarantined_before_selection": True,
            "train_dev_full_and_quarantine_images_are_pairwise_disjoint": True,
            "dev_full_has_exactly_requested_images": (
                len(selection.dev_full_keys) == dev_images
            ),
            "dev_screen_has_exactly_requested_images": (
                len(selection.dev_screen_keys) == screen_images
            ),
            "dev_screen_is_nested_in_dev_full": (
                selection.dev_screen_keys.issubset(selection.dev_full_keys)
            ),
            "dev_sets_are_disjoint_from_official_ref8": not bool(
                selection.dev_full_keys.intersection(selection.official_keys)
            ),
            "all_rows_for_one_global_image_share_the_same_main_partition": True,
            "D0_and_D1_partition_row_counts_match": all(
                output_records[VARIANTS[0]][partition][name]["rows"]
                == output_records[VARIANTS[1]][partition][name]["rows"]
                for partition in PARTITIONS
                for name in MANIFESTS
            ),
            "D0_and_D1_partition_identity_streams_and_counts_match": all(
                all(
                    output_records[VARIANTS[0]][partition][name][field]
                    == output_records[VARIANTS[1]][partition][name][field]
                    for field in (
                        "rows",
                        "unique_identities",
                        "unique_image_keys",
                        "ordered_identity_stream_sha256",
                    )
                )
                for partition in PARTITIONS
                for name in MANIFESTS
            ),
            "outputs_are_raw_byte_for_byte_input_subsequences": True,
            "selection_is_deterministic_model_score_free_and_hash_bound": True,
        },
    }
    if any(value is not True for value in receipt["invariants"].values()):
        raise NewHeadPartitionError("one or more output invariants failed")
    receipt["canonical_payload_sha256"] = _sha256_bytes(_canonical_bytes(receipt))
    return BuildPlan(
        receipt=receipt,
        selection=selection,
        source_records=source_records,
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


def build(**kwargs: Any) -> dict[str, Any]:
    output_root = Path(kwargs.get("output_root", OUTPUT_ROOT)).expanduser().resolve()
    if output_root.exists():
        raise NewHeadPartitionError(
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
        observed_outputs = _stream_partition_outputs(
            d0_root=Path(kwargs.get("d0_root", D0_ROOT)).expanduser().resolve(
                strict=True
            ),
            d1_root=Path(kwargs.get("d1_root", D1_ROOT)).expanduser().resolve(
                strict=True
            ),
            output_root=output_root,
            selection=plan.selection,
            source_records=plan.source_records,
            write_root=temporary_root,
        )
        if observed_outputs != plan.receipt["outputs"]:
            raise NewHeadPartitionError("written output streams differ from the plan")
        receipt_path = temporary_root / "receipt.json"
        with receipt_path.open("xb") as handle:
            handle.write(_receipt_bytes(plan.receipt))
            handle.flush()
            os.fsync(handle.fileno())
        for directory in sorted(
            (path for path in temporary_root.rglob("*") if path.is_dir()),
            key=lambda path: len(path.parts),
            reverse=True,
        ):
            _fsync_directory(directory)
        _fsync_directory(temporary_root)
        if output_root.exists():
            raise NewHeadPartitionError(
                f"refusing concurrent overwrite of output root: {output_root}"
            )
        os.rename(temporary_root, output_root)
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
        raise NewHeadPartitionError(f"output root is not a directory: {output_root}")
    plan = make_plan(**kwargs)
    expected_files = {Path("receipt.json")}
    for variant in VARIANTS:
        for partition in PARTITIONS:
            for name in MANIFESTS:
                expected_files.add(Path(variant) / partition / name)
    observed_files = {
        path.relative_to(output_root)
        for path in output_root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if observed_files != expected_files:
        raise NewHeadPartitionError(
            "output file set drifted: "
            f"missing={sorted(expected_files - observed_files)}, "
            f"extra={sorted(observed_files - expected_files)}"
        )
    receipt_path = output_root / "receipt.json"
    if receipt_path.is_symlink() or receipt_path.read_bytes() != _receipt_bytes(
        plan.receipt
    ):
        raise NewHeadPartitionError("partition receipt does not replay exactly")
    for variant in VARIANTS:
        for partition in PARTITIONS:
            for name in MANIFESTS:
                path = output_root / variant / partition / name
                if path.is_symlink():
                    raise NewHeadPartitionError(f"output must not be a symlink: {path}")
                observed = _file_record(path)
                expected = plan.receipt["outputs"][variant][partition][name]
                if (
                    observed["sha256"] != expected["sha256"]
                    or observed["size_bytes"] != expected["size_bytes"]
                ):
                    raise NewHeadPartitionError(
                        "partition output does not replay exactly: "
                        f"{variant}/{partition}/{name}"
                    )
                row_count = 0
                with path.open("rb") as handle:
                    for row_count, raw in enumerate(handle, start=1):
                        if not raw.endswith(b"\n"):
                            raise NewHeadPartitionError(
                                f"output row lost its line ending: {path}:{row_count}"
                            )
                if row_count != expected["rows"]:
                    raise NewHeadPartitionError(f"output row count drifted: {path}")
    return plan.receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-receipt", type=Path, default=INPUT_RECEIPT)
    parser.add_argument("--d0-root", type=Path, default=D0_ROOT)
    parser.add_argument("--d1-root", type=Path, default=D1_ROOT)
    parser.add_argument("--heldout-root", type=Path, default=HELDOUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="replay all contracts and print the planned receipt without writing",
    )
    mode.add_argument(
        "--verify",
        action="store_true",
        help="strictly replay and verify an existing output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kwargs = {
        "input_receipt": args.input_receipt,
        "d0_root": args.d0_root,
        "d1_root": args.d1_root,
        "heldout_root": args.heldout_root,
        "output_root": args.output_root,
    }
    if args.dry_run:
        receipt = make_plan(**kwargs).receipt
    elif args.verify:
        receipt = verify(**kwargs)
    else:
        receipt = build(**kwargs)
    print(json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
