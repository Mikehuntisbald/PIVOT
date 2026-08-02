#!/usr/bin/env python3
"""Build the clean role-routed official-assignment training manifests.

The builder intersects the sealed official-assignment manifests with the
sealed D1 new-head train partition by the exact seven-field expression
identity.  Selected assignment rows are copied byte for byte and in their
original order.  Assignment partners are never recomputed or replaced.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


ASSIGNMENT_ROOT = (
    REPO_ROOT / "data/ablations/stageb_data_driven_assignment_pairs_20260722"
)
ASSIGNMENT_RECEIPT = ASSIGNMENT_ROOT / "receipt.json"
PARTITION_ROOT = (
    REPO_ROOT / "data/ablations/stageb_data_driven_new_head_partition_20260723"
)
PARTITION_RECEIPT = PARTITION_ROOT / "receipt.json"
CLEAN_TRAIN_ROOT = PARTITION_ROOT / "d1_category_complete/train"
OUTPUT_ROOT = (
    REPO_ROOT
    / "data/ablations/stageb_data_driven_role_routed_clean_assignment_20260727"
)

RECEIPT_SCHEMA = (
    "pivot.stageb.data_driven.role_routed_clean_assignment_receipt/v1"
)
SCOPE = "official_assignment_clean_train_263661_v1"
ROW_SCHEMA = "pivot.stageb.data_driven.official_assignment_pair/v1"
UPSTREAM_ASSIGNMENT_RECEIPT_SCHEMA = (
    "pivot.stageb.data_driven.official_assignment_pair_receipt/v1"
)
UPSTREAM_PARTITION_RECEIPT_SCHEMA = (
    "pivot.stageb.data_driven.new_head_partition_receipt/v1"
)
D1_ROW_SCHEMA = "pivot.stageb.u2_category_complete_ref/v1"
SELECTION_POLICY = "exact_seven_field_identity_train_intersection_v1"
STREAM_ENCODING = "raw_upstream_assignment_record_including_line_ending_v1"

MANIFESTS = (
    "refcoco_stageb_phrase_v1.jsonl",
    "refcocoplus_stageb_phrase_v1.jsonl",
    "refcocog_stageb_phrase_v1.jsonl",
)
IDENTITY_KEYS = (
    "source",
    "image_id",
    "ann_id",
    "ref_id",
    "sent_id",
    "split",
    "filename",
)
ASSIGNMENT_FIELDS = (
    "stage_b_data_driven_assignment_pair",
    "stage_b_data_driven_assignment_pair_schema",
    "assignment_pair_valid",
    "assignment_pair",
    "assignment_pair_invalid_reason",
)

EXPECTED_ASSIGNMENT_RECEIPT_SHA256 = (
    "7b9ce1c911a2e1f0b67464243df8290fc2baf0786a2a3b131ddc57a6a6d2ddaa"
)
EXPECTED_PARTITION_RECEIPT_SHA256 = (
    "56de31d883ed137f3f9332c34de846839d82c0724120f42a49c5c1c302f38506"
)
EXPECTED_ASSIGNMENT = {
    "refcoco_stageb_phrase_v1.jsonl": {
        "rows": 120624,
        "valid_rows": 110956,
        "invalid_rows": 9668,
        "sha256": "f253c8bec4d15e421b11c42d8114e17c41bc32ed28f2614e34fe341e4da32592",
    },
    "refcocoplus_stageb_phrase_v1.jsonl": {
        "rows": 120191,
        "valid_rows": 110128,
        "invalid_rows": 10063,
        "sha256": "69039abbd5baeb1173c849c19c55128aea8053271ab79f0cc16fa679000deaa8",
    },
    "refcocog_stageb_phrase_v1.jsonl": {
        "rows": 80512,
        "valid_rows": 53498,
        "invalid_rows": 27014,
        "sha256": "378c5e34899e4113cd5dca1fd60352b362924e7969a717ea729b8278ce97a553",
    },
}
EXPECTED_CLEAN_TRAIN = {
    "refcoco_stageb_phrase_v1.jsonl": {
        "rows": 98934,
        "unique_image_keys": 13835,
        "sha256": "defa48cd85659c689734ba717e94baf0416c5391c77df80a7fc4cbc8f1202cc4",
        "ordered_identity_stream_sha256": (
            "b5ecebbcc27bb17018dcd75a5ab2ca4126e8a0c1cdf47b08d8ae8ffefb14264a"
        ),
    },
    "refcocoplus_stageb_phrase_v1.jsonl": {
        "rows": 98491,
        "unique_image_keys": 13833,
        "sha256": "7b5f4540ba565e692417a6f15ed2460f4a856e35521e284d4b5db6056b03332c",
        "ordered_identity_stream_sha256": (
            "fb8d7a6e81120ec0536d6fd1cc029b700c19cb530d62d32a8195e4f135d72096"
        ),
    },
    "refcocog_stageb_phrase_v1.jsonl": {
        "rows": 66236,
        "unique_image_keys": 18354,
        "sha256": "b45ee9494b05a57aa13a0fce075f3fdbda7ab580cce46fa90bce31fee6a10dfa",
        "ordered_identity_stream_sha256": (
            "8e5f15e0355ed61b07140f5592e9d66140801e9520bfa0079fa7d28b05af787b"
        ),
    },
}
EXPECTED_OUTPUT = {
    "refcoco_stageb_phrase_v1.jsonl": {
        "rows": 98934,
        "valid_rows": 91259,
        "invalid_rows": 7675,
        "unique_image_keys": 13835,
        "size_bytes": 257429921,
        "sha256": "9cf00f8c1cead0b5741e9f3bf74b29a3a58000982c0c3bcf18f5762512de20cc",
        "ordered_identity_stream_sha256": (
            "b5ecebbcc27bb17018dcd75a5ab2ca4126e8a0c1cdf47b08d8ae8ffefb14264a"
        ),
        "base_row_stream_sha256": (
            "da77363dc685fefe9323d5cb4b969ab31b91ba2496c740ea2d8f32e58c4c9beb"
        ),
    },
    "refcocoplus_stageb_phrase_v1.jsonl": {
        "rows": 98491,
        "valid_rows": 90508,
        "invalid_rows": 7983,
        "unique_image_keys": 13833,
        "size_bytes": 256426719,
        "sha256": "c4d6aec09049381d3d49688e9bd5337767515bc732d957f9727f4892ca8847d5",
        "ordered_identity_stream_sha256": (
            "fb8d7a6e81120ec0536d6fd1cc029b700c19cb530d62d32a8195e4f135d72096"
        ),
        "base_row_stream_sha256": (
            "7bcb8097e24f2f1a8469abbd13f82060e35766cf78c818d0332489409983ff7c"
        ),
    },
    "refcocog_stageb_phrase_v1.jsonl": {
        "rows": 66236,
        "valid_rows": 42956,
        "invalid_rows": 23280,
        "unique_image_keys": 18354,
        "size_bytes": 155514587,
        "sha256": "b530c4d838a85496b8713a14014e80fc71db342237fde93649b0d25adb43033a",
        "ordered_identity_stream_sha256": (
            "8e5f15e0355ed61b07140f5592e9d66140801e9520bfa0079fa7d28b05af787b"
        ),
        "base_row_stream_sha256": (
            "36d3be2f0b7c25ca10b825c3853ff9d5634dd461ef043eac54c2e66c46eeb696"
        ),
    },
}
EXPECTED_GLOBAL_UNIQUE_IMAGE_KEYS = 22359

_HEX64_RE = re.compile(r"[0-9a-f]{64}")
_COCO_FILENAME_RE = re.compile(
    r"^COCO_(?P<split>train2014|val2014)_(?P<image_id>[0-9]{12})"
    r"\.(?:jpg|jpeg)$",
    re.IGNORECASE,
)

Identity = tuple[Any, ...]
MemberId = tuple[str, int, int, int, int]
ImageKey = tuple[str, int]


class CleanAssignmentBuildError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CleanMembership:
    identities: frozenset[Identity]
    member_ids: frozenset[MemberId]
    image_keys: frozenset[ImageKey]


@dataclass(frozen=True, slots=True)
class BuildPlan:
    receipt: dict[str, Any]
    memberships: Mapping[str, CleanMembership]
    assignment_records: Mapping[str, Mapping[str, Any]]
    clean_records: Mapping[str, Mapping[str, Any]]


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


def _base_row_sha256_line(row: Mapping[str, Any]) -> bytes:
    """Encode one base row using the upstream receipt stream contract."""
    return _sha256_bytes(_canonical_bytes(row)).encode("ascii") + b"\n"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, *, reported_path: Path | None = None) -> dict[str, Any]:
    requested_path = path.expanduser()
    if requested_path.is_symlink():
        raise CleanAssignmentBuildError(f"symlinks are forbidden: {requested_path}")
    path = requested_path.resolve(strict=True)
    if not path.is_file():
        raise CleanAssignmentBuildError(f"not a regular file: {path}")
    before = path.stat()
    digest = _sha256_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (
        after.st_size,
        after.st_mtime_ns,
    ):
        raise CleanAssignmentBuildError(f"file changed while hashing: {path}")
    return {
        "path": str((reported_path or path).expanduser().resolve()),
        "size_bytes": int(before.st_size),
        "sha256": digest,
    }


def _predicted_file_record(
    path: Path, *, size_bytes: int, sha256: str
) -> dict[str, Any]:
    return {
        "path": str(path.expanduser().resolve()),
        "size_bytes": int(size_bytes),
        "sha256": sha256,
    }


def _validate_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _HEX64_RE.fullmatch(value) is None:
        raise CleanAssignmentBuildError(f"{label} is not a lowercase SHA-256")
    return value


def _required_int(value: Any, *, field: str, context: str) -> int:
    if type(value) is not int:
        raise CleanAssignmentBuildError(
            f"{context}: {field} must be an exact integer"
        )
    return int(value)


def _required_text(value: Any, *, field: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CleanAssignmentBuildError(f"{context}: {field} must be non-empty text")
    return value


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CleanAssignmentBuildError(
            f"could not load {label}: {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise CleanAssignmentBuildError(f"{label} must be a JSON object: {path}")
    return value


def _load_jsonl_row(raw: bytes, *, path: Path, line_number: int) -> dict[str, Any]:
    if not raw.endswith(b"\n"):
        raise CleanAssignmentBuildError(
            f"input row lacks a terminating LF at {path}:{line_number}"
        )
    if not raw.rstrip(b"\r\n"):
        raise CleanAssignmentBuildError(f"blank JSONL row at {path}:{line_number}")
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise CleanAssignmentBuildError(
            f"invalid JSON at {path}:{line_number}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise CleanAssignmentBuildError(
            f"row is not an object at {path}:{line_number}"
        )
    return value


def _identity(row: Mapping[str, Any], *, context: str) -> Identity:
    for field in ("image_id", "ann_id", "ref_id", "sent_id"):
        _required_int(row.get(field), field=field, context=context)
    for field in ("source", "split", "filename"):
        _required_text(row.get(field), field=field, context=context)
    return tuple(row[key] for key in IDENTITY_KEYS)


def _member_id(row: Mapping[str, Any], *, context: str) -> MemberId:
    return (
        _required_text(row.get("source"), field="source", context=context),
        _required_int(row.get("image_id"), field="image_id", context=context),
        _required_int(row.get("ann_id"), field="ann_id", context=context),
        _required_int(row.get("ref_id"), field="ref_id", context=context),
        _required_int(row.get("sent_id"), field="sent_id", context=context),
    )


def _image_key(row: Mapping[str, Any], *, context: str) -> ImageKey:
    image_id = _required_int(row.get("image_id"), field="image_id", context=context)
    filename = _required_text(row.get("filename"), field="filename", context=context)
    match = _COCO_FILENAME_RE.fullmatch(Path(filename).name)
    if match is None:
        raise CleanAssignmentBuildError(
            f"{context}: filename is not a canonical COCO 2014 image: {filename}"
        )
    filename_image_id = int(match.group("image_id"))
    if filename_image_id != image_id:
        raise CleanAssignmentBuildError(
            f"{context}: filename/image_id drifted: {filename_image_id} != {image_id}"
        )
    return match.group("split").lower(), image_id


def _validate_canonical_payload_hash(
    receipt: Mapping[str, Any], *, label: str
) -> None:
    claimed = _validate_sha256(
        receipt.get("canonical_payload_sha256"),
        label=f"{label} canonical payload hash",
    )
    payload = dict(receipt)
    payload.pop("canonical_payload_sha256", None)
    observed = _sha256_bytes(_canonical_bytes(payload))
    if observed != claimed:
        raise CleanAssignmentBuildError(
            f"{label} canonical payload hash drifted: {observed} != {claimed}"
        )


def _validate_expected_maps(
    *,
    manifest_names: Sequence[str],
    expected_assignment: Mapping[str, Mapping[str, Any]],
    expected_clean_train: Mapping[str, Mapping[str, Any]],
    expected_output: Mapping[str, Mapping[str, Any]],
    expected_global_unique_image_keys: int,
) -> None:
    names = tuple(manifest_names)
    required = set(names)
    if len(required) != len(names) or any(Path(name).name != name for name in names):
        raise CleanAssignmentBuildError("manifest names must be unique basenames")
    if (
        set(expected_assignment) != required
        or set(expected_clean_train) != required
        or set(expected_output) != required
    ):
        raise CleanAssignmentBuildError(
            "expected maps must exactly match the manifest names"
        )
    if (
        type(expected_global_unique_image_keys) is not int
        or expected_global_unique_image_keys <= 0
    ):
        raise CleanAssignmentBuildError("invalid expected global image-key count")

    for name in names:
        assignment = expected_assignment[name]
        clean = expected_clean_train[name]
        output = expected_output[name]
        for field in ("rows", "valid_rows", "invalid_rows"):
            if type(assignment.get(field)) is not int or assignment[field] < 0:
                raise CleanAssignmentBuildError(
                    f"invalid assignment {field} for {name}"
                )
        if assignment["valid_rows"] + assignment["invalid_rows"] != assignment["rows"]:
            raise CleanAssignmentBuildError(f"assignment totals disagree for {name}")
        _validate_sha256(
            assignment.get("sha256"), label=f"{name} assignment expected hash"
        )

        for field in ("rows", "unique_image_keys"):
            if type(clean.get(field)) is not int or clean[field] <= 0:
                raise CleanAssignmentBuildError(f"invalid clean {field} for {name}")
        for field in ("sha256", "ordered_identity_stream_sha256"):
            _validate_sha256(clean.get(field), label=f"{name} clean {field}")

        for field in (
            "rows",
            "valid_rows",
            "invalid_rows",
            "unique_image_keys",
            "size_bytes",
        ):
            if type(output.get(field)) is not int or output[field] < 0:
                raise CleanAssignmentBuildError(f"invalid output {field} for {name}")
        for field in (
            "sha256",
            "ordered_identity_stream_sha256",
            "base_row_stream_sha256",
        ):
            _validate_sha256(output.get(field), label=f"{name} output {field}")
        if output["valid_rows"] + output["invalid_rows"] != output["rows"]:
            raise CleanAssignmentBuildError(f"output totals disagree for {name}")
        if (
            output["rows"] != clean["rows"]
            or output["unique_image_keys"] != clean["unique_image_keys"]
            or output["ordered_identity_stream_sha256"]
            != clean["ordered_identity_stream_sha256"]
            or assignment["rows"] < clean["rows"]
        ):
            raise CleanAssignmentBuildError(
                f"expected assignment/clean/output contracts disagree for {name}"
            )


def _load_assignment_receipt(
    *,
    input_root: Path,
    input_receipt: Path,
    manifest_names: Sequence[str],
    expected_receipt_sha256: str,
    expected_assignment: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    _validate_sha256(expected_receipt_sha256, label="assignment receipt expected hash")
    receipt_record = _file_record(input_receipt)
    if receipt_record["sha256"] != expected_receipt_sha256:
        raise CleanAssignmentBuildError(
            "assignment receipt SHA-256 mismatch: "
            f"{receipt_record['sha256']} != {expected_receipt_sha256}"
        )
    receipt = _load_json_object(input_receipt, label="assignment receipt")
    _validate_canonical_payload_hash(receipt, label="assignment receipt")
    expected_rows = sum(values["rows"] for values in expected_assignment.values())
    expected_valid = sum(
        values["valid_rows"] for values in expected_assignment.values()
    )
    expected_invalid = sum(
        values["invalid_rows"] for values in expected_assignment.values()
    )
    invariants = receipt.get("invariants")
    if (
        receipt.get("schema") != UPSTREAM_ASSIGNMENT_RECEIPT_SCHEMA
        or receipt.get("row_schema") != ROW_SCHEMA
        or receipt.get("manifest_order") != list(manifest_names)
        or receipt.get("rows") != expected_rows
        or receipt.get("valid_rows") != expected_valid
        or receipt.get("invalid_rows") != expected_invalid
        or receipt.get("unique_identities") != expected_rows
        or not isinstance(invariants, dict)
        or not invariants
        or any(value is not True for value in invariants.values())
    ):
        raise CleanAssignmentBuildError("assignment receipt contract drifted")

    manifests = receipt.get("manifests")
    if not isinstance(manifests, dict) or set(manifests) != set(manifest_names):
        raise CleanAssignmentBuildError("assignment receipt manifest set drifted")
    records: dict[str, dict[str, Any]] = {}
    for name in manifest_names:
        expected = expected_assignment[name]
        entry = manifests.get(name)
        output = entry.get("output") if isinstance(entry, dict) else None
        record = _file_record(input_root / name)
        if (
            record["sha256"] != expected["sha256"]
            or not isinstance(output, dict)
            or output.get("sha256") != record["sha256"]
            or output.get("size_bytes") != record["size_bytes"]
            or not isinstance(output.get("path"), str)
            or Path(output["path"]).expanduser().resolve() != Path(record["path"])
            or entry.get("rows") != expected["rows"]
            or entry.get("valid_rows") != expected["valid_rows"]
            or entry.get("invalid_rows") != expected["invalid_rows"]
        ):
            raise CleanAssignmentBuildError(
                f"assignment receipt binding drifted for {name}"
            )
        records[name] = record
    return {
        "record": receipt_record,
        "schema": receipt["schema"],
        "row_schema": receipt["row_schema"],
        "rows": receipt["rows"],
        "valid_rows": receipt["valid_rows"],
        "invalid_rows": receipt["invalid_rows"],
        "unique_identities": receipt["unique_identities"],
        "canonical_payload_sha256": receipt["canonical_payload_sha256"],
    }, records


def _load_partition_receipt(
    *,
    clean_train_root: Path,
    partition_receipt: Path,
    manifest_names: Sequence[str],
    expected_receipt_sha256: str,
    expected_clean_train: Mapping[str, Mapping[str, Any]],
    expected_global_unique_image_keys: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    _validate_sha256(expected_receipt_sha256, label="partition receipt expected hash")
    receipt_record = _file_record(partition_receipt)
    if receipt_record["sha256"] != expected_receipt_sha256:
        raise CleanAssignmentBuildError(
            "partition receipt SHA-256 mismatch: "
            f"{receipt_record['sha256']} != {expected_receipt_sha256}"
        )
    receipt = _load_json_object(partition_receipt, label="partition receipt")
    _validate_canonical_payload_hash(receipt, label="partition receipt")
    invariants = receipt.get("invariants")
    if (
        receipt.get("schema") != UPSTREAM_PARTITION_RECEIPT_SCHEMA
        or receipt.get("source_manifest_order") != list(manifest_names)
        or receipt.get("output_stream_encoding")
        != "raw_input_record_including_original_line_ending_v1"
        or not isinstance(invariants, dict)
        or not invariants
        or any(value is not True for value in invariants.values())
    ):
        raise CleanAssignmentBuildError("partition receipt contract drifted")

    variants = receipt.get("outputs")
    d1 = variants.get("d1_category_complete") if isinstance(variants, dict) else None
    train = d1.get("train") if isinstance(d1, dict) else None
    if not isinstance(train, dict) or set(train) != set(manifest_names):
        raise CleanAssignmentBuildError("partition D1 train manifest set drifted")
    summary = receipt.get("partition_summary")
    train_summary = summary.get("train") if isinstance(summary, dict) else None
    expected_rows = sum(values["rows"] for values in expected_clean_train.values())
    if (
        not isinstance(train_summary, dict)
        or train_summary.get("rows") != expected_rows
        or train_summary.get("unique_image_keys")
        != expected_global_unique_image_keys
        or train_summary.get("rows_by_manifest")
        != {name: expected_clean_train[name]["rows"] for name in manifest_names}
    ):
        raise CleanAssignmentBuildError("partition train summary drifted")

    records: dict[str, dict[str, Any]] = {}
    for name in manifest_names:
        expected = expected_clean_train[name]
        sealed = train[name]
        record = _file_record(clean_train_root / name)
        if (
            record["sha256"] != expected["sha256"]
            or sealed.get("sha256") != record["sha256"]
            or sealed.get("size_bytes") != record["size_bytes"]
            or not isinstance(sealed.get("path"), str)
            or Path(sealed["path"]).expanduser().resolve() != Path(record["path"])
            or sealed.get("rows") != expected["rows"]
            or sealed.get("unique_identities") != expected["rows"]
            or sealed.get("unique_image_keys") != expected["unique_image_keys"]
            or sealed.get("ordered_identity_stream_sha256")
            != expected["ordered_identity_stream_sha256"]
        ):
            raise CleanAssignmentBuildError(
                f"partition train binding drifted for {name}"
            )
        records[name] = record
    return {
        "record": receipt_record,
        "schema": receipt["schema"],
        "canonical_payload_sha256": receipt["canonical_payload_sha256"],
        "train_rows": expected_rows,
        "train_unique_image_keys": expected_global_unique_image_keys,
        "train_ordered_image_key_stream_sha256": train_summary.get(
            "ordered_image_key_stream_sha256"
        ),
    }, records


def _scan_clean_train(
    *,
    clean_train_root: Path,
    manifest_names: Sequence[str],
    clean_records: Mapping[str, Mapping[str, Any]],
    expected_clean_train: Mapping[str, Mapping[str, Any]],
    expected_global_unique_image_keys: int,
) -> tuple[dict[str, CleanMembership], dict[str, Any]]:
    memberships: dict[str, CleanMembership] = {}
    scan: dict[str, Any] = {}
    global_identities: set[Identity] = set()
    global_images: set[ImageKey] = set()
    global_identity_digest = hashlib.sha256()
    total_rows = 0

    for name in manifest_names:
        path = clean_train_root / name
        input_digest = hashlib.sha256()
        identity_digest = hashlib.sha256()
        identities: set[Identity] = set()
        member_ids: set[MemberId] = set()
        images: set[ImageKey] = set()
        rows = 0
        with path.open("rb") as handle:
            for line_number, raw in enumerate(handle, start=1):
                input_digest.update(raw)
                row = _load_jsonl_row(raw, path=path, line_number=line_number)
                context = f"clean train {name}:{line_number}"
                if (
                    row.get("stage_b_u2_category_complete") is not True
                    or row.get("stage_b_u2_category_complete_schema") != D1_ROW_SCHEMA
                    or any(field in row for field in ASSIGNMENT_FIELDS)
                ):
                    raise CleanAssignmentBuildError(
                        f"{context}: clean D1 row contract drifted"
                    )
                identity = _identity(row, context=context)
                if identity in identities or identity in global_identities:
                    raise CleanAssignmentBuildError(
                        f"{context}: duplicate clean-train identity"
                    )
                identities.add(identity)
                global_identities.add(identity)
                member = _member_id(row, context=context)
                if member in member_ids:
                    raise CleanAssignmentBuildError(
                        f"{context}: duplicate clean-train expression member"
                    )
                member_ids.add(member)
                image = _image_key(row, context=context)
                images.add(image)
                global_images.add(image)
                identity_bytes = _canonical_bytes(identity) + b"\n"
                identity_digest.update(identity_bytes)
                global_identity_digest.update(identity_bytes)
                rows += 1

        expected = expected_clean_train[name]
        observed = {
            "rows": rows,
            "unique_identities": len(identities),
            "unique_member_ids": len(member_ids),
            "unique_image_keys": len(images),
            "ordered_identity_stream_sha256": identity_digest.hexdigest(),
        }
        if (
            input_digest.hexdigest() != clean_records[name]["sha256"]
            or input_digest.hexdigest() != expected["sha256"]
            or observed["rows"] != expected["rows"]
            or observed["unique_identities"] != expected["rows"]
            or observed["unique_member_ids"] != expected["rows"]
            or observed["unique_image_keys"] != expected["unique_image_keys"]
            or observed["ordered_identity_stream_sha256"]
            != expected["ordered_identity_stream_sha256"]
        ):
            raise CleanAssignmentBuildError(f"clean-train stream drifted for {name}")
        memberships[name] = CleanMembership(
            identities=frozenset(identities),
            member_ids=frozenset(member_ids),
            image_keys=frozenset(images),
        )
        scan[name] = observed
        total_rows += rows

    if len(global_images) != expected_global_unique_image_keys:
        raise CleanAssignmentBuildError(
            "global clean-train image-key count drifted: "
            f"{len(global_images)} != {expected_global_unique_image_keys}"
        )
    return memberships, {
        "rows": total_rows,
        "unique_identities": len(global_identities),
        "unique_image_keys": len(global_images),
        "ordered_identity_stream_sha256": global_identity_digest.hexdigest(),
        "manifests": scan,
    }


def _assignment_base_and_validity(
    row: Mapping[str, Any],
    *,
    membership: CleanMembership,
    context: str,
) -> tuple[dict[str, Any], bool]:
    if any(field not in row for field in ASSIGNMENT_FIELDS):
        raise CleanAssignmentBuildError(
            f"{context}: assignment row is missing one of the five assignment fields"
        )
    if (
        row.get("stage_b_data_driven_assignment_pair") is not True
        or row.get("stage_b_data_driven_assignment_pair_schema") != ROW_SCHEMA
        or type(row.get("assignment_pair_valid")) is not bool
    ):
        raise CleanAssignmentBuildError(f"{context}: assignment row contract drifted")
    valid = bool(row["assignment_pair_valid"])
    pair = row.get("assignment_pair")
    if not isinstance(pair, dict) or pair.get("schema") != ROW_SCHEMA:
        raise CleanAssignmentBuildError(f"{context}: assignment pair contract drifted")
    anchor = pair.get("anchor")
    if not isinstance(anchor, dict):
        raise CleanAssignmentBuildError(f"{context}: assignment anchor is invalid")
    anchor_member = _member_id(anchor, context=f"{context} anchor")
    row_member = _member_id(row, context=context)
    if anchor_member != row_member:
        raise CleanAssignmentBuildError(
            f"{context}: assignment anchor no longer matches the source row"
        )

    partner = pair.get("partner")
    reason = row.get("assignment_pair_invalid_reason")
    if valid:
        if reason is not None or not isinstance(partner, dict):
            raise CleanAssignmentBuildError(
                f"{context}: valid assignment must have one partner and no reason"
            )
        partner_member = _member_id(partner, context=f"{context} partner")
        if partner_member not in membership.member_ids:
            raise CleanAssignmentBuildError(
                f"{context}: valid partner is not a clean-train row"
            )
        if (
            partner_member[0] != row_member[0]
            or partner_member[1] != row_member[1]
            or partner_member[2] == row_member[2]
        ):
            raise CleanAssignmentBuildError(
                f"{context}: valid partner is not a distinct same-image row"
            )
    else:
        if partner is not None or not isinstance(reason, str) or not reason.strip():
            raise CleanAssignmentBuildError(
                f"{context}: invalid assignment must retain null partner and reason"
            )

    base = dict(row)
    for field in ASSIGNMENT_FIELDS:
        base.pop(field)
    return base, valid


def _next_clean(
    iterator: Any,
    *,
    path: Path,
    input_digest: Any,
) -> tuple[int, bytes, dict[str, Any], Identity] | None:
    try:
        line_number, raw = next(iterator)
    except StopIteration:
        return None
    input_digest.update(raw)
    row = _load_jsonl_row(raw, path=path, line_number=line_number)
    return line_number, raw, row, _identity(
        row, context=f"clean train {path.name}:{line_number}"
    )


def _stream_outputs(
    *,
    assignment_root: Path,
    clean_train_root: Path,
    output_root: Path,
    manifest_names: Sequence[str],
    memberships: Mapping[str, CleanMembership],
    assignment_records: Mapping[str, Mapping[str, Any]],
    clean_records: Mapping[str, Mapping[str, Any]],
    expected_assignment: Mapping[str, Mapping[str, Any]],
    expected_clean_train: Mapping[str, Mapping[str, Any]],
    expected_output: Mapping[str, Mapping[str, Any]],
    expected_global_unique_image_keys: int,
    write_root: Path | None,
) -> dict[str, Any]:
    manifest_records: dict[str, Any] = {}
    global_identity_digest = hashlib.sha256()
    global_base_digest = hashlib.sha256()
    global_output_digest = hashlib.sha256()
    global_images: set[ImageKey] = set()
    totals = {"rows": 0, "valid_rows": 0, "invalid_rows": 0}

    for name in manifest_names:
        assignment_path = assignment_root / name
        clean_path = clean_train_root / name
        output_digest = hashlib.sha256()
        assignment_digest = hashlib.sha256()
        clean_digest = hashlib.sha256()
        identity_digest = hashlib.sha256()
        base_digest = hashlib.sha256()
        output_size = 0
        assignment_rows = 0
        output_rows = 0
        valid_rows = 0
        invalid_rows = 0
        output_images: set[ImageKey] = set()
        output_handle: BinaryIO | None = None
        if write_root is not None:
            output_handle = (write_root / name).open("xb")
        try:
            with assignment_path.open("rb") as assignment_handle, clean_path.open(
                "rb"
            ) as clean_handle:
                clean_iterator = enumerate(clean_handle, start=1)
                current_clean = _next_clean(
                    clean_iterator, path=clean_path, input_digest=clean_digest
                )
                for line_number, raw in enumerate(assignment_handle, start=1):
                    assignment_digest.update(raw)
                    assignment_rows += 1
                    row = _load_jsonl_row(
                        raw, path=assignment_path, line_number=line_number
                    )
                    identity = _identity(
                        row, context=f"assignment {name}:{line_number}"
                    )
                    if current_clean is None or identity != current_clean[3]:
                        continue

                    clean_line_number, _clean_raw, clean_row, clean_identity = (
                        current_clean
                    )
                    context = f"selected assignment {name}:{line_number}"
                    base, valid = _assignment_base_and_validity(
                        row, membership=memberships[name], context=context
                    )
                    if base != clean_row:
                        raise CleanAssignmentBuildError(
                            f"{context}: removing the five assignment fields does not "
                            f"replay clean train line {clean_line_number}"
                        )
                    if identity != clean_identity:
                        raise CleanAssignmentBuildError(
                            f"{context}: selected identity changed during replay"
                        )
                    image = _image_key(clean_row, context=context)
                    if image not in memberships[name].image_keys:
                        raise CleanAssignmentBuildError(
                            f"{context}: selected image is outside clean train"
                        )

                    identity_bytes = _canonical_bytes(identity) + b"\n"
                    # Keep the upstream assignment receipt's base-row stream
                    # semantics: one lowercase canonical row digest per line.
                    # This is intentionally different from hashing the raw or
                    # canonicalized base-row byte stream directly.
                    base_sha256_line = _base_row_sha256_line(base)
                    output_digest.update(raw)
                    global_output_digest.update(raw)
                    identity_digest.update(identity_bytes)
                    global_identity_digest.update(identity_bytes)
                    base_digest.update(base_sha256_line)
                    global_base_digest.update(base_sha256_line)
                    output_size += len(raw)
                    output_rows += 1
                    valid_rows += int(valid)
                    invalid_rows += int(not valid)
                    output_images.add(image)
                    global_images.add(image)
                    if output_handle is not None:
                        output_handle.write(raw)
                    current_clean = _next_clean(
                        clean_iterator, path=clean_path, input_digest=clean_digest
                    )

                if current_clean is not None:
                    raise CleanAssignmentBuildError(
                        f"{name}: clean train identity is missing from assignment input: "
                        f"line {current_clean[0]}"
                    )
            if output_handle is not None:
                output_handle.flush()
                os.fsync(output_handle.fileno())
        finally:
            if output_handle is not None:
                output_handle.close()

        observed = {
            "rows": output_rows,
            "valid_rows": valid_rows,
            "invalid_rows": invalid_rows,
            "unique_identities": output_rows,
            "unique_image_keys": len(output_images),
            "ordered_identity_stream_sha256": identity_digest.hexdigest(),
            "base_row_stream_sha256": base_digest.hexdigest(),
            "size_bytes": output_size,
            "sha256": output_digest.hexdigest(),
        }
        expected = expected_output[name]
        if (
            assignment_digest.hexdigest() != assignment_records[name]["sha256"]
            or assignment_digest.hexdigest() != expected_assignment[name]["sha256"]
            or assignment_rows != expected_assignment[name]["rows"]
        ):
            raise CleanAssignmentBuildError(
                f"assignment input changed while filtering {name}"
            )
        if (
            clean_digest.hexdigest() != clean_records[name]["sha256"]
            or clean_digest.hexdigest() != expected_clean_train[name]["sha256"]
            or output_rows != expected_clean_train[name]["rows"]
        ):
            raise CleanAssignmentBuildError(
                f"clean train changed while filtering {name}"
            )
        expected_observed = {
            key: expected[key]
            for key in (
                "rows",
                "valid_rows",
                "invalid_rows",
                "unique_image_keys",
                "ordered_identity_stream_sha256",
                "base_row_stream_sha256",
                "size_bytes",
                "sha256",
            )
        }
        if {key: observed[key] for key in expected_observed} != expected_observed:
            raise CleanAssignmentBuildError(
                f"filtered output does not match preregistered statistics for {name}: "
                f"observed={observed}, expected={expected_observed}"
            )

        manifest_records[name] = {
            "rows": output_rows,
            "valid_rows": valid_rows,
            "invalid_rows": invalid_rows,
            "unique_identities": output_rows,
            "unique_image_keys": len(output_images),
            "ordered_identity_stream_sha256": identity_digest.hexdigest(),
            "base_row_stream_sha256": base_digest.hexdigest(),
            "valid_partner_rows_verified_in_clean_train": valid_rows,
            "assignment_input": dict(assignment_records[name]),
            "clean_train_input": dict(clean_records[name]),
            "output": _predicted_file_record(
                output_root / name,
                size_bytes=output_size,
                sha256=output_digest.hexdigest(),
            ),
        }
        totals["rows"] += output_rows
        totals["valid_rows"] += valid_rows
        totals["invalid_rows"] += invalid_rows

    if len(global_images) != expected_global_unique_image_keys:
        raise CleanAssignmentBuildError(
            "filtered global image-key count drifted: "
            f"{len(global_images)} != {expected_global_unique_image_keys}"
        )
    return {
        **totals,
        "unique_identities": totals["rows"],
        "unique_image_keys": len(global_images),
        "ordered_identity_stream_sha256": global_identity_digest.hexdigest(),
        "base_row_stream_sha256": global_base_digest.hexdigest(),
        "raw_output_stream_sha256": global_output_digest.hexdigest(),
        "manifests": manifest_records,
    }


def make_plan(
    *,
    assignment_root: Path = ASSIGNMENT_ROOT,
    assignment_receipt: Path = ASSIGNMENT_RECEIPT,
    partition_receipt: Path = PARTITION_RECEIPT,
    clean_train_root: Path = CLEAN_TRAIN_ROOT,
    output_root: Path = OUTPUT_ROOT,
    manifest_names: Sequence[str] = MANIFESTS,
    expected_assignment_receipt_sha256: str = (
        EXPECTED_ASSIGNMENT_RECEIPT_SHA256
    ),
    expected_partition_receipt_sha256: str = EXPECTED_PARTITION_RECEIPT_SHA256,
    expected_assignment: Mapping[str, Mapping[str, Any]] = EXPECTED_ASSIGNMENT,
    expected_clean_train: Mapping[str, Mapping[str, Any]] = EXPECTED_CLEAN_TRAIN,
    expected_output: Mapping[str, Mapping[str, Any]] = EXPECTED_OUTPUT,
    expected_global_unique_image_keys: int = EXPECTED_GLOBAL_UNIQUE_IMAGE_KEYS,
    scope: str = SCOPE,
) -> BuildPlan:
    _validate_expected_maps(
        manifest_names=manifest_names,
        expected_assignment=expected_assignment,
        expected_clean_train=expected_clean_train,
        expected_output=expected_output,
        expected_global_unique_image_keys=expected_global_unique_image_keys,
    )
    _required_text(scope, field="scope", context="builder")
    assignment_root = assignment_root.expanduser().resolve(strict=True)
    assignment_receipt = assignment_receipt.expanduser().resolve(strict=True)
    partition_receipt = partition_receipt.expanduser().resolve(strict=True)
    clean_train_root = clean_train_root.expanduser().resolve(strict=True)
    output_root = output_root.expanduser().resolve(strict=False)
    if not assignment_root.is_dir() or not clean_train_root.is_dir():
        raise CleanAssignmentBuildError(
            "assignment and clean-train roots must be directories"
        )
    for source_root in (assignment_root, clean_train_root):
        try:
            output_root.relative_to(source_root)
        except ValueError:
            pass
        else:
            raise CleanAssignmentBuildError(
                f"output root must not be inside source root: {source_root}"
            )

    upstream_assignment, assignment_records = _load_assignment_receipt(
        input_root=assignment_root,
        input_receipt=assignment_receipt,
        manifest_names=manifest_names,
        expected_receipt_sha256=expected_assignment_receipt_sha256,
        expected_assignment=expected_assignment,
    )
    upstream_partition, clean_records = _load_partition_receipt(
        clean_train_root=clean_train_root,
        partition_receipt=partition_receipt,
        manifest_names=manifest_names,
        expected_receipt_sha256=expected_partition_receipt_sha256,
        expected_clean_train=expected_clean_train,
        expected_global_unique_image_keys=expected_global_unique_image_keys,
    )
    memberships, clean_scan = _scan_clean_train(
        clean_train_root=clean_train_root,
        manifest_names=manifest_names,
        clean_records=clean_records,
        expected_clean_train=expected_clean_train,
        expected_global_unique_image_keys=expected_global_unique_image_keys,
    )
    replay = _stream_outputs(
        assignment_root=assignment_root,
        clean_train_root=clean_train_root,
        output_root=output_root,
        manifest_names=manifest_names,
        memberships=memberships,
        assignment_records=assignment_records,
        clean_records=clean_records,
        expected_assignment=expected_assignment,
        expected_clean_train=expected_clean_train,
        expected_output=expected_output,
        expected_global_unique_image_keys=expected_global_unique_image_keys,
        write_root=None,
    )
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "scope": scope,
        "row_schema": ROW_SCHEMA,
        "builder": _file_record(Path(__file__)),
        "upstream_assignment_receipt": upstream_assignment,
        "upstream_new_head_partition_receipt": upstream_partition,
        "manifest_order": list(manifest_names),
        "selection_contract": {
            "policy": SELECTION_POLICY,
            "clean_role": "d1_category_complete/train",
            "identity_fields": list(IDENTITY_KEYS),
            "assignment_fields_removed_only_for_base_row_validation": list(
                ASSIGNMENT_FIELDS
            ),
            "source_order": "upstream_assignment_manifest_order",
            "output_encoding": STREAM_ENCODING,
            "pair_reselection_or_repair_allowed": False,
            "valid_partner_contract": (
                "exact_expression_member_in_same_clean_train_manifest_and_image"
            ),
            "model_score_free": True,
            "forbidden_inputs": [
                "teacher_scores",
                "teacher_logits",
                "model_scores",
                "model_logits",
                "checkpoint_outputs",
            ],
        },
        "rows": replay["rows"],
        "valid_rows": replay["valid_rows"],
        "invalid_rows": replay["invalid_rows"],
        "unique_identities": replay["unique_identities"],
        "unique_image_keys": replay["unique_image_keys"],
        "ordered_identity_stream_sha256": replay[
            "ordered_identity_stream_sha256"
        ],
        "base_row_stream_sha256": replay["base_row_stream_sha256"],
        "raw_output_stream_sha256": replay["raw_output_stream_sha256"],
        "clean_train_scan": clean_scan,
        "output_layout": "<source_manifest>",
        "output_stream_encoding": STREAM_ENCODING,
        "manifests": replay["manifests"],
        "invariants": {
            "upstream_assignment_receipt_matches_preregistered_sha256": True,
            "upstream_new_head_partition_receipt_matches_preregistered_sha256": True,
            "all_assignment_and_clean_train_manifests_match_sealed_receipts": True,
            "output_is_raw_byte_for_byte_upstream_assignment_subsequence": True,
            "output_identity_and_order_exactly_match_clean_D1_train": True,
            "removing_exactly_five_assignment_fields_replays_clean_rows": True,
            "assignment_pairs_are_never_reselected_repaired_or_reserialized": True,
            "all_valid_partners_are_exact_clean_train_rows_in_same_image": True,
            "all_invalid_rows_retain_null_partner_and_nonempty_reason": True,
            "clean_train_receipt_seals_image_disjoint_dev_and_quarantine": True,
            "all_outputs_match_preregistered_rows_valid_invalid_size_and_sha256": True,
            "selection_is_deterministic_model_score_free_and_hash_bound": True,
            "builder_uses_create_new_atomic_directory_rename": True,
        },
    }
    if any(value is not True for value in receipt["invariants"].values()):
        raise CleanAssignmentBuildError("one or more output invariants failed")
    receipt["canonical_payload_sha256"] = _sha256_bytes(_canonical_bytes(receipt))
    return BuildPlan(
        receipt=receipt,
        memberships=memberships,
        assignment_records=assignment_records,
        clean_records=clean_records,
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


def _path_lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing any existing entry."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise CleanAssignmentBuildError(
            "renameat2(RENAME_NOREPLACE) is unavailable; refusing unsafe commit"
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
        -100,  # AT_FDCWD
        os.fsencode(source),
        -100,  # AT_FDCWD
        os.fsencode(destination),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise CleanAssignmentBuildError(
            f"refusing concurrent overwrite of output root: {destination}"
        )
    raise CleanAssignmentBuildError(
        "atomic no-replace directory commit failed: "
        f"{source} -> {destination}: {os.strerror(error_number)}"
    )


def build(**kwargs: Any) -> dict[str, Any]:
    requested_root = Path(kwargs.get("output_root", OUTPUT_ROOT)).expanduser()
    if _path_lexists(requested_root):
        raise CleanAssignmentBuildError(
            f"refusing to replace existing output root: {requested_root}"
        )
    output_root = requested_root.resolve()
    plan = make_plan(**kwargs)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.tmp-", dir=str(output_root.parent)
        )
    )
    committed = False
    try:
        replay = _stream_outputs(
            assignment_root=Path(
                kwargs.get("assignment_root", ASSIGNMENT_ROOT)
            ).expanduser().resolve(strict=True),
            clean_train_root=Path(
                kwargs.get("clean_train_root", CLEAN_TRAIN_ROOT)
            ).expanduser().resolve(strict=True),
            output_root=output_root,
            manifest_names=kwargs.get("manifest_names", MANIFESTS),
            memberships=plan.memberships,
            assignment_records=plan.assignment_records,
            clean_records=plan.clean_records,
            expected_assignment=kwargs.get(
                "expected_assignment", EXPECTED_ASSIGNMENT
            ),
            expected_clean_train=kwargs.get(
                "expected_clean_train", EXPECTED_CLEAN_TRAIN
            ),
            expected_output=kwargs.get("expected_output", EXPECTED_OUTPUT),
            expected_global_unique_image_keys=kwargs.get(
                "expected_global_unique_image_keys",
                EXPECTED_GLOBAL_UNIQUE_IMAGE_KEYS,
            ),
            write_root=temporary_root,
        )
        if replay["manifests"] != plan.receipt["manifests"]:
            raise CleanAssignmentBuildError(
                "written output streams differ from the planned streams"
            )
        receipt_path = temporary_root / "receipt.json"
        with receipt_path.open("xb") as handle:
            handle.write(_receipt_bytes(plan.receipt))
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(temporary_root)
        _rename_directory_noreplace(temporary_root, output_root)
        committed = True
        _fsync_directory(output_root.parent)
        return plan.receipt
    finally:
        if not committed and temporary_root.exists():
            shutil.rmtree(temporary_root)


def verify(**kwargs: Any) -> dict[str, Any]:
    requested_root = Path(kwargs.get("output_root", OUTPUT_ROOT)).expanduser()
    if requested_root.is_symlink():
        raise CleanAssignmentBuildError(
            f"output root must not be a symlink: {requested_root}"
        )
    output_root = requested_root.resolve(strict=True)
    if not output_root.is_dir():
        raise CleanAssignmentBuildError(
            f"output root is not a regular directory: {output_root}"
        )
    plan = make_plan(**kwargs)
    manifest_names = tuple(kwargs.get("manifest_names", MANIFESTS))
    expected_files = {Path("receipt.json"), *(Path(name) for name in manifest_names)}
    observed_files = {
        path.relative_to(output_root)
        for path in output_root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if observed_files != expected_files:
        raise CleanAssignmentBuildError(
            "output file set drifted: "
            f"missing={sorted(expected_files - observed_files)}, "
            f"extra={sorted(observed_files - expected_files)}"
        )
    receipt_path = output_root / "receipt.json"
    if receipt_path.is_symlink() or receipt_path.read_bytes() != _receipt_bytes(
        plan.receipt
    ):
        raise CleanAssignmentBuildError("clean assignment receipt does not replay exactly")
    for name in manifest_names:
        path = output_root / name
        if path.is_symlink():
            raise CleanAssignmentBuildError(f"output must not be a symlink: {path}")
        observed = _file_record(path)
        expected = plan.receipt["manifests"][name]["output"]
        if (
            observed["sha256"] != expected["sha256"]
            or observed["size_bytes"] != expected["size_bytes"]
        ):
            raise CleanAssignmentBuildError(
                f"clean assignment output does not replay exactly: {name}"
            )
        rows = 0
        with path.open("rb") as handle:
            for rows, raw in enumerate(handle, start=1):
                if not raw.endswith(b"\n"):
                    raise CleanAssignmentBuildError(
                        f"output row lost its line ending: {path}:{rows}"
                    )
        if rows != plan.receipt["manifests"][name]["rows"]:
            raise CleanAssignmentBuildError(f"output row count drifted: {path}")
    return plan.receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment-root", type=Path, default=ASSIGNMENT_ROOT)
    parser.add_argument(
        "--assignment-receipt", type=Path, default=ASSIGNMENT_RECEIPT
    )
    parser.add_argument("--partition-receipt", type=Path, default=PARTITION_RECEIPT)
    parser.add_argument("--clean-train-root", type=Path, default=CLEAN_TRAIN_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="replay all contracts and print the exact receipt without writing",
    )
    mode.add_argument(
        "--verify",
        action="store_true",
        help="replay all contracts and verify an existing output root",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kwargs = {
        "assignment_root": args.assignment_root,
        "assignment_receipt": args.assignment_receipt,
        "partition_receipt": args.partition_receipt,
        "clean_train_root": args.clean_train_root,
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
