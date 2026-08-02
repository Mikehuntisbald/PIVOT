#!/usr/bin/env python3
"""Build the fixed, model-score-free DD1 PairTop1 Overfit64 subset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
import shutil
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.stageb_ref_split_contract import (  # noqa: E402
    REF_SPLIT_CONTRACT,
    REF_SPLIT_MANIFEST_FILES,
    REF_SPLITS,
)


INPUT_ROOT = (
    REPO_ROOT / "data/ablations/stageb_data_driven_assignment_pairs_20260722"
)
INPUT_RECEIPT = INPUT_ROOT / "receipt.json"
HELDOUT_ROOT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/baseline_b58_ref8_seed42/refcoco_eval_inputs"
)
SUPPORT_TSV = (
    Path("/media/haoyi/T9/data")
    / "patches_quality_emb/emb_index_from_quality.tsv"
)
SUPPORT_BANK_CACHE = Path(str(SUPPORT_TSV) + ".bank.clean.img.pkl")
SUPPORT_IMAGE_ROOT = Path("/media/haoyi/T9/data/patches_quality")
CANONICAL_CLASSES = Path(
    "/media/haoyi/T9/data/canonical_classes_with_aliases.json"
)
OUTPUT_ROOT = (
    REPO_ROOT
    / "data/ablations/stageb_data_driven_assignment_overfit64_20260722"
)
OUTPUT_MANIFEST = "overfit64.jsonl"
OUTPUT_SUPPORT_TSV = "overfit64_support_clean.tsv"

RECEIPT_SCHEMA = (
    "pivot.stageb.data_driven.assignment_overfit64_receipt/v1"
)
ROW_SCHEMA = "pivot.stageb.data_driven.official_assignment_pair/v1"
UPSTREAM_RECEIPT_SCHEMA = (
    "pivot.stageb.data_driven.official_assignment_pair_receipt/v1"
)
PAIR_ID_SCHEMA = "pivot.stageb.data_driven.assignment_pair_member_id/v1"
SELECTION_POLICY = "sha256_pair_id_priority_quota_order_greedy_v1"
SELECTION_NAMESPACE = (
    "pivot.stageb.data_driven.assignment_pair_overfit64.selection/v1"
)
STREAM_ENCODING = "utf8_text_record_plus_lf_v1"

MANIFEST_QUOTAS = (
    ("refcoco_stageb_phrase_v1.jsonl", 22),
    ("refcocoplus_stageb_phrase_v1.jsonl", 21),
    ("refcocog_stageb_phrase_v1.jsonl", 21),
)
EXPECTED_INPUT_RECEIPT_SHA256 = (
    "7b9ce1c911a2e1f0b67464243df8290fc2baf0786a2a3b131ddc57a6a6d2ddaa"
)
EXPECTED_INPUT_SHA256 = {
    "refcoco_stageb_phrase_v1.jsonl": (
        "f253c8bec4d15e421b11c42d8114e17c41bc32ed28f2614e34fe341e4da32592"
    ),
    "refcocoplus_stageb_phrase_v1.jsonl": (
        "69039abbd5baeb1173c849c19c55128aea8053271ab79f0cc16fa679000deaa8"
    ),
    "refcocog_stageb_phrase_v1.jsonl": (
        "378c5e34899e4113cd5dca1fd60352b362924e7969a717ea729b8278ce97a553"
    ),
}
EXPECTED_INPUT_ROWS = {
    "refcoco_stageb_phrase_v1.jsonl": 120624,
    "refcocoplus_stageb_phrase_v1.jsonl": 120191,
    "refcocog_stageb_phrase_v1.jsonl": 80512,
}
EXPECTED_HELDOUT_UNION_IMAGES = 6549
EXPECTED_SUPPORT_SHA256 = {
    "support_tsv": (
        "93b1de99dd611577470960c5194faee15909af52b066713f956bbb6f25f78d47"
    ),
    "support_bank_cache": (
        "01ff270cef70bcf93e9884e89fb58d9a64ce749a1db5c2f7ca7a1ba3fde799c1"
    ),
    "canonical_classes": (
        "9074350284a759a8de8dc619cae86a0c76bfe6fa5db597f279507443089fd853"
    ),
}
EXPECTED_CATEGORY_COMPLETE_RECEIPT_SHA256 = (
    "fab09c61a8f53f05d75eedff25039a843ff27cb2d491d6c6576fe2b1e8aedd74"
)
_HEX64_RE = re.compile(r"[0-9a-f]{64}")


class Overfit64BuildError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Candidate:
    manifest_name: str
    line_number: int
    pair_id: str
    priority_sha256: str
    source_row_sha256: str
    source: str
    image_id: int
    anchor_ann_id: int
    partner_ann_id: int
    class_id: int
    ref_id: int
    sent_id: int

    @property
    def edge(self) -> tuple[int, int]:
        return tuple(sorted((self.anchor_ann_id, self.partner_ann_id)))


@dataclass(frozen=True, slots=True)
class BuildPlan:
    manifest_bytes: bytes
    support_tsv_bytes: bytes
    receipt: dict[str, Any]


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
        raise Overfit64BuildError(f"not a file: {path}")
    before = path.stat()
    digest = _sha256_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (
        after.st_size,
        after.st_mtime_ns,
    ):
        raise Overfit64BuildError(f"file changed while hashing: {path}")
    return {
        "path": str((reported_path or path).expanduser().resolve()),
        "size_bytes": int(before.st_size),
        "sha256": digest,
    }


def _predicted_file_record(path: Path, payload: bytes) -> dict[str, Any]:
    return {
        "path": str(path.expanduser().resolve()),
        "size_bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise Overfit64BuildError(f"could not load {label}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise Overfit64BuildError(f"{label} must be a JSON object: {path}")
    return value


def _required_int(value: Any, *, field: str, context: str) -> int:
    if type(value) is not int:
        raise Overfit64BuildError(f"{context}: {field} must be an exact integer")
    return int(value)


def _required_text(value: Any, *, field: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Overfit64BuildError(f"{context}: {field} must be non-empty text")
    return value.strip()


def _validate_sha256(value: str, *, label: str) -> None:
    if not isinstance(value, str) or _HEX64_RE.fullmatch(value) is None:
        raise Overfit64BuildError(f"{label} is not a lowercase SHA-256")


def _record_stream_sha256(records: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _validate_expected_inputs(
    manifest_quotas: Sequence[tuple[str, int]],
    expected_input_sha256: Mapping[str, str],
    expected_input_rows: Mapping[str, int],
) -> None:
    names = [name for name, _quota in manifest_quotas]
    if len(set(names)) != len(names):
        raise Overfit64BuildError("manifest quota names must be unique")
    if set(names) != set(expected_input_sha256) or set(names) != set(
        expected_input_rows
    ):
        raise Overfit64BuildError(
            "expected input maps must exactly match manifest quota names"
        )
    for name, quota in manifest_quotas:
        if Path(name).name != name:
            raise Overfit64BuildError(f"manifest name is not a basename: {name}")
        if type(quota) is not int or quota <= 0:
            raise Overfit64BuildError(f"invalid quota for {name}: {quota!r}")
        _validate_sha256(expected_input_sha256[name], label=f"{name} expected hash")
        if type(expected_input_rows[name]) is not int or expected_input_rows[name] <= 0:
            raise Overfit64BuildError(f"invalid expected row count for {name}")


def _load_upstream_receipt(
    *,
    input_receipt: Path,
    expected_receipt_sha256: str,
    input_root: Path,
    manifest_quotas: Sequence[tuple[str, int]],
    expected_input_sha256: Mapping[str, str],
    expected_input_rows: Mapping[str, int],
    expected_category_complete_receipt_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _validate_sha256(expected_receipt_sha256, label="expected upstream receipt hash")
    receipt_record = _file_record(input_receipt)
    if receipt_record["sha256"] != expected_receipt_sha256:
        raise Overfit64BuildError(
            "upstream assignment receipt SHA-256 mismatch: "
            f"expected={expected_receipt_sha256}, "
            f"observed={receipt_record['sha256']}"
        )
    receipt = _load_json_object(input_receipt, label="upstream assignment receipt")
    expected_total = sum(expected_input_rows.values())
    if (
        receipt.get("schema") != UPSTREAM_RECEIPT_SCHEMA
        or receipt.get("row_schema") != ROW_SCHEMA
        or receipt.get("rows") != expected_total
        or receipt.get("unique_identities") != expected_total
        or receipt.get("manifest_order")
        != [name for name, _quota in manifest_quotas]
        or any(value is not True for value in (receipt.get("invariants") or {}).values())
    ):
        raise Overfit64BuildError("upstream assignment receipt contract drifted")

    manifests = receipt.get("manifests")
    if not isinstance(manifests, dict) or set(manifests) != {
        name for name, _quota in manifest_quotas
    }:
        raise Overfit64BuildError("upstream assignment receipt manifests drifted")
    source_records: dict[str, Any] = {}
    for name, _quota in manifest_quotas:
        path = input_root / name
        record = _file_record(path)
        upstream = manifests.get(name)
        output = upstream.get("output") if isinstance(upstream, dict) else None
        if (
            record["sha256"] != expected_input_sha256[name]
            or not isinstance(output, dict)
            or output.get("sha256") != record["sha256"]
            or output.get("size_bytes") != record["size_bytes"]
            or upstream.get("rows") != expected_input_rows[name]
        ):
            raise Overfit64BuildError(f"upstream manifest binding drifted for {name}")
        source_records[name] = record

    category_record = receipt.get("category_complete_receipt")
    if not isinstance(category_record, dict):
        raise Overfit64BuildError("upstream category-complete receipt is absent")
    category_path_value = category_record.get("path")
    if not isinstance(category_path_value, str) or not category_path_value.strip():
        raise Overfit64BuildError("upstream category-complete receipt path is invalid")
    category_path = Path(category_path_value).expanduser().resolve(strict=True)
    observed_category_record = _file_record(category_path)
    if (
        category_record.get("sha256") != expected_category_complete_receipt_sha256
        or observed_category_record["sha256"]
        != expected_category_complete_receipt_sha256
    ):
        raise Overfit64BuildError("category-complete receipt SHA-256 drifted")
    return receipt_record, source_records, observed_category_record


def _load_heldout_images(
    *,
    heldout_root: Path,
    heldout_contract: Mapping[str, Mapping[str, Any]],
    heldout_manifest_files: Mapping[str, str],
    heldout_splits: Sequence[str],
    expected_union_images: int,
) -> tuple[set[int], dict[str, Any]]:
    if set(heldout_splits) != set(heldout_contract) or set(
        heldout_splits
    ) != set(heldout_manifest_files):
        raise Overfit64BuildError("heldout split contract is incomplete")
    image_ids: set[int] = set()
    split_records: dict[str, Any] = {}
    total_rows = 0
    for split in heldout_splits:
        contract = heldout_contract[split]
        filename = heldout_manifest_files[split]
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise Overfit64BuildError(f"heldout filename is invalid for {split}")
        expected_rows = contract.get("rows")
        expected_sha = contract.get("sha256")
        if type(expected_rows) is not int or expected_rows <= 0:
            raise Overfit64BuildError(f"heldout row count is invalid for {split}")
        _validate_sha256(expected_sha, label=f"{split} heldout hash")
        path = heldout_root / filename
        record = _file_record(path)
        if record["sha256"] != expected_sha:
            raise Overfit64BuildError(f"heldout manifest SHA-256 drifted for {split}")
        rows = 0
        with path.open("rb") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    raise Overfit64BuildError(
                        f"blank heldout row at {path}:{line_number}"
                    )
                try:
                    row = json.loads(raw)
                except (UnicodeError, json.JSONDecodeError) as error:
                    raise Overfit64BuildError(
                        f"invalid heldout JSON at {path}:{line_number}: {error}"
                    ) from error
                if not isinstance(row, dict):
                    raise Overfit64BuildError(
                        f"heldout row is not an object at {path}:{line_number}"
                    )
                image_ids.add(
                    _required_int(
                        row.get("image_id"),
                        field="image_id",
                        context=f"{path}:{line_number}",
                    )
                )
                rows += 1
        if rows != expected_rows:
            raise Overfit64BuildError(
                f"heldout row count drifted for {split}: {rows} != {expected_rows}"
            )
        total_rows += rows
        split_records[split] = {"rows": rows, "manifest": record}
    if len(image_ids) != expected_union_images:
        raise Overfit64BuildError(
            "all-eight heldout image union drifted: "
            f"observed={len(image_ids)}, expected={expected_union_images}"
        )
    union_payload = _canonical_bytes(sorted(image_ids))
    return image_ids, {
        "split_order": list(heldout_splits),
        "splits": split_records,
        "rows": total_rows,
        "unique_images": len(image_ids),
        "sorted_image_id_json_sha256": _sha256_bytes(union_payload),
    }


class SupportBank:
    def __init__(
        self,
        *,
        support_tsv: Path,
        support_bank_cache: Path,
        support_image_root: Path,
        canonical_classes: Path,
        expected_sha256: Mapping[str, str],
    ) -> None:
        required = {"support_tsv", "support_bank_cache", "canonical_classes"}
        if set(expected_sha256) != required:
            raise Overfit64BuildError("support expected-hash map is incomplete")
        paths = {
            "support_tsv": support_tsv,
            "support_bank_cache": support_bank_cache,
            "canonical_classes": canonical_classes,
        }
        self.records: dict[str, Any] = {}
        for role, path in paths.items():
            _validate_sha256(expected_sha256[role], label=f"{role} expected hash")
            record = _file_record(path)
            if record["sha256"] != expected_sha256[role]:
                raise Overfit64BuildError(f"{role} SHA-256 drifted")
            self.records[role] = record

        support_image_root = support_image_root.expanduser().resolve(strict=True)
        if not support_image_root.is_dir():
            raise Overfit64BuildError(
                f"support image root is not a directory: {support_image_root}"
            )
        self.support_image_root = support_image_root
        try:
            with support_bank_cache.open("rb") as handle:
                payload = pickle.load(handle)
        except (OSError, pickle.PickleError, EOFError) as error:
            raise Overfit64BuildError(
                f"could not load support bank cache: {support_bank_cache}: {error}"
            ) from error
        if not isinstance(payload, dict):
            raise Overfit64BuildError("support bank cache must be a dictionary")
        meta = payload.get("meta")
        bank = payload.get("bank")
        if not isinstance(meta, dict) or not isinstance(bank, dict):
            raise Overfit64BuildError("support bank cache payload is malformed")
        expected_meta_paths = {
            "tsv_path": support_tsv,
            "canonical_classes_json": canonical_classes,
            "support_patch_image_root": support_image_root,
        }
        for field, expected_path in expected_meta_paths.items():
            value = meta.get(field)
            if not isinstance(value, str) or Path(value).expanduser().resolve() != (
                expected_path.expanduser().resolve()
            ):
                raise Overfit64BuildError(f"support bank cache {field} drifted")
        if (
            meta.get("bucket") != "clean"
            or meta.get("use_embedding") is not False
            or type(meta.get("max_per_class")) is not int
            or int(meta["max_per_class"]) <= 0
        ):
            raise Overfit64BuildError("support bank cache mode is not clean image mode")

        normalized: dict[int, tuple[str, ...]] = {}
        path_stream: list[str] = []
        for raw_class_id, raw_paths in bank.items():
            if isinstance(raw_class_id, bool):
                raise Overfit64BuildError("support bank contains a boolean class id")
            try:
                class_id = int(raw_class_id)
            except (TypeError, ValueError) as error:
                raise Overfit64BuildError("support bank contains a non-integer class id") from error
            if class_id < 0 or not isinstance(raw_paths, list):
                raise Overfit64BuildError(
                    f"support bank entry is malformed for class {class_id}"
                )
            paths_for_class: list[str] = []
            for raw_path in raw_paths:
                if not isinstance(raw_path, str) or not raw_path.strip():
                    raise Overfit64BuildError(
                        f"support bank contains an invalid path for class {class_id}"
                    )
                resolved = Path(raw_path).expanduser().resolve()
                try:
                    resolved.relative_to(support_image_root)
                except ValueError as error:
                    raise Overfit64BuildError(
                        "support bank image escapes the external support root: "
                        f"{resolved}"
                    ) from error
                paths_for_class.append(str(resolved))
            unique_paths = tuple(sorted(set(paths_for_class)))
            if unique_paths:
                normalized[class_id] = unique_paths
            for path in unique_paths:
                path_stream.append(f"{class_id}\t{path}")
        self.bank = normalized
        self.meta = dict(meta)
        self.path_stream_sha256 = _record_stream_sha256(path_stream)
        self.file_count = sum(len(paths) for paths in normalized.values())
        self._witness_cache: dict[int, Path | None] = {}

    def witness(self, class_id: int) -> Path | None:
        if class_id in self._witness_cache:
            return self._witness_cache[class_id]
        witness = None
        for value in self.bank.get(class_id, ()):
            path = Path(value)
            if path.is_file():
                witness = path
                break
        self._witness_cache[class_id] = witness
        return witness

    def receipt(self, selected_class_ids: set[int]) -> dict[str, Any]:
        witnesses = []
        for class_id in sorted(selected_class_ids):
            path = self.witness(class_id)
            if path is None:
                raise Overfit64BuildError(
                    f"selected class lost external support coverage: {class_id}"
                )
            witnesses.append({"class_id": class_id, "image": _file_record(path)})
        return {
            **self.records,
            "support_image_root": str(self.support_image_root),
            "cache_contract": {
                "version": self.meta.get("version"),
                "bucket": self.meta.get("bucket"),
                "use_embedding": self.meta.get("use_embedding"),
                "max_per_class": self.meta.get("max_per_class"),
            },
            "bank_classes": len(self.bank),
            "bank_files": self.file_count,
            "ordered_class_path_stream_encoding": STREAM_ENCODING,
            "ordered_class_path_stream_sha256": self.path_stream_sha256,
            "selected_class_witnesses": witnesses,
            "target_crop_fallback_allowed": False,
        }

    def mini_tsv(self, selected_class_ids: set[int]) -> bytes:
        lines = ["class_id\tbucket\tpath\temb_rel_path\n"]
        for class_id in sorted(selected_class_ids):
            path = self.witness(class_id)
            if path is None:
                raise Overfit64BuildError(
                    f"selected class lost external support coverage: {class_id}"
                )
            relative = path.relative_to(self.support_image_root)
            embedding_relative = relative.with_suffix(".npy")
            fields = (
                str(class_id),
                "clean",
                str(path),
                str(embedding_relative),
            )
            if any("\t" in value or "\n" in value or "\r" in value for value in fields):
                raise Overfit64BuildError(
                    f"support witness cannot be serialized as TSV: {path}"
                )
            lines.append("\t".join(fields) + "\n")
        return "".join(lines).encode("utf-8")


def _candidate_from_row(
    row: Any,
    *,
    manifest_name: str,
    line_number: int,
    source_row_sha256: str,
) -> Candidate | None:
    context = f"{manifest_name}:{line_number}"
    if not isinstance(row, dict):
        raise Overfit64BuildError(f"{context}: row must be an object")
    if row.get("assignment_pair_valid") is not True:
        return None
    if (
        row.get("stage_b_data_driven_assignment_pair") is not True
        or row.get("stage_b_data_driven_assignment_pair_schema") != ROW_SCHEMA
    ):
        raise Overfit64BuildError(f"{context}: valid row lost assignment schema")
    pair = row.get("assignment_pair")
    anchor = pair.get("anchor") if isinstance(pair, dict) else None
    partner = pair.get("partner") if isinstance(pair, dict) else None
    if (
        not isinstance(pair, dict)
        or pair.get("schema") != ROW_SCHEMA
        or not isinstance(anchor, dict)
        or not isinstance(partner, dict)
    ):
        raise Overfit64BuildError(f"{context}: assignment payload is malformed")
    image_id = _required_int(row.get("image_id"), field="image_id", context=context)
    anchor_image_id = _required_int(
        anchor.get("image_id"), field="anchor.image_id", context=context
    )
    partner_image_id = _required_int(
        partner.get("image_id"), field="partner.image_id", context=context
    )
    anchor_ann_id = _required_int(
        anchor.get("coco_ann_id"), field="anchor.coco_ann_id", context=context
    )
    partner_ann_id = _required_int(
        partner.get("coco_ann_id"), field="partner.coco_ann_id", context=context
    )
    if (
        image_id != anchor_image_id
        or image_id != partner_image_id
        or anchor_ann_id == partner_ann_id
        or _required_int(row.get("ann_id"), field="ann_id", context=context)
        != anchor_ann_id
    ):
        raise Overfit64BuildError(f"{context}: assignment endpoint identity drifted")
    instances = row.get("instances")
    if (
        row.get("primary_support_instance_index") != 0
        or not isinstance(instances, list)
        or len(instances) < 2
        or not isinstance(instances[0], dict)
    ):
        raise Overfit64BuildError(f"{context}: primary support instance is malformed")
    primary = instances[0]
    if (
        primary.get("category_complete_primary") is not True
        or _required_int(
            primary.get("coco_ann_id"),
            field="instances[0].coco_ann_id",
            context=context,
        )
        != anchor_ann_id
    ):
        raise Overfit64BuildError(f"{context}: primary instance is not the anchor")
    class_id = _required_int(
        primary.get("class_id"), field="instances[0].class_id", context=context
    )
    source = _required_text(row.get("source"), field="source", context=context)
    ref_id = _required_int(row.get("ref_id"), field="ref_id", context=context)
    sent_id = _required_int(row.get("sent_id"), field="sent_id", context=context)
    pair_id_payload = {
        "schema": PAIR_ID_SCHEMA,
        "manifest": manifest_name,
        "source": source,
        "image_id": image_id,
        "anchor_coco_ann_id": anchor_ann_id,
        "partner_coco_ann_id": partner_ann_id,
        "ref_id": ref_id,
        "sent_id": sent_id,
    }
    pair_id = _sha256_bytes(_canonical_bytes(pair_id_payload))
    priority_sha256 = _sha256_bytes(
        SELECTION_NAMESPACE.encode("ascii") + b"\x00" + pair_id.encode("ascii")
    )
    return Candidate(
        manifest_name=manifest_name,
        line_number=line_number,
        pair_id=pair_id,
        priority_sha256=priority_sha256,
        source_row_sha256=source_row_sha256,
        source=source,
        image_id=image_id,
        anchor_ann_id=anchor_ann_id,
        partner_ann_id=partner_ann_id,
        class_id=class_id,
        ref_id=ref_id,
        sent_id=sent_id,
    )


def _select_candidates(
    *,
    input_root: Path,
    manifest_quotas: Sequence[tuple[str, int]],
    expected_input_rows: Mapping[str, int],
    heldout_images: set[int],
    support_bank: SupportBank,
) -> tuple[list[Candidate], dict[str, Any]]:
    selected: list[Candidate] = []
    used_images: set[int] = set()
    used_edges: set[tuple[int, int]] = set()
    used_endpoints: set[int] = set()
    manifest_statistics: dict[str, Any] = {}
    for manifest_name, quota in manifest_quotas:
        path = input_root / manifest_name
        candidates: list[Candidate] = []
        stats = Counter()
        with path.open("rb") as handle:
            for line_number, raw in enumerate(handle, start=1):
                stats["rows"] += 1
                stripped = raw.rstrip(b"\r\n")
                if not stripped:
                    raise Overfit64BuildError(
                        f"blank input row at {path}:{line_number}"
                    )
                try:
                    row = json.loads(stripped)
                except (UnicodeError, json.JSONDecodeError) as error:
                    raise Overfit64BuildError(
                        f"invalid input JSON at {path}:{line_number}: {error}"
                    ) from error
                candidate = _candidate_from_row(
                    row,
                    manifest_name=manifest_name,
                    line_number=line_number,
                    source_row_sha256=_sha256_bytes(stripped),
                )
                if candidate is None:
                    stats["invalid_assignment_rows"] += 1
                    continue
                stats["valid_assignment_rows"] += 1
                if candidate.image_id in heldout_images:
                    stats["heldout_image_excluded_rows"] += 1
                    continue
                if support_bank.witness(candidate.class_id) is None:
                    stats["external_support_uncovered_rows"] += 1
                    continue
                stats["eligible_rows"] += 1
                candidates.append(candidate)
        if stats["rows"] != expected_input_rows[manifest_name]:
            raise Overfit64BuildError(
                f"input row count drifted for {manifest_name}: "
                f"{stats['rows']} != {expected_input_rows[manifest_name]}"
            )
        candidates.sort(
            key=lambda value: (
                value.priority_sha256,
                value.pair_id,
                value.line_number,
            )
        )
        skip_counts = Counter()
        chosen_for_manifest: list[Candidate] = []
        for candidate in candidates:
            if candidate.image_id in used_images:
                skip_counts["duplicate_image"] += 1
                continue
            if candidate.edge in used_edges:
                skip_counts["duplicate_unordered_edge"] += 1
                continue
            endpoints = {candidate.anchor_ann_id, candidate.partner_ann_id}
            if endpoints.intersection(used_endpoints):
                skip_counts["duplicate_endpoint"] += 1
                continue
            chosen_for_manifest.append(candidate)
            used_images.add(candidate.image_id)
            used_edges.add(candidate.edge)
            used_endpoints.update(endpoints)
            if len(chosen_for_manifest) == quota:
                break
        if len(chosen_for_manifest) != quota:
            raise Overfit64BuildError(
                f"could not satisfy quota for {manifest_name}: "
                f"selected={len(chosen_for_manifest)}, quota={quota}"
            )
        selected.extend(chosen_for_manifest)
        manifest_statistics[manifest_name] = {
            **dict(sorted(stats.items())),
            "quota": quota,
            "selected_rows": len(chosen_for_manifest),
            "greedy_skip_histogram_before_quota": dict(sorted(skip_counts.items())),
        }
    expected_total = sum(quota for _name, quota in manifest_quotas)
    if (
        len(selected) != expected_total
        or len(used_images) != expected_total
        or len(used_edges) != expected_total
        or len(used_endpoints) != 2 * expected_total
    ):
        raise Overfit64BuildError("selected member uniqueness contract failed")
    return selected, manifest_statistics


def _load_selected_row_bytes(
    *, input_root: Path, selected: Sequence[Candidate]
) -> list[bytes]:
    by_manifest: dict[str, dict[int, Candidate]] = {}
    for candidate in selected:
        by_manifest.setdefault(candidate.manifest_name, {})[
            candidate.line_number
        ] = candidate
    loaded: dict[str, bytes] = {}
    for manifest_name, line_map in by_manifest.items():
        path = input_root / manifest_name
        remaining = set(line_map)
        with path.open("rb") as handle:
            for line_number, raw in enumerate(handle, start=1):
                candidate = line_map.get(line_number)
                if candidate is None:
                    continue
                stripped = raw.rstrip(b"\r\n")
                if _sha256_bytes(stripped) != candidate.source_row_sha256:
                    raise Overfit64BuildError(
                        f"selected source row changed at {path}:{line_number}"
                    )
                row = json.loads(stripped)
                replay = _candidate_from_row(
                    row,
                    manifest_name=manifest_name,
                    line_number=line_number,
                    source_row_sha256=candidate.source_row_sha256,
                )
                if replay != candidate:
                    raise Overfit64BuildError(
                        f"selected source identity changed at {path}:{line_number}"
                    )
                loaded[candidate.pair_id] = stripped + b"\n"
                remaining.remove(line_number)
                if not remaining:
                    break
        if remaining:
            raise Overfit64BuildError(
                f"selected source rows are missing from {manifest_name}: {sorted(remaining)}"
            )
    if len(loaded) != len(selected):
        raise Overfit64BuildError("selected pair IDs are not unique")
    return [loaded[candidate.pair_id] for candidate in selected]


def make_plan(
    *,
    input_root: Path = INPUT_ROOT,
    input_receipt: Path = INPUT_RECEIPT,
    heldout_root: Path = HELDOUT_ROOT,
    support_tsv: Path = SUPPORT_TSV,
    support_bank_cache: Path = SUPPORT_BANK_CACHE,
    support_image_root: Path = SUPPORT_IMAGE_ROOT,
    canonical_classes: Path = CANONICAL_CLASSES,
    output_root: Path = OUTPUT_ROOT,
    output_manifest: str = OUTPUT_MANIFEST,
    output_support_tsv: str = OUTPUT_SUPPORT_TSV,
    manifest_quotas: Sequence[tuple[str, int]] = MANIFEST_QUOTAS,
    expected_input_receipt_sha256: str = EXPECTED_INPUT_RECEIPT_SHA256,
    expected_input_sha256: Mapping[str, str] = EXPECTED_INPUT_SHA256,
    expected_input_rows: Mapping[str, int] = EXPECTED_INPUT_ROWS,
    heldout_contract: Mapping[str, Mapping[str, Any]] = REF_SPLIT_CONTRACT,
    heldout_manifest_files: Mapping[str, str] = REF_SPLIT_MANIFEST_FILES,
    heldout_splits: Sequence[str] = REF_SPLITS,
    expected_heldout_union_images: int = EXPECTED_HELDOUT_UNION_IMAGES,
    expected_support_sha256: Mapping[str, str] = EXPECTED_SUPPORT_SHA256,
    expected_category_complete_receipt_sha256: str = (
        EXPECTED_CATEGORY_COMPLETE_RECEIPT_SHA256
    ),
) -> BuildPlan:
    _validate_expected_inputs(
        manifest_quotas, expected_input_sha256, expected_input_rows
    )
    if Path(output_manifest).name != output_manifest or not output_manifest.endswith(
        ".jsonl"
    ):
        raise Overfit64BuildError("output manifest must be a JSONL basename")
    if (
        Path(output_support_tsv).name != output_support_tsv
        or not output_support_tsv.endswith(".tsv")
    ):
        raise Overfit64BuildError("output support TSV must be a TSV basename")
    input_root = input_root.expanduser().resolve(strict=True)
    heldout_root = heldout_root.expanduser().resolve(strict=True)
    output_root = output_root.expanduser().resolve(strict=False)
    if not input_root.is_dir() or not heldout_root.is_dir():
        raise Overfit64BuildError("input and heldout roots must be directories")
    upstream_receipt, source_records, category_receipt = _load_upstream_receipt(
        input_receipt=input_receipt,
        expected_receipt_sha256=expected_input_receipt_sha256,
        input_root=input_root,
        manifest_quotas=manifest_quotas,
        expected_input_sha256=expected_input_sha256,
        expected_input_rows=expected_input_rows,
        expected_category_complete_receipt_sha256=(
            expected_category_complete_receipt_sha256
        ),
    )
    heldout_images, heldout_receipt = _load_heldout_images(
        heldout_root=heldout_root,
        heldout_contract=heldout_contract,
        heldout_manifest_files=heldout_manifest_files,
        heldout_splits=heldout_splits,
        expected_union_images=expected_heldout_union_images,
    )
    support_bank = SupportBank(
        support_tsv=support_tsv,
        support_bank_cache=support_bank_cache,
        support_image_root=support_image_root,
        canonical_classes=canonical_classes,
        expected_sha256=expected_support_sha256,
    )
    selected, manifest_statistics = _select_candidates(
        input_root=input_root,
        manifest_quotas=manifest_quotas,
        expected_input_rows=expected_input_rows,
        heldout_images=heldout_images,
        support_bank=support_bank,
    )
    selected_row_bytes = _load_selected_row_bytes(
        input_root=input_root, selected=selected
    )
    manifest_payload = b"".join(selected_row_bytes)
    output_path = output_root / output_manifest
    selected_class_ids = {candidate.class_id for candidate in selected}
    mini_support_payload = support_bank.mini_tsv(selected_class_ids)
    mini_support_path = output_root / output_support_tsv
    pair_ids = [candidate.pair_id for candidate in selected]
    image_ids = [str(candidate.image_id) for candidate in selected]
    edges = [
        f"{candidate.edge[0]}\t{candidate.edge[1]}" for candidate in selected
    ]
    endpoints = [
        f"{candidate.anchor_ann_id}\t{candidate.partner_ann_id}"
        for candidate in selected
    ]
    member_records = [
        {
            "output_index": index,
            "manifest": candidate.manifest_name,
            "source_line_number": candidate.line_number,
            "source_row_sha256": candidate.source_row_sha256,
            "pair_id": candidate.pair_id,
            "priority_sha256": candidate.priority_sha256,
            "source": candidate.source,
            "image_id": candidate.image_id,
            "anchor_coco_ann_id": candidate.anchor_ann_id,
            "partner_coco_ann_id": candidate.partner_ann_id,
            "class_id": candidate.class_id,
            "ref_id": candidate.ref_id,
            "sent_id": candidate.sent_id,
        }
        for index, candidate in enumerate(selected)
    ]
    source_counts = Counter(candidate.manifest_name for candidate in selected)
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "row_schema": ROW_SCHEMA,
        "builder": _file_record(Path(__file__)),
        "upstream_assignment_receipt": upstream_receipt,
        "upstream_category_complete_receipt": category_receipt,
        "source_manifest_order": [name for name, _quota in manifest_quotas],
        "source_manifests": {
            name: {
                "input": source_records[name],
                **manifest_statistics[name],
            }
            for name, _quota in manifest_quotas
        },
        "heldout": heldout_receipt,
        "support": {
            **support_bank.receipt(selected_class_ids),
            "mini_support_tsv": _predicted_file_record(
                mini_support_path, mini_support_payload
            ),
            "mini_support_rows": len(selected_class_ids),
            "mini_support_candidates_per_class": 1,
        },
        "selection_contract": {
            "policy": SELECTION_POLICY,
            "namespace": SELECTION_NAMESPACE,
            "pair_id_schema": PAIR_ID_SCHEMA,
            "pair_id_payload_fields": [
                "schema",
                "manifest",
                "source",
                "image_id",
                "anchor_coco_ann_id",
                "partner_coco_ann_id",
                "ref_id",
                "sent_id",
            ],
            "pair_id_serialization": "canonical_ascii_json_sort_keys_compact_v1",
            "priority_serialization": "namespace_ascii_nul_pair_id_ascii_v1",
            "priority_order": "priority_sha256_pair_id_source_line_ascending",
            "quota_order": [
                {"manifest": name, "rows": quota}
                for name, quota in manifest_quotas
            ],
            "greedy_rejections": [
                "heldout_image",
                "external_support_uncovered",
                "duplicate_image",
                "duplicate_unordered_annotation_edge",
                "duplicate_annotation_endpoint",
            ],
            "all_eight_official_ref_splits_are_heldout": True,
            "external_clean_support_required": True,
            "runtime_support_source": output_support_tsv,
            "runtime_support_candidates_per_selected_class": 1,
            "target_crop_fallback_allowed": False,
            "model_score_free": True,
            "forbidden_inputs": [
                "teacher_scores",
                "teacher_logits",
                "model_scores",
                "model_logits",
                "checkpoint_outputs",
            ],
        },
        "rows": len(selected),
        "valid_rows": len(selected),
        "invalid_rows": 0,
        "unique_images": len({candidate.image_id for candidate in selected}),
        "unique_unordered_annotation_edges": len(
            {candidate.edge for candidate in selected}
        ),
        "unique_annotation_endpoints": len(
            {
                endpoint
                for candidate in selected
                for endpoint in (
                    candidate.anchor_ann_id,
                    candidate.partner_ann_id,
                )
            }
        ),
        "source_counts": dict(source_counts),
        "ordered_member_stream_encoding": STREAM_ENCODING,
        "ordered_member_pair_id_stream_sha256": _record_stream_sha256(pair_ids),
        "ordered_image_id_stream_sha256": _record_stream_sha256(image_ids),
        "ordered_unordered_edge_stream_sha256": _record_stream_sha256(edges),
        "ordered_endpoint_stream_sha256": _record_stream_sha256(endpoints),
        "members": member_records,
        "output_manifest": output_manifest,
        "output": _predicted_file_record(output_path, manifest_payload),
        "invariants": {
            "all_source_manifests_match_preregistered_sha256": True,
            "upstream_assignment_receipt_matches_preregistered_sha256": True,
            "upstream_category_complete_receipt_matches_preregistered_sha256": True,
            "all_eight_heldout_manifests_match_the_official_contract": True,
            "selected_images_are_disjoint_from_all_eight_heldout_splits": True,
            "selected_rows_are_valid_official_assignment_pairs": True,
            "selected_rows_retain_the_complete_upstream_row_bytes": True,
            "selected_primary_classes_have_external_clean_support": True,
            "runtime_mini_support_has_one_candidate_per_selected_class": True,
            "selected_support_witness_images_are_content_hash_bound": True,
            "target_image_crop_fallback_is_forbidden": True,
            "source_quotas_are_exact": True,
            "selected_images_are_unique": True,
            "selected_unordered_annotation_edges_are_unique": True,
            "selected_annotation_endpoints_are_unique": True,
            "selection_is_deterministic_and_model_score_free": True,
            "output_row_count_matches_preregistered_quotas": len(selected)
            == sum(quota for _name, quota in manifest_quotas),
        },
    }
    receipt["canonical_payload_sha256"] = _sha256_bytes(
        _canonical_bytes(receipt)
    )
    return BuildPlan(
        manifest_bytes=manifest_payload,
        support_tsv_bytes=mini_support_payload,
        receipt=receipt,
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
        raise Overfit64BuildError(
            f"refusing to replace existing output root: {output_root}"
        )
    plan = make_plan(**kwargs)
    output_manifest = str(kwargs.get("output_manifest", OUTPUT_MANIFEST))
    output_support_tsv = str(
        kwargs.get("output_support_tsv", OUTPUT_SUPPORT_TSV)
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.tmp-", dir=str(output_root.parent)
        )
    )
    committed = False
    try:
        manifest_path = temporary_root / output_manifest
        with manifest_path.open("xb") as handle:
            handle.write(plan.manifest_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        support_tsv_path = temporary_root / output_support_tsv
        with support_tsv_path.open("xb") as handle:
            handle.write(plan.support_tsv_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        receipt_path = temporary_root / "receipt.json"
        with receipt_path.open("xb") as handle:
            handle.write(_receipt_bytes(plan.receipt))
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(temporary_root)
        if output_root.exists():
            raise Overfit64BuildError(
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
        raise Overfit64BuildError(f"output root is not a directory: {output_root}")
    plan = make_plan(**kwargs)
    output_manifest = str(kwargs.get("output_manifest", OUTPUT_MANIFEST))
    output_support_tsv = str(
        kwargs.get("output_support_tsv", OUTPUT_SUPPORT_TSV)
    )
    manifest_path = output_root / output_manifest
    support_tsv_path = output_root / output_support_tsv
    receipt_path = output_root / "receipt.json"
    try:
        observed_manifest = manifest_path.read_bytes()
        observed_support_tsv = support_tsv_path.read_bytes()
        observed_receipt = receipt_path.read_bytes()
    except OSError as error:
        raise Overfit64BuildError(f"could not read output artifact: {error}") from error
    if observed_manifest != plan.manifest_bytes:
        raise Overfit64BuildError("Overfit64 output manifest does not replay exactly")
    if observed_support_tsv != plan.support_tsv_bytes:
        raise Overfit64BuildError("Overfit64 mini support TSV does not replay exactly")
    expected_receipt = _receipt_bytes(plan.receipt)
    if observed_receipt != expected_receipt:
        raise Overfit64BuildError("Overfit64 receipt does not replay exactly")
    return plan.receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--input-receipt", type=Path, default=INPUT_RECEIPT)
    parser.add_argument("--heldout-root", type=Path, default=HELDOUT_ROOT)
    parser.add_argument("--support-tsv", type=Path, default=SUPPORT_TSV)
    parser.add_argument(
        "--support-bank-cache", type=Path, default=SUPPORT_BANK_CACHE
    )
    parser.add_argument("--support-image-root", type=Path, default=SUPPORT_IMAGE_ROOT)
    parser.add_argument("--canonical-classes", type=Path, default=CANONICAL_CLASSES)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="replay all inputs and print the exact planned receipt without writing",
    )
    mode.add_argument(
        "--verify",
        action="store_true",
        help="rebuild the selection in memory and verify an existing output root",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kwargs = {
        "input_root": args.input_root,
        "input_receipt": args.input_receipt,
        "heldout_root": args.heldout_root,
        "support_tsv": args.support_tsv,
        "support_bank_cache": args.support_bank_cache,
        "support_image_root": args.support_image_root,
        "canonical_classes": args.canonical_classes,
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
