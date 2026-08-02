#!/usr/bin/env python3
"""Expand the sealed O64 assignment pairs into directed rank-only ODVG rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = (
    REPO_ROOT
    / "data/ablations/stageb_data_driven_assignment_overfit64_20260722"
)
SOURCE_MANIFEST = SOURCE_ROOT / "overfit64.jsonl"
SOURCE_RECEIPT = SOURCE_ROOT / "receipt.json"
OUTPUT_ROOT = (
    REPO_ROOT
    / "data/ablations/stageb_gdino_adapter_o64_direct_rank_20260723"
)
OUTPUT_MANIFEST = "o64_direct_rank_vg.jsonl"

SOURCE_RECEIPT_SCHEMA = (
    "pivot.stageb.data_driven.assignment_overfit64_receipt/v1"
)
SOURCE_ROW_SCHEMA = "pivot.stageb.data_driven.official_assignment_pair/v1"
OUTPUT_RECEIPT_SCHEMA = "pivot.stageb.gdino_adapter.o64_direct_rank_receipt/v1"
OUTPUT_ROW_SCHEMA = "pivot.stageb.gdino_adapter.o64_direct_rank_row/v1"
EXPECTED_SOURCE_PAIRS = 64
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "c9a763428bfdeff14e910978ca4fb423bec84de5e8bf5cc2a9664ea2959a529b"
)
EXPECTED_SOURCE_RECEIPT_SHA256 = (
    "359924240b43eea3052ae5e18d4afd014a5d0b7e094deac117d2a9d826d57521"
)
EXPECTED_SOURCE_INVARIANTS = frozenset(
    {
        "all_eight_heldout_manifests_match_the_official_contract",
        "all_source_manifests_match_preregistered_sha256",
        "output_row_count_matches_preregistered_quotas",
        "runtime_mini_support_has_one_candidate_per_selected_class",
        "selected_annotation_endpoints_are_unique",
        "selected_images_are_disjoint_from_all_eight_heldout_splits",
        "selected_images_are_unique",
        "selected_primary_classes_have_external_clean_support",
        "selected_rows_are_valid_official_assignment_pairs",
        "selected_rows_retain_the_complete_upstream_row_bytes",
        "selected_support_witness_images_are_content_hash_bound",
        "selected_unordered_annotation_edges_are_unique",
        "selection_is_deterministic_and_model_score_free",
        "source_quotas_are_exact",
        "target_image_crop_fallback_is_forbidden",
        "upstream_assignment_receipt_matches_preregistered_sha256",
        "upstream_category_complete_receipt_matches_preregistered_sha256",
    }
)
FORBIDDEN_SOURCE_KEYS = frozenset(
    {
        "teacher_scores",
        "teacher_logits",
        "model_scores",
        "model_logits",
        "checkpoint_outputs",
    }
)
DIRECTIONS = ("anchor", "partner")
STREAM_ENCODING = "utf8_text_record_plus_lf_v1"
_HEX64_RE = re.compile(r"[0-9a-f]{64}")


class O64DirectRankBuildError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class BuildPlan:
    manifest_bytes: bytes
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


def _validate_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _HEX64_RE.fullmatch(value) is None:
        raise O64DirectRankBuildError(f"{label} is not a lowercase SHA-256")
    return value


def _file_record(path: Path) -> dict[str, Any]:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise O64DirectRankBuildError(f"required file is missing: {path}") from error
    if not resolved.is_file():
        raise O64DirectRankBuildError(f"not a file: {resolved}")
    before = resolved.stat()
    digest = _sha256_file(resolved)
    after = resolved.stat()
    if (before.st_size, before.st_mtime_ns) != (
        after.st_size,
        after.st_mtime_ns,
    ):
        raise O64DirectRankBuildError(f"file changed while hashing: {resolved}")
    return {
        "path": str(resolved),
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
        raise O64DirectRankBuildError(
            f"could not load {label}: {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise O64DirectRankBuildError(f"{label} must be a JSON object: {path}")
    return value


def _required_int(value: Any, *, field: str, context: str) -> int:
    if type(value) is not int:
        raise O64DirectRankBuildError(f"{context}: {field} must be an exact integer")
    return int(value)


def _required_text(value: Any, *, field: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise O64DirectRankBuildError(f"{context}: {field} must be non-empty text")
    return value.strip()


def _xywh(value: Any, *, field: str, context: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise O64DirectRankBuildError(f"{context}: {field} must be one xywh box")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise O64DirectRankBuildError(
                f"{context}: {field} must contain only finite numbers"
            )
        number = float(item)
        if not math.isfinite(number):
            raise O64DirectRankBuildError(
                f"{context}: {field} must contain only finite numbers"
            )
        result.append(number)
    x, y, width, height = result
    if x < 0.0 or y < 0.0 or width <= 0.0 or height <= 0.0:
        raise O64DirectRankBuildError(
            f"{context}: {field} must have non-negative origin and positive size"
        )
    return x, y, width, height


def _xywh_to_xyxy(box: Sequence[float]) -> list[float]:
    x, y, width, height = box
    return [x, y, x + width, y + height]


def _record_stream_sha256(records: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _canonical_payload_sha256(value: Mapping[str, Any]) -> str:
    payload = {
        str(key): item
        for key, item in value.items()
        if str(key) != "canonical_payload_sha256"
    }
    return _sha256_bytes(_canonical_bytes(payload))


def _contains_forbidden_key(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            if key_text in FORBIDDEN_SOURCE_KEYS:
                return key_text
            nested = _contains_forbidden_key(item)
            if nested is not None:
                return nested
    elif isinstance(value, (list, tuple)):
        for item in value:
            nested = _contains_forbidden_key(item)
            if nested is not None:
                return nested
    return None


def _validate_source_receipt(
    *,
    source_manifest: Path,
    source_receipt: Path,
    manifest_record: Mapping[str, Any],
    expected_receipt_sha256: str,
    expected_pairs: int,
    expected_invariants: frozenset[str],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    receipt_record = _file_record(source_receipt)
    if receipt_record["sha256"] != expected_receipt_sha256:
        raise O64DirectRankBuildError(
            "source receipt SHA-256 mismatch: "
            f"expected={expected_receipt_sha256}, observed={receipt_record['sha256']}"
        )
    receipt = _load_json_object(source_receipt, label="O64 source receipt")
    declared_payload_sha = _validate_sha256(
        receipt.get("canonical_payload_sha256"),
        label="source receipt canonical payload hash",
    )
    if _canonical_payload_sha256(receipt) != declared_payload_sha:
        raise O64DirectRankBuildError("source receipt canonical payload hash drifted")

    invariants = receipt.get("invariants")
    if (
        receipt.get("schema") != SOURCE_RECEIPT_SCHEMA
        or receipt.get("row_schema") != SOURCE_ROW_SCHEMA
        or receipt.get("output_manifest") != source_manifest.name
        or receipt.get("rows") != expected_pairs
        or receipt.get("valid_rows") != expected_pairs
        or receipt.get("invalid_rows") != 0
        or receipt.get("unique_images") != expected_pairs
        or receipt.get("unique_unordered_annotation_edges") != expected_pairs
        or receipt.get("unique_annotation_endpoints") != 2 * expected_pairs
        or not isinstance(invariants, dict)
        or set(invariants) != set(expected_invariants)
        or any(value is not True for value in invariants.values())
    ):
        raise O64DirectRankBuildError("source receipt counts or invariants drifted")

    output = receipt.get("output")
    if not isinstance(output, dict):
        raise O64DirectRankBuildError("source receipt output binding is absent")
    try:
        receipt_output_path = Path(output.get("path", "")).expanduser().resolve()
    except (OSError, TypeError) as error:
        raise O64DirectRankBuildError("source receipt output path is invalid") from error
    if (
        receipt_output_path != source_manifest
        or output.get("sha256") != manifest_record["sha256"]
        or output.get("size_bytes") != manifest_record["size_bytes"]
    ):
        raise O64DirectRankBuildError("source receipt output binding drifted")

    selection = receipt.get("selection_contract")
    forbidden_inputs = (
        selection.get("forbidden_inputs") if isinstance(selection, dict) else None
    )
    if (
        not isinstance(selection, dict)
        or selection.get("model_score_free") is not True
        or not isinstance(forbidden_inputs, list)
        or len(forbidden_inputs) != len(FORBIDDEN_SOURCE_KEYS)
        or set(forbidden_inputs) != set(FORBIDDEN_SOURCE_KEYS)
    ):
        raise O64DirectRankBuildError("source receipt model-score-free contract drifted")

    members = receipt.get("members")
    if not isinstance(members, list) or len(members) != expected_pairs:
        raise O64DirectRankBuildError("source receipt member list drifted")
    normalized_members: list[dict[str, Any]] = []
    pair_ids = []
    image_ids = []
    edges = []
    endpoints = []
    for pair_index, member in enumerate(members):
        context = f"source receipt member {pair_index}"
        if not isinstance(member, dict):
            raise O64DirectRankBuildError(f"{context} must be an object")
        output_index = _required_int(
            member.get("output_index"), field="output_index", context=context
        )
        image_id = _required_int(
            member.get("image_id"), field="image_id", context=context
        )
        anchor_id = _required_int(
            member.get("anchor_coco_ann_id"),
            field="anchor_coco_ann_id",
            context=context,
        )
        partner_id = _required_int(
            member.get("partner_coco_ann_id"),
            field="partner_coco_ann_id",
            context=context,
        )
        pair_id = _validate_sha256(member.get("pair_id"), label=f"{context} pair_id")
        priority = _validate_sha256(
            member.get("priority_sha256"), label=f"{context} priority_sha256"
        )
        row_sha = _validate_sha256(
            member.get("source_row_sha256"), label=f"{context} source_row_sha256"
        )
        if output_index != pair_index or anchor_id == partner_id:
            raise O64DirectRankBuildError(f"{context} identity drifted")
        normalized = dict(member)
        normalized.update(
            {
                "pair_id": pair_id,
                "priority_sha256": priority,
                "source_row_sha256": row_sha,
                "image_id": image_id,
                "anchor_coco_ann_id": anchor_id,
                "partner_coco_ann_id": partner_id,
            }
        )
        normalized_members.append(normalized)
        pair_ids.append(pair_id)
        image_ids.append(str(image_id))
        edge = tuple(sorted((anchor_id, partner_id)))
        edges.append(f"{edge[0]}\t{edge[1]}")
        endpoints.append(f"{anchor_id}\t{partner_id}")

    expected_streams = {
        "ordered_member_pair_id_stream_sha256": _record_stream_sha256(pair_ids),
        "ordered_image_id_stream_sha256": _record_stream_sha256(image_ids),
        "ordered_unordered_edge_stream_sha256": _record_stream_sha256(edges),
        "ordered_endpoint_stream_sha256": _record_stream_sha256(endpoints),
    }
    if receipt.get("ordered_member_stream_encoding") != STREAM_ENCODING or any(
        receipt.get(key) != value for key, value in expected_streams.items()
    ):
        raise O64DirectRankBuildError("source receipt ordered member streams drifted")
    if (
        len(set(pair_ids)) != expected_pairs
        or len(set(image_ids)) != expected_pairs
        or len(set(edges)) != expected_pairs
        or len({item for pair in endpoints for item in pair.split("\t")})
        != 2 * expected_pairs
    ):
        raise O64DirectRankBuildError("source receipt member uniqueness drifted")
    return receipt, receipt_record, normalized_members


def _endpoint_payload(
    *,
    endpoint: Mapping[str, Any],
    direction: str,
    instances: Sequence[Any],
    row_source: str,
    image_id: int,
    expected_class_id: int,
    context: str,
) -> tuple[int, str, list[float]]:
    endpoint_context = f"{context} {direction}"
    endpoint_image = _required_int(
        endpoint.get("image_id"), field="image_id", context=endpoint_context
    )
    ann_id = _required_int(
        endpoint.get("coco_ann_id"), field="coco_ann_id", context=endpoint_context
    )
    expression = _required_text(
        endpoint.get("expression"), field="expression", context=endpoint_context
    )
    endpoint_source = _required_text(
        endpoint.get("source"), field="source", context=endpoint_context
    )
    endpoint_box = _xywh(
        endpoint.get("bbox"), field="bbox", context=endpoint_context
    )
    if endpoint_image != image_id or endpoint_source != row_source:
        raise O64DirectRankBuildError(f"{endpoint_context}: endpoint source drifted")

    matches = []
    for instance in instances:
        if not isinstance(instance, dict):
            raise O64DirectRankBuildError(f"{context}: instance must be an object")
        value = instance.get("coco_ann_id")
        if type(value) is int and value == ann_id:
            matches.append(instance)
    if len(matches) != 1:
        raise O64DirectRankBuildError(
            f"{endpoint_context}: endpoint must match exactly one instance"
        )
    instance_box = _xywh(
        matches[0].get("bbox"), field="instance bbox", context=endpoint_context
    )
    if instance_box != endpoint_box:
        raise O64DirectRankBuildError(
            f"{endpoint_context}: endpoint and instance boxes differ"
        )
    instance_class_id = _required_int(
        matches[0].get("class_id"), field="instance class_id", context=endpoint_context
    )
    if instance_class_id != expected_class_id:
        raise O64DirectRankBuildError(
            f"{endpoint_context}: endpoint instance class drifted"
        )
    return ann_id, expression, _xywh_to_xyxy(endpoint_box)


def _expand_source_rows(
    *,
    source_manifest: Path,
    manifest_record: Mapping[str, Any],
    receipt_record: Mapping[str, Any],
    members: Sequence[Mapping[str, Any]],
    expected_pairs: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    output_members: list[dict[str, Any]] = []
    source_images: set[int] = set()
    source_edges: set[tuple[int, int]] = set()
    source_endpoints: set[int] = set()
    try:
        handle = source_manifest.open("rb")
    except OSError as error:
        raise O64DirectRankBuildError(
            f"could not open source manifest: {source_manifest}"
        ) from error
    with handle:
        for pair_index, raw in enumerate(handle):
            context = f"{source_manifest}:{pair_index + 1}"
            if pair_index >= expected_pairs:
                raise O64DirectRankBuildError(
                    f"source manifest has more than {expected_pairs} rows"
                )
            if not raw.endswith(b"\n") or raw.endswith(b"\r\n"):
                raise O64DirectRankBuildError(
                    f"{context}: source row must end in one LF"
                )
            stripped = raw[:-1]
            if not stripped:
                raise O64DirectRankBuildError(f"{context}: source row is blank")
            try:
                row = json.loads(stripped)
            except (UnicodeError, json.JSONDecodeError) as error:
                raise O64DirectRankBuildError(
                    f"{context}: invalid source JSON: {error}"
                ) from error
            if not isinstance(row, dict):
                raise O64DirectRankBuildError(f"{context}: source row must be an object")
            forbidden = _contains_forbidden_key(row)
            if forbidden is not None:
                raise O64DirectRankBuildError(
                    f"{context}: forbidden model-derived field {forbidden!r}"
                )
            member = members[pair_index]
            source_row_sha = _sha256_bytes(stripped)
            if source_row_sha != member["source_row_sha256"]:
                raise O64DirectRankBuildError(
                    f"{context}: source row SHA-256 does not match receipt member"
                )
            if (
                row.get("stage_b_data_driven_assignment_pair") is not True
                or row.get("stage_b_data_driven_assignment_pair_schema")
                != SOURCE_ROW_SCHEMA
                or row.get("assignment_pair_valid") is not True
                or row.get("assignment_pair_invalid_reason") is not None
                or row.get("stage_b_u2_category_complete") is not True
                or row.get("stage_b_u2_category_complete_schema")
                != "pivot.stageb.u2_category_complete_ref/v1"
            ):
                raise O64DirectRankBuildError(f"{context}: source row contract drifted")

            pair = row.get("assignment_pair")
            anchor = pair.get("anchor") if isinstance(pair, dict) else None
            partner = pair.get("partner") if isinstance(pair, dict) else None
            instances = row.get("instances")
            if (
                not isinstance(pair, dict)
                or pair.get("schema") != SOURCE_ROW_SCHEMA
                or not isinstance(anchor, dict)
                or not isinstance(partner, dict)
                or not isinstance(instances, list)
                or len(instances) < 2
                or row.get("primary_support_instance_index") != 0
            ):
                raise O64DirectRankBuildError(f"{context}: assignment pair is malformed")
            image_id = _required_int(row.get("image_id"), field="image_id", context=context)
            anchor_id = _required_int(
                anchor.get("coco_ann_id"), field="anchor coco_ann_id", context=context
            )
            partner_id = _required_int(
                partner.get("coco_ann_id"), field="partner coco_ann_id", context=context
            )
            row_source = _required_text(row.get("source"), field="source", context=context)
            filename = _required_text(
                row.get("filename"), field="filename", context=context
            )
            ref_id = _required_int(row.get("ref_id"), field="ref_id", context=context)
            sent_id = _required_int(row.get("sent_id"), field="sent_id", context=context)
            ann_id = _required_int(row.get("ann_id"), field="ann_id", context=context)
            if (
                ann_id != anchor_id
                or anchor_id == partner_id
                or member.get("image_id") != image_id
                or member.get("anchor_coco_ann_id") != anchor_id
                or member.get("partner_coco_ann_id") != partner_id
                or member.get("source") != row_source
                or member.get("ref_id") != ref_id
                or member.get("sent_id") != sent_id
            ):
                raise O64DirectRankBuildError(f"{context}: row/member identity drifted")
            if (
                not isinstance(instances[0], dict)
                or instances[0].get("category_complete_primary") is not True
                or instances[0].get("coco_ann_id") != anchor_id
            ):
                raise O64DirectRankBuildError(f"{context}: primary instance drifted")
            class_id = _required_int(
                instances[0].get("class_id"),
                field="primary instance class_id",
                context=context,
            )
            if member.get("class_id") != class_id:
                raise O64DirectRankBuildError(f"{context}: member class_id drifted")

            endpoints = {"anchor": anchor, "partner": partner}
            pair_output_indices = []
            for direction in DIRECTIONS:
                target_id, expression, bbox_xyxy = _endpoint_payload(
                    endpoint=endpoints[direction],
                    direction=direction,
                    instances=instances,
                    row_source=row_source,
                    image_id=image_id,
                    expected_class_id=class_id,
                    context=context,
                )
                if target_id != (anchor_id if direction == "anchor" else partner_id):
                    raise O64DirectRankBuildError(
                        f"{context}: {direction} target identity drifted"
                    )
                output_index = len(rows)
                output_row = {
                    "direction": direction,
                    "filename": filename,
                    "grounding": {
                        "regions": [{"bbox": bbox_xyxy, "phrase": expression}]
                    },
                    "image_id": image_id,
                    "pair_index": pair_index,
                    "row_schema": OUTPUT_ROW_SCHEMA,
                    "source_manifest_sha256": manifest_record["sha256"],
                    "source_member_pair_id": member["pair_id"],
                    "source_o64_line_number": pair_index + 1,
                    "source_receipt_sha256": receipt_record["sha256"],
                    "source_row_sha256": source_row_sha,
                    "target_coco_ann_id": target_id,
                }
                rows.append(output_row)
                output_members.append(
                    {
                        "direction": direction,
                        "image_id": image_id,
                        "output_index": output_index,
                        "output_row_sha256": _sha256_bytes(
                            _canonical_bytes(output_row)
                        ),
                        "pair_index": pair_index,
                        "source_member_pair_id": member["pair_id"],
                        "source_row_sha256": source_row_sha,
                        "target_coco_ann_id": target_id,
                    }
                )
                pair_output_indices.append(output_index)
            if pair_output_indices != [2 * pair_index, 2 * pair_index + 1]:
                raise O64DirectRankBuildError(f"{context}: direction ordering drifted")
            source_images.add(image_id)
            source_edges.add(tuple(sorted((anchor_id, partner_id))))
            source_endpoints.update((anchor_id, partner_id))

    if len(rows) != 2 * expected_pairs:
        raise O64DirectRankBuildError(
            f"source manifest row count drifted: {len(rows) // 2} != {expected_pairs}"
        )
    if (
        len(source_images) != expected_pairs
        or len(source_edges) != expected_pairs
        or len(source_endpoints) != 2 * expected_pairs
    ):
        raise O64DirectRankBuildError("source row uniqueness contract drifted")
    if Counter(row["direction"] for row in rows) != {
        "anchor": expected_pairs,
        "partner": expected_pairs,
    }:
        raise O64DirectRankBuildError("directed expansion count drifted")
    return rows, output_members


def make_plan(
    *,
    source_manifest: Path = SOURCE_MANIFEST,
    source_receipt: Path = SOURCE_RECEIPT,
    output_root: Path = OUTPUT_ROOT,
    output_manifest: str = OUTPUT_MANIFEST,
    expected_source_manifest_sha256: str = EXPECTED_SOURCE_MANIFEST_SHA256,
    expected_source_receipt_sha256: str = EXPECTED_SOURCE_RECEIPT_SHA256,
    expected_pairs: int = EXPECTED_SOURCE_PAIRS,
    expected_invariants: frozenset[str] = EXPECTED_SOURCE_INVARIANTS,
) -> BuildPlan:
    _validate_sha256(
        expected_source_manifest_sha256, label="expected source manifest hash"
    )
    _validate_sha256(
        expected_source_receipt_sha256, label="expected source receipt hash"
    )
    if type(expected_pairs) is not int or expected_pairs <= 0:
        raise O64DirectRankBuildError("expected_pairs must be a positive exact integer")
    if not isinstance(expected_invariants, frozenset) or not expected_invariants:
        raise O64DirectRankBuildError("expected_invariants must be a non-empty frozenset")
    if Path(output_manifest).name != output_manifest or not output_manifest.endswith(
        ".jsonl"
    ):
        raise O64DirectRankBuildError("output manifest must be a JSONL basename")
    source_manifest = source_manifest.expanduser().resolve(strict=True)
    source_receipt = source_receipt.expanduser().resolve(strict=True)
    output_root = output_root.expanduser().resolve(strict=False)
    manifest_record = _file_record(source_manifest)
    if manifest_record["sha256"] != expected_source_manifest_sha256:
        raise O64DirectRankBuildError(
            "source manifest SHA-256 mismatch: "
            f"expected={expected_source_manifest_sha256}, "
            f"observed={manifest_record['sha256']}"
        )
    receipt, receipt_record, members = _validate_source_receipt(
        source_manifest=source_manifest,
        source_receipt=source_receipt,
        manifest_record=manifest_record,
        expected_receipt_sha256=expected_source_receipt_sha256,
        expected_pairs=expected_pairs,
        expected_invariants=expected_invariants,
    )
    rows, output_members = _expand_source_rows(
        source_manifest=source_manifest,
        manifest_record=manifest_record,
        receipt_record=receipt_record,
        members=members,
        expected_pairs=expected_pairs,
    )
    manifest_payload = b"".join(_canonical_bytes(row) + b"\n" for row in rows)
    output_path = output_root / output_manifest
    receipt_payload: dict[str, Any] = {
        "schema": OUTPUT_RECEIPT_SCHEMA,
        "row_schema": OUTPUT_ROW_SCHEMA,
        "builder": _file_record(Path(__file__)),
        "source": {
            "manifest": manifest_record,
            "receipt": receipt_record,
            "receipt_canonical_payload_sha256": receipt[
                "canonical_payload_sha256"
            ],
            "receipt_schema": receipt["schema"],
            "row_schema": receipt["row_schema"],
        },
        "conversion_contract": {
            "directions_in_pair_order": list(DIRECTIONS),
            "input_box_format": "xywh",
            "output_box_format": "xyxy",
            "output_dataset_mode": "odvg",
            "regions_per_row": 1,
            "expressions_per_row": 1,
            "teacher_or_model_scores_consumed": False,
        },
        "pairs": expected_pairs,
        "rows": len(rows),
        "direction_counts": dict(sorted(Counter(row["direction"] for row in rows).items())),
        "unique_images": len({row["image_id"] for row in rows}),
        "unique_target_annotation_ids": len(
            {row["target_coco_ann_id"] for row in rows}
        ),
        "members": output_members,
        "ordered_output_row_stream_encoding": STREAM_ENCODING,
        "ordered_output_row_sha256_stream_sha256": _record_stream_sha256(
            [member["output_row_sha256"] for member in output_members]
        ),
        "output_manifest": output_manifest,
        "output": _predicted_file_record(output_path, manifest_payload),
        "invariants": {
            "source_manifest_matches_preregistered_sha256": True,
            "source_receipt_matches_preregistered_sha256": True,
            "source_receipt_canonical_payload_recomputed": True,
            "source_receipt_invariants_are_exact_and_true": True,
            "source_rows_match_receipt_member_hashes_and_order": True,
            "source_pairs_are_model_score_free": True,
            "each_pair_emits_anchor_then_partner": True,
            "each_output_row_has_one_expression_and_one_box": True,
            "all_output_boxes_are_xyxy": True,
            "all_source_pairs_emit_exactly_two_directions": True,
            "source_pair_image_edge_and_endpoint_uniqueness_is_preserved": True,
        },
    }
    receipt_payload["canonical_payload_sha256"] = _canonical_payload_sha256(
        receipt_payload
    )
    return BuildPlan(manifest_bytes=manifest_payload, receipt=receipt_payload)


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
        raise O64DirectRankBuildError(
            f"refusing to replace existing output root: {output_root}"
        )
    plan = make_plan(**kwargs)
    output_manifest = str(kwargs.get("output_manifest", OUTPUT_MANIFEST))
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
        receipt_path = temporary_root / "receipt.json"
        with receipt_path.open("xb") as handle:
            handle.write(_receipt_bytes(plan.receipt))
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(temporary_root)
        if output_root.exists():
            raise O64DirectRankBuildError(
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
        raise O64DirectRankBuildError(f"output root is not a directory: {output_root}")
    plan = make_plan(**kwargs)
    output_manifest = str(kwargs.get("output_manifest", OUTPUT_MANIFEST))
    try:
        observed_manifest = (output_root / output_manifest).read_bytes()
        observed_receipt = (output_root / "receipt.json").read_bytes()
    except OSError as error:
        raise O64DirectRankBuildError(f"could not read output artifact: {error}") from error
    if observed_manifest != plan.manifest_bytes:
        raise O64DirectRankBuildError("direct-rank manifest does not replay exactly")
    if observed_receipt != _receipt_bytes(plan.receipt):
        raise O64DirectRankBuildError("direct-rank receipt does not replay exactly")
    return plan.receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--source-receipt", type=Path, default=SOURCE_RECEIPT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the planned receipt without writing",
    )
    mode.add_argument(
        "--verify",
        action="store_true",
        help="replay and byte-verify an existing output directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kwargs = {
        "source_manifest": args.source_manifest,
        "source_receipt": args.source_receipt,
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
