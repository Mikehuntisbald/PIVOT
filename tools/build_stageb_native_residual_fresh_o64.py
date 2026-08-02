#!/usr/bin/env python3
"""Build a fresh, exposure-disjoint 64-pair native-residual ODVG set."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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

from tools import build_stageb_data_driven_assignment_overfit64 as selection  # noqa: E402
from tools.stageb_ref_split_contract import (
    REF_SPLIT_CONTRACT,
    REF_SPLIT_MANIFEST_FILES,
    REF_SPLITS,
)  # noqa: E402


SELECTION_LIBRARY = REPO_ROOT / "tools/build_stageb_data_driven_assignment_overfit64.py"
INPUT_ROOT = REPO_ROOT / "data/ablations/stageb_data_driven_assignment_pairs_20260722"
INPUT_RECEIPT = INPUT_ROOT / "receipt.json"
HELDOUT_ROOT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/baseline_b58_ref8_seed42/refcoco_eval_inputs"
)
OLD_O64_RECEIPT = (
    REPO_ROOT
    / "data/ablations/stageb_data_driven_assignment_overfit64_20260722/receipt.json"
)
NEW_HEAD_RECEIPT = (
    REPO_ROOT
    / "data/ablations/stageb_data_driven_new_head_partition_20260723/receipt.json"
)
SUPPORT_TSV = Path(
    "/media/haoyi/T9/data/patches_quality_emb/emb_index_from_quality.tsv"
)
SUPPORT_BANK_CACHE = Path(str(SUPPORT_TSV) + ".bank.clean.img.pkl")
SUPPORT_IMAGE_ROOT = Path("/media/haoyi/T9/data/patches_quality")
CANONICAL_CLASSES = Path("/media/haoyi/T9/data/canonical_classes_with_aliases.json")
OUTPUT_ROOT = (
    REPO_ROOT / "data/ablations/stageb_native_residual_fresh_o64_20260724"
)
OUTPUT_MANIFEST = "fresh_o64_directed_vg.jsonl"

RECEIPT_SCHEMA = "pivot.stageb.native_residual.fresh_o64_receipt/v1"
ROW_SCHEMA = "pivot.stageb.native_residual.fresh_o64_directed_row/v1"
OUTPUT_RECEIPT_SCHEMA = RECEIPT_SCHEMA
OUTPUT_ROW_SCHEMA = ROW_SCHEMA
ASSIGNMENT_ROW_SCHEMA = "pivot.stageb.data_driven.official_assignment_pair/v1"
OLD_O64_RECEIPT_SCHEMA = (
    "pivot.stageb.data_driven.assignment_overfit64_receipt/v1"
)
NEW_HEAD_RECEIPT_SCHEMA = "pivot.stageb.data_driven.new_head_partition_receipt/v1"
STREAM_ENCODING = "utf8_text_record_plus_lf_v1"
DIRECTIONS = ("anchor", "partner")

MANIFEST_QUOTAS = selection.MANIFEST_QUOTAS
EXPECTED_INPUT_RECEIPT_SHA256 = selection.EXPECTED_INPUT_RECEIPT_SHA256
EXPECTED_INPUT_SHA256 = selection.EXPECTED_INPUT_SHA256
EXPECTED_INPUT_ROWS = selection.EXPECTED_INPUT_ROWS
EXPECTED_CATEGORY_COMPLETE_RECEIPT_SHA256 = (
    selection.EXPECTED_CATEGORY_COMPLETE_RECEIPT_SHA256
)
EXPECTED_HELDOUT_UNION_IMAGES = selection.EXPECTED_HELDOUT_UNION_IMAGES
EXPECTED_SUPPORT_SHA256 = selection.EXPECTED_SUPPORT_SHA256
EXPECTED_SELECTION_LIBRARY_SHA256 = (
    "c40547a41f6bcd2f0d4198fb6898f58a3198ee99d8b8aa0c98013843513dadae"
)
EXPECTED_OLD_O64_RECEIPT_SHA256 = (
    "359924240b43eea3052ae5e18d4afd014a5d0b7e094deac117d2a9d826d57521"
)
EXPECTED_NEW_HEAD_RECEIPT_SHA256 = (
    "56de31d883ed137f3f9332c34de846839d82c0724120f42a49c5c1c302f38506"
)
EXPECTED_OLD_O64_IMAGES = 64
EXPECTED_NEW_HEAD_DEV_FULL_IMAGES = 2048
EXPECTED_NEW_HEAD_DEV_SCREEN_IMAGES = 512
EXPECTED_SELECTED_STREAMS = {
    "ordered_pair_id_stream_sha256": (
        "0084c0330dcb034fd72b9fdaa6f7f04303595f0d6fdcce19d835a58ecaaebac4"
    ),
    "ordered_image_id_stream_sha256": (
        "d43c69435188e8f9802e794ab4b0bebdb220e3971ff33a62aa03566177ed83c3"
    ),
    "sorted_image_id_json_sha256": (
        "904ff9f83b55ea3ba1084eb8e2cb355c99e6436e02f415a18e086e99bcafd506"
    ),
    "ordered_unordered_edge_stream_sha256": (
        "2a01f6824a6b07752ea41d01c85227a2e9cef3fdbc0fdf423bb2794fed524d62"
    ),
    "ordered_endpoint_stream_sha256": (
        "6f71bd3a6339e8a156e3084e04e81af49dfd6ff614b55a4849296ca6e3e91997"
    ),
}

FORBIDDEN_SOURCE_KEYS = frozenset(
    {
        "teacher_score",
        "teacher_scores",
        "teacher_logit",
        "teacher_logits",
        "teacher_probability",
        "teacher_probabilities",
        "model_score",
        "model_scores",
        "model_logit",
        "model_logits",
        "checkpoint_score",
        "checkpoint_scores",
        "checkpoint_output",
        "checkpoint_outputs",
    }
)
LEGACY_FORBIDDEN_INPUTS = frozenset(
    {
        "teacher_scores",
        "teacher_logits",
        "model_scores",
        "model_logits",
        "checkpoint_outputs",
    }
)


class FreshO64BuildError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class BuildPlan:
    manifest_bytes: bytes
    receipt: dict[str, Any]


_canonical_bytes = selection._canonical_bytes
_sha256_bytes = selection._sha256_bytes
_sha256_file = selection._sha256_file
_file_record = selection._file_record
_predicted_file_record = selection._predicted_file_record
_record_stream_sha256 = selection._record_stream_sha256


def _canonical_payload_sha256(value: Mapping[str, Any]) -> str:
    payload = {str(key): item for key, item in value.items() if key != "canonical_payload_sha256"}
    return _sha256_bytes(_canonical_bytes(payload))


def _validate_sha256(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise FreshO64BuildError(f"{label} is not a lowercase SHA-256")
    return value


def _required_int(value: Any, *, field: str, context: str) -> int:
    if type(value) is not int:
        raise FreshO64BuildError(f"{context}: {field} must be an exact integer")
    return int(value)


def _required_text(value: Any, *, field: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FreshO64BuildError(f"{context}: {field} must be non-empty text")
    return value.strip()


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FreshO64BuildError(f"could not load {label}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise FreshO64BuildError(f"{label} must be a JSON object: {path}")
    return value


def _contains_forbidden_key(value: Any, *, prefix: str = "") -> str | None:
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            path = f"{prefix}.{key}" if prefix else key
            if key.casefold() in FORBIDDEN_SOURCE_KEYS:
                return path
            nested = _contains_forbidden_key(item, prefix=path)
            if nested is not None:
                return nested
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            path = f"{prefix}[{index}]"
            nested = _contains_forbidden_key(item, prefix=path)
            if nested is not None:
                return nested
    return None


def _read_bound_receipt(
    path: Path, *, expected_sha256: str, schema: str, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    _validate_sha256(expected_sha256, label=f"expected {label} hash")
    path = path.expanduser().resolve(strict=True)
    record = _file_record(path)
    if record["sha256"] != expected_sha256:
        raise FreshO64BuildError(
            f"{label} SHA-256 mismatch: expected={expected_sha256}, "
            f"observed={record['sha256']}"
        )
    receipt = _load_json_object(path, label=label)
    if receipt.get("schema") != schema:
        raise FreshO64BuildError(f"{label} schema drifted")
    declared = _validate_sha256(
        receipt.get("canonical_payload_sha256"),
        label=f"{label} canonical payload hash",
    )
    if _canonical_payload_sha256(receipt) != declared:
        raise FreshO64BuildError(f"{label} canonical payload hash drifted")
    invariants = receipt.get("invariants")
    if not isinstance(invariants, dict) or not invariants or any(
        value is not True for value in invariants.values()
    ):
        raise FreshO64BuildError(f"{label} invariants drifted")
    return receipt, record


def _load_old_o64_blacklist(
    *, receipt_path: Path, expected_sha256: str, expected_images: int
) -> tuple[set[int], dict[str, Any]]:
    receipt, record = _read_bound_receipt(
        receipt_path,
        expected_sha256=expected_sha256,
        schema=OLD_O64_RECEIPT_SCHEMA,
        label="old O64 receipt",
    )
    members = receipt.get("members")
    if (
        receipt.get("row_schema") != ASSIGNMENT_ROW_SCHEMA
        or receipt.get("rows") != expected_images
        or receipt.get("valid_rows") != expected_images
        or receipt.get("invalid_rows") != 0
        or receipt.get("unique_images") != expected_images
        or receipt.get("unique_unordered_annotation_edges") != expected_images
        or receipt.get("unique_annotation_endpoints") != 2 * expected_images
        or not isinstance(members, list)
        or len(members) != expected_images
    ):
        raise FreshO64BuildError("old O64 receipt counts drifted")
    contract = receipt.get("selection_contract")
    if (
        not isinstance(contract, dict)
        or contract.get("model_score_free") is not True
        or contract.get("namespace") != selection.SELECTION_NAMESPACE
        or contract.get("policy") != selection.SELECTION_POLICY
        or not LEGACY_FORBIDDEN_INPUTS.issubset(
            set(contract.get("forbidden_inputs") or [])
        )
    ):
        raise FreshO64BuildError("old O64 selection contract drifted")

    image_ids: list[int] = []
    for index, member in enumerate(members):
        context = f"old O64 member {index}"
        if not isinstance(member, dict) or member.get("output_index") != index:
            raise FreshO64BuildError(f"{context}: member order drifted")
        image_ids.append(
            _required_int(member.get("image_id"), field="image_id", context=context)
        )
        _validate_sha256(member.get("pair_id"), label=f"{context} pair_id")
        _validate_sha256(
            member.get("source_row_sha256"), label=f"{context} source row hash"
        )
    if len(set(image_ids)) != expected_images:
        raise FreshO64BuildError("old O64 blacklist images are not unique")
    image_stream = _record_stream_sha256([str(value) for value in image_ids])
    if (
        receipt.get("ordered_member_stream_encoding") != STREAM_ENCODING
        or receipt.get("ordered_image_id_stream_sha256") != image_stream
    ):
        raise FreshO64BuildError("old O64 image stream drifted")
    return set(image_ids), {
        "receipt": record,
        "receipt_canonical_payload_sha256": receipt["canonical_payload_sha256"],
        "images": expected_images,
        "ordered_image_id_stream_encoding": STREAM_ENCODING,
        "ordered_image_id_stream_sha256": image_stream,
        "sorted_image_id_json_sha256": _sha256_bytes(
            _canonical_bytes(sorted(image_ids))
        ),
    }


def _load_new_head_dev_blacklist(
    *,
    receipt_path: Path,
    expected_sha256: str,
    expected_dev_full_images: int,
    expected_dev_screen_images: int,
) -> tuple[set[int], dict[str, Any]]:
    receipt, record = _read_bound_receipt(
        receipt_path,
        expected_sha256=expected_sha256,
        schema=NEW_HEAD_RECEIPT_SCHEMA,
        label="new-head partition receipt",
    )
    contract = receipt.get("selection_contract")
    summary = receipt.get("partition_summary")
    dev_full = receipt.get("dev_full_members")
    dev_screen = receipt.get("dev_screen_members")
    if (
        not isinstance(contract, dict)
        or contract.get("model_score_free") is not True
        or contract.get("dev_full_target_images") != expected_dev_full_images
        or contract.get("dev_screen_target_images") != expected_dev_screen_images
        or contract.get("dev_screen_is_nested_in_dev_full") is not True
        or not LEGACY_FORBIDDEN_INPUTS.issubset(
            set(contract.get("forbidden_inputs") or [])
        )
        or not isinstance(summary, dict)
        or not isinstance(dev_full, list)
        or len(dev_full) != expected_dev_full_images
        or not isinstance(dev_screen, list)
        or len(dev_screen) != expected_dev_screen_images
    ):
        raise FreshO64BuildError("new-head dev selection contract drifted")

    def member_stream(
        members: Sequence[Any], *, label: str
    ) -> tuple[list[int], list[str]]:
        ids: list[int] = []
        keys: list[str] = []
        for index, member in enumerate(members):
            context = f"{label} member {index}"
            if not isinstance(member, dict):
                raise FreshO64BuildError(f"{context} must be an object")
            ids.append(
                _required_int(member.get("image_id"), field="image_id", context=context)
            )
            keys.append(
                _required_text(member.get("image_key"), field="image_key", context=context)
            )
            _validate_sha256(
                member.get("selection_priority_sha256"),
                label=f"{context} selection priority",
            )
        if len(set(ids)) != len(ids) or len(set(keys)) != len(keys):
            raise FreshO64BuildError(f"{label} members are not unique")
        return ids, keys

    full_ids, full_keys = member_stream(dev_full, label="dev_full")
    screen_ids, screen_keys = member_stream(dev_screen, label="dev_screen")
    full_summary = summary.get("dev_full")
    screen_summary = summary.get("dev_screen")
    if (
        not isinstance(full_summary, dict)
        or full_summary.get("unique_image_keys") != expected_dev_full_images
        or full_summary.get("ordered_image_key_stream_sha256")
        != _record_stream_sha256(sorted(full_keys))
        or not isinstance(screen_summary, dict)
        or screen_summary.get("unique_image_keys") != expected_dev_screen_images
        or screen_summary.get("ordered_image_key_stream_sha256")
        != _record_stream_sha256(sorted(screen_keys))
        or not set(screen_ids).issubset(full_ids)
    ):
        raise FreshO64BuildError("new-head dev member streams drifted")
    return set(full_ids), {
        "receipt": record,
        "receipt_canonical_payload_sha256": receipt["canonical_payload_sha256"],
        "dev_full_images": len(full_ids),
        "dev_screen_images": len(screen_ids),
        "dev_screen_is_nested_in_dev_full": True,
        "dev_full_image_key_stream_encoding": STREAM_ENCODING,
        "member_ordered_dev_full_image_key_stream_sha256": _record_stream_sha256(
            full_keys
        ),
        "sorted_dev_full_image_key_stream_sha256": _record_stream_sha256(
            sorted(full_keys)
        ),
        "ordered_dev_full_image_id_stream_sha256": _record_stream_sha256(
            [str(value) for value in full_ids]
        ),
        "sorted_dev_full_image_id_json_sha256": _sha256_bytes(
            _canonical_bytes(sorted(full_ids))
        ),
    }


def _load_selection_library(expected_sha256: str) -> dict[str, Any]:
    _validate_sha256(expected_sha256, label="expected selection library hash")
    record = _file_record(SELECTION_LIBRARY)
    if record["sha256"] != expected_sha256:
        raise FreshO64BuildError(
            "selection library SHA-256 mismatch: "
            f"expected={expected_sha256}, observed={record['sha256']}"
        )
    if (
        selection.SELECTION_NAMESPACE
        != "pivot.stageb.data_driven.assignment_pair_overfit64.selection/v1"
        or selection.SELECTION_POLICY
        != "sha256_pair_id_priority_quota_order_greedy_v1"
    ):
        raise FreshO64BuildError("selection library policy constants drifted")
    return record


def _decode_source_row(raw: bytes, *, context: str) -> tuple[dict[str, Any], bytes]:
    if not raw.endswith(b"\n") or raw.endswith(b"\r\n"):
        raise FreshO64BuildError(f"{context}: source row must end in exactly one LF")
    stripped = raw[:-1]
    if not stripped:
        raise FreshO64BuildError(f"{context}: source row is blank")
    try:
        row = json.loads(stripped)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise FreshO64BuildError(f"{context}: invalid JSON: {error}") from error
    if not isinstance(row, dict):
        raise FreshO64BuildError(f"{context}: source row must be an object")
    forbidden = _contains_forbidden_key(row)
    if forbidden is not None:
        raise FreshO64BuildError(
            f"{context}: forbidden model-derived field {forbidden!r}"
        )
    return row, stripped


def _select_candidates(
    *,
    input_root: Path,
    manifest_quotas: Sequence[tuple[str, int]],
    expected_input_rows: Mapping[str, int],
    heldout_images: set[int],
    old_o64_images: set[int],
    evaluated_dev_images: set[int],
    support_bank: selection.SupportBank,
) -> tuple[list[selection.Candidate], dict[str, Any]]:
    selected: list[selection.Candidate] = []
    used_images: set[int] = set()
    used_edges: set[tuple[int, int]] = set()
    used_endpoints: set[int] = set()
    statistics: dict[str, Any] = {}
    for manifest_name, quota in manifest_quotas:
        path = input_root / manifest_name
        candidates: list[selection.Candidate] = []
        fresh_images: set[int] = set()
        stats = Counter()
        with path.open("rb") as handle:
            for line_number, raw in enumerate(handle, start=1):
                stats["rows"] += 1
                context = f"{path}:{line_number}"
                row, stripped = _decode_source_row(raw, context=context)
                candidate = selection._candidate_from_row(
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
                    stats["official_ref8_image_excluded_rows"] += 1
                    continue
                if support_bank.witness(candidate.class_id) is None:
                    stats["external_support_uncovered_rows"] += 1
                    continue
                stats["support_eligible_rows"] += 1
                if candidate.image_id in old_o64_images:
                    stats["old_o64_image_excluded_rows"] += 1
                    continue
                if candidate.image_id in evaluated_dev_images:
                    stats["evaluated_dev_full_image_excluded_rows"] += 1
                    continue
                stats["fresh_eligible_rows"] += 1
                fresh_images.add(candidate.image_id)
                candidates.append(candidate)
        if stats["rows"] != expected_input_rows[manifest_name]:
            raise FreshO64BuildError(
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
        skips = Counter()
        chosen: list[selection.Candidate] = []
        for candidate in candidates:
            if candidate.image_id in used_images:
                skips["duplicate_image"] += 1
                continue
            if candidate.edge in used_edges:
                skips["duplicate_unordered_edge"] += 1
                continue
            endpoints = {candidate.anchor_ann_id, candidate.partner_ann_id}
            if endpoints.intersection(used_endpoints):
                skips["duplicate_endpoint"] += 1
                continue
            chosen.append(candidate)
            used_images.add(candidate.image_id)
            used_edges.add(candidate.edge)
            used_endpoints.update(endpoints)
            if len(chosen) == quota:
                break
        if len(chosen) != quota:
            raise FreshO64BuildError(
                f"could not satisfy quota for {manifest_name}: "
                f"selected={len(chosen)}, quota={quota}"
            )
        selected.extend(chosen)
        statistics[manifest_name] = {
            **dict(sorted(stats.items())),
            "fresh_eligible_unique_images": len(fresh_images),
            "quota": quota,
            "selected_pairs": len(chosen),
            "greedy_skip_histogram_before_quota": dict(sorted(skips.items())),
        }
    expected_pairs = sum(quota for _name, quota in manifest_quotas)
    if (
        len(selected) != expected_pairs
        or len(used_images) != expected_pairs
        or len(used_edges) != expected_pairs
        or len(used_endpoints) != 2 * expected_pairs
    ):
        raise FreshO64BuildError("selected pair uniqueness contract failed")
    return selected, statistics


def _load_selected_rows(
    *, input_root: Path, candidates: Sequence[selection.Candidate]
) -> list[tuple[dict[str, Any], bytes]]:
    by_manifest: dict[str, dict[int, selection.Candidate]] = {}
    for candidate in candidates:
        line_map = by_manifest.setdefault(candidate.manifest_name, {})
        if candidate.line_number in line_map:
            raise FreshO64BuildError("selected source line is duplicated")
        line_map[candidate.line_number] = candidate
    loaded: dict[str, tuple[dict[str, Any], bytes]] = {}
    for manifest_name, line_map in by_manifest.items():
        path = input_root / manifest_name
        remaining = set(line_map)
        with path.open("rb") as handle:
            for line_number, raw in enumerate(handle, start=1):
                candidate = line_map.get(line_number)
                if candidate is None:
                    continue
                context = f"{path}:{line_number}"
                row, stripped = _decode_source_row(raw, context=context)
                if _sha256_bytes(stripped) != candidate.source_row_sha256:
                    raise FreshO64BuildError(f"{context}: selected row hash drifted")
                replay = selection._candidate_from_row(
                    row,
                    manifest_name=manifest_name,
                    line_number=line_number,
                    source_row_sha256=candidate.source_row_sha256,
                )
                if replay != candidate:
                    raise FreshO64BuildError(f"{context}: selected identity drifted")
                loaded[candidate.pair_id] = (row, stripped)
                remaining.remove(line_number)
                if not remaining:
                    break
        if remaining:
            raise FreshO64BuildError(
                f"selected rows are missing from {manifest_name}: {sorted(remaining)}"
            )
    if len(loaded) != len(candidates):
        raise FreshO64BuildError("selected pair IDs are not unique")
    return [loaded[candidate.pair_id] for candidate in candidates]


def _xywh(value: Any, *, field: str, context: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise FreshO64BuildError(f"{context}: {field} must be one xywh box")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise FreshO64BuildError(f"{context}: {field} must contain finite numbers")
        number = float(item)
        if not math.isfinite(number):
            raise FreshO64BuildError(f"{context}: {field} must contain finite numbers")
        result.append(number)
    x, y, width, height = result
    if x < 0.0 or y < 0.0 or width <= 0.0 or height <= 0.0:
        raise FreshO64BuildError(f"{context}: {field} has invalid xywh geometry")
    return tuple(result)


def _endpoint_payload(
    *,
    endpoint: Mapping[str, Any],
    direction: str,
    instances: Sequence[Any],
    row_source: str,
    image_id: int,
    class_id: int,
    context: str,
) -> tuple[int, str, list[float]]:
    endpoint_context = f"{context} {direction}"
    if (
        _required_int(endpoint.get("image_id"), field="image_id", context=endpoint_context)
        != image_id
        or _required_text(endpoint.get("source"), field="source", context=endpoint_context)
        != row_source
    ):
        raise FreshO64BuildError(f"{endpoint_context}: endpoint source drifted")
    ann_id = _required_int(
        endpoint.get("coco_ann_id"), field="coco_ann_id", context=endpoint_context
    )
    expression = _required_text(
        endpoint.get("expression"), field="expression", context=endpoint_context
    )
    endpoint_box = _xywh(endpoint.get("bbox"), field="bbox", context=endpoint_context)
    matches = [
        item
        for item in instances
        if isinstance(item, dict) and type(item.get("coco_ann_id")) is int
        and item["coco_ann_id"] == ann_id
    ]
    if len(matches) != 1:
        raise FreshO64BuildError(
            f"{endpoint_context}: endpoint must match exactly one instance"
        )
    if (
        _xywh(matches[0].get("bbox"), field="instance bbox", context=endpoint_context)
        != endpoint_box
        or _required_int(
            matches[0].get("class_id"), field="instance class_id", context=endpoint_context
        )
        != class_id
    ):
        raise FreshO64BuildError(f"{endpoint_context}: endpoint instance drifted")
    x, y, width, height = endpoint_box
    return ann_id, expression, [x, y, x + width, y + height]


def _expand_rows(
    *,
    candidates: Sequence[selection.Candidate],
    source_rows: Sequence[tuple[dict[str, Any], bytes]],
    source_records: Mapping[str, Mapping[str, Any]],
    upstream_receipt_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    pair_members: list[dict[str, Any]] = []
    output_members: list[dict[str, Any]] = []
    for pair_index, (candidate, source_item) in enumerate(
        zip(candidates, source_rows, strict=True)
    ):
        row, stripped = source_item
        context = f"{candidate.manifest_name}:{candidate.line_number}"
        pair = row.get("assignment_pair")
        anchor = pair.get("anchor") if isinstance(pair, dict) else None
        partner = pair.get("partner") if isinstance(pair, dict) else None
        instances = row.get("instances")
        if (
            row.get("stage_b_data_driven_assignment_pair") is not True
            or row.get("stage_b_data_driven_assignment_pair_schema")
            != ASSIGNMENT_ROW_SCHEMA
            or row.get("assignment_pair_valid") is not True
            or row.get("assignment_pair_invalid_reason") is not None
            or row.get("stage_b_u2_category_complete") is not True
            or row.get("stage_b_u2_category_complete_schema")
            != "pivot.stageb.u2_category_complete_ref/v1"
            or not isinstance(pair, dict)
            or pair.get("schema") != ASSIGNMENT_ROW_SCHEMA
            or not isinstance(anchor, dict)
            or not isinstance(partner, dict)
            or not isinstance(instances, list)
            or len(instances) < 2
            or row.get("primary_support_instance_index") != 0
            or not isinstance(instances[0], dict)
            or instances[0].get("category_complete_primary") is not True
        ):
            raise FreshO64BuildError(f"{context}: selected assignment row drifted")
        image_id = _required_int(row.get("image_id"), field="image_id", context=context)
        row_source = _required_text(row.get("source"), field="source", context=context)
        filename = _required_text(row.get("filename"), field="filename", context=context)
        anchor_id = _required_int(
            anchor.get("coco_ann_id"), field="anchor coco_ann_id", context=context
        )
        partner_id = _required_int(
            partner.get("coco_ann_id"), field="partner coco_ann_id", context=context
        )
        class_id = _required_int(
            instances[0].get("class_id"), field="primary class_id", context=context
        )
        if (
            image_id != candidate.image_id
            or row_source != candidate.source
            or anchor_id != candidate.anchor_ann_id
            or partner_id != candidate.partner_ann_id
            or class_id != candidate.class_id
            or instances[0].get("coco_ann_id") != anchor_id
            or row.get("ann_id") != anchor_id
            or row.get("ref_id") != candidate.ref_id
            or row.get("sent_id") != candidate.sent_id
        ):
            raise FreshO64BuildError(f"{context}: candidate/source identity drifted")

        pair_members.append(
            {
                "pair_index": pair_index,
                "manifest": candidate.manifest_name,
                "source_line_number": candidate.line_number,
                "source_row_sha256": candidate.source_row_sha256,
                "pair_id": candidate.pair_id,
                "priority_sha256": candidate.priority_sha256,
                "source": candidate.source,
                "image_id": image_id,
                "anchor_coco_ann_id": anchor_id,
                "partner_coco_ann_id": partner_id,
                "class_id": class_id,
                "ref_id": candidate.ref_id,
                "sent_id": candidate.sent_id,
            }
        )
        endpoints = {"anchor": anchor, "partner": partner}
        for direction in DIRECTIONS:
            target_id, expression, bbox = _endpoint_payload(
                endpoint=endpoints[direction],
                direction=direction,
                instances=instances,
                row_source=row_source,
                image_id=image_id,
                class_id=class_id,
                context=context,
            )
            expected_target = anchor_id if direction == "anchor" else partner_id
            if target_id != expected_target:
                raise FreshO64BuildError(f"{context}: {direction} identity drifted")
            output_index = len(rows)
            output_row = {
                "direction": direction,
                "filename": filename,
                "grounding": {"regions": [{"bbox": bbox, "phrase": expression}]},
                "image_id": image_id,
                "pair_index": pair_index,
                "row_schema": ROW_SCHEMA,
                "source_assignment_line_number": candidate.line_number,
                "source_assignment_manifest": candidate.manifest_name,
                "source_assignment_manifest_sha256": source_records[
                    candidate.manifest_name
                ]["sha256"],
                "source_assignment_receipt_sha256": upstream_receipt_sha256,
                "source_member_pair_id": candidate.pair_id,
                "source_priority_sha256": candidate.priority_sha256,
                "source_row_sha256": _sha256_bytes(stripped),
                "target_coco_ann_id": target_id,
            }
            output_row_sha = _sha256_bytes(_canonical_bytes(output_row))
            rows.append(output_row)
            output_members.append(
                {
                    "output_index": output_index,
                    "pair_index": pair_index,
                    "direction": direction,
                    "image_id": image_id,
                    "target_coco_ann_id": target_id,
                    "source_member_pair_id": candidate.pair_id,
                    "source_row_sha256": candidate.source_row_sha256,
                    "output_row_sha256": output_row_sha,
                }
            )
    expected_pairs = len(candidates)
    if (
        len(rows) != 2 * expected_pairs
        or Counter(row["direction"] for row in rows)
        != {"anchor": expected_pairs, "partner": expected_pairs}
        or len({row["target_coco_ann_id"] for row in rows}) != 2 * expected_pairs
    ):
        raise FreshO64BuildError("directed expansion contract failed")
    return rows, pair_members, output_members


def make_plan(
    *,
    input_root: Path = INPUT_ROOT,
    input_receipt: Path = INPUT_RECEIPT,
    heldout_root: Path = HELDOUT_ROOT,
    old_o64_receipt: Path = OLD_O64_RECEIPT,
    new_head_receipt: Path = NEW_HEAD_RECEIPT,
    support_tsv: Path = SUPPORT_TSV,
    support_bank_cache: Path = SUPPORT_BANK_CACHE,
    support_image_root: Path = SUPPORT_IMAGE_ROOT,
    canonical_classes: Path = CANONICAL_CLASSES,
    output_root: Path = OUTPUT_ROOT,
    output_manifest: str = OUTPUT_MANIFEST,
    manifest_quotas: Sequence[tuple[str, int]] = MANIFEST_QUOTAS,
    expected_input_receipt_sha256: str = EXPECTED_INPUT_RECEIPT_SHA256,
    expected_input_sha256: Mapping[str, str] = EXPECTED_INPUT_SHA256,
    expected_input_rows: Mapping[str, int] = EXPECTED_INPUT_ROWS,
    expected_category_complete_receipt_sha256: str = (
        EXPECTED_CATEGORY_COMPLETE_RECEIPT_SHA256
    ),
    heldout_contract: Mapping[str, Mapping[str, Any]] = REF_SPLIT_CONTRACT,
    heldout_manifest_files: Mapping[str, str] = REF_SPLIT_MANIFEST_FILES,
    heldout_splits: Sequence[str] = REF_SPLITS,
    expected_heldout_union_images: int = EXPECTED_HELDOUT_UNION_IMAGES,
    expected_support_sha256: Mapping[str, str] = EXPECTED_SUPPORT_SHA256,
    expected_selection_library_sha256: str = EXPECTED_SELECTION_LIBRARY_SHA256,
    expected_old_o64_receipt_sha256: str = EXPECTED_OLD_O64_RECEIPT_SHA256,
    expected_new_head_receipt_sha256: str = EXPECTED_NEW_HEAD_RECEIPT_SHA256,
    expected_old_o64_images: int = EXPECTED_OLD_O64_IMAGES,
    expected_new_head_dev_full_images: int = EXPECTED_NEW_HEAD_DEV_FULL_IMAGES,
    expected_new_head_dev_screen_images: int = EXPECTED_NEW_HEAD_DEV_SCREEN_IMAGES,
    expected_selected_streams: Mapping[str, str] | None = EXPECTED_SELECTED_STREAMS,
) -> BuildPlan:
    selection._validate_expected_inputs(
        manifest_quotas, expected_input_sha256, expected_input_rows
    )
    if Path(output_manifest).name != output_manifest or not output_manifest.endswith(
        ".jsonl"
    ):
        raise FreshO64BuildError("output manifest must be a JSONL basename")
    if type(expected_old_o64_images) is not int or expected_old_o64_images <= 0:
        raise FreshO64BuildError("expected old O64 image count is invalid")
    input_root = input_root.expanduser().resolve(strict=True)
    heldout_root = heldout_root.expanduser().resolve(strict=True)
    output_root = output_root.expanduser().resolve(strict=False)

    selection_library_record = _load_selection_library(
        expected_selection_library_sha256
    )
    try:
        upstream_receipt, source_records, category_receipt = (
            selection._load_upstream_receipt(
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
        )
        heldout_images, heldout_receipt = selection._load_heldout_images(
            heldout_root=heldout_root,
            heldout_contract=heldout_contract,
            heldout_manifest_files=heldout_manifest_files,
            heldout_splits=heldout_splits,
            expected_union_images=expected_heldout_union_images,
        )
        support_bank = selection.SupportBank(
            support_tsv=support_tsv,
            support_bank_cache=support_bank_cache,
            support_image_root=support_image_root,
            canonical_classes=canonical_classes,
            expected_sha256=expected_support_sha256,
        )
    except selection.Overfit64BuildError as error:
        raise FreshO64BuildError(str(error)) from error

    old_images, old_binding = _load_old_o64_blacklist(
        receipt_path=old_o64_receipt,
        expected_sha256=expected_old_o64_receipt_sha256,
        expected_images=expected_old_o64_images,
    )
    dev_images, dev_binding = _load_new_head_dev_blacklist(
        receipt_path=new_head_receipt,
        expected_sha256=expected_new_head_receipt_sha256,
        expected_dev_full_images=expected_new_head_dev_full_images,
        expected_dev_screen_images=expected_new_head_dev_screen_images,
    )
    selected, source_statistics = _select_candidates(
        input_root=input_root,
        manifest_quotas=manifest_quotas,
        expected_input_rows=expected_input_rows,
        heldout_images=heldout_images,
        old_o64_images=old_images,
        evaluated_dev_images=dev_images,
        support_bank=support_bank,
    )
    selected_rows = _load_selected_rows(input_root=input_root, candidates=selected)
    directed_rows, pair_members, output_members = _expand_rows(
        candidates=selected,
        source_rows=selected_rows,
        source_records=source_records,
        upstream_receipt_sha256=upstream_receipt["sha256"],
    )

    image_ids = [candidate.image_id for candidate in selected]
    pair_ids = [candidate.pair_id for candidate in selected]
    edges = [f"{candidate.edge[0]}\t{candidate.edge[1]}" for candidate in selected]
    endpoints = [
        f"{candidate.anchor_ann_id}\t{candidate.partner_ann_id}"
        for candidate in selected
    ]
    selected_streams = {
        "ordered_pair_id_stream_sha256": _record_stream_sha256(pair_ids),
        "ordered_image_id_stream_sha256": _record_stream_sha256(
            [str(value) for value in image_ids]
        ),
        "sorted_image_id_json_sha256": _sha256_bytes(
            _canonical_bytes(sorted(image_ids))
        ),
        "ordered_unordered_edge_stream_sha256": _record_stream_sha256(edges),
        "ordered_endpoint_stream_sha256": _record_stream_sha256(endpoints),
    }
    if expected_selected_streams is not None:
        if set(expected_selected_streams) != set(EXPECTED_SELECTED_STREAMS):
            raise FreshO64BuildError("expected selected-stream map is incomplete")
        for key, expected in expected_selected_streams.items():
            _validate_sha256(expected, label=f"expected {key}")
            if selected_streams[key] != expected:
                raise FreshO64BuildError(
                    f"selected stream drifted for {key}: "
                    f"expected={expected}, observed={selected_streams[key]}"
                )

    selected_images = set(image_ids)
    expected_pairs = sum(quota for _name, quota in manifest_quotas)
    if (
        selected_images.intersection(heldout_images)
        or selected_images.intersection(old_images)
        or selected_images.intersection(dev_images)
    ):
        raise FreshO64BuildError("selected images overlap a sealed exclusion set")
    selected_class_ids = {candidate.class_id for candidate in selected}
    manifest_payload = b"".join(
        _canonical_bytes(row) + b"\n" for row in directed_rows
    )
    output_path = output_root / output_manifest
    support_receipt = support_bank.receipt(selected_class_ids)
    support_receipt.update(
        {
            "selection_only_not_consumed_by_odvg_loader": True,
            "selected_class_count": len(selected_class_ids),
            "external_clean_support_required": True,
        }
    )
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "row_schema": ROW_SCHEMA,
        "builder": _file_record(Path(__file__)),
        "selection_library": selection_library_record,
        "inputs": {
            "upstream_assignment_receipt": upstream_receipt,
            "upstream_category_complete_receipt": category_receipt,
            "source_manifest_order": [name for name, _quota in manifest_quotas],
            "source_manifests": {
                name: source_records[name] for name, _quota in manifest_quotas
            },
            "official_ref8": heldout_receipt,
            "old_o64_exclusion": old_binding,
            "evaluated_new_head_dev_full_exclusion": dev_binding,
            "external_support": support_receipt,
        },
        "selection_contract": {
            "policy": selection.SELECTION_POLICY,
            "namespace": selection.SELECTION_NAMESPACE,
            "pair_id_schema": selection.PAIR_ID_SCHEMA,
            "priority_order": "priority_sha256_pair_id_source_line_ascending",
            "manifest_quota_order": [
                {"manifest": name, "pairs": quota}
                for name, quota in manifest_quotas
            ],
            "filter_order": [
                "valid_official_assignment_pair",
                "official_ref8_image_exclusion",
                "external_clean_support_coverage",
                "old_o64_image_exclusion",
                "evaluated_new_head_dev_full_image_exclusion",
                "global_image_edge_endpoint_uniqueness",
            ],
            "old_priority_namespace_reused_without_a_new_random_seed": True,
            "model_score_free": True,
            "teacher_score_free": True,
            "checkpoint_output_free": True,
            "forbidden_source_keys": sorted(FORBIDDEN_SOURCE_KEYS),
            "target_crop_fallback_allowed": False,
            "expected_selected_streams": (
                dict(expected_selected_streams)
                if expected_selected_streams is not None
                else None
            ),
        },
        "conversion_contract": {
            "directions_in_pair_order": list(DIRECTIONS),
            "input_box_format": "xywh",
            "output_box_format": "xyxy",
            "output_dataset_mode": "odvg",
            "regions_per_row": 1,
            "expressions_per_row": 1,
            "external_support_is_selection_only": True,
            "teacher_model_or_checkpoint_scores_consumed": False,
        },
        "source_statistics": source_statistics,
        "pairs": expected_pairs,
        "rows": len(directed_rows),
        "direction_counts": dict(
            sorted(Counter(row["direction"] for row in directed_rows).items())
        ),
        "unique_images": len(selected_images),
        "unique_unordered_annotation_edges": len(set(edges)),
        "unique_target_annotation_ids": len(
            {row["target_coco_ann_id"] for row in directed_rows}
        ),
        "selected_class_count": len(selected_class_ids),
        "selected_stream_encoding": STREAM_ENCODING,
        "selected_streams": selected_streams,
        "pair_members": pair_members,
        "directed_members": output_members,
        "ordered_output_row_sha256_stream_sha256": _record_stream_sha256(
            [member["output_row_sha256"] for member in output_members]
        ),
        "output_manifest": output_manifest,
        "output": _predicted_file_record(output_path, manifest_payload),
        "scope_limitations": {
            "globally_unseen_to_b58_finetuning": False,
            "eligible_checkpoint_lineage": "b58_only_plus_fresh_random_adapter",
            "forbidden_initializers": [
                "old_full_assignment_DD1_checkpoints",
                "new_head_partition_trained_checkpoints",
            ],
            "formal_holdout_warning": (
                "A checkpoint trained on this O64 artifact cannot claim it as a "
                "formal generalization holdout."
            ),
        },
        "invariants": {
            "selection_library_matches_preregistered_sha256": True,
            "upstream_assignment_receipt_matches_preregistered_sha256": True,
            "upstream_category_complete_receipt_matches_preregistered_sha256": True,
            "all_source_manifests_match_preregistered_sha256": True,
            "all_eight_official_ref_manifests_match_contract": True,
            "old_o64_receipt_matches_preregistered_sha256": True,
            "old_o64_receipt_canonical_payload_recomputed": True,
            "new_head_receipt_matches_preregistered_sha256": True,
            "new_head_receipt_canonical_payload_recomputed": True,
            "selected_images_are_disjoint_from_official_ref8": True,
            "selected_images_are_disjoint_from_old_o64": True,
            "selected_images_are_disjoint_from_evaluated_dev_full": True,
            "selected_primary_classes_have_external_clean_support": True,
            "selected_support_witness_images_are_content_hash_bound": True,
            "target_crop_fallback_is_forbidden": True,
            "selection_reuses_the_sealed_model_score_free_priority": True,
            "source_rows_contain_no_forbidden_model_derived_fields": True,
            "source_quotas_are_exact": True,
            "selected_images_edges_and_endpoints_are_unique": True,
            "each_pair_emits_anchor_then_partner": True,
            "each_output_row_has_one_expression_and_one_xyxy_box": True,
            "all_128_target_annotations_are_unique": True,
            "selected_stream_hash_contract_satisfied": True,
        },
    }
    receipt["canonical_payload_sha256"] = _canonical_payload_sha256(receipt)
    return BuildPlan(manifest_bytes=manifest_payload, receipt=receipt)


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
        raise FreshO64BuildError(
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
            raise FreshO64BuildError(
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
        raise FreshO64BuildError(f"output root is not a directory: {output_root}")
    plan = make_plan(**kwargs)
    output_manifest = str(kwargs.get("output_manifest", OUTPUT_MANIFEST))
    try:
        observed_manifest = (output_root / output_manifest).read_bytes()
        observed_receipt = (output_root / "receipt.json").read_bytes()
    except OSError as error:
        raise FreshO64BuildError(f"could not read output artifact: {error}") from error
    if observed_manifest != plan.manifest_bytes:
        raise FreshO64BuildError("fresh O64 manifest does not replay exactly")
    if observed_receipt != _receipt_bytes(plan.receipt):
        raise FreshO64BuildError("fresh O64 receipt does not replay exactly")
    return plan.receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kwargs = {"output_root": args.output_root}
    if args.verify:
        receipt = verify(**kwargs)
    elif args.dry_run:
        receipt = make_plan(**kwargs).receipt
    else:
        receipt = build(**kwargs)
    print(json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
