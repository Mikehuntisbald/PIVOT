#!/usr/bin/env python3
"""Build model-score-free official assignment pairs for data-driven Stage-B."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT_ROOT = (
    REPO_ROOT
    / "data/ablations/stageb_refexp_three_train_category_complete_20260720"
)
OUTPUT_ROOT = (
    REPO_ROOT / "data/ablations/stageb_data_driven_assignment_pairs_20260722"
)
ROW_SCHEMA = "pivot.stageb.data_driven.official_assignment_pair/v1"
RECEIPT_SCHEMA = (
    "pivot.stageb.data_driven.official_assignment_pair_receipt/v1"
)
SELECTION_POLICY = (
    "least_used_partner_annotation_then_max_normalized_token_jaccard_v1"
)
MAX_TARGET_IOU_EXCLUSIVE = 0.3
MANIFESTS = (
    "refcoco_stageb_phrase_v1.jsonl",
    "refcocoplus_stageb_phrase_v1.jsonl",
    "refcocog_stageb_phrase_v1.jsonl",
)
EXPECTED_INPUT_SHA256 = {
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
EXPECTED_CATEGORY_COMPLETE_RECEIPT_SHA256 = (
    "fab09c61a8f53f05d75eedff25039a843ff27cb2d491d6c6576fe2b1e8aedd74"
)
EXPECTED_ROWS_BY_MANIFEST = {
    "refcoco_stageb_phrase_v1.jsonl": 120624,
    "refcocoplus_stageb_phrase_v1.jsonl": 120191,
    "refcocog_stageb_phrase_v1.jsonl": 80512,
}
IDENTITY_KEYS = (
    "source",
    "image_id",
    "ann_id",
    "ref_id",
    "sent_id",
    "split",
    "filename",
)
ADDED_KEYS = (
    "stage_b_data_driven_assignment_pair",
    "stage_b_data_driven_assignment_pair_schema",
    "assignment_pair_valid",
    "assignment_pair",
    "assignment_pair_invalid_reason",
)
_TOKEN_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)?", flags=re.UNICODE)


class AssignmentPairBuildError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class RowMeta:
    line_number: int
    source: str
    image_id: int
    filename: str
    coco_split: str
    category_id: int
    class_id: int
    ann_id: int
    coco_ann_id: int
    ref_id: int
    sent_id: int
    split: str
    expression: str
    normalized_expression: str
    tokens: frozenset[str]
    bbox: tuple[float, float, float, float]
    source_row_sha256: str

    @property
    def official_identity(self) -> tuple[str, int, int, int]:
        return (self.source, self.ref_id, self.sent_id, self.ann_id)

    @property
    def ordered_identity(self) -> tuple[Any, ...]:
        return (
            self.source,
            self.image_id,
            self.ann_id,
            self.ref_id,
            self.sent_id,
            self.split,
            self.filename,
        )

    @property
    def group_key(self) -> tuple[str, int, str, int]:
        return (
            self.coco_split,
            self.image_id,
            self.filename,
            self.category_id,
        )


@dataclass(frozen=True, slots=True)
class Assignment:
    partner_index: int | None
    target_iou: float | None
    candidate_rows: int
    candidate_annotations: int
    lexical_overlap_tokens: int | None
    lexical_union_tokens: int | None
    invalid_reason: str | None

    @property
    def valid(self) -> bool:
        return self.partner_index is not None


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, *, reported_path: Path | None = None) -> dict[str, Any]:
    if not path.is_file():
        raise AssignmentPairBuildError(f"not a file: {path}")
    stat = path.stat()
    return {
        "path": str((reported_path or path).resolve()),
        "size_bytes": int(stat.st_size),
        "sha256": _sha256_file(path),
    }


def _load_row(raw: str, *, path: Path, line_number: int) -> dict[str, Any]:
    if not raw.strip():
        raise AssignmentPairBuildError(f"blank line at {path}:{line_number}")
    try:
        row = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise AssignmentPairBuildError(
            f"invalid JSON at {path}:{line_number}: {error}"
        ) from error
    if not isinstance(row, dict):
        raise AssignmentPairBuildError(
            f"expected object at {path}:{line_number}"
        )
    return row


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AssignmentPairBuildError(f"could not load {label}: {path}: {error}") from error
    if not isinstance(value, dict):
        raise AssignmentPairBuildError(f"{label} must be a JSON object: {path}")
    return value


def _required_int(value: Any, *, field: str, context: str) -> int:
    if type(value) is not int:
        raise AssignmentPairBuildError(f"{context}: {field} must be an integer")
    return int(value)


def _required_text(value: Any, *, field: str, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AssignmentPairBuildError(f"{context}: {field} must be non-empty text")
    return value.strip()


def normalize_expression(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(_TOKEN_RE.findall(normalized))


def _bbox(value: Any, *, field: str, context: str) -> tuple[float, float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise AssignmentPairBuildError(f"{context}: {field} must be xywh[4]")
    numbers: list[float] = []
    for item in value:
        if isinstance(item, bool):
            raise AssignmentPairBuildError(f"{context}: {field} contains a boolean")
        try:
            number = float(item)
        except (TypeError, ValueError) as error:
            raise AssignmentPairBuildError(
                f"{context}: {field} contains a non-number"
            ) from error
        if not math.isfinite(number):
            raise AssignmentPairBuildError(f"{context}: {field} must be finite")
        numbers.append(number)
    if numbers[2] <= 0.0 or numbers[3] <= 0.0:
        raise AssignmentPairBuildError(
            f"{context}: {field} width and height must be positive"
        )
    return tuple(numbers)  # type: ignore[return-value]


def xywh_iou(
    left: Sequence[float], right: Sequence[float]
) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    ix1, iy1 = max(lx, rx), max(ly, ry)
    ix2, iy2 = min(lx + lw, rx + rw), min(ly + lh, ry + rh)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = lw * lh + rw * rh - intersection
    if union <= 0.0:
        raise AssignmentPairBuildError("bbox union must be positive")
    return intersection / union


def _row_meta(row: Mapping[str, Any], *, line_number: int, context: str) -> RowMeta:
    present_reserved = sorted(set(row).intersection(ADDED_KEYS))
    if present_reserved:
        raise AssignmentPairBuildError(
            f"{context}: input already contains assignment fields: {present_reserved}"
        )
    if row.get("stage_b_u2_category_complete") is not True:
        raise AssignmentPairBuildError(
            f"{context}: category-complete marker is not exact true"
        )
    if (
        row.get("stage_b_u2_category_complete_schema")
        != "pivot.stageb.u2_category_complete_ref/v1"
    ):
        raise AssignmentPairBuildError(
            f"{context}: category-complete row schema drifted"
        )
    if type(row.get("primary_support_instance_index")) is not int or row.get(
        "primary_support_instance_index"
    ) != 0:
        raise AssignmentPairBuildError(
            f"{context}: primary_support_instance_index must be integer zero"
        )

    source = _required_text(row.get("source"), field="source", context=context)
    filename = _required_text(
        row.get("filename"), field="filename", context=context
    )
    split = _required_text(row.get("split"), field="split", context=context)
    coco_split = _required_text(
        row.get("category_complete_coco_split"),
        field="category_complete_coco_split",
        context=context,
    )
    image_id = _required_int(row.get("image_id"), field="image_id", context=context)
    ann_id = _required_int(row.get("ann_id"), field="ann_id", context=context)
    ref_id = _required_int(row.get("ref_id"), field="ref_id", context=context)
    sent_id = _required_int(row.get("sent_id"), field="sent_id", context=context)
    category_id = _required_int(
        row.get("category_complete_coco_category_id"),
        field="category_complete_coco_category_id",
        context=context,
    )

    instances = row.get("instances")
    if not isinstance(instances, list) or not instances or not all(
        isinstance(instance, dict) for instance in instances
    ):
        raise AssignmentPairBuildError(
            f"{context}: instances must be a non-empty object list"
        )
    instance_count = _required_int(
        row.get("category_complete_instance_count"),
        field="category_complete_instance_count",
        context=context,
    )
    if instance_count != len(instances):
        raise AssignmentPairBuildError(
            f"{context}: category-complete instance count drifted"
        )
    primary = instances[0]
    if primary.get("category_complete_primary") is not True:
        raise AssignmentPairBuildError(
            f"{context}: instances[0] is not the category-complete primary"
        )
    if primary.get("text_is_negative") is not False:
        raise AssignmentPairBuildError(
            f"{context}: official primary expression must be positive"
        )
    coco_ann_id = _required_int(
        primary.get("coco_ann_id"), field="instances[0].coco_ann_id", context=context
    )
    if coco_ann_id != ann_id:
        raise AssignmentPairBuildError(
            f"{context}: row ann_id differs from primary coco_ann_id"
        )
    primary_category = _required_int(
        primary.get("refcoco_category_id"),
        field="instances[0].refcoco_category_id",
        context=context,
    )
    if primary_category != category_id:
        raise AssignmentPairBuildError(
            f"{context}: primary category differs from category-complete category"
        )
    class_id = _required_int(
        primary.get("class_id"), field="instances[0].class_id", context=context
    )
    target_bbox = _bbox(
        primary.get("bbox"), field="instances[0].bbox", context=context
    )
    seen_coco_ann_ids: set[int] = set()
    for index, instance in enumerate(instances):
        instance_context = f"{context}:instances[{index}]"
        instance_ann_id = _required_int(
            instance.get("coco_ann_id"), field="coco_ann_id", context=instance_context
        )
        if instance_ann_id in seen_coco_ann_ids:
            raise AssignmentPairBuildError(
                f"{context}: duplicate coco_ann_id={instance_ann_id}"
            )
        seen_coco_ann_ids.add(instance_ann_id)
        if (
            _required_int(
                instance.get("refcoco_category_id"),
                field="refcoco_category_id",
                context=instance_context,
            )
            != category_id
            or _required_int(
                instance.get("class_id"), field="class_id", context=instance_context
            )
            != class_id
        ):
            raise AssignmentPairBuildError(
                f"{context}: category-complete row contains another category"
            )
        _bbox(instance.get("bbox"), field="bbox", context=instance_context)
        if index > 0 and instance.get("category_complete_auxiliary") is not True:
            raise AssignmentPairBuildError(
                f"{instance_context}: missing category-complete auxiliary marker"
            )

    expression = _required_text(
        primary.get("raw_phrase"), field="instances[0].raw_phrase", context=context
    )
    positive_expression = _required_text(
        primary.get("positive_phrase"),
        field="instances[0].positive_phrase",
        context=context,
    )
    normalized_expression = normalize_expression(expression)
    if normalize_expression(positive_expression) != normalized_expression:
        raise AssignmentPairBuildError(
            f"{context}: raw and positive official expressions differ"
        )
    return RowMeta(
        line_number=line_number,
        source=source,
        image_id=image_id,
        filename=filename,
        coco_split=coco_split,
        category_id=category_id,
        class_id=class_id,
        ann_id=ann_id,
        coco_ann_id=coco_ann_id,
        ref_id=ref_id,
        sent_id=sent_id,
        split=split,
        expression=expression,
        normalized_expression=normalized_expression,
        tokens=frozenset(normalized_expression.split()),
        bbox=target_bbox,
        source_row_sha256=hashlib.sha256(_canonical_bytes(row)).hexdigest(),
    )


def _stream_digest_update(digest: Any, value: Any) -> None:
    digest.update(_canonical_bytes(value))
    digest.update(b"\n")


def _load_metadata(
    source: Path,
    *,
    expected_sha256: str,
    expected_rows: int,
) -> tuple[list[RowMeta], dict[tuple[str, int, str, int], list[int]], dict[str, Any]]:
    input_record = _file_record(source)
    if input_record["sha256"] != expected_sha256:
        raise AssignmentPairBuildError(
            f"{source.name}: input SHA256 mismatch: expected={expected_sha256}, "
            f"observed={input_record['sha256']}"
        )
    metas: list[RowMeta] = []
    groups: dict[tuple[str, int, str, int], list[int]] = defaultdict(list)
    identities: set[tuple[str, int, int, int]] = set()
    ordered_identity_digest = hashlib.sha256()
    source_row_digest = hashlib.sha256()
    image_filename: dict[tuple[str, int], str] = {}
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            row = _load_row(raw, path=source, line_number=line_number)
            meta = _row_meta(
                row, line_number=line_number, context=f"{source}:{line_number}"
            )
            if meta.official_identity in identities:
                raise AssignmentPairBuildError(
                    f"{source}:{line_number}: duplicate official expression identity"
                )
            identities.add(meta.official_identity)
            image_key = (meta.coco_split, meta.image_id)
            prior_filename = image_filename.setdefault(image_key, meta.filename)
            if prior_filename != meta.filename:
                raise AssignmentPairBuildError(
                    f"{source}:{line_number}: one COCO image id maps to two filenames"
                )
            index = len(metas)
            metas.append(meta)
            groups[meta.group_key].append(index)
            _stream_digest_update(ordered_identity_digest, meta.ordered_identity)
            source_row_digest.update(meta.source_row_sha256.encode("ascii") + b"\n")
    if len(metas) != expected_rows:
        raise AssignmentPairBuildError(
            f"{source.name}: expected {expected_rows} rows, observed {len(metas)}"
        )
    if not metas:
        raise AssignmentPairBuildError(f"source manifest is empty: {source}")
    if _sha256_file(source) != input_record["sha256"]:
        raise AssignmentPairBuildError(f"input changed while reading: {source}")
    return metas, dict(groups), {
        "input": input_record,
        "ordered_identity_stream_sha256": ordered_identity_digest.hexdigest(),
        "source_row_stream_sha256": source_row_digest.hexdigest(),
        "unique_official_expression_identities": len(identities),
        "official_identity_source_namespaces": sorted(
            {meta.source for meta in metas}
        ),
        "image_category_groups": len(groups),
    }


def _lexical_counts(left: RowMeta, right: RowMeta) -> tuple[int, int, float]:
    overlap = len(left.tokens.intersection(right.tokens))
    union = len(left.tokens.union(right.tokens))
    if union <= 0:
        raise AssignmentPairBuildError("normalized token union is empty")
    return overlap, union, overlap / union


def select_assignments(
    metas: Sequence[RowMeta],
    groups: Mapping[tuple[str, int, str, int], Sequence[int]],
) -> list[Assignment]:
    annotation_usage: Counter[tuple[tuple[str, int, str, int], int]] = Counter()
    row_usage: Counter[int] = Counter()
    assignments: list[Assignment] = []
    for anchor_index, anchor in enumerate(metas):
        if not anchor.normalized_expression:
            assignments.append(
                Assignment(
                    partner_index=None,
                    target_iou=None,
                    candidate_rows=0,
                    candidate_annotations=0,
                    lexical_overlap_tokens=None,
                    lexical_union_tokens=None,
                    invalid_reason="empty_normalized_official_expression",
                )
            )
            continue
        other_annotation_exists = False
        spatial_candidate_exists = False
        eligible_by_annotation: dict[
            int, list[tuple[int, float, int, int, float]]
        ] = defaultdict(list)
        for candidate_index in groups[anchor.group_key]:
            candidate = metas[candidate_index]
            if candidate.coco_ann_id == anchor.coco_ann_id:
                continue
            other_annotation_exists = True
            target_iou = xywh_iou(anchor.bbox, candidate.bbox)
            if not target_iou < MAX_TARGET_IOU_EXCLUSIVE:
                continue
            spatial_candidate_exists = True
            if not candidate.normalized_expression:
                continue
            if candidate.normalized_expression == anchor.normalized_expression:
                continue
            overlap, union, score = _lexical_counts(anchor, candidate)
            eligible_by_annotation[candidate.coco_ann_id].append(
                (candidate_index, target_iou, overlap, union, score)
            )

        if not eligible_by_annotation:
            if not other_annotation_exists:
                reason = "no_distinct_official_annotation"
            elif not spatial_candidate_exists:
                reason = "no_partner_below_target_iou"
            else:
                reason = "no_distinct_normalized_official_expression"
            assignments.append(
                Assignment(
                    partner_index=None,
                    target_iou=None,
                    candidate_rows=0,
                    candidate_annotations=0,
                    lexical_overlap_tokens=None,
                    lexical_union_tokens=None,
                    invalid_reason=reason,
                )
            )
            continue

        best_by_annotation: list[tuple[int, float, int, int, float]] = []
        candidate_rows = 0
        for candidate_ann_id, candidates in eligible_by_annotation.items():
            candidate_rows += len(candidates)
            best = min(
                candidates,
                key=lambda item: (
                    -item[4],
                    row_usage[item[0]],
                    metas[item[0]].line_number,
                ),
            )
            if metas[best[0]].coco_ann_id != candidate_ann_id:
                raise AssignmentPairBuildError("candidate annotation partition drifted")
            best_by_annotation.append(best)
        selected = min(
            best_by_annotation,
            key=lambda item: (
                annotation_usage[(anchor.group_key, metas[item[0]].coco_ann_id)],
                -item[4],
                row_usage[item[0]],
                metas[item[0]].coco_ann_id,
                metas[item[0]].ref_id,
                metas[item[0]].sent_id,
                metas[item[0]].line_number,
            ),
        )
        partner_index, target_iou, overlap, union, _score = selected
        partner = metas[partner_index]
        if partner_index == anchor_index or partner.group_key != anchor.group_key:
            raise AssignmentPairBuildError("selected partner escaped its source group")
        annotation_usage[(anchor.group_key, partner.coco_ann_id)] += 1
        row_usage[partner_index] += 1
        assignments.append(
            Assignment(
                partner_index=partner_index,
                target_iou=target_iou,
                candidate_rows=candidate_rows,
                candidate_annotations=len(eligible_by_annotation),
                lexical_overlap_tokens=overlap,
                lexical_union_tokens=union,
                invalid_reason=None,
            )
        )
    if len(assignments) != len(metas):
        raise AssignmentPairBuildError("assignment count drifted")
    return assignments


def _identity_payload(meta: RowMeta, *, include_target_iou: float | None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "expression": meta.expression,
        "normalized_expression": meta.normalized_expression,
        "bbox": list(meta.bbox),
        "coco_ann_id": meta.coco_ann_id,
        "source": meta.source,
        "ref_id": meta.ref_id,
        "sent_id": meta.sent_id,
        "ann_id": meta.ann_id,
        "image_id": meta.image_id,
        "category_id": meta.category_id,
        "manifest_line_number": meta.line_number,
    }
    if include_target_iou is not None:
        payload["target_iou"] = include_target_iou
    return payload


def _added_fields(
    anchor: RowMeta,
    assignment: Assignment,
    metas: Sequence[RowMeta],
) -> dict[str, Any]:
    partner = (
        None
        if assignment.partner_index is None
        else _identity_payload(
            metas[assignment.partner_index],
            include_target_iou=assignment.target_iou,
        )
    )
    lexical_jaccard = (
        None
        if assignment.lexical_union_tokens is None
        else assignment.lexical_overlap_tokens / assignment.lexical_union_tokens
    )
    pair = {
        "schema": ROW_SCHEMA,
        "anchor": _identity_payload(anchor, include_target_iou=None),
        "partner": partner,
        "selection": {
            "policy": SELECTION_POLICY,
            "model_score_free": True,
            "max_target_iou_exclusive": MAX_TARGET_IOU_EXCLUSIVE,
            "candidate_rows": assignment.candidate_rows,
            "candidate_annotations": assignment.candidate_annotations,
            "lexical_metric": "normalized_token_set_jaccard",
            "lexical_overlap_tokens": assignment.lexical_overlap_tokens,
            "lexical_union_tokens": assignment.lexical_union_tokens,
            "lexical_jaccard": lexical_jaccard,
        },
    }
    return {
        "stage_b_data_driven_assignment_pair": True,
        "stage_b_data_driven_assignment_pair_schema": ROW_SCHEMA,
        "assignment_pair_valid": assignment.valid,
        "assignment_pair": pair,
        "assignment_pair_invalid_reason": assignment.invalid_reason,
    }


def _count_histogram(values: Sequence[int]) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(Counter(values).items())
    }


def _iou_bucket(value: float) -> str:
    boundaries = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
    lower = 0.0
    for upper in boundaries:
        if value < upper:
            return f"[{lower:.2f},{upper:.2f})"
        lower = upper
    raise AssignmentPairBuildError(f"valid target IoU escaped contract: {value}")


def _lexical_bucket(value: float) -> str:
    index = min(int(value * 10.0), 9)
    return f"[{index / 10.0:.1f},{(index + 1) / 10.0:.1f}]" if index == 9 else (
        f"[{index / 10.0:.1f},{(index + 1) / 10.0:.1f})"
    )


def _manifest_statistics(
    metas: Sequence[RowMeta], assignments: Sequence[Assignment]
) -> dict[str, Any]:
    invalid_reasons: Counter[str] = Counter()
    candidate_rows: list[int] = []
    candidate_annotations: list[int] = []
    target_iou_histogram: Counter[str] = Counter()
    lexical_histogram: Counter[str] = Counter()
    partner_row_usage: Counter[int] = Counter()
    partner_annotation_usage: Counter[
        tuple[tuple[str, int, str, int], int]
    ] = Counter()
    valid_rows = 0
    for anchor, assignment in zip(metas, assignments, strict=True):
        candidate_rows.append(assignment.candidate_rows)
        candidate_annotations.append(assignment.candidate_annotations)
        if not assignment.valid:
            invalid_reasons[str(assignment.invalid_reason)] += 1
            continue
        valid_rows += 1
        if (
            assignment.partner_index is None
            or assignment.target_iou is None
            or assignment.lexical_overlap_tokens is None
            or assignment.lexical_union_tokens is None
        ):
            raise AssignmentPairBuildError(
                "valid assignment is missing partner or selection evidence"
            )
        partner = metas[assignment.partner_index]
        partner_row_usage[assignment.partner_index] += 1
        partner_annotation_usage[(anchor.group_key, partner.coco_ann_id)] += 1
        target_iou_histogram[_iou_bucket(assignment.target_iou)] += 1
        lexical_histogram[
            _lexical_bucket(
                assignment.lexical_overlap_tokens
                / assignment.lexical_union_tokens
            )
        ] += 1
    invalid_rows = len(assignments) - valid_rows
    return {
        "rows": len(assignments),
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
        "invalid_reason_histogram": dict(sorted(invalid_reasons.items())),
        "candidate_row_count_histogram": _count_histogram(candidate_rows),
        "candidate_annotation_count_histogram": _count_histogram(
            candidate_annotations
        ),
        "target_iou_histogram": dict(sorted(target_iou_histogram.items())),
        "lexical_jaccard_histogram": dict(sorted(lexical_histogram.items())),
        "partner_usage": {
            "unique_partner_expression_rows_used": len(partner_row_usage),
            "unique_partner_annotations_used": len(partner_annotation_usage),
            "partner_expression_row_use_count_histogram": _count_histogram(
                list(partner_row_usage.values())
            ),
            "partner_annotation_use_count_histogram": _count_histogram(
                list(partner_annotation_usage.values())
            ),
            "max_partner_expression_row_uses": max(
                partner_row_usage.values(), default=0
            ),
            "max_partner_annotation_uses": max(
                partner_annotation_usage.values(), default=0
            ),
        },
    }


def _write_manifest(
    *,
    source: Path,
    temporary_output: Path,
    final_output: Path,
    metas: Sequence[RowMeta],
    assignments: Sequence[Assignment],
    source_record: Mapping[str, Any],
) -> dict[str, Any]:
    if temporary_output.exists() or final_output.exists():
        raise AssignmentPairBuildError(
            f"refusing to replace existing output: {final_output}"
        )
    with source.open("r", encoding="utf-8") as input_handle, temporary_output.open(
        "x", encoding="ascii"
    ) as output_handle:
        observed_rows = 0
        for observed_rows, raw in enumerate(input_handle, start=1):
            if observed_rows > len(metas):
                raise AssignmentPairBuildError(
                    f"{source.name}: source gained rows between passes"
                )
            row = _load_row(raw, path=source, line_number=observed_rows)
            meta = metas[observed_rows - 1]
            if hashlib.sha256(_canonical_bytes(row)).hexdigest() != meta.source_row_sha256:
                raise AssignmentPairBuildError(
                    f"{source}:{observed_rows}: source changed between passes"
                )
            output = dict(row)
            output.update(_added_fields(meta, assignments[observed_rows - 1], metas))
            output_handle.write(_canonical_bytes(output).decode("ascii") + "\n")
        if observed_rows != len(metas):
            raise AssignmentPairBuildError(
                f"{source.name}: source lost rows between passes"
            )
        output_handle.flush()
        os.fsync(output_handle.fileno())
    if _sha256_file(source) != source_record["sha256"]:
        raise AssignmentPairBuildError(f"input changed while writing: {source}")

    output_identity_digest = hashlib.sha256()
    output_base_digest = hashlib.sha256()
    with temporary_output.open("r", encoding="ascii") as handle:
        audited_rows = 0
        for audited_rows, raw in enumerate(handle, start=1):
            output = _load_row(raw, path=temporary_output, line_number=audited_rows)
            if audited_rows > len(metas):
                raise AssignmentPairBuildError(
                    f"{source.name}: output contains extra rows"
                )
            meta = metas[audited_rows - 1]
            expected_fields = _added_fields(
                meta, assignments[audited_rows - 1], metas
            )
            for key, expected in expected_fields.items():
                if output.get(key) != expected:
                    raise AssignmentPairBuildError(
                        f"{temporary_output}:{audited_rows}: {key} drifted"
                    )
            base = dict(output)
            for key in ADDED_KEYS:
                base.pop(key, None)
            base_sha256 = hashlib.sha256(_canonical_bytes(base)).hexdigest()
            if base_sha256 != meta.source_row_sha256:
                raise AssignmentPairBuildError(
                    f"{temporary_output}:{audited_rows}: source row was not preserved"
                )
            output_base_digest.update(base_sha256.encode("ascii") + b"\n")
            _stream_digest_update(output_identity_digest, meta.ordered_identity)
        if audited_rows != len(metas):
            raise AssignmentPairBuildError(
                f"{source.name}: output row count drifted"
            )
    statistics = _manifest_statistics(metas, assignments)
    statistics.update(
        {
            "output": _file_record(
                temporary_output, reported_path=final_output
            ),
            "output_ordered_identity_stream_sha256": (
                output_identity_digest.hexdigest()
            ),
            "output_base_row_stream_sha256": output_base_digest.hexdigest(),
        }
    )
    return statistics


def _validate_expected_maps(
    manifest_names: Sequence[str],
    expected_input_sha256: Mapping[str, str],
    expected_rows_by_manifest: Mapping[str, int],
) -> None:
    names = set(manifest_names)
    if len(names) != len(manifest_names) or names != set(expected_input_sha256):
        raise AssignmentPairBuildError(
            "expected input SHA256 keys must exactly match unique manifest names"
        )
    if names != set(expected_rows_by_manifest):
        raise AssignmentPairBuildError(
            "expected row-count keys must exactly match manifest names"
        )
    for name in manifest_names:
        expected_hash = expected_input_sha256[name]
        if not isinstance(expected_hash, str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_hash
        ):
            raise AssignmentPairBuildError(f"invalid expected SHA256 for {name}")
        if type(expected_rows_by_manifest[name]) is not int or expected_rows_by_manifest[
            name
        ] <= 0:
            raise AssignmentPairBuildError(f"invalid expected row count for {name}")


def _category_complete_receipt_record(
    *,
    input_root: Path,
    manifest_names: Sequence[str],
    expected_sha256: str,
    expected_input_sha256: Mapping[str, str],
    expected_rows_by_manifest: Mapping[str, int],
) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise AssignmentPairBuildError(
            "invalid expected category-complete receipt SHA256"
        )
    path = input_root / "receipt.json"
    record = _file_record(path)
    if record["sha256"] != expected_sha256:
        raise AssignmentPairBuildError(
            "category-complete receipt SHA256 mismatch: "
            f"expected={expected_sha256}, observed={record['sha256']}"
        )
    payload = _load_json_object(path, label="category-complete receipt")
    if (
        payload.get("schema") != "pivot.stageb.u2_category_complete_receipt/v1"
        or payload.get("row_schema") != "pivot.stageb.u2_category_complete_ref/v1"
    ):
        raise AssignmentPairBuildError("category-complete receipt schema drifted")
    manifests = payload.get("manifests")
    if not isinstance(manifests, dict) or set(manifests) != set(manifest_names):
        raise AssignmentPairBuildError(
            "category-complete receipt manifest keys drifted"
        )
    for name in manifest_names:
        manifest = manifests[name]
        if not isinstance(manifest, dict) or manifest.get("rows") != (
            expected_rows_by_manifest[name]
        ):
            raise AssignmentPairBuildError(
                f"category-complete receipt row count drifted for {name}"
            )
        output = manifest.get("output")
        if not isinstance(output, dict) or output.get("sha256") != (
            expected_input_sha256[name]
        ):
            raise AssignmentPairBuildError(
                f"category-complete receipt output hash drifted for {name}"
            )
    return record


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_all(
    *,
    input_root: Path,
    output_root: Path,
    manifest_names: Sequence[str] = MANIFESTS,
    expected_input_sha256: Mapping[str, str] = EXPECTED_INPUT_SHA256,
    expected_rows_by_manifest: Mapping[str, int] = EXPECTED_ROWS_BY_MANIFEST,
    expected_category_complete_receipt_sha256: str = (
        EXPECTED_CATEGORY_COMPLETE_RECEIPT_SHA256
    ),
) -> dict[str, Any]:
    _validate_expected_maps(
        manifest_names, expected_input_sha256, expected_rows_by_manifest
    )
    input_root = input_root.expanduser().resolve(strict=True)
    output_root = output_root.expanduser().resolve(strict=False)
    if not input_root.is_dir():
        raise AssignmentPairBuildError(f"input root is not a directory: {input_root}")
    if output_root.exists():
        raise AssignmentPairBuildError(
            f"refusing to replace existing output root: {output_root}"
        )
    try:
        output_root.relative_to(input_root)
    except ValueError:
        pass
    else:
        raise AssignmentPairBuildError("output root must not be inside input root")
    category_complete_receipt = _category_complete_receipt_record(
        input_root=input_root,
        manifest_names=manifest_names,
        expected_sha256=expected_category_complete_receipt_sha256,
        expected_input_sha256=expected_input_sha256,
        expected_rows_by_manifest=expected_rows_by_manifest,
    )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.tmp-", dir=str(output_root.parent)
        )
    )
    committed = False
    try:
        manifest_receipts: dict[str, Any] = {}
        total_rows = 0
        total_valid_rows = 0
        unique_identities = 0
        identity_source_namespaces: set[str] = set()
        global_identity_digest = hashlib.sha256()
        global_source_row_digest = hashlib.sha256()
        for name in manifest_names:
            if Path(name).name != name:
                raise AssignmentPairBuildError(f"manifest name is not a basename: {name}")
            source = input_root / name
            metas, groups, metadata_receipt = _load_metadata(
                source,
                expected_sha256=expected_input_sha256[name],
                expected_rows=expected_rows_by_manifest[name],
            )
            assignments = select_assignments(metas, groups)
            final_output = output_root / name
            manifest_receipt = {
                **metadata_receipt,
                **_write_manifest(
                    source=source,
                    temporary_output=temporary_root / name,
                    final_output=final_output,
                    metas=metas,
                    assignments=assignments,
                    source_record=metadata_receipt["input"],
                ),
            }
            if (
                manifest_receipt["ordered_identity_stream_sha256"]
                != manifest_receipt["output_ordered_identity_stream_sha256"]
                or manifest_receipt["source_row_stream_sha256"]
                != manifest_receipt["output_base_row_stream_sha256"]
            ):
                raise AssignmentPairBuildError(
                    f"{name}: output identity/order/source preservation audit failed"
                )
            manifest_receipts[name] = manifest_receipt
            total_rows += int(manifest_receipt["rows"])
            total_valid_rows += int(manifest_receipt["valid_rows"])
            unique_identities += int(
                manifest_receipt["unique_official_expression_identities"]
            )
            manifest_namespaces = set(
                manifest_receipt["official_identity_source_namespaces"]
            )
            overlap = identity_source_namespaces.intersection(manifest_namespaces)
            if overlap:
                raise AssignmentPairBuildError(
                    "official identity source namespace occurs in multiple manifests: "
                    f"{sorted(overlap)}"
                )
            identity_source_namespaces.update(manifest_namespaces)
            _stream_digest_update(global_identity_digest, name)
            global_identity_digest.update(
                manifest_receipt["ordered_identity_stream_sha256"].encode("ascii")
            )
            global_identity_digest.update(b"\n")
            _stream_digest_update(global_source_row_digest, name)
            global_source_row_digest.update(
                manifest_receipt["source_row_stream_sha256"].encode("ascii")
            )
            global_source_row_digest.update(b"\n")

        expected_total_rows = sum(expected_rows_by_manifest[name] for name in manifest_names)
        if total_rows != expected_total_rows:
            raise AssignmentPairBuildError(
                f"expected {expected_total_rows} total rows, observed {total_rows}"
            )
        if unique_identities != total_rows:
            raise AssignmentPairBuildError(
                "global official identity count differs from total row count"
            )
        receipt: dict[str, Any] = {
            "schema": RECEIPT_SCHEMA,
            "row_schema": ROW_SCHEMA,
            "builder": _file_record(Path(__file__)),
            "category_complete_receipt": category_complete_receipt,
            "selection_contract": {
                "policy": SELECTION_POLICY,
                "model_score_free": True,
                "same_manifest_only": True,
                "same_image_and_category_only": True,
                "partner_annotation_distinct": True,
                "normalized_expression_distinct": True,
                "max_target_iou_exclusive": MAX_TARGET_IOU_EXCLUSIVE,
                "balance_unit": "official_partner_coco_annotation_within_image_category",
                "hardness_metric": "normalized_token_set_jaccard",
                "tie_break": (
                    "partner_row_use_then_coco_ann_id_ref_id_sent_id_line_number"
                ),
                "forbidden_inputs": [
                    "teacher_scores",
                    "teacher_logits",
                    "model_scores",
                    "model_logits",
                    "checkpoint_outputs",
                ],
            },
            "manifest_order": list(manifest_names),
            "rows": total_rows,
            "unique_identities": unique_identities,
            "valid_rows": total_valid_rows,
            "invalid_rows": total_rows - total_valid_rows,
            "ordered_identity_stream_sha256": global_identity_digest.hexdigest(),
            "source_row_stream_sha256": global_source_row_digest.hexdigest(),
            "manifests": manifest_receipts,
            "invariants": {
                "all_input_sha256_values_match_preregistered_sources": True,
                "category_complete_receipt_binds_every_input_manifest": True,
                "official_expression_identities_are_globally_unique": True,
                "all_input_rows_emitted_exactly_once_in_original_order": True,
                "source_rows_preserved_except_assignment_fields": True,
                "invalid_anchors_retained_with_null_partner": True,
                "valid_partners_are_official_rows_from_same_manifest": True,
                "valid_partners_share_image_filename_and_category": True,
                "valid_partner_annotations_are_distinct": True,
                "valid_target_bbox_iou_is_strictly_below_0_3": True,
                "valid_normalized_expressions_are_distinct": True,
                "selection_is_deterministic_and_model_score_free": True,
                "outputs_committed_as_one_atomic_directory_rename": True,
            },
        }
        receipt["canonical_payload_sha256"] = hashlib.sha256(
            _canonical_bytes(receipt)
        ).hexdigest()
        receipt_path = temporary_root / "receipt.json"
        with receipt_path.open("x", encoding="ascii") as handle:
            json.dump(receipt, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(temporary_root)
        if output_root.exists():
            raise AssignmentPairBuildError(
                f"refusing concurrent overwrite of output root: {output_root}"
            )
        os.rename(temporary_root, output_root)
        committed = True
        _fsync_directory(output_root.parent)
        return receipt
    finally:
        if not committed and temporary_root.exists():
            shutil.rmtree(temporary_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = build_all(input_root=args.input_root, output_root=args.output_root)
    print(json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
