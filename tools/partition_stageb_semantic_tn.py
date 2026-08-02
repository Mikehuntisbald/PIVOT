#!/usr/bin/env python3
"""Build a deterministic, holdout-isolated semantic-TN train/calibration split.

Rows whose ``image_id`` occurs in either sealed strict manifest are removed
before partitioning.  Eligible images are assigned by a seeded SHA-256
threshold, so every row for one image stays in exactly one partition.  Source
JSONL rows are copied byte-for-byte; the tool never adds a split field or
otherwise rewrites training examples.

The all-pairs outputs support confidence data-quality ablations.  The parallel
single-edit outputs are the stricter token-supervision mainline and contain
only rows with one valid ``tn_edits`` entry whose category/from/to/span values
exactly match the corresponding top-level provenance lists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    REPO_ROOT
    / "data/ablations/stageb_gdino_adapter_semantic_verified_20260711"
    / "semantic_verified_pairs.jsonl"
)
DEFAULT_STRICT_DIR = (
    REPO_ROOT
    / "data/eval_manifests/stageb_vlm_verified_strict_ann_umd_val_20260711"
)
DEFAULT_STRICT2031 = DEFAULT_STRICT_DIR / "eval_manifest.jsonl"
DEFAULT_STRICT1607 = (
    DEFAULT_STRICT_DIR / "semantic_stageb_union_image_disjoint_manifest.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "data/ablations/stageb_gdino_adapter_semantic_partition_20260717"
)
DEFAULT_SEED = "20260717"
DEFAULT_CALIBRATION_RATIO = Fraction(1, 10)

AUDIT_SCHEMA = "stage-b-semantic-tn-leakage-isolated-partition-v1"
CONTRACT_SCHEMA = "stage-b-semantic-tn-image-hash-contract-v1"
OUTPUT_NAMES = {
    "train": "train.jsonl",
    "calibration": "calibration.jsonl",
    "single_edit_train": "single_edit_train.jsonl",
    "single_edit_calibration": "single_edit_calibration.jsonl",
}
AUDIT_NAME = "audit.json"
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_INTEGER_RE = re.compile(r"^(?:0|[1-9][0-9]*)$")


class SemanticTNPartitionError(RuntimeError):
    pass


@dataclass(frozen=True)
class EditProvenance:
    edit_count: int | None
    single_edit: bool
    multi_edit: bool
    valid_tn_edits: bool
    valid_replace_category: bool
    valid_replace_from: bool
    valid_replace_to: bool
    valid_replace_span: bool
    category_consistent: bool
    replace_from_consistent: bool
    replace_to_consistent: bool
    replace_span_consistent: bool
    metadata_consistent: bool
    token_mask_provenance_valid: bool
    single_edit_token_eligible: bool
    categories: tuple[str, ...]
    invalid_reasons: tuple[str, ...]


@dataclass(frozen=True)
class ParsedRow:
    line_number: int
    raw: bytes
    value: Dict[str, Any]
    image_id: int
    sample_id: str | None
    dataset: str | None
    provenance: EditProvenance | None


@dataclass(frozen=True)
class LoadedJSONL:
    path: Path
    sha256: str
    size_bytes: int
    rows: tuple[ParsedRow, ...]


@dataclass(frozen=True)
class PartitionPlan:
    source: LoadedJSONL
    strict2031: LoadedJSONL
    strict1607: LoadedJSONL
    seed: str
    calibration_ratio: Fraction
    filtered: tuple[ParsedRow, ...]
    eligible: tuple[ParsedRow, ...]
    train: tuple[ParsedRow, ...]
    calibration: tuple[ParsedRow, ...]
    single_edit_train: tuple[ParsedRow, ...]
    single_edit_calibration: tuple[ParsedRow, ...]
    assignments: Mapping[int, str]


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _required_text(row: Mapping[str, Any], key: str, *, context: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SemanticTNPartitionError(
            f"missing non-empty critical field {key!r} at {context}"
        )
    return value


def _image_id(row: Mapping[str, Any], *, context: str) -> int:
    value = row.get("image_id")
    if isinstance(value, bool):
        raise SemanticTNPartitionError(f"invalid image_id at {context}: {value!r}")
    if isinstance(value, int):
        image_id = value
    elif isinstance(value, str) and _INTEGER_RE.fullmatch(value):
        image_id = int(value)
    else:
        raise SemanticTNPartitionError(f"invalid image_id at {context}: {value!r}")
    if image_id < 0:
        raise SemanticTNPartitionError(f"negative image_id at {context}: {image_id}")
    return image_id


def _clean_label(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _valid_text_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(item, str) and bool(item.strip()) for item in value)
    )


def _valid_span(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
        and 0 <= value[0] < value[1]
    )


def _valid_span_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(
        _valid_span(item) for item in value
    )


def _edit_provenance(row: Mapping[str, Any]) -> EditProvenance:
    edits = row.get("tn_edits")
    edit_count = len(edits) if isinstance(edits, list) else None
    valid_tn_edits = isinstance(edits, list) and bool(edits)
    if valid_tn_edits:
        for edit in edits:
            if not isinstance(edit, Mapping):
                valid_tn_edits = False
                break
            category = edit.get("category")
            replace_from = edit.get("replace_from")
            replace_to = edit.get("replace_to")
            if (
                not isinstance(category, str)
                or not category.strip()
                or not isinstance(replace_from, str)
                or not replace_from.strip()
                or not isinstance(replace_to, str)
                or not replace_to.strip()
                or _clean_label(replace_from) == _clean_label(replace_to)
                or not _valid_span(edit.get("replace_span"))
            ):
                valid_tn_edits = False
                break

    top_category = row.get("replace_category")
    top_from = row.get("replace_from")
    top_to = row.get("replace_to")
    top_span = row.get("replace_span")
    valid_category = _valid_text_list(top_category)
    valid_from = _valid_text_list(top_from)
    valid_to = _valid_text_list(top_to)
    valid_spans = _valid_span_list(top_span)

    comparable_edits = (
        isinstance(edits, list)
        and bool(edits)
        and all(isinstance(edit, Mapping) for edit in edits)
    )
    expected_category = (
        [edit.get("category") for edit in edits] if comparable_edits else None
    )
    expected_from = (
        [edit.get("replace_from") for edit in edits] if comparable_edits else None
    )
    expected_to = (
        [edit.get("replace_to") for edit in edits] if comparable_edits else None
    )
    expected_span = (
        [edit.get("replace_span") for edit in edits] if comparable_edits else None
    )
    category_consistent = valid_category and top_category == expected_category
    from_consistent = valid_from and top_from == expected_from
    to_consistent = valid_to and top_to == expected_to
    span_consistent = valid_spans and top_span == expected_span
    metadata_consistent = all(
        (category_consistent, from_consistent, to_consistent, span_consistent)
    )
    token_mask_valid = bool(valid_tn_edits and metadata_consistent)
    single_edit = edit_count == 1
    multi_edit = edit_count is not None and edit_count > 1

    if valid_category:
        categories = tuple(_clean_label(value) for value in top_category)
    elif valid_tn_edits:
        categories = tuple(_clean_label(edit["category"]) for edit in edits)
    else:
        categories = ("__missing_or_invalid__",)

    reasons = []
    checks = (
        (valid_tn_edits, "invalid_tn_edits"),
        (valid_category, "invalid_replace_category"),
        (valid_from, "invalid_replace_from"),
        (valid_to, "invalid_replace_to"),
        (valid_spans, "invalid_replace_span"),
        (category_consistent, "inconsistent_replace_category"),
        (from_consistent, "inconsistent_replace_from"),
        (to_consistent, "inconsistent_replace_to"),
        (span_consistent, "inconsistent_replace_span"),
    )
    reasons.extend(reason for passed, reason in checks if not passed)
    if multi_edit:
        reasons.append("multiple_edits")
    elif not single_edit:
        reasons.append("missing_or_zero_edits")

    return EditProvenance(
        edit_count=edit_count,
        single_edit=single_edit,
        multi_edit=multi_edit,
        valid_tn_edits=bool(valid_tn_edits),
        valid_replace_category=valid_category,
        valid_replace_from=valid_from,
        valid_replace_to=valid_to,
        valid_replace_span=valid_spans,
        category_consistent=category_consistent,
        replace_from_consistent=from_consistent,
        replace_to_consistent=to_consistent,
        replace_span_consistent=span_consistent,
        metadata_consistent=metadata_consistent,
        token_mask_provenance_valid=token_mask_valid,
        single_edit_token_eligible=bool(single_edit and token_mask_valid),
        categories=categories,
        invalid_reasons=tuple(reasons),
    )


def _normalize_expected_sha256(value: str | None, *, label: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise SemanticTNPartitionError(
            f"{label} expected SHA-256 must be exactly 64 hexadecimal characters"
        )
    return value.lower()


def _load_jsonl(
    path: Path,
    *,
    label: str,
    semantic_source: bool,
    expected_sha256: str | None = None,
) -> LoadedJSONL:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise SemanticTNPartitionError(f"missing {label} JSONL: {path}")
    expected = _normalize_expected_sha256(expected_sha256, label=label)
    observed_sha = sha256_file(path)
    if expected is not None and observed_sha != expected:
        raise SemanticTNPartitionError(
            f"{label} input hash drift: expected {expected}, got {observed_sha}"
        )
    raw_lines = path.read_bytes().splitlines(keepends=True)
    if not raw_lines:
        raise SemanticTNPartitionError(f"empty {label} JSONL: {path}")

    parsed = []
    sample_ids: set[str] = set()
    for line_number, raw in enumerate(raw_lines, 1):
        context = f"{path}:{line_number}"
        if not raw.strip():
            raise SemanticTNPartitionError(f"blank row at {context}")
        try:
            text = raw.decode("utf-8")
            row = json.loads(text, parse_constant=_nonfinite_json_constant)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise SemanticTNPartitionError(f"invalid JSON at {context}: {error}") from error
        if not isinstance(row, dict):
            raise SemanticTNPartitionError(f"non-object JSON row at {context}")
        image_id = _image_id(row, context=context)

        sample_id = None
        dataset = None
        provenance = None
        if semantic_source:
            sample_id = _required_text(row, "sample_id", context=context)
            dataset = _clean_label(_required_text(row, "dataset", context=context))
            _required_text(row, "sent", context=context)
            _required_text(row, "try_tn", context=context)
            if sample_id in sample_ids:
                raise SemanticTNPartitionError(
                    f"duplicate sample_id {sample_id!r} at {context}"
                )
            sample_ids.add(sample_id)
            provenance = _edit_provenance(row)
        elif "sample_id" in row:
            sample_id = _required_text(row, "sample_id", context=context)
            if sample_id in sample_ids:
                raise SemanticTNPartitionError(
                    f"duplicate sample_id {sample_id!r} inside {label} at {context}"
                )
            sample_ids.add(sample_id)

        parsed.append(
            ParsedRow(
                line_number=line_number,
                raw=raw,
                value=row,
                image_id=image_id,
                sample_id=sample_id,
                dataset=dataset,
                provenance=provenance,
            )
        )
    return LoadedJSONL(
        path=path,
        sha256=observed_sha,
        size_bytes=path.stat().st_size,
        rows=tuple(parsed),
    )


def _ratio(value: Fraction | str | float | int) -> Fraction:
    try:
        ratio = value if isinstance(value, Fraction) else Fraction(str(value))
    except (ValueError, ZeroDivisionError) as error:
        raise SemanticTNPartitionError(
            f"invalid calibration ratio {value!r}"
        ) from error
    if ratio <= 0 or ratio >= 1:
        raise SemanticTNPartitionError("calibration ratio must be strictly between 0 and 1")
    return ratio


def _seed(value: Any) -> str:
    seed = str(value)
    if not seed or "\0" in seed:
        raise SemanticTNPartitionError("partition seed must be non-empty and contain no NUL")
    return seed


def _is_calibration_image(
    image_id: int, *, seed: str, calibration_ratio: Fraction
) -> bool:
    digest = hashlib.sha256(f"{seed}\0{image_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest, byteorder="big", signed=False)
    return (
        value * calibration_ratio.denominator
        < calibration_ratio.numerator * (1 << 256)
    )


def _build_plan(
    *,
    source_path: Path,
    strict2031_path: Path,
    strict1607_path: Path,
    seed: Any,
    calibration_ratio: Fraction | str | float | int,
    expected_input_sha256: str | None = None,
    expected_strict2031_sha256: str | None = None,
    expected_strict1607_sha256: str | None = None,
) -> PartitionPlan:
    source = _load_jsonl(
        source_path,
        label="semantic_verified_pairs",
        semantic_source=True,
        expected_sha256=expected_input_sha256,
    )
    strict2031 = _load_jsonl(
        strict2031_path,
        label="strict2031",
        semantic_source=False,
        expected_sha256=expected_strict2031_sha256,
    )
    strict1607 = _load_jsonl(
        strict1607_path,
        label="strict1607",
        semantic_source=False,
        expected_sha256=expected_strict1607_sha256,
    )
    normalized_seed = _seed(seed)
    normalized_ratio = _ratio(calibration_ratio)

    strict_union = {row.image_id for row in strict2031.rows}
    strict_union.update(row.image_id for row in strict1607.rows)
    filtered = tuple(row for row in source.rows if row.image_id in strict_union)
    eligible = tuple(row for row in source.rows if row.image_id not in strict_union)
    if not eligible:
        raise SemanticTNPartitionError(
            "holdout filtering removed every semantic TN row"
        )

    assignments = {
        image_id: (
            "calibration"
            if _is_calibration_image(
                image_id,
                seed=normalized_seed,
                calibration_ratio=normalized_ratio,
            )
            else "train"
        )
        for image_id in sorted({row.image_id for row in eligible})
    }
    train = tuple(row for row in eligible if assignments[row.image_id] == "train")
    calibration = tuple(
        row for row in eligible if assignments[row.image_id] == "calibration"
    )
    single_train = tuple(
        row
        for row in train
        if row.provenance is not None
        and row.provenance.single_edit_token_eligible
    )
    single_calibration = tuple(
        row
        for row in calibration
        if row.provenance is not None
        and row.provenance.single_edit_token_eligible
    )
    plan = PartitionPlan(
        source=source,
        strict2031=strict2031,
        strict1607=strict1607,
        seed=normalized_seed,
        calibration_ratio=normalized_ratio,
        filtered=filtered,
        eligible=eligible,
        train=train,
        calibration=calibration,
        single_edit_train=single_train,
        single_edit_calibration=single_calibration,
        assignments=assignments,
    )
    _validate_plan(plan)
    return plan


def _sample_ids(rows: Iterable[ParsedRow]) -> list[str]:
    return [str(row.sample_id) for row in rows]


def _validate_plan(plan: PartitionPlan) -> None:
    strict_images = {row.image_id for row in plan.strict2031.rows}
    strict_images.update(row.image_id for row in plan.strict1607.rows)
    train_images = {row.image_id for row in plan.train}
    calibration_images = {row.image_id for row in plan.calibration}
    if train_images & calibration_images:
        raise SemanticTNPartitionError("an image crosses train and calibration")
    if (train_images | calibration_images) & strict_images:
        raise SemanticTNPartitionError("a strict holdout image survived filtering")
    eligible_ids = _sample_ids(plan.eligible)
    train_ids = _sample_ids(plan.train)
    calibration_ids = _sample_ids(plan.calibration)
    if train_ids + calibration_ids == eligible_ids:
        # Source-order subsequences need not concatenate into source order, so this
        # fast path is uncommon; set checks below define exact coverage.
        pass
    if set(train_ids) & set(calibration_ids):
        raise SemanticTNPartitionError("a sample crosses train and calibration")
    if set(train_ids) | set(calibration_ids) != set(eligible_ids):
        raise SemanticTNPartitionError("train/calibration union differs from eligible rows")
    for name, rows in (
        ("single_edit_train", plan.single_edit_train),
        ("single_edit_calibration", plan.single_edit_calibration),
    ):
        if not all(
            row.provenance is not None
            and row.provenance.single_edit_token_eligible
            for row in rows
        ):
            raise SemanticTNPartitionError(
                f"{name} contains a row without valid single-edit provenance"
            )
    if not set(_sample_ids(plan.single_edit_train)).issubset(set(train_ids)):
        raise SemanticTNPartitionError("single-edit train is not a train subset")
    if not set(_sample_ids(plan.single_edit_calibration)).issubset(
        set(calibration_ids)
    ):
        raise SemanticTNPartitionError(
            "single-edit calibration is not a calibration subset"
        )


def _loaded_record(
    loaded: LoadedJSONL, *, expected_sha256: str | None
) -> Dict[str, Any]:
    expected = _normalize_expected_sha256(expected_sha256, label=str(loaded.path))
    return {
        "path": str(loaded.path),
        "size_bytes": loaded.size_bytes,
        "sha256": loaded.sha256,
        "rows": len(loaded.rows),
        "unique_images": len({row.image_id for row in loaded.rows}),
        "expected_sha256": expected,
        "expected_sha256_enforced": expected is not None,
    }


def _output_record(path: Path, rows: Sequence[ParsedRow]) -> Dict[str, Any]:
    expected_bytes = b"".join(row.raw for row in rows)
    if path.read_bytes() != expected_bytes:
        raise SemanticTNPartitionError(
            f"output rows are not byte-for-byte source subsequences: {path}"
        )
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": len(rows),
        "unique_images": len({row.image_id for row in rows}),
    }


def _rate(count: int, total: int) -> float:
    return float(Fraction(count, total)) if total else 0.0


def _distribution(rows: Sequence[ParsedRow]) -> Dict[str, Any]:
    dataset_rows: Counter[str] = Counter()
    taxonomy_row_membership: Counter[str] = Counter()
    taxonomy_edit_occurrences: Counter[str] = Counter()
    dataset_taxonomy: Dict[str, Counter[str]] = defaultdict(Counter)
    edit_counts: Counter[str] = Counter()
    invalid_reasons: Counter[str] = Counter()
    facts: Counter[str] = Counter()
    for row in rows:
        dataset = str(row.dataset)
        dataset_rows[dataset] += 1
        provenance = row.provenance
        if provenance is None:
            continue
        categories = provenance.categories
        taxonomy_edit_occurrences.update(categories)
        for category in set(categories):
            taxonomy_row_membership[category] += 1
            dataset_taxonomy[dataset][category] += 1
        edit_key = (
            str(provenance.edit_count)
            if provenance.edit_count is not None
            else "invalid"
        )
        edit_counts[edit_key] += 1
        invalid_reasons.update(provenance.invalid_reasons)
        for key, value in (
            ("single_edit_rows", provenance.single_edit),
            ("multi_edit_rows", provenance.multi_edit),
            ("valid_tn_edits_rows", provenance.valid_tn_edits),
            ("valid_replace_category_rows", provenance.valid_replace_category),
            ("valid_replace_from_rows", provenance.valid_replace_from),
            ("valid_replace_to_rows", provenance.valid_replace_to),
            ("valid_replace_span_rows", provenance.valid_replace_span),
            ("consistent_replace_category_rows", provenance.category_consistent),
            ("consistent_replace_from_rows", provenance.replace_from_consistent),
            ("consistent_replace_to_rows", provenance.replace_to_consistent),
            ("consistent_replace_span_rows", provenance.replace_span_consistent),
            ("consistent_all_edit_metadata_rows", provenance.metadata_consistent),
            (
                "token_mask_provenance_valid_rows",
                provenance.token_mask_provenance_valid,
            ),
            (
                "single_edit_token_eligible_rows",
                provenance.single_edit_token_eligible,
            ),
        ):
            facts[key] += int(value)
    total = len(rows)
    provenance_payload = {key: int(value) for key, value in sorted(facts.items())}
    for key in (
        "single_edit_rows",
        "multi_edit_rows",
        "valid_tn_edits_rows",
        "valid_replace_category_rows",
        "valid_replace_from_rows",
        "valid_replace_to_rows",
        "valid_replace_span_rows",
        "consistent_replace_category_rows",
        "consistent_replace_from_rows",
        "consistent_replace_to_rows",
        "consistent_replace_span_rows",
        "consistent_all_edit_metadata_rows",
        "token_mask_provenance_valid_rows",
        "single_edit_token_eligible_rows",
    ):
        provenance_payload.setdefault(key, 0)
    provenance_payload.update(
        {
            "token_mask_provenance_coverage": _rate(
                provenance_payload["token_mask_provenance_valid_rows"], total
            ),
            "single_edit_token_eligibility_coverage": _rate(
                provenance_payload["single_edit_token_eligible_rows"], total
            ),
            "edit_count_rows": dict(sorted(edit_counts.items())),
            "single_edit_exclusion_reason_rows": dict(sorted(invalid_reasons.items())),
        }
    )
    return {
        "rows": total,
        "unique_images": len({row.image_id for row in rows}),
        "dataset_rows": dict(sorted(dataset_rows.items())),
        "taxonomy": {
            "row_membership": dict(sorted(taxonomy_row_membership.items())),
            "edit_occurrences": dict(sorted(taxonomy_edit_occurrences.items())),
            "dataset_row_membership": {
                dataset: dict(sorted(counts.items()))
                for dataset, counts in sorted(dataset_taxonomy.items())
            },
        },
        "edit_provenance": provenance_payload,
    }


def _filtered_overlap(plan: PartitionPlan) -> Dict[str, Any]:
    strict2031_images = {row.image_id for row in plan.strict2031.rows}
    strict1607_images = {row.image_id for row in plan.strict1607.rows}
    source_images = {row.image_id for row in plan.source.rows}

    def membership(image_id: int) -> str | None:
        in_2031 = image_id in strict2031_images
        in_1607 = image_id in strict1607_images
        if in_2031 and in_1607:
            return "both"
        if in_2031:
            return "strict2031_only"
        if in_1607:
            return "strict1607_only"
        return None

    image_counts: Counter[str] = Counter()
    for image_id in source_images:
        key = membership(image_id)
        if key is not None:
            image_counts[key] += 1
    row_counts: Counter[str] = Counter()
    for row in plan.source.rows:
        key = membership(row.image_id)
        if key is not None:
            row_counts[key] += 1
    return {
        "strict_manifest_image_overlap": len(strict2031_images & strict1607_images),
        "strict_manifest_union_unique_images": len(
            strict2031_images | strict1607_images
        ),
        "semantic_source_image_overlap": {
            **{
                key: int(image_counts[key])
                for key in ("strict2031_only", "strict1607_only", "both")
            },
            "union": len(source_images & (strict2031_images | strict1607_images)),
        },
        "filtered_semantic_rows": {
            **{
                key: int(row_counts[key])
                for key in ("strict2031_only", "strict1607_only", "both")
            },
            "union": len(plan.filtered),
        },
        "eligible_rows_after_union_filter": len(plan.eligible),
        "eligible_unique_images_after_union_filter": len(
            {row.image_id for row in plan.eligible}
        ),
    }


def _membership_sha256(rows: Sequence[ParsedRow]) -> str:
    return canonical_sha256(_sample_ids(rows))


def _contract(plan: PartitionPlan) -> Dict[str, Any]:
    strict_union = sorted(
        {row.image_id for row in plan.strict2031.rows}
        | {row.image_id for row in plan.strict1607.rows}
    )
    assignments = [
        [image_id, plan.assignments[image_id]] for image_id in sorted(plan.assignments)
    ]
    return {
        "schema": CONTRACT_SCHEMA,
        "inputs": {
            "semantic_verified_pairs_sha256": plan.source.sha256,
            "strict2031_sha256": plan.strict2031.sha256,
            "strict1607_sha256": plan.strict1607.sha256,
        },
        "holdout_filter": {
            "key": "normalized_nonnegative_integer_image_id",
            "operation": "remove semantic rows whose image_id is in strict2031 union strict1607",
            "strict_union_image_ids_sha256": canonical_sha256(strict_union),
        },
        "partition": {
            "group_key": "normalized_nonnegative_integer_image_id",
            "seed": plan.seed,
            "calibration_ratio": {
                "numerator": plan.calibration_ratio.numerator,
                "denominator": plan.calibration_ratio.denominator,
            },
            "hash": "sha256(utf8(seed) || NUL || ascii(canonical_decimal_image_id))",
            "threshold": "uint256_be(digest) / 2**256 < calibration_ratio",
            "assignment_sha256": canonical_sha256(assignments),
        },
        "membership_sha256": {
            "eligible": _membership_sha256(plan.eligible),
            "train": _membership_sha256(plan.train),
            "calibration": _membership_sha256(plan.calibration),
            "single_edit_train": _membership_sha256(plan.single_edit_train),
            "single_edit_calibration": _membership_sha256(
                plan.single_edit_calibration
            ),
        },
        "row_contract": {
            "all_pairs_purpose": "confidence_data_quality_ablation",
            "single_edit_purpose": "token_supervision_mainline",
            "single_edit_gate": (
                "exactly_one_valid_tn_edits_entry_and_exact_top_level_"
                "category_from_to_span_consistency"
            ),
            "source_rows_byte_preserved": True,
            "source_order_preserved_within_each_output": True,
            "output_rows_are_source_subsequences": True,
        },
    }


def _make_audit(
    plan: PartitionPlan,
    *,
    output_records: Mapping[str, Mapping[str, Any]],
    expected_input_sha256: str | None,
    expected_strict2031_sha256: str | None,
    expected_strict1607_sha256: str | None,
) -> Dict[str, Any]:
    contract = _contract(plan)
    train_images = {row.image_id for row in plan.train}
    calibration_images = {row.image_id for row in plan.calibration}
    strict_images = {row.image_id for row in plan.strict2031.rows}
    strict_images.update(row.image_id for row in plan.strict1607.rows)
    return {
        "schema": AUDIT_SCHEMA,
        "kind": "completed_semantic_tn_leakage_isolated_partition",
        "inputs": {
            "semantic_verified_pairs": _loaded_record(
                plan.source, expected_sha256=expected_input_sha256
            ),
            "strict2031": _loaded_record(
                plan.strict2031, expected_sha256=expected_strict2031_sha256
            ),
            "strict1607": _loaded_record(
                plan.strict1607, expected_sha256=expected_strict1607_sha256
            ),
        },
        "outputs": {key: dict(value) for key, value in sorted(output_records.items())},
        "policy": {
            "seed": plan.seed,
            "calibration_ratio": {
                "numerator": plan.calibration_ratio.numerator,
                "denominator": plan.calibration_ratio.denominator,
                "float": float(plan.calibration_ratio),
            },
            "filter_before_partition": True,
            "group_key": "image_id",
        },
        "filtered_overlap": _filtered_overlap(plan),
        "distributions": {
            "input": _distribution(plan.source.rows),
            "filtered_holdout_overlap": _distribution(plan.filtered),
            "eligible": _distribution(plan.eligible),
            "train": _distribution(plan.train),
            "calibration": _distribution(plan.calibration),
            "single_edit_train": _distribution(plan.single_edit_train),
            "single_edit_calibration": _distribution(
                plan.single_edit_calibration
            ),
        },
        "partition_contract": contract,
        "partition_contract_sha256": canonical_sha256(contract),
        "invariants": {
            "duplicate_sample_ids_rejected": True,
            "critical_fields_required": [
                "sample_id",
                "image_id",
                "dataset",
                "sent",
                "try_tn",
            ],
            "expected_input_hashes_checked_when_supplied": True,
            "strict_union_filtered_by_image_id": True,
            "eligible_strict_union_image_overlap": len(
                (train_images | calibration_images) & strict_images
            ),
            "train_calibration_image_overlap": len(
                train_images & calibration_images
            ),
            "all_pairs_sample_union_exact": True,
            "single_edit_outputs_are_all_pairs_subsets": True,
            "single_edit_invalid_metadata_rows_excluded": True,
            "source_rows_byte_preserved": True,
        },
    }


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise SemanticTNPartitionError(f"stale temporary output exists: {temporary}")
    try:
        with temporary.open("xb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (
        json.dumps(
            value,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    _atomic_write_bytes(path, payload)


def create_partition(
    *,
    source_path: Path,
    strict2031_path: Path,
    strict1607_path: Path,
    output_dir: Path,
    seed: Any = DEFAULT_SEED,
    calibration_ratio: Fraction | str | float | int = DEFAULT_CALIBRATION_RATIO,
    expected_input_sha256: str | None = None,
    expected_strict2031_sha256: str | None = None,
    expected_strict1607_sha256: str | None = None,
) -> Dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    output_paths = {
        name: output_dir / filename for name, filename in OUTPUT_NAMES.items()
    }
    audit_path = output_dir / AUDIT_NAME
    existing = [
        str(path)
        for path in (*output_paths.values(), audit_path)
        if path.exists()
    ]
    if existing:
        raise SemanticTNPartitionError(
            f"refusing to overwrite existing partition outputs: {existing}"
        )

    plan = _build_plan(
        source_path=source_path,
        strict2031_path=strict2031_path,
        strict1607_path=strict1607_path,
        seed=seed,
        calibration_ratio=calibration_ratio,
        expected_input_sha256=expected_input_sha256,
        expected_strict2031_sha256=expected_strict2031_sha256,
        expected_strict1607_sha256=expected_strict1607_sha256,
    )
    rows_by_output = {
        "train": plan.train,
        "calibration": plan.calibration,
        "single_edit_train": plan.single_edit_train,
        "single_edit_calibration": plan.single_edit_calibration,
    }
    created = []
    try:
        for name, path in output_paths.items():
            _atomic_write_bytes(path, b"".join(row.raw for row in rows_by_output[name]))
            created.append(path)
        output_records = {
            name: _output_record(path, rows_by_output[name])
            for name, path in output_paths.items()
        }
        audit = _make_audit(
            plan,
            output_records=output_records,
            expected_input_sha256=expected_input_sha256,
            expected_strict2031_sha256=expected_strict2031_sha256,
            expected_strict1607_sha256=expected_strict1607_sha256,
        )
        _atomic_write_json(audit_path, audit)
        created.append(audit_path)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return audit


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SemanticTNPartitionError(f"could not read audit {path}: {error}") from error
    if not isinstance(value, dict):
        raise SemanticTNPartitionError(f"partition audit is not an object: {path}")
    return value


def verify_partition(audit_path: Path) -> Dict[str, Any]:
    audit_path = audit_path.expanduser().resolve()
    audit = _read_json(audit_path)
    if (
        audit.get("schema") != AUDIT_SCHEMA
        or audit.get("kind")
        != "completed_semantic_tn_leakage_isolated_partition"
    ):
        raise SemanticTNPartitionError("partition audit schema/kind mismatch")
    contract = audit.get("partition_contract")
    if not isinstance(contract, Mapping) or audit.get(
        "partition_contract_sha256"
    ) != canonical_sha256(contract):
        raise SemanticTNPartitionError("partition contract hash mismatch")
    inputs = audit.get("inputs")
    outputs = audit.get("outputs")
    policy = audit.get("policy")
    if not all(isinstance(value, Mapping) for value in (inputs, outputs, policy)):
        raise SemanticTNPartitionError("partition audit is incomplete")

    def input_record(name: str) -> Mapping[str, Any]:
        record = inputs.get(name)
        if not isinstance(record, Mapping):
            raise SemanticTNPartitionError(f"missing audit input record {name!r}")
        return record

    semantic_record = input_record("semantic_verified_pairs")
    strict2031_record = input_record("strict2031")
    strict1607_record = input_record("strict1607")
    ratio_record = policy.get("calibration_ratio")
    if not isinstance(ratio_record, Mapping):
        raise SemanticTNPartitionError("audit calibration ratio is missing")
    try:
        ratio = Fraction(
            int(ratio_record["numerator"]), int(ratio_record["denominator"])
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise SemanticTNPartitionError("audit calibration ratio is invalid") from error

    plan = _build_plan(
        source_path=Path(str(semantic_record.get("path", ""))),
        strict2031_path=Path(str(strict2031_record.get("path", ""))),
        strict1607_path=Path(str(strict1607_record.get("path", ""))),
        seed=policy.get("seed"),
        calibration_ratio=ratio,
        expected_input_sha256=str(semantic_record.get("sha256", "")),
        expected_strict2031_sha256=str(strict2031_record.get("sha256", "")),
        expected_strict1607_sha256=str(strict1607_record.get("sha256", "")),
    )
    rows_by_output = {
        "train": plan.train,
        "calibration": plan.calibration,
        "single_edit_train": plan.single_edit_train,
        "single_edit_calibration": plan.single_edit_calibration,
    }
    output_records = {}
    for name, rows in rows_by_output.items():
        record = outputs.get(name)
        if not isinstance(record, Mapping):
            raise SemanticTNPartitionError(f"missing audit output record {name!r}")
        path = Path(str(record.get("path", ""))).expanduser().resolve()
        if not path.is_file():
            raise SemanticTNPartitionError(f"missing partition output: {path}")
        current = _output_record(path, rows)
        if dict(record) != current:
            raise SemanticTNPartitionError(f"partition output identity drifted: {path}")
        output_records[name] = current

    rebuilt = _make_audit(
        plan,
        output_records=output_records,
        expected_input_sha256=semantic_record.get("expected_sha256"),
        expected_strict2031_sha256=strict2031_record.get("expected_sha256"),
        expected_strict1607_sha256=strict1607_record.get("expected_sha256"),
    )
    if audit != rebuilt:
        raise SemanticTNPartitionError(
            "partition audit differs from deterministic replay"
        )
    return {
        "schema": AUDIT_SCHEMA,
        "audit": str(audit_path),
        "partition_contract_sha256": audit["partition_contract_sha256"],
        "eligible_rows": len(plan.eligible),
        "verified": True,
    }


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--strict2031", type=Path, default=DEFAULT_STRICT2031)
    parser.add_argument("--strict1607", type=Path, default=DEFAULT_STRICT1607)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument(
        "--calibration-ratio",
        default=str(float(DEFAULT_CALIBRATION_RATIO)),
        help="Exact decimal or fraction in (0,1); default: 0.1",
    )
    parser.add_argument("--expected-input-sha256")
    parser.add_argument("--expected-strict2031-sha256")
    parser.add_argument("--expected-strict1607-sha256")
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Replay and verify output-dir/audit.json without writing files",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        help="Audit path for --verify-only; defaults to output-dir/audit.json",
    )
    return parser


def main() -> None:
    args = make_parser().parse_args()
    try:
        if args.verify_only:
            audit_path = args.audit or (args.output_dir / AUDIT_NAME)
            result = verify_partition(audit_path)
        else:
            if args.audit is not None:
                raise SemanticTNPartitionError("--audit is only valid with --verify-only")
            result = create_partition(
                source_path=args.input,
                strict2031_path=args.strict2031,
                strict1607_path=args.strict1607,
                output_dir=args.output_dir,
                seed=args.seed,
                calibration_ratio=args.calibration_ratio,
                expected_input_sha256=args.expected_input_sha256,
                expected_strict2031_sha256=args.expected_strict2031_sha256,
                expected_strict1607_sha256=args.expected_strict1607_sha256,
            )
    except SemanticTNPartitionError as error:
        raise SystemExit(f"[ERROR] {error}") from error
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
