#!/usr/bin/env python3
"""Deterministically materialize the sealed Stage-B screen calibration set.

The paper screen uses the scope-preserving D3 single-edit calibration rows.
Those rows intentionally retain their compact training representation, while
the evaluator consumes ordinary ``patch_episode`` rows with one explicit
instance.  This module is the small, CPU-only transform and binding contract
between the two representations.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]


def _artifact_repository_root() -> Path:
    outputs_entry = REPO_ROOT / "outputs"
    if not outputs_entry.exists():
        return REPO_ROOT
    resolved_outputs = outputs_entry.resolve(strict=True)
    if not resolved_outputs.is_dir():
        raise RuntimeError("execution repository outputs entry is invalid")
    candidates = [REPO_ROOT]
    if outputs_entry.is_symlink():
        target = Path(os.readlink(outputs_entry))
        if not target.is_absolute():
            target = outputs_entry.parent / target
        if target.name == "outputs":
            candidates.append(target.parent)
    candidates.append(resolved_outputs.parent)
    for root in dict.fromkeys(candidates):
        if (
            (root / "data").is_dir()
            and (root / "outputs").resolve(strict=True) == resolved_outputs
        ):
            return root.resolve(strict=True)
    raise RuntimeError("execution repository outputs target has no data root")


ARTIFACT_REPOSITORY_ROOT = _artifact_repository_root()
SCHEMA = "pivot.stageb.screen_calibration_binding/v1"
DERIVATION_ALGORITHM = "stageb_scope_preserving_single_edit_to_eval_v2"
EVAL_SPLIT = "screen_calibration"

DEFAULT_SOURCE = (
    ARTIFACT_REPOSITORY_ROOT
    / "data/ablations/stageb_tn_table_b_equal_exposure_20260717/"
    "d3_proposal_covered_calibration.jsonl"
)
DEFAULT_SOURCE_SHA256 = (
    "6295bdf88e0851024315f95d5d582bdd435ec4e26ab1a4522d647780385ae730"
)
DEFAULT_SOURCE_ROWS = 1570
DEFAULT_SOURCE_UNIQUE_IMAGES = 868
DEFAULT_AUDIT = (
    ARTIFACT_REPOSITORY_ROOT
    / "data/ablations/stageb_tn_table_b_equal_exposure_20260717/audit.json"
)
DEFAULT_AUDIT_SHA256 = (
    "7d74d541529a3e9abfbe84b192f2d0d3608d291bf46d19263c7c06a6ccb2291d"
)


class ScreenCalibrationError(ValueError):
    """Raised when the screen-calibration transform cannot be proven."""


@dataclass(frozen=True)
class ScreenCalibrationBinding:
    path: Path
    source_manifest: Mapping[str, Any]
    source_audit: Mapping[str, Any]
    derived_manifest: Mapping[str, Any]
    data_root: Path
    image_root: Path
    eval_split: str
    row_mapping_sha256: str
    row_mapping: tuple[Mapping[str, Any], ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": int(stat.st_size),
    }
    if rows is not None:
        result["rows"] = int(rows)
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ScreenCalibrationError(
                    f"invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ScreenCalibrationError(
                    f"screen calibration row must be an object: {path}:{line_number}"
                )
            rows.append(row)
    if not rows:
        raise ScreenCalibrationError(f"screen calibration manifest is empty: {path}")
    return rows


def _canonical_json_sha256(value: Any) -> str:
    rendered = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(rendered).hexdigest()


def _nonempty(value: Any, *, field: str, index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScreenCalibrationError(
            f"screen calibration row {index} has invalid {field}"
        )
    return value.strip()


def _integer(value: Any, *, field: str, index: int) -> int:
    if isinstance(value, bool):
        raise ScreenCalibrationError(
            f"screen calibration row {index} has invalid {field}"
        )
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ScreenCalibrationError(
            f"screen calibration row {index} has invalid {field}"
        ) from exc
    return result


def validate_source_row(row: Mapping[str, Any], *, index: int) -> None:
    """Validate the exact weak-scope, certified-single-edit row contract."""

    for field in ("sample_id", "file_name", "sent", "try_tn", "pair_source"):
        _nonempty(row.get(field), field=field, index=index)
    for field in ("image_id", "ann_id", "ref_id", "sent_id", "class_id"):
        _integer(row.get(field), field=field, index=index)
    bbox = row.get("target_bbox_used")
    if not (
        isinstance(bbox, list)
        and len(bbox) == 4
        and all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in bbox
        )
        and float(bbox[2]) > 0.0
        and float(bbox[3]) > 0.0
    ):
        raise ScreenCalibrationError(
            f"screen calibration row {index} has invalid target_bbox_used"
        )
    edits = row.get("tn_edits")
    if not (
        isinstance(edits, list)
        and len(edits) == 1
        and isinstance(edits[0], Mapping)
    ):
        raise ScreenCalibrationError(
            f"screen calibration row {index} is not a certified single edit"
        )
    edit = edits[0]
    expected = {
        "replace_category": [edit.get("category")],
        "replace_from": [edit.get("replace_from")],
        "replace_to": [edit.get("replace_to")],
        "replace_span": [edit.get("replace_span")],
    }
    for field, value in expected.items():
        if row.get(field) != value:
            raise ScreenCalibrationError(
                f"screen calibration row {index} has inconsistent {field}"
            )
    if not (
        row.get("table_b_pair_schema")
        == "stage-b-paper-table-b-scope-preserving-pair-v1"
        and row.get("table_b_id") == "D3"
        and row.get("tn_scope") == "proposal_covered_verified"
        and row.get("proposal_covered_verified") is True
        and row.get("traceable_counterfactual_edit") is True
        and row.get("visual_verified_negative") is True
        and row.get("global_tn_verified") is False
        and row.get("proposalset_proxy_verified") is False
        and row.get("cached_proposal_coverage_only") is True
        and row.get("all_900_gdino_queries_verified") is False
        and row.get("global_max_label_is_semantic_extrapolation") is True
    ):
        raise ScreenCalibrationError(
            f"screen calibration row {index} changed its proposal-covered scope"
        )


def derive_row(
    row: Mapping[str, Any],
    *,
    data_root: Path,
    index: int,
    audit_sha256: str,
) -> dict[str, Any]:
    """Create the evaluator row without upgrading the row's TN scope."""

    validate_source_row(row, index=index)
    if not (
        isinstance(audit_sha256, str)
        and len(audit_sha256) == 64
        and all(character in "0123456789abcdef" for character in audit_sha256)
    ):
        raise ScreenCalibrationError(
            f"screen calibration row {index} has invalid audit SHA-256"
        )
    data_root = Path(data_root).resolve()
    image_root = data_root / "COCO/coco2014/train2014"
    positive = str(row["sent"]).strip()
    negative = str(row["try_tn"]).strip()
    canonical = (
        row.get("class_norm_name")
        or row.get("category_name")
        or row.get("try_tn_head")
    )
    instance = {
        "bbox": [float(value) for value in row["target_bbox_used"]],
        "class_id": int(row["class_id"]),
        "raw_phrase": negative,
        "phrase": negative,
        "head_phrase": canonical,
        "head": canonical,
        "canonical_name": canonical,
        "positive_phrase": positive,
        "negative_phrase": negative,
        "try_tn": negative,
        "try_tn_head": row.get("try_tn_head"),
        "try_tn_head_phrase": positive,
        "replace_from": row.get("replace_from"),
        "replace_to": row.get("replace_to"),
        "replace_category": row.get("replace_category"),
        "replace_span": row.get("replace_span"),
        "text_is_negative": True,
        "pair_source": row.get("pair_source"),
        "category_name": row.get("category_name"),
        "global_tn_verified": False,
        "proposalset_proxy_verified": False,
        "tn_scope": "proposal_covered_verified",
        "table_b_audit_sha256": audit_sha256,
    }
    derived = dict(row)
    derived.update(
        {
            "filename": str((image_root / Path(str(row["file_name"])).name).resolve()),
            "tn_eval_split": EVAL_SPLIT,
            "tn_eval_pair_source": str(row["pair_source"]),
            "tn_eval_source_split": "sealed_image_disjoint_calibration",
            "table_b_audit_sha256": audit_sha256,
            "instances": [instance],
        }
    )
    return derived


def validate_locked_source(
    source_path: Path = DEFAULT_SOURCE,
    audit_path: Path = DEFAULT_AUDIT,
) -> dict[str, Any]:
    """Validate the immutable paper screen source and its upstream audit."""

    source_path = Path(source_path).resolve(strict=True)
    audit_path = Path(audit_path).resolve(strict=True)
    source_record = _file_record(source_path)
    audit_record = _file_record(audit_path)
    if source_path != DEFAULT_SOURCE.resolve(strict=True):
        raise ScreenCalibrationError("screen calibration source path is not locked")
    if source_record["sha256"] != DEFAULT_SOURCE_SHA256:
        raise ScreenCalibrationError("screen calibration source SHA-256 mismatch")
    if audit_path != DEFAULT_AUDIT.resolve(strict=True):
        raise ScreenCalibrationError("screen calibration audit path is not locked")
    if audit_record["sha256"] != DEFAULT_AUDIT_SHA256:
        raise ScreenCalibrationError("screen calibration audit SHA-256 mismatch")
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScreenCalibrationError(f"screen calibration audit is invalid: {exc}") from exc
    if not isinstance(audit, Mapping) or audit.get("schema") != (
        "stage-b-paper-table-b-equal-exposure-v1"
    ):
        raise ScreenCalibrationError("screen calibration audit schema mismatch")
    invariants = audit.get("invariants")
    expected_invariants = (
        isinstance(invariants, Mapping)
        and invariants.get("scope_preserved_without_global_upgrade") is True
        and invariants.get("train_calibration_image_overlap", {}).get("d3") == 0
        and invariants.get("strict_union_overlap_D1_D3", {}).get("d3_calibration") == 0
        and invariants.get("runtime_global_verified_true_rows", {}).get(
            "d3_calibration"
        )
        == 0
    )
    if not expected_invariants:
        raise ScreenCalibrationError("screen calibration audit invariants are incomplete")
    declared = audit.get("outputs", {}).get("d3_calibration")
    expected_declared = {
        **source_record,
        "rows": DEFAULT_SOURCE_ROWS,
        "unique_images": DEFAULT_SOURCE_UNIQUE_IMAGES,
    }
    if not isinstance(declared, Mapping) or dict(declared) != expected_declared:
        raise ScreenCalibrationError("screen calibration audit output record mismatch")
    rows = _read_jsonl(source_path)
    if len(rows) != DEFAULT_SOURCE_ROWS:
        raise ScreenCalibrationError("screen calibration row count mismatch")
    sample_ids: set[str] = set()
    image_ids: set[int] = set()
    for index, row in enumerate(rows):
        validate_source_row(row, index=index)
        sample_id = str(row["sample_id"])
        if sample_id in sample_ids:
            raise ScreenCalibrationError(
                f"duplicate screen calibration sample_id: {sample_id}"
            )
        sample_ids.add(sample_id)
        image_ids.add(int(row["image_id"]))
    if len(image_ids) != DEFAULT_SOURCE_UNIQUE_IMAGES:
        raise ScreenCalibrationError("screen calibration unique-image count mismatch")
    return {
        "source_manifest": {**source_record, "rows": len(rows)},
        "source_audit": audit_record,
        "unique_images": len(image_ids),
        "scope": "proposal_covered_verified",
        "single_edit_provenance": True,
        "strict_union_image_overlap": 0,
        "train_calibration_image_overlap": 0,
    }


def binding_path(derived_path: Path) -> Path:
    return Path(derived_path).with_suffix(Path(derived_path).suffix + ".screen-binding.json")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="ascii",
        )
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def build_manifest(
    *,
    source_path: Path,
    audit_path: Path,
    derived_path: Path,
    data_root: Path,
) -> ScreenCalibrationBinding:
    """Materialize the full source in source order and write a sealed binding."""

    source_path = Path(source_path).resolve(strict=True)
    audit_path = Path(audit_path).resolve(strict=True)
    derived_path = Path(derived_path).resolve()
    data_root = Path(data_root).resolve(strict=True)
    if derived_path.exists() or binding_path(derived_path).exists():
        raise FileExistsError(
            f"screen calibration derived output must be fresh: {derived_path}"
        )
    source_rows = _read_jsonl(source_path)
    audit_record = _file_record(audit_path)
    derived_rows = [
        derive_row(
            row,
            data_root=data_root,
            index=index,
            audit_sha256=str(audit_record["sha256"]),
        )
        for index, row in enumerate(source_rows)
    ]
    derived_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = derived_path.with_name(derived_path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="ascii") as handle:
            for row in derived_rows:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=True,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
        os.replace(temporary, derived_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    mapping = [
        {
            "derived_index": index,
            "source_index": index,
            "sample_id": str(row["sample_id"]),
            "image_id": int(row["image_id"]),
        }
        for index, row in enumerate(source_rows)
    ]
    payload = {
        "schema": SCHEMA,
        "kind": "deterministic_full_screen_calibration_derivation",
        "derivation": {
            "algorithm": DERIVATION_ALGORITHM,
            "eval_split": EVAL_SPLIT,
            "source_order_preserved": True,
            "source_rows_selected": "all",
            "scope_upgrade_forbidden": True,
            "table_b_audit_sha256_injected": True,
        },
        "source_manifest": _file_record(source_path, rows=len(source_rows)),
        "source_audit": audit_record,
        "derived_manifest": _file_record(derived_path, rows=len(derived_rows)),
        "data_root": str(data_root),
        "image_root": str((data_root / "COCO/coco2014/train2014").resolve()),
        "row_mapping_sha256": _canonical_json_sha256(mapping),
        "row_mapping": mapping,
    }
    sidecar = binding_path(derived_path)
    _write_json_atomic(sidecar, payload)
    return load_binding(sidecar, expected_derived=derived_path)


def load_binding(
    path: Path,
    *,
    expected_derived: Path | None = None,
) -> ScreenCalibrationBinding:
    path = Path(path).resolve(strict=True)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScreenCalibrationError(f"invalid screen calibration binding: {exc}") from exc
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema",
        "kind",
        "derivation",
        "source_manifest",
        "source_audit",
        "derived_manifest",
        "data_root",
        "image_root",
        "row_mapping_sha256",
        "row_mapping",
    }:
        raise ScreenCalibrationError("screen calibration binding field set mismatch")
    if payload.get("schema") != SCHEMA or payload.get("kind") != (
        "deterministic_full_screen_calibration_derivation"
    ):
        raise ScreenCalibrationError("screen calibration binding schema mismatch")
    expected_derivation = {
        "algorithm": DERIVATION_ALGORITHM,
        "eval_split": EVAL_SPLIT,
        "source_order_preserved": True,
        "source_rows_selected": "all",
        "scope_upgrade_forbidden": True,
        "table_b_audit_sha256_injected": True,
    }
    if payload.get("derivation") != expected_derivation:
        raise ScreenCalibrationError("screen calibration derivation contract mismatch")
    source_record = payload["source_manifest"]
    audit_record = payload["source_audit"]
    derived_record = payload["derived_manifest"]
    if not all(
        isinstance(record, Mapping)
        for record in (source_record, audit_record, derived_record)
    ):
        raise ScreenCalibrationError("screen calibration binding lacks file records")
    source_path = Path(str(source_record.get("path", ""))).resolve(strict=True)
    audit_path = Path(str(audit_record.get("path", ""))).resolve(strict=True)
    derived_path = Path(str(derived_record.get("path", ""))).resolve(strict=True)
    if expected_derived is not None and derived_path != Path(expected_derived).resolve(
        strict=True
    ):
        raise ScreenCalibrationError("screen calibration binding points elsewhere")
    source_rows = _read_jsonl(source_path)
    derived_rows = _read_jsonl(derived_path)
    expected_records = (
        (source_record, _file_record(source_path, rows=len(source_rows))),
        (audit_record, _file_record(audit_path)),
        (derived_record, _file_record(derived_path, rows=len(derived_rows))),
    )
    for declared, observed in expected_records:
        if dict(declared) != observed:
            raise ScreenCalibrationError("screen calibration bound file changed")
    if len(source_rows) != len(derived_rows):
        raise ScreenCalibrationError("screen calibration derivation dropped rows")
    data_root = Path(str(payload.get("data_root", ""))).resolve(strict=True)
    image_root = Path(str(payload.get("image_root", ""))).resolve()
    if image_root != (data_root / "COCO/coco2014/train2014").resolve():
        raise ScreenCalibrationError("screen calibration image root mismatch")
    mapping = payload.get("row_mapping")
    if not isinstance(mapping, list) or len(mapping) != len(source_rows):
        raise ScreenCalibrationError("screen calibration row mapping length mismatch")
    if payload.get("row_mapping_sha256") != _canonical_json_sha256(mapping):
        raise ScreenCalibrationError("screen calibration row mapping hash mismatch")
    for index, (source, derived, row_mapping) in enumerate(
        zip(source_rows, derived_rows, mapping)
    ):
        if not isinstance(row_mapping, Mapping) or dict(row_mapping) != {
            "derived_index": index,
            "source_index": index,
            "sample_id": str(source.get("sample_id", "")),
            "image_id": int(source.get("image_id", -1)),
        }:
            raise ScreenCalibrationError(
                f"screen calibration row mapping drift at index {index}"
            )
        if derived != derive_row(
            source,
            data_root=data_root,
            index=index,
            audit_sha256=str(audit_record["sha256"]),
        ):
            raise ScreenCalibrationError(
                f"screen calibration derived row drift at index {index}"
            )
    return ScreenCalibrationBinding(
        path=path,
        source_manifest=dict(source_record),
        source_audit=dict(audit_record),
        derived_manifest=dict(derived_record),
        data_root=data_root,
        image_root=image_root,
        eval_split=EVAL_SPLIT,
        row_mapping_sha256=str(payload["row_mapping_sha256"]),
        row_mapping=tuple(dict(value) for value in mapping),
    )


def meta_rows(binding: ScreenCalibrationBinding) -> list[dict[str, Any]]:
    rows = _read_jsonl(Path(str(binding.derived_manifest["path"])))
    return [
        {
            "eval_split": EVAL_SPLIT,
            "pair_source": row.get("pair_source"),
            "source_split": "sealed_image_disjoint_calibration",
            "image_id": int(row["image_id"]),
            "ann_id": int(row["ann_id"]),
            "ref_id": int(row["ref_id"]),
            "sent_id": int(row["sent_id"]),
            "negative_phrase": row["try_tn"],
            "positive_phrase": row["sent"],
            "category": str(row["replace_category"][0]),
        }
        for row in rows
    ]


def summary_fields(binding: ScreenCalibrationBinding) -> dict[str, Any]:
    return {
        "screen_calibration_binding_schema": SCHEMA,
        "screen_calibration_derivation_algorithm": DERIVATION_ALGORITHM,
        "screen_calibration_binding_path": str(binding.path),
        "screen_calibration_binding_sha256": sha256_file(binding.path),
        "screen_calibration_source_path": str(binding.source_manifest["path"]),
        "screen_calibration_source_sha256": str(
            binding.source_manifest["sha256"]
        ),
        "screen_calibration_source_n": int(binding.source_manifest["rows"]),
        "screen_calibration_audit_path": str(binding.source_audit["path"]),
        "screen_calibration_audit_sha256": str(binding.source_audit["sha256"]),
        "screen_calibration_derived_path": str(binding.derived_manifest["path"]),
        "screen_calibration_derived_sha256": str(
            binding.derived_manifest["sha256"]
        ),
        "screen_calibration_row_mapping_sha256": binding.row_mapping_sha256,
        "screen_calibration_scope": "proposal_covered_verified",
        "screen_calibration_single_edit": True,
    }


def iter_rows(path: Path) -> Iterable[dict[str, Any]]:
    """Expose the strict reader for postflight code without duplicating it."""

    return iter(_read_jsonl(path))
