#!/usr/bin/env python3
"""Derive audit-bound paired calibration manifests for U2-v5 D rows."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "pivot.stageb.u2v5_ablation_calibration_binding/v1"
EVAL_SPLIT = "u2v5_ablation_calibration"


class CalibrationError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    path = path.resolve(strict=True)
    result = {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha256(path)}
    if rows is not None:
        result["rows"] = rows
    return result


def _read(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise CalibrationError(f"invalid calibration source: {path}")
    return rows


def _audit_output_key(table_id: str) -> str:
    return f"{table_id.lower()}_calibration"


def _validate_audit(audit_path: Path, source_path: Path, table_id: str) -> dict[str, Any]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(audit, Mapping):
        raise CalibrationError("calibration audit is not an object")
    output = audit.get("outputs", {}).get(_audit_output_key(table_id))
    if not isinstance(output, Mapping):
        raise CalibrationError(f"audit does not bind {table_id} calibration")
    observed = _record(source_path, rows=sum(1 for line in source_path.open("r", encoding="utf-8") if line.strip()))
    for key in ("path", "sha256", "rows"):
        declared = output.get(key)
        if key == "path":
            if Path(str(declared)).resolve() != source_path.resolve():
                raise CalibrationError("audit calibration path drifted")
        elif declared != observed[key]:
            raise CalibrationError(f"audit calibration {key} drifted")
    invariants = audit.get("invariants", {})
    if table_id in {"D1", "D2", "D3"}:
        if not (
            invariants.get("scope_preserved_without_global_upgrade") is True
            and invariants.get("strict_union_overlap_D1_D3", {}).get(
                _audit_output_key(table_id)
            ) == 0
            and invariants.get("train_calibration_image_overlap", {}).get(
                table_id.lower()
            ) == 0
        ):
            raise CalibrationError("broad calibration leakage invariants failed")
    elif not (
        invariants.get("strict_union_image_overlap") == 0
        and invariants.get("train_calibration_image_overlap") == 0
    ):
        raise CalibrationError("matched calibration leakage invariants failed")
    return dict(audit)


def _derive(row: Mapping[str, Any], *, index: int, data_root: Path, audit_sha: str, table_id: str, scope: str) -> dict[str, Any]:
    required = (
        "sample_id", "image_id", "ann_id", "ref_id", "sent_id", "file_name",
        "target_bbox_used", "class_id", "sent", "try_tn", "category_name",
    )
    if any(key not in row for key in required):
        raise CalibrationError(f"row {index} lacks required fields")
    if row.get("table_b_id") != table_id or row.get("tn_scope") != scope:
        raise CalibrationError(f"row {index} scope/table drifted")
    if row.get("proposalset_proxy_verified") is not False:
        raise CalibrationError(f"row {index} is a proposal proxy")
    bbox = row["target_bbox_used"]
    if not isinstance(bbox, list) or len(bbox) != 4 or float(bbox[2]) <= 0 or float(bbox[3]) <= 0:
        raise CalibrationError(f"row {index} has invalid bbox")
    positive, negative = str(row["sent"]).strip(), str(row["try_tn"]).strip()
    canonical = row.get("class_norm_name") or row.get("category_name") or row.get("try_tn_head")
    instance = {
        "bbox": [float(value) for value in bbox],
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
        "tn_scope": scope,
        "table_b_audit_sha256": audit_sha,
    }
    derived = dict(row)
    derived.update(
        {
            "filename": str(
                (data_root / "COCO/coco2014/train2014" / Path(str(row["file_name"])).name).resolve()
            ),
            "tn_eval_split": EVAL_SPLIT,
            "tn_eval_pair_source": str(row.get("pair_source", "")),
            "tn_eval_source_split": "sealed_image_disjoint_calibration",
            "table_b_audit_sha256": audit_sha,
            "instances": [instance],
        }
    )
    return derived


@dataclass(frozen=True)
class Binding:
    path: Path
    source: dict[str, Any]
    audit: dict[str, Any]
    derived: dict[str, Any]
    table_id: str
    scope: str


def build_manifest(*, source_path: Path, audit_path: Path, derived_path: Path, data_root: Path, table_id: str, scope: str) -> Binding:
    source_path = source_path.resolve(strict=True)
    audit_path = audit_path.resolve(strict=True)
    derived_path = derived_path.resolve()
    data_root = data_root.resolve(strict=True)
    sidecar = Path(str(derived_path) + ".u2v5-binding.json")
    if derived_path.exists() or sidecar.exists():
        raise CalibrationError(f"calibration output must be fresh: {derived_path}")
    _validate_audit(audit_path, source_path, table_id)
    source_rows = _read(source_path)
    audit_record = _record(audit_path)
    derived_rows = [
        _derive(row, index=index, data_root=data_root, audit_sha=audit_record["sha256"], table_id=table_id, scope=scope)
        for index, row in enumerate(source_rows)
    ]
    derived_path.parent.mkdir(parents=True, exist_ok=True)
    with derived_path.open("x", encoding="ascii") as handle:
        for row in derived_rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    payload = {
        "schema": SCHEMA,
        "table_id": table_id,
        "scope": scope,
        "source": _record(source_path, rows=len(source_rows)),
        "audit": audit_record,
        "derived": _record(derived_path, rows=len(derived_rows)),
        "data_root": str(data_root),
        "source_order_preserved": True,
        "scope_upgrade_forbidden": True,
    }
    temporary = Path(str(sidecar) + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, sidecar)
    return Binding(sidecar, payload["source"], payload["audit"], payload["derived"], table_id, scope)


def meta_rows(binding: Binding) -> list[dict[str, Any]]:
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
            "category": str((row.get("replace_category") or ["unknown"])[0]),
        }
        for row in _read(Path(str(binding.derived["path"])))
    ]


def summary_fields(binding: Binding) -> dict[str, Any]:
    return {
        "u2v5_calibration_binding_schema": SCHEMA,
        "u2v5_calibration_binding_path": str(binding.path),
        "u2v5_calibration_binding_sha256": _sha256(binding.path),
        "u2v5_calibration_source_sha256": binding.source["sha256"],
        "u2v5_calibration_audit_sha256": binding.audit["sha256"],
        "u2v5_calibration_derived_sha256": binding.derived["sha256"],
        "u2v5_calibration_rows": binding.derived["rows"],
        "u2v5_calibration_table_id": binding.table_id,
        "u2v5_calibration_scope": binding.scope,
    }


__all__ = ["Binding", "CalibrationError", "EVAL_SPLIT", "SCHEMA", "build_manifest", "meta_rows", "summary_fields"]
