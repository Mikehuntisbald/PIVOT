#!/usr/bin/env python3
"""Build the canonical proposal-covered Table-B matched evaluation surface."""

from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.build_stageb_tn_matched_causal_panel import (  # noqa: E402
    MatchedPanelError,
    PAIR_SCHEMA,
    SCHEMA as AUDIT_SCHEMA,
    verify_panel,
)


SCHEMA = "pivot.stageb.table_b_matched_eval_surface_binding/v1"
SURFACE_ROW_SCHEMA = "pivot.stageb.table_b_matched_eval_surface_row/v1"
DERIVATION_ALGORITHM = "stageb_d3m_matched_calibration_to_eval_surface_v1"
EVAL_SPLIT = "matched_calibration"
DECLARED_SCOPE = "proposal_covered_verified"
SUPPORT_POOL_ALGORITHM = (
    "patch_episode_clean_image_bank_relevant_classes_seed42_reservoir_v1"
)
SUPPORT_BANK_SEED = 42
SUPPORT_PATCH_MAX_PER_CLASS = 200
SUPPORT_PATCH_BUCKET = "clean"

DEFAULT_ROOT = (
    REPO_ROOT
    / "data/ablations/stageb_tn_c2_parent_matched_class_aligned_20260718_v2"
)
DEFAULT_AUDIT = DEFAULT_ROOT / "audit.json"
DEFAULT_LEDGER = DEFAULT_ROOT / "matched_pairs_calibration.jsonl"
DEFAULT_D3M_SOURCE = DEFAULT_ROOT / "d3m_proposal_covered_calibration.jsonl"


class MatchedEvalSurfaceError(ValueError):
    """Raised when the matched evaluation surface cannot be proven."""


@dataclass(frozen=True)
class MatchedEvalSurfaceBinding:
    path: Path
    source_manifest: Mapping[str, Any]
    source_audit: Mapping[str, Any]
    pair_ledger: Mapping[str, Any]
    derived_manifest: Mapping[str, Any]
    data_root: Path
    image_root: Path
    image_files_sha256: str
    image_files: tuple[Mapping[str, Any], ...]
    support_pool_mapping_sha256: str
    support_pool_mapping: tuple[Mapping[str, Any], ...]
    support_pool_files_sha256: str
    support_pool_files: tuple[Mapping[str, Any], ...]
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
    result: dict[str, Any] = {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }
    if rows is not None:
        result["rows"] = int(rows)
    return result


def _read_json(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MatchedEvalSurfaceError(f"{label}: invalid JSON: {error}") from error
    if not isinstance(value, Mapping):
        raise MatchedEvalSurfaceError(f"{label}: expected an object")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise MatchedEvalSurfaceError(f"{label}: invalid JSONL: {error}") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise MatchedEvalSurfaceError(
                f"{label}:{line_number}: blank rows are forbidden"
            )
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise MatchedEvalSurfaceError(
                f"{label}:{line_number}: invalid JSON: {error}"
            ) from error
        if not isinstance(row, dict):
            raise MatchedEvalSurfaceError(
                f"{label}:{line_number}: expected an object"
            )
        rows.append(row)
    if not rows:
        raise MatchedEvalSurfaceError(f"{label}: empty JSONL")
    return rows


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _exact_int(value: Any, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise MatchedEvalSurfaceError(f"{label}: expected a non-negative integer")
    return value


def _nonempty(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MatchedEvalSurfaceError(f"{label}: expected a non-empty string")
    return value.strip()


def _sha(value: Any, *, label: str) -> str:
    rendered = str(value or "")
    if len(rendered) != 64 or any(ch not in "0123456789abcdef" for ch in rendered):
        raise MatchedEvalSurfaceError(f"{label}: invalid SHA-256")
    return rendered


def _audited_output(
    audit: Mapping[str, Any],
    *,
    audit_path: Path,
    key: str,
    supplied: Path,
) -> dict[str, Any]:
    outputs = audit.get("outputs")
    declared = outputs.get(key) if isinstance(outputs, Mapping) else None
    if not isinstance(declared, Mapping):
        raise MatchedEvalSurfaceError(f"audit.outputs.{key} is missing")
    declared_path = Path(str(declared.get("path", ""))).expanduser()
    if not declared_path.is_absolute():
        declared_path = audit_path.parent / declared_path
    declared_path = declared_path.resolve(strict=True)
    supplied = Path(supplied).resolve(strict=True)
    if declared_path != supplied:
        raise MatchedEvalSurfaceError(f"{key}: path is not sealed by the audit")
    rows = _exact_int(declared.get("rows"), label=f"audit.outputs.{key}.rows")
    observed = _file_record(supplied, rows=rows)
    if any(observed.get(field) != declared.get(field) for field in observed):
        raise MatchedEvalSurfaceError(f"{key}: audited file identity drift")
    if "unique_images" in declared:
        unique_images = len(
            {
                _exact_int(row.get("image_id"), label=f"{key}.image_id")
                for row in _read_jsonl(supplied, label=key)
            }
        )
        if declared.get("unique_images") != unique_images:
            raise MatchedEvalSurfaceError(f"{key}: audited unique-image count drift")
    return dict(declared)


def _validate_source_row(row: Mapping[str, Any], *, index: int) -> None:
    prefix = f"D3m calibration row {index}"
    for field in (
        "sample_id",
        "file_name",
        "sent",
        "try_tn",
        "pair_source",
        "matched_pair_id",
        "matched_parent_key_sha256",
    ):
        _nonempty(row.get(field), label=f"{prefix}.{field}")
    for field in ("image_id", "ann_id", "ref_id", "sent_id", "class_id"):
        _exact_int(row.get(field), label=f"{prefix}.{field}")
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
        raise MatchedEvalSurfaceError(f"{prefix}.target_bbox_used is invalid")
    edits = row.get("tn_edits")
    if not (
        isinstance(edits, list)
        and len(edits) == 1
        and isinstance(edits[0], Mapping)
    ):
        raise MatchedEvalSurfaceError(f"{prefix} is not a single-edit row")
    edit = edits[0]
    expected_edit_fields = {
        "replace_category": [edit.get("category")],
        "replace_from": [edit.get("replace_from")],
        "replace_to": [edit.get("replace_to")],
        "replace_span": [edit.get("replace_span")],
    }
    if any(row.get(field) != value for field, value in expected_edit_fields.items()):
        raise MatchedEvalSurfaceError(f"{prefix} edit metadata is inconsistent")
    if not (
        row.get("matched_pair_schema") == PAIR_SCHEMA
        and row.get("matched_split") == "calibration"
        and row.get("table_b_id") == "D3m"
        and row.get("tn_scope") == DECLARED_SCOPE
        and row.get("proposal_covered_verified") is True
        and row.get("traceable_counterfactual_edit") is True
        and row.get("visual_verified_negative") is True
        and row.get("global_tn_verified") is False
        and row.get("proposalset_proxy_verified") is False
        and row.get("cached_proposal_coverage_only") is True
        and row.get("all_900_gdino_queries_verified") is False
        and row.get("global_max_label_is_semantic_extrapolation") is True
    ):
        raise MatchedEvalSurfaceError(f"{prefix} changed proposal-covered scope")
    _sha(
        row.get("matched_parent_key_sha256"),
        label=f"{prefix}.matched_parent_key_sha256",
    )
    if type(row.get("canonical_class_id_match")) is not bool:
        raise MatchedEvalSurfaceError(f"{prefix} lacks canonical-class accounting")


def _validate_pair_alignment(
    source: Mapping[str, Any], ledger: Mapping[str, Any], *, index: int
) -> None:
    prefix = f"matched calibration row {index}"
    if not (
        ledger.get("matched_pair_schema") == PAIR_SCHEMA
        and ledger.get("matched_split") == "calibration"
    ):
        raise MatchedEvalSurfaceError(f"{prefix}: ledger pair schema/split mismatch")
    side = ledger.get("d3m")
    if not isinstance(side, Mapping):
        raise MatchedEvalSurfaceError(f"{prefix}: ledger lacks D3m side")
    exact_fields = (
        "matched_pair_id",
        "matched_parent_key",
        "matched_parent_key_sha256",
        "matched_stratum",
        "d2_canonical_class_id",
        "d3_canonical_class_id",
        "canonical_class_id_match",
        "model_input_component_exact_matches",
        "base_parent_input_exact_match",
        "complete_model_input_exact_match",
        "class_aligned_identical_complete_input",
    )
    for field in exact_fields:
        if source.get(field) != ledger.get(field):
            raise MatchedEvalSurfaceError(f"{prefix}: {field} alignment drift")
    if not (
        source.get("sample_id") == side.get("sample_id")
        and source.get("class_id") == side.get("class_id")
        and source.get("file_name") == side.get("file_name")
        and source.get("target_bbox_used") == side.get("target_bbox_used")
        and source.get("sent") == side.get("sent")
        and source.get("try_tn") == side.get("try_tn")
        and side.get("tn_scope") == DECLARED_SCOPE
        and source.get("image_id") == ledger.get("image_id")
        and source.get("sent_id") == ledger.get("sent_id")
    ):
        raise MatchedEvalSurfaceError(f"{prefix}: D3m side identity drift")


def _query_image_files(
    rows: Sequence[Mapping[str, Any]], *, data_root: Path
) -> list[dict[str, Any]]:
    image_root = (Path(data_root) / "COCO/coco2014/train2014").resolve(strict=True)
    by_image_id: dict[int, tuple[str, Path]] = {}
    by_path: dict[Path, int] = {}
    for index, row in enumerate(rows):
        raw_name = _nonempty(row.get("file_name"), label=f"D3m row {index}.file_name")
        name_path = Path(raw_name)
        if name_path.is_absolute() or name_path.name != raw_name:
            raise MatchedEvalSurfaceError(
                f"D3m row {index}.file_name is not one canonical basename"
            )
        image_id = _exact_int(row.get("image_id"), label=f"D3m row {index}.image_id")
        path = (image_root / raw_name).resolve(strict=True)
        try:
            path.relative_to(image_root)
        except ValueError as error:
            raise MatchedEvalSurfaceError(
                f"D3m row {index} image escaped the canonical image root"
            ) from error
        previous = by_image_id.get(image_id)
        if previous is not None and previous != (raw_name, path):
            raise MatchedEvalSurfaceError(
                f"image_id {image_id} maps to multiple canonical image files"
            )
        previous_id = by_path.get(path)
        if previous_id is not None and previous_id != image_id:
            raise MatchedEvalSurfaceError(
                f"canonical image file {path} maps to multiple image IDs"
            )
        by_image_id[image_id] = (raw_name, path)
        by_path[path] = image_id
    return [
        {
            **_file_record(path),
            "image_id": image_id,
            "file_name": raw_name,
        }
        for image_id, (raw_name, path) in sorted(by_image_id.items())
    ]


def _support_pool_contract(
    rows: Sequence[Mapping[str, Any]], *, data_root: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Replay PatchEpisode's exact clean-image bank for relevant class IDs."""

    from datasets.patch_episode import (  # imported lazily for summary-only tools
        PatchEpisodeJsonlDataset,
        _build_name_to_canonical_id,
    )

    data_root = Path(data_root).resolve(strict=True)
    canonical_path = (data_root / "canonical_classes_with_aliases.json").resolve(
        strict=True
    )
    tsv_path = (
        data_root / "patches_quality_emb/emb_index_from_quality.tsv"
    ).resolve(strict=True)
    patch_root = (data_root / "patches_quality").resolve(strict=True)
    shim = SimpleNamespace(
        cfg=SimpleNamespace(
            support_patch_use_embedding=False,
            support_patch_max_per_class=SUPPORT_PATCH_MAX_PER_CLASS,
            support_patch_bucket=SUPPORT_PATCH_BUCKET,
            support_patch_image_root=str(patch_root),
        ),
        patch_class_map=None,
        name2cid=_build_name_to_canonical_id(str(canonical_path)),
    )
    random_state = random.getstate()
    try:
        random.seed(SUPPORT_BANK_SEED)
        bank = PatchEpisodeJsonlDataset._load_patch_bank_fast(shim, tsv_path)
    finally:
        random.setstate(random_state)
    relevant_classes = sorted(
        {
            _exact_int(row.get("class_id"), label=f"D3m row {index}.class_id")
            for index, row in enumerate(rows)
        }
    )
    mapping: list[dict[str, Any]] = []
    file_classes: dict[Path, set[int]] = {}
    for class_id in relevant_classes:
        raw_candidates = bank.get(class_id, [])
        if not raw_candidates:
            raise MatchedEvalSurfaceError(
                f"support patch bank has no clean-image candidates for class {class_id}"
            )
        candidates: list[str] = []
        for raw_path in raw_candidates:
            path = Path(str(raw_path)).expanduser().resolve(strict=True)
            candidates.append(str(path))
            file_classes.setdefault(path, set()).add(class_id)
        if len(candidates) > SUPPORT_PATCH_MAX_PER_CLASS:
            raise MatchedEvalSurfaceError(
                f"support patch bank exceeded max-per-class for class {class_id}"
            )
        mapping.append({"class_id": class_id, "candidate_paths": candidates})
    files = [
        {**_file_record(path), "class_ids": sorted(class_ids)}
        for path, class_ids in sorted(file_classes.items(), key=lambda item: str(item[0]))
    ]
    return mapping, files


def derive_row(
    source: Mapping[str, Any],
    ledger: Mapping[str, Any],
    *,
    data_root: Path,
    audit_sha256: str,
    index: int,
) -> dict[str, Any]:
    """Derive one evaluator row while preserving the proposal-covered scope."""

    _validate_source_row(source, index=index)
    _validate_pair_alignment(source, ledger, index=index)
    audit_sha256 = _sha(audit_sha256, label="matched audit SHA-256")
    data_root = Path(data_root).resolve(strict=True)
    image_root = data_root / "COCO/coco2014/train2014"
    positive = str(source["sent"]).strip()
    negative = str(source["try_tn"]).strip()
    canonical = _nonempty(
        source.get("class_norm_name")
        or source.get("category_name")
        or source.get("try_tn_head"),
        label=f"D3m calibration row {index}.canonical_name",
    )
    pair_fields = {
        field: source[field]
        for field in (
            "matched_pair_schema",
            "matched_pair_id",
            "matched_split",
            "matched_parent_key",
            "matched_parent_key_sha256",
            "matched_stratum",
            "d2_canonical_class_id",
            "d3_canonical_class_id",
            "canonical_class_id_match",
            "complete_model_input_exact_match",
            "class_aligned_identical_complete_input",
        )
    }
    instance = {
        "bbox": [float(value) for value in source["target_bbox_used"]],
        "class_id": int(source["class_id"]),
        "raw_phrase": negative,
        "phrase": negative,
        "head_phrase": canonical,
        "head": canonical,
        "canonical_name": canonical,
        "positive_phrase": positive,
        "negative_phrase": negative,
        "try_tn": negative,
        "try_tn_head": source.get("try_tn_head"),
        "try_tn_head_phrase": positive,
        "replace_from": source.get("replace_from"),
        "replace_to": source.get("replace_to"),
        "replace_category": source.get("replace_category"),
        "replace_span": source.get("replace_span"),
        "text_is_negative": True,
        "pair_source": source.get("pair_source"),
        "category_name": source.get("category_name"),
        "global_tn_verified": False,
        "proposalset_proxy_verified": False,
        "tn_scope": DECLARED_SCOPE,
        "table_b_id": "D3m",
        "table_b_audit_sha256": audit_sha256,
        **pair_fields,
    }
    derived = dict(source)
    derived.update(
        {
            "matched_eval_surface_schema": SURFACE_ROW_SCHEMA,
            "filename": str(
                (image_root / Path(str(source["file_name"])).name).resolve()
            ),
            "tn_eval_split": EVAL_SPLIT,
            "tn_eval_pair_source": str(source["pair_source"]),
            "tn_eval_source_split": "sealed_parent_matched_calibration",
            "table_b_audit_sha256": audit_sha256,
            "instances": [instance],
        }
    )
    return derived


def binding_path(derived_path: Path) -> Path:
    path = Path(derived_path)
    return path.with_suffix(path.suffix + ".matched-binding.json")


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n",
            encoding="ascii",
        )
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="ascii") as handle:
            for row in rows:
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
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def build_surface(
    *,
    audit_path: Path = DEFAULT_AUDIT,
    ledger_path: Path = DEFAULT_LEDGER,
    source_path: Path = DEFAULT_D3M_SOURCE,
    derived_path: Path,
    data_root: Path,
) -> MatchedEvalSurfaceBinding:
    audit_path = Path(audit_path).resolve(strict=True)
    ledger_path = Path(ledger_path).resolve(strict=True)
    source_path = Path(source_path).resolve(strict=True)
    derived_path = Path(derived_path).resolve()
    data_root = Path(data_root).resolve(strict=True)
    sidecar = binding_path(derived_path)
    if derived_path.exists() or sidecar.exists():
        raise FileExistsError(f"matched evaluation surface must be fresh: {derived_path}")
    try:
        audit = verify_panel(audit_path)
    except (OSError, KeyError, TypeError, MatchedPanelError) as error:
        raise MatchedEvalSurfaceError(f"matched v2 audit verification failed: {error}") from error
    if audit.get("schema") != AUDIT_SCHEMA:
        raise MatchedEvalSurfaceError("matched audit schema mismatch")
    source_record = _audited_output(
        audit,
        audit_path=audit_path,
        key="d3m_calibration",
        supplied=source_path,
    )
    ledger_record = _audited_output(
        audit,
        audit_path=audit_path,
        key="pairs_calibration",
        supplied=ledger_path,
    )
    audit_record = _file_record(audit_path)
    source_rows = _read_jsonl(source_path, label="D3m calibration source")
    ledger_rows = _read_jsonl(ledger_path, label="matched calibration ledger")
    if not (
        len(source_rows)
        == len(ledger_rows)
        == int(source_record["rows"])
        == int(ledger_record["rows"])
    ):
        raise MatchedEvalSurfaceError("source/ledger row counts differ")
    derived_rows = [
        derive_row(
            source,
            ledger,
            data_root=data_root,
            audit_sha256=str(audit_record["sha256"]),
            index=index,
        )
        for index, (source, ledger) in enumerate(zip(source_rows, ledger_rows))
    ]
    image_files = _query_image_files(source_rows, data_root=data_root)
    if source_record.get("unique_images") != len(image_files):
        raise MatchedEvalSurfaceError(
            "canonical query-image count differs from the audited D3m source"
        )
    support_pool_mapping, support_pool_files = _support_pool_contract(
        source_rows, data_root=data_root
    )
    _write_jsonl_atomic(derived_path, derived_rows)
    mapping = [
        {
            "source_index": index,
            "ledger_index": index,
            "derived_index": index,
            "sample_id": str(source["sample_id"]),
            "image_id": int(source["image_id"]),
            "matched_pair_id": str(source["matched_pair_id"]),
            "matched_parent_key_sha256": str(
                source["matched_parent_key_sha256"]
            ),
        }
        for index, source in enumerate(source_rows)
    ]
    payload = {
        "schema": SCHEMA,
        "kind": "deterministic_full_d3m_matched_calibration_derivation",
        "derivation": {
            "algorithm": DERIVATION_ALGORITHM,
            "eval_split": EVAL_SPLIT,
            "source_order_preserved": True,
            "source_rows_selected": "all_matched_calibration_rows",
            "scope_upgrade_forbidden": True,
            "table_b_audit_sha256_injected_into_row_and_instance": True,
            "pair_provenance_preserved": True,
        },
        "source_manifest": source_record,
        "source_audit": audit_record,
        "pair_ledger": ledger_record,
        "derived_manifest": _file_record(derived_path, rows=len(derived_rows)),
        "data_root": str(data_root),
        "image_root": str(
            (data_root / "COCO/coco2014/train2014").resolve()
        ),
        "image_files_sha256": _canonical_sha256(image_files),
        "image_files": image_files,
        "support_pool": {
            "algorithm": SUPPORT_POOL_ALGORITHM,
            "bucket": SUPPORT_PATCH_BUCKET,
            "max_per_class": SUPPORT_PATCH_MAX_PER_CLASS,
            "seed": SUPPORT_BANK_SEED,
            "cache_enabled": False,
            "cache_write_enabled": False,
            "mapping_sha256": _canonical_sha256(support_pool_mapping),
            "mapping": support_pool_mapping,
            "files_sha256": _canonical_sha256(support_pool_files),
            "files": support_pool_files,
        },
        "row_mapping_sha256": _canonical_sha256(mapping),
        "row_mapping": mapping,
    }
    _write_json_atomic(sidecar, payload)
    return load_binding(sidecar, expected_derived=derived_path)


def load_binding(
    path: Path, *, expected_derived: Path | None = None
) -> MatchedEvalSurfaceBinding:
    path = Path(path).resolve(strict=True)
    payload = _read_json(path, label="matched evaluation surface binding")
    expected_fields = {
        "schema",
        "kind",
        "derivation",
        "source_manifest",
        "source_audit",
        "pair_ledger",
        "derived_manifest",
        "data_root",
        "image_root",
        "image_files_sha256",
        "image_files",
        "support_pool",
        "row_mapping_sha256",
        "row_mapping",
    }
    if set(payload) != expected_fields:
        raise MatchedEvalSurfaceError("matched surface binding field set mismatch")
    if not (
        payload.get("schema") == SCHEMA
        and payload.get("kind")
        == "deterministic_full_d3m_matched_calibration_derivation"
    ):
        raise MatchedEvalSurfaceError("matched surface binding schema mismatch")
    expected_derivation = {
        "algorithm": DERIVATION_ALGORITHM,
        "eval_split": EVAL_SPLIT,
        "source_order_preserved": True,
        "source_rows_selected": "all_matched_calibration_rows",
        "scope_upgrade_forbidden": True,
        "table_b_audit_sha256_injected_into_row_and_instance": True,
        "pair_provenance_preserved": True,
    }
    if payload.get("derivation") != expected_derivation:
        raise MatchedEvalSurfaceError("matched surface derivation contract drift")
    records = {}
    for key in (
        "source_manifest",
        "source_audit",
        "pair_ledger",
        "derived_manifest",
    ):
        declared = payload.get(key)
        if not isinstance(declared, Mapping):
            raise MatchedEvalSurfaceError(f"binding lacks {key} file record")
        file_path = Path(str(declared.get("path", ""))).resolve(strict=True)
        rows = None if key == "source_audit" else len(
            _read_jsonl(file_path, label=key)
        )
        observed = _file_record(file_path, rows=rows)
        if any(declared.get(field) != value for field, value in observed.items()):
            raise MatchedEvalSurfaceError(f"bound {key} changed")
        if "unique_images" in declared:
            file_rows = _read_jsonl(file_path, label=key)
            if declared.get("unique_images") != len(
                {int(row["image_id"]) for row in file_rows}
            ):
                raise MatchedEvalSurfaceError(f"bound {key} unique images changed")
        records[key] = dict(declared)
    derived_path = Path(str(records["derived_manifest"]["path"]))
    if expected_derived is not None and derived_path != Path(expected_derived).resolve(
        strict=True
    ):
        raise MatchedEvalSurfaceError("binding points to another derived manifest")
    audit_path = Path(str(records["source_audit"]["path"]))
    try:
        audit = verify_panel(audit_path)
    except (OSError, KeyError, TypeError, MatchedPanelError) as error:
        raise MatchedEvalSurfaceError(f"bound audit verification failed: {error}") from error
    if dict(audit.get("outputs", {}).get("d3m_calibration", {})) != dict(
        records["source_manifest"]
    ):
        raise MatchedEvalSurfaceError("binding source is not audited D3m calibration")
    if dict(audit.get("outputs", {}).get("pairs_calibration", {})) != dict(
        records["pair_ledger"]
    ):
        raise MatchedEvalSurfaceError("binding ledger is not audited calibration")
    source_rows = _read_jsonl(
        Path(str(records["source_manifest"]["path"])), label="bound D3m source"
    )
    ledger_rows = _read_jsonl(
        Path(str(records["pair_ledger"]["path"])), label="bound pair ledger"
    )
    derived_rows = _read_jsonl(derived_path, label="bound derived surface")
    if not len(source_rows) == len(ledger_rows) == len(derived_rows):
        raise MatchedEvalSurfaceError("binding dropped source or ledger rows")
    data_root = Path(str(payload.get("data_root", ""))).resolve(strict=True)
    image_root = Path(str(payload.get("image_root", ""))).resolve()
    if image_root != (data_root / "COCO/coco2014/train2014").resolve():
        raise MatchedEvalSurfaceError("binding image root mismatch")
    image_files = payload.get("image_files")
    expected_image_files = _query_image_files(source_rows, data_root=data_root)
    if not isinstance(image_files, list) or image_files != expected_image_files:
        raise MatchedEvalSurfaceError("binding canonical query-image files drifted")
    if payload.get("image_files_sha256") != _canonical_sha256(image_files):
        raise MatchedEvalSurfaceError("binding canonical query-image digest drifted")
    if records["source_manifest"].get("unique_images") != len(image_files):
        raise MatchedEvalSurfaceError("binding canonical query-image count drifted")
    support_pool = payload.get("support_pool")
    expected_support_mapping, expected_support_files = _support_pool_contract(
        source_rows, data_root=data_root
    )
    if not isinstance(support_pool, Mapping):
        raise MatchedEvalSurfaceError("binding support pool is missing")
    expected_support_header = {
        "algorithm": SUPPORT_POOL_ALGORITHM,
        "bucket": SUPPORT_PATCH_BUCKET,
        "max_per_class": SUPPORT_PATCH_MAX_PER_CLASS,
        "seed": SUPPORT_BANK_SEED,
        "cache_enabled": False,
        "cache_write_enabled": False,
    }
    if any(support_pool.get(key) != value for key, value in expected_support_header.items()):
        raise MatchedEvalSurfaceError("binding support-pool contract drifted")
    if support_pool.get("mapping") != expected_support_mapping or support_pool.get(
        "mapping_sha256"
    ) != _canonical_sha256(expected_support_mapping):
        raise MatchedEvalSurfaceError("binding support-pool mapping drifted")
    if support_pool.get("files") != expected_support_files or support_pool.get(
        "files_sha256"
    ) != _canonical_sha256(expected_support_files):
        raise MatchedEvalSurfaceError("binding support-pool file identities drifted")
    mapping = payload.get("row_mapping")
    if not isinstance(mapping, list) or len(mapping) != len(source_rows):
        raise MatchedEvalSurfaceError("binding row mapping length mismatch")
    if payload.get("row_mapping_sha256") != _canonical_sha256(mapping):
        raise MatchedEvalSurfaceError("binding row mapping digest mismatch")
    audit_sha = str(records["source_audit"]["sha256"])
    for index, (source, ledger, derived, mapped) in enumerate(
        zip(source_rows, ledger_rows, derived_rows, mapping)
    ):
        expected_mapping = {
            "source_index": index,
            "ledger_index": index,
            "derived_index": index,
            "sample_id": str(source["sample_id"]),
            "image_id": int(source["image_id"]),
            "matched_pair_id": str(source["matched_pair_id"]),
            "matched_parent_key_sha256": str(
                source["matched_parent_key_sha256"]
            ),
        }
        if not isinstance(mapped, Mapping) or dict(mapped) != expected_mapping:
            raise MatchedEvalSurfaceError(f"row mapping drift at index {index}")
        expected_row = derive_row(
            source,
            ledger,
            data_root=data_root,
            audit_sha256=audit_sha,
            index=index,
        )
        if derived != expected_row:
            raise MatchedEvalSurfaceError(f"derived row drift at index {index}")
    return MatchedEvalSurfaceBinding(
        path=path,
        source_manifest=dict(records["source_manifest"]),
        source_audit=dict(records["source_audit"]),
        pair_ledger=dict(records["pair_ledger"]),
        derived_manifest=dict(records["derived_manifest"]),
        data_root=data_root,
        image_root=image_root,
        image_files_sha256=str(payload["image_files_sha256"]),
        image_files=tuple(dict(value) for value in image_files),
        support_pool_mapping_sha256=str(support_pool["mapping_sha256"]),
        support_pool_mapping=tuple(
            dict(value) for value in support_pool["mapping"]
        ),
        support_pool_files_sha256=str(support_pool["files_sha256"]),
        support_pool_files=tuple(dict(value) for value in support_pool["files"]),
        row_mapping_sha256=str(payload["row_mapping_sha256"]),
        row_mapping=tuple(dict(value) for value in mapping),
    )


def meta_rows(binding: MatchedEvalSurfaceBinding) -> list[dict[str, Any]]:
    rows = _read_jsonl(
        Path(str(binding.derived_manifest["path"])), label="matched eval surface"
    )
    result = []
    for row in rows:
        category = row.get("replace_category")
        if isinstance(category, list):
            category = category[0] if category else "unknown"
        result.append(
            {
                "sample_id": str(row["sample_id"]),
                "eval_split": EVAL_SPLIT,
                "pair_source": row.get("pair_source"),
                "source_split": "sealed_parent_matched_calibration",
                "image_id": int(row["image_id"]),
                "ann_id": int(row["ann_id"]),
                "ref_id": int(row["ref_id"]),
                "sent_id": int(row["sent_id"]),
                "negative_phrase": row["try_tn"],
                "positive_phrase": row["sent"],
                "category": str(category or "unknown"),
                "matched_pair_id": str(row["matched_pair_id"]),
                "matched_parent_key_sha256": str(
                    row["matched_parent_key_sha256"]
                ),
            }
        )
    return result


def summary_fields(binding: MatchedEvalSurfaceBinding) -> dict[str, Any]:
    return {
        "matched_eval_surface_binding_schema": SCHEMA,
        "matched_eval_surface_derivation_algorithm": DERIVATION_ALGORITHM,
        "matched_eval_surface_binding_path": str(binding.path),
        "matched_eval_surface_binding_sha256": sha256_file(binding.path),
        "matched_eval_surface_source_path": str(binding.source_manifest["path"]),
        "matched_eval_surface_source_sha256": str(
            binding.source_manifest["sha256"]
        ),
        "matched_eval_surface_source_n": int(binding.source_manifest["rows"]),
        "matched_eval_surface_audit_sha256": str(binding.source_audit["sha256"]),
        "matched_eval_surface_ledger_sha256": str(binding.pair_ledger["sha256"]),
        "matched_eval_surface_derived_path": str(binding.derived_manifest["path"]),
        "matched_eval_surface_derived_sha256": str(
            binding.derived_manifest["sha256"]
        ),
        "matched_eval_surface_row_mapping_sha256": binding.row_mapping_sha256,
        "matched_eval_surface_scope": DECLARED_SCOPE,
        "matched_eval_surface_split": EVAL_SPLIT,
        "matched_eval_surface_image_files_n": len(binding.image_files),
        "matched_eval_surface_image_files_sha256": binding.image_files_sha256,
        "matched_eval_surface_support_pool_classes_n": len(
            binding.support_pool_mapping
        ),
        "matched_eval_surface_support_pool_files_n": len(
            binding.support_pool_files
        ),
        "matched_eval_surface_support_pool_mapping_sha256": (
            binding.support_pool_mapping_sha256
        ),
        "matched_eval_surface_support_pool_files_sha256": (
            binding.support_pool_files_sha256
        ),
        "formal_global_fpr_eligible": False,
    }


def iter_rows(path: Path) -> Iterable[dict[str, Any]]:
    return iter(_read_jsonl(path, label="matched evaluation rows"))
