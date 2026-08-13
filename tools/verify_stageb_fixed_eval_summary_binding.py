#!/usr/bin/env python3
"""Replay fixed Stage-B summaries and bind every record to its checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.stageb_eval_records import (  # noqa: E402
    RECORD_SCHEMA,
    TN_DERIVATION_ALGORITHM,
    TN_DERIVED_MANIFEST_BINDING_SCHEMA,
    load_tn_derived_manifest_binding,
    sample_id_from_meta,
    sha256_file,
    tn_manifest_derivation_contract,
)
from tools.stageb_ref_split_contract import (  # noqa: E402
    REF_SPLIT_CONTRACT,
    REF_SPLIT_MANIFEST_FILES,
    REF_SPLITS,
)
from tools.stageb_gdino_fixed_top1_selection import (  # noqa: E402
    SELECTION_SCHEMA,
    SelectionError,
    verify_selection,
)
SCHEMA = "stageb-fixed-eval-summary-binding-v1"
TN_SECTIONS = {"strict2031": 2031, "strict1607": 1607}
TN_SOURCE_MANIFESTS = {
    "strict2031": (
        "data/eval_manifests/stageb_vlm_verified_strict_ann_umd_val_20260711/"
        "eval_manifest.jsonl"
    ),
    "strict1607": (
        "data/eval_manifests/stageb_vlm_verified_strict_ann_umd_val_20260711/"
        "semantic_stageb_union_image_disjoint_manifest.jsonl"
    ),
}

P0_SCHEMA = "stageb-gdino-adapter-p0-v1"
TWO_PHASE_SCHEMA = "stageb-gdino-adapter-two-phase-probe-v1"
TOTAL_TRUST_SCHEMA = "stageb-gdino-adapter-total-trust-probe-v1"
SEMANTIC_SCHEMA = "stageb-gdino-adapter-semantic-confidence-probe-v1"
FIXED_TOP1_SCHEMA = "stageb-gdino-adapter-fixed-top1-confidence-probe-v1"
ADAPTER_TWO_PHASE_SCHEMAS = {TWO_PHASE_SCHEMA, TOTAL_TRUST_SCHEMA}


class SummaryBindingError(RuntimeError):
    pass


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = REPO_ROOT / value
    return value.resolve()


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SummaryBindingError(f"{label} is missing or is not a file: {path}")


def _read_json(path: Path, label: str) -> Dict[str, Any]:
    _require_file(path, label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SummaryBindingError(f"could not read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise SummaryBindingError(f"{label} must be a JSON object: {path}")
    return value


def _read_jsonl_with_record(
    path: Path, label: str
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    _require_file(path, label)
    try:
        raw = path.read_bytes()
        rendered = raw.decode("utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise SummaryBindingError(f"could not read {label} {path}: {error}") from error
    rows: list[Dict[str, Any]] = []
    for line_number, line in enumerate(rendered.splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise SummaryBindingError(
                f"invalid JSON at {path}:{line_number}: {error}"
            ) from error
        if not isinstance(row, dict):
            raise SummaryBindingError(f"expected an object at {path}:{line_number}")
        rows.append(row)
    if not rows:
        raise SummaryBindingError(f"{label} is empty: {path}")
    return rows, {
        "path": str(path.resolve()),
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _read_jsonl(path: Path, label: str) -> list[Dict[str, Any]]:
    rows, _ = _read_jsonl_with_record(path, label)
    return rows


def _file_record(path: Path) -> Dict[str, Any]:
    _require_file(path, "audited file")
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "run"


def checkpoint_run_prefix(checkpoint: Path) -> str:
    parent = checkpoint.parent.name
    stem = checkpoint.stem
    return _safe_name(f"{parent}_{stem}") if parent else _safe_name(stem)


def _exact_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise SummaryBindingError(f"{label} must be an exact integer")
    return int(value)


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SummaryBindingError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise SummaryBindingError(f"{label} must be finite")
    return result


def _unit_float(value: Any, label: str) -> float:
    result = _finite_float(value, label)
    if result < 0.0 or result > 1.0:
        raise SummaryBindingError(f"{label} must be in [0, 1], got {result}")
    return result


def _require_exact_float(row: Mapping[str, Any], key: str, expected: float, label: str) -> None:
    observed = _finite_float(row.get(key), f"{label}.{key}")
    if observed != float(expected):
        raise SummaryBindingError(
            f"{label}.{key} does not match records: expected {expected!r}, got {observed!r}"
        )


def _require_exact_int(row: Mapping[str, Any], key: str, expected: int, label: str) -> None:
    observed = _exact_int(row.get(key), f"{label}.{key}")
    if observed != int(expected):
        raise SummaryBindingError(
            f"{label}.{key} mismatch: expected {expected}, got {observed}"
        )


def _declared_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SummaryBindingError(f"{label} must be a non-empty path string")
    return _resolve(value)


def _current_file_identity(value: Any, label: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SummaryBindingError(f"{label} must be a file-record object")
    path = _declared_path(value.get("path"), f"{label}.path")
    current = _file_record(path)
    if value.get("size_bytes") != current["size_bytes"] or value.get("sha256") != current[
        "sha256"
    ]:
        raise SummaryBindingError(f"{label} file record drifted: {path}")
    return current


def _require_same_file_identity(
    observed: Mapping[str, Any], expected: Mapping[str, Any], label: str
) -> None:
    if observed != expected:
        raise SummaryBindingError(
            f"{label} file identity mismatch: expected {expected}, got {observed}"
        )


def _two_phase_root_baseline(
    milestone_value: Any,
    *,
    expected_checkpoint: Mapping[str, Any],
    expected_schema: str | None = None,
    allow_historical_rank_root: bool = False,
    visited: set[Path] | None = None,
) -> Dict[str, Any]:
    milestone_record = _current_file_identity(
        milestone_value, "two-phase milestone audit"
    )
    milestone_path = Path(milestone_record["path"])
    if visited is None:
        visited = set()
    if milestone_path in visited:
        raise SummaryBindingError("two-phase lineage contains an audit cycle")
    visited.add(milestone_path)
    try:
        milestone = _read_json(milestone_path, "two-phase milestone audit")
        milestone_schema = milestone.get("schema")
        schema_matches = milestone_schema in ADAPTER_TWO_PHASE_SCHEMAS
        if expected_schema is not None:
            schema_matches = milestone_schema == expected_schema
            if allow_historical_rank_root and expected_schema == TOTAL_TRUST_SCHEMA:
                schema_matches = milestone_schema in {
                    TOTAL_TRUST_SCHEMA,
                    TWO_PHASE_SCHEMA,
                }
        if not schema_matches or milestone.get("kind") != "milestone_checkpoint":
            raise SummaryBindingError("two-phase lineage has an invalid milestone audit")
        checkpoint = _current_file_identity(
            milestone.get("checkpoint"), "two-phase milestone checkpoint"
        )
        _require_same_file_identity(
            checkpoint, expected_checkpoint, "two-phase milestone checkpoint"
        )
        preflight_record = _current_file_identity(
            milestone.get("preflight"), "two-phase preflight"
        )
        preflight = _read_json(Path(preflight_record["path"]), "two-phase preflight")
        phase = preflight.get("phase")
        preflight_schema = preflight.get("schema")
        historical_rank = (
            allow_historical_rank_root
            and expected_schema == TOTAL_TRUST_SCHEMA
            and milestone_schema == TWO_PHASE_SCHEMA
            and preflight_schema == TWO_PHASE_SCHEMA
            and phase == "rank"
        )
        if expected_schema is None:
            schema_pair_matches = (
                milestone_schema in ADAPTER_TWO_PHASE_SCHEMAS
                and preflight_schema == milestone_schema
            )
        else:
            schema_pair_matches = (
                milestone_schema == expected_schema
                and preflight_schema == expected_schema
            )
            if historical_rank:
                schema_pair_matches = True
        if (
            not schema_pair_matches
            or preflight.get("kind") != "phase_preflight"
            or phase not in {"rank", "confidence"}
        ):
            raise SummaryBindingError("two-phase lineage has an invalid phase preflight")
        initial_checkpoint = _current_file_identity(
            preflight.get("initial_checkpoint"), "two-phase initial checkpoint"
        )
        if phase == "rank":
            return initial_checkpoint
        initial_audit = preflight.get("initial_audit")
        if not isinstance(initial_audit, Mapping):
            raise SummaryBindingError(
                "two-phase confidence preflight has no rank initial audit"
            )
        return _two_phase_root_baseline(
            initial_audit,
            expected_checkpoint=initial_checkpoint,
            expected_schema=expected_schema,
            allow_historical_rank_root=allow_historical_rank_root,
            visited=visited,
        )
    finally:
        visited.remove(milestone_path)


def _preflight_root_baseline(
    preflight_value: Any,
    *,
    schema: str,
    phase: str,
) -> Dict[str, Any]:
    preflight_record = _current_file_identity(preflight_value, f"{phase} preflight")
    preflight = _read_json(Path(preflight_record["path"]), f"{phase} preflight")
    if (
        preflight.get("schema") != schema
        or preflight.get("kind") != "phase_preflight"
        or preflight.get("phase") != phase
    ):
        raise SummaryBindingError(f"{phase} lineage has an invalid preflight")
    initial_checkpoint = _current_file_identity(
        preflight.get("initial_checkpoint"), f"{phase} initial checkpoint"
    )
    initial_audit = preflight.get("initial_audit")
    if not isinstance(initial_audit, Mapping):
        raise SummaryBindingError(f"{phase} preflight has no two-phase source audit")
    return _two_phase_root_baseline(
        initial_audit,
        expected_checkpoint=initial_checkpoint,
    )


def _audit_trusted_lineage(
    lineage_path: Path,
    *,
    candidate_checkpoint: Mapping[str, Any],
    expected_baseline_checkpoint: Path,
) -> Dict[str, Any]:
    lineage_path = _resolve(lineage_path)
    lineage = _read_json(lineage_path, "trusted candidate lineage output")
    lineage_record = _file_record(lineage_path)
    baseline_record = _file_record(_resolve(expected_baseline_checkpoint))
    schema = lineage.get("schema")
    kind = lineage.get("kind")
    selection_binding = None
    lineage_checkpoint = _current_file_identity(
        lineage.get("checkpoint"), "trusted lineage candidate checkpoint"
    )
    _require_same_file_identity(
        lineage_checkpoint,
        candidate_checkpoint,
        "evaluation preflight/trusted lineage candidate checkpoint",
    )

    if schema == P0_SCHEMA:
        if kind != "p0_checkpoint_and_sidecar_verified":
            raise SummaryBindingError("P0 lineage output has the wrong kind")
        audit = lineage.get("audit")
        if not isinstance(audit, Mapping):
            raise SummaryBindingError("P0 lineage output has no recomputed audit")
        if audit.get("schema") != P0_SCHEMA or audit.get("kind") != "p0_checkpoint_audit":
            raise SummaryBindingError("P0 lineage audit has the wrong schema or kind")
        p0_checkpoint = _current_file_identity(
            audit.get("p0_checkpoint"), "P0 audit checkpoint"
        )
        _require_same_file_identity(
            p0_checkpoint, candidate_checkpoint, "P0 audit/evaluation checkpoint"
        )
        root_baseline = _current_file_identity(
            audit.get("baseline"), "P0 root baseline checkpoint"
        )
        sidecar_record = _current_file_identity(lineage.get("sidecar"), "P0 sidecar")
        sidecar = _read_json(Path(sidecar_record["path"]), "P0 sidecar")
        if sidecar != dict(audit):
            raise SummaryBindingError("P0 sidecar no longer matches the trusted audit")
        identity = audit.get("functional_identity")
        if not isinstance(identity, Mapping) or any(
            identity.get(key) is not True
            for key in (
                "rank_score_equals_base",
                "confidence_score_equals_base",
                "rank_residual_exact_zero",
                "confidence_gate_exact_zero",
            )
        ):
            raise SummaryBindingError("P0 functional identity contract is incomplete")
        if audit.get("intended_use") != "evaluation_only_same_records_parity":
            raise SummaryBindingError("P0 intended-use contract is invalid")
        evaluation_config = _current_file_identity(
            audit.get("config"), "P0 evaluation config"
        )
    elif schema in ADAPTER_TWO_PHASE_SCHEMAS:
        if kind != "evaluation_checkpoint_verified":
            raise SummaryBindingError("two-phase lineage output has the wrong kind")
        root_baseline = _two_phase_root_baseline(
            lineage.get("audit"),
            expected_checkpoint=candidate_checkpoint,
            expected_schema=schema,
            allow_historical_rank_root=(schema == TOTAL_TRUST_SCHEMA),
        )
        evaluation_config = _current_file_identity(
            lineage.get("config"), "two-phase evaluation config"
        )
    elif schema == SEMANTIC_SCHEMA:
        if kind != "evaluation_checkpoint_verification" or lineage.get("verified") is not True:
            raise SummaryBindingError("semantic lineage output has the wrong kind")
        checkpoint_audit_record = _current_file_identity(
            lineage.get("checkpoint_audit"), "semantic checkpoint audit"
        )
        checkpoint_audit = _read_json(
            Path(checkpoint_audit_record["path"]), "semantic checkpoint audit"
        )
        if (
            checkpoint_audit.get("schema") != SEMANTIC_SCHEMA
            or checkpoint_audit.get("kind") != "milestone_checkpoint"
        ):
            raise SummaryBindingError("semantic checkpoint audit is invalid")
        audited_checkpoint = _current_file_identity(
            checkpoint_audit.get("checkpoint"), "semantic audited checkpoint"
        )
        _require_same_file_identity(
            audited_checkpoint, candidate_checkpoint, "semantic audited/evaluation checkpoint"
        )
        root_baseline = _preflight_root_baseline(
            lineage.get("preflight"),
            schema=SEMANTIC_SCHEMA,
            phase="semantic-confidence",
        )
        evaluation_config = _current_file_identity(
            lineage.get("config"), "semantic evaluation config"
        )
    elif schema == FIXED_TOP1_SCHEMA:
        if kind != "evaluation_checkpoint_verification" or lineage.get("verified") is not True:
            raise SummaryBindingError("fixed-top1 lineage output has the wrong kind")
        checkpoint_audit_record = _current_file_identity(
            lineage.get("checkpoint_audit"), "fixed-top1 checkpoint audit"
        )
        checkpoint_audit = _read_json(
            Path(checkpoint_audit_record["path"]), "fixed-top1 checkpoint audit"
        )
        if (
            checkpoint_audit.get("schema") != FIXED_TOP1_SCHEMA
            or checkpoint_audit.get("kind") != "milestone_checkpoint"
        ):
            raise SummaryBindingError("fixed-top1 checkpoint audit is invalid")
        audited_checkpoint = _current_file_identity(
            checkpoint_audit.get("checkpoint"), "fixed-top1 audited checkpoint"
        )
        _require_same_file_identity(
            audited_checkpoint,
            candidate_checkpoint,
            "fixed-top1 audited/evaluation checkpoint",
        )
        root_baseline = _preflight_root_baseline(
            lineage.get("preflight"),
            schema=FIXED_TOP1_SCHEMA,
            phase="fixed-top1-confidence",
        )
        source_binding = lineage.get("fixed_gdino_source_binding")
        if (
            not isinstance(source_binding, Mapping)
            or source_binding.get("matches_rank_initial_baseline") is not True
            or source_binding.get("checkpoint_sha256") != root_baseline["sha256"]
        ):
            raise SummaryBindingError("fixed-top1 root-baseline binding is invalid")
        evaluation_config = _current_file_identity(
            lineage.get("config"), "fixed-top1 evaluation config"
        )
        authorization = lineage.get("selection_authorization")
        if (
            not isinstance(authorization, Mapping)
            or authorization.get("schema") != SELECTION_SCHEMA
            or authorization.get("input_scope") != "calibration_only"
            or authorization.get("strict_paths_consumed_for_scoring") is not False
            or authorization.get("formal_strict_authorization") is not True
        ):
            raise SummaryBindingError(
                "fixed-top1 lineage has no calibration-only selection authorization"
            )
        selection_record = _current_file_identity(
            authorization.get("audit"), "fixed-top1 selection audit"
        )
        try:
            selection = verify_selection(
                Path(selection_record["path"]),
                expected_checkpoint=Path(candidate_checkpoint["path"]),
                expected_milestone_audit=Path(checkpoint_audit_record["path"]),
                expected_calibration_root=(
                    Path(selection_record["path"]).parent.parent
                    / "calibration_selection"
                ),
            )
        except SelectionError as error:
            raise SummaryBindingError(
                f"fixed-top1 selection audit replay failed: {error}"
            ) from error
        for key, expected in (
            ("selected_checkpoint", selection["selected_checkpoint"]),
            ("selected_milestone_audit", selection["selected_milestone_audit"]),
            ("selected_iteration", selection["selected_iteration"]),
            ("calibration_root", selection["calibration_root"]),
        ):
            if authorization.get(key) != expected:
                raise SummaryBindingError(
                    f"fixed-top1 lineage selection authorization drifted in {key}"
                )
        selection_binding = {
            "pass": True,
            "selection_audit": selection_record,
            "selected_checkpoint": selection["selected_checkpoint"],
            "selected_milestone_audit": selection["selected_milestone_audit"],
            "selected_iteration": selection["selected_iteration"],
            "calibration_root": selection["calibration_root"],
            "input_scope": selection["input_scope"],
            "strict_paths_consumed_for_scoring": selection[
                "strict_paths_consumed_for_scoring"
            ],
        }
    else:
        raise SummaryBindingError(f"unsupported trusted lineage schema: {schema!r}")

    _require_same_file_identity(
        root_baseline,
        baseline_record,
        "candidate lineage root/current authoritative baseline checkpoint",
    )
    return {
        "pass": True,
        "schema": schema,
        "kind": kind,
        "trusted_lineage_output": lineage_record,
        "candidate_checkpoint": dict(candidate_checkpoint),
        "evaluation_config": evaluation_config,
        "root_authoritative_baseline_checkpoint": baseline_record,
        "eval_preflight_current_lineage_checkpoint_same_file_record": True,
        "lineage_root_current_baseline_same_file_record": True,
        "selection_binding": selection_binding,
    }


def _validate_summary_binding(
    row: Mapping[str, Any],
    *,
    checkpoint: Path,
    run_id: str,
    records_path: Path,
    label: str,
) -> None:
    if row.get("run_id") != run_id:
        raise SummaryBindingError(
            f"{label}.run_id mismatch: expected {run_id!r}, got {row.get('run_id')!r}"
        )
    if row.get("checkpoint_name") != checkpoint.name:
        raise SummaryBindingError(f"{label}.checkpoint_name does not match preflight checkpoint")
    if _declared_path(row.get("checkpoint"), f"{label}.checkpoint") != checkpoint:
        raise SummaryBindingError(f"{label}.checkpoint does not match preflight checkpoint")
    if _declared_path(row.get("records_jsonl"), f"{label}.records_jsonl") != records_path:
        raise SummaryBindingError(f"{label}.records_jsonl does not match the audited record file")


def _validate_runtime(
    row: Mapping[str, Any],
    runtime: Mapping[str, Any],
    *,
    expected_seed: int,
    label: str,
) -> None:
    for key in ("batch_size", "num_workers"):
        expected = _exact_int(runtime.get(key), f"preflight.runtime.{key}")
        _require_exact_int(row, key, expected, label)
    _require_exact_int(row, "seed", expected_seed, label)
    _require_exact_int(row, "max_batches", 0, label)


def _record_identity(
    row: Mapping[str, Any],
    *,
    task: str,
    manifest_key: str,
    run_id: str,
    index: int,
    total: int,
    label: str,
) -> tuple[str, str]:
    if row.get("schema") != RECORD_SCHEMA:
        raise SummaryBindingError(f"{label} has the wrong record schema")
    if row.get("task") != task or row.get("manifest_key") != manifest_key:
        raise SummaryBindingError(f"{label} has the wrong task or manifest_key")
    if row.get("run_id") != run_id:
        raise SummaryBindingError(
            f"{label}.run_id mismatch: expected {run_id!r}, got {row.get('run_id')!r}"
        )
    if row.get("valid") is not True:
        raise SummaryBindingError(f"{label} must have valid=true")
    if _exact_int(row.get("manifest_index"), f"{label}.manifest_index") != index:
        raise SummaryBindingError(f"{label} is not in exact manifest order")
    if _exact_int(row.get("manifest_n"), f"{label}.manifest_n") != total:
        raise SummaryBindingError(f"{label}.manifest_n does not match record count")
    sample_id = row.get("sample_id")
    if not isinstance(sample_id, str) or not sample_id:
        raise SummaryBindingError(f"{label}.sample_id must be non-empty")
    manifest_sha = row.get("manifest_sha256")
    if (
        not isinstance(manifest_sha, str)
        or len(manifest_sha) != 64
        or any(char not in "0123456789abcdef" for char in manifest_sha.lower())
    ):
        raise SummaryBindingError(f"{label}.manifest_sha256 is invalid")
    return sample_id, manifest_sha.lower()


def _audit_ref_records(
    path: Path,
    *,
    split: str,
    run_id: str,
    manifest_path: Path,
) -> Dict[str, Any]:
    rows, records_file = _read_jsonl_with_record(path, f"{split} records")
    manifest_rows, manifest_file = _read_jsonl_with_record(
        manifest_path, f"{split} canonical manifest"
    )
    contract = REF_SPLIT_CONTRACT[split]
    if len(manifest_rows) != int(contract["rows"]):
        raise SummaryBindingError(
            f"{split} is not the locked official split: expected "
            f"{contract['rows']} manifest rows, got {len(manifest_rows)}"
        )
    if manifest_file["sha256"] != str(contract["sha256"]):
        raise SummaryBindingError(
            f"{split} is not the locked official split: canonical manifest SHA-256 mismatch"
        )
    if len(rows) != len(manifest_rows):
        raise SummaryBindingError(
            f"{split} records do not cover the complete canonical manifest"
        )
    sample_ids: list[str] = []
    manifest_hashes: set[str] = set()
    top1_sum = 0.0
    all_query_sum = 0.0
    top1_correct50 = 0
    top1_correct25 = 0
    all_query_correct50 = 0
    all_query_correct25 = 0
    for index, (row, manifest_row) in enumerate(zip(rows, manifest_rows)):
        label = f"{split} record {index}"
        sample_id, manifest_sha = _record_identity(
            row,
            task="ref",
            manifest_key=f"ref:{split}",
            run_id=run_id,
            index=index,
            total=len(rows),
            label=label,
        )
        if row.get("split") != split:
            raise SummaryBindingError(f"{label}.split mismatch")
        try:
            expected_sample_id = sample_id_from_meta(
                manifest_row,
                task="ref",
                split=split,
                index=index,
            )
        except (TypeError, ValueError) as error:
            raise SummaryBindingError(
                f"{split} canonical manifest row {index} has invalid identity: {error}"
            ) from error
        if sample_id != expected_sample_id:
            raise SummaryBindingError(
                f"{label}.sample_id does not match the canonical manifest"
            )
        for identity_key in ("image_id", "ann_id", "ref_id", "sent_id"):
            manifest_identity = _exact_int(
                manifest_row.get(identity_key),
                f"{split} canonical manifest row {index}.{identity_key}",
            )
            record_identity = _exact_int(
                row.get(identity_key), f"{label}.{identity_key}"
            )
            if record_identity != manifest_identity:
                raise SummaryBindingError(
                    f"{label}.{identity_key} does not match the canonical manifest"
                )
        top1_iou = _unit_float(row.get("top1_iou"), f"{label}.top1_iou")
        all_query_iou = _unit_float(
            row.get("all_query_best_iou"), f"{label}.all_query_best_iou"
        )
        if all_query_iou < top1_iou:
            raise SummaryBindingError(f"{label} oracle IoU is lower than top1 IoU")
        correct50 = row.get("correct50")
        if type(correct50) is not bool or correct50 is not (top1_iou >= 0.5):
            raise SummaryBindingError(f"{label}.correct50 is inconsistent with top1_iou")
        top1_sum += top1_iou
        all_query_sum += all_query_iou
        top1_correct50 += int(correct50)
        top1_correct25 += int(top1_iou >= 0.25)
        all_query_correct50 += int(all_query_iou >= 0.5)
        all_query_correct25 += int(all_query_iou >= 0.25)
        sample_ids.append(sample_id)
        manifest_hashes.add(manifest_sha)
    if len(set(sample_ids)) != len(sample_ids):
        raise SummaryBindingError(f"{split} records contain duplicate sample IDs")
    if len(manifest_hashes) != 1:
        raise SummaryBindingError(f"{split} records contain multiple manifest hashes")
    total = len(rows)
    return {
        "records": records_file,
        "canonical_manifest": manifest_file,
        "n": total,
        "manifest_sha256": next(iter(manifest_hashes)),
        "metrics": {
            "acc50": float(top1_correct50 / total),
            "acc25": float(top1_correct25 / total),
            "mean_iou": float(top1_sum / total),
            "recall50@all_queries": float(all_query_correct50 / total),
            "recall25@all_queries": float(all_query_correct25 / total),
            "mean_best_iou@all_queries": float(all_query_sum / total),
        },
    }


def _audit_ref_section(
    eval_dir: Path,
    *,
    checkpoint: Path,
    run_id: str,
    runtime: Mapping[str, Any],
) -> Dict[str, Any]:
    section = eval_dir / "ref8"
    summary_path = section / "summary.json"
    summary = _read_json(summary_path, "ref8 summary")
    ref_rows = summary.get("refcoco")
    tn_rows = summary.get("tn")
    if not isinstance(ref_rows, list) or not isinstance(tn_rows, list) or tn_rows:
        raise SummaryBindingError("ref8 summary must contain refcoco rows and no TN rows")
    if len(ref_rows) != len(REF_SPLITS):
        raise SummaryBindingError(
            f"ref8 summary must contain exactly {len(REF_SPLITS)} rows"
        )
    observed_splits = [row.get("dataset") if isinstance(row, Mapping) else None for row in ref_rows]
    if observed_splits != list(REF_SPLITS):
        raise SummaryBindingError(
            f"ref8 summary split order mismatch: expected {list(REF_SPLITS)}, got {observed_splits}"
        )
    record_dir = section / "per_example_records"
    record_files = sorted(record_dir.glob("*.records.jsonl"))
    expected_files = {
        split: record_dir / f"{run_id}__{_safe_name(split)}.records.jsonl"
        for split in REF_SPLITS
    }
    if {path.resolve() for path in record_files} != {
        path.resolve() for path in expected_files.values()
    }:
        raise SummaryBindingError("ref8 record filenames do not exactly match the checkpoint run")
    manifest_dir = section / "refcoco_eval_inputs"
    manifest_files = sorted(manifest_dir.glob("*.jsonl"))
    expected_manifests = {
        split: manifest_dir / REF_SPLIT_MANIFEST_FILES[split]
        for split in REF_SPLITS
    }
    if {path.resolve() for path in manifest_files} != {
        path.resolve() for path in expected_manifests.values()
    }:
        raise SummaryBindingError(
            "ref8 canonical manifest filenames do not exactly match the eight official splits"
        )
    result: Dict[str, Any] = {}
    base_seed = _exact_int(runtime.get("seed"), "preflight.runtime.seed")
    for split_index, (split, summary_row) in enumerate(zip(REF_SPLITS, ref_rows)):
        if not isinstance(summary_row, Mapping):
            raise SummaryBindingError(f"ref8 summary row {split_index} is not an object")
        record_path = expected_files[split].resolve()
        audit = _audit_ref_records(
            record_path,
            split=split,
            run_id=run_id,
            manifest_path=expected_manifests[split].resolve(),
        )
        if audit["manifest_sha256"] != audit["canonical_manifest"]["sha256"]:
            raise SummaryBindingError(
                f"{split} records declare a manifest SHA-256 that differs from the canonical file"
            )
        label = f"ref8 summary {split}"
        _validate_summary_binding(
            summary_row,
            checkpoint=checkpoint,
            run_id=run_id,
            records_path=record_path,
            label=label,
        )
        _validate_runtime(
            summary_row,
            runtime,
            expected_seed=base_seed + split_index * 100000,
            label=label,
        )
        for key, expected in (
            ("num_expressions", audit["n"]),
            ("valid_mask_expressions", audit["n"]),
            ("invalid_mask_expressions", 0),
            ("manifest_n", audit["n"]),
            ("invalid_records", 0),
        ):
            _require_exact_int(summary_row, key, expected, label)
        if summary_row.get("manifest_sha256") != audit["manifest_sha256"]:
            raise SummaryBindingError(f"{label}.manifest_sha256 does not match records")
        for key, expected in audit["metrics"].items():
            _require_exact_float(summary_row, key, expected, label)
        result[split] = audit
    return {"summary": _file_record(summary_path), "splits": result}


def _threshold_for_tpr(scores: np.ndarray, target_tpr: float) -> float:
    import numpy as np

    accepted = max(1, int(math.ceil(float(target_tpr) * int(scores.size))))
    ascending_index = int(scores.size) - accepted
    return float(np.partition(scores, ascending_index)[ascending_index])


def _tn_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    threshold_tprs: Iterable[float],
    score_thresholds: Iterable[float],
) -> Dict[str, float | int]:
    import numpy as np

    positive = np.asarray([row["pos_score"] for row in rows], dtype=np.float32)
    negative = np.asarray([row["neg_score"] for row in rows], dtype=np.float32)
    pos_iou = np.asarray([row["pos_iou"] for row in rows], dtype=np.float32)
    neg_iou = np.asarray([row["neg_iou"] for row in rows], dtype=np.float32)
    gap = positive - negative
    metrics: Dict[str, float | int] = {
        "num_pairs": int(positive.size),
        "pair_win_rate": float(np.mean(positive > negative)),
        "pair_tie_rate": float(np.mean(positive == negative)),
        "score_gap_mean": float(gap.mean()),
        "score_gap_median": float(np.median(gap)),
        "pos_score_mean": float(positive.mean()),
        "tn_score_mean": float(negative.mean()),
        "pos_top1_iou50": float(np.mean(pos_iou >= 0.5)),
        "tn_top1_iou50": float(np.mean(neg_iou >= 0.5)),
    }
    for tpr in threshold_tprs:
        key = f"{int(round(float(tpr) * 100)):02d}"
        threshold = _threshold_for_tpr(positive, float(tpr))
        metrics[f"threshold_at_{key}tpr"] = threshold
        metrics[f"actual_tpr_at_{key}tpr"] = float(np.mean(positive >= threshold))
        metrics[f"fpr{key}tpr"] = float(np.mean(negative >= threshold))
    for threshold in score_thresholds:
        key = f"{float(threshold):.2f}".replace(".", "p")
        metrics[f"tpr_at_score_{key}"] = float(np.mean(positive >= float(threshold)))
        metrics[f"fpr_at_score_{key}"] = float(np.mean(negative >= float(threshold)))
    metrics["tn_fpr"] = float(metrics["fpr95tpr"])
    return metrics


def _audit_tn_section(
    eval_dir: Path,
    section_name: str,
    expected_n: int,
    *,
    checkpoint: Path,
    run_id: str,
    runtime: Mapping[str, Any],
) -> Dict[str, Any]:
    section = eval_dir / section_name
    summary_path = section / "summary.json"
    summary = _read_json(summary_path, f"{section_name} summary")
    ref_rows = summary.get("refcoco")
    tn_rows = summary.get("tn")
    if not isinstance(ref_rows, list) or ref_rows or not isinstance(tn_rows, list) or len(tn_rows) != 1:
        raise SummaryBindingError(
            f"{section_name} summary must contain exactly one TN row and no Ref rows"
        )
    summary_row = tn_rows[0]
    if not isinstance(summary_row, Mapping):
        raise SummaryBindingError(f"{section_name} TN summary row is not an object")
    record_dir = section / "per_example_records"
    record_files = sorted(record_dir.glob("*.records.jsonl"))
    expected_path = (record_dir / f"{run_id}__tn_global.records.jsonl").resolve()
    if len(record_files) != 1 or record_files[0].resolve() != expected_path:
        raise SummaryBindingError(
            f"{section_name} must contain the checkpoint-bound TN record filename"
        )
    rows, records_file = _read_jsonl_with_record(
        expected_path, f"{section_name} records"
    )
    if len(rows) != expected_n:
        raise SummaryBindingError(
            f"{section_name} must contain exactly {expected_n} records, got {len(rows)}"
        )
    source_manifest_path = _resolve(TN_SOURCE_MANIFESTS[section_name])
    source_rows, source_file = _read_jsonl_with_record(
        source_manifest_path, f"{section_name} locked source manifest"
    )
    if len(source_rows) != expected_n:
        raise SummaryBindingError(
            f"{section_name} locked source manifest row count changed: "
            f"expected {expected_n}, got {len(source_rows)}"
        )
    derived_manifest_path = (
        section
        / "tn_eval_inputs"
        / "tn_refcocop_val_refcocog_umd_val.jsonl"
    ).resolve()
    binding_path = derived_manifest_path.with_suffix(
        derived_manifest_path.suffix + ".binding.json"
    )
    try:
        manifest_binding = load_tn_derived_manifest_binding(
            binding_path, expected_derived_manifest=derived_manifest_path
        )
    except (OSError, TypeError, ValueError) as error:
        raise SummaryBindingError(
            f"{section_name} source-to-derived manifest binding failed: {error}"
        ) from error
    expected_source_record = {**source_file, "rows": len(source_rows)}
    if dict(manifest_binding.source_manifest) != expected_source_record:
        raise SummaryBindingError(
            f"{section_name} binding does not use the current locked source manifest"
        )
    if int(manifest_binding.derived_manifest["rows"]) != expected_n:
        raise SummaryBindingError(
            f"{section_name} derived data manifest row count mismatch"
        )
    source_indices = [
        int(mapping["source_index"]) for mapping in manifest_binding.row_mapping
    ]
    if source_indices != list(range(expected_n)):
        raise SummaryBindingError(
            f"{section_name} derivation does not preserve every locked source row in order"
        )
    for key, expected in (
        ("requested_splits", ["refcocop_val", "refcocog_umd_val"]),
        ("max_pairs", 0),
        ("max_pairs_per_split", 0),
        ("holdout_level", "none"),
    ):
        if manifest_binding.derivation.get(key) != expected:
            raise SummaryBindingError(
                f"{section_name} derivation.{key} violates the fixed protocol"
            )
    sample_ids: list[str] = []
    manifest_hashes: set[str] = set()
    normalized: list[Dict[str, Any]] = []
    for index, (row, source_row, mapping) in enumerate(
        zip(rows, source_rows, manifest_binding.row_mapping)
    ):
        label = f"{section_name} record {index}"
        sample_id, manifest_sha = _record_identity(
            row,
            task="tn",
            manifest_key="tn_global",
            run_id=run_id,
            index=index,
            total=expected_n,
            label=label,
        )
        normalized_row = dict(row)
        for key in ("pos_score", "neg_score"):
            normalized_row[key] = _finite_float(row.get(key), f"{label}.{key}")
        for key in ("pos_iou", "neg_iou"):
            normalized_row[key] = _unit_float(row.get(key), f"{label}.{key}")
        normalized.append(normalized_row)
        expected_sample_id = sample_id_from_meta(
            source_row, task="tn", split="global", index=index
        )
        if sample_id != expected_sample_id or sample_id != mapping.get("sample_id"):
            raise SummaryBindingError(
                f"{label}.sample_id does not match the locked source row mapping"
            )
        expected_binding_fields = {
            "manifest_sha256": str(manifest_binding.derived_manifest["sha256"]),
            "manifest_path": str(manifest_binding.derived_manifest["path"]),
            "manifest_size_bytes": int(
                manifest_binding.derived_manifest["size_bytes"]
            ),
            "manifest_binding_schema": TN_DERIVED_MANIFEST_BINDING_SCHEMA,
            "manifest_derivation_algorithm": TN_DERIVATION_ALGORITHM,
            "manifest_binding_path": str(manifest_binding.path),
            "manifest_binding_sha256": manifest_binding.sha256,
            "manifest_binding_size_bytes": manifest_binding.size_bytes,
            "manifest_row_mapping_sha256": manifest_binding.row_mapping_sha256,
            "source_manifest_path": str(manifest_binding.source_manifest["path"]),
            "source_manifest_sha256": str(
                manifest_binding.source_manifest["sha256"]
            ),
            "source_manifest_size_bytes": int(
                manifest_binding.source_manifest["size_bytes"]
            ),
            "source_manifest_n": int(manifest_binding.source_manifest["rows"]),
            "source_manifest_index": int(mapping["source_index"]),
            "split": str(mapping["eval_split"]),
        }
        for field, expected in expected_binding_fields.items():
            if row.get(field) != expected:
                raise SummaryBindingError(
                    f"{label}.{field} does not match the two-layer manifest binding"
                )
        for identity_key in ("image_id", "ann_id", "ref_id", "sent_id"):
            if _exact_int(row.get(identity_key), f"{label}.{identity_key}") != _exact_int(
                source_row.get(identity_key),
                f"{section_name} source row {index}.{identity_key}",
            ):
                raise SummaryBindingError(
                    f"{label}.{identity_key} does not match the locked source manifest"
                )
        sample_ids.append(sample_id)
        manifest_hashes.add(manifest_sha)
    if len(set(sample_ids)) != len(sample_ids):
        raise SummaryBindingError(f"{section_name} records contain duplicate sample IDs")
    if len(manifest_hashes) != 1:
        raise SummaryBindingError(f"{section_name} records contain multiple manifest hashes")
    label = f"{section_name} summary"
    _validate_summary_binding(
        summary_row,
        checkpoint=checkpoint,
        run_id=run_id,
        records_path=expected_path,
        label=label,
    )
    base_seed = _exact_int(runtime.get("seed"), "preflight.runtime.seed")
    _validate_runtime(summary_row, runtime, expected_seed=base_seed, label=label)
    for key, expected in (
        ("manifest_n", expected_n),
        ("invalid_records", 0),
        ("invalid_positive_pairs", 0),
        ("invalid_negative_pairs", 0),
    ):
        _require_exact_int(summary_row, key, expected, label)
    manifest_sha = next(iter(manifest_hashes))
    if summary_row.get("manifest_sha256") != manifest_sha:
        raise SummaryBindingError(f"{label}.manifest_sha256 does not match records")
    expected_summary_binding = {
        "manifest_path": str(manifest_binding.derived_manifest["path"]),
        "manifest_size_bytes": int(manifest_binding.derived_manifest["size_bytes"]),
        "manifest_binding_schema": TN_DERIVED_MANIFEST_BINDING_SCHEMA,
        "manifest_derivation_algorithm": TN_DERIVATION_ALGORITHM,
        "manifest_binding_path": str(manifest_binding.path),
        "manifest_binding_sha256": manifest_binding.sha256,
        "manifest_binding_size_bytes": manifest_binding.size_bytes,
        "manifest_row_mapping_sha256": manifest_binding.row_mapping_sha256,
        "source_manifest_path": str(manifest_binding.source_manifest["path"]),
        "source_manifest_sha256": str(manifest_binding.source_manifest["sha256"]),
        "source_manifest_size_bytes": int(
            manifest_binding.source_manifest["size_bytes"]
        ),
        "source_manifest_n": int(manifest_binding.source_manifest["rows"]),
    }
    for field, expected in expected_summary_binding.items():
        if summary_row.get(field) != expected:
            raise SummaryBindingError(
                f"{label}.{field} does not match the two-layer manifest binding"
            )
    threshold_tprs = runtime.get("threshold_tprs")
    score_thresholds = runtime.get("score_thresholds")
    if not isinstance(threshold_tprs, list) or not isinstance(score_thresholds, list):
        raise SummaryBindingError("preflight runtime has invalid score threshold lists")
    metrics = _tn_metrics(
        normalized,
        threshold_tprs=[_finite_float(value, "threshold_tpr") for value in threshold_tprs],
        score_thresholds=[
            _finite_float(value, "score_threshold") for value in score_thresholds
        ],
    )
    for key, expected in metrics.items():
        if type(expected) is int:
            _require_exact_int(summary_row, key, expected, label)
        else:
            _require_exact_float(summary_row, key, float(expected), label)
    return {
        "summary": _file_record(summary_path),
        "records": records_file,
        "n": expected_n,
        "manifest_sha256": manifest_sha,
        "source_manifest_sha256": str(manifest_binding.source_manifest["sha256"]),
        "manifest_binding": {
            "schema": TN_DERIVED_MANIFEST_BINDING_SCHEMA,
            "algorithm": TN_DERIVATION_ALGORITHM,
            "source_manifest": dict(manifest_binding.source_manifest),
            "derived_manifest": dict(manifest_binding.derived_manifest),
            "binding": _file_record(manifest_binding.path),
            "row_mapping_sha256": manifest_binding.row_mapping_sha256,
            "row_mapping_verified_against_source_and_derived": True,
            "all_locked_source_rows_preserved_in_order": True,
        },
        "metrics": metrics,
    }


def _validate_completion_outputs(
    completion: Mapping[str, Any],
    *,
    ref8: Mapping[str, Any],
    tn_sections: Mapping[str, Mapping[str, Any]],
) -> None:
    outputs = completion.get("outputs")
    if not isinstance(outputs, Mapping):
        raise SummaryBindingError("completion audit has no outputs object")
    summaries = outputs.get("summaries")
    ref_records = outputs.get("ref_records")
    tn_records = outputs.get("tn_records")
    if not all(isinstance(value, Mapping) for value in (summaries, ref_records, tn_records)):
        raise SummaryBindingError("completion output records are incomplete")
    expected_summaries = {"ref8": ref8["summary"]}
    expected_summaries.update(
        {name: audit["summary"] for name, audit in tn_sections.items()}
    )
    if dict(summaries) != expected_summaries:
        raise SummaryBindingError("completion summary file records do not match current summaries")
    for split, audit in ref8["splits"].items():
        completed = ref_records.get(split)
        if not isinstance(completed, Mapping):
            raise SummaryBindingError(f"completion is missing Ref records for {split}")
        if completed.get("path") != audit["records"]["path"] or completed.get(
            "sha256"
        ) != audit["records"]["sha256"]:
            raise SummaryBindingError(f"completion Ref record hash mismatch for {split}")
    if set(ref_records) != set(REF_SPLITS):
        raise SummaryBindingError("completion Ref record groups are not exactly the eight splits")
    for section, audit in tn_sections.items():
        completed = tn_records.get(section)
        if not isinstance(completed, Mapping):
            raise SummaryBindingError(f"completion is missing TN records for {section}")
        if completed.get("path") != audit["records"]["path"] or completed.get(
            "sha256"
        ) != audit["records"]["sha256"]:
            raise SummaryBindingError(f"completion TN record hash mismatch for {section}")
        if completed.get("manifest_binding") != audit["manifest_binding"]:
            raise SummaryBindingError(
                f"completion TN source/derived manifest binding mismatch for {section}"
            )
    if set(tn_records) != set(TN_SECTIONS):
        raise SummaryBindingError("completion TN record groups are not exact")


def _strict_file_record(value: Any, label: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "size_bytes",
        "sha256",
    }:
        raise SummaryBindingError(f"{label} must be an exact file-record object")
    path = value.get("path")
    size = value.get("size_bytes")
    digest = value.get("sha256")
    if not isinstance(path, str) or not path:
        raise SummaryBindingError(f"{label}.path must be non-empty")
    if type(size) is not int or size < 0:
        raise SummaryBindingError(f"{label}.size_bytes must be a non-negative integer")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(char not in "0123456789abcdef" for char in digest.lower())
    ):
        raise SummaryBindingError(f"{label}.sha256 is invalid")
    return {"path": path, "size_bytes": size, "sha256": digest.lower()}


def _binding_metric_records(
    binding: Mapping[str, Any], label: str
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    if (
        binding.get("schema") != SCHEMA
        or binding.get("kind") != "completed_fixed_eval_summary_binding"
        or binding.get("pass") is not True
    ):
        raise SummaryBindingError(f"{label} is not a completed summary binding")
    ref8 = binding.get("ref8")
    tn = binding.get("tn")
    if not isinstance(ref8, Mapping) or not isinstance(tn, Mapping):
        raise SummaryBindingError(f"{label} has no Ref/TN binding sections")
    splits = ref8.get("splits")
    if not isinstance(splits, Mapping) or set(splits) != set(REF_SPLITS):
        raise SummaryBindingError(f"{label} does not contain exactly Ref8")
    if set(tn) != set(TN_SECTIONS):
        raise SummaryBindingError(f"{label} does not contain exact TN sections")
    ref_records = {
        split: _strict_file_record(
            splits[split].get("records") if isinstance(splits[split], Mapping) else None,
            f"{label}.{split}.records",
        )
        for split in REF_SPLITS
    }
    tn_records = {
        section: _strict_file_record(
            tn[section].get("records") if isinstance(tn[section], Mapping) else None,
            f"{label}.{section}.records",
        )
        for section in TN_SECTIONS
    }
    return {"ref": ref_records, "tn": tn_records}


def _file_record_map(values: Any, label: str) -> Dict[str, Dict[str, Any]]:
    if not isinstance(values, list):
        raise SummaryBindingError(f"{label} must be a file-record list")
    result: Dict[str, Dict[str, Any]] = {}
    for index, value in enumerate(values):
        record = _strict_file_record(value, f"{label}[{index}]")
        if record["path"] in result:
            raise SummaryBindingError(f"{label} contains duplicate path {record['path']}")
        result[record["path"]] = record
    return result


def _expected_gate_record_map(
    records: Mapping[str, Mapping[str, Mapping[str, Any]]], section: str
) -> Dict[str, Dict[str, Any]]:
    values = [records["ref"][split] for split in REF_SPLITS]
    values.append(records["tn"][section])
    return {str(value["path"]): dict(value) for value in values}


def validate_final_metric_input_binding(
    *,
    baseline_binding: Mapping[str, Any],
    candidate_binding: Mapping[str, Any],
    fpr_reports: Mapping[str, Mapping[str, Any]],
    dual_gate_reports: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Bind metric reports to summary-audited bytes and their current files."""

    if set(fpr_reports) != set(TN_SECTIONS) or set(dual_gate_reports) != set(
        TN_SECTIONS
    ):
        raise SummaryBindingError("final metric evidence must contain both strict sections")
    bindings = {
        "baseline": _binding_metric_records(baseline_binding, "baseline binding"),
        "candidate": _binding_metric_records(candidate_binding, "candidate binding"),
    }
    section_reports: Dict[str, Any] = {}
    current_records: Dict[str, Dict[str, Any]] = {}
    for section in TN_SECTIONS:
        gate = dual_gate_reports[section]
        if gate.get("schema") != "stageb-dual-gate-v1":
            raise SummaryBindingError(f"{section} dual gate has the wrong schema")
        gate_inputs = gate.get("input_files")
        if (
            not isinstance(gate_inputs, Mapping)
            or gate_inputs.get("identity_is_from_the_same_bytes_used_for_metrics")
            is not True
        ):
            raise SummaryBindingError(
                f"{section} dual gate has no same-byte input identity contract"
            )
        for side in ("baseline", "candidate"):
            observed = _file_record_map(
                gate_inputs.get(side), f"{section} dual gate {side} inputs"
            )
            expected = _expected_gate_record_map(bindings[side], section)
            if observed != expected:
                raise SummaryBindingError(
                    f"{section} dual gate {side} inputs differ from summary binding"
                )
            for path, record in expected.items():
                previous = current_records.get(path)
                if previous is not None and previous != record:
                    raise SummaryBindingError(
                        f"metric input path has conflicting file identities: {path}"
                    )
                current_records[path] = record

        fpr = fpr_reports[section]
        if fpr.get("schema") != "stageb-fpr95-record-comparison-v1":
            raise SummaryBindingError(f"{section} FPR report has the wrong schema")
        fpr_inputs = fpr.get("input_files")
        if (
            not isinstance(fpr_inputs, Mapping)
            or fpr_inputs.get("identity_is_from_the_same_bytes_used_for_metrics")
            is not True
        ):
            raise SummaryBindingError(
                f"{section} FPR report has no same-byte input identity contract"
            )
        for side, input_key in (
            ("baseline", "baseline_records"),
            ("candidate", "candidate_records"),
        ):
            observed = _strict_file_record(
                fpr_inputs.get(input_key), f"{section} FPR {side} records"
            )
            expected = bindings[side]["tn"][section]
            if observed != expected:
                raise SummaryBindingError(
                    f"{section} FPR {side} input differs from summary binding"
                )
        manifest = _strict_file_record(
            fpr_inputs.get("manifest"), f"{section} FPR manifest"
        )
        expected_manifest = _file_record(_resolve(TN_SOURCE_MANIFESTS[section]))
        if manifest != expected_manifest:
            raise SummaryBindingError(
                f"{section} FPR report did not parse the locked current manifest"
            )
        for side, binding in (
            ("baseline", baseline_binding),
            ("candidate", candidate_binding),
        ):
            declared = binding["tn"][section].get("source_manifest_sha256")
            if declared != manifest["sha256"]:
                raise SummaryBindingError(
                    f"{section} {side} TN source binding differs from the FPR manifest"
                )
        validation = fpr.get("validation")
        if not isinstance(validation, Mapping):
            raise SummaryBindingError(f"{section} FPR report has no validation object")
        for side in ("baseline", "candidate"):
            if validation.get(f"{side}_manifest_binding_mode") != "source_to_derived_v1":
                raise SummaryBindingError(
                    f"{section} FPR {side} records do not use the auditable "
                    "source-to-derived manifest binding"
                )
        validation_paths = {
            "manifest_path": manifest,
            "baseline_records": bindings["baseline"]["tn"][section],
            "candidate_records": bindings["candidate"]["tn"][section],
        }
        for key, expected in validation_paths.items():
            value = validation.get(key)
            if not isinstance(value, str) or _resolve(value) != _resolve(expected["path"]):
                raise SummaryBindingError(
                    f"{section} FPR validation.{key} differs from its parsed input"
                )
        if validation.get("manifest_sha256") != manifest["sha256"]:
            raise SummaryBindingError(
                f"{section} FPR validation manifest hash differs from its parsed input"
            )
        section_reports[section] = {
            "dual_gate_inputs_match_summary_binding": True,
            "fpr_inputs_match_summary_binding": True,
            "locked_manifest": manifest,
        }

    expected_current_records = 2 * (len(REF_SPLITS) + len(TN_SECTIONS))
    if len(current_records) != expected_current_records:
        raise SummaryBindingError(
            "baseline and candidate metric inputs must be distinct complete record sets"
        )
    for label, record in sorted(current_records.items()):
        current = _file_record(_resolve(record["path"]))
        if current != record:
            raise SummaryBindingError(
                f"final metric input changed after it was parsed: {label}"
            )
    return {
        "pass": True,
        "same_bytes_parsed_by_metrics_match_summary_binding": True,
        "current_disk_records_match_metric_inputs": True,
        "record_files_per_side": len(REF_SPLITS) + len(TN_SECTIONS),
        "distinct_current_record_files": len(current_records),
        "sections": section_reports,
    }


def audit_evaluation(
    eval_dir: Path,
    expected_checkpoint: Path,
    *,
    trusted_lineage: Path | None = None,
    expected_baseline_checkpoint: Path | None = None,
) -> Dict[str, Any]:
    eval_dir = _resolve(eval_dir)
    expected_checkpoint = _resolve(expected_checkpoint)
    _require_file(expected_checkpoint, "expected checkpoint")
    preflight_path = eval_dir / "protocol_eval_preflight.json"
    complete_path = eval_dir / "protocol_eval_complete.json"
    preflight = _read_json(preflight_path, "evaluation preflight")
    completion = _read_json(complete_path, "evaluation completion")
    if preflight.get("schema") != "stageb-fixed-protocol-v1" or preflight.get(
        "kind"
    ) != "fixed_stageb_eval_preflight":
        raise SummaryBindingError("evaluation preflight has the wrong schema or kind")
    if completion.get("schema") != "stageb-fixed-protocol-v1" or completion.get(
        "kind"
    ) != "fixed_stageb_eval_complete":
        raise SummaryBindingError("evaluation completion has the wrong schema or kind")
    if completion.get("preflight") != _file_record(preflight_path):
        raise SummaryBindingError("evaluation completion is not linked to its current preflight")
    checkpoint_record = _file_record(expected_checkpoint)
    if preflight.get("checkpoint") != checkpoint_record:
        raise SummaryBindingError(
            "expected checkpoint does not exactly match the evaluation preflight"
        )
    evaluation_config = _current_file_identity(
        preflight.get("config"), "evaluation preflight config"
    )
    if (trusted_lineage is None) != (expected_baseline_checkpoint is None):
        raise SummaryBindingError(
            "trusted lineage and expected baseline checkpoint must be provided together"
        )
    runtime = preflight.get("runtime")
    if not isinstance(runtime, Mapping):
        raise SummaryBindingError("evaluation preflight has no runtime object")
    if preflight.get(
        "tn_manifest_derivation_contract"
    ) != tn_manifest_derivation_contract():
        raise SummaryBindingError(
            "evaluation preflight predates the auditable two-layer TN manifest "
            "contract; legacy derived artifacts must be rerun"
        )
    strict_preflight = preflight.get("strict_manifests")
    if not isinstance(strict_preflight, Mapping) or set(strict_preflight) != set(
        TN_SECTIONS
    ):
        raise SummaryBindingError(
            "evaluation preflight does not bind both locked TN source manifests"
        )
    for section, expected_n in TN_SECTIONS.items():
        source_path = _resolve(TN_SOURCE_MANIFESTS[section])
        source_rows, source_file = _read_jsonl_with_record(
            source_path, f"{section} preflight source manifest replay"
        )
        expected = {
            "path": str(source_path),
            "rows": len(source_rows),
            "size_bytes": source_file["size_bytes"],
            "sha256": source_file["sha256"],
        }
        observed = strict_preflight.get(section)
        if not isinstance(observed, Mapping):
            raise SummaryBindingError(
                f"evaluation preflight is missing locked source {section}"
            )
        for key, value in expected.items():
            if observed.get(key) != value:
                raise SummaryBindingError(
                    f"evaluation preflight locked source {section}.{key} changed"
                )
        if len(source_rows) != expected_n:
            raise SummaryBindingError(
                f"locked source {section} has {len(source_rows)} rows, expected {expected_n}"
            )
    run_id = checkpoint_run_prefix(expected_checkpoint)
    ref8 = _audit_ref_section(
        eval_dir,
        checkpoint=expected_checkpoint,
        run_id=run_id,
        runtime=runtime,
    )
    tn_sections = {
        section: _audit_tn_section(
            eval_dir,
            section,
            expected_n,
            checkpoint=expected_checkpoint,
            run_id=run_id,
            runtime=runtime,
        )
        for section, expected_n in TN_SECTIONS.items()
    }
    _validate_completion_outputs(
        completion,
        ref8=ref8,
        tn_sections=tn_sections,
    )
    lineage_binding = (
        _audit_trusted_lineage(
            trusted_lineage,
            candidate_checkpoint=checkpoint_record,
            expected_baseline_checkpoint=expected_baseline_checkpoint,
        )
        if trusted_lineage is not None and expected_baseline_checkpoint is not None
        else None
    )
    if (
        lineage_binding is not None
        and lineage_binding.get("evaluation_config") != evaluation_config
    ):
        raise SummaryBindingError(
            "trusted lineage config does not exactly match the candidate evaluation preflight"
        )
    return {
        "schema": SCHEMA,
        "kind": "completed_fixed_eval_summary_binding",
        "pass": True,
        "eval_dir": str(eval_dir),
        "expected_run_id": run_id,
        "checkpoint": checkpoint_record,
        "evaluation_config": evaluation_config,
        "preflight": _file_record(preflight_path),
        "completion": _file_record(complete_path),
        "official_ref_contract": {
            "method": (
                "canonical evaluator _build_split_jsonl with holdout_level=none and "
                "the fixed official data/phrase-map inputs"
            ),
            "splits": REF_SPLIT_CONTRACT,
            "all_eight_exact_rows_and_manifest_sha256": True,
        },
        "ref8": ref8,
        "tn": tn_sections,
        "lineage_binding": lineage_binding,
        "headroom_replayed_from_per_example_all_query_best_iou": True,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = _resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--expected-checkpoint", required=True)
    parser.add_argument("--trusted-lineage")
    parser.add_argument("--expected-baseline-checkpoint")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        result = audit_evaluation(
            Path(args.eval_dir),
            Path(args.expected_checkpoint),
            trusted_lineage=(
                Path(args.trusted_lineage) if args.trusted_lineage else None
            ),
            expected_baseline_checkpoint=(
                Path(args.expected_baseline_checkpoint)
                if args.expected_baseline_checkpoint
                else None
            ),
        )
        _write_json(Path(args.output), result)
    except SummaryBindingError as error:
        raise SystemExit(f"[FAIL] {error}") from error
    print(f"[OK] fixed evaluation summaries replayed and checkpoint-bound: {_resolve(args.output)}")


if __name__ == "__main__":
    main()
