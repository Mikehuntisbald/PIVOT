#!/usr/bin/env python3
"""Fail-closed Stage-B summary comparison against an explicit baseline manifest.

This is a lightweight directional report for completed Ref8/TN summaries.  The
record-level final acceptance gate remains ``verify_stageb_dual_gate.py``.

Exit codes:
  0: every formal, protocol-matched metric strictly improves;
  1: the formal protocols match, but at least one metric does not improve;
  2: malformed/missing evidence, a protocol mismatch, or a non-formal baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence


BASELINE_SCHEMA = "stageb-summary-gate-baseline-v1"
REPORT_SCHEMA = "stageb-summary-gate-report-v1"
FORMAL_STATUS = "formal_fixed_protocol"
REF_SPLITS = (
    "refcoco_val",
    "refcoco_testA",
    "refcoco_testB",
    "refcocop_val",
    "refcocop_testA",
    "refcocop_testB",
    "refcocog_val",
    "refcocog_test",
)


class SummaryGateError(ValueError):
    pass


def _read_json(path: Path, label: str) -> Dict[str, Any]:
    if not path.is_file():
        raise SummaryGateError(f"{label} is missing or is not a file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SummaryGateError(f"could not read {label} {path}: {error}") from error
    if not isinstance(value, dict):
        raise SummaryGateError(f"{label} must be a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise SummaryGateError(f"evidence file is missing: {path}")
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _count_nonempty_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def _resolve_evidence_path(value: Any, base_dir: Path, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SummaryGateError(f"{label}.path must be a non-empty string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _validate_sha(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise SummaryGateError(f"{label} must be a SHA-256 string")
    normalized = value.lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise SummaryGateError(f"{label} must be a lowercase hexadecimal SHA-256")
    return normalized


def _verify_file_record(value: Any, base_dir: Path, label: str) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SummaryGateError(f"{label} must be a file-record object")
    path = _resolve_evidence_path(value.get("path"), base_dir, label)
    current = _file_record(path)
    expected_sha = _validate_sha(value.get("sha256"), f"{label}.sha256")
    if current["sha256"] != expected_sha:
        raise SummaryGateError(f"{label} SHA-256 mismatch: {path}")
    if "size_bytes" in value and _exact_int(value["size_bytes"], f"{label}.size_bytes") != current[
        "size_bytes"
    ]:
        raise SummaryGateError(f"{label} size mismatch: {path}")
    return current


def _exact_int(value: Any, label: str) -> int:
    if type(value) is not int:
        raise SummaryGateError(f"{label} must be an exact integer")
    return int(value)


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SummaryGateError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise SummaryGateError(f"{label} must be finite")
    return result


def _unit_float(value: Any, label: str) -> float:
    result = _finite_float(value, label)
    if result < 0.0 or result > 1.0:
        raise SummaryGateError(f"{label} must be in [0, 1]")
    return result


def _summary_rows(summary: Mapping[str, Any], label: str) -> tuple[list[Any], list[Any]]:
    ref_rows = summary.get("refcoco")
    tn_rows = summary.get("tn")
    if not isinstance(ref_rows, list) or not isinstance(tn_rows, list):
        raise SummaryGateError(f"{label} must contain refcoco and tn lists")
    return ref_rows, tn_rows


def _validate_ref_row(row: Any, split: str, label: str) -> Dict[str, Any]:
    if not isinstance(row, Mapping):
        raise SummaryGateError(f"{label} must be an object")
    if row.get("dataset") != split:
        raise SummaryGateError(f"{label}.dataset must be {split!r}")
    n = _exact_int(row.get("manifest_n"), f"{label}.manifest_n")
    if n <= 0 or _exact_int(row.get("num_expressions"), f"{label}.num_expressions") != n:
        raise SummaryGateError(f"{label} does not cover its complete positive manifest")
    if _exact_int(row.get("valid_mask_expressions"), f"{label}.valid_mask_expressions") != n:
        raise SummaryGateError(f"{label} has invalid positive masks")
    for field in ("invalid_mask_expressions", "invalid_records", "max_batches"):
        if _exact_int(row.get(field), f"{label}.{field}") != 0:
            raise SummaryGateError(f"{label}.{field} must be zero for a complete fixed evaluation")
    if not isinstance(row.get("records_jsonl"), str) or not row["records_jsonl"]:
        raise SummaryGateError(f"{label}.records_jsonl is required")
    checkpoint = row.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise SummaryGateError(f"{label}.checkpoint is required")
    return {
        "acc50": _unit_float(row.get("acc50"), f"{label}.acc50"),
        "manifest_n": n,
        "manifest_sha256": _validate_sha(
            row.get("manifest_sha256"), f"{label}.manifest_sha256"
        ),
        "seed": _exact_int(row.get("seed"), f"{label}.seed"),
        "checkpoint": checkpoint,
    }


def _load_candidate_ref(path: Path) -> Dict[str, Any]:
    summary = _read_json(path, "candidate Ref8 summary")
    ref_rows, tn_rows = _summary_rows(summary, "candidate Ref8 summary")
    if tn_rows:
        raise SummaryGateError("candidate Ref8 summary must not contain TN rows")
    observed = [row.get("dataset") if isinstance(row, Mapping) else None for row in ref_rows]
    if observed != list(REF_SPLITS):
        raise SummaryGateError(
            f"candidate Ref8 split order/completeness mismatch: expected {list(REF_SPLITS)}, "
            f"got {observed}"
        )
    rows = {
        split: _validate_ref_row(row, split, f"candidate Ref8 {split}")
        for split, row in zip(REF_SPLITS, ref_rows)
    }
    checkpoints = {row["checkpoint"] for row in rows.values()}
    if len(checkpoints) != 1:
        raise SummaryGateError("candidate Ref8 rows do not use one checkpoint")
    return {"file": _file_record(path), "rows": rows, "checkpoint": next(iter(checkpoints))}


def _load_candidate_tn(section: str, path: Path) -> Dict[str, Any]:
    summary = _read_json(path, f"candidate TN {section} summary")
    ref_rows, tn_rows = _summary_rows(summary, f"candidate TN {section} summary")
    if ref_rows or len(tn_rows) != 1 or not isinstance(tn_rows[0], Mapping):
        raise SummaryGateError(
            f"candidate TN {section} summary must contain exactly one TN row and no Ref rows"
        )
    row = tn_rows[0]
    label = f"candidate TN {section}"
    n = _exact_int(row.get("manifest_n"), f"{label}.manifest_n")
    if n <= 0 or _exact_int(row.get("num_pairs"), f"{label}.num_pairs") != n:
        raise SummaryGateError(f"{label} does not cover its complete TN manifest")
    for field in (
        "invalid_positive_pairs",
        "invalid_negative_pairs",
        "invalid_records",
        "max_batches",
    ):
        if _exact_int(row.get(field), f"{label}.{field}") != 0:
            raise SummaryGateError(f"{label}.{field} must be zero for a complete fixed evaluation")
    actual_tpr = _unit_float(row.get("actual_tpr_at_95tpr"), f"{label}.actual_tpr_at_95tpr")
    if actual_tpr < 0.95:
        raise SummaryGateError(f"{label} did not attain the requested 0.95 TPR")
    if not isinstance(row.get("records_jsonl"), str) or not row["records_jsonl"]:
        raise SummaryGateError(f"{label}.records_jsonl is required")
    checkpoint = row.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise SummaryGateError(f"{label}.checkpoint is required")
    return {
        "file": _file_record(path),
        "fpr95tpr": _unit_float(row.get("fpr95tpr"), f"{label}.fpr95tpr"),
        "manifest_n": n,
        "manifest_sha256": _validate_sha(
            row.get("source_manifest_sha256"),
            f"{label}.source_manifest_sha256",
        ),
        "derived_manifest_sha256": _validate_sha(
            row.get("manifest_sha256"), f"{label}.manifest_sha256"
        ),
        "manifest_binding_sha256": _validate_sha(
            row.get("manifest_binding_sha256"),
            f"{label}.manifest_binding_sha256",
        ),
        "seed": _exact_int(row.get("seed"), f"{label}.seed"),
        "checkpoint": checkpoint,
        "actual_tpr_at_95tpr": actual_tpr,
    }


def _ref_source_rows(summary: Mapping[str, Any], label: str) -> Dict[str, Mapping[str, Any]]:
    ref_rows, tn_rows = _summary_rows(summary, label)
    if tn_rows or len(ref_rows) != len(REF_SPLITS):
        raise SummaryGateError(f"{label} must contain exactly the eight Ref rows and no TN rows")
    result: Dict[str, Mapping[str, Any]] = {}
    for row in ref_rows:
        if not isinstance(row, Mapping) or row.get("dataset") not in REF_SPLITS:
            raise SummaryGateError(f"{label} contains an unknown Ref row")
        split = str(row["dataset"])
        if split in result:
            raise SummaryGateError(f"{label} contains duplicate split {split}")
        result[split] = row
    if set(result) != set(REF_SPLITS):
        raise SummaryGateError(f"{label} does not contain the exact Ref8 split set")
    return result


def _load_baseline_manifest(path: Path) -> Dict[str, Any]:
    manifest = _read_json(path, "baseline manifest")
    if manifest.get("schema") != BASELINE_SCHEMA:
        raise SummaryGateError(f"baseline manifest schema must be {BASELINE_SCHEMA!r}")
    baseline_id = manifest.get("baseline_id")
    if not isinstance(baseline_id, str) or not baseline_id:
        raise SummaryGateError("baseline manifest baseline_id is required")
    base_dir = path.resolve().parent
    checkpoint = _verify_file_record(manifest.get("checkpoint"), base_dir, "baseline checkpoint")

    ref = manifest.get("ref8")
    if not isinstance(ref, Mapping):
        raise SummaryGateError("baseline manifest ref8 must be an object")
    status = ref.get("status")
    if not isinstance(status, str) or not status:
        raise SummaryGateError("baseline ref8.status is required")
    ref_reason = ref.get("reason")
    if status != FORMAL_STATUS and (not isinstance(ref_reason, str) or not ref_reason):
        raise SummaryGateError("non-formal baseline ref8 must state a reason")
    source_file = _verify_file_record(ref.get("source_summary"), base_dir, "baseline Ref8 summary")
    source_summary = _read_json(Path(source_file["path"]), "baseline Ref8 summary")
    source_rows = _ref_source_rows(source_summary, "baseline Ref8 summary")
    metrics = ref.get("metrics")
    protocol = ref.get("protocol")
    if not isinstance(metrics, Mapping) or set(metrics) != set(REF_SPLITS):
        raise SummaryGateError("baseline ref8.metrics must contain the exact Ref8 split set")
    if not isinstance(protocol, Mapping) or not isinstance(protocol.get("id"), str):
        raise SummaryGateError("baseline ref8.protocol.id is required")
    contracts = protocol.get("splits")
    if not isinstance(contracts, Mapping) or set(contracts) != set(REF_SPLITS):
        raise SummaryGateError("baseline ref8.protocol.splits must contain the exact Ref8 split set")

    parsed_metrics: Dict[str, float] = {}
    parsed_contracts: Dict[str, Dict[str, Any]] = {}
    for split in REF_SPLITS:
        metric = metrics[split]
        contract = contracts[split]
        if not isinstance(metric, Mapping) or not isinstance(contract, Mapping):
            raise SummaryGateError(f"baseline {split} metric/contract must be objects")
        acc50 = _unit_float(metric.get("acc50"), f"baseline {split}.acc50")
        n = _exact_int(contract.get("manifest_n"), f"baseline {split}.manifest_n")
        if n <= 0:
            raise SummaryGateError(f"baseline {split}.manifest_n must be positive")
        manifest_sha = _validate_sha(
            contract.get("manifest_sha256"), f"baseline {split}.manifest_sha256"
        )
        seed = _exact_int(contract.get("seed"), f"baseline {split}.seed")
        source_row = source_rows[split]
        if _unit_float(source_row.get("acc50"), f"baseline source {split}.acc50") != acc50:
            raise SummaryGateError(f"baseline {split}.acc50 differs from its source summary")
        if _exact_int(source_row.get("num_expressions"), f"baseline source {split}.num_expressions") != n:
            raise SummaryGateError(f"baseline {split}.manifest_n differs from its source summary")
        if _exact_int(source_row.get("seed"), f"baseline source {split}.seed") != seed:
            raise SummaryGateError(f"baseline {split}.seed differs from its source summary")
        if status == FORMAL_STATUS:
            if source_row.get("manifest_sha256") != manifest_sha or source_row.get("manifest_n") != n:
                raise SummaryGateError(
                    f"formal baseline {split} source summary is not bound to its declared manifest"
                )
            if (
                source_row.get("valid_mask_expressions") != n
                or source_row.get("invalid_mask_expressions") != 0
                or source_row.get("invalid_records") != 0
                or source_row.get("max_batches") != 0
                or not isinstance(source_row.get("records_jsonl"), str)
                or not source_row["records_jsonl"]
            ):
                raise SummaryGateError(f"formal baseline {split} is incomplete or invalid")
        parsed_metrics[split] = acc50
        parsed_contracts[split] = {
            "manifest_n": n,
            "manifest_sha256": manifest_sha,
            "seed": seed,
        }

    tn = manifest.get("tn")
    if not isinstance(tn, Mapping) or not tn:
        raise SummaryGateError("baseline manifest tn must declare every required TN section")
    parsed_tn: Dict[str, Dict[str, Any]] = {}
    for section, value in tn.items():
        if not isinstance(section, str) or not isinstance(value, Mapping):
            raise SummaryGateError("baseline TN section names/values are invalid")
        section_status = value.get("status")
        if not isinstance(section_status, str) or not section_status:
            raise SummaryGateError(f"baseline TN {section}.status is required")
        parsed: Dict[str, Any] = {
            "status": section_status,
            "reason": value.get("reason"),
        }
        if section_status == FORMAL_STATUS:
            source = _verify_file_record(
                value.get("source_summary"), base_dir, f"baseline TN {section} summary"
            )
            summary = _read_json(Path(source["path"]), f"baseline TN {section} summary")
            source_ref, source_tn = _summary_rows(summary, f"baseline TN {section} summary")
            if source_ref or len(source_tn) != 1 or not isinstance(source_tn[0], Mapping):
                raise SummaryGateError(f"baseline TN {section} source summary has the wrong shape")
            section_protocol = value.get("protocol")
            if not isinstance(section_protocol, Mapping) or not isinstance(
                section_protocol.get("id"), str
            ):
                raise SummaryGateError(f"baseline TN {section}.protocol.id is required")
            n = _exact_int(
                section_protocol.get("manifest_n"), f"baseline TN {section}.manifest_n"
            )
            manifest_sha = _validate_sha(
                section_protocol.get("manifest_sha256"),
                f"baseline TN {section}.manifest_sha256",
            )
            seed = _exact_int(section_protocol.get("seed"), f"baseline TN {section}.seed")
            target_tpr = _finite_float(
                section_protocol.get("target_tpr"), f"baseline TN {section}.target_tpr"
            )
            if target_tpr != 0.95:
                raise SummaryGateError(f"baseline TN {section} target_tpr must be exactly 0.95")
            fpr = _unit_float(value.get("fpr95tpr"), f"baseline TN {section}.fpr95tpr")
            source_row = source_tn[0]
            for field, expected in (
                ("manifest_n", n),
                ("num_pairs", n),
                ("seed", seed),
                ("invalid_positive_pairs", 0),
                ("invalid_negative_pairs", 0),
                ("invalid_records", 0),
                ("max_batches", 0),
            ):
                if source_row.get(field) != expected:
                    raise SummaryGateError(
                        f"formal baseline TN {section} source field {field} mismatch"
                    )
            if source_row.get("source_manifest_sha256") != manifest_sha:
                raise SummaryGateError(
                    f"formal baseline TN {section} locked source manifest SHA mismatch"
                )
            _validate_sha(
                source_row.get("manifest_sha256"),
                f"formal baseline TN {section} derived manifest SHA",
            )
            _validate_sha(
                source_row.get("manifest_binding_sha256"),
                f"formal baseline TN {section} manifest binding SHA",
            )
            if _unit_float(source_row.get("fpr95tpr"), f"baseline source TN {section}.fpr95tpr") != fpr:
                raise SummaryGateError(f"baseline TN {section} metric differs from its source summary")
            if _unit_float(
                source_row.get("actual_tpr_at_95tpr"),
                f"baseline source TN {section}.actual_tpr_at_95tpr",
            ) < 0.95:
                raise SummaryGateError(f"formal baseline TN {section} did not attain 0.95 TPR")
            if not isinstance(source_row.get("records_jsonl"), str) or not source_row[
                "records_jsonl"
            ]:
                raise SummaryGateError(f"formal baseline TN {section} has no record binding")
            parsed.update(
                {
                    "source_summary": source,
                    "protocol_id": section_protocol["id"],
                    "manifest_n": n,
                    "manifest_sha256": manifest_sha,
                    "seed": seed,
                    "target_tpr": target_tpr,
                    "fpr95tpr": fpr,
                }
            )
        else:
            if not isinstance(value.get("reason"), str) or not value["reason"]:
                raise SummaryGateError(f"non-formal baseline TN {section} must state a reason")
            if "fpr95tpr" in value:
                raise SummaryGateError(
                    f"non-formal baseline TN {section} must not expose a formal comparator"
                )
        parsed_tn[section] = parsed

    diagnostics_value = manifest.get("diagnostics", {})
    if not isinstance(diagnostics_value, Mapping):
        raise SummaryGateError("baseline manifest diagnostics must be an object")
    diagnostics: Dict[str, Any] = {}
    for name, value in diagnostics_value.items():
        if not isinstance(name, str) or not isinstance(value, Mapping):
            raise SummaryGateError("baseline diagnostics names/values are invalid")
        label = f"baseline diagnostic {name}"
        if value.get("kind") != "tn_summary_reference_v1":
            raise SummaryGateError(f"{label}.kind is unsupported")
        if value.get("status") != "diagnostic_only" or value.get(
            "formal_gate_eligible"
        ) is not False:
            raise SummaryGateError(f"{label} must be explicitly diagnostic-only")
        source_summary_file = _verify_file_record(
            value.get("source_summary"), base_dir, f"{label} summary"
        )
        source_manifest_file = _verify_file_record(
            value.get("source_manifest"), base_dir, f"{label} manifest"
        )
        source_summary = _read_json(Path(source_summary_file["path"]), f"{label} summary")
        diagnostic_ref, diagnostic_tn = _summary_rows(source_summary, f"{label} summary")
        if len(diagnostic_tn) != 1 or not isinstance(diagnostic_tn[0], Mapping):
            raise SummaryGateError(f"{label} summary must contain exactly one TN row")
        source_row = diagnostic_tn[0]
        fpr = _unit_float(value.get("fpr95tpr"), f"{label}.fpr95tpr")
        actual_tpr = _unit_float(
            value.get("actual_tpr_at_95tpr"), f"{label}.actual_tpr_at_95tpr"
        )
        source_rows = _exact_int(value.get("num_source_rows"), f"{label}.num_source_rows")
        valid_pairs = _exact_int(value.get("num_valid_pairs"), f"{label}.num_valid_pairs")
        invalid_positive = _exact_int(
            value.get("invalid_positive_pairs"), f"{label}.invalid_positive_pairs"
        )
        if _count_nonempty_lines(Path(source_manifest_file["path"])) != source_rows:
            raise SummaryGateError(f"{label} source manifest row count mismatch")
        for field, expected in (
            ("fpr95tpr", fpr),
            ("actual_tpr_at_95tpr", actual_tpr),
            ("num_pairs", valid_pairs),
            ("invalid_positive_pairs", invalid_positive),
        ):
            if source_row.get(field) != expected:
                raise SummaryGateError(f"{label} differs from source summary field {field}")
        diagnostics[name] = {
            "kind": value["kind"],
            "status": "diagnostic_only",
            "formal_gate_eligible": False,
            "protocol_id": value.get("protocol_id"),
            "fpr95tpr": fpr,
            "actual_tpr_at_95tpr": actual_tpr,
            "num_source_rows": source_rows,
            "num_valid_pairs": valid_pairs,
            "invalid_positive_pairs": invalid_positive,
            "source_summary": source_summary_file,
            "source_manifest": source_manifest_file,
            "source_ref_rows_ignored": len(diagnostic_ref),
        }

    return {
        "file": _file_record(path),
        "baseline_id": baseline_id,
        "checkpoint": checkpoint,
        "ref8": {
            "status": status,
            "reason": ref_reason,
            "source_summary": source_file,
            "protocol_id": protocol["id"],
            "metrics": parsed_metrics,
            "contracts": parsed_contracts,
        },
        "tn": parsed_tn,
        "diagnostics": diagnostics,
    }


def compare_summary_gate(
    *,
    baseline_manifest: Path,
    candidate_ref_summary: Path,
    candidate_tn_summaries: Mapping[str, Path],
) -> Dict[str, Any]:
    baseline = _load_baseline_manifest(baseline_manifest)
    candidate_ref = _load_candidate_ref(candidate_ref_summary)
    expected_tn = set(baseline["tn"])
    observed_tn = set(candidate_tn_summaries)
    if observed_tn != expected_tn:
        raise SummaryGateError(
            f"candidate TN sections differ from the baseline manifest: "
            f"missing={sorted(expected_tn - observed_tn)}, extra={sorted(observed_tn - expected_tn)}"
        )
    candidate_tn = {
        section: _load_candidate_tn(section, candidate_tn_summaries[section])
        for section in sorted(expected_tn)
    }
    candidate_checkpoints = {candidate_ref["checkpoint"]}.union(
        row["checkpoint"] for row in candidate_tn.values()
    )
    if len(candidate_checkpoints) != 1:
        raise SummaryGateError("candidate Ref8/TN summaries do not use the same checkpoint")

    ineligibility: list[str] = []
    ref_status = baseline["ref8"]["status"]
    if ref_status != FORMAL_STATUS:
        ineligibility.append(f"baseline Ref8 status is {ref_status!r}, not {FORMAL_STATUS!r}")
    ref_report: Dict[str, Any] = {}
    ref_numeric_pass = True
    ref_formal_pass = True
    ref_protocol_pass = True
    for split in REF_SPLITS:
        base_value = baseline["ref8"]["metrics"][split]
        candidate_value = candidate_ref["rows"][split]["acc50"]
        contract = baseline["ref8"]["contracts"][split]
        candidate_contract = {
            key: candidate_ref["rows"][split][key]
            for key in ("manifest_n", "manifest_sha256", "seed")
        }
        protocol_match = candidate_contract == contract
        if not protocol_match:
            ref_protocol_pass = False
            ineligibility.append(f"Ref split {split} protocol contract mismatch")
        numeric = candidate_value > base_value
        ref_numeric_pass = ref_numeric_pass and numeric
        eligible = ref_status == FORMAL_STATUS and protocol_match
        if eligible:
            ref_formal_pass = ref_formal_pass and numeric
        else:
            ref_formal_pass = False
        ref_report[split] = {
            "baseline_acc50": base_value,
            "candidate_acc50": candidate_value,
            "candidate_minus_baseline_acc50": candidate_value - base_value,
            "numeric_strictly_higher": numeric,
            "protocol_match": protocol_match,
            "baseline_protocol_contract": contract,
            "candidate_protocol_contract": candidate_contract,
            "strictly_higher": numeric if eligible else None,
        }

    tn_report: Dict[str, Any] = {}
    tn_formal_pass = True
    tn_eligible = True
    for section in sorted(expected_tn):
        base = baseline["tn"][section]
        candidate = candidate_tn[section]
        status = base["status"]
        if status != FORMAL_STATUS:
            tn_eligible = False
            tn_formal_pass = False
            ineligibility.append(
                f"baseline TN {section} status is {status!r}, not {FORMAL_STATUS!r}: "
                f"{base.get('reason') or 'no reason provided'}"
            )
            tn_report[section] = {
                "baseline_fpr95tpr": None,
                "candidate_fpr95tpr": candidate["fpr95tpr"],
                "candidate_minus_baseline_fpr95tpr": None,
                "protocol_match": False,
                "candidate_protocol_contract": {
                    key: candidate[key]
                    for key in ("manifest_n", "manifest_sha256", "seed")
                },
                "strictly_lower": None,
                "reason": base.get("reason"),
            }
            continue
        protocol_match = all(
            candidate[key] == base[key]
            for key in ("manifest_n", "manifest_sha256", "seed")
        )
        if not protocol_match:
            tn_eligible = False
            tn_formal_pass = False
            ineligibility.append(f"TN section {section} protocol contract mismatch")
        numeric = candidate["fpr95tpr"] < base["fpr95tpr"]
        eligible = protocol_match
        if eligible:
            tn_formal_pass = tn_formal_pass and numeric
        baseline_value = base["fpr95tpr"]
        tn_report[section] = {
            "baseline_fpr95tpr": baseline_value,
            "candidate_fpr95tpr": candidate["fpr95tpr"],
            "candidate_minus_baseline_fpr95tpr": candidate["fpr95tpr"] - baseline_value,
            "numeric_strictly_lower": numeric,
            "protocol_match": protocol_match,
            "baseline_protocol_contract": {
                key: base[key] for key in ("manifest_n", "manifest_sha256", "seed")
            },
            "candidate_protocol_contract": {
                key: candidate[key] for key in ("manifest_n", "manifest_sha256", "seed")
            },
            "strictly_lower": numeric if eligible else None,
        }

    ref_eligible = ref_status == FORMAL_STATUS and ref_protocol_pass
    eligible = (
        ref_eligible
        and tn_eligible
        and all(baseline["tn"][section]["status"] == FORMAL_STATUS for section in expected_tn)
    )
    passed = bool(eligible and ref_formal_pass and tn_formal_pass)
    return {
        "schema": REPORT_SCHEMA,
        "validation": {"pass": True, "errors": []},
        "baseline": {
            "id": baseline["baseline_id"],
            "manifest": baseline["file"],
            "checkpoint": baseline["checkpoint"],
            "ref8_status": ref_status,
            "ref8_status_reason": baseline["ref8"]["reason"],
            "tn_status": {
                section: baseline["tn"][section]["status"] for section in sorted(expected_tn)
            },
            "diagnostics": baseline["diagnostics"],
        },
        "candidate": {
            "checkpoint": next(iter(candidate_checkpoints)),
            "ref_summary": candidate_ref["file"],
            "tn_summaries": {
                section: candidate_tn[section]["file"] for section in sorted(expected_tn)
            },
        },
        "ref8": {
            "protocol_id": baseline["ref8"]["protocol_id"],
            "eligible": ref_eligible,
            "numeric_all_strictly_higher": ref_numeric_pass,
            "all_strictly_higher": ref_formal_pass if ref_eligible else None,
            "splits": ref_report,
        },
        "tn": tn_report,
        "gate": {
            "eligible": eligible,
            "pass": passed,
            "ref8_pass": ref_formal_pass if eligible else None,
            "tn_pass": tn_formal_pass if eligible else None,
            "ineligibility_reasons": sorted(set(ineligibility)),
        },
    }


def _parse_section_paths(values: Sequence[str]) -> Dict[str, Path]:
    result: Dict[str, Path] = {}
    for value in values:
        section, separator, raw_path = value.partition("=")
        if not separator or not section or not raw_path:
            raise SummaryGateError(
                f"invalid --candidate-tn-summary {value!r}; expected SECTION=PATH"
            )
        if section in result:
            raise SummaryGateError(f"duplicate candidate TN section: {section}")
        result[section] = Path(raw_path)
    return result


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(report), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-manifest", required=True)
    parser.add_argument("--candidate-ref-summary", required=True)
    parser.add_argument(
        "--candidate-tn-summary",
        action="append",
        default=[],
        metavar="SECTION=PATH",
        help="Repeat once for every TN section declared by the baseline manifest.",
    )
    parser.add_argument("--output", help="Optional JSON report path; inputs are never modified.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = Path(args.output) if args.output else None
    try:
        tn_paths = _parse_section_paths(args.candidate_tn_summary)
        report = compare_summary_gate(
            baseline_manifest=Path(args.baseline_manifest),
            candidate_ref_summary=Path(args.candidate_ref_summary),
            candidate_tn_summaries=tn_paths,
        )
    except SummaryGateError as error:
        report = {
            "schema": REPORT_SCHEMA,
            "validation": {"pass": False, "errors": [str(error)]},
            "gate": {"eligible": False, "pass": False},
        }
        if output is not None:
            _write_report(output, report)
        print(f"[FAIL] {error}", file=sys.stderr)
        return 2
    if output is not None:
        _write_report(output, report)
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True))
    if not report["gate"]["eligible"]:
        return 2
    return 0 if report["gate"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
