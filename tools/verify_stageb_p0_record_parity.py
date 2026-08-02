#!/usr/bin/env python3
"""Require exact per-record parity between pure baseline and zero-init P0."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


IDENTITY_FIELDS = (
    "task",
    "manifest_key",
    "manifest_sha256",
    "manifest_n",
    "manifest_index",
    "sample_id",
    "split",
    "image_id",
    "ann_id",
    "ref_id",
    "sent_id",
    "valid",
)
VALUE_FIELDS = {
    "tn": ("pos_score", "neg_score", "pos_iou", "neg_iou"),
    "ref": ("top1_iou", "all_query_best_iou", "correct50"),
}


class ParityError(RuntimeError):
    pass


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ParityError(f"invalid JSON at {path}:{line_number}: {error}") from error
            if not isinstance(row, dict):
                raise ParityError(f"expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def _load_groups(path: Path) -> Dict[str, list[Dict[str, Any]]]:
    if not path.is_dir():
        raise ParityError(f"record directory is missing: {path}")
    files = sorted(path.glob("*.records.jsonl"))
    if not files:
        raise ParityError(f"record directory is empty: {path}")
    groups: Dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for file_path in files:
        for row in _read_jsonl(file_path):
            key = row.get("manifest_key")
            if not isinstance(key, str) or not key:
                raise ParityError(f"record has no manifest_key: {file_path}")
            groups[key].append(row)
    return dict(groups)


def _equal_value(left: Any, right: Any, *, atol: float) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        left_float = float(left)
        right_float = float(right)
        return (
            math.isfinite(left_float)
            and math.isfinite(right_float)
            and abs(left_float - right_float) <= atol
        )
    return left == right


def compare_record_groups(
    baseline: Mapping[str, Sequence[Mapping[str, Any]]],
    p0: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    atol: float = 0.0,
    compare_values: bool = True,
) -> Dict[str, Any]:
    if set(baseline) != set(p0):
        raise ParityError(
            f"manifest groups differ: baseline={sorted(baseline)}, p0={sorted(p0)}"
        )
    report: Dict[str, Any] = {}
    for key in sorted(baseline):
        left_rows = list(baseline[key])
        right_rows = list(p0[key])
        if len(left_rows) != len(right_rows):
            raise ParityError(
                f"{key}: record count differs: baseline={len(left_rows)}, p0={len(right_rows)}"
            )
        if not left_rows:
            raise ParityError(f"{key}: empty record group")
        expected_n = left_rows[0].get("manifest_n")
        if expected_n != len(left_rows):
            raise ParityError(f"{key}: records do not cover the complete manifest")
        task = left_rows[0].get("task")
        if task not in VALUE_FIELDS:
            raise ParityError(f"{key}: unsupported task {task!r}")
        for index, (left, right) in enumerate(zip(left_rows, right_rows)):
            for field in IDENTITY_FIELDS:
                if left.get(field) != right.get(field):
                    raise ParityError(
                        f"{key}[{index}] identity mismatch for {field}: "
                        f"baseline={left.get(field)!r}, p0={right.get(field)!r}"
                    )
            if left.get("valid") is not True or right.get("valid") is not True:
                raise ParityError(f"{key}[{index}] is invalid; parity requires full coverage")
            if compare_values:
                for field in VALUE_FIELDS[str(task)]:
                    if field not in left and field not in right:
                        continue
                    if not _equal_value(left.get(field), right.get(field), atol=atol):
                        raise ParityError(
                            f"{key}[{index}] value mismatch for {field}: "
                            f"baseline={left.get(field)!r}, p0={right.get(field)!r}, atol={atol}"
                        )
        report[key] = {
            "task": task,
            "records": len(left_rows),
            "valid": len(left_rows),
            "identity_aligned": True,
            "values_equal": True if compare_values else None,
        }
    return report


def _require_complete_eval(path: Path) -> Dict[str, Any]:
    complete = path / "protocol_eval_complete.json"
    if not complete.is_file():
        raise ParityError(f"fixed-protocol evaluation is incomplete: {complete}")
    try:
        payload = json.loads(complete.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ParityError(f"could not read {complete}: {error}") from error
    if not isinstance(payload, dict):
        raise ParityError(f"evaluation completion audit is not an object: {complete}")
    return payload


def evaluate_parity(
    baseline_dir: Path,
    p0_dir: Path,
    *,
    atol: float,
) -> Dict[str, Any]:
    _require_complete_eval(baseline_dir)
    _require_complete_eval(p0_dir)
    sections = {
        "ref8": None,
        "strict2031": 2031,
        "strict1607": 1607,
    }
    report: Dict[str, Any] = {}
    for section, expected_n in sections.items():
        baseline = _load_groups(baseline_dir / section / "per_example_records")
        p0 = _load_groups(p0_dir / section / "per_example_records")
        comparison = compare_record_groups(
            baseline,
            p0,
            atol=atol,
            compare_values=True,
        )
        if expected_n is not None:
            if set(comparison) != {"tn_global"}:
                raise ParityError(f"{section}: expected exactly the tn_global record group")
            observed = int(comparison["tn_global"]["records"])
            if observed != expected_n:
                raise ParityError(
                    f"{section}: expected {expected_n}/{expected_n} valid records, got {observed}"
                )
        report[section] = comparison
    if len(report["ref8"]) != 8:
        raise ParityError(
            f"ref8 parity requires all eight official splits, got {sorted(report['ref8'])}"
        )
    return {
        "schema": "stageb-gdino-adapter-p0-record-parity-v1",
        "pass": True,
        "absolute_tolerance": float(atol),
        "baseline_eval_dir": str(baseline_dir.resolve()),
        "p0_eval_dir": str(p0_dir.resolve()),
        "sections": report,
        "strict2031_valid": 2031,
        "strict2031_identity_aligned": True,
    }


def evaluate_identity_alignment(
    baseline_dir: Path,
    candidate_dir: Path,
) -> Dict[str, Any]:
    _require_complete_eval(baseline_dir)
    _require_complete_eval(candidate_dir)
    sections = {
        "ref8": None,
        "strict2031": 2031,
        "strict1607": 1607,
    }
    report: Dict[str, Any] = {}
    for section, expected_n in sections.items():
        baseline = _load_groups(baseline_dir / section / "per_example_records")
        candidate = _load_groups(candidate_dir / section / "per_example_records")
        comparison = compare_record_groups(
            baseline,
            candidate,
            compare_values=False,
        )
        if expected_n is not None:
            if set(comparison) != {"tn_global"}:
                raise ParityError(f"{section}: expected exactly the tn_global record group")
            observed = int(comparison["tn_global"]["records"])
            if observed != expected_n:
                raise ParityError(
                    f"{section}: expected {expected_n}/{expected_n} valid records, got {observed}"
                )
        report[section] = comparison
    if len(report["ref8"]) != 8:
        raise ParityError("identity audit requires all eight official Ref splits")
    return {
        "schema": "stageb-gdino-adapter-paired-record-identity-v1",
        "pass": True,
        "baseline_eval_dir": str(baseline_dir.resolve()),
        "candidate_eval_dir": str(candidate_dir.resolve()),
        "sections": report,
        "strict2031_valid": 2031,
        "strict2031_identity_aligned": True,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-eval-dir", required=True)
    candidate_group = parser.add_mutually_exclusive_group(required=True)
    candidate_group.add_argument("--p0-eval-dir")
    candidate_group.add_argument("--candidate-eval-dir")
    parser.add_argument("--identity-only", action="store_true")
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not math.isfinite(args.atol) or args.atol < 0.0:
        raise SystemExit("[FAIL] --atol must be finite and non-negative")
    try:
        candidate_dir = Path(args.p0_eval_dir or args.candidate_eval_dir)
        if args.identity_only:
            report = evaluate_identity_alignment(
                Path(args.baseline_eval_dir),
                candidate_dir,
            )
        else:
            if not args.p0_eval_dir:
                raise ParityError("exact P0 parity requires --p0-eval-dir")
            report = evaluate_parity(
                Path(args.baseline_eval_dir),
                candidate_dir,
                atol=float(args.atol),
            )
        _write_json(Path(args.output), report)
        label = "paired record identity" if args.identity_only else "exact P0 record parity"
        print(f"[OK] {label} passed: {args.output}")
    except ParityError as error:
        raise SystemExit(f"[FAIL] {error}") from error


if __name__ == "__main__":
    main()
