#!/usr/bin/env python3
"""Select a U2-v2 milestone using only the preregistered Ref val gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping


class SelectionError(RuntimeError):
    pass


SPLITS = ("refcoco_val", "refcocop_val", "refcocog_val")


def _load(path: Path) -> Any:
    return json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))


def _file(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def _rows(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows = summary.get("refcoco")
    if not isinstance(rows, list):
        raise SelectionError("summary lacks Ref rows")
    return rows


def _by_checkpoint(summary: Mapping[str, Any]) -> dict[str, dict[str, Mapping[str, Any]]]:
    grouped: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in _rows(summary):
        checkpoint = str(row.get("checkpoint", ""))
        split = str(row.get("dataset", ""))
        if split not in SPLITS or not checkpoint:
            continue
        if split in grouped.setdefault(checkpoint, {}):
            raise SelectionError(f"duplicate row for {checkpoint} {split}")
        grouped[checkpoint][split] = row
    if not grouped or any(set(rows) != set(SPLITS) for rows in grouped.values()):
        raise SelectionError("each checkpoint must have exactly the three val splits")
    return grouped


def _micro(rows: Mapping[str, Mapping[str, Any]]) -> float:
    correct = sum(float(row["acc50"]) * int(row["num_expressions"]) for row in rows.values())
    total = sum(int(row["num_expressions"]) for row in rows.values())
    return correct / total


def _records(row: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    path = Path(str(row.get("records_jsonl", "")))
    if not path.is_absolute():
        path = Path.cwd() / path
    records: dict[str, Mapping[str, Any]] = {}
    with path.resolve(strict=True).open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            sample_id = str(record.get("sample_id", ""))
            if not sample_id or sample_id in records:
                raise SelectionError(f"invalid/duplicate sample id in {path}")
            records[sample_id] = record
    return records


def _update(checkpoint: str) -> int:
    match = re.search(r"(?:_|-)0*(25|50|100)(?:\D|$)", Path(checkpoint).stem)
    if match is None:
        raise SelectionError(f"cannot infer U25/U50/U100 from {checkpoint}")
    return int(match.group(1))


def select(
    *, milestone_summary: Path, c0_summary: Path, c0_gate_receipt: Path,
    baseline_summary: Path,
) -> dict[str, Any]:
    milestone_payload = _load(milestone_summary)
    c0_payload = _load(c0_summary)
    gate_payload = _load(c0_gate_receipt)
    baseline_payload = _load(baseline_summary)
    milestone_groups = _by_checkpoint(milestone_payload)
    c0_groups = _by_checkpoint(c0_payload)
    if len(c0_groups) != 1:
        raise SelectionError("C0 parity summary must contain one checkpoint")
    c0_rows = next(iter(c0_groups.values()))
    baseline_rows = {
        str(row["dataset"]): row for row in _rows(baseline_payload)
        if str(row.get("dataset")) in SPLITS
    }
    if set(baseline_rows) != set(SPLITS):
        raise SelectionError("baseline summary lacks the three val splits")
    selected_gate = gate_payload.get("selected")
    if not isinstance(selected_gate, Mapping) or float(selected_gate.get("gap", -1)) != 5.0:
        raise SelectionError("C0 gate receipt is not the selected Gap5 contract")
    raw_r100_micro = float(selected_gate["raw_r100_aggregate_acc50"])
    c0_micro = _micro(c0_rows)
    if abs(c0_micro - float(selected_gate["aggregate_acc50"])) > 1e-12:
        raise SelectionError("C0 fixed-gap rerun drifted from the gate receipt")

    c0_records = {split: _records(row) for split, row in c0_rows.items()}
    candidates = []
    for checkpoint, rows in milestone_groups.items():
        update = _update(checkpoint)
        split_results = {}
        correct_to_wrong = 0
        wrong_to_correct = 0
        eligibility_equal = True
        for split in SPLITS:
            candidate_records = _records(rows[split])
            reference_records = c0_records[split]
            if set(candidate_records) != set(reference_records):
                raise SelectionError(f"record universe drifted for {checkpoint} {split}")
            local_c2w = local_w2c = 0
            for sample_id, candidate in candidate_records.items():
                reference = reference_records[sample_id]
                candidate_hash = candidate.get("stage_b_u2v2_eligible_mask_sha256")
                reference_hash = reference.get("stage_b_u2v2_eligible_mask_sha256")
                if not candidate_hash or candidate_hash != reference_hash:
                    eligibility_equal = False
                before = bool(reference.get("correct50"))
                after = bool(candidate.get("correct50"))
                local_c2w += int(before and not after)
                local_w2c += int((not before) and after)
            correct_to_wrong += local_c2w
            wrong_to_correct += local_w2c
            acc = float(rows[split]["acc50"])
            c0_acc = float(c0_rows[split]["acc50"])
            b58_acc = float(baseline_rows[split]["acc50"])
            split_results[split] = {
                "acc50": acc,
                "b58_c100_ref_safe_acc50": b58_acc,
                "c0_acc50": c0_acc,
                "strict_b58_win": acc > b58_acc,
                "c0_non_regression": acc >= c0_acc,
                "c0_strict_gain": acc > c0_acc,
                "c0_correct_to_wrong": local_c2w,
                "c0_wrong_to_correct": local_w2c,
            }
        micro = _micro(rows)
        admitted = (
            eligibility_equal
            and all(item["strict_b58_win"] for item in split_results.values())
            and micro > raw_r100_micro
            and micro > c0_micro
            and all(item["c0_non_regression"] for item in split_results.values())
            and any(item["c0_strict_gain"] for item in split_results.values())
        )
        candidates.append(
            {
                "checkpoint": _file(Path(checkpoint)),
                "update": update,
                "aggregate_acc50": micro,
                "raw_r100_aggregate_acc50": raw_r100_micro,
                "c0_aggregate_acc50": c0_micro,
                "aggregate_strict_raw_win": micro > raw_r100_micro,
                "aggregate_strict_c0_win": micro > c0_micro,
                "patch_eligibility_bitwise_equal_to_c0": eligibility_equal,
                "c0_correct_to_wrong": correct_to_wrong,
                "c0_wrong_to_correct": wrong_to_correct,
                "net_gain": wrong_to_correct - correct_to_wrong,
                "splits": split_results,
                "admitted": admitted,
            }
        )
    candidates.sort(key=lambda item: item["update"])
    admitted = [item for item in candidates if item["admitted"]]
    selected = min(
        admitted,
        key=lambda item: (
            item["c0_correct_to_wrong"], -item["net_gain"], item["update"]
        ),
    ) if admitted else None
    return {
        "schema": "pivot.stageb.u2v2_milestone_selection/v1",
        "status": "selected" if selected is not None else "no_candidate",
        "policy": {
            "splits": list(SPLITS),
            "strict_b58_c100_ref_safe_each_split": True,
            "strict_raw_r100_and_c0_aggregate": True,
            "c0_each_split_non_regression_and_one_strict_gain": True,
            "patch_eligibility_bitwise_equal_to_c0": True,
            "tie_break": ["min_c0_correct_to_wrong", "max_net_gain", "earlier_update"],
        },
        "inputs": {
            "milestone_summary": _file(milestone_summary),
            "c0_summary": _file(c0_summary),
            "c0_gate_receipt": _file(c0_gate_receipt),
            "baseline_summary": _file(baseline_summary),
        },
        "candidates": candidates,
        "selected": selected,
        "test_and_strict_authorized": selected is not None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--milestone-summary", required=True)
    parser.add_argument("--c0-summary", required=True)
    parser.add_argument("--c0-gate-receipt", required=True)
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise SelectionError(f"refusing to overwrite {output}")
    result = select(
        milestone_summary=Path(args.milestone_summary),
        c0_summary=Path(args.c0_summary),
        c0_gate_receipt=Path(args.c0_gate_receipt),
        baseline_summary=Path(args.baseline_summary),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, SelectionError) as error:
        raise SystemExit(f"[FAIL] {error}") from error
