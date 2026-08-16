#!/usr/bin/env python3
"""Select the U2-v2 training gate using val-only accuracy and eligibility recall."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


VAL_SPLITS = ("refcoco_val", "refcocop_val", "refcocog_val")
SCHEMA = "pivot.stageb.u2v2_gate_selection/v1"


class U2V2GateSelectionError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = path.resolve(strict=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("refcoco"), list):
        raise U2V2GateSelectionError(f"invalid summary: {path}")
    return payload, {"path": str(path), "sha256": _sha(path), "size_bytes": path.stat().st_size}


def _records(summary_path: Path, row: Mapping[str, Any]) -> list[bool]:
    path = Path(str(row.get("records_jsonl", "")))
    if not path.is_absolute():
        candidates = ((summary_path.parent / path), (Path.cwd() / path))
        path = next((item for item in candidates if item.exists()), candidates[0])
    values = []
    with path.resolve(strict=True).open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("valid") is not True or not isinstance(record.get("correct50"), bool):
                raise U2V2GateSelectionError(f"invalid Ref record in {path}")
            values.append(bool(record["correct50"]))
    if len(values) != int(row["manifest_n"]):
        raise U2V2GateSelectionError(f"record count drifted: {path}")
    return values


def select(sweep_summary: Path, baseline_summary: Path) -> dict[str, Any]:
    sweep, sweep_artifact = _summary(sweep_summary)
    baseline, baseline_artifact = _summary(baseline_summary)
    baseline_by_split = {
        str(row["dataset"]): row for row in baseline["refcoco"]
        if str(row.get("dataset")) in VAL_SPLITS
    }
    if set(baseline_by_split) != set(VAL_SPLITS):
        raise U2V2GateSelectionError("baseline does not contain the exact val splits")
    by_gap: dict[float, dict[str, Mapping[str, Any]]] = {}
    for row in sweep["refcoco"]:
        split = str(row.get("dataset"))
        if split not in VAL_SPLITS:
            raise U2V2GateSelectionError("gate selection is val-only")
        gap = float(row.get("category_gate_max_gap"))
        if "recall50@eligible_queries" not in row:
            raise U2V2GateSelectionError("sweep lacks eligible-query recall")
        by_gap.setdefault(gap, {})[split] = row
    if not by_gap or any(set(rows) != set(VAL_SPLITS) for rows in by_gap.values()):
        raise U2V2GateSelectionError("incomplete split/gap grid")
    reference_gap = max(by_gap)
    reference = by_gap[reference_gap]
    reference_correct = {
        split: _records(sweep_summary, reference[split]) for split in VAL_SPLITS
    }

    candidates = []
    for gap, rows in sorted(by_gap.items()):
        if gap == reference_gap:
            continue
        split_metrics = {}
        pooled_n = pooled_recall = pooled_correct = 0.0
        pooled_raw_correct = 0.0
        regressions = gains = 0
        recall_pass = True
        strict_b58 = True
        for split in VAL_SPLITS:
            row = rows[split]
            raw = reference[split]
            n = int(row["manifest_n"])
            recall = float(row["recall50@eligible_queries"])
            raw_recall = float(raw["recall50@eligible_queries"])
            delta = recall - raw_recall
            recall_pass &= delta >= -0.005
            strict_b58 &= float(row["acc50"]) > float(baseline_by_split[split]["acc50"])
            correctness = _records(sweep_summary, row)
            raw_correctness = reference_correct[split]
            split_regression = sum(a and not b for a, b in zip(raw_correctness, correctness))
            split_gain = sum((not a) and b for a, b in zip(raw_correctness, correctness))
            regressions += split_regression
            gains += split_gain
            pooled_n += n
            pooled_recall += recall * n
            pooled_correct += float(row["acc50"]) * n
            pooled_raw_correct += float(raw["acc50"]) * n
            split_metrics[split] = {
                "acc50": float(row["acc50"]),
                "b58_acc50": float(baseline_by_split[split]["acc50"]),
                "raw_r100_acc50": float(raw["acc50"]),
                "eligible_recall50": recall,
                "raw_eligible_recall50": raw_recall,
                "eligible_recall_delta": delta,
                "raw_correct_to_wrong": split_regression,
                "raw_wrong_to_correct": split_gain,
            }
        pooled_recall /= pooled_n
        raw_pooled_recall = sum(
            float(reference[s]["recall50@eligible_queries"]) * int(reference[s]["manifest_n"])
            for s in VAL_SPLITS
        ) / pooled_n
        pooled_recall_delta = pooled_recall - raw_pooled_recall
        recall_pass &= pooled_recall_delta >= -0.002
        aggregate = pooled_correct / pooled_n
        raw_aggregate = pooled_raw_correct / pooled_n
        preferred = recall_pass and strict_b58 and aggregate > raw_aggregate
        candidates.append({
            "gap": gap, "recall_admitted": recall_pass,
            "strict_b58_all_splits": strict_b58,
            "aggregate_acc50": aggregate, "raw_r100_aggregate_acc50": raw_aggregate,
            "aggregate_strict_raw_win": aggregate > raw_aggregate,
            "pooled_eligible_recall50": pooled_recall,
            "raw_pooled_eligible_recall50": raw_pooled_recall,
            "pooled_eligible_recall_delta": pooled_recall_delta,
            "raw_correct_to_wrong": regressions, "raw_wrong_to_correct": gains,
            "net_gain": gains - regressions, "preferred": preferred,
            "splits": split_metrics,
        })
    admitted = [row for row in candidates if row["recall_admitted"]]
    if not admitted:
        raise U2V2GateSelectionError("no nontrivial gap passes eligible recall admission")
    preferred = [row for row in admitted if row["preferred"]]
    pool = preferred or admitted
    selected = min(
        pool,
        key=lambda row: (
            row["raw_correct_to_wrong"], -row["aggregate_acc50"], -row["gap"]
        ),
    )
    return {
        "schema": SCHEMA, "status": "selected", "val_only": True,
        "sweep_summary": sweep_artifact, "baseline_summary": baseline_artifact,
        "reference_gap": reference_gap,
        "policy": {
            "pooled_eligible_recall_max_drop": 0.002,
            "per_split_eligible_recall_max_drop": 0.005,
            "preferred": "strict_b58_each_split_and_aggregate_strict_raw_r100",
            "tie_break": "min_raw_regression_then_max_aggregate_then_max_gap",
        },
        "selected_gap": selected["gap"], "selected": selected,
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-summary", required=True)
    parser.add_argument("--baseline-summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = select(Path(args.sweep_summary), Path(args.baseline_summary))
    output = Path(args.output).resolve()
    if output.exists():
        raise U2V2GateSelectionError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, U2V2GateSelectionError) as error:
        raise SystemExit(f"[FAIL] {error}") from error
