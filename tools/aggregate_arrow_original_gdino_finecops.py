#!/usr/bin/env python3
"""Aggregate the preregistered original GroundingDINO-T FineCops replay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.aggregate_arrow_finecops import _surface_metrics
from tools.arrow_finecops_common import (
    file_record,
    load_json,
    load_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)
from tools.arrow_original_gdino_common import (
    PRIMARY_SCORE,
    RECORD_SCHEMA,
    RESULTS_SCHEMA,
    RUN_SCHEMA,
    SENSITIVITY_SCORE,
    load_preregistration,
    verify_file,
)


DEFAULT_PREREG = (
    REPO_ROOT / "outputs/arrow_original_gdino_ogc_finecops_20260819/preregistration.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "outputs/arrow_original_gdino_ogc_finecops_20260819/results.json"
)


def _load_complete_run(prereg: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = Path(str(prereg["execution"]["results_root"])).resolve()
    receipt_path = root / "run_receipt.json"
    receipt = load_json(receipt_path)
    if receipt.get("schema") != RUN_SCHEMA or receipt.get("status") != "complete":
        raise ValueError("original OGC FineCops run is not complete")
    records_path = verify_file(receipt["records"], label="original OGC records")
    rows = load_jsonl(records_path)
    if len(rows) != 27926 or receipt.get("count") != 27926:
        raise ValueError("original OGC FineCops record count drifted")
    if any(row.get("schema") != RECORD_SCHEMA for row in rows):
        raise ValueError("original OGC FineCops record schema drifted")
    sample_ids = [str(row.get("sample_id")) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("original OGC FineCops sample IDs are not unique")
    required_routes = {PRIMARY_SCORE, SENSITIVITY_SCORE}
    if any(set(row.get("routes", {})) != required_routes for row in rows):
        raise ValueError("original OGC FineCops score routes drifted")
    return receipt, rows


def _official_predictions(
    output_dir: Path, rows: list[dict[str, Any]]
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Any] = {}
    for route in (PRIMARY_SCORE, SENSITIVITY_SCORE):
        output = output_dir / f"original_ogc_{route}.jsonl"
        predictions = [
            {
                "image_id": int(row["finecops_annotation_id"]),
                "bboxes": [row["routes"][route]["top1_box_xyxy"]],
                "scores": [row["routes"][route]["official_probability"]],
            }
            for row in rows
        ]
        write_jsonl_atomic(output, predictions)
        artifacts[route] = file_record(output, rows=len(predictions))
    return artifacts


def aggregate(preregistration_path: Path, output_path: Path) -> dict[str, Any]:
    prereg = load_preregistration(preregistration_path)
    receipt, rows = _load_complete_run(prereg)
    metrics = {
        route: _surface_metrics(
            rows,
            output_route=route,
            support_matched=False,
            fixed_threshold=None,
        )
        for route in (PRIMARY_SCORE, SENSITIVITY_SCORE)
    }
    payload = {
        "schema": RESULTS_SCHEMA,
        "status": "aggregated_pending_official_exact",
        "evidence_status": prereg["evidence_status"],
        "preregistration": file_record(preregistration_path),
        "run_receipt": file_record(
            Path(str(prereg["execution"]["results_root"])) / "run_receipt.json"
        ),
        "checkpoint": dict(receipt["checkpoint"]),
        "primary_score": PRIMARY_SCORE,
        "sensitivity_score": SENSITIVITY_SCORE,
        "metrics": metrics,
        "official_prediction_artifacts": _official_predictions(
            output_path.parent / "official_predictions", rows
        ),
        "finecops_threshold_fitted": False,
        "training_performed": False,
        "official_exact_status": "pending_external_pinned_evaluator",
    }
    write_json_atomic(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    payload = aggregate(args.preregistration.resolve(strict=True), args.output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

