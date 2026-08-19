#!/usr/bin/env python3
"""Bind gRefCOCO v2 results without rewriting the failed v1 paths."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.arrow_grefcoco_common import FINAL_SCHEMA, RESULTS_SCHEMA, SEEDS, file_record, load_json, write_json_atomic

ROOT = REPO_ROOT / "outputs/arrow_grefcoco_20260820"


def build(preregistration: Path, results: Path, table: Path, evaluations: Path, amendment: Path, output: Path) -> dict:
    result = load_json(results)
    if result.get("schema") != RESULTS_SCHEMA or result.get("status") != "complete":
        raise ValueError("gRefCOCO results are incomplete")
    amendment_payload = load_json(amendment)
    if amendment_payload.get("schema") != "arrow.grefcoco.preregistration_amendment/v1" or amendment_payload.get("change_scope") != "final_receipt_v2_path_resolution_only":
        raise ValueError("gRefCOCO final packaging amendment is invalid")
    if file_record(preregistration) != amendment_payload.get("base_preregistration"):
        raise ValueError("gRefCOCO amendment does not bind preregistration v2")
    runs = {}
    for seed in SEEDS:
        receipt = evaluations / f"seed{seed}" / "run_receipt.json"
        payload = load_json(receipt)
        if payload.get("status") != "complete" or payload.get("rows") != 29589 or payload.get("model_training") is not False or payload.get("optimizer_created") is not False:
            raise ValueError(f"seed{seed} run contract failed")
        runs[str(seed)] = file_record(receipt)
    payload = {
        "schema": FINAL_SCHEMA,
        "status": "sealed_v2",
        "preregistration": file_record(preregistration),
        "packaging_amendment": file_record(amendment),
        "results": file_record(results),
        "paper_table": file_record(table),
        "runs": runs,
        "decision": result["decision"],
        "claim_boundary": result["claim_boundary"],
        "failed_v1_preserved": True,
        "weights_committed": False,
        "records_committed": False,
    }
    write_json_atomic(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, default=ROOT / "preregistration_v2.json")
    parser.add_argument("--results", type=Path, default=ROOT / "results.json")
    parser.add_argument("--table", type=Path, default=ROOT / "paper_table.md")
    parser.add_argument("--evaluations", type=Path, default=ROOT / "evaluations_v2")
    parser.add_argument("--amendment", type=Path, default=ROOT / "preregistration_amendment_v2_final_packaging.json")
    parser.add_argument("--output", type=Path, default=ROOT / "final_receipt.json")
    args = parser.parse_args()
    result = build(args.preregistration.resolve(strict=True), args.results.resolve(strict=True), args.table.resolve(strict=True), args.evaluations.resolve(strict=True), args.amendment.resolve(strict=True), args.output.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
