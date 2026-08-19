#!/usr/bin/env python3
"""Bind the completed ARROW gRefCOCO external transfer evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.arrow_grefcoco_common import FINAL_SCHEMA, RESULTS_SCHEMA, SEEDS, file_record, load_json, write_json_atomic

DEFAULT_ROOT = REPO_ROOT / "outputs/arrow_grefcoco_20260820"


def build(root: Path) -> dict:
    prereg = root / "preregistration.json"
    results = root / "results.json"
    table = root / "paper_table.md"
    result = load_json(results)
    if result.get("schema") != RESULTS_SCHEMA or result.get("status") != "complete":
        raise ValueError("gRefCOCO results are incomplete")
    runs = {}
    for seed in SEEDS:
        receipt = root / "evaluations" / f"seed{seed}" / "run_receipt.json"
        payload = load_json(receipt)
        if payload.get("status") != "complete" or payload.get("model_training") is not False or payload.get("optimizer_created") is not False:
            raise ValueError(f"seed{seed} run contract failed")
        runs[str(seed)] = file_record(receipt)
    payload = {"schema": FINAL_SCHEMA, "status": "sealed", "preregistration": file_record(prereg), "results": file_record(results), "paper_table": file_record(table), "runs": runs, "decision": result["decision"], "claim_boundary": result["claim_boundary"], "weights_committed": False, "records_committed": False}
    write_json_atomic(root / "final_receipt.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    print(json.dumps(build(args.root.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
