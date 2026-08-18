#!/usr/bin/env python3
"""Build the terminal fail-closed ARROW FineCops receipt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.arrow_finecops_common import FINAL_SCHEMA, file_record, load_json, write_json_atomic


def build(
    preregistration: Path,
    results: Path,
    table: Path,
    gqa_zip_verification: Path,
    output: Path,
) -> dict:
    prereg = load_json(preregistration)
    result = load_json(results)
    if prereg.get("schema") != "arrow.finecops.preregistration/v1":
        raise ValueError("FineCops preregistration schema drifted")
    if result.get("schema") != "arrow.finecops.results/v1":
        raise ValueError("FineCops results schema drifted")
    if result.get("finecops_threshold_fitted") is not False:
        raise ValueError("FineCops results fitted a test threshold")
    if result.get("parity", {}).get("status") != "bitwise_equal":
        raise ValueError("FineCops frozen-route parity did not pass")
    if result.get("official_exact_status") != "complete_external_pinned_evaluator":
        raise ValueError("pinned official evaluator has not completed")
    gqa = load_json(gqa_zip_verification)
    if (
        gqa.get("schema") != "arrow.finecops.gqa_zip_verification/v1"
        or gqa.get("required_image_crc_parity") is not True
        or int(gqa.get("required_image_count", -1)) != 4313
    ):
        raise ValueError("official GQA zip verification is incomplete")
    payload = {
        "schema": FINAL_SCHEMA,
        "status": "complete",
        "claim": "finecops_specific_external_zero_shot_not_image_disjoint",
        "preregistration": file_record(preregistration),
        "results": file_record(results),
        "paper_table": file_record(table),
        "official_gqa_zip_verification": file_record(gqa_zip_verification),
        "dataset": dict(prereg["dataset"]),
        "checkpoints": dict(prereg["checkpoints"]),
        "official_exact": result["official_exact"],
        "parity": result["parity"],
        "finecops_train_or_val_used": False,
        "finecops_threshold_fitted": False,
    }
    write_json_atomic(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=REPO_ROOT / "outputs/arrow_finecops_20260819/preregistration.json",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=REPO_ROOT / "outputs/arrow_finecops_20260819/results.json",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=REPO_ROOT / "outputs/arrow_finecops_20260819/paper_table.md",
    )
    parser.add_argument(
        "--gqa-zip-verification",
        type=Path,
        default=Path(
            "/media/haoyi/T9/data/FineCops-Ref/v1/manifests/"
            "official_gqa_zip_verification.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs/arrow_finecops_20260819/final_receipt.json",
    )
    args = parser.parse_args()
    payload = build(
        args.preregistration.resolve(strict=True),
        args.results.resolve(strict=True),
        args.table.resolve(strict=True),
        args.gqa_zip_verification.resolve(strict=True),
        args.output,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
