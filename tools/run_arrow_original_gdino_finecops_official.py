#!/usr/bin/env python3
"""Replay the pinned external FineCops evaluator for original OGC outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.arrow_finecops_common import (
    OFFICIAL_REPO_COMMIT,
    file_record,
    load_json,
    load_jsonl,
    write_json_atomic,
)
from tools.arrow_original_gdino_common import (
    PRIMARY_SCORE,
    RESULTS_SCHEMA,
    SENSITIVITY_SCORE,
    load_preregistration,
)
from tools.run_arrow_finecops_official import _load_official, _read_table


DEFAULT_RESULTS = (
    REPO_ROOT / "outputs/arrow_original_gdino_ogc_finecops_20260819/results.json"
)
DEFAULT_OFFICIAL_REPO = Path(
    "/media/haoyi/T9/data/FineCops-Ref/v1/official_repo"
)
DEFAULT_ANNOTATION = Path(
    "/media/haoyi/T9/data/FineCops-Ref/v1/raw/benchmark/"
    "test_expression_all_coco_format.json"
)


def run(results_path: Path, official_repo: Path, annotation: Path) -> dict[str, object]:
    results = load_json(results_path)
    if results.get("schema") != RESULTS_SCHEMA:
        raise ValueError("original OGC FineCops results schema drifted")
    prereg = load_preregistration(Path(results["preregistration"]["path"]))
    module, official_source = _load_official(official_repo.resolve(strict=True))
    official_root = results_path.parent / "official_exact"
    receipts: dict[str, object] = {}
    for route in (PRIMARY_SCORE, SENSITIVITY_SCORE):
        artifact = results["official_prediction_artifacts"][route]
        prediction_path = Path(artifact["path"]).resolve(strict=True)
        if file_record(prediction_path)["sha256"] != artifact["sha256"]:
            raise ValueError(f"original OGC official predictions drifted: {route}")
        rows = load_jsonl(prediction_path)
        predictions = [
            {
                "img_id": int(row["image_id"]),
                "bboxes": np.asarray(row["bboxes"], dtype=np.float32),
                "scores": np.asarray(row["scores"], dtype=np.float32),
            }
            for row in rows
        ]
        output_dir = official_root / route
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError(f"original OGC official output exists: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        evaluator = module.RefRecallAUROC(
            ann_file=str(annotation.resolve(strict=True)),
            topk=(1, 5, 10),
            iou_thrs=0.5,
            score_thrs=0.0,
            save_dir=str(output_dir),
        )
        returned = evaluator.compute_metrics(predictions)
        precision = output_dir / "precision.csv"
        recall = output_dir / "recall_['negative_type', 'negative_level', 'negative_cate'].csv"
        auroc = output_dir / "auroc_['negative_type', 'negative_level', 'negative_cate'].csv"
        receipt = {
            "schema": "arrow.original_gdino_ogc.finecops_official_receipt/v1",
            "route": route,
            "official_repo_commit": OFFICIAL_REPO_COMMIT,
            "official_source": file_record(official_source),
            "annotation": file_record(annotation),
            "predictions": dict(artifact),
            "precision_csv": file_record(precision),
            "recall_csv": file_record(recall),
            "auroc_csv": file_record(auroc),
            "precision": _read_table(precision),
            "recall": _read_table(recall),
            "auroc": _read_table(auroc),
            "returned_result_keys": sorted(str(key) for key in returned),
        }
        receipt_path = output_dir / "receipt.json"
        write_json_atomic(receipt_path, receipt)
        receipts[route] = file_record(receipt_path)
    results["official_exact"] = {
        "status": "complete_external_pinned_evaluator",
        "repo_commit": OFFICIAL_REPO_COMMIT,
        "source": file_record(official_source),
        "annotation": file_record(annotation),
        "receipts": receipts,
    }
    results["official_exact_status"] = "complete_external_pinned_evaluator"
    results["status"] = "complete"
    write_json_atomic(results_path, results)
    return results["official_exact"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--official-repo", type=Path, default=DEFAULT_OFFICIAL_REPO)
    parser.add_argument("--annotation", type=Path, default=DEFAULT_ANNOTATION)
    args = parser.parse_args()
    payload = run(
        args.results.resolve(strict=True),
        args.official_repo,
        args.annotation.resolve(strict=True),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
