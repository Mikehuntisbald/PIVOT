#!/usr/bin/env python3
"""Derive the validation-only Admission-gap table from sealed records.

This script performs no model forward pass.  It reads the gap-sweep summary,
verifies every referenced per-example record file, and materializes exact
pooled Val3 accuracy, eligible recall, and candidate-count statistics together
with byte-level provenance for every input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/u2v5_cvpr_ablation_20260817/zero_training_h/gap_sweep/summary.json"
OUTPUT = ROOT / "paper/data/gap_sensitivity.json"
EXPECTED_SOURCE_SHA256 = "f0c51ada90bc5786d17d4704db5bc20cb7d08b2a866b7bfc3761438d9ec5d8f3"
EXPECTED_SPLITS = {"refcoco_val", "refcocop_val", "refcocog_val"}
EXPECTED_GAPS = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 1_000_000_000.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


def derive() -> dict[str, Any]:
    actual_source_sha = sha256(SOURCE)
    if actual_source_sha != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(
            f"sealed gap summary drift: expected {EXPECTED_SOURCE_SHA256}, got {actual_source_sha}"
        )
    payload = json.loads(SOURCE.read_text(encoding="utf-8"))
    runs = payload.get("refcoco")
    if not isinstance(runs, list) or len(runs) != len(EXPECTED_GAPS) * len(EXPECTED_SPLITS):
        raise RuntimeError("gap summary does not contain the expected 8 gaps x 3 Val splits")

    grouped: dict[float, list[dict[str, Any]]] = {gap: [] for gap in EXPECTED_GAPS}
    for run in runs:
        gap = float(run["category_gate_max_gap"])
        split = str(run["dataset"])
        if gap not in grouped or split not in EXPECTED_SPLITS:
            raise RuntimeError(f"unexpected gap/split in sealed summary: {gap}/{split}")
        grouped[gap].append(run)

    rows: list[dict[str, Any]] = []
    for gap in EXPECTED_GAPS:
        split_runs = grouped[gap]
        if {str(run["dataset"]) for run in split_runs} != EXPECTED_SPLITS:
            raise RuntimeError(f"incomplete split set for gap {gap}")
        total = correct = eligible = candidate_sum = 0
        inputs: list[dict[str, Any]] = []
        by_split: dict[str, dict[str, float | int]] = {}
        checkpoint_shas: set[str] = set()
        for run in sorted(split_runs, key=lambda item: str(item["dataset"])):
            split = str(run["dataset"])
            record_path = ROOT / str(run["records_jsonl"])
            if not record_path.is_file():
                raise FileNotFoundError(record_path)
            n = split_correct = split_eligible = split_candidates = 0
            with record_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    if record.get("split") != split or not record.get("valid", False):
                        raise RuntimeError(f"invalid or misaligned gap record in {record_path}")
                    if not math.isclose(float(record["category_gate_max_gap"]), gap):
                        raise RuntimeError(f"gap mismatch in {record_path}")
                    n += 1
                    split_correct += int(bool(record["correct50"]))
                    split_eligible += int(float(record["eligible_query_best_iou"]) >= 0.5)
                    split_candidates += int(record["category_gate_eligible_queries"])
            if n != int(run["num_expressions"]):
                raise RuntimeError(f"record count mismatch for {record_path}: {n}")
            if not math.isclose(split_correct / n, float(run["acc50"]), abs_tol=1e-12):
                raise RuntimeError(f"accuracy mismatch for {record_path}")
            if not math.isclose(
                split_eligible / n, float(run["recall50@eligible_queries"]), abs_tol=1e-12
            ):
                raise RuntimeError(f"eligible-recall mismatch for {record_path}")
            by_split[split] = {
                "n": n,
                "acc50": split_correct / n,
                "eligible_recall50": split_eligible / n,
                "mean_eligible_queries": split_candidates / n,
            }
            total += n
            correct += split_correct
            eligible += split_eligible
            candidate_sum += split_candidates
            checkpoint_shas.add(str(run["checkpoint_sha256"]))
            inputs.append(
                {
                    "path": relative(record_path),
                    "sha256": sha256(record_path),
                    "size_bytes": record_path.stat().st_size,
                    "split": split,
                }
            )
        if len(checkpoint_shas) != 1:
            raise RuntimeError(f"multiple checkpoints in gap {gap}: {sorted(checkpoint_shas)}")
        rows.append(
            {
                "gap": gap,
                "label": "infinity" if gap >= 1_000_000_000 else f"{gap:g}",
                "n": total,
                "acc50": correct / total,
                "eligible_recall50": eligible / total,
                "mean_eligible_queries": candidate_sum / total,
                "by_split": by_split,
                "checkpoint_sha256": next(iter(checkpoint_shas)),
                "record_inputs": inputs,
            }
        )
    return {
        "schema": "arrow.paper.gap_sensitivity/v1",
        "status": "exploratory_validation_only",
        "selected_gap": 3.0,
        "selection_surface": "Val3 only",
        "source_summary": {
            "path": relative(SOURCE),
            "sha256": actual_source_sha,
            "size_bytes": SOURCE.stat().st_size,
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    rendered = json.dumps(derive(), indent=2, sort_keys=True) + "\n"
    if args.check:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != rendered:
            raise SystemExit("committed gap sensitivity receipt is stale")
        print(f"verified {OUTPUT.relative_to(ROOT)}")
        return
    OUTPUT.write_text(rendered, encoding="utf-8")
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
