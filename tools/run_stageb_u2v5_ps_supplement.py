#!/usr/bin/env python3
"""Run and aggregate the val-only U2-v5 P/S counterfactual supplement."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/haoyi/miniconda/envs/cvpr/bin/python")
EVALUATOR = ROOT / "tools/eval_text_groundingdino_refcoco_tn.py"
CONFIG = ROOT / "config/ablations/cfg_stageb_u2v5_clean_confidence_d3_u100.py"
CHECKPOINT = ROOT / "outputs/u2v5_leakage_clean_anchor_20260817/formal/confidence_seed42_u50/checkpoint_iter.pth"
DEFAULT_OUTPUT = ROOT / "outputs/u2v5_cvpr_ablation_20260817/zero_training_ps"
SPLITS = ("refcoco_val", "refcocop_val", "refcocog_val")
SCHEMA = "pivot.stageb.u2v5_ps_supplement/v1"
VARIANTS = {
    "P0_S0": ("full", "full", "bound", 42),
    "P1": ("canonical", "full", "bound", 42),
    "P2": ("object", "full", "bound", 42),
    "P3": ("full", "canonical", "bound", 42),
    "S1": ("full", "full", "bound", 43),
    "S2": ("full", "full", "same_class_shuffle", 42),
    "S3": ("full", "full", "wrong_category", 42),
    "S4": ("full", "full", "zero", 42),
}


class SupplementError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": _sha(path)}


def _command(variant: str, output_root: Path) -> list[str]:
    query, rank, support, seed = VARIANTS[variant]
    return [
        str(PYTHON), str(EVALUATOR), "--config", str(CONFIG),
        "--ckpts", str(CHECKPOINT), "--output_dir", str(output_root / variant),
        "--data_root", "/media/haoyi/T9/data", "--device", "cuda:0",
        "--batch_size", "16", "--num_workers", "4", "--seed", str(seed),
        "--amp", "--ref_splits", *SPLITS, "--skip_tn", "--topk", "1",
        "--holdout_level", "none", "--u2v5_ref_query_caption_mode", query,
        "--u2v5_ref_rank_caption_mode", rank, "--u2v5_ref_support_mode", support,
        "--u2v5_emit_eligible_indices",
    ]


def _records(summary_path: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    result, inputs = {}, []
    for row in summary.get("refcoco", []):
        if row.get("dataset") not in SPLITS:
            continue
        path = Path(row["records_jsonl"])
        inputs.append(_record(path))
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                sample_id = str(record["sample_id"])
                if sample_id in result or not record.get("valid", False):
                    raise SupplementError(f"invalid/duplicate record {sample_id}")
                result[sample_id] = record
    if len(result) != 26488:
        raise SupplementError(f"expected 26,488 val3 records, got {len(result)}")
    return result, inputs


def _box_l1(a: list[float], b: list[float]) -> float:
    return sum(abs(float(x) - float(y)) for x, y in zip(a, b)) / 4.0


def _metrics(
    records: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]],
    *,
    variant: str,
) -> dict[str, Any]:
    if set(records) != set(baseline):
        raise SupplementError(f"{variant} sample IDs do not align with baseline")
    correct = oracle = eligible_recall = churn = 0
    eligible_sizes = []
    hamming = []
    box_l1 = []
    valid_intervention = 0
    support_changed = 0
    for sample_id in sorted(records):
        row, base = records[sample_id], baseline[sample_id]
        correct += int(bool(row["correct50"]))
        oracle += int(float(row["all_query_best_iou"]) >= 0.5)
        eligible_recall += int(float(row["eligible_query_best_iou"]) >= 0.5)
        eligible = set(map(int, row["stage_b_u2v5_eligible_indices"]))
        base_eligible = set(map(int, base["stage_b_u2v5_eligible_indices"]))
        eligible_sizes.append(len(eligible))
        hamming.append(len(eligible.symmetric_difference(base_eligible)))
        churn += int(int(row["top1_query_index"]) != int(base["top1_query_index"]))
        box_l1.append(_box_l1(row["top1_box_cxcywh"], base["top1_box_cxcywh"]))
        valid_intervention += int(bool(row["stage_b_u2v5_support_intervention_valid"]))
        support_changed += int(
            row["stage_b_u2v5_support_tensor_sha256"]
            != base["stage_b_u2v5_support_tensor_sha256"]
        )
    total = len(records)
    return {
        "expressions": total,
        "acc50": correct / total,
        "all_query_oracle_recall50": oracle / total,
        "eligible_recall50": eligible_recall / total,
        "eligible_query_count_mean": statistics.mean(eligible_sizes),
        "eligible_mask_hamming_mean": statistics.mean(hamming),
        "top1_query_churn": churn / total,
        "top1_box_l1_mean": statistics.mean(box_l1),
        "support_intervention_valid_rate": valid_intervention / total,
        "support_tensor_changed_rate": support_changed / total,
    }


def _aggregate(output_root: Path, output: Path) -> None:
    loaded, inputs = {}, {}
    for variant in VARIANTS:
        summary = output_root / variant / "summary.json"
        if not summary.is_file():
            raise SupplementError(f"missing completed variant {variant}: {summary}")
        loaded[variant], inputs[variant] = _records(summary)
    baseline = loaded["P0_S0"]
    payload = {
        "schema": SCHEMA,
        "checkpoint": _record(CHECKPOINT),
        "config": _record(CONFIG),
        "variants": {
            variant: {
                "contract": {
                    "query_caption": VARIANTS[variant][0],
                    "rank_caption": VARIANTS[variant][1],
                    "support": VARIANTS[variant][2],
                    "loader_seed": VARIANTS[variant][3],
                },
                "metrics": _metrics(records, baseline, variant=variant),
                "record_inputs": inputs[variant],
            }
            for variant, records in loaded.items()
        },
        "surface": "val3_exploratory_only",
        "confirmatory_surfaces_used": False,
    }
    if output.exists():
        raise SupplementError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"status": "complete", "receipt": _record(output)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "dry-run", "run", "status", "aggregate"))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--variants", nargs="*", choices=tuple(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT / "summary.json"))
    args = parser.parse_args()
    output_root = Path(args.output_root)
    if args.action == "list":
        print(json.dumps(VARIANTS, indent=2))
        return
    if args.action == "aggregate":
        _aggregate(output_root, Path(args.output))
        return
    if args.action == "status":
        print(json.dumps({variant: (output_root / variant / "summary.json").is_file() for variant in VARIANTS}, indent=2))
        return
    for variant in args.variants:
        command = _command(variant, output_root)
        if args.action == "dry-run":
            print(json.dumps({"variant": variant, "command": command}))
            continue
        if (output_root / variant).exists():
            raise SupplementError(f"variant output already exists: {output_root / variant}")
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, SupplementError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"[FAIL] {error}") from error
