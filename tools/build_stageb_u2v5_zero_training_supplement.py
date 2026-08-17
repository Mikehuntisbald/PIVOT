#!/usr/bin/env python3
"""Build zero-training M/G/H evidence from sealed U2-v5 records."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
ANCHOR = ROOT / "outputs/u2v5_leakage_clean_anchor_20260817"
BLOCK = ROOT / "outputs/u2v5_cvpr_ablation_20260817"
SEEDS = (17, 42, 73)
VAL3 = ("refcoco_val", "refcocop_val", "refcocog_val")
SCHEMA = "pivot.stageb.u2v5_zero_training_supplement/v1"


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


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SupplementError(f"expected JSON object: {path}")
    return value


def _seed(row: dict[str, Any]) -> int:
    match = re.search(r"seed(17|42|73)", str(row.get("run_id", "")))
    if not match:
        raise SupplementError("cannot recover formal seed")
    return int(match.group(1))


def _mean_sd(values: Iterable[float]) -> dict[str, float]:
    values = list(map(float, values))
    if len(values) != 3:
        raise SupplementError("formal route requires three seeds")
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values)}


def _route_table(summary_path: Path) -> dict[str, Any]:
    rows = _read(summary_path).get("refcoco")
    if not isinstance(rows, list):
        raise SupplementError("anchor val3 summary lacks Ref rows")
    grouped: dict[str, dict[int, list[tuple[float, int]]]] = {
        route: {seed: [] for seed in SEEDS}
        for route in ("b58_base", "raw_r100", "admission_r100", "deployed")
    }
    seconds = []
    for row in rows:
        if row.get("dataset") not in VAL3:
            continue
        seed = _seed(row)
        routes = row.get("stage_b_u2v5_causal_ref_routes")
        if not isinstance(routes, dict):
            raise SupplementError("anchor val3 summary lacks causal routes")
        count = int(row["num_expressions"])
        for route in grouped:
            grouped[route][seed].append((float(routes[route]["acc50"]), count))
        seconds.append(float(row["seconds"]))
    table = {}
    for route, by_seed in grouped.items():
        values = {}
        for seed in SEEDS:
            if len(by_seed[seed]) != 3:
                raise SupplementError(f"route {route} seed{seed} lacks val3")
            total = sum(count for _, count in by_seed[seed])
            values[seed] = sum(value * count for value, count in by_seed[seed]) / total
        table[route] = {
            "by_seed": {str(seed): values[seed] for seed in SEEDS},
            **_mean_sd(values[seed] for seed in SEEDS),
        }
    table["single_forward_seconds"] = {
        "mean_per_split_seed": statistics.mean(seconds),
        "total_recorded": sum(seconds),
    }
    return table


def _deployed_micro(summary_path: Path, *, allowed_seeds: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    rows = _read(summary_path).get("refcoco")
    grouped: dict[int, list[tuple[float, int]]] = {seed: [] for seed in allowed_seeds}
    for row in rows:
        if row.get("dataset") in VAL3:
            seed = _seed(row)
            if seed in grouped:
                grouped[seed].append((float(row["acc50"]), int(row["num_expressions"])))
    values = {}
    for seed in allowed_seeds:
        if len(grouped[seed]) != 3:
            raise SupplementError(f"deployed seed{seed} lacks val3")
        total = sum(count for _, count in grouped[seed])
        values[seed] = sum(value * count for value, count in grouped[seed]) / total
    result = {"by_seed": {str(seed): values[seed] for seed in allowed_seeds}}
    if len(allowed_seeds) == 3:
        result.update(_mean_sd(values[seed] for seed in allowed_seeds))
    return result


def attribute_ref_error(record: dict[str, Any]) -> str:
    all_best = float(record["all_query_best_iou"])
    eligible_best = float(record["eligible_query_best_iou"])
    top1 = float(record["top1_iou"])
    if all_best < 0.5:
        return "geometry"
    if eligible_best < 0.5:
        return "admission"
    if top1 < 0.5:
        return "rank"
    return "correct"


def _error_attribution(summary_path: Path) -> dict[str, Any]:
    rows = _read(summary_path)["refcoco"]
    by_seed: dict[int, Counter[str]] = {seed: Counter() for seed in SEEDS}
    eligible_counts: dict[int, list[int]] = {seed: [] for seed in SEEDS}
    inputs = []
    for row in rows:
        if row.get("dataset") not in VAL3:
            continue
        seed = _seed(row)
        path = Path(row["records_jsonl"])
        inputs.append(_record(path))
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if not record.get("valid", False):
                    raise SupplementError(f"invalid sealed Ref record in {path}")
                by_seed[seed][attribute_ref_error(record)] += 1
                eligible_counts[seed].append(
                    int(record["stage_b_u2v2_eligible_queries"])
                )
    result = {}
    for seed in SEEDS:
        total = sum(by_seed[seed].values())
        result[str(seed)] = {
            "count": dict(by_seed[seed]),
            "rate": {key: value / total for key, value in by_seed[seed].items()},
            "expressions": total,
            "eligible_query_count": {
                "mean": statistics.mean(eligible_counts[seed]),
                "median": statistics.median(eligible_counts[seed]),
                "min": min(eligible_counts[seed]),
                "max": max(eligible_counts[seed]),
            },
        }
    return {"by_seed": result, "record_inputs": inputs}


def _confidence_milestones(path: Path) -> dict[str, Any]:
    rows = _read(path)["tn"]
    result: dict[str, dict[str, float]] = {}
    for row in rows:
        match = re.search(r"seed(17|42|73)_u(25|50|100)", row["run_id"])
        if not match:
            continue
        result.setdefault(match.group(2), {})[match.group(1)] = float(row["fpr95tpr"])
    if any(set(values) != {"17", "42", "73"} for values in result.values()):
        raise SupplementError("confidence milestone grid is incomplete")
    return {
        update: {"by_seed": values, **_mean_sd(values[str(seed)] for seed in SEEDS)}
        for update, values in sorted(result.items(), key=lambda item: int(item[0]))
    }


def _write(path: Path, value: dict[str, Any]) -> None:
    path = path.resolve()
    if path.exists():
        raise SupplementError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    val3 = ANCHOR / "formal/ref_val3_admission_u100/summary.json"
    causal = BLOCK / "evaluations/mechanism/A1/val3/summary.json"
    static = BLOCK / "partial_attempts/A2_val3_interrupted_after_seed17/val3/summary.json"
    confidence = ANCHOR / "formal/calibration_u25_u50_u100/summary.json"
    routes = _route_table(causal)
    deployed = _deployed_micro(val3)
    payload = {
        "schema": SCHEMA,
        "inputs": {
            "anchor_val3": _record(val3),
            "causal_routes": _record(causal),
            "static_seed17": _record(static),
            "confidence_grid": _record(confidence),
        },
        "M_cumulative_routes": {
            "M0_B58": routes["b58_base"],
            "M1_positive_only_R100": routes["raw_r100"],
            "M2_static_admission_R100": {
                "status": "bound_by_A2_deployment_parity",
                "parity_receipt": _record(BLOCK / "evaluations/mechanism/A2/deployment_parity.json"),
                "seed17_val3_micro": _deployed_micro(static, allowed_seeds=(17,)),
            },
            "M3_trained_admission_identity_confidence": deployed,
            "M4_full_U2v5": deployed,
            "M3_M4_ref_bitwise_same_route": True,
        },
        "G_error_attribution": _error_attribution(val3),
        "H_sensitivity": {
            "confidence_milestone": _confidence_milestones(confidence),
            "latency": routes["single_forward_seconds"],
            "gap": {"selected": 3.0, "selection_surface": "val3 only"},
        },
        "P_query_geometry": {"status": "requires_additional_zero_training_forward"},
        "S_support_perturbation": {"status": "requires_additional_zero_training_forward"},
        "confirmatory_surfaces_used": False,
    }
    _write(Path(args.output), payload)
    print(json.dumps({"status": "complete", "receipt": _record(Path(args.output))}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, SupplementError) as error:
        raise SystemExit(f"[FAIL] {error}") from error
