#!/usr/bin/env python3
"""Build a zero-training cross-benchmark confidence-anatomy receipt.

The receipt is derived exclusively from sealed per-example records and result
artifacts.  It distinguishes per-domain diagnostic FPR95 (whose q05 is
recomputed within that domain) from the deployment threshold sealed on D3
calibration.  No score or threshold is fitted here.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "data" / "confidence_anatomy.json"
QUANTILES_OUT = ROOT / "paper" / "data" / "plot_sources" / "confidence_quantiles.csv"

BASE_INTERNAL = ROOT / "outputs/paper_cvpr_v1/baseline_b58_strict2031_seed42_v3_contract/per_example_records/gdino_ft_stageb_from_gdino_ft_e1_with_tn_bs19_nopatchbranch_checkpoint0001__tn_global.records.jsonl"
D3_INTERNAL = {
    seed: ROOT / f"outputs/u2v5_leakage_clean_anchor_20260817/final_once/strict2031_u50/per_example_records/confidence_seed{seed}_u50_checkpoint_iter__tn_global.records.jsonl"
    for seed in (17, 42, 73)
}
CALIBRATION = {
    seed: ROOT / f"outputs/u2v5_leakage_clean_anchor_20260817/formal/calibration_u25_u50_u100/confidence_seed{seed}_u50_checkpoint_iter__tn_val.json"
    for seed in (17, 42, 73)
}
FINE_RESULTS = ROOT / "outputs/arrow_finecops_20260819/results.json"
FINE_RECORDS = {
    seed: ROOT / f"outputs/arrow_finecops_20260819/evaluations/B/seed{seed}/records.jsonl"
    for seed in (17, 42, 73)
}
GREF_RESULTS = ROOT / "outputs/arrow_grefcoco_20260820/results.json"
GREF_RECORDS = {
    seed: ROOT / f"outputs/arrow_grefcoco_20260820/evaluations_v2/seed{seed}/records.jsonl"
    for seed in (17, 42, 73)
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256(path), "size_bytes": path.stat().st_size}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def exact_threshold_at_tpr(positive: np.ndarray, target_tpr: float = 0.95) -> float:
    accepted = max(1, int(math.ceil(target_tpr * positive.size)))
    index = positive.size - accepted
    return float(np.partition(positive, index)[index])


def auroc(positive: np.ndarray, negative: np.ndarray) -> float:
    values = np.concatenate([positive, negative])
    labels = np.concatenate([np.ones(positive.size, dtype=bool), np.zeros(negative.size, dtype=bool)])
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    rank_sum = float(ranks[labels].sum())
    numerator = rank_sum - positive.size * (positive.size + 1) / 2.0
    return numerator / float(positive.size * negative.size)


def metrics(positive: Iterable[float], negative: Iterable[float]) -> dict[str, float]:
    pos = np.asarray(list(positive), dtype=np.float64)
    neg = np.asarray(list(negative), dtype=np.float64)
    if pos.size == 0 or neg.size == 0 or not np.isfinite(pos).all() or not np.isfinite(neg).all():
        raise ValueError("binary score arrays must be non-empty and finite")
    threshold = exact_threshold_at_tpr(pos)
    return {
        "positive_count": int(pos.size),
        "negative_count": int(neg.size),
        "auroc": auroc(pos, neg),
        "fpr95": float(np.mean(neg >= threshold)),
        "q05_threshold": threshold,
        "actual_tpr": float(np.mean(pos >= threshold)),
    }


def quantile_row(benchmark: str, route: str, label: str, seed: int | str, values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    result: dict[str, Any] = {
        "benchmark": benchmark, "route": route, "label": label, "seed": seed,
        "count": int(array.size), "mean": float(np.mean(array)), "sd": float(np.std(array)),
    }
    for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99):
        result[f"q{int(q * 100):02d}"] = float(np.quantile(array, q))
    return result


def internal_surface(quantiles: list[dict[str, Any]]) -> dict[str, Any]:
    base_rows = read_jsonl(BASE_INTERNAL)
    base_ids = [row["sample_id"] for row in base_rows]
    base_pos = [float(row["pos_score"]) for row in base_rows]
    base_neg = [float(row["neg_score"]) for row in base_rows]
    base_metric = metrics(base_pos, base_neg)
    quantiles.extend([
        quantile_row("Internal Strict-TN2031", "Frozen base", "positive", "fixed", base_pos),
        quantile_row("Internal Strict-TN2031", "Frozen base", "negative", "fixed", base_neg),
    ])

    by_seed: dict[str, Any] = {}
    for seed, path in D3_INTERNAL.items():
        rows = read_jsonl(path)
        if [row["sample_id"] for row in rows] != base_ids:
            raise RuntimeError(f"internal sample alignment failed for seed {seed}")
        pos = [float(row["pos_score"]) for row in rows]
        neg = [float(row["neg_score"]) for row in rows]
        metric = metrics(pos, neg)
        calibration = json.loads(CALIBRATION[seed].read_text(encoding="utf-8"))
        tau = float(calibration["threshold_at_95tpr"])
        metric.update({
            "sealed_d3_threshold": tau,
            "fixed_threshold_positive_tpr": float(np.mean(np.asarray(pos) >= tau)),
            "fixed_threshold_negative_fpr": float(np.mean(np.asarray(neg) >= tau)),
        })
        by_seed[str(seed)] = metric
        quantiles.extend([
            quantile_row("Internal Strict-TN2031", "Isolated rejector", "positive", seed, pos),
            quantile_row("Internal Strict-TN2031", "Isolated rejector", "negative", seed, neg),
        ])

    return {
        "surface": "Strict-TN2031",
        "frozen_base": base_metric,
        "isolated_rejector_by_seed": by_seed,
        "isolated_rejector_mean": {
            key: statistics.fmean(float(by_seed[str(seed)][key]) for seed in (17, 42, 73))
            for key in ("auroc", "fpr95", "actual_tpr", "fixed_threshold_positive_tpr", "fixed_threshold_negative_fpr")
        },
        "auroc_gain": statistics.fmean(by_seed[str(seed)]["auroc"] for seed in (17, 42, 73)) - base_metric["auroc"],
        "fpr95_reduction": base_metric["fpr95"] - statistics.fmean(by_seed[str(seed)]["fpr95"] for seed in (17, 42, 73)),
        "fixed_target_tpr": 0.95,
        "fixed_threshold_contract": "each seed threshold is copied from sealed 1,570-row D3 calibration",
    }


def finecops_surface(quantiles: list[dict[str, Any]]) -> dict[str, Any]:
    result = json.loads(FINE_RESULTS.read_text(encoding="utf-8"))
    route_metrics: dict[str, dict[str, float]] = {}
    for route_key, output_name in (("B58", "frozen_base"), ("R100_D3", "isolated_rejector")):
        values: dict[str, list[float]] = {key: [] for key in ("auroc", "fpr95")}
        fixed_tpr: list[float] = []
        for seed in (17, 42, 73):
            rejection = result["metrics"][route_key][str(seed)]["rejection"]
            for kind in ("text", "image"):
                audited = rejection[kind]["audited_all_positive"]
                values["auroc"].append(float(audited["auroc"]))
                values["fpr95"].append(float(audited["fpr95"]))
            if route_key == "R100_D3":
                fixed_tpr.append(float(rejection["text"]["sealed_d3_threshold"]["positive_tpr"]))
        route_metrics[output_name] = {
            "auroc_type_macro": statistics.fmean(values["auroc"]),
            "fpr95_type_macro": statistics.fmean(values["fpr95"]),
        }
        if fixed_tpr:
            route_metrics[output_name]["fixed_threshold_positive_tpr"] = statistics.fmean(fixed_tpr)

    for seed, path in FINE_RECORDS.items():
        rows = read_jsonl(path)
        for route, record_key in (("Frozen base", "b58"), ("Isolated rejector", "deployed")):
            for label, predicate in (
                ("positive", lambda row: row["kind"] == "positive"),
                ("negative-text", lambda row: row["kind"] == "text"),
                ("negative-image", lambda row: row["kind"] == "image"),
            ):
                values = [row["routes"][record_key]["raw_confidence"] for row in rows if predicate(row)]
                quantiles.append(quantile_row("FineCops-Ref", route, label, seed, values))

    return {
        "surface": "FineCops audited all-positive type macro",
        **route_metrics,
        "auroc_gain": route_metrics["isolated_rejector"]["auroc_type_macro"] - route_metrics["frozen_base"]["auroc_type_macro"],
        "fpr95_reduction": route_metrics["frozen_base"]["fpr95_type_macro"] - route_metrics["isolated_rejector"]["fpr95_type_macro"],
        "fixed_target_tpr": 0.95,
        "fixed_threshold_contract": "sealed D3 threshold; no FineCops fitting",
        "inference": "point estimates only; no D3-vs-base rejection bootstrap receipt exists",
    }


def gref_surface(quantiles: list[dict[str, Any]]) -> dict[str, Any]:
    result = json.loads(GREF_RESULTS.read_text(encoding="utf-8"))
    surface = result["surfaces"]["full"]
    for seed, path in GREF_RECORDS.items():
        rows = [row for row in read_jsonl(path) if row["split"] in {"testA", "testB"}]
        for route, key in (("Frozen base", "b58"), ("Isolated rejector", "d3")):
            for label, wanted in (("positive", 1), ("no-target", 0)):
                values = [row["scores"][key] for row in rows if int(row["label"]) == wanted]
                quantiles.append(quantile_row("gRefCOCO restricted Full", route, label, seed, values))
    d3 = surface["d3_summary"]
    return {
        "surface": "gRefCOCO restricted Full single/no-target",
        "frozen_base": {key: float(surface["b58"][key]) for key in ("auroc", "fpr95")},
        "isolated_rejector": {
            "auroc": float(d3["auroc"]["mean"]),
            "fpr95": float(d3["fpr95"]["mean"]),
            "fixed_threshold_positive_tpr": float(surface["fixed_threshold_summary"]["positive_tpr"]["mean"]),
        },
        "auroc_gain": float(d3["auroc"]["mean"] - surface["b58"]["auroc"]),
        "fpr95_reduction": float(surface["b58"]["fpr95"] - d3["fpr95"]["mean"]),
        "fixed_target_tpr": 0.95,
        "fixed_threshold_contract": "sealed D3 threshold; no gRefCOCO fitting",
    }


def main() -> None:
    source_paths = [BASE_INTERNAL, *D3_INTERNAL.values(), *CALIBRATION.values(), FINE_RESULTS,
                    *FINE_RECORDS.values(), GREF_RESULTS, *GREF_RECORDS.values()]
    inputs = {str(index): artifact(path) for index, path in enumerate(source_paths)}
    quantiles: list[dict[str, Any]] = []
    surfaces = {
        "internal": internal_surface(quantiles),
        "finecops": finecops_surface(quantiles),
        "grefcoco": gref_surface(quantiles),
    }
    payload = {
        "schema": "arrow.paper.confidence_anatomy/v1",
        "status": "zero_training_derived_from_sealed_records",
        "inputs": inputs,
        "surfaces": surfaces,
        "shared_conclusion": "rejection ordering improves on all three surfaces; the sealed 95%-TPR source operating point does not transfer to either external benchmark",
        "claim_boundary": "domain-derived FPR95 is diagnostic and recomputes domain q05; it is not a deployment threshold",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    QUANTILES_OUT.parent.mkdir(parents=True, exist_ok=True)
    with QUANTILES_OUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(quantiles[0]))
        writer.writeheader(); writer.writerows(quantiles)
    print(json.dumps({name: {
        "auroc_gain": value["auroc_gain"], "fpr95_reduction": value["fpr95_reduction"],
        "fixed_tpr": (value.get("isolated_rejector_mean") or value.get("isolated_rejector"))["fixed_threshold_positive_tpr"],
    } for name, value in surfaces.items()}, indent=2))


if __name__ == "__main__":
    main()
