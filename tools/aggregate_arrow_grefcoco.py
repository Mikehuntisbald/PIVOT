#!/usr/bin/env python3
"""Aggregate sealed ARROW gRefCOCO records with paired cluster bootstrap."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import stdev
from typing import Any, Callable, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.arrow_grefcoco_common import (
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    PREREG_SCHEMA,
    RECORD_SCHEMA,
    RESULTS_SCHEMA,
    SEALED_THRESHOLDS,
    SEEDS,
    file_record,
    load_json,
    load_jsonl,
    verify_record,
    write_json_atomic,
)

DEFAULT_PREREG = REPO_ROOT / "outputs/arrow_grefcoco_20260820/preregistration.json"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/arrow_grefcoco_20260820/results.json"
DEFAULT_TABLE = REPO_ROOT / "outputs/arrow_grefcoco_20260820/paper_table.md"


def _threshold_at_tpr(scores: np.ndarray, tpr: float = 0.95, weights: np.ndarray | None = None) -> float:
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        raise ValueError("positive score set is empty")
    if weights is None:
        weights = np.ones(values.size, dtype=np.float64)
    order = np.argsort(-values, kind="mergesort")
    ordered_scores = values[order]
    ordered_weights = np.asarray(weights, dtype=np.float64)[order]
    target = tpr * float(ordered_weights.sum())
    index = int(np.searchsorted(np.cumsum(ordered_weights), target, side="left"))
    return float(ordered_scores[min(index, values.size - 1)])


def _weighted_auc(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(scores, kind="mergesort")
    labels = labels[order]
    scores = scores[order]
    weights = weights[order]
    pos_total = float(weights[labels == 1].sum())
    neg_total = float(weights[labels == 0].sum())
    if pos_total <= 0 or neg_total <= 0:
        raise ValueError("binary metric surface lost a class")
    starts = np.flatnonzero(np.r_[True, scores[1:] != scores[:-1]])
    group_pos = np.add.reduceat(weights * (labels == 1), starts)
    group_neg = np.add.reduceat(weights * (labels == 0), starts)
    cumulative_before = np.cumsum(group_neg) - group_neg
    numerator = float(np.sum(group_pos * (cumulative_before + 0.5 * group_neg)))
    return numerator / (pos_total * neg_total)


def _binary_metrics(labels: np.ndarray, scores: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if weights is None:
        weights = np.ones(scores.size, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if labels.shape != scores.shape or weights.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("binary arrays are malformed")
    auroc = _weighted_auc(labels, scores, weights)
    pos_mask = labels == 1
    neg_mask = ~pos_mask
    threshold = _threshold_at_tpr(scores[pos_mask], weights=weights[pos_mask])
    pos_weight = float(weights[pos_mask].sum())
    neg_weight = float(weights[neg_mask].sum())
    fpr95 = float(weights[neg_mask & (scores >= threshold)].sum() / neg_weight)
    actual_tpr = float(weights[pos_mask & (scores >= threshold)].sum() / pos_weight)
    order = np.argsort(-scores, kind="mergesort")
    ordered_scores = scores[order]
    ordered_labels = labels[order]
    ordered_weights = weights[order]
    starts = np.flatnonzero(np.r_[True, ordered_scores[1:] != ordered_scores[:-1]])
    group_pos = np.add.reduceat(ordered_weights * (ordered_labels == 1), starts)
    group_neg = np.add.reduceat(ordered_weights * (ordered_labels == 0), starts)
    tp = np.cumsum(group_pos)
    fp = np.cumsum(group_neg)
    recall = tp / pos_weight
    precision = tp / np.maximum(tp + fp, 1e-12)
    recall_before = np.r_[0.0, recall[:-1]]
    aupr = float(np.sum((recall - recall_before) * precision))
    return {"auroc": float(auroc), "aupr": aupr, "threshold_at_95tpr": threshold, "actual_tpr": actual_tpr, "fpr95": fpr95}


def _surface_predicate(name: str) -> Callable[[Mapping[str, Any]], bool]:
    if name == "full":
        return lambda row: row["split"] in {"testA", "testB"}
    if name == "d3_disjoint":
        return lambda row: row["split"] in {"testA", "testB"} and bool(row["surface_d3_disjoint"])
    if name == "d3_finecops_disjoint":
        return lambda row: row["split"] in {"testA", "testB"} and bool(row["surface_d3_finecops_disjoint"])
    if name in {"testA", "testB"}:
        return lambda row: row["split"] == name
    raise ValueError(f"unknown surface {name}")


def _surface_metrics(rows: list[dict[str, Any]], surface: str, route: str, fixed_threshold: float | None) -> dict[str, Any]:
    selected = [row for row in rows if _surface_predicate(surface)(row)]
    labels = np.asarray([int(row["label"]) for row in selected])
    scores = np.asarray([float(row["scores"][route]) for row in selected])
    result: dict[str, Any] = {"rows": len(selected), "images": len({int(row["image_id"]) for row in selected}), **_binary_metrics(labels, scores)}
    positives = [row for row in selected if row["label"] == 1]
    if positives:
        result["positive_localization"] = {
            "b58_p1": float(np.mean([row["positive_localization"]["b58_top1_iou"] >= 0.5 for row in positives])),
            "r100_p1": float(np.mean([row["positive_localization"]["r100_top1_iou"] >= 0.5 for row in positives])),
        }
    if fixed_threshold is not None:
        pos = scores[labels == 1]
        neg = scores[labels == 0]
        localized = np.asarray([row["positive_localization"]["r100_top1_iou"] >= 0.5 and float(row["scores"][route]) >= fixed_threshold for row in positives])
        result["sealed_threshold"] = {"threshold": fixed_threshold, "positive_tpr": float(np.mean(pos >= fixed_threshold)), "negative_fpr": float(np.mean(neg >= fixed_threshold)), "no_target_n_acc": float(np.mean(neg < fixed_threshold)), "localized_and_accepted": float(np.mean(localized))}
    return result


def _load_runs(prereg: Mapping[str, Any]) -> dict[int, list[dict[str, Any]]]:
    root = Path(prereg["execution"]["results_root"])
    runs: dict[int, list[dict[str, Any]]] = {}
    expected_ids: list[str] | None = None
    b58_reference: list[float] | None = None
    for seed in SEEDS:
        receipt = load_json(root / f"seed{seed}/run_receipt.json")
        if receipt.get("status") != "complete" or receipt.get("rows") != 29589:
            raise ValueError(f"seed{seed} is incomplete")
        path = verify_record(receipt["records"], label=f"seed{seed} records")
        rows = load_jsonl(path)
        if any(row.get("schema") != RECORD_SCHEMA for row in rows):
            raise ValueError(f"seed{seed} record schema drifted")
        ids = [str(row["sample_id"]) for row in rows]
        b58 = [float(row["scores"]["b58"]) for row in rows]
        if expected_ids is None:
            expected_ids, b58_reference = ids, b58
        elif ids != expected_ids or b58 != b58_reference:
            raise ValueError("sample alignment or cross-seed B58 bitwise parity failed")
        runs[seed] = rows
    return runs


def _seed_stats(values: Mapping[int, float]) -> dict[str, Any]:
    numbers = list(values.values())
    return {"by_seed": {str(key): value for key, value in values.items()}, "mean": float(np.mean(numbers)), "sample_sd": float(stdev(numbers))}


def _bootstrap(runs: Mapping[int, list[dict[str, Any]]]) -> dict[str, Any]:
    base_rows = runs[17]
    clusters_by_split = {split: sorted({int(row["image_id"]) for row in base_rows if row["split"] == split and row["split"] in {"testA", "testB"}}) for split in ("testA", "testB")}
    all_clusters = sorted(set().union(*clusters_by_split.values()))
    cluster_index = {value: index for index, value in enumerate(all_clusters)}
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    draws = {surface: {"auroc_gain": [], "fpr95_gain": [], "fixed_tpr": []} for surface in ("full", "d3_disjoint")}
    prepared: dict[str, dict[int | str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(dict)
    for surface in draws:
        predicate = _surface_predicate(surface)
        selected = [row for row in base_rows if predicate(row)]
        labels = np.asarray([int(row["label"]) for row in selected])
        clusters = np.asarray([cluster_index[int(row["image_id"])] for row in selected])
        prepared[surface]["b58"] = (labels, np.asarray([float(row["scores"]["b58"]) for row in selected]), clusters)
        for seed in SEEDS:
            rows = [row for row in runs[seed] if predicate(row)]
            if [row["sample_id"] for row in rows] != [row["sample_id"] for row in selected]:
                raise ValueError("bootstrap rows are not aligned")
            prepared[surface][seed] = (labels, np.asarray([float(row["scores"]["d3"]) for row in rows]), clusters)
    for _ in range(BOOTSTRAP_ITERATIONS):
        weights = np.zeros(len(all_clusters), dtype=np.float64)
        for split in ("testA", "testB"):
            population = np.asarray([cluster_index[value] for value in clusters_by_split[split]], dtype=np.int64)
            sampled = rng.choice(population, size=population.size, replace=True)
            weights += np.bincount(sampled, minlength=len(all_clusters))
        for surface in draws:
            labels, b58_scores, clusters = prepared[surface]["b58"]
            row_weights = weights[clusters]
            keep = row_weights > 0
            b58 = _binary_metrics(labels[keep], b58_scores[keep], row_weights[keep])
            d3_metrics = []
            fixed_tprs = []
            for seed in SEEDS:
                _, scores, _ = prepared[surface][seed]
                metric = _binary_metrics(labels[keep], scores[keep], row_weights[keep])
                d3_metrics.append(metric)
                pos = labels == 1
                fixed_tprs.append(float(row_weights[pos & (scores >= SEALED_THRESHOLDS[seed])].sum() / row_weights[pos].sum()))
            draws[surface]["auroc_gain"].append(float(np.mean([row["auroc"] for row in d3_metrics]) - b58["auroc"]))
            draws[surface]["fpr95_gain"].append(float(b58["fpr95"] - np.mean([row["fpr95"] for row in d3_metrics])))
            draws[surface]["fixed_tpr"].append(float(np.mean(fixed_tprs)))
    result: dict[str, Any] = {}
    for surface, metrics in draws.items():
        result[surface] = {}
        for name, values in metrics.items():
            array = np.asarray(values)
            row = {"ci95": [float(np.quantile(array, 0.025)), float(np.quantile(array, 0.975))]}
            if name != "fixed_tpr":
                row["one_sided_p"] = float((1 + np.sum(array <= 0)) / (array.size + 1))
            result[surface][name] = row
    return {"iterations": BOOTSTRAP_ITERATIONS, "pcg64_seed": BOOTSTRAP_SEED, "unit": "testA_testB_stratified_image_cluster", "surfaces": result}


def aggregate(preregistration: Path, output: Path, table: Path) -> dict[str, Any]:
    prereg = load_json(preregistration)
    if prereg.get("schema") != PREREG_SCHEMA:
        raise ValueError("gRefCOCO preregistration drifted")
    runs = _load_runs(prereg)
    surfaces: dict[str, Any] = {}
    for surface in ("testA", "testB", "full", "d3_disjoint", "d3_finecops_disjoint"):
        b58 = _surface_metrics(runs[17], surface, "b58", None)
        d3_by_seed = {seed: _surface_metrics(runs[seed], surface, "d3", SEALED_THRESHOLDS[seed]) for seed in SEEDS}
        surfaces[surface] = {
            "b58": b58,
            "d3": d3_by_seed,
            "d3_summary": {
                metric: _seed_stats({seed: float(d3_by_seed[seed][metric]) for seed in SEEDS})
                for metric in ("auroc", "aupr", "fpr95")
            },
            "fixed_threshold_summary": {
                metric: _seed_stats({seed: float(d3_by_seed[seed]["sealed_threshold"][metric]) for seed in SEEDS})
                for metric in ("positive_tpr", "negative_fpr", "no_target_n_acc", "localized_and_accepted")
            },
        }
    val: dict[str, Any] = {}
    for seed in SEEDS:
        rows = [row for row in runs[seed] if row["split"] == "val"]
        scores = np.asarray([float(row["scores"]["d3"]) for row in rows])
        val[str(seed)] = {"rows": len(rows), "threshold": SEALED_THRESHOLDS[seed], "negative_fpr": float(np.mean(scores >= SEALED_THRESHOLDS[seed])), "no_target_n_acc": float(np.mean(scores < SEALED_THRESHOLDS[seed]))}
    bootstrap = _bootstrap(runs)
    primary = bootstrap["surfaces"]["d3_disjoint"]
    full = bootstrap["surfaces"]["full"]
    ordering_gate = all(row[metric]["ci95"][0] > 0 for row in (primary, full) for metric in ("auroc_gain", "fpr95_gain"))
    fixed_ci = primary["fixed_tpr"]["ci95"]
    operating_shift = not (fixed_ci[0] <= 0.95 <= fixed_ci[1])
    claim = (
        "parameter-isolated confidence consistently improves cross-benchmark rejection ordering, while absolute operating-point calibration remains domain dependent"
        if ordering_gate and operating_shift
        else "ordering_improvement_only" if ordering_gate
        else "finecops_phenomenon_not_confirmed_on_d3_disjoint_grefcoco"
    )
    payload = {"schema": RESULTS_SCHEMA, "status": "complete", "preregistration": file_record(preregistration), "surfaces": surfaces, "val_no_target": val, "bootstrap": bootstrap, "decision": {"ordering_gate_passed": ordering_gate, "operating_point_shift": operating_shift, "claim": claim}, "claim_boundary": prereg["claim_boundary"]}
    write_json_atomic(output, payload)
    lines = ["# ARROW × gRefCOCO rejection transfer", "", "| Surface | Model | AUROC | AUPR | FPR95 | Fixed TPR | Fixed N-acc |", "|---|---|---:|---:|---:|---:|---:|"]
    for surface in ("testA", "testB", "full", "d3_disjoint", "d3_finecops_disjoint"):
        row = surfaces[surface]
        lines.append(f"| {surface} | B58 | {row['b58']['auroc']:.4f} | {row['b58']['aupr']:.4f} | {row['b58']['fpr95']:.4f} | — | — |")
        lines.append(f"| {surface} | D3 mean | {row['d3_summary']['auroc']['mean']:.4f} | {row['d3_summary']['aupr']['mean']:.4f} | {row['d3_summary']['fpr95']['mean']:.4f} | {row['fixed_threshold_summary']['positive_tpr']['mean']:.4f} | {row['fixed_threshold_summary']['no_target_n_acc']['mean']:.4f} |")
    lines.extend(["", f"Decision: `{claim}`", "", "This is annotation/task-zero-shot transfer on previously exposed COCO imagery, not image-disjoint zero-shot.", ""])
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text("\n".join(lines), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, default=DEFAULT_PREREG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    args = parser.parse_args()
    result = aggregate(args.preregistration.resolve(strict=True), args.output.resolve(), args.table.resolve())
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
