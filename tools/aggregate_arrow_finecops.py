#!/usr/bin/env python3
"""Aggregate ARROW FineCops records with audited paired statistics."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import stdev
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.arrow_finecops_common import (
    BOOTSTRAP_ITERATIONS,
    BOOTSTRAP_SEED,
    PREREG_SCHEMA,
    RECORD_SCHEMA,
    RESULTS_SCHEMA,
    file_record,
    load_json,
    load_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)


DEFAULT_PREREG = REPO_ROOT / "outputs/arrow_finecops_20260819/preregistration.json"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/arrow_finecops_20260819/results.json"


def _verify_record(expected: Mapping[str, Any], *, label: str) -> Path:
    path = Path(str(expected["path"])).resolve(strict=True)
    observed = file_record(path)
    for key in ("sha256", "size_bytes"):
        if observed[key] != expected.get(key):
            raise ValueError(f"{label} {key} drifted")
    return path


def _load_runs(prereg: Mapping[str, Any]) -> dict[str, dict[int, list[dict[str, Any]]]]:
    root = Path(str(prereg["execution"]["results_root"])).resolve()
    result: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for route in ("A", "B", "C"):
        result[route] = {}
        for seed in (17, 42, 73):
            receipt_path = root / route / f"seed{seed}" / "run_receipt.json"
            receipt = load_json(receipt_path)
            if receipt.get("schema") != "arrow.finecops.run_receipt/v1" or receipt.get("status") != "complete":
                raise ValueError(f"{route}/seed{seed} is not a complete formal run")
            records_path = _verify_record(receipt["records"], label=f"{route}/seed{seed} records")
            rows = load_jsonl(records_path)
            if len(rows) != 27926:
                raise ValueError(f"{route}/seed{seed} rows {len(rows)} != 27926")
            if any(row.get("schema") != RECORD_SCHEMA for row in rows):
                raise ValueError(f"{route}/seed{seed} record schema drifted")
            ids = [str(row["sample_id"]) for row in rows]
            if len(set(ids)) != len(ids):
                raise ValueError(f"{route}/seed{seed} duplicate sample IDs")
            result[route][seed] = rows
    return result


def _assert_parity(runs: Mapping[str, Mapping[int, list[dict[str, Any]]]]) -> dict[str, Any]:
    route_maps = {
        route: {seed: {row["sample_id"]: row for row in rows} for seed, rows in seeds.items()}
        for route, seeds in runs.items()
    }
    reference_ids = list(route_maps["A"][17])
    for route in ("A", "B", "C"):
        for seed in (17, 42, 73):
            if list(route_maps[route][seed]) != reference_ids:
                raise ValueError(f"{route}/seed{seed} sample order drifted")
    compared = 0
    for sample_id in reference_ids:
        baseline_reference = route_maps["A"][17][sample_id]["routes"]["b58"]
        rank_reference = route_maps["A"][17][sample_id]["routes"]["r100_d3"]
        for route in ("A", "B", "C"):
            for seed in (17, 42, 73):
                row = route_maps[route][seed][sample_id]
                b58 = row["routes"]["b58"]
                rank = row["routes"]["r100_d3"]
                for key in ("top1_query_index", "top1_box_xyxy", "top1_iou", "raw_confidence"):
                    if b58[key] != baseline_reference[key]:
                        raise ValueError(f"B58 parity drift at {sample_id}/{route}/{seed}/{key}")
                for key in ("top1_query_index", "top1_box_xyxy", "top1_iou"):
                    if rank[key] != rank_reference[key]:
                        raise ValueError(f"R100 parity drift at {sample_id}/{route}/{seed}/{key}")
        for seed in (17, 42, 73):
            confidence_reference = route_maps["A"][seed][sample_id]["routes"]["deployed"]["raw_confidence"]
            for route in ("B", "C"):
                if route_maps[route][seed][sample_id]["routes"]["deployed"]["raw_confidence"] != confidence_reference:
                    raise ValueError(f"D3 parity drift at {sample_id}/{route}/seed{seed}")
        compared += 1
    return {
        "status": "bitwise_equal",
        "records_compared": compared,
        "b58_across_routes_and_seeds": True,
        "r100_geometry_across_routes_and_seeds": True,
        "d3_confidence_across_routes_within_seed": True,
    }


def _threshold_at_tpr(scores: np.ndarray, tpr: float = 0.95) -> float:
    values = np.sort(np.asarray(scores, dtype=np.float64))[::-1]
    if values.size == 0:
        raise ValueError("positive score set is empty")
    index = max(0, min(values.size - 1, int(math.ceil(tpr * values.size)) - 1))
    return float(values[index])


def _binary_metrics(positive: list[float], negative: list[float]) -> dict[str, Any]:
    pos = np.asarray(positive, dtype=np.float64)
    neg = np.asarray(negative, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        raise ValueError("binary metric surface must contain both labels")
    labels = np.concatenate((np.ones(pos.size, dtype=np.int64), np.zeros(neg.size, dtype=np.int64)))
    scores = np.concatenate((pos, neg))
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    rank_sum = float(ranks[labels == 1].sum())
    auroc = (
        rank_sum - pos.size * (pos.size + 1) / 2.0
    ) / float(pos.size * neg.size)

    descending = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[descending]
    sorted_desc_scores = scores[descending]
    tp = 0
    fp = 0
    previous_recall = 0.0
    aupr = 0.0
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_desc_scores[end] == sorted_desc_scores[start]:
            end += 1
        group = sorted_labels[start:end]
        tp += int(group.sum())
        fp += int(group.size - group.sum())
        recall = tp / float(pos.size)
        precision = tp / float(tp + fp)
        aupr += (recall - previous_recall) * precision
        previous_recall = recall
        start = end
    threshold = _threshold_at_tpr(pos)
    return {
        "positive_count": int(pos.size),
        "negative_count": int(neg.size),
        "auroc": float(auroc),
        "aupr": float(aupr),
        "threshold_at_95tpr": threshold,
        "actual_tpr": float(np.mean(pos >= threshold)),
        "fpr95": float(np.mean(neg >= threshold)),
    }


def _surface_metrics(
    rows: list[dict[str, Any]],
    *,
    output_route: str,
    support_matched: bool,
    fixed_threshold: float | None,
) -> dict[str, Any]:
    positives = {
        int(row["finecops_annotation_id"]): row
        for row in rows
        if row["kind"] == "positive"
        and (not support_matched or bool(row["support_covered"]))
    }
    by_level: dict[str, list[bool]] = defaultdict(list)
    for row in positives.values():
        by_level[str(int(row["level"]))].append(
            float(row["routes"][output_route]["top1_iou"]) >= 0.5
        )
    precision = {key: float(np.mean(value)) for key, value in sorted(by_level.items())}
    precision_macro = float(np.mean(list(precision.values())))
    precision_micro = float(np.mean([value for values in by_level.values() for value in values]))

    rejection: dict[str, Any] = {}
    for kind in ("text", "image"):
        negative_rows = [
            row
            for row in rows
            if row["kind"] == kind
            and int(row["parent_positive_id"]) in positives
            and (not support_matched or bool(row["support_covered"]))
        ]
        wins: list[bool] = []
        paired_positive_scores: list[float] = []
        negative_scores: list[float] = []
        by_type: dict[str, list[bool]] = defaultdict(list)
        for row in negative_rows:
            parent = positives[int(row["parent_positive_id"])]
            pos_payload = parent["routes"][output_route]
            neg_payload = row["routes"][output_route]
            success = (
                float(pos_payload["top1_iou"]) >= 0.5
                and float(pos_payload["official_probability"])
                > float(neg_payload["official_probability"])
            )
            wins.append(success)
            key = f"{row.get('negative_type')}|L{row.get('negative_level')}"
            by_type[key].append(success)
            paired_positive_scores.append(float(pos_payload["official_probability"]))
            negative_scores.append(float(neg_payload["official_probability"]))
        unique_positive_scores = [
            float(row["routes"][output_route]["official_probability"])
            for row in positives.values()
        ]
        level1_positive_scores = [
            float(row["routes"][output_route]["official_probability"])
            for row in positives.values()
            if int(row["level"]) == 1
        ]
        row_result: dict[str, Any] = {
            "negative_count": len(negative_rows),
            "recall1_strict_tie_fail": float(np.mean(wins)),
            "recall1_by_type_level": {
                key: {"count": len(value), "recall1": float(np.mean(value))}
                for key, value in sorted(by_type.items())
            },
            "official_exact_scope": _binary_metrics(level1_positive_scores, negative_scores),
            "audited_all_positive": _binary_metrics(unique_positive_scores, negative_scores),
            "paired_one_to_one": _binary_metrics(paired_positive_scores, negative_scores),
        }
        if fixed_threshold is not None:
            raw_pos = np.asarray(
                [float(row["routes"][output_route]["raw_confidence"]) for row in positives.values()]
            )
            raw_neg = np.asarray(
                [float(row["routes"][output_route]["raw_confidence"]) for row in negative_rows]
            )
            localization_accept = np.asarray(
                [
                    float(row["routes"][output_route]["top1_iou"]) >= 0.5
                    and float(row["routes"][output_route]["raw_confidence"]) >= fixed_threshold
                    for row in positives.values()
                ]
            )
            row_result["sealed_d3_threshold"] = {
                "threshold": fixed_threshold,
                "positive_tpr": float(np.mean(raw_pos >= fixed_threshold)),
                "negative_fpr": float(np.mean(raw_neg >= fixed_threshold)),
                "localized_and_accepted": float(np.mean(localization_accept)),
            }
        rejection[kind] = row_result
    return {
        "positive_count": len(positives),
        "precision1_by_level": precision,
        "precision1_macro": precision_macro,
        "precision1_micro": precision_micro,
        "rejection": rejection,
    }


def _seed_summary(values: Mapping[int, Mapping[str, Any]], path: tuple[str, ...]) -> dict[str, Any]:
    selected: dict[str, float] = {}
    for seed, payload in values.items():
        current: Any = payload
        for key in path:
            current = current[key]
        selected[str(seed)] = float(current)
    numbers = list(selected.values())
    return {
        "by_seed": selected,
        "mean": float(np.mean(numbers)),
        "sample_sd": float(stdev(numbers)),
    }


def _holm(raw: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=raw.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, key in enumerate(ordered):
        value = min(1.0, (count - index) * float(raw[key]))
        running = max(running, value)
        adjusted[key] = running
    return adjusted


def _bootstrap(runs: Mapping[str, Mapping[int, list[dict[str, Any]]]]) -> dict[str, Any]:
    clusters = sorted(
        {
            int(row["cluster_gqa_image_id"])
            for row in runs["A"][17]
            if row["kind"] == "positive"
        }
    )
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    cluster_index = {cluster: index for index, cluster in enumerate(clusters)}
    # [cluster, level, (correct,total)] keeps the 5,000 draws in NumPy rather
    # than rescanning 27,926 JSON records for every replicate.
    stats: dict[str, dict[int, np.ndarray]] = {
        route: {
            seed: np.zeros((len(clusters), 3, 2), dtype=np.float64)
            for seed in (17, 42, 73)
        }
        for route in ("A", "B", "C")
    }
    rejection_stats: dict[str, dict[int, dict[str, np.ndarray]]] = {
        route: {
            seed: {
                kind: np.zeros((len(clusters), 2), dtype=np.float64)
                for kind in ("text", "image")
            }
            for seed in (17, 42, 73)
        }
        for route in ("A", "B", "C")
    }
    for route in ("A", "B", "C"):
        for seed in (17, 42, 73):
            array = stats[route][seed]
            for row in runs[route][seed]:
                if row["kind"] != "positive" or not bool(row["support_covered"]):
                    continue
                cluster = cluster_index[int(row["cluster_gqa_image_id"])]
                level = int(row["level"]) - 1
                array[cluster, level, 1] += 1.0
                array[cluster, level, 0] += float(
                    float(row["routes"]["deployed"]["top1_iou"]) >= 0.5
                )
            positives = {
                int(row["finecops_annotation_id"]): row
                for row in runs[route][seed]
                if row["kind"] == "positive" and bool(row["support_covered"])
            }
            for row in runs[route][seed]:
                kind = str(row["kind"])
                if (
                    kind not in {"text", "image"}
                    or not bool(row["support_covered"])
                    or int(row["parent_positive_id"]) not in positives
                ):
                    continue
                parent = positives[int(row["parent_positive_id"])]
                cluster = cluster_index[int(row["cluster_gqa_image_id"])]
                success = (
                    float(parent["routes"]["deployed"]["top1_iou"]) >= 0.5
                    and float(parent["routes"]["deployed"]["official_probability"])
                    > float(row["routes"]["deployed"]["official_probability"])
                )
                rejection_stats[route][seed][kind][cluster, 0] += float(success)
                rejection_stats[route][seed][kind][cluster, 1] += 1.0

    draws: dict[str, list[float]] = {
        "positive_A_minus_B": [],
        "positive_B_minus_C": [],
        "text_A_minus_B": [],
        "text_B_minus_C": [],
        "image_A_minus_B": [],
        "image_B_minus_C": [],
    }
    point: dict[str, float] = {}

    def mean_route(route: str, weights: np.ndarray) -> float:
        seed_metrics: list[float] = []
        for seed in (17, 42, 73):
            totals = np.tensordot(weights, stats[route][seed], axes=(0, 0))
            if bool((totals[:, 1] <= 0).any()):
                raise ValueError("bootstrap draw lost a FineCops difficulty level")
            seed_metrics.append(float(np.mean(totals[:, 0] / totals[:, 1])))
        return float(np.mean(seed_metrics))

    def mean_rejection(route: str, kind: str, weights: np.ndarray) -> float:
        values: list[float] = []
        for seed in (17, 42, 73):
            totals = np.tensordot(
                weights, rejection_stats[route][seed][kind], axes=(0, 0)
            )
            if totals[1] <= 0:
                raise ValueError("bootstrap draw lost a FineCops rejection family")
            values.append(float(totals[0] / totals[1]))
        return float(np.mean(values))

    unit_weights = np.ones((len(clusters),), dtype=np.float64)
    point_values = {route: mean_route(route, unit_weights) for route in ("A", "B", "C")}
    point["positive_A_minus_B"] = point_values["A"] - point_values["B"]
    point["positive_B_minus_C"] = point_values["B"] - point_values["C"]
    for kind in ("text", "image"):
        rejection_values = {
            route: mean_rejection(route, kind, unit_weights)
            for route in ("A", "B", "C")
        }
        point[f"{kind}_A_minus_B"] = rejection_values["A"] - rejection_values["B"]
        point[f"{kind}_B_minus_C"] = rejection_values["B"] - rejection_values["C"]
    for _ in range(BOOTSTRAP_ITERATIONS):
        sampled = rng.integers(0, len(clusters), size=len(clusters))
        counts = np.bincount(sampled, minlength=len(clusters))
        weights = counts.astype(np.float64, copy=False)
        values = {route: mean_route(route, weights) for route in ("A", "B", "C")}
        draws["positive_A_minus_B"].append(values["A"] - values["B"])
        draws["positive_B_minus_C"].append(values["B"] - values["C"])
        for kind in ("text", "image"):
            rejection_values = {
                route: mean_rejection(route, kind, weights)
                for route in ("A", "B", "C")
            }
            draws[f"{kind}_A_minus_B"].append(
                rejection_values["A"] - rejection_values["B"]
            )
            draws[f"{kind}_B_minus_C"].append(
                rejection_values["B"] - rejection_values["C"]
            )
    raw_p = {
        key: float((1 + np.sum(np.asarray(value) <= 0.0)) / (BOOTSTRAP_ITERATIONS + 1))
        for key, value in draws.items()
    }
    families = {
        "positive": ("positive_A_minus_B", "positive_B_minus_C"),
        "text": ("text_A_minus_B", "text_B_minus_C"),
        "image": ("image_A_minus_B", "image_B_minus_C"),
    }
    adjusted: dict[str, float] = {}
    for keys in families.values():
        adjusted.update(_holm({key: raw_p[key] for key in keys}))
    return {
        "iterations": BOOTSTRAP_ITERATIONS,
        "seed": BOOTSTRAP_SEED,
        "cluster": "parent_positive_gqa_image",
        "contrasts": {
            key: {
                "gain": point[key],
                "ci95": [
                    float(np.percentile(value, 2.5)),
                    float(np.percentile(value, 97.5)),
                ],
                "one_sided_p": raw_p[key],
                "holm_adjusted_p": adjusted[key],
                "family": key.split("_", 1)[0],
            }
            for key, value in draws.items()
        },
    }


def _write_official_predictions(
    output_dir: Path,
    runs: Mapping[str, Mapping[int, list[dict[str, Any]]]],
) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    surfaces = {
        "A": ("A", "deployed"),
        "B": ("B", "deployed"),
        "C": ("C", "deployed"),
        "B58": ("B", "b58"),
        "R100_D3": ("B", "r100_d3"),
    }
    for surface, (record_route, output_route) in surfaces.items():
        artifacts[surface] = {}
        for seed in (17, 42, 73):
            rows = runs[record_route][seed]
            output = output_dir / f"{surface}_seed{seed}.jsonl"
            covered_parents = {
                int(row["finecops_annotation_id"])
                for row in rows
                if row["kind"] == "positive" and bool(row["support_covered"])
            }
            predictions = []
            for row in rows:
                if surface == "A" and (
                    not bool(row["support_covered"])
                    or int(row["parent_positive_id"]) not in covered_parents
                ):
                    continue
                payload = row["routes"][output_route]
                predictions.append(
                    {
                        "image_id": int(row["finecops_annotation_id"]),
                        "bboxes": [payload["top1_box_xyxy"]],
                        "scores": [payload["official_probability"]],
                    }
                )
            write_jsonl_atomic(output, predictions)
            artifacts[surface][str(seed)] = file_record(output, rows=len(predictions))
    return artifacts


def aggregate(preregistration_path: Path, output_path: Path) -> dict[str, Any]:
    prereg = load_json(preregistration_path)
    if prereg.get("schema") != PREREG_SCHEMA:
        raise ValueError("FineCops preregistration schema drifted")
    runs = _load_runs(prereg)
    parity = _assert_parity(runs)
    thresholds = {
        int(seed): float(row["raw_threshold_at_95tpr"])
        for seed, row in prereg["calibration"]["thresholds"].items()
    }
    metrics: dict[str, dict[int, dict[str, Any]]] = {}
    for surface in ("A", "B", "C"):
        metrics[surface] = {
            seed: _surface_metrics(
                runs[surface][seed],
                output_route="deployed",
                support_matched=(surface == "A"),
                fixed_threshold=thresholds[seed],
            )
            for seed in (17, 42, 73)
        }
    metrics["B58"] = {
        seed: _surface_metrics(
            runs["B"][seed], output_route="b58", support_matched=False, fixed_threshold=None
        )
        for seed in (17, 42, 73)
    }
    metrics["R100_D3"] = {
        seed: _surface_metrics(
            runs["B"][seed],
            output_route="r100_d3",
            support_matched=False,
            fixed_threshold=thresholds[seed],
        )
        for seed in (17, 42, 73)
    }
    seed_aggregates = {
        surface: {
            "precision1_macro": _seed_summary(values, ("precision1_macro",)),
            "precision1_micro": _seed_summary(values, ("precision1_micro",)),
            "text_recall1": _seed_summary(values, ("rejection", "text", "recall1_strict_tie_fail")),
            "image_recall1": _seed_summary(values, ("rejection", "image", "recall1_strict_tie_fail")),
        }
        for surface, values in metrics.items()
    }
    a_lower_bound_by_seed = {
        str(seed): sum(
            float(row["routes"]["deployed"]["top1_iou"]) >= 0.5
            for row in runs["A"][seed]
            if row["kind"] == "positive" and bool(row["support_covered"])
        )
        / 9605
        for seed in (17, 42, 73)
    }
    prediction_artifacts = _write_official_predictions(
        output_path.parent / "official_predictions", runs
    )
    payload = {
        "schema": RESULTS_SCHEMA,
        "status": "aggregated",
        "preregistration": file_record(preregistration_path),
        "claim": "finecops_specific_external_zero_shot_not_image_disjoint",
        "parity": parity,
        "metrics": metrics,
        "seed_aggregates": seed_aggregates,
        "a_support_contract": {
            "covered_positive_count": 9182,
            "total_positive_count": 9605,
            "coverage": 9182 / 9605,
            "coverage_penalized_precision1_lower_bound_by_seed": a_lower_bound_by_seed,
            "coverage_penalized_precision1_lower_bound_mean": float(
                np.mean(list(a_lower_bound_by_seed.values()))
            ),
            "unsupported_negative_counted_as_rejection": False,
        },
        "bootstrap": _bootstrap(runs),
        "official_prediction_artifacts": prediction_artifacts,
        "official_exact_status": "pending_external_pinned_evaluator",
        "finecops_threshold_fitted": False,
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
