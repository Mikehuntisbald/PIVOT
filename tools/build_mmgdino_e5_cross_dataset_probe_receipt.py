#!/usr/bin/env python3
"""Validate and aggregate the zero-update e5 cross-dataset gradient probe."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from tools.probe_mmgdino_e5_cross_dataset_gradients import ROUTES, SCHEMA as PROBE_SCHEMA
from tools.responsibility_isolation_cache import file_sha256

SCHEMA = "arrow.mmgdino_e5_cross_dataset_probe.final_receipt/v1"
DATASETS = ("refcoco", "refcocoplus", "refcocog")
SEEDS = (17, 42, 73)
METRICS = (
    "cosine",
    "sign_conflict_fraction",
    "rank_gradient_l2",
    "confidence_gradient_l2",
    "native_p1",
    "rank_loss",
    "rank_fix_loss",
    "rank_preserve_loss",
    "native_top1_runnerup_margin",
    "native_oracle_positive_negative_gap",
)


class ProbeReceiptError(RuntimeError):
    pass


def _binding(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _mean_sd(values: Sequence[float]) -> dict[str, Any]:
    return {
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": list(values),
    }


def aggregate(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != PROBE_SCHEMA or payload.get("status") != "complete_zero_update":
        raise ProbeReceiptError("probe result is incomplete or has wrong schema")
    if payload.get("contract", {}).get("parameters_updated") != 0:
        raise ProbeReceiptError("probe updated parameters")
    if payload.get("contract", {}).get("optimizer_created") is not False:
        raise ProbeReceiptError("probe created an optimizer")
    aggregate_results = {}
    negative_cosines = []
    for route in ROUTES:
        route_result = {}
        for dataset in DATASETS:
            metrics = {}
            for metric in METRICS:
                seed_values = [
                    float(
                        payload["results"][route][str(seed)]["datasets"][dataset]
                        ["summary"][metric]["mean"]
                    )
                    for seed in SEEDS
                ]
                metrics[metric] = _mean_sd(seed_values)
            metrics["rows_per_seed"] = 256
            route_result[dataset] = metrics
            negative_cosines.extend(
                {
                    "route": route,
                    "dataset": dataset,
                    "seed": seed,
                    "cosine": metrics["cosine"]["values"][index],
                }
                for index, seed in enumerate(SEEDS)
                if metrics["cosine"]["values"][index] < 0.0
            )
        aggregate_results[route] = route_result

    for route in ROUTES:
        for seed in SEEDS:
            row = payload["results"][route][str(seed)]
            checkpoint = row["checkpoint"]
            if checkpoint["model_state_sha256_before"] != checkpoint["model_state_sha256_after"]:
                raise ProbeReceiptError("checkpoint state changed during zero-update probe")
            reference_shas = row["confidence_probe"]["gradient_sha256_by_batch"]
            if len(reference_shas) != 8 or len(set(reference_shas)) != 8:
                raise ProbeReceiptError("confidence probe batch fingerprints drifted")
            for batch_index in range(8):
                observed = {
                    row["datasets"][dataset]["batches"][batch_index]
                    ["confidence_gradient_sha256"]
                    for dataset in DATASETS
                }
                if observed != {reference_shas[batch_index]}:
                    raise ProbeReceiptError("confidence gradient changed with rank dataset")

    for dataset in DATASETS:
        for seed in SEEDS:
            left = payload["results"][ROUTES[0]][str(seed)]["datasets"][dataset]["summary"]
            right = payload["results"][ROUTES[1]][str(seed)]["datasets"][dataset]["summary"]
            for metric in (
                "native_p1",
                "native_top1_runnerup_margin",
                "native_oracle_positive_negative_gap",
            ):
                if left[metric]["mean"] != right[metric]["mean"]:
                    raise ProbeReceiptError(f"native diagnostic changed across routes: {metric}")
    return {
        "aggregate": aggregate_results,
        "negative_cosine_cells": negative_cosines,
        "negative_cosine_cell_count": len(negative_cosines),
        "total_cosine_cells": len(ROUTES) * len(DATASETS) * len(SEEDS),
    }


def build(*, probe_result: Path, preregistration: Path, output: Path) -> dict[str, Any]:
    output = output.resolve()
    if output.exists():
        raise ProbeReceiptError(f"receipt output already exists: {output}")
    probe_result = probe_result.resolve(strict=True)
    preregistration = preregistration.resolve(strict=True)
    payload = json.loads(probe_result.read_text(encoding="utf-8"))
    prereg = json.loads(preregistration.read_text(encoding="utf-8"))
    if prereg.get("status") != "locked_before_selected_candidate_extraction_or_gradient_probe":
        raise ProbeReceiptError("preregistration status drifted")
    if prereg["output_absence"]["path"] != str(probe_result.relative_to(ROOT)):
        raise ProbeReceiptError("probe output differs from preregistered target")
    summary = aggregate(payload)
    cache_receipts = {
        dataset: _binding(
            ROOT
            / f"outputs/mmgdino_e5_cross_dataset_probe_20260821/caches/{dataset}_receipt.json"
        )
        for dataset in DATASETS
    }
    result = {
        "schema": SCHEMA,
        "status": "complete_zero_update_cross_dataset_probe",
        "provenance": {
            "preregistration": _binding(preregistration),
            "probe_result": _binding(probe_result),
            "selection_receipt": _binding(
                ROOT
                / "outputs/mmgdino_e5_cross_dataset_probe_20260821/selection/selection_receipt.json"
            ),
            "candidate_cache_receipts": cache_receipts,
        },
        "contract": payload["contract"],
        **summary,
        "paper_claims": {
            "rank_dataset_induces_consistent_negative_shared_gradient_alignment": False,
            "allowed": "Changing the rank probe among RefCOCO, RefCOCO+, and RefCOCOg changes rank difficulty and gradient magnitude, but does not reveal consistent negative rank/rejection alignment on the strong e5 shared owners.",
            "prohibited": "The strong e5 shared owner exhibits dataset-independent harmful rank/rejection gradient conflict."
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-result", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(build(
        probe_result=args.probe_result,
        preregistration=args.preregistration,
        output=args.output,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
