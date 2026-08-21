#!/usr/bin/env python3
"""Bind the strong-e5 ownership matrix, statistics, and allowed paper claims."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval_mmgdino_e5_ownership_cache import binary_metrics
from tools.mmgdino_e5_ownership import OWNERSHIP_MODES
from tools.responsibility_isolation_cache import file_sha256
from tools.train_mmgdino_e5_ownership import FORMAL_SEEDS


SCHEMA = "arrow.mmgdino_e5_ownership.final_receipt/v1"
EXPERIMENT = ROOT / "outputs/mmgdino_e5_ownership_transfer_20260821"
ROUTES = ("native",) + OWNERSHIP_MODES


class FinalReceiptError(RuntimeError):
    pass


def _record(path: Path, rows: int | None = None) -> dict[str, Any]:
    path = path.resolve(strict=True)
    value = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }
    if rows is not None:
        with path.open("r", encoding="utf-8") as handle:
            actual = sum(1 for _ in handle)
        if actual != rows:
            raise FinalReceiptError(
                f"row count drift: expected {rows}, got {actual}: {path}"
            )
        value["rows"] = actual
    return value


def _summary(surface: str, route: str, seed: int | None) -> tuple[dict, Path]:
    suffix = "native" if route == "native" else f"{route}/seed{seed}"
    path = EXPERIMENT / f"evaluation/{surface}/{suffix}/summary.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "complete":
        raise FinalReceiptError(f"incomplete evaluation summary: {path}")
    records = Path(value["records"]["path"])
    if value["records"]["sha256"] != file_sha256(records):
        raise FinalReceiptError(f"record SHA drift: {path}")
    return value, path


def _mean_sd(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "sample_sd": None, "by_seed": None}
    return {
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
        "by_seed": {
            str(seed): value for seed, value in zip(FORMAL_SEEDS, values)
        } if len(values) == len(FORMAL_SEEDS) else None,
    }


def _strict1607_ids() -> tuple[set[str], dict[str, Any]]:
    path = (
        ROOT
        / "outputs/u2v5_leakage_clean_anchor_20260817/final_once/"
        "strict1607_u50/tn_eval_inputs/tn_refcocop_val_refcocog_umd_val.jsonl"
    )
    ids = set()
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            ids.add(f"eval:{json.loads(raw)['sample_id']}")
    if len(ids) != 1607:
        raise FinalReceiptError("strict1607 identities drifted")
    return ids, _record(path, 1607)


def _derived_strict1607(route: str, seed: int | None, ids: set[str]) -> dict[str, float]:
    summary, _ = _summary("strict2031", route, seed)
    records = Path(summary["records"]["path"])
    positive = []
    negative = []
    with records.open("r", encoding="utf-8") as handle:
        for raw in handle:
            row = json.loads(raw)
            if row["pair_id"] in ids:
                positive.append(float(row["positive_score"]))
                negative.append(float(row["negative_score"]))
    if len(positive) != 1607:
        raise FinalReceiptError("strict1607 is not an exact strict2031 record subset")
    return binary_metrics(np.asarray(positive), np.asarray(negative))


def build(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FinalReceiptError("final receipt already exists")
    aggregate_path = EXPERIMENT / "aggregate.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if aggregate.get("status") != "complete":
        raise FinalReceiptError("ownership aggregate is incomplete")
    training_preregistration_path = (
        ROOT / "paper/data/mmgdino_e5_ownership_preregistration.json"
    )
    training_preregistration = json.loads(
        training_preregistration_path.read_text(encoding="utf-8")
    )
    training_bindings = {}
    gradient_trajectories = {}
    model_accounting = {"native": {
        "trainable_parameters": 0,
        "macs_per_query_both_outputs": 0,
        "rank_representation_dim": 0,
        "confidence_representation_dim": 0,
    }}
    for route in OWNERSHIP_MODES:
        route_bindings = {}
        route_gradients = {}
        canonical_architecture = None
        for seed in FORMAL_SEEDS:
            receipt_path = EXPERIMENT / f"formal/{route}/seed{seed}/training_receipt.json"
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if receipt.get("status") != "complete":
                raise FinalReceiptError(f"incomplete training receipt: {receipt_path}")
            if receipt["optimizers"].get("weight_decay") != 0.0:
                raise FinalReceiptError(f"nonzero primary weight decay: {receipt_path}")
            if not receipt["optimizers"].get("task_specific_states"):
                raise FinalReceiptError(f"missing task-specific optimizer states: {receipt_path}")
            if receipt["updates"] != {
                "amp_skips": 0,
                "confidence": 50,
                "nonfinite": 0,
                "rank": 100,
                "total": 150,
            }:
                raise FinalReceiptError(f"formal update contract drift: {receipt_path}")
            architecture = receipt["architecture"]
            if canonical_architecture is None:
                canonical_architecture = architecture
            elif architecture != canonical_architecture:
                raise FinalReceiptError(f"cross-seed architecture drift: {route}")
            route_bindings[str(seed)] = {
                "training_receipt": _record(receipt_path),
                "checkpoint": _record(Path(receipt["checkpoint"]["path"])),
                "rank_optimizer_state_tensors": receipt["optimizers"]["rank_state_tensor_count"],
                "confidence_optimizer_state_tensors": receipt["optimizers"]["confidence_state_tensor_count"],
            }
            route_gradients[str(seed)] = receipt["gradient_probes"]
        assert canonical_architecture is not None
        model_accounting[route] = canonical_architecture
        training_bindings[route] = route_bindings
        gradient_trajectories[route] = route_gradients
    localization = {}
    for route in ROUTES:
        seeds = (None,) if route == "native" else FORMAL_SEEDS
        per_seed = {}
        for seed in seeds:
            surface_metrics = {}
            total_correct = 0
            total_rows = 0
            for surface, short_name in (
                ("refcoco_testA", "testA"),
                ("refcoco_testB", "testB"),
            ):
                summary, _ = _summary(surface, route, seed)
                metrics = summary["metrics"]
                surface_metrics[short_name] = float(metrics["p1_iou50"])
                total_correct += int(metrics["correct"])
                total_rows += int(metrics["rows"])
            surface_metrics["testAB_micro"] = total_correct / total_rows
            per_seed["native" if seed is None else str(seed)] = surface_metrics
        localization[route] = {
            metric: _mean_sd([float(row[metric]) for row in per_seed.values()])
            for metric in ("testA", "testB", "testAB_micro")
        }
    strict_secondary = {}
    evaluation_bindings = {}
    for route in ROUTES:
        seeds = (None,) if route == "native" else FORMAL_SEEDS
        strict_values = []
        bindings = {}
        for seed in seeds:
            summary, path = _summary("strict2031", route, seed)
            strict_values.append(summary["metrics"])
            bindings["native" if seed is None else str(seed)] = {
                "summary": _record(path),
                "records": _record(Path(summary["records"]["path"]), 2031),
            }
        evaluation_bindings[route] = bindings
        strict_secondary[route] = {}
        for metric in ("auroc", "aupr", "fpr95", "fixed_tpr", "fixed_fpr"):
            values = [float(value[metric]) for value in strict_values if value[metric] is not None]
            strict_secondary[route][metric] = _mean_sd(values)
    strict1607_ids, strict1607_manifest = _strict1607_ids()
    strict1607 = {}
    for route in ROUTES:
        seeds = (None,) if route == "native" else FORMAL_SEEDS
        metrics = [
            _derived_strict1607(route, seed, strict1607_ids) for seed in seeds
        ]
        strict1607[route] = {
            key: _mean_sd([float(value[key]) for value in metrics])
            for key in ("auroc", "aupr", "fpr95")
        }
    val = {}
    for route in ROUTES:
        seeds = (None,) if route == "native" else FORMAL_SEEDS
        values = []
        bindings = {}
        for seed in seeds:
            summary, path = _summary("refcoco_val", route, seed)
            values.append(float(summary["metrics"]["p1_iou50"]))
            bindings["native" if seed is None else str(seed)] = _record(path)
        val[route] = {"p1": _mean_sd(values), "summaries": bindings}
    native_gain = aggregate["contrasts"]["isolated_128-native"]
    shared128 = aggregate["contrasts"]["isolated_128-shared_128"]
    sharedwide = aggregate["contrasts"]["isolated_128-shared_wide"]
    payload = {
        "schema": SCHEMA,
        "status": "complete_all_preregistered_trajectories_and_surfaces",
        "provenance": {
            "training_preregistration": _record(
                training_preregistration_path
            ),
            "evaluation_preregistration": _record(
                ROOT
                / "paper/data/mmgdino_e5_ownership_evaluation_preregistration.json"
            ),
            "runtime_amendments": [
                _record(ROOT / "paper/data/mmgdino_e5_ownership_runtime_amendment.json"),
                _record(ROOT / "paper/data/mmgdino_e5_ownership_order_statistic_amendment.json"),
                _record(ROOT / "paper/data/mmgdino_e5_ownership_strict_path_amendment.json"),
                _record(ROOT / "paper/data/mmgdino_e5_ownership_val_schema_amendment.json"),
                _record(ROOT / "paper/data/mmgdino_e5_ownership_val_runtime_amendment.json"),
                _record(ROOT / "paper/data/mmgdino_e5_ownership_val_runner_amendment.json"),
            ],
            "aggregate": _record(aggregate_path),
            "strict1607_manifest": strict1607_manifest,
        },
        "strong_trunk": {
            "local_epoch5_replay": training_preregistration[
                "frozen_candidate_generator"
            ],
            "official_mmdetection_model_zoo_reference_p1": {
                "refcoco_val": 0.895,
                "refcoco_testA": 0.914,
                "refcoco_testB": 0.866,
                "status": "published_reference_not_a_local_forward",
            },
            "evidence_status": training_preregistration["evidence_status"],
        },
        "primary": {
            "point_metrics": aggregate["point_metrics"],
            "contrasts": aggregate["contrasts"],
            "claim_gate": aggregate["claim_gate"],
            "bootstrap": aggregate["bootstrap"],
        },
        "model_accounting": model_accounting,
        "training_bindings": training_bindings,
        "gradient_trajectories": gradient_trajectories,
        "secondary": {
            "refcoco_localization": localization,
            "strict2031": strict_secondary,
            "strict1607_derived_from_same_forward": strict1607,
            "refcoco_val_mechanism_only": val,
        },
        "gradient_u150": aggregate["gradient_u150"],
        "evaluation_bindings": evaluation_bindings,
        "paper_claims": {
            "isolated_vs_native_rejection_gain_with_rec_noninferiority": bool(
                native_gain["passes"]
            ),
            "isolated_superior_to_shared128": bool(shared128["passes"]),
            "isolated_superior_to_capacity_matched_sharedwide": bool(
                sharedwide["passes"]
            ),
            "shared_gradient_conflict_consistent_across_arms_and_seeds": bool(
                aggregate["claim_gate"]["shared_all_seed_cosines_negative"]
            ),
            "allowed_summary": (
                "Adding a learned absolute rejector to the strong frozen e5 "
                "candidate representation improves Strict2031 FPR95 without "
                "degrading RefCOCO. Responsibility isolation is sufficient "
                "for route preservation, but it is not superior to the "
                "capacity-matched Shared-Wide control, and shared gradient "
                "cosines are not consistently negative."
            ),
            "prohibited_summary": (
                "Responsibility isolation universally outperforms shared "
                "optimization on strong query representations."
            ),
        },
        "failed_attempts": [
            {
                "path": str(EXPERIMENT / "failed_attempts/shared_128_seed17_missing_cublas_before_u1"),
                "optimizer_updates": 0,
            },
            {
                "path": str(EXPERIMENT / "failed_attempts/shared_128_seed17_kthvalue_after_u1"),
                "rank_updates": 1,
                "confidence_updates": 0,
            },
            {
                "surface": "refcoco_val_mechanism_only",
                "reason": "training-only oracle precondition in eval cache validator",
                "result_rows_written": 0,
                "amendment": "paper/data/mmgdino_e5_ownership_val_schema_amendment.json",
            },
            {
                "surface": "refcoco_val_mechanism_only",
                "reason": "PyTorch weights_only default before model forward",
                "result_rows_written": 0,
                "amendment": "paper/data/mmgdino_e5_ownership_val_runtime_amendment.json",
            },
            {
                "surface": "refcoco_val_mechanism_only",
                "reason": "runner did not recognize sealed Val-only code hashes",
                "result_rows_written": 0,
                "amendment": "paper/data/mmgdino_e5_ownership_val_runner_amendment.json",
            },
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    value = build(args.output.resolve())
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
