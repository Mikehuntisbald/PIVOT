#!/usr/bin/env python3
"""Aggregate and bootstrap the capacity-controlled e5 ownership transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.eval_mmgdino_e5_ownership_cache import exact_q05
from tools.mmgdino_e5_ownership import (
    OWNERSHIP_ISOLATED_128,
    OWNERSHIP_SHARED_128,
    OWNERSHIP_SHARED_WIDE,
)
from tools.responsibility_isolation_cache import file_sha256
from tools.train_mmgdino_e5_ownership import FORMAL_SEEDS


SCHEMA = "arrow.mmgdino_e5_ownership.aggregate/v1"
BOOTSTRAP_SEED = 20260821
BOOTSTRAP_REPLICATES = 5000
REC_MARGIN = 0.005
ROUTES = (
    "native",
    OWNERSHIP_SHARED_128,
    OWNERSHIP_SHARED_WIDE,
    OWNERSHIP_ISOLATED_128,
)
CONTRASTS = (
    (OWNERSHIP_ISOLATED_128, OWNERSHIP_SHARED_128),
    (OWNERSHIP_ISOLATED_128, OWNERSHIP_SHARED_WIDE),
    (OWNERSHIP_ISOLATED_128, "native"),
)


class AggregateError(RuntimeError):
    pass


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.endswith("\n") or not raw.strip():
                raise AggregateError(f"malformed records at {path}:{line_number}")
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise AggregateError("record must be an object")
            rows.append(value)
    if not rows:
        raise AggregateError(f"records are empty: {path}")
    return rows


def _record_path(root: Path, surface: str, route: str, seed: int | None) -> Path:
    suffix = "native" if route == "native" else f"{route}/seed{seed}"
    return root / surface / suffix / "records.jsonl"


def _load_ref(root: Path, surface: str) -> tuple[list[str], list[str], dict[str, np.ndarray]]:
    paths = {}
    rows_by_key = {}
    for route in ROUTES:
        seeds = (None,) if route == "native" else FORMAL_SEEDS
        for seed in seeds:
            key = f"{route}:{seed}"
            path = _record_path(root, surface, route, seed)
            paths[key] = path
            rows_by_key[key] = _read_jsonl(path)
    reference = rows_by_key["native:None"]
    ids = [row["sample_id"] for row in reference]
    images = [str(row["image_id"]) for row in reference]
    if len(ids) != len(set(ids)):
        raise AggregateError(f"duplicate Ref identity on {surface}")
    arrays = {}
    for key, rows in rows_by_key.items():
        if [row["sample_id"] for row in rows] != ids:
            raise AggregateError(f"Ref record alignment drift: {surface}/{key}")
        if [str(row["image_id"]) for row in rows] != images:
            raise AggregateError(f"Ref image alignment drift: {surface}/{key}")
        arrays[key] = np.asarray(
            [bool(row["correct_iou50"]) for row in rows], dtype=np.float64
        )
    return ids, images, arrays


def _load_tn(root: Path) -> tuple[list[str], list[str], dict[str, tuple[np.ndarray, np.ndarray]]]:
    rows_by_key = {}
    for route in ROUTES:
        seeds = (None,) if route == "native" else FORMAL_SEEDS
        for seed in seeds:
            key = f"{route}:{seed}"
            rows_by_key[key] = _read_jsonl(
                _record_path(root, "strict2031", route, seed)
            )
    reference = rows_by_key["native:None"]
    ids = [row["pair_id"] for row in reference]
    images = [str(row["image_id"]) for row in reference]
    if len(ids) != len(set(ids)):
        raise AggregateError("duplicate Strict2031 pair identity")
    arrays = {}
    for key, rows in rows_by_key.items():
        if [row["pair_id"] for row in rows] != ids:
            raise AggregateError(f"Strict2031 record alignment drift: {key}")
        if [str(row["image_id"]) for row in rows] != images:
            raise AggregateError(f"Strict2031 image alignment drift: {key}")
        arrays[key] = (
            np.asarray([row["positive_score"] for row in rows], dtype=np.float64),
            np.asarray([row["negative_score"] for row in rows], dtype=np.float64),
        )
    return ids, images, arrays


def _groups(images: Sequence[str]) -> list[np.ndarray]:
    mapping: dict[str, list[int]] = {}
    for index, image in enumerate(images):
        mapping.setdefault(image, []).append(index)
    return [np.asarray(mapping[key], dtype=np.int64) for key in sorted(mapping)]


def _draw_indices(groups: Sequence[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    draw = rng.integers(0, len(groups), size=len(groups))
    return np.concatenate([groups[index] for index in draw])


def _route_mean_ref(arrays: Mapping[str, np.ndarray], route: str, indices: np.ndarray) -> float:
    if route == "native":
        return float(arrays["native:None"][indices].mean())
    return float(
        np.mean(
            [arrays[f"{route}:{seed}"][indices].mean() for seed in FORMAL_SEEDS]
        )
    )


def _fpr(positive: np.ndarray, negative: np.ndarray, indices: np.ndarray) -> float:
    pos = positive[indices]
    neg = negative[indices]
    threshold = exact_q05(pos)
    return float(np.mean(neg >= threshold))


def _route_mean_fpr(
    arrays: Mapping[str, tuple[np.ndarray, np.ndarray]],
    route: str,
    indices: np.ndarray,
) -> float:
    if route == "native":
        positive, negative = arrays["native:None"]
        return _fpr(positive, negative, indices)
    return float(
        np.mean(
            [
                _fpr(*arrays[f"{route}:{seed}"], indices)
                for seed in FORMAL_SEEDS
            ]
        )
    )


def _percentile(values: Sequence[float]) -> list[float]:
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def _one_sided(values: Sequence[float], boundary: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float((1 + np.sum(array <= boundary)) / (len(array) + 1))


def _holm(raw: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=raw.get)
    adjusted = {}
    running = 0.0
    total = len(ordered)
    for index, name in enumerate(ordered):
        value = min(1.0, (total - index) * raw[name])
        running = max(running, value)
        adjusted[name] = running
    return adjusted


def aggregate(*, evaluation_root: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise AggregateError("aggregate output already exists")
    _, images_a, ref_a = _load_ref(evaluation_root, "refcoco_testA")
    _, images_b, ref_b = _load_ref(evaluation_root, "refcoco_testB")
    _, tn_images, tn = _load_tn(evaluation_root)
    groups_a, groups_b, groups_tn = _groups(images_a), _groups(images_b), _groups(tn_images)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws: dict[str, dict[str, list[float]]] = {
        f"{candidate}-{reference}": {
            "rec_gain": [],
            "testA_gain": [],
            "testB_gain": [],
            "fpr_gain": [],
        }
        for candidate, reference in CONTRASTS
    }
    for _ in range(BOOTSTRAP_REPLICATES):
        ia = _draw_indices(groups_a, rng)
        ib = _draw_indices(groups_b, rng)
        itn = _draw_indices(groups_tn, rng)
        for candidate, reference in CONTRASTS:
            key = f"{candidate}-{reference}"
            candidate_a = _route_mean_ref(ref_a, candidate, ia)
            reference_a = _route_mean_ref(ref_a, reference, ia)
            candidate_b = _route_mean_ref(ref_b, candidate, ib)
            reference_b = _route_mean_ref(ref_b, reference, ib)
            candidate_correct = candidate_a * len(ia) + candidate_b * len(ib)
            reference_correct = reference_a * len(ia) + reference_b * len(ib)
            draws[key]["rec_gain"].append(
                (candidate_correct - reference_correct) / (len(ia) + len(ib))
            )
            draws[key]["testA_gain"].append(candidate_a - reference_a)
            draws[key]["testB_gain"].append(candidate_b - reference_b)
            draws[key]["fpr_gain"].append(
                _route_mean_fpr(tn, reference, itn)
                - _route_mean_fpr(tn, candidate, itn)
            )
    full_a = np.arange(len(images_a))
    full_b = np.arange(len(images_b))
    full_tn = np.arange(len(tn_images))
    point = {}
    for route in ROUTES:
        a = _route_mean_ref(ref_a, route, full_a)
        b = _route_mean_ref(ref_b, route, full_b)
        pooled = (a * len(full_a) + b * len(full_b)) / (
            len(full_a) + len(full_b)
        )
        point[route] = {
            "testA_p1": a,
            "testB_p1": b,
            "testAB_micro_p1": pooled,
            "strict2031_fpr95": _route_mean_fpr(tn, route, full_tn),
        }
    contrasts = {}
    raw_iut = {}
    for candidate, reference in CONTRASTS:
        key = f"{candidate}-{reference}"
        rec_values = draws[key]["rec_gain"]
        fpr_values = draws[key]["fpr_gain"]
        rec_p = _one_sided(rec_values, -REC_MARGIN)
        fpr_p = _one_sided(fpr_values, 0.0)
        iut = max(rec_p, fpr_p)
        raw_iut[key] = iut
        point_rec = point[candidate]["testAB_micro_p1"] - point[reference]["testAB_micro_p1"]
        point_fpr = point[reference]["strict2031_fpr95"] - point[candidate]["strict2031_fpr95"]
        contrasts[key] = {
            "candidate": candidate,
            "reference": reference,
            "rec_gain": point_rec,
            "rec_ci95": _percentile(rec_values),
            "rec_noninferiority_margin": REC_MARGIN,
            "rec_noninferiority_p": rec_p,
            "testA_gain": point[candidate]["testA_p1"] - point[reference]["testA_p1"],
            "testA_ci95": _percentile(draws[key]["testA_gain"]),
            "testB_gain": point[candidate]["testB_p1"] - point[reference]["testB_p1"],
            "testB_ci95": _percentile(draws[key]["testB_gain"]),
            "fpr95_gain": point_fpr,
            "fpr95_ci95": _percentile(fpr_values),
            "fpr95_superiority_p": fpr_p,
            "iut_p": iut,
            "rec_noninferior": _percentile(rec_values)[0] > -REC_MARGIN,
            "testA_no_collapse": _percentile(draws[key]["testA_gain"])[0] > -REC_MARGIN,
            "testB_no_collapse": _percentile(draws[key]["testB_gain"])[0] > -REC_MARGIN,
            "fpr95_superior": _percentile(fpr_values)[0] > 0.0,
        }
    holm = _holm(raw_iut)
    for key, value in contrasts.items():
        value["holm_iut_p"] = holm[key]
        value["passes"] = bool(
            value["rec_noninferior"]
            and value["testA_no_collapse"]
            and value["testB_no_collapse"]
            and value["fpr95_superior"]
            and holm[key] < 0.05
        )
    gradients = {}
    for route in (OWNERSHIP_SHARED_128, OWNERSHIP_SHARED_WIDE, OWNERSHIP_ISOLATED_128):
        gradients[route] = {}
        for seed in FORMAL_SEEDS:
            receipt_path = (
                ROOT
                / f"outputs/mmgdino_e5_ownership_transfer_20260821/formal/{route}/seed{seed}/training_receipt.json"
            )
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            gradients[route][str(seed)] = receipt["gradient_probes"]["150"]
    shared_all_negative = all(
        gradients[route][str(seed)]["cosine_mean"] < 0.0
        for route in (OWNERSHIP_SHARED_128, OWNERSHIP_SHARED_WIDE)
        for seed in FORMAL_SEEDS
    )
    full_claim = bool(
        all(value["passes"] for value in contrasts.values())
        and shared_all_negative
    )
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "evaluation_root": str(evaluation_root.resolve(strict=True)),
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "cluster": "image_id; TestA/TestB sampled as separate strata",
            "fpr95_threshold_recomputed_from_each_replicate_positive_q05": True,
        },
        "point_metrics": point,
        "contrasts": contrasts,
        "gradient_u150": gradients,
        "claim_gate": {
            "shared_all_seed_cosines_negative": shared_all_negative,
            "all_three_iut_contrasts_pass": all(
                value["passes"] for value in contrasts.values()
            ),
            "full_conflict_and_endpoint_claim_supported": full_claim,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    result = aggregate(
        evaluation_root=args.evaluation_root.resolve(strict=True),
        output=args.output.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
