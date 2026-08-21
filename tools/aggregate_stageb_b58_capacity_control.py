#!/usr/bin/env python3
"""Aggregate the preregistered B58 Shared-Wide versus Isolated replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.compare_stageb_fpr95_records import exact_fpr95


SEEDS = (17, 42, 73)
SPLITS = (
    "refcoco_testA",
    "refcoco_testB",
    "refcocop_testA",
    "refcocop_testB",
    "refcocog_test",
)


class CapacityAggregationError(RuntimeError):
    pass


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_name(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError as error:
        raise CapacityAggregationError(f"source is outside repository: {path}") from error


def _rows(path: Path) -> list[dict[str, Any]]:
    result = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line.strip():
                row = json.loads(line)
                if row.get("valid") is not True:
                    raise CapacityAggregationError(f"{path}:{line_no} invalid row")
                result.append(row)
    if not result:
        raise CapacityAggregationError(f"{path} is empty")
    return result


def _ref_path(root: Path, arm: str, seed: int, split: str) -> Path:
    return root / arm / "test5/per_example_records" / (
        f"seed{seed}_checkpoint_iter__{split}.records.jsonl"
    )


def _tn_path(root: Path, arm: str, seed: int) -> Path:
    return root / arm / "strict2031/per_example_records" / (
        f"seed{seed}_checkpoint_iter__tn_global.records.jsonl"
    )


def _summary(values: np.ndarray, observed: float, *, margin: float = 0.0) -> dict[str, Any]:
    low, high = np.quantile(values, [0.025, 0.975])
    return {
        "observed_gain": float(observed),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "one_sided_p_gain_le_margin": float((1 + np.count_nonzero(values <= margin)) / (values.size + 1)),
        "margin": float(margin),
    }


def _binary_metrics(positive: np.ndarray, negative: np.ndarray) -> dict[str, float]:
    # Pairwise definition with half credit for ties; exact and dependency-free.
    order = np.argsort(np.concatenate([positive, negative]), kind="mergesort")
    scores = np.concatenate([positive, negative])[order]
    labels = np.concatenate([
        np.ones(positive.size, dtype=np.int8),
        np.zeros(negative.size, dtype=np.int8),
    ])[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and scores[end] == scores[start]:
            end += 1
        ranks[start:end] = 0.5 * (start + 1 + end)
        start = end
    positive_rank_sum = float(ranks[labels == 1].sum())
    auroc = (
        positive_rank_sum - positive.size * (positive.size + 1) / 2
    ) / (positive.size * negative.size)
    descending = np.argsort(-np.concatenate([positive, negative]), kind="mergesort")
    descending_labels = np.concatenate([
        np.ones(positive.size, dtype=np.int8),
        np.zeros(negative.size, dtype=np.int8),
    ])[descending]
    precision = np.cumsum(descending_labels) / np.arange(1, descending_labels.size + 1)
    aupr = float(precision[descending_labels == 1].sum() / positive.size)
    return {"auroc": float(auroc), "positive_aupr": aupr}


def aggregate(root: Path, *, iterations: int, seed: int) -> dict[str, Any]:
    if iterations != 5000:
        raise CapacityAggregationError("formal aggregation requires 5000 replicates")
    sources: dict[str, Any] = {}
    formal_root = root.parent / "formal"
    training_audit: dict[str, Any] = {}
    for arm in ("shared_wide", "isolated"):
        per_seed = {}
        capacities = []
        for train_seed in SEEDS:
            path = (formal_root / arm / f"seed{train_seed}/ownership_receipt.json").resolve(strict=True)
            receipt = json.loads(path.read_text(encoding="utf-8"))
            runtime = receipt["runtime_audit"]
            if not (
                runtime["successful_optimizer_steps"] == 150
                and runtime["amp_skipped_optimizer_steps"] == 0
                and runtime["nonfinite_gradient_boundaries"] == 0
                and receipt["frozen_tensor_sha256"]
                == receipt["initializer_frozen_tensor_sha256"]
                and receipt["optimizer_ownership"]["task_specific_states"] is True
                and receipt["optimizer_ownership"]["weight_decay"] == 0.0
            ):
                raise CapacityAggregationError(f"{arm} seed{train_seed} training audit failed")
            capacity = receipt["parameter_accounting"]["capacity_control"]
            capacities.append(capacity)
            per_seed[str(train_seed)] = {
                "gradient_audit": receipt["gradient_audit"],
                "minimum_device_free_bytes": runtime["minimum_device_free_bytes"],
                "peak_reserved_bytes": runtime["peak_reserved_bytes"],
            }
            sources[_source_name(path)] = {"sha256": _sha(path)}
        if any(value != capacities[0] for value in capacities[1:]):
            raise CapacityAggregationError(f"{arm} capacity accounting differs by seed")
        training_audit[arm] = {"capacity": capacities[0], "per_seed": per_seed}

    refs: dict[str, dict[int, dict[str, list[dict[str, Any]]]]] = {
        arm: {train_seed: {} for train_seed in SEEDS}
        for arm in ("shared_wide", "isolated")
    }
    canonical_images: set[int] = set()
    for split in SPLITS:
        reference_ids = None
        reference_images = None
        for arm in refs:
            for train_seed in SEEDS:
                path = _ref_path(root, arm, train_seed, split).resolve(strict=True)
                rows = _rows(path)
                identities = tuple(str(row["sample_id"]) for row in rows)
                images = tuple(int(row["image_id"]) for row in rows)
                if reference_ids is None:
                    reference_ids, reference_images = identities, images
                if identities != reference_ids or images != reference_images:
                    raise CapacityAggregationError(f"{split} record alignment drift")
                refs[arm][train_seed][split] = rows
                sources[_source_name(path)] = {"sha256": _sha(path), "rows": len(rows)}
        canonical_images.update(reference_images or ())
    image_ids = tuple(sorted(canonical_images))
    image_position = {value: index for index, value in enumerate(image_ids)}
    counts = {}
    hits = {arm: {train_seed: {} for train_seed in SEEDS} for arm in refs}
    for split in SPLITS:
        rows = refs["shared_wide"][17][split]
        positions = np.asarray([image_position[int(row["image_id"])] for row in rows])
        counts[split] = np.bincount(positions, minlength=len(image_ids)).astype(np.float64)
        for arm in refs:
            for train_seed in SEEDS:
                values = np.asarray(
                    [bool(row["correct50"]) for row in refs[arm][train_seed][split]],
                    dtype=np.float64,
                )
                hits[arm][train_seed][split] = np.bincount(
                    positions, weights=values, minlength=len(image_ids)
                )
    rng = np.random.default_rng(seed)
    ref_micro = np.empty(iterations, dtype=np.float64)
    ref_split = {split: np.empty(iterations, dtype=np.float64) for split in SPLITS}
    for iteration in range(iterations):
        multiplicity = np.bincount(
            rng.integers(0, len(image_ids), size=len(image_ids)),
            minlength=len(image_ids),
        ).astype(np.float64)
        denominators = {split: float(counts[split] @ multiplicity) for split in SPLITS}
        if any(value == 0 for value in denominators.values()):
            raise CapacityAggregationError("global image draw omitted a Ref split")
        seed_micro = []
        seed_split = {split: [] for split in SPLITS}
        for train_seed in SEEDS:
            arm_micro = {}
            for arm in refs:
                numerator = sum(
                    float(hits[arm][train_seed][split] @ multiplicity)
                    for split in SPLITS
                )
                denominator = sum(denominators.values())
                arm_micro[arm] = numerator / denominator
            seed_micro.append(arm_micro["isolated"] - arm_micro["shared_wide"])
            for split in SPLITS:
                isolated = float(hits["isolated"][train_seed][split] @ multiplicity) / denominators[split]
                shared = float(hits["shared_wide"][train_seed][split] @ multiplicity) / denominators[split]
                seed_split[split].append(isolated - shared)
        ref_micro[iteration] = float(np.mean(seed_micro))
        for split in SPLITS:
            ref_split[split][iteration] = float(np.mean(seed_split[split]))

    ref_points: dict[str, Any] = {"per_seed": {}}
    micro_gains = []
    split_gains = defaultdict(list)
    for train_seed in SEEDS:
        per_arm = {}
        for arm in refs:
            numerator = sum(
                sum(bool(row["correct50"]) for row in refs[arm][train_seed][split])
                for split in SPLITS
            )
            denominator = sum(len(refs[arm][train_seed][split]) for split in SPLITS)
            per_arm[arm] = numerator / denominator
        micro_gain = per_arm["isolated"] - per_arm["shared_wide"]
        micro_gains.append(micro_gain)
        per_split = {}
        for split in SPLITS:
            isolated = np.mean([bool(row["correct50"]) for row in refs["isolated"][train_seed][split]])
            shared = np.mean([bool(row["correct50"]) for row in refs["shared_wide"][train_seed][split]])
            per_split[split] = {"shared_wide": float(shared), "isolated": float(isolated), "gain": float(isolated - shared)}
            split_gains[split].append(float(isolated - shared))
        ref_points["per_seed"][str(train_seed)] = {
            "micro": {**per_arm, "gain": float(micro_gain)},
            "splits": per_split,
        }
    ref_bootstrap = {
        "micro": _summary(ref_micro, float(np.mean(micro_gains)), margin=-0.005),
        "splits": {
            split: _summary(ref_split[split], float(np.mean(split_gains[split])), margin=-0.005)
            for split in SPLITS
        },
        "image_clusters": len(image_ids),
    }

    tn: dict[str, dict[int, list[dict[str, Any]]]] = {
        arm: {} for arm in ("shared_wide", "isolated")
    }
    tn_reference_ids = None
    tn_images = None
    for arm in tn:
        for train_seed in SEEDS:
            path = _tn_path(root, arm, train_seed).resolve(strict=True)
            rows = _rows(path)
            identities = tuple(str(row["sample_id"]) for row in rows)
            images = np.asarray([int(row["image_id"]) for row in rows], dtype=np.int64)
            if tn_reference_ids is None:
                tn_reference_ids, tn_images = identities, images
            if identities != tn_reference_ids or not np.array_equal(images, tn_images):
                raise CapacityAggregationError("Strict2031 record alignment drift")
            tn[arm][train_seed] = rows
            sources[_source_name(path)] = {"sha256": _sha(path), "rows": len(rows)}
    assert tn_images is not None
    tn_image_ids = tuple(sorted(set(int(value) for value in tn_images.tolist())))
    tn_image_position = {value: index for index, value in enumerate(tn_image_ids)}
    record_cluster = np.asarray([tn_image_position[int(value)] for value in tn_images])
    tn_scores = {
        arm: {
            train_seed: (
                np.asarray([float(row["pos_score"]) for row in rows]),
                np.asarray([float(row["neg_score"]) for row in rows]),
            )
            for train_seed, rows in values.items()
        }
        for arm, values in tn.items()
    }
    tn_gain = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        multiplicity = np.bincount(
            rng.integers(0, len(tn_image_ids), size=len(tn_image_ids)),
            minlength=len(tn_image_ids),
        )
        sampled = np.repeat(np.arange(tn_images.size), multiplicity[record_cluster])
        seed_gains = []
        for train_seed in SEEDS:
            operating = {}
            for arm in tn:
                positive, negative = tn_scores[arm][train_seed]
                operating[arm] = exact_fpr95(positive[sampled], negative[sampled])["fpr"]
            seed_gains.append(operating["shared_wide"] - operating["isolated"])
        tn_gain[iteration] = float(np.mean(seed_gains))
    tn_points = {"per_seed": {}}
    tn_gains = []
    for train_seed in SEEDS:
        values = {}
        for arm in tn:
            positive, negative = tn_scores[arm][train_seed]
            values[arm] = {
                "fpr95": float(exact_fpr95(positive, negative)["fpr"]),
                **_binary_metrics(positive, negative),
            }
        gain = values["shared_wide"]["fpr95"] - values["isolated"]["fpr95"]
        tn_gains.append(gain)
        tn_points["per_seed"][str(train_seed)] = {**values, "fpr95_gain": float(gain)}
    tn_bootstrap = {
        "fpr95_gain_shared_minus_isolated": _summary(
            tn_gain, float(np.mean(tn_gains)), margin=0.0
        ),
        "image_clusters": len(tn_image_ids),
        "recomputes_each_arm_seed_positive_q05_per_replicate": True,
    }
    ref_pass = ref_bootstrap["micro"]["ci95_low"] > -0.005 and all(
        value["ci95_low"] > -0.005 for value in ref_bootstrap["splits"].values()
    )
    tn_pass = tn_bootstrap["fpr95_gain_shared_minus_isolated"]["ci95_low"] > 0.0
    return {
        "schema": "arrow.stageb.b58_capacity_control.results/v1",
        "comparison": "isolated_minus_shared_wide",
        "bootstrap": {
            "iterations": iterations,
            "rng": "PCG64",
            "seed": seed,
            "unit": "global image_id cluster",
            "same_draw_across_arms_and_three_seeds": True,
            "training_seeds_resampled": False,
        },
        "training_audit": training_audit,
        "ref_test5": {"points": ref_points, "bootstrap": ref_bootstrap},
        "strict2031": {"points": tn_points, "bootstrap": tn_bootstrap},
        "decision_gate": {
            "ref_micro_and_all_split_noninferiority_margin_0p005": ref_pass,
            "strict2031_fpr95_superiority": tn_pass,
            "joint_pass": bool(ref_pass and tn_pass),
        },
        "sources": sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-root",
        default="outputs/b58_capacity_control_20260821/evaluation_v2",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()
    result = aggregate(
        (ROOT / args.evaluation_root).resolve(),
        iterations=args.iterations,
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "decision_gate": result["decision_gate"]}, indent=2))


if __name__ == "__main__":
    main()
