#!/usr/bin/env python3
"""Aggregate the paired MM-GDINO e6 Shared-Wide/Isolated 2x2."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tools.eval_mmgdino_e5_ownership_cache import exact_q05
from tools.mmgdino_e6_ownership_2x2 import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    E5_REFERENCE,
    FORMAL_SEEDS,
    OWNERS,
    REC_NONINFERIORITY_MARGIN,
    ROOT,
    TRUNK_SPECS,
)


SCHEMA = "arrow.mmgdino_e6_ownership_2x2.aggregate/v1"
SHARED, ISOLATED = OWNERS


class AggregateError(RuntimeError):
    pass


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.endswith("\n") or not raw.strip():
                raise AggregateError(f"malformed record at {path}:{line_number}")
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise AggregateError("record must be a JSON object")
            rows.append(value)
    if not rows:
        raise AggregateError(f"records are empty: {path}")
    return rows


def _record_path(
    root: Path, trunk: str, surface: str, route: str, seed: int | None
) -> Path:
    suffix = "native" if route == "native" else f"{route}/seed{seed}"
    return root / trunk / surface / suffix / "records.jsonl"


def _keys() -> list[tuple[str, str, int | None]]:
    result = []
    for trunk in TRUNK_SPECS:
        result.append((trunk, "native", None))
        for route in OWNERS:
            result.extend((trunk, route, seed) for seed in FORMAL_SEEDS)
    return result


def _load_ref(
    root: Path, surface: str
) -> tuple[list[str], list[str], dict[tuple[str, str, int | None], np.ndarray]]:
    arrays = {}
    canonical_ids = None
    canonical_images = None
    for trunk, route, seed in _keys():
        rows = _read_jsonl(_record_path(root, trunk, surface, route, seed))
        ids = [str(row["sample_id"]) for row in rows]
        images = [str(row["image_id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise AggregateError(f"duplicate Ref identities on {surface}")
        if canonical_ids is None:
            canonical_ids, canonical_images = ids, images
        elif ids != canonical_ids or images != canonical_images:
            raise AggregateError(
                f"Ref alignment drift on {surface}/{trunk}/{route}/{seed}"
            )
        arrays[(trunk, route, seed)] = np.asarray(
            [bool(row["correct_iou50"]) for row in rows], dtype=np.float64
        )
    assert canonical_ids is not None and canonical_images is not None
    return canonical_ids, canonical_images, arrays


def _load_tn(
    root: Path,
) -> tuple[
    list[str], list[str],
    dict[tuple[str, str, int | None], tuple[np.ndarray, np.ndarray]],
]:
    arrays = {}
    canonical_ids = None
    canonical_images = None
    for trunk, route, seed in _keys():
        rows = _read_jsonl(_record_path(root, trunk, "strict2031", route, seed))
        ids = [str(row["pair_id"]) for row in rows]
        images = [str(row["image_id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise AggregateError("duplicate Strict2031 identities")
        if canonical_ids is None:
            canonical_ids, canonical_images = ids, images
        elif ids != canonical_ids or images != canonical_images:
            raise AggregateError(
                f"Strict alignment drift for {trunk}/{route}/{seed}"
            )
        arrays[(trunk, route, seed)] = (
            np.asarray([row["positive_score"] for row in rows], dtype=np.float64),
            np.asarray([row["negative_score"] for row in rows], dtype=np.float64),
        )
    assert canonical_ids is not None and canonical_images is not None
    return canonical_ids, canonical_images, arrays


def _groups(images: Sequence[str]) -> list[np.ndarray]:
    mapping: dict[str, list[int]] = {}
    for index, image in enumerate(images):
        mapping.setdefault(image, []).append(index)
    return [np.asarray(mapping[key], dtype=np.int64) for key in sorted(mapping)]


def _draw(groups: Sequence[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    selected = rng.integers(0, len(groups), size=len(groups))
    return np.concatenate([groups[index] for index in selected])


def _ref_mean(
    arrays: Mapping[tuple[str, str, int | None], np.ndarray],
    trunk: str,
    route: str,
    indices: np.ndarray,
) -> float:
    if route == "native":
        return float(arrays[(trunk, route, None)][indices].mean())
    return float(np.mean([
        arrays[(trunk, route, seed)][indices].mean() for seed in FORMAL_SEEDS
    ]))


def _fpr(positive: np.ndarray, negative: np.ndarray, indices: np.ndarray) -> float:
    threshold = exact_q05(positive[indices])
    return float(np.mean(negative[indices] >= threshold))


def _fpr_mean(
    arrays: Mapping[
        tuple[str, str, int | None], tuple[np.ndarray, np.ndarray]
    ],
    trunk: str,
    route: str,
    indices: np.ndarray,
) -> float:
    if route == "native":
        return _fpr(*arrays[(trunk, route, None)], indices)
    return float(np.mean([
        _fpr(*arrays[(trunk, route, seed)], indices)
        for seed in FORMAL_SEEDS
    ]))


def _ci(values: Sequence[float]) -> list[float]:
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def _one_sided(values: Sequence[float], boundary: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float((1 + np.sum(array <= boundary)) / (len(array) + 1))


def _holm(raw: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=raw.get)
    result = {}
    running = 0.0
    total = len(ordered)
    for index, key in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * raw[key]))
        result[key] = running
    return result


def _mean_sd(values: Sequence[float]) -> dict[str, Any]:
    return {
        "mean": float(statistics.fmean(values)),
        "sample_sd": float(statistics.stdev(values)),
        "by_seed": {
            str(seed): float(value) for seed, value in zip(FORMAL_SEEDS, values)
        },
    }


def _point_metrics(
    ref_a: Mapping[tuple[str, str, int | None], np.ndarray],
    ref_b: Mapping[tuple[str, str, int | None], np.ndarray],
    tn: Mapping[tuple[str, str, int | None], tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    ia = np.arange(len(next(iter(ref_a.values()))))
    ib = np.arange(len(next(iter(ref_b.values()))))
    itn = np.arange(len(next(iter(tn.values()))[0]))
    result = {}
    for trunk in TRUNK_SPECS:
        result[trunk] = {}
        for route in ("native", *OWNERS):
            seeds: Sequence[int | None] = (None,) if route == "native" else FORMAL_SEEDS
            test_a, test_b, pooled, fpr = [], [], [], []
            for seed in seeds:
                a = float(ref_a[(trunk, route, seed)][ia].mean())
                b = float(ref_b[(trunk, route, seed)][ib].mean())
                test_a.append(a); test_b.append(b)
                pooled.append((a * len(ia) + b * len(ib)) / (len(ia) + len(ib)))
                fpr.append(_fpr(*tn[(trunk, route, seed)], itn))
            if route == "native":
                result[trunk][route] = {
                    "testA_p1": test_a[0], "testB_p1": test_b[0],
                    "testAB_micro_p1": pooled[0], "strict2031_fpr95": fpr[0],
                }
            else:
                result[trunk][route] = {
                    "testA_p1": _mean_sd(test_a),
                    "testB_p1": _mean_sd(test_b),
                    "testAB_micro_p1": _mean_sd(pooled),
                    "strict2031_fpr95": _mean_sd(fpr),
                }
    return result


def _gradient_statistics(formal_root: Path) -> dict[str, Any]:
    result = {}
    for trunk in TRUNK_SPECS:
        result[trunk] = {}
        for route in OWNERS:
            all_cosines = []
            u150_cosines = []
            per_seed = {}
            isolated_checks = []
            for seed in FORMAL_SEEDS:
                receipt_path = formal_root / f"{trunk}/{route}/seed{seed}/training_receipt.json"
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                probes = receipt["gradient_probes"]
                seed_all = []
                for milestone in ("25", "50", "100", "150"):
                    probe = probes[milestone]
                    isolated_checks.append(bool(probe["structurally_isolated"]))
                    seed_all.extend(float(value) for value in probe.get("cosines", []))
                seed_u150 = [float(value) for value in probes["150"].get("cosines", [])]
                all_cosines.extend(seed_all)
                u150_cosines.extend(seed_u150)
                per_seed[str(seed)] = {
                    "u150_mean": float(probes["150"]["cosine_mean"]),
                    "u150_cosines": seed_u150,
                    "u150_p_negative": (
                        float(np.mean(np.asarray(seed_u150) < 0.0))
                        if seed_u150 else None
                    ),
                    "all_milestone_p_negative": (
                        float(np.mean(np.asarray(seed_all) < 0.0))
                        if seed_all else None
                    ),
                }
            if route == ISOLATED:
                if not isolated_checks or not all(isolated_checks):
                    raise AggregateError(f"{trunk} isolated cross-path audit failed")
                result[trunk][route] = {
                    "structurally_isolated": True,
                    "all_cross_task_autograd_paths_absent": True,
                    "per_seed": per_seed,
                }
                continue
            all_array = np.asarray(all_cosines, dtype=np.float64)
            u150_array = np.asarray(u150_cosines, dtype=np.float64)
            if all_array.size != 96 or u150_array.size != 24:
                raise AggregateError(f"{trunk} shared gradient probe count drifted")
            result[trunk][route] = {
                "structurally_isolated": False,
                "per_seed": per_seed,
                "all_milestones": {
                    "count": int(all_array.size),
                    "mean": float(all_array.mean()),
                    "p_negative": float(np.mean(all_array < 0.0)),
                    "q05": float(np.quantile(all_array, 0.05)),
                    "minimum": float(all_array.min()),
                },
                "u150": {
                    "count": int(u150_array.size),
                    "mean": float(u150_array.mean()),
                    "p_negative": float(np.mean(u150_array < 0.0)),
                    "q05": float(np.quantile(u150_array, 0.05)),
                    "minimum": float(u150_array.min()),
                },
            }
    pos = result["e6_posctrl"][SHARED]
    tn = result["e6_tn10"][SHARED]
    result["cross_trunk_tail_shift"] = {
        "all_milestones_p_negative_gain": (
            tn["all_milestones"]["p_negative"]
            - pos["all_milestones"]["p_negative"]
        ),
        "u150_p_negative_gain": tn["u150"]["p_negative"] - pos["u150"]["p_negative"],
        "all_milestones_q05_shift": tn["all_milestones"]["q05"] - pos["all_milestones"]["q05"],
        "status": "paired fixed-probe descriptive",
    }
    return result


def aggregate(
    *, evaluation_root: Path, formal_root: Path, output: Path
) -> dict[str, Any]:
    if output.exists():
        raise AggregateError("aggregate output already exists")
    _, images_a, ref_a = _load_ref(evaluation_root, "refcoco_testA")
    _, images_b, ref_b = _load_ref(evaluation_root, "refcoco_testB")
    _, images_tn, tn = _load_tn(evaluation_root)
    groups_a, groups_b, groups_tn = _groups(images_a), _groups(images_b), _groups(images_tn)
    draws = {
        trunk: {"rec_gain": [], "testA_gain": [], "testB_gain": [], "fpr_gain": []}
        for trunk in TRUNK_SPECS
    }
    did = {"rec": [], "fpr": []}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for _ in range(BOOTSTRAP_REPLICATES):
        ia, ib, itn = _draw(groups_a, rng), _draw(groups_b, rng), _draw(groups_tn, rng)
        replicate = {}
        for trunk in TRUNK_SPECS:
            iso_a = _ref_mean(ref_a, trunk, ISOLATED, ia)
            shared_a = _ref_mean(ref_a, trunk, SHARED, ia)
            iso_b = _ref_mean(ref_b, trunk, ISOLATED, ib)
            shared_b = _ref_mean(ref_b, trunk, SHARED, ib)
            rec_gain = (
                (iso_a - shared_a) * len(ia)
                + (iso_b - shared_b) * len(ib)
            ) / (len(ia) + len(ib))
            fpr_gain = (
                _fpr_mean(tn, trunk, SHARED, itn)
                - _fpr_mean(tn, trunk, ISOLATED, itn)
            )
            draws[trunk]["rec_gain"].append(rec_gain)
            draws[trunk]["testA_gain"].append(iso_a - shared_a)
            draws[trunk]["testB_gain"].append(iso_b - shared_b)
            draws[trunk]["fpr_gain"].append(fpr_gain)
            replicate[trunk] = (rec_gain, fpr_gain)
        did["rec"].append(replicate["e6_tn10"][0] - replicate["e6_posctrl"][0])
        did["fpr"].append(replicate["e6_tn10"][1] - replicate["e6_posctrl"][1])

    points = _point_metrics(ref_a, ref_b, tn)
    contrasts = {}
    raw_iut = {}
    for trunk in TRUNK_SPECS:
        shared = points[trunk][SHARED]
        isolated = points[trunk][ISOLATED]
        rec_gain = (
            isolated["testAB_micro_p1"]["mean"]
            - shared["testAB_micro_p1"]["mean"]
        )
        fpr_gain = (
            shared["strict2031_fpr95"]["mean"]
            - isolated["strict2031_fpr95"]["mean"]
        )
        rec_ci = _ci(draws[trunk]["rec_gain"])
        fpr_ci = _ci(draws[trunk]["fpr_gain"])
        rec_noninferiority_p = _one_sided(
            draws[trunk]["rec_gain"], -REC_NONINFERIORITY_MARGIN
        )
        fpr_superiority_p = _one_sided(draws[trunk]["fpr_gain"], 0.0)
        iut = max(rec_noninferiority_p, fpr_superiority_p)
        raw_iut[trunk] = iut
        contrasts[trunk] = {
            "candidate": ISOLATED, "reference": SHARED,
            "rec_gain": rec_gain, "rec_ci95": rec_ci,
            "rec_superiority_p": _one_sided(draws[trunk]["rec_gain"], 0.0),
            "rec_superior": rec_ci[0] > 0.0,
            "rec_noninferiority_margin": REC_NONINFERIORITY_MARGIN,
            "rec_noninferiority_p": rec_noninferiority_p,
            "rec_noninferior": rec_ci[0] > -REC_NONINFERIORITY_MARGIN,
            "testA_gain": isolated["testA_p1"]["mean"] - shared["testA_p1"]["mean"],
            "testA_ci95": _ci(draws[trunk]["testA_gain"]),
            "testB_gain": isolated["testB_p1"]["mean"] - shared["testB_p1"]["mean"],
            "testB_ci95": _ci(draws[trunk]["testB_gain"]),
            "fpr95_gain": fpr_gain, "fpr95_ci95": fpr_ci,
            "fpr95_superiority_p": fpr_superiority_p,
            "fpr95_superior": fpr_ci[0] > 0.0,
            "iut_p": iut,
        }
    holm = _holm(raw_iut)
    for trunk in contrasts:
        contrasts[trunk]["holm_iut_p"] = holm[trunk]
        contrasts[trunk]["iut_passes"] = bool(
            contrasts[trunk]["rec_noninferior"]
            and contrasts[trunk]["fpr95_superior"]
            and holm[trunk] < 0.05
        )

    gradients = _gradient_statistics(formal_root)
    e5_reference = json.loads(E5_REFERENCE.read_text(encoding="utf-8"))
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "cluster": "image_id; TestA/TestB sampled as separate strata",
            "same_draw_across_trunks_owners_and_seeds": True,
            "fpr95_recomputes_each_model_seed_positive_q05_per_replicate": True,
        },
        "point_metrics": points,
        "within_trunk_contrasts": contrasts,
        "cross_trunk_difference_in_differences": {
            "rec_isolation_gap_gain_tn10_minus_posctrl": (
                contrasts["e6_tn10"]["rec_gain"]
                - contrasts["e6_posctrl"]["rec_gain"]
            ),
            "rec_ci95": _ci(did["rec"]),
            "rec_superiority_p": _one_sided(did["rec"], 0.0),
            "fpr95_isolation_gap_gain_tn10_minus_posctrl": (
                contrasts["e6_tn10"]["fpr95_gain"]
                - contrasts["e6_posctrl"]["fpr95_gain"]
            ),
            "fpr95_ci95": _ci(did["fpr"]),
            "fpr95_superiority_p": _one_sided(did["fpr"], 0.0),
        },
        "gradient_probes": gradients,
        "e5_reference": {
            "point_metrics": e5_reference["primary"]["point_metrics"],
            "gradient_u150": e5_reference["gradient_u150"],
            "source": str(E5_REFERENCE),
        },
        "claim_gate": {
            "posctrl_near_zero_shared_gradient": (
                abs(gradients["e6_posctrl"][SHARED]["u150"]["mean"]) < 0.05
            ),
            "tn10_negative_tail_probability_increased": (
                gradients["cross_trunk_tail_shift"]["all_milestones_p_negative_gain"] > 0.0
            ),
            "tn10_isolated_rec_superior": contrasts["e6_tn10"]["rec_superior"],
            "complete_requested_pattern": bool(
                abs(gradients["e6_posctrl"][SHARED]["u150"]["mean"]) < 0.05
                and gradients["cross_trunk_tail_shift"]["all_milestones_p_negative_gain"] > 0.0
                and contrasts["e6_tn10"]["rec_superior"]
            ),
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
            handle.flush(); os.fsync(handle.fileno())
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
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    print(json.dumps(aggregate(
        evaluation_root=args.evaluation_root.resolve(strict=True),
        formal_root=args.formal_root.resolve(strict=True),
        output=args.output.resolve(),
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
