#!/usr/bin/env python3
"""Aggregate the pretrained MM-GDINO-T Shared-Wide/Isolated replay."""

from __future__ import annotations

import json
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from tools.eval_mmgdino_e5_ownership_cache import (
    binary_auroc,
    exact_q05,
    positive_average_precision,
)
from tools.mmgdino_pretrain_ownership import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    B58_REFERENCE,
    E5_REFERENCE,
    FORMAL_SEEDS,
    OWNERS,
    REC_NONINFERIORITY_MARGIN,
    REF_INPUTS,
    TEST5_SURFACES,
    TESTAB_SURFACES,
    TRUNK_ID,
)


SCHEMA = "arrow.mmgdino_pretrain_ownership.aggregate/v1"
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
    root: Path, surface: str, route: str, seed: int | None
) -> Path:
    suffix = "native" if route == "native" else f"{route}/seed{seed}"
    return root / TRUNK_ID / surface / suffix / "records.jsonl"


def _keys() -> list[tuple[str, int | None]]:
    result = [("native", None)]
    for route in OWNERS:
        result.extend((route, seed) for seed in FORMAL_SEEDS)
    return result


def _load_ref(
    root: Path,
) -> tuple[
    dict[str, list[str]],
    dict[str, dict[tuple[str, int | None], np.ndarray]],
]:
    images_by_surface: dict[str, list[str]] = {}
    arrays: dict[str, dict[tuple[str, int | None], np.ndarray]] = {}
    for surface in TEST5_SURFACES:
        canonical_ids = None
        canonical_images = None
        arrays[surface] = {}
        for route, seed in _keys():
            rows = _read_jsonl(_record_path(root, surface, route, seed))
            ids = [str(row["sample_id"]) for row in rows]
            images = [str(row["image_id"]) for row in rows]
            if len(ids) != len(set(ids)):
                raise AggregateError(f"duplicate identities on {surface}")
            if canonical_ids is None:
                canonical_ids, canonical_images = ids, images
            elif ids != canonical_ids or images != canonical_images:
                raise AggregateError(f"record alignment drift on {surface}")
            arrays[surface][(route, seed)] = np.asarray(
                [bool(row["correct_iou50"]) for row in rows],
                dtype=np.float64,
            )
        assert canonical_images is not None
        if len(canonical_images) != REF_INPUTS[surface]["rows"]:
            raise AggregateError(f"row count drift on {surface}")
        images_by_surface[surface] = canonical_images
    return images_by_surface, arrays


def _load_tn(
    root: Path,
) -> tuple[
    list[str],
    dict[tuple[str, int | None], tuple[np.ndarray, np.ndarray]],
]:
    canonical_ids = None
    canonical_images = None
    arrays = {}
    for route, seed in _keys():
        rows = _read_jsonl(_record_path(root, "strict2031", route, seed))
        ids = [str(row["pair_id"]) for row in rows]
        images = [str(row["image_id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise AggregateError("duplicate Strict2031 pair identities")
        if canonical_ids is None:
            canonical_ids, canonical_images = ids, images
        elif ids != canonical_ids or images != canonical_images:
            raise AggregateError("Strict2031 record alignment drift")
        arrays[(route, seed)] = (
            np.asarray([row["positive_score"] for row in rows], dtype=np.float64),
            np.asarray([row["negative_score"] for row in rows], dtype=np.float64),
        )
    assert canonical_images is not None
    return canonical_images, arrays


def _mean_sd(values: Sequence[float]) -> dict[str, Any]:
    return {
        "mean": float(statistics.fmean(values)),
        "sample_sd": float(statistics.stdev(values)),
        "by_seed": {
            str(seed): float(value) for seed, value in zip(FORMAL_SEEDS, values)
        },
    }


def _exact_fpr(positive: np.ndarray, negative: np.ndarray) -> float:
    return float(np.mean(negative >= exact_q05(positive)))


def _point_metrics(
    refs: Mapping[str, Mapping[tuple[str, int | None], np.ndarray]],
    tn: Mapping[tuple[str, int | None], tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    result = {}
    for route in ("native", *OWNERS):
        seeds: Sequence[int | None] = (None,) if route == "native" else FORMAL_SEEDS
        split_values: dict[str, list[float]] = {
            surface: [] for surface in TEST5_SURFACES
        }
        test5_values, testab_values, fpr_values, auroc_values, aupr_values = (
            [], [], [], [], []
        )
        for seed in seeds:
            for surface in TEST5_SURFACES:
                split_values[surface].append(
                    float(refs[surface][(route, seed)].mean())
                )
            test5_values.append(
                float(
                    sum(refs[s][(route, seed)].sum() for s in TEST5_SURFACES)
                    / sum(len(refs[s][(route, seed)]) for s in TEST5_SURFACES)
                )
            )
            testab_values.append(
                float(
                    sum(refs[s][(route, seed)].sum() for s in TESTAB_SURFACES)
                    / sum(len(refs[s][(route, seed)]) for s in TESTAB_SURFACES)
                )
            )
            positive, negative = tn[(route, seed)]
            fpr_values.append(_exact_fpr(positive, negative))
            auroc_values.append(binary_auroc(positive, negative))
            aupr_values.append(positive_average_precision(positive, negative))
        if route == "native":
            result[route] = {
                "splits": {key: value[0] for key, value in split_values.items()},
                "test5_micro_p1": test5_values[0],
                "testab_micro_p1": testab_values[0],
                "strict2031_fpr95": fpr_values[0],
                "strict2031_auroc": auroc_values[0],
                "strict2031_aupr": aupr_values[0],
            }
        else:
            result[route] = {
                "splits": {
                    key: _mean_sd(value) for key, value in split_values.items()
                },
                "test5_micro_p1": _mean_sd(test5_values),
                "testab_micro_p1": _mean_sd(testab_values),
                "strict2031_fpr95": _mean_sd(fpr_values),
                "strict2031_auroc": _mean_sd(auroc_values),
                "strict2031_aupr": _mean_sd(aupr_values),
            }
    return result


def _image_accounting(
    images_by_surface: Mapping[str, Sequence[str]],
    refs: Mapping[str, Mapping[tuple[str, int | None], np.ndarray]],
) -> tuple[
    tuple[str, ...],
    dict[str, np.ndarray],
    dict[str, dict[tuple[str, int | None], np.ndarray]],
]:
    image_ids = tuple(
        sorted({image for images in images_by_surface.values() for image in images})
    )
    image_position = {image: index for index, image in enumerate(image_ids)}
    counts = {}
    hits = {}
    for surface in TEST5_SURFACES:
        positions = np.asarray(
            [image_position[image] for image in images_by_surface[surface]],
            dtype=np.int64,
        )
        counts[surface] = np.bincount(
            positions, minlength=len(image_ids)
        ).astype(np.float64)
        hits[surface] = {
            key: np.bincount(
                positions, weights=value, minlength=len(image_ids)
            ).astype(np.float64)
            for key, value in refs[surface].items()
        }
    return image_ids, counts, hits


def _draw_ref_metric(
    multiplicity: np.ndarray,
    surfaces: Sequence[str],
    route: str,
    seed: int | None,
    counts: Mapping[str, np.ndarray],
    hits: Mapping[str, Mapping[tuple[str, int | None], np.ndarray]],
) -> float:
    numerator = sum(float(hits[s][(route, seed)] @ multiplicity) for s in surfaces)
    denominator = sum(float(counts[s] @ multiplicity) for s in surfaces)
    if denominator <= 0:
        raise AggregateError("bootstrap draw omitted an evaluation surface")
    return numerator / denominator


def _ci(values: Sequence[float]) -> list[float]:
    return [float(value) for value in np.percentile(values, [2.5, 97.5])]


def _one_sided(values: Sequence[float], boundary: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float((1 + np.sum(array <= boundary)) / (len(array) + 1))


def _gradient_statistics(formal_root: Path) -> dict[str, Any]:
    result = {}
    for route in OWNERS:
        all_cosines = []
        u150_cosines = []
        per_seed = {}
        isolated_checks = []
        for seed in FORMAL_SEEDS:
            receipt_path = (
                formal_root
                / f"{TRUNK_ID}/{route}/seed{seed}/training_receipt.json"
            )
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            probes = receipt["gradient_probes"]
            seed_all = []
            for milestone in ("25", "50", "100", "150"):
                probe = probes[milestone]
                isolated_checks.append(bool(probe["structurally_isolated"]))
                seed_all.extend(float(value) for value in probe.get("cosines", []))
            seed_u150 = [
                float(value) for value in probes["150"].get("cosines", [])
            ]
            all_cosines.extend(seed_all)
            u150_cosines.extend(seed_u150)
            mean = probes["150"]["cosine_mean"]
            per_seed[str(seed)] = {
                "u150_mean": None if mean is None else float(mean),
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
                raise AggregateError("isolated cross-task path audit failed")
            result[route] = {
                "structurally_isolated": True,
                "all_cross_task_autograd_paths_absent": True,
                "per_seed": per_seed,
            }
            continue
        all_array = np.asarray(all_cosines, dtype=np.float64)
        u150_array = np.asarray(u150_cosines, dtype=np.float64)
        if all_array.size != 96 or u150_array.size != 24:
            raise AggregateError("shared gradient probe count drifted")

        def summary(value: np.ndarray) -> dict[str, Any]:
            return {
                "count": int(value.size),
                "mean": float(value.mean()),
                "p_negative": float(np.mean(value < 0.0)),
                "q05": float(np.quantile(value, 0.05)),
                "minimum": float(value.min()),
            }

        result[route] = {
            "structurally_isolated": False,
            "per_seed": per_seed,
            "all_milestones": summary(all_array),
            "u150": summary(u150_array),
        }
    return result


def aggregate(
    *, evaluation_root: Path, formal_root: Path, output: Path
) -> dict[str, Any]:
    if output.exists():
        raise AggregateError("aggregate output already exists")
    images_by_surface, refs = _load_ref(evaluation_root)
    tn_images, tn = _load_tn(evaluation_root)
    image_ids, counts, hits = _image_accounting(images_by_surface, refs)
    tn_groups: dict[str, list[int]] = {}
    for index, image in enumerate(tn_images):
        tn_groups.setdefault(image, []).append(index)
    ordered_tn_groups = [
        np.asarray(tn_groups[key], dtype=np.int64) for key in sorted(tn_groups)
    ]
    draws = {
        "test5_gain": [],
        "testab_gain": [],
        "fpr95_gain": [],
        "split_gain": {surface: [] for surface in TEST5_SURFACES},
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    for _ in range(BOOTSTRAP_REPLICATES):
        multiplicity = np.bincount(
            rng.integers(0, len(image_ids), size=len(image_ids)),
            minlength=len(image_ids),
        ).astype(np.float64)
        sampled_tn_groups = rng.integers(
            0, len(ordered_tn_groups), size=len(ordered_tn_groups)
        )
        tn_indices = np.concatenate(
            [ordered_tn_groups[index] for index in sampled_tn_groups]
        )
        seed_test5, seed_testab, seed_fpr = [], [], []
        seed_splits = {surface: [] for surface in TEST5_SURFACES}
        for seed in FORMAL_SEEDS:
            shared_test5 = _draw_ref_metric(
                multiplicity, TEST5_SURFACES, SHARED, seed, counts, hits
            )
            isolated_test5 = _draw_ref_metric(
                multiplicity, TEST5_SURFACES, ISOLATED, seed, counts, hits
            )
            shared_testab = _draw_ref_metric(
                multiplicity, TESTAB_SURFACES, SHARED, seed, counts, hits
            )
            isolated_testab = _draw_ref_metric(
                multiplicity, TESTAB_SURFACES, ISOLATED, seed, counts, hits
            )
            seed_test5.append(isolated_test5 - shared_test5)
            seed_testab.append(isolated_testab - shared_testab)
            for surface in TEST5_SURFACES:
                shared = _draw_ref_metric(
                    multiplicity, (surface,), SHARED, seed, counts, hits
                )
                isolated = _draw_ref_metric(
                    multiplicity, (surface,), ISOLATED, seed, counts, hits
                )
                seed_splits[surface].append(isolated - shared)
            shared_positive, shared_negative = tn[(SHARED, seed)]
            isolated_positive, isolated_negative = tn[(ISOLATED, seed)]
            seed_fpr.append(
                _exact_fpr(
                    shared_positive[tn_indices], shared_negative[tn_indices]
                )
                - _exact_fpr(
                    isolated_positive[tn_indices], isolated_negative[tn_indices]
                )
            )
        draws["test5_gain"].append(float(np.mean(seed_test5)))
        draws["testab_gain"].append(float(np.mean(seed_testab)))
        draws["fpr95_gain"].append(float(np.mean(seed_fpr)))
        for surface in TEST5_SURFACES:
            draws["split_gain"][surface].append(
                float(np.mean(seed_splits[surface]))
            )

    points = _point_metrics(refs, tn)
    shared, isolated = points[SHARED], points[ISOLATED]
    test5_gain = (
        isolated["test5_micro_p1"]["mean"]
        - shared["test5_micro_p1"]["mean"]
    )
    testab_gain = (
        isolated["testab_micro_p1"]["mean"]
        - shared["testab_micro_p1"]["mean"]
    )
    fpr95_gain = (
        shared["strict2031_fpr95"]["mean"]
        - isolated["strict2031_fpr95"]["mean"]
    )
    test5_ci = _ci(draws["test5_gain"])
    testab_ci = _ci(draws["testab_gain"])
    fpr_ci = _ci(draws["fpr95_gain"])
    rec_noninferiority_p = _one_sided(
        draws["test5_gain"], -REC_NONINFERIORITY_MARGIN
    )
    fpr_superiority_p = _one_sided(draws["fpr95_gain"], 0.0)
    contrast = {
        "candidate": ISOLATED,
        "reference": SHARED,
        "test5_gain": test5_gain,
        "test5_ci95": test5_ci,
        "test5_noninferiority_margin": REC_NONINFERIORITY_MARGIN,
        "test5_noninferiority_p": rec_noninferiority_p,
        "test5_noninferior": test5_ci[0] > -REC_NONINFERIORITY_MARGIN,
        "test5_superior": test5_ci[0] > 0.0,
        "testab_gain": testab_gain,
        "testab_ci95": testab_ci,
        "testab_noninferior": testab_ci[0] > -REC_NONINFERIORITY_MARGIN,
        "fpr95_gain": fpr95_gain,
        "fpr95_ci95": fpr_ci,
        "fpr95_superiority_p": fpr_superiority_p,
        "fpr95_superior": fpr_ci[0] > 0.0,
        "split_gains": {
            surface: {
                "gain": (
                    isolated["splits"][surface]["mean"]
                    - shared["splits"][surface]["mean"]
                ),
                "ci95": _ci(draws["split_gain"][surface]),
            }
            for surface in TEST5_SURFACES
        },
    }
    contrast["iut_p"] = max(rec_noninferiority_p, fpr_superiority_p)
    contrast["iut_passes"] = bool(
        contrast["test5_noninferior"]
        and contrast["testab_noninferior"]
        and contrast["fpr95_superior"]
        and contrast["iut_p"] < 0.05
    )
    gradients = _gradient_statistics(formal_root)
    payload = {
        "schema": SCHEMA,
        "status": "complete",
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "ref_cluster": "global image_id across all Test5 splits",
            "strict_cluster": "image_id carrying the complete pair",
            "same_draw_across_owners_and_seeds": True,
            "fpr95_recomputes_each_owner_seed_positive_q05_per_replicate": True,
        },
        "point_metrics": points,
        "isolated_minus_shared_wide": contrast,
        "gradient_probes": gradients,
        "references": {
            "e5": json.loads(E5_REFERENCE.read_text(encoding="utf-8"))[
                "primary"
            ]["point_metrics"],
            "b58": {
                "source": str(B58_REFERENCE),
                "sha256_bound_by_preregistration": True,
            },
        },
        "claim_gate": {
            "isolated_test5_noninferior": contrast["test5_noninferior"],
            "isolated_testab_noninferior": contrast["testab_noninferior"],
            "isolated_strict2031_fpr95_superior": contrast["fpr95_superior"],
            "joint_gate": contrast["iut_passes"],
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
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return payload


__all__ = [
    "AggregateError", "SCHEMA", "_ci", "_one_sided", "aggregate",
]
