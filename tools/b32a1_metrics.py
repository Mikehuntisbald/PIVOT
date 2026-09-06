"""Formal metrics and paired image-cluster bootstrap for B32A1."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import stdev
from typing import Any, Mapping, Sequence

import numpy as np

from tools.b32a1_heads import COUPLED_SCALAR, ISOLATED, SHARED_WIDE


FORMAL_SEEDS = (17, 42, 73)
FORMAL_ARMS = (COUPLED_SCALAR, SHARED_WIDE, ISOLATED)
BOOTSTRAP_ITERATIONS = 5_000
BOOTSTRAP_SEED = 320_105_000
PRIMARY_METRICS = (
    "p_at_1_macro",
    "text_recall_at_1",
    "image_recall_at_1",
    "text_auroc",
    "image_auroc",
    "aurc",
)


class B32A1MetricError(ValueError):
    """Raised when records or a formal statistic violate the protocol."""


def binary_auroc(positive: Sequence[float], negative: Sequence[float]) -> float:
    """Tie-aware AUROC, with a tied positive/negative pair worth one half."""
    pos = np.asarray(positive, dtype=np.float64)
    neg = np.asarray(negative, dtype=np.float64)
    if pos.ndim != 1 or neg.ndim != 1 or not len(pos) or not len(neg):
        raise B32A1MetricError("AUROC requires nonempty one-dimensional classes")
    if not np.isfinite(pos).all() or not np.isfinite(neg).all():
        raise B32A1MetricError("AUROC scores must be finite")
    labels = np.concatenate(
        (np.ones(len(pos), dtype=np.float64), np.zeros(len(neg), dtype=np.float64))
    )
    scores = np.concatenate((pos, neg))
    weights = np.ones(len(scores), dtype=np.float64)
    return _weighted_auroc(scores, labels, weights)


def selective_localization_aurc(
    confidence: Sequence[float], localization_error: Sequence[bool | int]
) -> float:
    """Tie-grouped trapezoidal area under the positive selective-risk curve.

    Examples are retained from highest to lowest confidence.  Equal confidence
    values enter together.  Risk is cumulative end-to-end top-1 localization
    error (IoU < 0.5) among retained examples.  The risk at zero coverage is
    defined as the risk of the first nonempty tie group, and linear
    interpolation is used between group coverage breakpoints on [0, 1].
    """
    scores = np.asarray(confidence, dtype=np.float64)
    errors = np.asarray(localization_error, dtype=np.float64)
    if scores.ndim != 1 or errors.ndim != 1 or len(scores) != len(errors) or not len(scores):
        raise B32A1MetricError("AURC inputs must be nonempty aligned vectors")
    if not np.isfinite(scores).all() or not np.isin(errors, (0.0, 1.0)).all():
        raise B32A1MetricError("AURC requires finite scores and binary errors")
    return _weighted_aurc(scores, errors, np.ones(len(scores), dtype=np.float64))


def _group_starts(sorted_scores: np.ndarray) -> np.ndarray:
    return np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]


def _weighted_auroc(
    scores: np.ndarray, labels: np.ndarray, weights: np.ndarray
) -> float:
    if not (scores.shape == labels.shape == weights.shape) or scores.ndim != 1:
        raise B32A1MetricError("weighted AUROC arrays must be aligned vectors")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    sorted_weights = weights[order]
    starts = _group_starts(sorted_scores)
    group_positive = np.add.reduceat(sorted_weights * sorted_labels, starts)
    group_negative = np.add.reduceat(sorted_weights * (1.0 - sorted_labels), starts)
    positive_total = float(group_positive.sum())
    negative_total = float(group_negative.sum())
    if positive_total <= 0.0 or negative_total <= 0.0:
        raise B32A1MetricError("weighted AUROC draw lost a class")
    negative_before = np.cumsum(group_negative) - group_negative
    numerator = np.sum(
        group_positive * (negative_before + 0.5 * group_negative), dtype=np.float64
    )
    return float(numerator / (positive_total * negative_total))


def _weighted_aurc(
    scores: np.ndarray, errors: np.ndarray, weights: np.ndarray
) -> float:
    if not (scores.shape == errors.shape == weights.shape) or scores.ndim != 1:
        raise B32A1MetricError("weighted AURC arrays must be aligned vectors")
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_errors = errors[order]
    sorted_weights = weights[order]
    starts = _group_starts(sorted_scores)
    group_count = np.add.reduceat(sorted_weights, starts)
    group_error = np.add.reduceat(sorted_weights * sorted_errors, starts)
    total = float(group_count.sum())
    if total <= 0.0:
        raise B32A1MetricError("weighted AURC draw lost all positives")
    retained = np.cumsum(group_count)
    cumulative_error = np.cumsum(group_error)
    risk = cumulative_error / retained
    coverage = retained / total
    previous_coverage = np.r_[0.0, coverage[:-1]]
    previous_risk = np.r_[risk[0], risk[:-1]]
    return float(np.sum((coverage - previous_coverage) * (risk + previous_risk) / 2.0))


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise B32A1MetricError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise B32A1MetricError(f"{name} must be finite")
    return result


def _validate_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise B32A1MetricError("metric records must not be empty")
    sample_ids: set[str] = set()
    annotation_ids: set[int] = set()
    positive_ids: set[int] = set()
    for row in rows:
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id or sample_id in sample_ids:
            raise B32A1MetricError("sample IDs must be nonempty and unique")
        sample_ids.add(sample_id)
        kind = row.get("kind")
        if kind not in {"positive", "text", "image"}:
            raise B32A1MetricError(f"invalid FineCops kind {kind!r}")
        annotation_id = row.get("annotation_id")
        if isinstance(annotation_id, bool) or not isinstance(annotation_id, int):
            raise B32A1MetricError("annotation_id must be an integer")
        if annotation_id in annotation_ids:
            raise B32A1MetricError("annotation IDs must be unique")
        annotation_ids.add(annotation_id)
        cluster = row.get("cluster_image_id")
        if not isinstance(cluster, str) or not cluster:
            raise B32A1MetricError("cluster_image_id must be a nonempty string")
        _finite_float(row.get("confidence"), name="confidence")
        if kind == "positive":
            level = row.get("level")
            if level not in {1, 2, 3}:
                raise B32A1MetricError("positive level must be 1, 2, or 3")
            iou = _finite_float(row.get("top1_iou"), name="top1_iou")
            if not 0.0 <= iou <= 1.0:
                raise B32A1MetricError("top1_iou must lie in [0,1]")
            positive_ids.add(annotation_id)
        else:
            parent = row.get("parent_positive_id")
            if isinstance(parent, bool) or not isinstance(parent, int):
                raise B32A1MetricError("negative parent_positive_id must be an integer")
    for row in rows:
        if row["kind"] != "positive" and row["parent_positive_id"] not in positive_ids:
            raise B32A1MetricError("negative record has no positive parent")


def compute_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute the seven sealed B32A1 metric surfaces for one arm/seed."""
    _validate_rows(rows)
    positives = {
        int(row["annotation_id"]): row for row in rows if row["kind"] == "positive"
    }
    by_level: dict[int, list[bool]] = {1: [], 2: [], 3: []}
    for row in positives.values():
        by_level[int(row["level"])].append(float(row["top1_iou"]) >= 0.5)
    if any(not values for values in by_level.values()):
        raise B32A1MetricError("P@1 requires all three FineCops levels")
    p_by_level = {
        str(level): float(np.mean(values)) for level, values in by_level.items()
    }
    positive_scores = [float(row["confidence"]) for row in positives.values()]
    level1_scores = [
        float(row["confidence"])
        for row in positives.values()
        if int(row["level"]) == 1
    ]
    errors = [float(row["top1_iou"]) < 0.5 for row in positives.values()]
    result: dict[str, Any] = {
        "counts": {"positive": len(positives)},
        "p_at_1_by_level": p_by_level,
        "p_at_1_macro": float(np.mean(list(p_by_level.values()))),
        "p_at_1_micro": float(np.mean([value for values in by_level.values() for value in values])),
        "aurc": selective_localization_aurc(positive_scores, errors),
    }
    for kind in ("text", "image"):
        negatives = [row for row in rows if row["kind"] == kind]
        if not negatives:
            raise B32A1MetricError(f"metric records contain no {kind} negatives")
        wins = []
        for row in negatives:
            parent = positives[int(row["parent_positive_id"])]
            if str(row["cluster_image_id"]) != str(parent["cluster_image_id"]):
                raise B32A1MetricError("negative and positive parent use different clusters")
            wins.append(
                float(parent["top1_iou"]) >= 0.5
                and float(parent["confidence"]) > float(row["confidence"])
            )
        negative_scores = [float(row["confidence"]) for row in negatives]
        result["counts"][kind] = len(negatives)
        result[f"{kind}_recall_at_1"] = float(np.mean(wins))
        result[f"{kind}_auroc"] = binary_auroc(level1_scores, negative_scores)
    return result


@dataclass(frozen=True)
class _PreparedRun:
    clusters: np.ndarray
    positive_cluster: np.ndarray
    positive_level: np.ndarray
    positive_correct: np.ndarray
    positive_confidence: np.ndarray
    positive_error: np.ndarray
    rejection_cluster: Mapping[str, np.ndarray]
    rejection_success: Mapping[str, np.ndarray]
    auroc_scores: Mapping[str, np.ndarray]
    auroc_labels: Mapping[str, np.ndarray]
    auroc_cluster: Mapping[str, np.ndarray]


def _prepare_run(
    rows: Sequence[Mapping[str, Any]], cluster_index: Mapping[str, int]
) -> _PreparedRun:
    _validate_rows(rows)
    positives = {
        int(row["annotation_id"]): row for row in rows if row["kind"] == "positive"
    }
    positive_rows = list(positives.values())
    level1_rows = [row for row in positive_rows if int(row["level"]) == 1]
    rejection_cluster: dict[str, np.ndarray] = {}
    rejection_success: dict[str, np.ndarray] = {}
    auroc_scores: dict[str, np.ndarray] = {}
    auroc_labels: dict[str, np.ndarray] = {}
    auroc_cluster: dict[str, np.ndarray] = {}
    for kind in ("text", "image"):
        negative_rows = [row for row in rows if row["kind"] == kind]
        rejection_cluster[kind] = np.asarray(
            [cluster_index[str(row["cluster_image_id"])] for row in negative_rows],
            dtype=np.int64,
        )
        rejection_success[kind] = np.asarray(
            [
                float(positives[int(row["parent_positive_id"])]["top1_iou"]) >= 0.5
                and float(positives[int(row["parent_positive_id"])]["confidence"])
                > float(row["confidence"])
                for row in negative_rows
            ],
            dtype=np.float64,
        )
        auroc_scores[kind] = np.asarray(
            [float(row["confidence"]) for row in level1_rows]
            + [float(row["confidence"]) for row in negative_rows],
            dtype=np.float64,
        )
        auroc_labels[kind] = np.r_[
            np.ones(len(level1_rows), dtype=np.float64),
            np.zeros(len(negative_rows), dtype=np.float64),
        ]
        auroc_cluster[kind] = np.asarray(
            [cluster_index[str(row["cluster_image_id"])] for row in level1_rows]
            + [cluster_index[str(row["cluster_image_id"])] for row in negative_rows],
            dtype=np.int64,
        )
    return _PreparedRun(
        clusters=np.asarray(sorted(cluster_index.values()), dtype=np.int64),
        positive_cluster=np.asarray(
            [cluster_index[str(row["cluster_image_id"])] for row in positive_rows],
            dtype=np.int64,
        ),
        positive_level=np.asarray([int(row["level"]) for row in positive_rows], dtype=np.int64),
        positive_correct=np.asarray(
            [float(row["top1_iou"]) >= 0.5 for row in positive_rows], dtype=np.float64
        ),
        positive_confidence=np.asarray(
            [float(row["confidence"]) for row in positive_rows], dtype=np.float64
        ),
        positive_error=np.asarray(
            [float(row["top1_iou"]) < 0.5 for row in positive_rows], dtype=np.float64
        ),
        rejection_cluster=rejection_cluster,
        rejection_success=rejection_success,
        auroc_scores=auroc_scores,
        auroc_labels=auroc_labels,
        auroc_cluster=auroc_cluster,
    )


def _weighted_metrics(prepared: _PreparedRun, cluster_weights: np.ndarray) -> dict[str, float]:
    positive_weights = cluster_weights[prepared.positive_cluster]
    level_values: list[float] = []
    for level in (1, 2, 3):
        selected = prepared.positive_level == level
        denominator = float(positive_weights[selected].sum())
        if denominator <= 0.0:
            raise B32A1MetricError("bootstrap draw lost a FineCops level")
        level_values.append(
            float(np.sum(positive_weights[selected] * prepared.positive_correct[selected]) / denominator)
        )
    result = {
        "p_at_1_macro": float(np.mean(level_values)),
        "aurc": _weighted_aurc(
            prepared.positive_confidence, prepared.positive_error, positive_weights
        ),
    }
    for kind in ("text", "image"):
        rejection_weights = cluster_weights[prepared.rejection_cluster[kind]]
        denominator = float(rejection_weights.sum())
        if denominator <= 0.0:
            raise B32A1MetricError(f"bootstrap draw lost {kind} negatives")
        result[f"{kind}_recall_at_1"] = float(
            np.sum(rejection_weights * prepared.rejection_success[kind]) / denominator
        )
        auc_weights = cluster_weights[prepared.auroc_cluster[kind]]
        result[f"{kind}_auroc"] = _weighted_auroc(
            prepared.auroc_scores[kind], prepared.auroc_labels[kind], auc_weights
        )
    return result


def _record_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("sample_id"),
        row.get("annotation_id"),
        row.get("kind"),
        row.get("parent_positive_id"),
        row.get("cluster_image_id"),
        row.get("level"),
    )


def _holm(raw: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(raw, key=raw.get)
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for index, key in enumerate(ordered):
        candidate = min(1.0, (count - index) * raw[key])
        running = max(running, candidate)
        adjusted[key] = running
    return adjusted


def paired_image_cluster_bootstrap(
    runs: Mapping[str, Mapping[int, Sequence[Mapping[str, Any]]]],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Bootstrap mean-over-seed arm contrasts with shared image-cluster draws."""
    if set(runs) != set(FORMAL_ARMS):
        raise B32A1MetricError(f"runs must contain exactly {FORMAL_ARMS}")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        raise B32A1MetricError("iterations must be a positive integer")
    reference = list(runs[COUPLED_SCALAR][FORMAL_SEEDS[0]])
    _validate_rows(reference)
    reference_identity = [_record_identity(row) for row in reference]
    clusters = sorted(
        {str(row["cluster_image_id"]) for row in reference if row["kind"] == "positive"}
    )
    cluster_index = {value: index for index, value in enumerate(clusters)}
    prepared: dict[str, dict[int, _PreparedRun]] = {}
    for arm in FORMAL_ARMS:
        if set(runs[arm]) != set(FORMAL_SEEDS):
            raise B32A1MetricError(f"{arm} must contain exactly seeds {FORMAL_SEEDS}")
        prepared[arm] = {}
        for run_seed in FORMAL_SEEDS:
            rows = list(runs[arm][run_seed])
            if [_record_identity(row) for row in rows] != reference_identity:
                raise B32A1MetricError("paired arm/seed record identities drifted")
            prepared[arm][run_seed] = _prepare_run(rows, cluster_index)

    contrast_specs = {
        "isolated_vs_coupled_scalar": (ISOLATED, COUPLED_SCALAR),
        "isolated_vs_shared_wide": (ISOLATED, SHARED_WIDE),
        "shared_wide_vs_coupled_scalar": (SHARED_WIDE, COUPLED_SCALAR),
    }
    draws = {
        contrast: {metric: [] for metric in PRIMARY_METRICS}
        for contrast in contrast_specs
    }

    def arm_mean(arm: str, weights: np.ndarray) -> dict[str, float]:
        values = [
            _weighted_metrics(prepared[arm][run_seed], weights)
            for run_seed in FORMAL_SEEDS
        ]
        return {
            metric: float(np.mean([value[metric] for value in values]))
            for metric in PRIMARY_METRICS
        }

    unit = np.ones(len(clusters), dtype=np.float64)
    point_by_arm = {arm: arm_mean(arm, unit) for arm in FORMAL_ARMS}
    rng = np.random.Generator(np.random.PCG64(seed))
    for _ in range(iterations):
        sampled = rng.integers(0, len(clusters), size=len(clusters))
        weights = np.bincount(sampled, minlength=len(clusters)).astype(np.float64)
        boot_by_arm = {arm: arm_mean(arm, weights) for arm in FORMAL_ARMS}
        for contrast, (candidate, reference_arm) in contrast_specs.items():
            for metric in PRIMARY_METRICS:
                if metric == "aurc":
                    gain = (
                        boot_by_arm[reference_arm][metric]
                        - boot_by_arm[candidate][metric]
                    )
                else:
                    gain = (
                        boot_by_arm[candidate][metric]
                        - boot_by_arm[reference_arm][metric]
                    )
                draws[contrast][metric].append(gain)

    contrasts: dict[str, Any] = {}
    for contrast, (candidate, reference_arm) in contrast_specs.items():
        metric_rows: dict[str, Any] = {}
        raw_p: dict[str, float] = {}
        for metric in PRIMARY_METRICS:
            values = np.asarray(draws[contrast][metric], dtype=np.float64)
            point = (
                point_by_arm[reference_arm][metric] - point_by_arm[candidate][metric]
                if metric == "aurc"
                else point_by_arm[candidate][metric] - point_by_arm[reference_arm][metric]
            )
            lower_tail = (1.0 + float(np.sum(values <= 0.0))) / (iterations + 1.0)
            upper_tail = (1.0 + float(np.sum(values >= 0.0))) / (iterations + 1.0)
            raw_p[metric] = min(1.0, 2.0 * min(lower_tail, upper_tail))
            metric_rows[metric] = {
                "gain_favoring_candidate": float(point),
                "ci95_percentile": [
                    float(np.percentile(values, 2.5)),
                    float(np.percentile(values, 97.5)),
                ],
                "two_sided_p": raw_p[metric],
                "direction": "reference_minus_candidate" if metric == "aurc" else "candidate_minus_reference",
            }
        adjusted = _holm(raw_p)
        for metric in PRIMARY_METRICS:
            metric_rows[metric]["holm_adjusted_p_six_endpoints"] = adjusted[metric]
        contrasts[contrast] = {
            "candidate": candidate,
            "reference": reference_arm,
            "role": "secondary" if contrast.startswith("shared_wide") else "primary",
            "metrics": metric_rows,
        }

    per_arm = {}
    for arm in FORMAL_ARMS:
        by_seed = {
            str(run_seed): compute_metrics(runs[arm][run_seed])
            for run_seed in FORMAL_SEEDS
        }
        per_arm[arm] = {
            "by_seed": by_seed,
            "mean": {
                metric: float(np.mean([by_seed[str(value)][metric] for value in FORMAL_SEEDS]))
                for metric in PRIMARY_METRICS
            },
            "sample_sd": {
                metric: float(stdev([by_seed[str(value)][metric] for value in FORMAL_SEEDS]))
                for metric in PRIMARY_METRICS
            },
        }
    return {
        "schema": "pivot.b32a1.formal_metrics_bootstrap/v1",
        "iterations": iterations,
        "seed": seed,
        "cluster_unit": "parent_positive_visual_genome_image",
        "seed_aggregation": "unweighted_mean_across_17_42_73_within_each_draw",
        "tie_policy": {
            "recall_at_1": "strict_positive_greater_than_negative_ties_fail",
            "auroc": "half_credit",
            "aurc": "equal_confidence_examples_enter_as_one_group",
        },
        "per_arm": per_arm,
        "contrasts": contrasts,
    }


__all__ = [
    "BOOTSTRAP_ITERATIONS",
    "BOOTSTRAP_SEED",
    "B32A1MetricError",
    "FORMAL_ARMS",
    "FORMAL_SEEDS",
    "PRIMARY_METRICS",
    "binary_auroc",
    "compute_metrics",
    "paired_image_cluster_bootstrap",
    "selective_localization_aurc",
]
