"""Offline paired analysis of existence and localization-confidence ordering.

This module consumes canonical, already-produced records.  It imports no model
code and never selects checkpoints or fits a deployment threshold.  Bootstrap
intervals describe image-sampling uncertainty conditional on the supplied fixed
checkpoints, not uncertainty over future training seeds.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA = "arrow.grounding_confidence_ordering/v1"
KINDS = ("positive", "text", "image", "no_target")
SCORES = ("baseline", "candidate")


class OrderingContractError(ValueError):
    """The canonical record or paired-resampling contract was violated."""


def _validate_runs(runs: Mapping[str, Sequence[Mapping[str, Any]]]):
    if not isinstance(runs, Mapping) or not runs:
        raise OrderingContractError("runs must be a nonempty seed-to-records mapping")
    if any(not isinstance(key, str) or not key for key in runs):
        raise OrderingContractError("seed keys must be nonempty strings")
    aligned: dict[str, list[Mapping[str, Any]]] = {}
    reference = None
    for seed in sorted(runs):
        rows = runs[seed]
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
            raise OrderingContractError(f"seed {seed}: records must be a nonempty sequence")
        by_id = {}
        for row in rows:
            if not isinstance(row, Mapping):
                raise OrderingContractError("each record must be a mapping")
            for name in ("sample_id", "cluster_id", "stratum"):
                if not isinstance(row.get(name), str) or not row[name]:
                    raise OrderingContractError(f"{name} must be a nonempty string")
            sample_id = row["sample_id"]
            if sample_id in by_id:
                raise OrderingContractError(f"duplicate sample_id: {sample_id}")
            kind = row.get("kind")
            if kind not in KINDS:
                raise OrderingContractError(f"invalid kind: {kind!r}")
            if "correct" not in row or (
                kind == "positive" and not isinstance(row["correct"], bool)
            ) or (kind != "positive" and row["correct"] is not None):
                raise OrderingContractError("positive correct must be bool; negative correct must be null")
            if "level" not in row or (
                row["level"] is not None
                and (isinstance(row["level"], bool) or not isinstance(row["level"], int) or row["level"] < 1)
            ):
                raise OrderingContractError("level must be a positive integer or null")
            for name in ("baseline_score", "candidate_score"):
                value = row.get(name)
                if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                    raise OrderingContractError(f"{name} must be finite numeric")
            by_id[sample_id] = row
        ordered = [by_id[key] for key in sorted(by_id)]
        identity = [
            tuple(row[name] for name in ("sample_id", "cluster_id", "stratum", "kind", "level", "correct"))
            for row in ordered
        ]
        if reference is not None and identity != reference:
            raise OrderingContractError("sample identities, labels, clusters or strata drift across seeds")
        reference = identity
        aligned[seed] = ordered
    rows = next(iter(aligned.values()))
    cluster_strata: dict[str, str] = {}
    for row in rows:
        cluster, stratum = row["cluster_id"], row["stratum"]
        if cluster in cluster_strata and cluster_strata[cluster] != stratum:
            raise OrderingContractError("one image cluster cannot belong to multiple bootstrap strata")
        cluster_strata[cluster] = stratum
    clusters = sorted(cluster_strata)
    lookup = {cluster: index for index, cluster in enumerate(clusters)}
    strata: dict[str, list[int]] = defaultdict(list)
    for cluster in clusters:
        strata[cluster_strata[cluster]].append(lookup[cluster])
    return aligned, lookup, {key: np.asarray(strata[key], dtype=np.int64) for key in sorted(strata)}


def _group_starts(sorted_scores: np.ndarray) -> np.ndarray:
    return np.r_[0, np.flatnonzero(sorted_scores[1:] != sorted_scores[:-1]) + 1]


@dataclass(frozen=True)
class _GroupedMetric:
    kind: str
    clusters: np.ndarray
    labels: np.ndarray
    starts: np.ndarray

    @classmethod
    def prepare(cls, kind: str, scores: np.ndarray, labels: np.ndarray, clusters: np.ndarray):
        if kind == "mean" or not scores.size:
            return cls(kind, clusters, labels, np.empty(0, dtype=np.int64))
        order = np.argsort(-scores if kind == "aurc" else scores, kind="stable")
        return cls(kind, clusters[order], labels[order], _group_starts(scores[order]))

    def evaluate(self, cluster_weights: np.ndarray) -> tuple[float | None, str | None]:
        if not self.clusters.size:
            return None, "no_examples"
        weights = cluster_weights[self.clusters]
        if self.kind == "mean":
            total = float(weights.sum())
            if total <= 0:
                return None, "no_positive_weight"
            return float(np.dot(weights, self.labels) / total), None
        total_by_group = np.add.reduceat(weights, self.starts)
        label_by_group = np.add.reduceat(weights * self.labels, self.starts)
        if self.kind == "auc":
            negatives = total_by_group - label_by_group
            positive_total, negative_total = float(label_by_group.sum()), float(negatives.sum())
            if positive_total <= 0 or negative_total <= 0:
                return None, "requires_both_classes_with_positive_weight"
            numerator = np.sum(label_by_group * (np.cumsum(negatives) - negatives + 0.5 * negatives))
            return float(numerator / (positive_total * negative_total)), None
        # A sampled-out top confidence group must not create an artificial 0/0.
        active = total_by_group > 0
        group_count, group_errors = total_by_group[active], label_by_group[active]
        if not group_count.size:
            return None, "no_positive_weight"
        retained = np.cumsum(group_count)
        risk = np.cumsum(group_errors) / retained
        coverage = retained / retained[-1]
        return float(np.sum(np.diff(np.r_[0.0, coverage]) * (risk + np.r_[risk[0], risk[:-1]]) / 2.0)), None


def _prepare_surface(rows, cluster_lookup, *, l1: bool, negative_kinds):
    positive = np.asarray([row["kind"] == "positive" and (not l1 or row["level"] == 1) for row in rows])
    kinds = np.asarray([row["kind"] for row in rows])
    clusters = np.asarray([cluster_lookup[row["cluster_id"]] for row in rows], dtype=np.int64)
    correct = np.asarray([bool(row["correct"]) for row in rows], dtype=np.float64)
    result = {}
    for score_name in SCORES:
        scores = np.asarray([row[f"{score_name}_score"] for row in rows], dtype=np.float64)
        metrics = {}
        for kind in (*negative_kinds, "pooled"):
            negative = kinds != "positive" if kind == "pooled" else kinds == kind
            selected = positive | negative
            metrics[f"existence_auroc_{kind}"] = _GroupedMetric.prepare(
                "auc", scores[selected], positive[selected].astype(np.float64), clusters[selected]
            )
        metrics["correctness_auroc"] = _GroupedMetric.prepare("auc", scores[positive], correct[positive], clusters[positive])
        metrics["positive_aurc"] = _GroupedMetric.prepare("aurc", scores[positive], 1.0 - correct[positive], clusters[positive])
        metrics["positive_p_at_1"] = _GroupedMetric.prepare("mean", scores[positive], correct[positive], clusters[positive])
        result[score_name] = metrics
    return result


def _draw_cluster_weights(rng: np.random.Generator, strata: Mapping[str, np.ndarray], cluster_count: int) -> np.ndarray:
    weights = np.zeros(cluster_count, dtype=np.float64)
    for indices in strata.values():
        sampled = indices[rng.integers(0, len(indices), size=len(indices))]
        weights += np.bincount(sampled, minlength=cluster_count)
    return weights


def analyze_runs(
    runs: Mapping[str, Sequence[Mapping[str, Any]]], iterations: int = 5_000, seed: int = 20260905
) -> dict[str, Any]:
    """Compare two scores on fixed localization labels with paired cluster CIs.

    All rows of a sampled image receive the same multiplicity.  Image draws are
    stratified, shared across scores and supplied training seeds, and also shared
    between all-positive and L1 supplementary surfaces.  Undefined replicates
    invalidate that metric's entire interval instead of being silently dropped.
    """
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        raise OrderingContractError("iterations must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise OrderingContractError("bootstrap seed must be a nonnegative integer")
    aligned, lookup, strata = _validate_runs(runs)
    reference = next(iter(aligned.values()))
    negative_kinds = tuple(kind for kind in KINDS[1:] if any(row["kind"] == kind for row in reference))
    surfaces = {"all_positive": False}
    if any(row["kind"] == "positive" and row["level"] is not None for row in reference):
        surfaces["l1_positive"] = True
    prepared = {
        name: {run_seed: _prepare_surface(rows, lookup, l1=l1, negative_kinds=negative_kinds) for run_seed, rows in aligned.items()}
        for name, l1 in surfaces.items()
    }
    unit = np.ones(len(lookup), dtype=np.float64)
    output_surfaces = {}
    draws = {}
    for surface_name, by_seed in prepared.items():
        per_seed = {}
        metric_names = tuple(next(iter(by_seed.values()))["baseline"])
        for run_seed, by_score in by_seed.items():
            per_seed[run_seed] = {}
            for score_name, metrics in by_score.items():
                per_seed[run_seed][score_name] = {}
                for metric_name, metric in metrics.items():
                    value, reason = metric.evaluate(unit)
                    per_seed[run_seed][score_name][metric_name] = {"value": value, "reason": reason}
        summary = {score: {} for score in SCORES}
        for score in SCORES:
            for metric in metric_names:
                values = [per_seed[run_seed][score][metric]["value"] for run_seed in by_seed]
                defined = all(value is not None for value in values)
                summary[score][metric] = {
                    "mean": float(np.mean(values)) if defined else None,
                    "sample_sd": float(np.std(values, ddof=1)) if defined and len(values) > 1 else None,
                    "reason": None if defined else "undefined_for_at_least_one_fixed_checkpoint",
                    "sample_sd_reason": None if defined and len(values) > 1 else "requires_at_least_two_defined_checkpoint_values",
                }
        positive_count = sum(row["kind"] == "positive" and (not surfaces[surface_name] or row["level"] == 1) for row in reference)
        output_surfaces[surface_name] = {
            "scope": "all positives" if not surfaces[surface_name] else "L1 positives; existence AUROC uses all negatives",
            "positive_records": positive_count,
            "negative_records_by_kind": dict(Counter(row["kind"] for row in reference if row["kind"] != "positive")),
            "per_seed": per_seed,
            "summary": summary,
            "contrasts": {},
        }
        draws[surface_name] = {metric: np.full(iterations, np.nan) for metric in metric_names}
    rng = np.random.Generator(np.random.PCG64(seed))
    for iteration in range(iterations):
        weights = _draw_cluster_weights(rng, strata, len(lookup))
        for surface_name, by_seed in prepared.items():
            for metric_name, delta_draws in draws[surface_name].items():
                deltas = []
                for by_score in by_seed.values():
                    baseline, _ = by_score["baseline"][metric_name].evaluate(weights)
                    candidate, _ = by_score["candidate"][metric_name].evaluate(weights)
                    if baseline is None or candidate is None:
                        break
                    deltas.append(candidate - baseline)
                if len(deltas) == len(by_seed):
                    delta_draws[iteration] = float(np.mean(deltas))
    for surface_name, metrics in draws.items():
        result = output_surfaces[surface_name]
        for name, values in metrics.items():
            invalid = int(np.count_nonzero(~np.isfinite(values)))
            baseline = result["summary"]["baseline"][name]["mean"]
            candidate = result["summary"]["candidate"][name]["mean"]
            result["contrasts"][name] = {
                "candidate_minus_baseline": None if baseline is None or candidate is None else candidate - baseline,
                "ci95_percentile": None if invalid else [float(value) for value in np.percentile(values, (2.5, 97.5))],
                "undefined_replicates": invalid,
                "defined_replicates": iterations - invalid,
                "reason": "at_least_one_undefined_replicate_no_dropping" if invalid else None,
                "improvement_direction": "negative" if name == "positive_aurc" else "positive",
            }
    return {
        "schema": SCHEMA,
        "status": "complete",
        "analysis_role": "posthoc_exploratory_no_selection_or_significance_decisions",
        "records_per_checkpoint": len(reference),
        "fixed_checkpoint_seeds": list(aligned),
        "bootstrap": {
            "iterations": iterations, "rng": "PCG64", "seed": seed,
            "unit": "image_cluster", "stratified": True,
            "strata_cluster_counts": {name: len(indices) for name, indices in strata.items()},
            "image_cluster_count": len(lookup),
            "shared_draws": "all records, both scores, all fixed checkpoint seeds and all surfaces",
            "seed_aggregation": "equal mean of fixed checkpoint metrics in each draw",
            "interval_scope": "image-sampling uncertainty conditional on supplied trained checkpoints; excludes training-seed uncertainty",
            "undefined_policy": "null interval if any replicate is undefined; no dropped replicates",
        },
        "metric_contract": {
            "localization_labels": "same fixed chosen-localization correctness for both scores",
            "existence_positive_class": "target exists",
            "correctness_positive_class": "chosen positive localization is correct",
            "auroc_ties": "half credit",
            "aurc": "positive localization error; equal-score groups enter together; trapezoidal integral; risk at zero is first nonempty group's risk",
            "deployment_threshold": "none fitted or evaluated",
        },
        "surfaces": output_surfaces,
    }
