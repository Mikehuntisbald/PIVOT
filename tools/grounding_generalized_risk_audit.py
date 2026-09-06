"""Record-only AUGRC robustness audit with fixed request-prior reweighting.

Generalized risk is the population mass of accepted failures, not failure risk
conditional on acceptance. Its curve starts at (coverage, generalized risk) =
(0, 0), including when the highest-scoring group contains failures. Equal-score
groups enter together and are connected by trapezoidal interpolation.

This module fits no score, policy, threshold, checkpoint, or deployment prior.
The ``observed`` population uses unweighted empirical requests and image-draw
multiplicities, including each replicate's fluctuating absent-request fraction.
The separate fixed prior grid is class-renormalized within every replicate.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from tools.grounding_confidence_ordering import _draw_cluster_weights, _validate_runs


SCHEMA = "arrow.generalized_risk_audit/v1"
PREVALENCES = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
ARMS = ("native", "exists", "emit")
METRICS = ("mixed_augrc", "wrong_box_augrc", "no_target_augrc")


@dataclass(frozen=True)
class GeneralizedRiskCurve:
    """Prepared descending score groups; state 0/1/2 = absent/wrong/correct."""

    clusters: np.ndarray
    positive: np.ndarray
    wrong: np.ndarray
    starts: np.ndarray

    @classmethod
    def prepare(cls, scores, state, clusters):
        scores = np.asarray(scores, dtype=np.float64)
        state, clusters = np.asarray(state), np.asarray(clusters)
        if (
            scores.ndim != 1
            or not scores.size
            or scores.shape != state.shape
            or scores.shape != clusters.shape
            or not np.isfinite(scores).all()
        ):
            raise ValueError("nonempty aligned finite scores, states and clusters required")
        if not np.issubdtype(state.dtype, np.integer) or not np.isin(state, (0, 1, 2)).all():
            raise ValueError("states must be integer absent=0, wrong=1, correct=2")
        if not np.issubdtype(clusters.dtype, np.integer) or (clusters < 0).any():
            raise ValueError("clusters must be nonnegative integer indices")
        order = np.argsort(-scores, kind="stable")
        starts = np.r_[0, np.flatnonzero(scores[order][1:] != scores[order][:-1]) + 1]
        return cls(clusters[order], state[order] > 0, state[order] == 1, starts)

    def evaluate(self, cluster_weights, prevalences):
        """Return columns METRICS per prior; unsupported class mass yields NaN.

        Weight inputs may be arbitrary nonnegative image masses. Formal bootstrap
        callers supply integer multiplicities. Missing positives invalidate pi<1;
        missing absent requests invalidate pi>0. A missing zero-mass class does
        not invalidate the supported endpoint.
        """
        weights = np.asarray(cluster_weights, dtype=np.float64)
        priors = np.asarray(prevalences, dtype=np.float64)
        if (
            weights.ndim != 1
            or len(weights) <= int(self.clusters.max())
            or not np.isfinite(weights).all()
            or (weights < 0).any()
        ):
            raise ValueError("finite nonnegative weights covering every cluster required")
        if (
            priors.ndim != 1
            or not len(priors)
            or not np.isfinite(priors).all()
            or (priors < 0).any()
            or (priors > 1).any()
        ):
            raise ValueError("finite nonempty prevalence vector in [0,1] required")

        row_weights = weights[self.clusters]
        gp = np.add.reduceat(row_weights * self.positive, self.starts)
        gn = np.add.reduceat(row_weights * ~self.positive, self.starts)
        gw = np.add.reduceat(row_weights * self.wrong, self.starts)
        positive_mass, absent_mass = float(gp.sum()), float(gn.sum())
        values = np.full((len(priors), len(METRICS)), np.nan)

        for index, pi in enumerate(priors):
            if (pi < 1 and positive_mass <= 0) or (pi > 0 and absent_mass <= 0):
                continue
            a = (1 - pi) / positive_mass if positive_mass else 0.0
            b = pi / absent_mass if absent_mass else 0.0
            group_mass = a * gp + b * gn
            active = group_mass > 0
            retained = np.cumsum(group_mass[active])
            total = retained[-1]
            coverage = retained / total
            # Crucially, these denominators are total population mass, not the
            # retained mass used by a conventional selective-risk curve.
            wrong_gr = np.cumsum(a * gw[active]) / total
            absent_gr = np.cumsum(b * gn[active]) / total

            def integral(generalized_risk):
                # Start at (0,0), even for a tied/incorrect top-score group.
                return float(np.sum(
                    np.diff(np.r_[0.0, coverage])
                    * (generalized_risk + np.r_[0.0, generalized_risk[:-1]]) / 2
                ))

            wrong_area, absent_area = integral(wrong_gr), integral(absent_gr)
            values[index] = (wrong_area + absent_area, wrong_area, absent_area)
        return values


def analyze_generalized_risk(
    runs: Mapping[str, Sequence[Mapping]], iterations=5000, seed=20260910
):
    """Paired, stratified-image AUGRC audit at a fixed grid and observed prior.

    Canonical score fields are native_score/baseline_score/candidate_score,
    displayed as native/exists/emit. Cross-seed sample identities, fixed labels
    and Native scores must agree. Any undefined replicate nulls its affected
    interval; no draw is silently discarded.
    """
    if type(iterations) is not int or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    aligned, lookup, strata = _validate_runs(runs)
    first = next(iter(aligned.values()))
    for rows in aligned.values():
        for row, reference in zip(rows, first, strict=True):
            score = row.get("native_score")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not np.isfinite(score):
                raise ValueError("finite numeric Native score required")
            if score != reference["native_score"]:
                raise ValueError("cross-seed Native parity failed")

    state = np.asarray([
        0 if row["kind"] != "positive" else (2 if row["correct"] else 1)
        for row in first
    ], dtype=np.int64)
    clusters = np.asarray([lookup[row["cluster_id"]] for row in first], dtype=np.int64)
    observed = float(np.mean(state == 0))
    prior_items = [(f"{pi:g}", pi) for pi in PREVALENCES] + [("observed", observed)]
    priors = np.asarray([pi for _, pi in prior_items], dtype=np.float64)
    columns = {"native": "native_score", "exists": "baseline_score", "emit": "candidate_score"}
    prepared = {
        s: {
            arm: GeneralizedRiskCurve.prepare(
                np.asarray([row[column] for row in rows]), state, clusters
            ) for arm, column in columns.items()
        } for s, rows in aligned.items()
    }
    units = np.ones(len(lookup))
    points = {s: {arm: p.evaluate(units, priors) for arm, p in owners.items()}
              for s, owners in prepared.items()}
    draws = {arm: np.full((iterations, len(priors), len(METRICS)), np.nan) for arm in ARMS}
    rng = np.random.Generator(np.random.PCG64(seed))
    native = next(iter(prepared.values()))["native"]
    for iteration in range(iterations):
        weights = _draw_cluster_weights(rng, strata, len(lookup))
        # The observed-population estimand must match the earlier ordinary
        # image bootstrap: it retains the draw's actual class composition.
        # Reweighting at that empirical prior is exactly the normalized image
        # multiplicity measure, not an additional class-prior intervention.
        row_weights = weights[clusters]
        draw_priors = priors.copy()
        draw_priors[-1] = np.dot(row_weights, state == 0) / row_weights.sum()
        draws["native"][iteration] = native.evaluate(weights, draw_priors)
        for arm in ("exists", "emit"):
            draws[arm][iteration] = np.mean(
                [owners[arm].evaluate(weights, draw_priors) for owners in prepared.values()], axis=0
            )

    def summarize(values, samples):
        undefined = int(np.count_nonzero(~np.isfinite(samples)))
        return {
            "mean": float(np.mean(values)),
            "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else None,
            "ci95": np.percentile(samples, [2.5, 97.5]).tolist() if not undefined else None,
            "undefined_replicates": undefined,
        }

    summary, contrasts = {}, {}
    for index, (key, _) in enumerate(prior_items):
        summary[key], contrasts[key] = {}, {}
        for arm in ARMS:
            seeds = list(points)[:1] if arm == "native" else list(points)
            summary[key][arm] = {
                metric: summarize([points[s][arm][index, k] for s in seeds], draws[arm][:, index, k])
                for k, metric in enumerate(METRICS)
            }
        for a, b in (("exists", "native"), ("emit", "native"), ("emit", "exists")):
            row = {}
            for k, metric in enumerate(METRICS):
                item = summarize(
                    [points[s][a][index, k] - points[s][b][index, k] for s in points],
                    draws[a][:, index, k] - draws[b][:, index, k],
                )
                item["delta"] = item.pop("mean")
                row[metric] = item
            contrasts[key][f"{a}_minus_{b}"] = row

    result = {
        "schema": SCHEMA,
        "role": "posthoc_generalized_risk_robustness_no_selection",
        "metrics": list(METRICS),
        "prevalences": list(PREVALENCES),
        "observed_no_target_fraction": observed,
        "evaluation_priors": dict(prior_items),
        "population": {
            "records": len(first), "images": len(lookup),
            "positive_correct": int((state == 2).sum()),
            "positive_wrong": int((state == 1).sum()), "no_target": int((state == 0).sum()),
        },
        "bootstrap": {
            "iterations": iterations, "seed": seed, "rng": "PCG64", "unit": "image_cluster",
            "strata": {s: len(v) for s, v in strata.items()},
            "same_draw_all_scores_seeds_prevalences": True,
            "renormalize_fixed_grid_classes_within_each_draw": True,
            "observed_prior": "unweighted empirical population; class fraction fluctuates with each image draw",
            "undefined_policy": "null affected interval if any replicate is undefined; no discarded draws",
            "uncertainty": "image sampling conditional on the supplied fixed checkpoints",
        },
        "reweighting": "positive m_i*(1-pi)/positive_mass; absent m_i*pi/absent_mass",
        "prior_semantics": "fixed grid changes class masses; observed retains image multiplicities without a fixed-prior intervention",
        "generalized_risk": "population mass of accepted wrong-box or no-target failures",
        "integration": "whole-score groups; trapezoidal integral over coverage starting at (0,0)",
        "identity": "err*acc*(1-AUC(correct versus failure)) + 0.5*err**2; half-credit ties",
        "policy_selection": False,
        "summary": summary,
        "contrasts": contrasts,
        "per_seed": {
            s: {arm: {key: dict(zip(METRICS, values[index].tolist()))
                      for index, (key, _) in enumerate(prior_items)}
                for arm, values in owners.items()}
            for s, owners in points.items()
        },
    }

    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    return clean(result)
