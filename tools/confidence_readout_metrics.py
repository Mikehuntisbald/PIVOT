"""Model-free target x readout audit; no model import or deployment fitting.

Inputs are localizer -> seed -> complete per-example rows.  Every localizer
shares the same sample/image universe, but correctness is localizer-specific.
Existing risk implementations are imported, never rewritten.  All quantities
are on their raw scale (AUROC [0,1], AUGRC [0,.5]).
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from numbers import Real
from typing import Mapping, Sequence

import numpy as np

from tools.grounding_confidence_ordering import _GroupedMetric, _draw_cluster_weights
from tools.grounding_emission_audit import Prepared
from tools.grounding_generalized_risk_audit import GeneralizedRiskCurve, PREVALENCES
from tools.grounding_prevalence_audit import PriorCurve

SCHEMA = "arrow.confidence_readout_metrics/v1"
SEEDS = ("17", "42", "73")
CELLS = (
    "global_max__exists", "global_max__emit",
    "native_selected__exists", "native_selected__emit",
)
EFFECTS = {
    "D_emit": {CELLS[3]: 1., CELLS[1]: -1.},
    "D_exists": {CELLS[2]: 1., CELLS[0]: -1.},
    "interaction": {CELLS[3]: 1., CELLS[2]: -1., CELLS[1]: -1., CELLS[0]: 1.},
    "global_emit_minus_exists": {CELLS[1]: 1., CELLS[0]: -1.},
    "selected_emit_minus_exists": {CELLS[3]: 1., CELLS[2]: -1.},
}


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def finite_json(value):
    if isinstance(value, Mapping):
        return {str(k): finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [finite_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def fit_sirc_statistics(rows, expected_count=83341):
    """Fit ONLY unique training positives; correctness is irrelevant to fitting.

    Each row must explicitly say split='train', kind='positive', and contain
    global_max__exists in scores.  A sorted identity+score SHA binds the actual
    fitted inputs; evaluation rows and duplicated pair-expanded positives fail.
    The caller additionally binds source provenance in its study receipt.
    """
    if type(expected_count) is not int or expected_count < 1:
        raise ValueError("positive integer expected_count required")
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} unique training positives")
    by_id = {}
    for row in rows:
        identity = row.get("sample_id")
        if not isinstance(identity, str) or not identity or identity in by_id:
            raise ValueError("unique nonempty training positive sample_id required")
        if row.get("split") != "train" or row.get("kind") != "positive":
            raise ValueError("SIRC statistics must use train positive records only")
        by_id[identity] = _number(row.get("scores", {}).get(CELLS[0]), CELLS[0])
    ordered = sorted(by_id.items())
    values = np.array([value for _, value in ordered], dtype=np.float64)
    mu, sigma = float(values.mean()), float(values.std(ddof=0))
    if not np.isfinite(mu) or not np.isfinite(sigma):
        raise ValueError("training statistics overflowed; finite mean and scale required")
    return {
        "schema": "arrow.confidence_readout_sirc_statistics/v1",
        "count": len(values), "mean": mu, "std_population": sigma,
        "a": mu - 3 * sigma, "ddof": 0, "split": "train",
        "score": CELLS[0], "zero_variance_native_fallback": sigma <= 1e-12,
        "inputs_sha256": hashlib.sha256(json.dumps(
            ordered, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode()).hexdigest(),
    }


def combine_scores(native, exists, statistics):
    """Return fixed product and SIRC-style rankings; never calibrated probabilities."""
    native, exists = np.asarray(native, dtype=np.float64), np.asarray(exists, dtype=np.float64)
    if native.shape != exists.shape or not np.isfinite(native).all() or not np.isfinite(exists).all():
        raise ValueError("aligned finite native/exists scores required")
    if (native < 0).any() or (native > 1).any():
        raise ValueError("native scores must be probabilities in [0,1]")
    if statistics.get("schema") != "arrow.confidence_readout_sirc_statistics/v1" or statistics.get("split") != "train":
        raise ValueError("bound train-only SIRC statistics required")
    if statistics.get("score") != CELLS[0] or statistics.get("ddof") != 0:
        raise ValueError("global exists and population standard deviation required")
    mu = _number(statistics.get("mean"), "mean")
    sigma = _number(statistics.get("std_population"), "std_population")
    if sigma < 0 or not np.isclose(statistics.get("a", np.nan), mu - 3 * sigma, rtol=0, atol=1e-12):
        raise ValueError("invalid SIRC scale or a statistic")
    product = np.log(np.maximum(native, 1e-6)) - np.logaddexp(0., -exists)
    sirc = native.copy() if sigma <= 1e-12 else (
        -np.log(np.maximum(1 - native, 1e-6))
        - np.logaddexp(0., -(exists - (mu - 3 * sigma)) / sigma)
    )
    return {"joint_product": product, "joint_sirc": sirc}


def add_combination_scores(rows, statistics):
    """Return copied rows; preserve all source scores and selection metadata."""
    joint = combine_scores(
        [r["native_score"] for r in rows], [r["scores"][CELLS[0]] for r in rows], statistics
    )
    output = []
    for i, row in enumerate(rows):
        scores = dict(row["scores"])
        if set(joint) & set(scores):
            raise ValueError("combination scores already present; do not overwrite")
        output.append({**row, "scores": {**scores, **{k: float(v[i]) for k, v in joint.items()}}})
    return output


def _with_cross_readouts(runs):
    """Use frozen head alternate reads only when supplied for the entire panel."""
    all_rows = [row for seeds in runs.values() for rows in seeds.values() for row in rows]
    present = [bool(row.get("readout_diagnostics")) for row in all_rows]
    if not any(present):
        return runs, {}
    if not all(present):
        raise ValueError("partial readout diagnostics would change the diagnostic population")
    names = {}
    output = {}
    for localizer, seeds in runs.items():
        output[localizer] = {}
        for seed, rows in seeds.items():
            result = []
            for row in rows:
                scores = dict(row["scores"])
                native_indices, native_ious = set(), set()
                for arm in CELLS:
                    diagnostic = row["readout_diagnostics"].get(arm)
                    if not isinstance(diagnostic, Mapping):
                        raise ValueError("all four trained head diagnostics are required")
                    for field in ("native_selected_index", "confidence_winner_index"):
                        index = diagnostic.get(field)
                        if type(index) is not int or index < 0:
                            raise ValueError("nonnegative integer winner indices required")
                    native_indices.add(diagnostic["native_selected_index"])
                    for field in ("winner_native_box_iou", "winner_gt_iou", "native_gt_iou"):
                        value = diagnostic.get(field)
                        if row.get("kind") != "positive" and field != "winner_native_box_iou":
                            if value is not None:
                                raise ValueError("no-target diagnostics cannot invent GT IoU")
                            continue
                        value = _number(value, field)
                        if not 0 <= value <= 1:
                            raise ValueError("IoU diagnostics must be in [0,1]")
                    native_ious.add(diagnostic.get("native_gt_iou"))
                    if row.get("kind") == "positive" and (diagnostic["native_gt_iou"] >= .5) != row.get("correct"):
                        raise ValueError("Native GT IoU disagrees with fixed correctness label")
                    maximum = _number(diagnostic.get("max_logit"), "max_logit")
                    selected = _number(diagnostic.get("selected_logit"), "selected_logit")
                    if selected > maximum:
                        raise ValueError("selected logit cannot exceed valid global max")
                    global_trained = arm.startswith("global_max")
                    matched = maximum if global_trained else selected
                    if scores[arm] != matched:
                        raise ValueError("matched score does not match its recorded readout")
                    name = arm + ("__eval_selected" if global_trained else "__eval_global")
                    if name in scores:
                        raise ValueError("off-diagonal diagnostic score would overwrite input")
                    scores[name] = selected if global_trained else maximum
                    names[name] = {"trained_head": arm, "eval_readout": "native_selected" if global_trained else "global_max"}
                if len(native_indices) != 1 or len(native_ious) != 1:
                    raise ValueError("all confidence heads must preserve the same Native selection")
                result.append({**row, "scores": scores})
            output[localizer][seed] = result
    return output, names


def _validate(runs, required_seeds):
    if not isinstance(runs, Mapping) or not runs:
        raise ValueError("localizer -> seed -> rows mapping required")
    aligned, common_identity, common_scores = {}, None, None
    for localizer in sorted(runs):
        if not isinstance(localizer, str) or not localizer:
            raise ValueError("nonempty localizer names required")
        seed_runs = runs[localizer]
        if set(seed_runs) != set(required_seeds):
            raise ValueError(f"{localizer}: exact required seeds {required_seeds} needed")
        aligned[localizer] = {}
        local_reference = None
        for seed in sorted(seed_runs):
            source = seed_runs[seed]
            if not isinstance(source, Sequence) or isinstance(source, (str, bytes)) or not source:
                raise ValueError("nonempty rows required")
            by_id = {}
            for row in source:
                for field in ("sample_id", "cluster_id", "stratum"):
                    if not isinstance(row.get(field), str) or not row[field]:
                        raise ValueError(f"nonempty {field} required")
                if row["sample_id"] in by_id:
                    raise ValueError("duplicate sample identity")
                if row.get("kind") not in ("positive", "text", "no_target"):
                    raise ValueError("only positive/text/no_target records are in scope")
                if (row["kind"] == "positive" and type(row.get("correct")) is not bool) or (
                    row["kind"] != "positive" and row.get("correct", "missing") is not None
                ):
                    raise ValueError("positive correctness bool; negative correctness null required")
                if row.get("split") == "train":
                    raise ValueError("train rows cannot enter evaluation")
                level = row.get("level")
                if level is not None and (type(level) is not int or level < 1):
                    raise ValueError("positive difficulty level must be positive integer or null")
                parent = row.get("parent_positive_id")
                if parent is not None and (not isinstance(parent, str) or not parent):
                    raise ValueError("parent_positive_id must be nonempty string or null")
                native = _number(row.get("native_score"), "native_score")
                if not 0 <= native <= 1:
                    raise ValueError("Native probability must be in [0,1]")
                scores = row.get("scores")
                if not isinstance(scores, Mapping) or not set(CELLS).issubset(scores):
                    raise ValueError("all four matched score cells required")
                if any(not isinstance(k, str) or not k for k in scores):
                    raise ValueError("nonempty score names required")
                for name, score in scores.items():
                    _number(score, name)
                if "native" in scores and scores["native"] != native:
                    raise ValueError("Native score slot mismatch")
                row = {**row, "scores": {**scores, "native": native}}
                if common_scores is None:
                    common_scores = tuple(sorted(row["scores"]))
                if tuple(sorted(row["scores"])) != common_scores:
                    raise ValueError("score slots must match across all rows/localizers/seeds")
                by_id[row["sample_id"]] = row
            rows = [by_id[key] for key in sorted(by_id)]
            identity = [tuple(r.get(k) for k in (
                "sample_id", "cluster_id", "stratum", "kind", "level", "parent_positive_id"
            )) for r in rows]
            if common_identity is not None and identity != common_identity:
                raise ValueError("sample/image/stratum/parent identities differ across runs")
            common_identity = identity
            local_identity = [(
                r["correct"], r["native_score"],
                r.get("readout_diagnostics", {}).get(CELLS[0], {}).get("native_selected_index"),
                r.get("readout_diagnostics", {}).get(CELLS[0], {}).get("native_gt_iou"),
            ) for r in rows]
            if local_reference is not None and local_identity != local_reference:
                raise ValueError("Native correctness/scores drift across same-localizer seeds")
            local_reference = local_identity
            for r in rows:
                parent = r.get("parent_positive_id")
                if parent is not None:
                    if r["kind"] == "positive" or parent not in by_id:
                        raise ValueError("negative parent must reference a present positive")
                    p = by_id[parent]
                    if p["kind"] != "positive" or p["cluster_id"] != r["cluster_id"]:
                        raise ValueError("parent must be a positive from the same image")
            aligned[localizer][seed] = rows
    first = next(iter(next(iter(aligned.values())).values()))
    cluster_strata = {}
    for r in first:
        if r["cluster_id"] in cluster_strata and cluster_strata[r["cluster_id"]] != r["stratum"]:
            raise ValueError("one image cannot belong to multiple strata")
        cluster_strata[r["cluster_id"]] = r["stratum"]
    lookup = {key: i for i, key in enumerate(sorted(cluster_strata))}
    strata = defaultdict(list)
    for key, i in lookup.items():
        strata[cluster_strata[key]].append(i)
    return aligned, lookup, {k: np.asarray(v, dtype=np.int64) for k, v in sorted(strata.items())}, common_scores


def augrc_from_states(a, pi, auc_cw, auc_cn):
    """Exact AUGRC identity; W-N never enters correct-vs-failure ranking."""
    good, wrong, absent = (1 - pi) * a, (1 - pi) * (1 - a), pi
    pair_error = (good * wrong * (1 - auc_cw) if good * wrong else 0.)
    pair_error += good * absent * (1 - auc_cn) if good * absent else 0.
    return pair_error + .5 * (wrong + absent) ** 2


def augrc_crossover(a, delta_cw, delta_cn):
    """Return only an identified interior crossover; never clip to [0,1]."""
    if not np.isfinite(a):
        return {"prior": None, "status": "undefined_state_auc"}
    if not 0 <= a <= 1:
        raise ValueError("positive Native accuracy must be in [0,1]")
    if a == 0:
        return {"prior": None, "status": "no_correct_output_mass"}
    # With no wrong positive mass, C-W AUC is undefined but irrelevant.
    if a == 1:
        delta_cw = 0.
    if not all(np.isfinite(v) for v in (delta_cw, delta_cn)):
        return {"prior": None, "status": "undefined_state_auc"}
    b, d = (1 - a) * delta_cw, delta_cn - (1 - a) * delta_cw
    # No absolute epsilon: rescaling an effect must not change whether its
    # linear factor has a root. Statistical precision is reported separately.
    if d == 0:
        return {"prior": None, "status": "identical_all_priors" if b == 0 else "no_root"}
    root = -b / d
    if not 0 < root < 1:
        return {"prior": None, "status": "no_interior_root", "unclipped_root": root}
    return {"prior": float(root), "status": "interior"}


def _state(rows):
    return np.asarray([0 if r["kind"] != "positive" else 2 if r["correct"] else 1 for r in rows], dtype=np.int64)


def _auc(scores, labels, weights):
    metric = _GroupedMetric.prepare("auc", scores, labels.astype(float), np.arange(len(scores)))
    value, _ = metric.evaluate(weights)
    return np.nan if value is None else value


class _Conditional:
    """Precomputed same-image pairs and paired edits, plus comparable surfaces.

    A resampled image copy contributes its within-copy pairs once (m_i), not
    m_i squared. Across-image AUC uses ordinary replicated request masses.
    """
    def __init__(self, rows, scores, state, clusters, nclusters):
        self.pairs, self.grouped, self.counts, self.difficulty = {}, {}, {}, {}
        for label, high, low in (("cw", 2, 1), ("cn", 2, 0), ("wn", 1, 0)):
            numerator, denominator = np.zeros(nclusters), np.zeros(nclusters)
            selected = np.zeros(len(rows), dtype=bool)
            for c in np.unique(clusters):
                h, l = (clusters == c) & (state == high), (clusters == c) & (state == low)
                count = int(h.sum() * l.sum())
                if not count:
                    continue
                mask = h | l
                auc = _auc(scores[mask], h[mask], np.ones(mask.sum()))
                numerator[c], denominator[c] = auc * count, count
                selected |= mask
            name = "same_image_" + label
            self.pairs[name] = (numerator, denominator)
            self.grouped[name + "_comparable_unconditional"] = _GroupedMetric.prepare(
                "auc", scores[selected], (state[selected] == high).astype(float), clusters[selected]
            )
            self.counts[name] = {
                "eligible_images": int(np.count_nonzero(denominator)),
                "within_image_pairs": int(denominator.sum()), "eligible_records": int(selected.sum()),
            }
        index = {r["sample_id"]: i for i, r in enumerate(rows)}
        for status in ("all", "C", "W"):
            numerator, denominator = np.zeros(nclusters), np.zeros(nclusters)
            for i, r in enumerate(rows):
                parent = r.get("parent_positive_id")
                if parent is None:
                    continue
                p = index[parent]
                if status != "all" and state[p] != (2 if status == "C" else 1):
                    continue
                numerator[clusters[i]] += float(scores[p] > scores[i]) + .5 * float(scores[p] == scores[i])
                denominator[clusters[i]] += 1
            self.pairs["parent_pair_" + status] = (numerator, denominator)
            self.counts["parent_pair_" + status] = {
                "pairs": int(denominator.sum()), "images": int(np.count_nonzero(denominator))
            }
        # Negative edit difficulty is deliberately ignored. When available,
        # negatives inherit the ORIGINAL parent positive expression difficulty.
        levels = np.full(len(rows), -1, dtype=int)
        for i, r in enumerate(rows):
            p = i if state[i] else index.get(r.get("parent_positive_id"), -1)
            if p >= 0 and rows[p].get("level") is not None:
                levels[i] = rows[p]["level"]
        for label, high, low in (("cw", 2, 1), ("cn", 2, 0), ("wn", 1, 0)):
            participating = (levels > 0) & ((state == high) | (state == low))
            level_masks = []
            for level in sorted(set(levels[participating])):
                mask = participating & (levels == level)
                name = f"difficulty_{label}_level{level}"
                self.grouped[name] = _GroupedMetric.prepare(
                    "auc", scores[mask], (state[mask] == high).astype(float), clusters[mask]
                )
                self.counts[name] = {"high": int((mask & (state == high)).sum()),
                                     "low": int((mask & (state == low)).sum()),
                                     "images": len(set(clusters[mask].tolist()))}
                level_masks.append((name, clusters[mask & (state == high)], clusters[mask & (state == low)]))
            self.grouped[f"difficulty_{label}_comparable_unconditional"] = _GroupedMetric.prepare(
                "auc", scores[participating], (state[participating] == high).astype(float), clusters[participating]
            )
            self.difficulty[label] = (level_masks, clusters[participating & (state == high)],
                                      clusters[participating & (state == low)])

    def evaluate(self, weights):
        values = {}
        for name, (n, d) in self.pairs.items():
            mass = np.dot(weights, d)
            values[name] = np.dot(weights, n) / mass if mass else np.nan
        for name, metric in self.grouped.items():
            value, _ = metric.evaluate(weights)
            values[name] = np.nan if value is None else value
        for label in ("cw", "cn", "wn"):
            base = f"same_image_{label}"
            # Applying the target/readout contrast to this per-score quantity
            # gives a paired CI for effect attenuation on a comparable surface.
            values[base + "_unconditional_minus_conditional"] = (
                values[base + "_comparable_unconditional"] - values[base]
            )
        for label, (levels, high_clusters, low_clusters) in self.difficulty.items():
            denominator = weights[high_clusters].sum() * weights[low_clusters].sum()
            pair_mass, numerator = 0., 0.
            for name, hi, lo in levels:
                mass = weights[hi].sum() * weights[lo].sum()
                if mass:
                    pair_mass += mass
                    numerator += mass * values[name]
            within = numerator / denominator if denominator else np.nan
            total = values[f"difficulty_{label}_comparable_unconditional"]
            values[f"difficulty_{label}_same_level_pair_auroc"] = numerator / pair_mass if pair_mass else np.nan
            values[f"difficulty_{label}_within_level_contribution"] = within
            values[f"difficulty_{label}_cross_level_contribution"] = total - within
            values[f"difficulty_{label}_same_level_pair_fraction"] = pair_mass / denominator if denominator else np.nan
            values[f"difficulty_{label}_unconditional_minus_same_level"] = (
                total - values[f"difficulty_{label}_same_level_pair_auroc"]
            )
        return values


class _PreparedReadout:
    def __init__(self, rows, arm, clusters, nclusters, conditionals):
        self.state, self.clusters = _state(rows), clusters
        scores = np.asarray([r["scores"][arm] for r in rows], dtype=np.float64)
        self.base = Prepared(scores, self.state, clusters)
        self.gr = GeneralizedRiskCurve.prepare(scores, self.state, clusters)
        self.prior = PriorCurve.prepare(scores, self.state, clusters)
        self.conditional = _Conditional(rows, scores, self.state, clusters, nclusters) if conditionals else None

    def evaluate(self, weights):
        values = self.base.evaluate(weights)
        masses = weights[self.clusters]
        pos = self.state > 0
        a = np.dot(masses, self.state == 2) / masses[pos].sum() if masses[pos].sum() else np.nan
        pi = np.dot(masses, ~pos) / masses.sum()
        values["native_p_at_1"] = a
        values["observed_no_target_fraction"] = pi
        priors = [*PREVALENCES, pi]
        gr, risk = self.gr.evaluate(weights, priors), self.prior.evaluate(weights, PREVALENCES)
        values.update(zip(("mixed_augrc", "wrong_box_augrc", "no_target_augrc"), gr[-1]))
        for i, p in enumerate(PREVALENCES):
            for j, name in enumerate(("mixed", "wrong_box", "no_target")):
                values[f"prior_{p:g}__{name}_augrc"] = gr[i, j]
                values[f"prior_{p:g}__{name}_aurc"] = risk[i, j]
        if self.conditional is not None:
            values.update(self.conditional.evaluate(weights))
        return values


def _summary(points, draws):
    points, draws = np.asarray(points, dtype=float), np.asarray(draws, dtype=float)
    invalid = int((~np.isfinite(draws)).sum())
    valid_point = np.isfinite(points).all()
    return {"mean": float(points.mean()) if valid_point else None,
            "sample_sd": float(points.std(ddof=1)) if valid_point and len(points) > 1 else None,
            "ci95": np.percentile(draws, [2.5, 97.5]).tolist() if not invalid else None,
            "undefined_replicates": invalid}


def _effect_values(by_arm, coefficients, metric):
    return sum(coefficient * by_arm[arm][metric] for arm, coefficient in coefficients.items())


def _winner_summary(rows, arm):
    """Descriptive geometry; missing diagnostics are missing, never zero."""
    groups = {}
    for state in ("C", "W", "N"):
        selected = [r for r in rows if ("N" if r["kind"] != "positive" else "C" if r["correct"] else "W") == state]
        present = [r["readout_diagnostics"][arm] for r in selected
                   if isinstance(r.get("readout_diagnostics"), Mapping) and arm in r["readout_diagnostics"]]
        out = {"records": len(selected), "diagnostic_records": len(present)}
        if present:
            fields = ("winner_native_box_iou", "native_gt_iou", "winner_gt_iou")
            for field in fields:
                vals = [d[field] for d in present if d.get(field) is not None]
                if any(not isinstance(v, (Real, bool)) or not np.isfinite(v) for v in vals):
                    raise ValueError("finite geometry diagnostics required")
                out[field + "_mean"] = float(np.mean(vals)) if vals else None
            differs = [d["confidence_winner_index"] != d["native_selected_index"] for d in present
                       if d.get("confidence_winner_index") is not None and d.get("native_selected_index") is not None]
            out["winner_differs_mean"] = float(np.mean(differs)) if differs else None
            if state == "W":
                gt = [d["winner_gt_iou"] for d in present if d.get("winner_gt_iou") is not None]
                out["confidence_winner_correct_fraction"] = float(np.mean(np.asarray(gt) >= .5)) if gt else None
        groups[state] = out
    return groups


def analyze_readout(runs, iterations=5000, seed=20260911, required_seeds=SEEDS,
                    conditionals=True, progress=None):
    """Common paired image draws across every localizer, head, and fixed seed.

    Conditional intervals are undefined if any draw lacks required states; no
    draw is deleted. Geometric diagnostics are descriptive only. Off-diagonal
    eval scores can be included as additional score slots; they get summaries
    but are not substituted into the four matched cells.
    """
    if type(iterations) is not int or iterations < 1 or type(seed) is not int or seed < 0:
        raise ValueError("positive iterations and nonnegative seed required")
    if not isinstance(runs, Mapping) or not runs or any(not isinstance(s, Mapping) or not s for s in runs.values()):
        raise ValueError("nonempty localizer -> seed -> rows mapping required")
    required_seeds = tuple(str(s) for s in required_seeds)
    if not required_seeds or len(set(required_seeds)) != len(required_seeds):
        raise ValueError("unique nonempty required seeds needed")
    runs, cross_readouts = _with_cross_readouts(runs)
    aligned, lookup, strata, arms = _validate(runs, required_seeds)
    first = next(iter(next(iter(aligned.values())).values()))
    clusters = np.asarray([lookup[r["cluster_id"]] for r in first], dtype=np.int64)
    prepared = {l: {s: {a: _PreparedReadout(rows, a, clusters, len(lookup), conditionals)
                        for a in arms} for s, rows in owners.items()} for l, owners in aligned.items()}
    unit = np.ones(len(lookup), dtype=float)
    points = {l: {s: {a: p.evaluate(unit) for a, p in owners.items()} for s, owners in seeds.items()}
              for l, seeds in prepared.items()}
    draws = {l: {a: {m: np.full(iterations, np.nan) for m in next(iter(values.values()))[a]}
                  for a in arms} for l, values in points.items()}
    rng, draw_hash = np.random.Generator(np.random.PCG64(seed)), hashlib.sha256()
    for i in range(iterations):
        weights = _draw_cluster_weights(rng, strata, len(lookup))
        draw_hash.update(weights.astype("<u4").tobytes())
        for localizer, seeds in prepared.items():
            for arm in arms:
                # Native is identical within a localizer, not between localizers.
                owners = list(seeds.values())[:1] if arm == "native" else list(seeds.values())
                values = [owner[arm].evaluate(weights) for owner in owners]
                for m in draws[localizer][arm]:
                    draws[localizer][arm][m][i] = np.mean([v[m] for v in values])
        if progress is not None and ((i + 1) % 25 == 0 or i + 1 == iterations):
            progress(i + 1, iterations)
    output = {}
    for localizer, values in points.items():
        local_draws = draws[localizer]
        metric_names = tuple(next(iter(values.values()))["native"])
        summary = {a: {m: _summary([v[a][m] for v in values.values()], local_draws[a][m])
                       for m in metric_names} for a in arms}
        effects = dict(EFFECTS)
        for joint in ("joint_product", "joint_sirc"):
            if joint in arms:
                for baseline in ("native", CELLS[0], CELLS[1]):
                    effects[joint + "_minus_" + baseline] = {joint: 1., baseline: -1.}
        for name, description in cross_readouts.items():
            effects["fixed_weights__" + name + "_minus_matched"] = {
                name: 1., description["trained_head"]: -1.,
            }
        effects_result, crossovers = {}, {}
        for effect, coefficients in effects.items():
            effect_draws = {m: sum(c * local_draws[a][m] for a, c in coefficients.items()) for m in metric_names}
            effects_result[effect] = {
                m: {**_summary([_effect_values(v, coefficients, m) for v in values.values()], effect_draws[m]),
                    "definition": coefficients} for m in metric_names
            }
            a = summary["native"]["native_p_at_1"]["mean"]
            point_root = augrc_crossover(
                np.nan if a is None else a,
                np.mean([_effect_values(v, coefficients, "correctness_auroc") for v in values.values()]),
                np.mean([_effect_values(v, coefficients, "correct_vs_no_target_auroc") for v in values.values()]),
            )
            roots = [augrc_crossover(local_draws["native"]["native_p_at_1"][i],
                                    effect_draws["correctness_auroc"][i],
                                    effect_draws["correct_vs_no_target_auroc"][i]) for i in range(iterations)]
            counts = {status: sum(r["status"] == status for r in roots) for status in sorted({r["status"] for r in roots})}
            valid = [r["prior"] for r in roots if r["prior"] is not None]
            crossovers[effect] = {"point": point_root, "bootstrap_status_counts": counts,
                "ci95": np.percentile(valid, [2.5, 97.5]).tolist() if len(valid) == iterations else None,
                "conditional_on_interior_ci95": np.percentile(valid, [2.5, 97.5]).tolist() if valid else None,
                "conditional_interval_warning": "conditional on an interior root; absent roots are counted, never imputed",
                "per_seed": {s: augrc_crossover(
                    v["native"]["native_p_at_1"],
                    _effect_values(v, coefficients, "correctness_auroc"),
                    _effect_values(v, coefficients, "correct_vs_no_target_auroc"),
                ) for s, v in values.items()},
                "point_root_estimand": "root of equal-seed mean risk difference, not mean of seed roots"}
        first_rows = next(iter(aligned[localizer].values()))
        states = _state(first_rows)
        output[localizer] = {
            "population": {"records": len(first_rows), "images": len(lookup),
                           "C": int((states == 2).sum()), "W": int((states == 1).sum()), "N": int((states == 0).sum())},
            "per_seed": values, "summary": summary, "effects": effects_result, "augrc_crossovers": crossovers,
            "conditional_counts": next(iter(prepared[localizer].values()))["native"].conditional.counts if conditionals else None,
            "winner_geometry": {s: {a: _winner_summary(rows, a) for a in CELLS} for s, rows in aligned[localizer].items()},
        }
        # Strict independent identity checks on points (including tied scores).
        residuals = []
        for per_seed in values.values():
            for v in per_seed.values():
                a, pi = v["native_p_at_1"], v["observed_no_target_fraction"]
                implied = augrc_from_states(a, pi, v["correctness_auroc"], v["correct_vs_no_target_auroc"])
                if np.isfinite(implied):
                    residuals.append(abs(implied - v["mixed_augrc"]))
                existence = a * v["correct_vs_no_target_auroc"] + (1 - a) * v["wrong_positive_vs_no_target_auroc"]
                if np.isfinite(existence):
                    residuals.append(abs(existence - v["existence_auroc"]))
        if residuals and max(residuals) > 2e-12:
            raise ValueError("three-state risk identity failed")
        output[localizer]["max_state_identity_error"] = max(residuals) if residuals else None
    return finite_json({
        "schema": SCHEMA, "role": "posthoc_motivated_prospectively_locked_mechanism_analysis",
        "score_scale": "raw; no percentages; AUGRC in [0,.5]", "primary_metric": "mixed_augrc",
        "matched_cells": list(CELLS), "localizers": output,
        "cross_readout_scores": cross_readouts,
        "bootstrap": {"iterations": iterations, "seed": seed, "rng": "PCG64", "unit": "image_cluster",
            "strata": {k: len(v) for k, v in strata.items()}, "required_seeds": list(required_seeds),
            "same_draw_all_localizers_heads_seeds": True, "draws_sha256": draw_hash.hexdigest(),
            "q05_recomputed_each_draw": True, "fixed_threshold_fit": False,
            "observed_mixture_varies_with_image_draw": True, "fixed_priors_class_renormalized_each_draw": True,
            "undefined_policy": "any undefined draw nulls its affected interval; no deleted draws",
            "uncertainty": "image sampling conditional on fixed checkpoints; seeds equally averaged"},
        "interpretation": {"negative_interaction_is_not_emit_repair": True,
            "spatial_alignment_is_unproven": True, "conditional_null_is_not_attribution": True,
            "failed_fixed_combinations_do_not_prove_noncomposability": True,
            "same_image_pair_weights": "image multiplicity times within-image pairs; never multiplicity squared",
            "difficulty_semantics": "positive expression level; negatives use parent positive level only, never edit level"},
    })
