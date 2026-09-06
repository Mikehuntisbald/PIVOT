import copy
import json

import numpy as np
import pytest

from tools.grounding_generalized_risk_audit import (
    GeneralizedRiskCurve, METRICS, PREVALENCES, analyze_generalized_risk,
)


def fixture():
    rows = []
    for image in range(4):
        for j, (kind, correct, score) in enumerate((
            ("positive", True, .9 - image * .15),
            ("positive", False, .5 + image * .1),
            ("no_target", None, .2 + image * .2),
        )):
            rows.append(dict(
                sample_id=f"{image}-{j}", cluster_id=str(image), stratum="A" if image < 2 else "B",
                kind=kind, correct=correct, level=None, native_score=score,
                baseline_score=score, candidate_score=1 - score,
            ))
    return rows


def direct_identity(scores, state, masses):
    """Independent pairwise half-tie AUC; no production grouping/integration."""
    good, bad = state == 2, state != 2
    total = masses.sum()
    err, acc = masses[bad].sum() / total, masses[good].sum() / total
    if err == 0:
        return 0.0
    if acc == 0:
        return 0.5
    numerator = 0.0
    for i in np.flatnonzero(good):
        for j in np.flatnonzero(bad):
            numerator += masses[i] * masses[j] * (
                float(scores[i] > scores[j]) + .5 * float(scores[i] == scores[j])
            )
    auc = numerator / (masses[good].sum() * masses[bad].sum())
    return err * acc * (1 - auc) + .5 * err ** 2


def class_normalized_masses(state, weights, pi):
    positive = state > 0
    return np.where(positive, weights * (1 - pi) / weights[positive].sum(),
                    weights * pi / weights[~positive].sum())


def test_origin_and_whole_tie_group_are_not_selective_risk_extension():
    state = np.array([0, 1, 2])
    curve = GeneralizedRiskCurve.prepare(np.ones(3), state, np.arange(3))
    got = curve.evaluate(np.ones(3), [0., .25, 1.])
    # Constant-score GR joins (0,0) to (1,error), hence area=error/2.
    assert got[1].tolist() == pytest.approx([.3125, .1875, .125])
    assert got[0, 0] == pytest.approx(.25)
    assert got[2].tolist() == pytest.approx([.5, 0., .5])
    bad_first = GeneralizedRiskCurve.prepare(np.array([1., 0.]), np.array([0, 2]), np.arange(2))
    assert bad_first.evaluate(np.ones(2), [.5])[0, 0] == pytest.approx(.375)


def test_weighted_draw_matches_expanded_records_including_sampled_out_top_group():
    scores = np.array([1., .8, .8, .3, .1])
    state = np.array([0, 2, 1, 2, 0])
    clusters = np.arange(5)
    weights = np.array([0, 2, 3, 1, 2])
    expanded = np.repeat(clusters, weights)
    a = GeneralizedRiskCurve.prepare(scores, state, clusters).evaluate(weights, PREVALENCES)
    b = GeneralizedRiskCurve.prepare(scores[expanded], state[expanded], np.arange(len(expanded))).evaluate(
        np.ones(len(expanded)), PREVALENCES
    )
    assert a == pytest.approx(b)
    assert a[:, 0] == pytest.approx(a[:, 1] + a[:, 2])


def test_independent_auc_identity_with_fractional_weights_and_ties():
    rng = np.random.default_rng(31)
    scores = rng.integers(0, 4, 24).astype(float)
    state = np.tile(np.array([0, 1, 2]), 8)
    weights = rng.uniform(.05, 3., len(state))
    weights[2] = 0.
    values = GeneralizedRiskCurve.prepare(scores, state, np.arange(len(state))).evaluate(weights, PREVALENCES)
    for index, pi in enumerate(PREVALENCES):
        masses = class_normalized_masses(state, weights, pi)
        assert values[index, 0] == pytest.approx(direct_identity(scores, state, masses), abs=1e-14)


def test_observed_prior_and_positive_endpoint_and_all_absent_endpoint():
    scores = np.array([.9, .8, .8, .4, .2])
    state = np.array([2, 1, 0, 2, 0])
    weights = np.ones(5)
    values = GeneralizedRiskCurve.prepare(scores, state, np.arange(5)).evaluate(weights, [0., .4, 1.])
    positive = state > 0
    assert values[0, 0] == pytest.approx(direct_identity(scores[positive], state[positive], weights[positive]))
    assert values[1, 0] == pytest.approx(direct_identity(scores, state, weights))
    assert values[2].tolist() == pytest.approx([.5, 0., .5])


def test_all_correct_positive_endpoint_remains_defined_without_failure_auc():
    curve = GeneralizedRiskCurve.prepare(np.array([.9, .8, .4]), np.array([2, 2, 0]), np.arange(3))
    assert curve.evaluate(np.ones(3), [0.])[0].tolist() == [0., 0., 0.]


def test_deterministic_paired_stratified_image_draws_and_seed_order():
    runs = {"17": fixture(), "42": list(reversed(fixture())), "73": fixture()}
    a = analyze_generalized_risk(runs, iterations=20, seed=7)
    assert a == analyze_generalized_risk(dict(reversed(list(runs.items()))), iterations=20, seed=7)
    assert a["bootstrap"]["same_draw_all_scores_seeds_prevalences"]
    assert a["bootstrap"]["strata"] == {"A": 2, "B": 2}
    assert a["evaluation_priors"]["observed"] == 1 / 3
    assert a["summary"]["observed"]["native"]["mixed_augrc"]["sample_sd"] is None
    for prior in a["summary"]:
        assert a["contrasts"][prior]["exists_minus_native"]["mixed_augrc"]["ci95"] == pytest.approx([0., 0.], abs=1e-15)
    json.dumps(a, allow_nan=False)


def test_bootstrap_matches_manual_whole_image_draws_not_expression_iid():
    rows = fixture()
    runs = {"17": rows, "42": list(reversed(rows))}
    got = analyze_generalized_risk(runs, iterations=25, seed=19)
    rng = np.random.Generator(np.random.PCG64(19))
    scores = np.array([r["candidate_score"] for r in rows])
    state = np.array([0 if r["kind"] != "positive" else (2 if r["correct"] else 1) for r in rows])
    manual = []
    for _ in range(25):
        # Deliberately independent of production _draw_cluster_weights.
        sampled = np.r_[rng.choice([0, 1], size=2, replace=True), rng.choice([2, 3], size=2, replace=True)]
        images = np.bincount(sampled, minlength=4)
        masses = class_normalized_masses(state, np.repeat(images, 3), .25)
        manual.append(direct_identity(scores, state, masses))
    assert got["summary"]["0.25"]["emit"]["mixed_augrc"]["ci95"] == pytest.approx(np.percentile(manual, [2.5, 97.5]))


def test_missing_class_draw_invalidates_affected_interval_without_discarding():
    rows = [fixture()[0], fixture()[-1]]
    rows[0]["stratum"] = rows[1]["stratum"] = "one"
    got = analyze_generalized_risk({"17": rows}, iterations=40, seed=3)
    item = got["summary"]["0.5"]["native"]["mixed_augrc"]
    assert item["mean"] is not None
    assert 0 < item["undefined_replicates"] < 40
    assert item["ci95"] is None
    assert got["contrasts"]["0.5"]["emit_minus_exists"]["mixed_augrc"]["ci95"] is None
    # Unlike a fixed mixed prior, the observed estimand remains defined when
    # an image draw happens to contain only one class.
    assert got["summary"]["observed"]["native"]["mixed_augrc"]["undefined_replicates"] == 0
    json.dumps(got, allow_nan=False)


def test_observed_bootstrap_retains_draw_prevalence_not_original_class_prior():
    rows = fixture()
    # Unequal positive/no-target composition per image makes class proportions
    # fluctuate within draws; retain both classes in each stratum overall.
    rows = [r for r in rows if not (r["cluster_id"] == "0" and r["kind"] == "no_target")]
    got = analyze_generalized_risk({"17": rows}, iterations=30, seed=13)
    scores = np.asarray([r["candidate_score"] for r in rows])
    state = np.asarray([0 if r["kind"] != "positive" else (2 if r["correct"] else 1) for r in rows])
    clusters = np.asarray([int(r["cluster_id"]) for r in rows])
    rng = np.random.Generator(np.random.PCG64(13))
    manual, normalized = [], []
    observed = np.mean(state == 0)
    for _ in range(30):
        sampled = np.r_[rng.choice([0, 1], 2, replace=True), rng.choice([2, 3], 2, replace=True)]
        masses = np.bincount(sampled, minlength=4)[clusters]
        manual.append(direct_identity(scores, state, masses))
        normalized.append(direct_identity(scores, state, class_normalized_masses(state, masses, observed)))
    assert got["summary"]["observed"]["emit"]["mixed_augrc"]["ci95"] == pytest.approx(np.percentile(manual, [2.5, 97.5]))
    assert not np.allclose(manual, normalized)


def test_single_class_supported_endpoint_and_missing_class_are_explicit():
    rows = [r for r in fixture() if r["kind"] == "no_target"]
    got = analyze_generalized_risk({"17": rows}, iterations=5)
    assert got["summary"]["1"]["native"]["mixed_augrc"]["mean"] == .5
    assert got["summary"]["observed"]["native"]["mixed_augrc"]["ci95"] == [.5, .5]
    unsupported = got["summary"]["0.5"]["native"]["mixed_augrc"]
    assert unsupported == {"mean": None, "sample_sd": None, "ci95": None, "undefined_replicates": 5}
    assert got["per_seed"]["17"]["native"]["0.5"] == dict.fromkeys(METRICS)


def test_contract_drift_and_invalid_controls_fail_closed():
    for kwargs in ({"iterations": True}, {"iterations": 0}, {"seed": True}, {"seed": -1}):
        with pytest.raises(ValueError):
            analyze_generalized_risk({"17": fixture()}, **kwargs)
    runs = {"17": fixture(), "42": copy.deepcopy(fixture())}
    runs["42"][0]["native_score"] += .1
    with pytest.raises(ValueError, match="Native parity"):
        analyze_generalized_risk(runs, iterations=1)
    curve = GeneralizedRiskCurve.prepare(np.array([.9, .3]), np.array([2, 0]), np.arange(2))
    for weights, prior in (([1., -1.], [.5]), ([1.], [.5]), ([1., 1.], [1.1])):
        with pytest.raises(ValueError):
            curve.evaluate(weights, prior)


def test_prepare_rejects_nonfinite_and_malformed_states():
    with pytest.raises(ValueError):
        GeneralizedRiskCurve.prepare(np.array([np.nan]), np.array([2]), np.array([0]))
    with pytest.raises(ValueError):
        GeneralizedRiskCurve.prepare(np.array([.1]), np.array([3]), np.array([0]))
    with pytest.raises(ValueError):
        GeneralizedRiskCurve.prepare(np.array([.1]), np.array([2]), np.array([-1]))
