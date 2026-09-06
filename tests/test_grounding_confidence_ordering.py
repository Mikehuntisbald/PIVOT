import copy

import numpy as np
import pytest

from tools.grounding_confidence_ordering import (
    OrderingContractError, _GroupedMetric, _draw_cluster_weights, analyze_runs,
)


def row(sample, cluster, kind, correct, baseline, candidate, *, stratum="test", level=None):
    return dict(sample_id=sample, cluster_id=cluster, stratum=stratum, kind=kind,
                correct=correct, baseline_score=baseline, candidate_score=candidate, level=level)


def fixture():
    return [
        row("p1", "a", "positive", True, .9, .5, level=1),
        row("n1", "a", "text", None, .1, .5, level=1),
        row("p2", "b", "positive", False, .2, .5, level=2),
        row("n2", "b", "image", None, .3, .5, level=2),
        row("p3", "c", "positive", True, .7, .5, level=1),
        row("n3", "c", "no_target", None, .4, .5),
    ]


def test_manual_ties_aurc_and_fixed_labels():
    result = analyze_runs({"17": fixture(), "42": list(reversed(fixture()))}, iterations=30)
    surface = result["surfaces"]["all_positive"]
    baseline = surface["per_seed"]["17"]["baseline"]
    candidate = surface["per_seed"]["17"]["candidate"]
    assert baseline["correctness_auroc"]["value"] == 1
    assert candidate["correctness_auroc"]["value"] == .5
    assert candidate["existence_auroc_pooled"]["value"] == .5
    assert baseline["positive_aurc"]["value"] == pytest.approx(1 / 18)
    assert candidate["positive_aurc"]["value"] == pytest.approx(1 / 3)
    assert surface["contrasts"]["positive_p_at_1"]["candidate_minus_baseline"] == 0
    assert surface["contrasts"]["positive_p_at_1"]["ci95_percentile"] == [0, 0]
    assert result["surfaces"]["l1_positive"]["positive_records"] == 2
    assert result["surfaces"]["l1_positive"]["per_seed"]["17"]["baseline"]["existence_auroc_pooled"]["value"] == 1


@pytest.mark.parametrize("kind", ["auc", "aurc", "mean"])
def test_zero_weight_tie_groups_equal_expanded_sample(kind):
    scores = np.array([1., .8, .8, .5, .1])
    labels = np.array([1., 0., 1., 0., 1.])
    clusters = np.array([0, 1, 1, 2, 3])
    weights = np.array([0., 2., 1., 0.])
    metric = _GroupedMetric.prepare(kind, scores, labels, clusters)
    expanded_indices = np.repeat(np.arange(len(scores)), weights[clusters].astype(int))
    expanded = _GroupedMetric.prepare(kind, scores[expanded_indices], labels[expanded_indices], np.arange(len(expanded_indices)))
    got, reason = metric.evaluate(weights)
    expected, expected_reason = expanded.evaluate(np.ones(len(expanded_indices)))
    assert got == pytest.approx(expected)
    assert reason == expected_reason is None


def test_bootstrap_determinism_strata_and_cluster_counts():
    strata = {"a": np.array([0, 1]), "b": np.array([2, 3, 4])}
    first = np.random.Generator(np.random.PCG64(10))
    second = np.random.Generator(np.random.PCG64(10))
    for _ in range(20):
        draw = _draw_cluster_weights(first, strata, 5)
        assert np.array_equal(draw, _draw_cluster_weights(second, strata, 5))
        assert draw[:2].sum() == 2 and draw[2:].sum() == 3
    runs = {"17": fixture(), "73": fixture()}
    a = analyze_runs(runs, iterations=40, seed=99)
    b = analyze_runs(runs, iterations=40, seed=99)
    assert a == b
    assert a["bootstrap"]["unit"] == "image_cluster"
    assert "excludes training-seed uncertainty" in a["bootstrap"]["interval_scope"]


@pytest.mark.parametrize("field,value", [
    ("correct", False), ("cluster_id", "other"), ("stratum", "other"),
    ("sample_id", "other"), ("level", 3), ("kind", "text"),
])
def test_cross_seed_identity_and_label_drift_fail(field, value):
    left, right = fixture(), fixture()
    right[0][field] = value
    with pytest.raises(OrderingContractError):
        analyze_runs({"17": left, "42": right}, iterations=2)


@pytest.mark.parametrize("mutation", [
    lambda rows: rows.append(copy.deepcopy(rows[0])),
    lambda rows: rows[0].update(baseline_score=float("nan")),
    lambda rows: rows[0].update(candidate_score=True),
    lambda rows: rows[1].update(correct=False),
    lambda rows: rows[0].update(correct=1),
    lambda rows: rows[1].update(stratum="different"),
])
def test_invalid_rows_fail_closed(mutation):
    rows = fixture()
    mutation(rows)
    with pytest.raises(OrderingContractError):
        analyze_runs({"17": rows}, iterations=2)


def test_undefined_replicates_invalidate_whole_interval():
    rows = [row("a", "a", "positive", True, 1., 1.), row("b", "b", "positive", False, 0., 0.)]
    result = analyze_runs({"17": rows}, iterations=100, seed=3)
    surface = result["surfaces"]["all_positive"]
    correctness = surface["contrasts"]["correctness_auroc"]
    assert correctness["candidate_minus_baseline"] == 0
    assert 0 < correctness["undefined_replicates"] < 100
    assert correctness["ci95_percentile"] is None
    existence = surface["per_seed"]["17"]["baseline"]["existence_auroc_pooled"]
    assert existence["value"] is None and existence["reason"]
    assert surface["summary"]["baseline"]["positive_aurc"]["sample_sd"] is None
    assert "l1_positive" not in result["surfaces"]


def test_one_class_correctness_is_explicitly_undefined():
    rows = [row("a", "a", "positive", True, 1., 2., level=1),
            row("n", "a", "text", None, 0., 0.)]
    result = analyze_runs({"17": rows}, iterations=5)
    metric = result["surfaces"]["all_positive"]["contrasts"]["correctness_auroc"]
    assert metric["candidate_minus_baseline"] is None
    assert metric["undefined_replicates"] == 5
    assert metric["ci95_percentile"] is None
