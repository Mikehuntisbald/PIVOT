import copy
import hashlib
import json

import numpy as np
import pytest

from tools import grounding_emission_audit as sealed_emission
from tools.confidence_readout_metrics import (
    CELLS, SEEDS, _Conditional, _effect_values, _state, add_combination_scores,
    analyze_readout, augrc_crossover, augrc_from_states, combine_scores,
    fit_sirc_statistics,
)
from tools.analyze_confidence_readout import load_manifest


def records():
    rows = []
    for image in range(4):
        for j, (kind, correct, native) in enumerate((
            ("positive", True, .9 - image * .1),
            ("positive", False, .35 + image * .1),
            ("text", None, .2 + image * .1),
        )):
            rows.append({
                "sample_id": f"{image}:{j}", "cluster_id": str(image),
                "stratum": "A" if image < 2 else "B", "kind": kind,
                "correct": correct, "level": (image % 2 + 1) if j < 2 else 99,
                "parent_positive_id": f"{image}:0" if j == 2 else None,
                "native_score": native,
                "scores": dict(zip(CELLS, (native, 1 - native, native + .1, 1.1 - native))),
            })
    return rows


def runs(second=True):
    a = {seed: copy.deepcopy(records()) for seed in SEEDS}
    result = {"mmgdino": a}
    if second:
        b = copy.deepcopy(a)
        for rows in b.values():
            for row in rows:
                if row["kind"] == "positive":
                    row["correct"] = not row["correct"]
        result["mdetr"] = b
    return result


def training():
    return [{"sample_id": str(i), "kind": "positive", "split": "train",
             "scores": {CELLS[0]: z}} for i, z in enumerate((-2., 0., 1., 3.))]


def test_train_only_sirc_statistics_rejects_duplicates_subset_and_evaluation():
    rows = training()
    stats = fit_sirc_statistics(rows, expected_count=4)
    assert stats["mean"] == .5
    assert stats["std_population"] == pytest.approx(np.std([-2., 0., 1., 3.], ddof=0))
    assert fit_sirc_statistics(list(reversed(rows)), 4) == stats
    with pytest.raises(ValueError, match="83341"):
        fit_sirc_statistics(rows)
    for field, value in (("split", "val"), ("kind", "text"), ("sample_id", "1")):
        bad = copy.deepcopy(rows)
        bad[0][field] = value
        with pytest.raises(ValueError):
            fit_sirc_statistics(bad, 4)


def test_joint_formulas_extremes_native_fallback_and_no_mutation():
    stats = fit_sirc_statistics(training(), 4)
    native, z = np.array([0., .2, .9, 1.]), np.array([-1000., -1., 1., 1000.])
    joint = combine_scores(native, z, stats)
    assert np.isfinite(joint["joint_product"]).all()
    assert np.isfinite(joint["joint_sirc"]).all()
    assert joint["joint_product"][2] == pytest.approx(np.log(.9) + np.log(1 / (1 + np.exp(-1))))
    assert joint["joint_sirc"][1] == pytest.approx(
        -np.log(.8) - np.logaddexp(0, -(-1 - stats["a"]) / stats["std_population"]))
    zero = training()
    for r in zero:
        r["scores"][CELLS[0]] = 4.
    zero_stats = fit_sirc_statistics(zero, 4)
    assert zero_stats["zero_variance_native_fallback"]
    assert np.array_equal(combine_scores(native, z, zero_stats)["joint_sirc"], native)
    rows = records()
    saved = copy.deepcopy(rows)
    augmented = add_combination_scores(rows, stats)
    assert rows == saved
    assert set(augmented[0]["scores"]) == set(CELLS) | {"joint_product", "joint_sirc"}
    with pytest.raises(ValueError, match="overwrite"):
        add_combination_scores(augmented, stats)
    with pytest.raises(ValueError, match="probabilities"):
        combine_scores([1.1], [0.], stats)


def test_two_localizers_share_draws_with_different_native_correctness():
    source = runs()
    result = analyze_readout(source, iterations=12, seed=7)
    reversed_source = {l: dict(reversed(list(s.items()))) for l, s in reversed(list(source.items()))}
    assert result == analyze_readout(reversed_source, iterations=12, seed=7)
    assert result["bootstrap"]["strata"] == {"A": 2, "B": 2}
    assert result["bootstrap"]["same_draw_all_localizers_heads_seeds"]
    assert result["bootstrap"]["unit"] == "image_cluster"
    assert result["localizers"]["mdetr"]["summary"]["native"]["correctness_auroc"]["mean"] != result["localizers"]["mmgdino"]["summary"]["native"]["correctness_auroc"]["mean"]
    assert result["localizers"]["mdetr"]["max_state_identity_error"] < 1e-14
    json.dumps(result, allow_nan=False)


def test_same_localizer_native_parity_and_cross_localizer_identity_are_required():
    for field, value in (("correct", False), ("native_score", .123), ("cluster_id", "drift")):
        source = runs()
        source["mmgdino"]["42"][0][field] = value
        with pytest.raises(ValueError, match="drift|identities"):
            analyze_readout(source, iterations=1)
    source = runs()
    del source["mmgdino"]["73"]
    with pytest.raises(ValueError, match="exact required seeds"):
        analyze_readout(source, iterations=1)


def test_interaction_does_not_substitute_for_emit_absolute_improvement():
    # Negative interaction produced ONLY by damaging exists.
    values = {CELLS[0]: {"r": .1}, CELLS[1]: {"r": .12},
              CELLS[2]: {"r": .2}, CELLS[3]: {"r": .12}}
    from tools.confidence_readout_metrics import EFFECTS
    effects = {k: _effect_values(values, c, "r") for k, c in EFFECTS.items()}
    assert effects["D_emit"] == 0
    assert effects["interaction"] == pytest.approx(-.1)
    assert effects["interaction"] == pytest.approx(effects["D_emit"] - effects["D_exists"])
    # Both targets benefit equally: no target-specific interaction.
    values[CELLS[2]]["r"], values[CELLS[3]]["r"] = .05, .07
    effects = {k: _effect_values(values, c, "r") for k, c in EFFECTS.items()}
    assert effects["D_emit"] == pytest.approx(-.05)
    assert effects["interaction"] == pytest.approx(0.)


def test_effect_outputs_keep_four_cells_and_exact_pairing():
    source = runs(False)
    result = analyze_readout(source, iterations=10, seed=1, conditionals=False)["localizers"]["mmgdino"]
    assert set(CELLS).issubset(result["summary"])
    e = result["effects"]
    assert e["interaction"]["mixed_augrc"]["mean"] == pytest.approx(
        e["D_emit"]["mixed_augrc"]["mean"] - e["D_exists"]["mixed_augrc"]["mean"])
    assert e["D_emit"]["mixed_augrc"]["ci95"] == pytest.approx([0., 0.], abs=1e-15)


def test_manual_stratified_image_bootstrap_matches_draw_hash_and_primary_interval():
    source = runs(False)
    result = analyze_readout(source, iterations=11, seed=23, conditionals=False)
    rng = np.random.Generator(np.random.PCG64(23))
    expected_hash, samples = hashlib.sha256(), []
    from tools.grounding_generalized_risk_audit import GeneralizedRiskCurve
    rows = source["mmgdino"]["17"]
    states = _state(rows)
    cluster = np.asarray([int(r["cluster_id"]) for r in rows])
    curve = GeneralizedRiskCurve.prepare(np.asarray([r["scores"][CELLS[0]] for r in rows]), states, cluster)
    for _ in range(11):
        picked = np.r_[rng.choice([0, 1], 2, replace=True), rng.choice([2, 3], 2, replace=True)]
        weights = np.bincount(picked, minlength=4)
        expected_hash.update(weights.astype("<u4").tobytes())
        masses = weights[cluster]
        pi = masses[states == 0].sum() / masses.sum()
        samples.append(curve.evaluate(weights, [pi])[0, 0])
    assert result["bootstrap"]["draws_sha256"] == expected_hash.hexdigest()
    interval = result["localizers"]["mmgdino"]["summary"][CELLS[0]]["mixed_augrc"]["ci95"]
    assert interval == pytest.approx(np.percentile(samples, [2.5, 97.5]))


def test_q05_is_recomputed_for_every_draw_and_ties_use_greater_equal(monkeypatch):
    calls = []
    original = sealed_emission.weighted_q05
    def traced(scores, weights):
        calls.append(weights.copy())
        return original(scores, weights)
    monkeypatch.setattr(sealed_emission, "weighted_q05", traced)
    source = runs(False)
    for rows in source["mmgdino"].values():
        for r in rows:
            r["scores"] = {k: 0. for k in CELLS}
    result = analyze_readout(source, iterations=5, seed=3, conditionals=False)
    assert len(calls) == 15 + 5 * 13  # 3 seeds*5 arms points; Native reused per draw.
    assert any(not np.array_equal(w, np.ones_like(w)) for w in calls)
    assert result["localizers"]["mmgdino"]["summary"][CELLS[0]]["diagnostic_fpr95"]["mean"] == 1.


def test_augrc_three_state_identity_and_interior_root():
    a, dcw, dcn = .75, .2, -.1
    root = augrc_crossover(a, dcw, dcn)
    assert root["prior"] == pytest.approx(1 / 3)
    for pi in (0., .25, .75, 1.):
        before = augrc_from_states(a, pi, .6, .7)
        after = augrc_from_states(a, pi, .8, .6)
        assert after - before == pytest.approx(-(1 - pi) * a * ((1 - pi) * (1 - a) * dcw + pi * dcn))
    assert augrc_crossover(a, .2, .1)["prior"] is None
    assert augrc_crossover(a, 0., 0.)["status"] == "identical_all_priors"
    assert augrc_crossover(a, np.nan, .1)["status"] == "undefined_state_auc"
    assert augrc_crossover(0., .2, -.1)["status"] == "no_correct_output_mass"


def test_root_absence_is_counted_not_clipped_or_silently_dropped():
    result = analyze_readout(runs(False), iterations=8, seed=9, conditionals=False)
    root = result["localizers"]["mmgdino"]["augrc_crossovers"]["D_emit"]
    assert root["point"]["prior"] is None
    assert sum(root["bootstrap_status_counts"].values()) == 8
    assert root["ci95"] is None


def test_conditional_image_copies_are_linear_not_squared_and_parent_pairs_align():
    rows = records()
    state = _state(rows)
    clusters = np.array([int(r["cluster_id"]) for r in rows])
    scores = np.array([r["native_score"] for r in rows])
    helper = _Conditional(rows, scores, state, clusters, 4)
    assert helper.counts["parent_pair_C"]["pairs"] == 4
    assert helper.counts["parent_pair_W"]["pairs"] == 0
    weights = np.array([3., 0., 1., 2.])
    got = helper.evaluate(weights)
    assert got["same_image_cw"] == pytest.approx(4 / 6)
    # Explicitly replicating image copies gives the same within-image estimand.
    numerator = sum(int(weights[i]) * float(scores[3*i] > scores[3*i+1]) for i in range(4))
    assert got["same_image_cw"] == pytest.approx(numerator / weights.sum())
    assert got["parent_pair_C"] == 1.


def test_negative_edit_levels_are_never_used_for_difficulty():
    rows = records()
    state = _state(rows)
    clusters = np.array([int(r["cluster_id"]) for r in rows])
    h = _Conditional(rows, np.array([r["native_score"] for r in rows]), state, clusters, 4)
    assert not any("level99" in key for key in h.counts)
    got = h.evaluate(np.ones(4))
    for pair in ("cw", "cn", "wn"):
        assert got[f"difficulty_{pair}_within_level_contribution"] + got[f"difficulty_{pair}_cross_level_contribution"] == pytest.approx(got[f"difficulty_{pair}_comparable_unconditional"])


def test_invalid_parent_pair_fail_closed():
    source = runs(False)
    for rows in source["mmgdino"].values():
        rows[2]["parent_positive_id"] = "1:0"
    with pytest.raises(ValueError, match="same image"):
        analyze_readout(source, iterations=1)


def test_cross_readout_diagnostics_have_matched_parity_and_winner_geometry():
    source = runs(False)
    for rows in source["mmgdino"].values():
        for r in rows:
            r["readout_diagnostics"] = {}
            for arm in CELLS:
                v = r["scores"][arm]
                glob = arm.startswith("global")
                r["readout_diagnostics"][arm] = {
                    "max_logit": v if glob else v + .2, "selected_logit": v - .2 if glob else v,
                    "confidence_winner_index": 9, "native_selected_index": 2,
                    "winner_native_box_iou": .4,
                    "native_gt_iou": None if r["kind"] != "positive" else .7 if r["correct"] else .3,
                    "winner_gt_iou": None if r["kind"] != "positive" else .8,
                }
    result = analyze_readout(source, iterations=3, seed=5, conditionals=False)
    assert len(result["cross_readout_scores"]) == 4
    geometry = result["localizers"]["mmgdino"]["winner_geometry"]["17"][CELLS[0]]["W"]
    assert geometry["confidence_winner_correct_fraction"] == 1.
    assert geometry["winner_differs_mean"] == 1.
    source["mmgdino"]["17"][0]["readout_diagnostics"][CELLS[0]]["max_logit"] += .01
    with pytest.raises(ValueError, match="matched score"):
        analyze_readout(source, iterations=1)


def test_manifest_requires_bound_protocol_and_record_sha(tmp_path, monkeypatch):
    from tools import analyze_confidence_readout as cli
    # Tiny synthetic surface only; production constants have official counts.
    monkeypatch.setitem(cli.SURFACES, "finecops_val", {"records": 12, "positive": 8, "no_target": 4})
    train = [{"sample_id": str(i), "kind": "positive", "split": "train",
              "scores": {CELLS[0]: float(i % 7)}} for i in range(83341)]
    statistics = fit_sirc_statistics(train)
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps(statistics))
    stats_binding = {"path": "stats.json", "sha256": hashlib.sha256(stats_path.read_bytes()).hexdigest()}
    rows = add_combination_scores(records(), statistics)
    for r in rows:
        r["readout_diagnostics"] = {"provided": True}  # Full shape is validated by analyze_readout.
    records_path = tmp_path / "rows.jsonl"
    records_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    sha = hashlib.sha256(records_path.read_bytes()).hexdigest()
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps({"schema": "arrow.confidence_readout.study_protocol/v1",
                                         "evaluation": {"surfaces": ["finecops_val"]}}))
    protocol_sha = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    path = tmp_path / "input.json"
    value = {"schema": "arrow.confidence_readout_analysis_input/v1", "protocol_sha256": protocol_sha,
             "protocol": {"path": "protocol.json", "sha256": protocol_sha}, "surface": "finecops_val",
             "expected_population": {"records": 12, "images": 4, "positive": 8, "no_target": 4},
             "sirc_statistics": {"mmgdino_positive": {s: stats_binding for s in SEEDS}},
             "runs": {"mmgdino_positive": {s: {"path": "rows.jsonl", "sha256": sha} for s in SEEDS}}}
    path.write_text(json.dumps(value))
    _, loaded, bindings = load_manifest(path, stage_mm_only=True)
    assert loaded["mmgdino_positive"]["17"] == rows
    assert bindings["mmgdino_positive"]["17"]["sha256"] == sha
    with pytest.raises(ValueError, match="both localizers"):
        load_manifest(path)
    saved = copy.deepcopy(value)
    value["expected_population"]["images"] = 5
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="population drift"):
        load_manifest(path, stage_mm_only=True)
    value = copy.deepcopy(saved)
    value["runs"]["mmgdino_positive"]["17"]["sha256"] = "b" * 64
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="SHA drift"):
        load_manifest(path, stage_mm_only=True)
    value = copy.deepcopy(saved)
    path.write_text(json.dumps(value))
    protocol_path.write_text("{}")
    with pytest.raises(ValueError, match="SHA drift"):
        load_manifest(path, stage_mm_only=True)


def test_missing_state_replicates_null_intervals_without_selection():
    source = runs(False)
    for seed in SEEDS:
        rows = [source["mmgdino"][seed][0], source["mmgdino"][seed][-1]]
        rows[1]["parent_positive_id"] = None
        for r in rows:
            r["stratum"] = "one"
        source["mmgdino"][seed] = rows
    result = analyze_readout(source, iterations=20, seed=4, conditionals=False)
    auc = result["localizers"]["mmgdino"]["summary"][CELLS[0]]["existence_auroc"]
    assert auc["ci95"] is None
    assert auc["undefined_replicates"] > 0


def test_crossover_scale_invariance_and_irrelevant_undefined_state():
    for scale in (1e-20, 1e-10, 1., 1e10):
        assert augrc_crossover(.75, .2 * scale, -.1 * scale)["prior"] == pytest.approx(1 / 3)
    assert augrc_crossover(0., np.nan, np.nan)["status"] == "no_correct_output_mass"
    assert augrc_crossover(1., np.nan, 0.)["status"] == "identical_all_priors"
    assert augrc_crossover(1., np.nan, .1)["status"] == "no_interior_root"


def test_tie_aware_augrc_identity_against_independent_pair_sum_stress():
    from tools.grounding_generalized_risk_audit import GeneralizedRiskCurve
    rng = np.random.default_rng(1081)
    for _ in range(30):
        states = np.tile([0, 1, 2], 4)
        scores = rng.integers(-2, 3, len(states)).astype(float)
        weights = rng.uniform(.1, 3., len(states))
        for pi in (0., .25, .5, .75, 1.):
            pos = states > 0
            masses = np.where(pos, weights * (1 - pi) / weights[pos].sum(), weights * pi / weights[~pos].sum())
            correct, failed = states == 2, states != 2
            correct_mass, failed_mass = masses[correct].sum(), masses[failed].sum()
            error_pair = 0.
            for i in np.flatnonzero(correct):
                for j in np.flatnonzero(failed):
                    error_pair += masses[i] * masses[j] * (float(scores[i] < scores[j]) + .5 * float(scores[i] == scores[j]))
            expected = error_pair + .5 * failed_mass ** 2
            got = GeneralizedRiskCurve.prepare(scores, states, np.arange(len(states))).evaluate(weights, [pi])[0, 0]
            assert got == pytest.approx(expected, abs=1e-14)


def test_same_image_attenuation_uses_comparable_surface_and_can_remain_unresolved():
    result = analyze_readout(runs(False), iterations=18, seed=19)
    effects = result["localizers"]["mmgdino"]["effects"]["global_emit_minus_exists"]
    delta = effects["same_image_cw_unconditional_minus_conditional"]
    assert delta["mean"] == pytest.approx(effects["same_image_cw_comparable_unconditional"]["mean"] - effects["same_image_cw"]["mean"])
    assert delta["ci95"] is not None
    assert result["interpretation"]["conditional_null_is_not_attribution"]
