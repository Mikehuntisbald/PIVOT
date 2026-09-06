"""V7 is an editorial/analytic reuse, never a new risk-bootstrap result."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "paper/scripts"))
import build_evidence_v7_assets as v7


def test_positive_pairwise_means_imply_no_interior_reversal():
    out = v7.analytic_l1(.8, .07, .002)
    assert out["interior_root"] is None
    assert out["status"] == "strictly_lower_mean_risk_for_0_le_pi_lt_1"
    for i in range(1000):
        pi = i/1000
        value = (1-pi)*((1-pi)*out["cw_coefficient"]+pi*out["cn_coefficient"])
        assert value < 0
    assert out["at_pi_one"] == 0


def test_opposing_pairwise_effects_can_have_an_internal_root():
    out = v7.analytic_l1(.8, .07, -.02)
    pi = out["interior_root"]
    assert 0 < pi < 1
    assert abs((1-pi)*out["cw_coefficient"]+pi*out["cn_coefficient"]) < 1e-15


def test_point_algebra_does_not_create_uncertainty_or_new_evaluation():
    out = v7.analytic_l1(.8, .07, .002)
    assert out["risk_ci95"] is None
    assert out["new_bootstrap_replicates"] == 0
    assert out["simultaneous_curve_guarantee"] is False
    assert out["role"] == "analytic_point_estimate_consequence"


@pytest.mark.parametrize("a", [0, 1, -1, 2])
def test_degenerate_accuracy_rejected(a):
    with pytest.raises(ValueError):
        v7.analytic_l1(a, .1, .1)


def test_zero_or_negative_changes_are_not_reported_as_improvement():
    assert v7.analytic_l1(.8, 0, 0)["status"] != "strictly_lower_mean_risk_for_0_le_pi_lt_1"
    assert v7.analytic_l1(.8, -.1, -.1)["status"] == "strictly_higher_mean_risk_for_0_le_pi_lt_1"


def test_sealed_sources_and_conditional_counts():
    data, bindings = v7.load_sources()
    f = v7.facts(data)
    assert len(bindings) == 3
    assert f["l1_population"] == {"C": 5506, "W": 591, "N": 9029, "positive": 6097}
    assert f["analytic_l1"]["interior_root"] is None
    assert f["new_risk_estimation"] is False
    assert f["new_model_forwards"] == f["new_training_updates"] == 0
    assert 7.084 < 100*f["l1_pairwise_effects"]["difficulty_cw_level1"]["mean"] < 7.086


def test_cross_readout_distinguishes_inference_change_and_retraining():
    data, _ = v7.load_sources()
    text = v7.snippets(data)
    assert "+1.608" in text["table_cross_readout.tex"]
    assert "-0.189" in text["table_cross_readout.tex"]
    assert "not continued from G" in text["table_cross_readout.tex"]
    assert "weighted contributions" in text["table_coverage.tex"]
    assert r"\newcommand{\LoneCW}{+7.085" in text["numbers.tex"]


def test_paper_centers_coverage_not_interaction_as_contribution():
    main = (ROOT / "paper/empirical_study_v7.tex").read_text()
    assert "supervision--readout\ninteraction and its boundaries" not in main
    assert "Supervision Coverage and Generalization" in main
    assert "no L1 risk bootstrap" in main
    assert main.index("Same Supervision, Opposite Risk Results") < main.index("Does Reading the Output Query Resolve It?")
    assert main.index("Which Comparisons Determine Complete Risk?") < main.index("Supervision Coverage and Generalization")
    assert "Re-reading is different from retraining" in main
