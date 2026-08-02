from pathlib import Path

import pytest
import torch

from models.GroundingDINO.stage_b_fixed_text_criterion import (
    StageBFixedTextCriterion,
)


ROOT = Path(__file__).resolve().parents[1]


def _routing_criterion(**overrides) -> StageBFixedTextCriterion:
    options = {
        "listwise_weight": 0.0,
        "local_tn_rank_weight": 0.0,
        "predicate_tn_rank_weight": 0.0,
        "local_anchor_weight": 0.0,
        "global_tn_negative_weight": 0.0,
        "global_tn_tail_weight": 0.0,
        "batch_tail_separation_weight": 0.0,
        "local_absolute_weight": 0.0,
        "predicate_absolute_weight": 0.0,
        "tail_queue_weight": 0.0,
        "token_weight": 0.0,
        "raw_veto_gate_weight": 0.0,
        "raw_veto_carrier_pair_weight": 0.0,
        "deployed_veto_routing_weight": 1.0,
        "deployed_veto_positive_max": 0.1,
        "deployed_veto_tn_min": 0.9,
    }
    options.update(overrides)
    return StageBFixedTextCriterion(**options)


def _routing_inputs(
    *,
    batch_size: int = 1,
    num_candidates: int = 3,
) -> dict[str, torch.Tensor]:
    return {
        "candidate_logits": torch.zeros(
            batch_size, num_candidates, requires_grad=True
        ),
        "candidate_ious": torch.full((batch_size, num_candidates), 0.5),
        "candidate_mask": torch.ones(
            batch_size, num_candidates, dtype=torch.bool
        ),
        "local_tn_logits": torch.zeros(
            batch_size, num_candidates, requires_grad=True
        ),
        "local_tn_mask": torch.ones(
            batch_size, num_candidates, dtype=torch.bool
        ),
        "expression_valid_mask": torch.ones(batch_size, 2, dtype=torch.bool),
        "confidence_tn_train_eligible": torch.ones(
            batch_size, dtype=torch.bool
        ),
        "confidence_mismatch_gate": torch.full(
            (batch_size, num_candidates, 2), 0.5, requires_grad=True
        ),
        "confidence_veto_coverage": torch.full(
            (batch_size, 2), 0.5, requires_grad=True
        ),
        "confidence_base_logits": torch.zeros(
            batch_size, num_candidates, 2, requires_grad=True
        ),
    }


def test_deployed_routing_selects_masked_base_winner_not_frozen_carrier():
    criterion = _routing_criterion()
    inputs = _routing_inputs(batch_size=2)
    inputs["candidate_mask"] = torch.tensor(
        [[True, True, False], [True, True, True]]
    )
    inputs["local_tn_mask"] = torch.tensor(
        [[True, False, True], [True, True, True]]
    )
    inputs["expression_valid_mask"] = torch.tensor(
        [[True, True], [False, False]]
    )
    inputs["confidence_base_logits"] = torch.tensor(
        [
            [[0.0, 0.0], [10.0, 99.0], [99.0, 8.0]],
            [[3.0, 3.0], [2.0, 2.0], [1.0, 1.0]],
        ],
        requires_grad=True,
    )
    inputs["confidence_mismatch_gate"] = torch.tensor(
        [
            [[0.95, 0.05], [0.40, 0.10], [0.80, 0.60]],
            [[0.50, 0.50], [0.50, 0.50], [0.50, 0.50]],
        ],
        requires_grad=True,
    )
    inputs["confidence_veto_coverage"] = torch.tensor(
        [[0.30, 0.70], [0.50, 0.50]], requires_grad=True
    )
    # These deliberately disagree with the detached base-logit winners (1, 2).
    inputs["confidence_veto_carrier_indices"] = torch.tensor([[0, 0], [2, 2]])

    result = criterion(**inputs)

    assert result["loss_fixed_text_deployed_veto_routing"].item() == pytest.approx(
        0.25
    )
    assert result[
        "fixed_text_deployed_veto_routing_positive_sample_count"
    ].item() == 1
    assert result["fixed_text_deployed_veto_routing_tn_sample_count"].item() == 1
    assert result[
        "fixed_text_deployed_veto_routing_positive_winner_gate_mean"
    ].item() == pytest.approx(0.4)
    assert result[
        "fixed_text_deployed_veto_routing_tn_winner_gate_mean"
    ].item() == pytest.approx(0.6)
    assert result[
        "fixed_text_deployed_veto_routing_winner_loss_mean"
    ].item() == pytest.approx(0.3)
    assert result[
        "fixed_text_deployed_veto_routing_coverage_loss_mean"
    ].item() == pytest.approx(0.2)

    result["loss_fixed_text_deployed_veto_routing"].backward()
    gate_grad = inputs["confidence_mismatch_gate"].grad
    coverage_grad = inputs["confidence_veto_coverage"].grad
    assert gate_grad[0, 1, 0].item() == pytest.approx(0.25)
    assert gate_grad[0, 2, 1].item() == pytest.approx(-0.25)
    expected_gate_nonzero = torch.zeros_like(gate_grad, dtype=torch.bool)
    expected_gate_nonzero[0, 1, 0] = True
    expected_gate_nonzero[0, 2, 1] = True
    assert torch.equal(gate_grad != 0, expected_gate_nonzero)
    assert coverage_grad[0, 0].item() == pytest.approx(0.25)
    assert coverage_grad[0, 1].item() == pytest.approx(-0.25)
    assert torch.equal(
        coverage_grad != 0,
        torch.tensor([[True, True], [False, False]]),
    )
    assert inputs["confidence_base_logits"].grad is None


def test_deployed_routing_enforces_expression_eligibility_and_candidate_presence():
    criterion = _routing_criterion()
    inputs = _routing_inputs(batch_size=4)
    inputs["expression_valid_mask"] = torch.tensor(
        [[True, True], [True, True], [False, False], [True, True]]
    )
    inputs["confidence_tn_train_eligible"] = torch.tensor(
        [True, False, True, True]
    )
    inputs["candidate_mask"][3].zero_()
    inputs["local_tn_mask"][3].zero_()
    with torch.no_grad():
        inputs["confidence_mismatch_gate"][0, 0] = torch.tensor([0.2, 0.4])
        inputs["confidence_mismatch_gate"][1, 0] = torch.tensor([0.6, 0.2])
        inputs["confidence_veto_coverage"][0] = torch.tensor([0.3, 0.8])
        inputs["confidence_veto_coverage"][1] = torch.tensor([0.7, 0.2])

    result = criterion(**inputs)
    result["loss_fixed_text_deployed_veto_routing"].backward()

    assert result[
        "fixed_text_deployed_veto_routing_positive_sample_count"
    ].item() == 2
    assert result["fixed_text_deployed_veto_routing_tn_sample_count"].item() == 1
    # Winner: .5 * mean(.1, .5) + .5 * .5 = .4.
    assert result[
        "fixed_text_deployed_veto_routing_winner_loss_mean"
    ].item() == pytest.approx(0.4)
    # Coverage: .5 * mean(.2, .6) + .5 * .1 = .25; components average to .325.
    assert result[
        "fixed_text_deployed_veto_routing_coverage_loss_mean"
    ].item() == pytest.approx(0.25)
    assert result["loss_fixed_text_deployed_veto_routing"].item() == pytest.approx(
        0.325
    )
    gate_grad = inputs["confidence_mismatch_gate"].grad
    coverage_grad = inputs["confidence_veto_coverage"].grad
    assert gate_grad[0].abs().sum().item() > 0.0
    assert gate_grad[1, :, 0].abs().sum().item() > 0.0
    assert gate_grad[1, :, 1].abs().sum().item() == 0.0
    assert gate_grad[2:].abs().sum().item() == 0.0
    assert coverage_grad[0].abs().sum().item() > 0.0
    assert coverage_grad[1, 0].abs().item() > 0.0
    assert coverage_grad[1, 1].abs().item() == 0.0
    assert coverage_grad[2:].abs().sum().item() == 0.0


def test_deployed_routing_top_quarter_cvar_updates_only_hardest_samples():
    criterion = _routing_criterion(
        deployed_veto_routing_reduction_contract=(
            "balanced_top_quarter_cvar_v2"
        )
    )
    inputs = _routing_inputs(batch_size=8)
    positive_gate = torch.arange(0.2, 1.0, 0.1)
    tn_gate = torch.arange(0.8, 0.0, -0.1)
    mismatch_gate = torch.full((8, 3, 2), 0.5)
    mismatch_gate[:, 0, 0] = positive_gate
    mismatch_gate[:, 0, 1] = tn_gate
    inputs["confidence_mismatch_gate"] = mismatch_gate.requires_grad_()
    inputs["confidence_veto_coverage"] = torch.stack(
        (positive_gate, tn_gate), dim=-1
    ).requires_grad_()

    result = criterion(**inputs)

    assert result["loss_fixed_text_deployed_veto_routing"].item() == pytest.approx(
        0.75
    )
    result["loss_fixed_text_deployed_veto_routing"].backward()
    gate_grad = inputs["confidence_mismatch_gate"].grad[:, 0]
    coverage_grad = inputs["confidence_veto_coverage"].grad
    expected = torch.tensor([False] * 6 + [True, True])
    assert torch.equal(gate_grad[:, 0] != 0.0, expected)
    assert torch.equal(gate_grad[:, 1] != 0.0, expected)
    assert torch.equal(coverage_grad[:, 0] != 0.0, expected)
    assert torch.equal(coverage_grad[:, 1] != 0.0, expected)
    assert bool((gate_grad[expected, 0] > 0.0).all())
    assert bool((gate_grad[expected, 1] < 0.0).all())


def test_deployed_routing_is_bounded_zero_and_preserves_live_graph():
    criterion = _routing_criterion()
    inputs = _routing_inputs()
    inputs["confidence_mismatch_gate"] = torch.tensor(
        [[[0.1, 0.9], [0.0, 1.0], [0.05, 0.95]]], requires_grad=True
    )
    inputs["confidence_veto_coverage"] = torch.tensor(
        [[0.1, 0.9]], requires_grad=True
    )
    inputs["confidence_base_logits"] = torch.tensor(
        [[[3.0, 3.0], [2.0, 2.0], [1.0, 1.0]]], requires_grad=True
    )

    result = criterion(**inputs)
    loss = result["loss_fixed_text_deployed_veto_routing"]
    assert loss.item() == 0.0
    loss.backward()
    assert inputs["confidence_mismatch_gate"].grad is not None
    assert inputs["confidence_veto_coverage"].grad is not None
    assert inputs["confidence_mismatch_gate"].grad.abs().sum().item() == 0.0
    assert inputs["confidence_veto_coverage"].grad.abs().sum().item() == 0.0
    assert inputs["confidence_base_logits"].grad is None


def test_deployed_routing_bounds_float32_coverage_roundoff_with_live_gradient():
    inputs = _routing_inputs()
    one = torch.tensor(1.0, dtype=torch.float32)
    one_ulp_high = torch.nextafter(one, torch.tensor(float("inf")))
    inputs["confidence_mismatch_gate"] = torch.tensor(
        [[[0.1, 0.9], [0.1, 0.9], [0.1, 0.9]]], requires_grad=True
    )
    inputs["confidence_veto_coverage"] = torch.tensor(
        [[one_ulp_high, 0.9]], requires_grad=True
    )

    result = _routing_criterion()(**inputs)

    assert torch.isfinite(result["loss_fixed_text_deployed_veto_routing"])
    assert result[
        "fixed_text_deployed_veto_routing_positive_coverage_mean"
    ].item() == 1.0
    result["loss_fixed_text_deployed_veto_routing"].backward()
    assert inputs["confidence_veto_coverage"].grad[0, 0].item() > 0.0


def test_deployed_routing_rejects_coverage_beyond_roundoff_tolerance():
    inputs = _routing_inputs()
    eps = torch.finfo(torch.float32).eps
    inputs["confidence_veto_coverage"] = torch.tensor([[0.0, 1.0 + 9.0 * eps]])

    with pytest.raises(ValueError, match="reduction tolerance"):
        _routing_criterion()(**inputs)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"deployed_veto_routing_weight": -0.1}, "routing_weight"),
        ({"deployed_veto_routing_weight": float("inf")}, "routing_weight"),
        ({"deployed_veto_positive_max": -0.1}, "positive_max"),
        ({"deployed_veto_tn_min": 1.1}, "tn_min"),
        (
            {"deployed_veto_positive_max": 0.9, "deployed_veto_tn_min": 0.9},
            "must be below",
        ),
        (
            {"deployed_veto_routing_reduction_contract": "unknown"},
            "reduction_contract",
        ),
    ),
)
def test_deployed_routing_constructor_rejects_invalid_contracts(overrides, message):
    with pytest.raises(ValueError, match=message):
        _routing_criterion(**overrides)


@pytest.mark.parametrize(
    ("name", "value", "error", "message"),
    (
        ("confidence_mismatch_gate", None, ValueError, "requires"),
        (
            "confidence_mismatch_gate",
            torch.zeros(1, 3, 2, dtype=torch.long),
            TypeError,
            "floating",
        ),
        (
            "confidence_mismatch_gate",
            torch.zeros(1, 3, 1),
            ValueError,
            "shape",
        ),
        (
            "confidence_mismatch_gate",
            torch.tensor([[[float("nan"), 0.0]]]).expand(1, 3, 2),
            ValueError,
            "finite",
        ),
        (
            "confidence_mismatch_gate",
            torch.full((1, 3, 2), 1.01),
            ValueError,
            "bounded",
        ),
        (
            "confidence_veto_coverage",
            torch.zeros(1, 1),
            ValueError,
            "shape",
        ),
        (
            "confidence_veto_coverage",
            torch.tensor([[0.0, -0.01]]),
            ValueError,
            "bounded",
        ),
        (
            "confidence_base_logits",
            torch.full((1, 3, 2), float("inf")),
            ValueError,
            "finite",
        ),
        (
            "candidate_mask",
            torch.ones(1, 1, dtype=torch.bool),
            ValueError,
            "exact shape",
        ),
        (
            "expression_valid_mask",
            torch.ones(1, dtype=torch.bool),
            ValueError,
            "exact shape",
        ),
        (
            "confidence_tn_train_eligible",
            torch.ones(1, 1, dtype=torch.bool),
            ValueError,
            "exact shape",
        ),
    ),
)
def test_deployed_routing_forward_rejects_invalid_contracts(
    name, value, error, message
):
    inputs = _routing_inputs()
    inputs[name] = value
    with pytest.raises(error, match=message):
        _routing_criterion()(**inputs)


def test_deployed_routing_rejects_cross_device_inputs():
    inputs = _routing_inputs()
    inputs["confidence_base_logits"] = torch.empty((1, 3, 2), device="meta")
    with pytest.raises(ValueError, match="device"):
        _routing_criterion()(**inputs)


def test_v43_model_and_engine_wiring_names_are_exact():
    model_source = (ROOT / "models/GroundingDINO/groundingdino.py").read_text()
    engine_source = (ROOT / "engine.py").read_text()
    for config_name in (
        "stage_b_dense_duty_deployed_veto_routing_weight",
        "stage_b_dense_duty_deployed_veto_positive_max",
        "stage_b_dense_duty_deployed_veto_tn_min",
    ):
        assert config_name in model_source
    for output_name in (
        "stage_b_dense_duty_confidence_mismatch_gate",
        "stage_b_dense_duty_confidence_veto_coverage",
        "stage_b_dense_duty_confidence_base_logits",
    ):
        assert output_name in engine_source
