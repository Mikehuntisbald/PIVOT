import pytest
import torch

from models.GroundingDINO.stage_b_fixed_text_criterion import (
    StageBFixedTextCriterion,
    _fpr95_negative_softplus_loss,
)


def _loss(
    negatives: torch.Tensor,
    *,
    threshold: torch.Tensor,
    contract: str = "exact_fpr95_active_set_mean_v1",
):
    return _fpr95_negative_softplus_loss(
        negatives,
        surrogate_threshold=threshold,
        operating_threshold=threshold.detach(),
        temperature=0.1,
        margin=0.3,
        reduction_contract=contract,
    )


def test_active_set_matches_exact_fpr95_ties_and_gradient_support():
    negatives = torch.tensor(
        [-0.4, -0.2, -0.1, -0.1, 0.3], requires_grad=True
    )
    threshold = torch.tensor(-0.1, requires_grad=True)

    loss, active, selected = _loss(negatives, threshold=threshold)

    expected_active = torch.tensor([False, False, True, True, True])
    assert torch.equal(active, expected_active)
    assert torch.equal(selected, expected_active)
    manual = 0.1 * torch.nn.functional.softplus(
        (negatives[expected_active] - threshold + 0.3) / 0.1
    ).mean()
    assert loss.item() == pytest.approx(manual.item())

    loss.backward()
    assert torch.equal(negatives.grad != 0.0, expected_active)
    assert threshold.grad is not None
    assert threshold.grad.item() < 0.0


def test_all_mean_preserves_legacy_denominator_and_gradients():
    negatives = torch.tensor([-0.4, -0.2, 0.1], requires_grad=True)
    threshold = torch.tensor(-0.1, requires_grad=True)

    loss, active, selected = _loss(
        negatives,
        threshold=threshold,
        contract="all_mean_v1",
    )

    assert torch.equal(active, torch.tensor([False, False, True]))
    assert bool(selected.all().item())
    manual = 0.1 * torch.nn.functional.softplus(
        (negatives - threshold + 0.3) / 0.1
    ).mean()
    assert loss.item() == pytest.approx(manual.item())
    loss.backward()
    assert bool((negatives.grad != 0.0).all().item())


def test_empty_active_set_is_exact_zero_with_live_graph():
    negatives = torch.tensor([-2.0, -1.0], requires_grad=True)
    threshold = torch.tensor(0.0, requires_grad=True)

    loss, active, selected = _loss(negatives, threshold=threshold)

    assert loss.item() == 0.0
    assert not bool(active.any().item())
    assert not bool(selected.any().item())
    loss.backward()
    assert torch.equal(negatives.grad, torch.zeros_like(negatives))
    assert threshold.grad is not None
    assert threshold.grad.item() == 0.0


@pytest.mark.parametrize("count", [1, 2, 7, 16])
def test_active_set_is_permutation_invariant_for_any_batch_size(count: int):
    negatives = torch.linspace(-0.5, 0.5, count, requires_grad=True)
    threshold = torch.tensor(-0.1, requires_grad=True)
    permutation = torch.arange(count - 1, -1, -1)

    loss, active, selected = _loss(negatives, threshold=threshold)
    permuted_loss, permuted_active, permuted_selected = _loss(
        negatives.detach()[permutation].requires_grad_(),
        threshold=threshold.detach().requires_grad_(),
    )

    assert permuted_loss.item() == pytest.approx(loss.item())
    assert torch.equal(permuted_active, active[permutation])
    assert torch.equal(permuted_selected, selected[permutation])


def test_unknown_negative_reduction_fails_closed():
    with pytest.raises(
        ValueError,
        match="tail_queue_negative_reduction_contract",
    ):
        StageBFixedTextCriterion(
            tail_queue_negative_reduction_contract="unknown"
        )
