import torch

from models.GroundingDINO.stage_b_fixed_text_criterion import (
    _fpr95_negative_softplus_loss,
)


def test_v58_active_set_uses_all_valid_count_normalization():
    negatives = torch.tensor([-1.0, 0.0, 0.5, 1.0], requires_grad=True)
    threshold = torch.tensor(0.5)
    loss, active, selected = _fpr95_negative_softplus_loss(
        negatives,
        surrogate_threshold=threshold,
        operating_threshold=threshold,
        temperature=0.1,
        margin=0.3,
        reduction_contract="exact_fpr95_active_set_all_count_mean_v2",
    )
    per_example = 0.1 * torch.nn.functional.softplus(
        (negatives - threshold + 0.3) / 0.1
    )
    expected = per_example[2:].sum() / 4.0
    torch.testing.assert_close(loss, expected)
    assert active.tolist() == [False, False, True, True]
    assert torch.equal(selected, active)

    loss.backward()
    assert negatives.grad is not None
    assert negatives.grad[:2].abs().sum().item() == 0.0
    assert torch.all(negatives.grad[2:] > 0.0)


def test_v58_does_not_amplify_each_active_gradient_when_active_set_shrinks():
    threshold = torch.tensor(0.5)
    many = torch.tensor([0.6, 0.7, 0.8, 0.9], requires_grad=True)
    few = torch.tensor([0.0, 0.1, 0.8, 0.9], requires_grad=True)

    many_loss, _, _ = _fpr95_negative_softplus_loss(
        many,
        surrogate_threshold=threshold,
        operating_threshold=threshold,
        temperature=0.1,
        margin=0.3,
        reduction_contract="exact_fpr95_active_set_all_count_mean_v2",
    )
    few_loss, _, _ = _fpr95_negative_softplus_loss(
        few,
        surrogate_threshold=threshold,
        operating_threshold=threshold,
        temperature=0.1,
        margin=0.3,
        reduction_contract="exact_fpr95_active_set_all_count_mean_v2",
    )
    many_loss.backward()
    few_loss.backward()
    torch.testing.assert_close(many.grad[2:], few.grad[2:])
    assert few.grad[:2].abs().sum().item() == 0.0
