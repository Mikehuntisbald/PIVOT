from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from engine import (
    _V53_FULLTEXT_GLOBAL_ABSOLUTE_CONTRACT,
    _select_dense_duty_confidence_loss_logits,
    _select_dense_duty_positive_confidence_trust_logits,
    _select_dense_duty_sample_confidence_logits,
)


def _candidate_logits(batch_size: int = 2) -> torch.Tensor:
    return torch.zeros(batch_size, 3, 2)


def _select_positive_trust(
    *,
    outputs,
    sample_positive,
    contract: str,
    candidate_logits: torch.Tensor | None = None,
):
    return _select_dense_duty_positive_confidence_trust_logits(
        outputs=outputs,
        candidate_logits=(
            _candidate_logits() if candidate_logits is None else candidate_logits
        ),
        sample_positive_confidence_logits=sample_positive,
        decoupled_confidence=True,
        positive_trust_contract=contract,
        head_gradient_contract=_V53_FULLTEXT_GLOBAL_ABSOLUTE_CONTRACT,
    )


def test_absolute_trust_contract_keeps_sample_absolute_route():
    sample_positive = torch.tensor([2.0, 3.0], requires_grad=True)
    confidence_delta = torch.tensor(
        [[-5.0, 7.0], [-6.0, 8.0]], requires_grad=True
    )

    selected = _select_positive_trust(
        outputs={
            "stage_b_dense_duty_confidence_delta_logits": confidence_delta,
        },
        sample_positive=sample_positive,
        contract="absolute_global_confidence_logit_v2",
    )

    assert selected is sample_positive


@pytest.mark.parametrize(
    "contract",
    (
        "net_total_confidence_delta_v1",
        "exact_frozen_rank_max_confidence_delta_v3",
    ),
)
def test_residual_trust_contract_uses_delta_slot_zero_even_with_sample_logits(
    contract,
):
    sample_positive = torch.tensor([20.0, 30.0], requires_grad=True)
    confidence_delta = torch.tensor(
        [[-0.5, 0.7], [-0.6, 0.8]], requires_grad=True
    )

    selected = _select_positive_trust(
        outputs={
            "stage_b_dense_duty_confidence_delta_logits": confidence_delta,
        },
        sample_positive=sample_positive,
        contract=contract,
    )

    assert torch.equal(selected, confidence_delta[:, 0])
    assert selected.data_ptr() == confidence_delta.data_ptr()
    assert not torch.equal(selected, sample_positive)


def test_residual_positive_trust_gradient_reaches_deployed_not_reference():
    deployed_global = torch.tensor(
        [[1.2, -0.3], [0.7, -1.1]], requires_grad=True
    )
    frozen_reference = torch.tensor(
        [[0.9, -0.1], [0.5, -0.8]], requires_grad=True
    )
    confidence_delta = deployed_global - frozen_reference.detach()

    selected = _select_positive_trust(
        outputs={
            "stage_b_dense_duty_confidence_delta_logits": confidence_delta,
        },
        sample_positive=deployed_global[:, 0],
        contract="exact_frozen_rank_max_confidence_delta_v3",
    )
    positive_trust_proxy = F.softplus(-selected).mean()
    positive_trust_proxy.backward()

    assert deployed_global.grad is not None
    assert torch.count_nonzero(deployed_global.grad[:, 0]).item() == 2
    assert torch.count_nonzero(deployed_global.grad[:, 1]).item() == 0
    assert frozen_reference.grad is None


def test_residual_trust_does_not_change_deployed_sample_or_tn_routes():
    candidate_rank = _candidate_logits()
    candidate_absolute = torch.randn(2, 3, 2, requires_grad=True)
    deployed_sample = torch.tensor(
        [[1.0, -2.0], [3.0, -4.0]], requires_grad=True
    )
    confidence_delta = torch.tensor(
        [[0.1, -0.2], [0.3, -0.4]], requires_grad=True
    )
    outputs = {
        "stage_b_dense_duty_confidence_base_logits": candidate_absolute,
        "stage_b_dense_duty_global_confidence_logits": deployed_sample,
        "stage_b_dense_duty_confidence_delta_logits": confidence_delta,
    }

    positive_local, tn_local = _select_dense_duty_confidence_loss_logits(
        outputs=outputs,
        candidate_logits=candidate_rank,
        confidence_logits=None,
        head_gradient_contract=_V53_FULLTEXT_GLOBAL_ABSOLUTE_CONTRACT,
    )
    sample_positive, sample_tn = _select_dense_duty_sample_confidence_logits(
        outputs=outputs,
        candidate_logits=candidate_rank,
        head_gradient_contract=_V53_FULLTEXT_GLOBAL_ABSOLUTE_CONTRACT,
    )
    positive_trust = _select_positive_trust(
        outputs=outputs,
        sample_positive=sample_positive,
        contract="exact_frozen_rank_max_confidence_delta_v3",
        candidate_logits=candidate_rank,
    )

    assert torch.equal(positive_local, candidate_absolute[..., 0])
    assert torch.equal(tn_local, candidate_absolute[..., 1])
    assert torch.equal(sample_positive, deployed_sample[:, 0])
    assert torch.equal(sample_tn, deployed_sample[:, 1])
    assert torch.equal(positive_trust, confidence_delta[:, 0])


@pytest.mark.parametrize(
    ("delta", "error"),
    (
        (None, "requires the exact confidence-delta output"),
        (torch.zeros(2, 1), "shape \\(2, 2\\)"),
        (torch.zeros(2, 3, 2), "shape \\(2, 2\\)"),
        (torch.zeros(2, 2, dtype=torch.int64), "floating tensor"),
        (
            torch.tensor([[0.0, float("nan")], [0.0, 0.0]]),
            "must be finite",
        ),
    ),
)
def test_exact_residual_trust_fails_closed_on_missing_or_drift(delta, error):
    outputs = {}
    if delta is not None:
        outputs["stage_b_dense_duty_confidence_delta_logits"] = delta

    with pytest.raises(RuntimeError, match=error):
        _select_positive_trust(
            outputs=outputs,
            sample_positive=torch.ones(2),
            contract="exact_frozen_rank_max_confidence_delta_v3",
        )


def test_positive_trust_fails_closed_on_unknown_contract_with_sample_logits():
    with pytest.raises(RuntimeError, match="unknown dense-duty positive trust"):
        _select_positive_trust(
            outputs={},
            sample_positive=torch.ones(2),
            contract="drifted_contract",
        )
