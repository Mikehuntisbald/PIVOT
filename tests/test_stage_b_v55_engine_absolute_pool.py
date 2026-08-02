from __future__ import annotations

import pytest
import torch

from engine import (
    _V55_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE_CONTRACT,
    _select_dense_duty_confidence_loss_logits,
    _select_dense_duty_positive_confidence_trust_logits,
    _select_dense_duty_sample_confidence_logits,
)


def _candidate_logits() -> torch.Tensor:
    return torch.zeros(2, 3, 2)


def test_v55_local_and_sample_losses_use_independent_declared_outputs():
    candidate_rank = _candidate_logits()
    candidate_absolute = torch.randn(2, 3, 2, requires_grad=True)
    global_absolute = torch.randn(2, 2, requires_grad=True)
    outputs = {
        "stage_b_dense_duty_confidence_base_logits": candidate_absolute,
        "stage_b_dense_duty_global_confidence_logits": global_absolute,
    }

    positive_local, tn_local = _select_dense_duty_confidence_loss_logits(
        outputs=outputs,
        candidate_logits=candidate_rank,
        confidence_logits=None,
        head_gradient_contract=_V55_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE_CONTRACT,
    )
    positive_global, tn_global = _select_dense_duty_sample_confidence_logits(
        outputs=outputs,
        candidate_logits=candidate_rank,
        head_gradient_contract=_V55_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE_CONTRACT,
    )

    assert torch.equal(positive_local, candidate_absolute[..., 0])
    assert torch.equal(tn_local, candidate_absolute[..., 1])
    assert torch.equal(positive_global, global_absolute[..., 0])
    assert torch.equal(tn_global, global_absolute[..., 1])


def test_v55_positive_trust_is_exact_deployed_pool_absolute_logit():
    candidate_rank = _candidate_logits()
    pool_absolute = torch.tensor(
        [[0.2, -0.3], [0.4, -0.5]], requires_grad=True
    )
    selected = _select_dense_duty_positive_confidence_trust_logits(
        outputs={
            "stage_b_dense_duty_confidence_pool_absolute_logits": pool_absolute,
        },
        candidate_logits=candidate_rank,
        sample_positive_confidence_logits=pool_absolute[..., 0],
        decoupled_confidence=True,
        positive_trust_contract="absolute_global_pool_logit_v4",
        head_gradient_contract=_V55_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE_CONTRACT,
    )

    assert torch.equal(selected, pool_absolute[..., 0])
    assert selected.data_ptr() == pool_absolute.data_ptr()


@pytest.mark.parametrize(
    ("pool", "error"),
    (
        (None, "requires the independent pool output"),
        (torch.zeros(2, 1), r"shape \(2, 2\)"),
        (torch.zeros(2, 2, dtype=torch.int64), "floating tensor"),
        (
            torch.tensor([[0.0, float("nan")], [0.0, 0.0]]),
            "must be finite",
        ),
    ),
)
def test_v55_positive_trust_fails_closed_on_missing_or_invalid_pool(pool, error):
    outputs = {}
    if pool is not None:
        outputs["stage_b_dense_duty_confidence_pool_absolute_logits"] = pool
    with pytest.raises(RuntimeError, match=error):
        _select_dense_duty_positive_confidence_trust_logits(
            outputs=outputs,
            candidate_logits=_candidate_logits(),
            sample_positive_confidence_logits=None,
            decoupled_confidence=True,
            positive_trust_contract="absolute_global_pool_logit_v4",
            head_gradient_contract=_V55_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE_CONTRACT,
        )


def test_v55_positive_trust_rejects_forward_drift_from_deployed_global():
    pool = torch.zeros(2, 2)
    with pytest.raises(RuntimeError, match="changed the deployed confidence value"):
        _select_dense_duty_positive_confidence_trust_logits(
            outputs={
                "stage_b_dense_duty_confidence_pool_absolute_logits": pool,
            },
            candidate_logits=_candidate_logits(),
            sample_positive_confidence_logits=torch.ones(2),
            decoupled_confidence=True,
            positive_trust_contract="absolute_global_pool_logit_v4",
            head_gradient_contract=_V55_LOCAL_CANDIDATE_GLOBAL_ABSOLUTE_CONTRACT,
        )
