from __future__ import annotations

import pytest
import torch

from engine import (
    _V56_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE_CONTRACT,
    _select_dense_duty_confidence_loss_logits,
    _select_dense_duty_positive_confidence_trust_logits,
    _select_dense_duty_sample_confidence_logits,
)


def test_v56_local_output_is_diagnostic_but_sample_route_is_deployed_global():
    rank = torch.zeros(2, 3, 2)
    diagnostic = torch.randn(2, 3, 2, requires_grad=False)
    deployed = torch.randn(2, 2, requires_grad=True)
    outputs = {
        "stage_b_dense_duty_confidence_base_logits": diagnostic,
        "stage_b_dense_duty_global_confidence_logits": deployed,
    }
    local_positive, local_tn = _select_dense_duty_confidence_loss_logits(
        outputs=outputs,
        candidate_logits=rank,
        confidence_logits=None,
        head_gradient_contract=_V56_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE_CONTRACT,
    )
    global_positive, global_tn = _select_dense_duty_sample_confidence_logits(
        outputs=outputs,
        candidate_logits=rank,
        head_gradient_contract=_V56_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE_CONTRACT,
    )
    assert local_positive.data_ptr() == diagnostic[..., 0].data_ptr()
    assert local_tn.data_ptr() == diagnostic[..., 1].data_ptr()
    assert global_positive.data_ptr() == deployed[..., 0].data_ptr()
    assert global_tn.data_ptr() == deployed[..., 1].data_ptr()


def test_v56_positive_q05_returns_true_deployed_tensor_not_pool_alias():
    rank = torch.zeros(2, 3, 2)
    pool = torch.tensor([[0.2, -0.3], [0.4, -0.5]], requires_grad=True)
    # A distinct tensor can be value-identical to the pool diagnostic. V56 must
    # return the sample-global object used by FPR95/inference.
    deployed_positive = pool[..., 0].clone()
    selected = _select_dense_duty_positive_confidence_trust_logits(
        outputs={"stage_b_dense_duty_confidence_pool_absolute_logits": pool},
        candidate_logits=rank,
        sample_positive_confidence_logits=deployed_positive,
        decoupled_confidence=True,
        positive_trust_contract="absolute_global_pool_logit_v4",
        head_gradient_contract=_V56_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE_CONTRACT,
    )
    assert selected.data_ptr() == deployed_positive.data_ptr()
    assert selected.data_ptr() != pool.data_ptr()


def test_v56_positive_q05_requires_and_matches_deployed_global():
    rank = torch.zeros(2, 3, 2)
    pool = torch.zeros(2, 2)
    with pytest.raises(RuntimeError, match="requires the true deployed"):
        _select_dense_duty_positive_confidence_trust_logits(
            outputs={"stage_b_dense_duty_confidence_pool_absolute_logits": pool},
            candidate_logits=rank,
            sample_positive_confidence_logits=None,
            decoupled_confidence=True,
            positive_trust_contract="absolute_global_pool_logit_v4",
            head_gradient_contract=_V56_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE_CONTRACT,
        )
    with pytest.raises(RuntimeError, match="changed the deployed confidence value"):
        _select_dense_duty_positive_confidence_trust_logits(
            outputs={"stage_b_dense_duty_confidence_pool_absolute_logits": pool},
            candidate_logits=rank,
            sample_positive_confidence_logits=torch.ones(2),
            decoupled_confidence=True,
            positive_trust_contract="absolute_global_pool_logit_v4",
            head_gradient_contract=_V56_DEPLOYMENT_OWNED_GLOBAL_ABSOLUTE_CONTRACT,
        )
