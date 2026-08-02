from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F

from models.GroundingDINO.stage_b_fixed_text_criterion import (
    StageBFixedTextCriterion,
)


def _criterion(**overrides) -> StageBFixedTextCriterion:
    values = {
        "listwise_weight": 0.0,
        "local_tn_rank_weight": 0.0,
        "predicate_tn_rank_weight": 0.0,
        "local_anchor_weight": 0.0,
        "global_tn_negative_weight": 0.0,
        "global_tn_tail_weight": 0.0,
        "batch_tail_separation_weight": 0.0,
        "local_absolute_weight": 0.0,
        "deployed_global_absolute_weight": 1.0,
        "deployed_global_absolute_gamma": 1.0,
        "predicate_absolute_weight": 0.0,
        "tail_queue_weight": 0.0,
    }
    values.update(overrides)
    return StageBFixedTextCriterion(**values)


def test_v57_balanced_absolute_value_and_gradient_use_only_sample_global_logits():
    criterion = _criterion()
    rank = torch.zeros(2, 3, requires_grad=True)
    iou = torch.tensor([[0.9, 0.8, 0.1], [0.7, 0.6, 0.2]])
    candidate_positive = torch.randn(2, 3, requires_grad=True)
    candidate_tn = torch.randn(2, 3, requires_grad=True)
    deployed_positive = torch.tensor([-0.4, 0.7], requires_grad=True)
    deployed_tn = torch.tensor([0.3, -0.8], requires_grad=True)

    losses = criterion(
        rank,
        iou,
        torch.ones_like(rank, dtype=torch.bool),
        local_tn_logits=rank,
        confidence_logits=candidate_positive,
        local_tn_confidence_logits=candidate_tn,
        local_tn_mask=torch.ones_like(rank, dtype=torch.bool),
        sample_positive_confidence_logits=deployed_positive,
        sample_tn_confidence_logits=deployed_tn,
    )

    positive_bce = F.binary_cross_entropy_with_logits(
        deployed_positive, torch.ones_like(deployed_positive), reduction="none"
    )
    positive_focal = positive_bce * (1.0 - deployed_positive.sigmoid())
    tn_bce = F.binary_cross_entropy_with_logits(
        deployed_tn, torch.zeros_like(deployed_tn), reduction="none"
    )
    tn_focal = tn_bce * deployed_tn.sigmoid()
    expected = 0.5 * positive_focal.mean() + 0.5 * tn_focal.mean()
    actual = losses["loss_fixed_text_deployed_global_absolute"]
    assert math.isclose(
        float(actual.detach()),
        float(expected.detach()),
        rel_tol=0.0,
        abs_tol=1e-7,
    )

    actual.backward()
    assert torch.all(deployed_positive.grad < 0)
    assert torch.all(deployed_tn.grad > 0)
    assert candidate_positive.grad is None
    assert candidate_tn.grad is None
    assert rank.grad is None
    assert int(
        losses[
            "fixed_text_deployed_global_absolute_positive_sample_count"
        ].item()
    ) == 2
    assert int(
        losses["fixed_text_deployed_global_absolute_tn_sample_count"].item()
    ) == 2


def test_v57_config_is_single_variable_over_v56_ownership_surface():
    from groundingdino.util.slconfig import SLConfig
    from tools.eval_refcoco_stageb import (
        _validate_v57_deployed_global_balanced_absolute_config,
    )

    repo = Path(__file__).resolve().parents[1]
    cfg = SLConfig.fromfile(
        str(
            repo
            / "config/ablations/"
            "cfg_stageb_dense_duty_confidence_adapter_deployed_global_"
            "balanced_absolute_probe_u0400_20260802.py"
        )
    )
    assert _validate_v57_deployed_global_balanced_absolute_config(cfg)
    assert cfg.stage_b_v14_local_absolute_weight == 0.0
    assert cfg.stage_b_dense_duty_deployed_global_absolute_weight == 1.0
    assert cfg.stage_b_dense_duty_deployed_global_absolute_gamma == 1.0
    assert cfg.stage_b_v11_trainable_params_min == 468_164
    assert cfg.stage_b_v11_trainable_params_max == 468_164
