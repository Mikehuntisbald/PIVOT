"""Pure-PyTorch objectives used by the single-process B32A1 head trainer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


def _graph_zero(tensor: Tensor) -> Tensor:
    finite = torch.where(torch.isfinite(tensor), tensor, torch.zeros_like(tensor))
    return (finite * 0.0).float().sum()


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    selected = mask.to(device=values.device, dtype=torch.bool)
    if not bool(selected.any().item()):
        return _graph_zero(values)
    return values[selected].mean()


@dataclass(frozen=True)
class BaselinePreservingRankOutput:
    loss: Tensor
    margin_loss: Tensor
    fix_loss: Tensor
    preserve_loss: Tensor
    residual_loss: Tensor
    valid_rows: Tensor
    fix_rows: Tensor
    preserve_rows: Tensor
    rows_no_positive: Tensor
    base_correct: Tensor
    adapted_correct: Tensor
    wrong_fixed: Tensor
    correct_regressed: Tensor


def baseline_preserving_top1_rank_loss(
    rank_score: Tensor,
    base_score: Tensor,
    rank_residual: Tensor,
    candidate_iou: Tensor,
    *,
    iou_threshold: float = 0.5,
    fix_margin: float = 0.05,
    preserve_margin: float = 0.02,
    temperature: float = 0.1,
    residual_weight: float = 1e-3,
) -> BaselinePreservingRankOutput:
    """Repair baseline top-1 failures while protecting baseline-correct rows."""
    if rank_score.dim() != 2 or not rank_score.is_floating_point():
        raise ValueError("rank_score must be floating point with shape (B,Q)")
    for name, value in (
        ("base_score", base_score),
        ("rank_residual", rank_residual),
        ("candidate_iou", candidate_iou),
    ):
        if tuple(value.shape) != tuple(rank_score.shape):
            raise ValueError(f"{name} must match rank_score")
    if not 0.0 < float(iou_threshold) <= 1.0:
        raise ValueError("iou_threshold must be in (0,1]")
    if float(fix_margin) < 0.0 or float(preserve_margin) < 0.0:
        raise ValueError("rank margins must be non-negative")
    if float(temperature) <= 0.0 or float(residual_weight) < 0.0:
        raise ValueError("rank temperature and residual weight are invalid")
    if not bool(torch.isfinite(rank_score).all().item()) or not bool(
        torch.isfinite(base_score).all().item()
    ):
        raise ValueError("rank and base scores must be finite")

    positive = candidate_iou >= float(iou_threshold)
    negative = candidate_iou < float(iou_threshold)
    valid = positive.any(dim=1) & negative.any(dim=1)
    rank_positive = rank_score.masked_fill(~positive, -torch.inf).max(dim=1).values
    rank_negative = rank_score.masked_fill(~negative, -torch.inf).max(dim=1).values
    base_positive = base_score.masked_fill(~positive, -torch.inf).max(dim=1).values
    base_negative = base_score.masked_fill(~negative, -torch.inf).max(dim=1).values
    rank_gap = torch.where(valid, rank_positive - rank_negative, torch.zeros_like(rank_positive))
    base_gap = torch.where(valid, base_positive - base_negative, torch.zeros_like(base_positive))

    base_top = base_score.argmax(dim=1)
    adapted_top = rank_score.argmax(dim=1)
    base_correct = positive.gather(1, base_top[:, None]).squeeze(1) & positive.any(dim=1)
    adapted_correct = positive.gather(1, adapted_top[:, None]).squeeze(1) & positive.any(dim=1)
    fix_rows = valid & (~base_correct)
    preserve_rows = valid & base_correct
    tau = float(temperature)
    fix_loss = _masked_mean(
        tau * F.softplus((float(fix_margin) - rank_gap) / tau), fix_rows
    )
    preserve_target = (base_gap.detach() - float(preserve_margin)).clamp_min(0.0)
    preserve_loss = _masked_mean(F.relu(preserve_target - rank_gap), preserve_rows)
    margin_loss = fix_loss + preserve_loss
    residual_loss = rank_residual.float().square().mean()
    loss = margin_loss + float(residual_weight) * residual_loss
    return BaselinePreservingRankOutput(
        loss=loss,
        margin_loss=margin_loss,
        fix_loss=fix_loss,
        preserve_loss=preserve_loss,
        residual_loss=residual_loss,
        valid_rows=valid.float().sum().detach(),
        fix_rows=fix_rows.float().sum().detach(),
        preserve_rows=preserve_rows.float().sum().detach(),
        rows_no_positive=(~positive.any(dim=1)).float().sum().detach(),
        base_correct=base_correct.float().sum().detach(),
        adapted_correct=adapted_correct.float().sum().detach(),
        wrong_fixed=((~base_correct) & adapted_correct).float().sum().detach(),
        correct_regressed=(base_correct & (~adapted_correct)).float().sum().detach(),
    )


def _validate_candidate_scores(
    scores: Tensor, candidate_mask: Optional[Tensor], *, name: str
) -> Tensor:
    if scores.dim() != 2 or not scores.is_floating_point() or not scores.numel():
        raise ValueError(f"{name} must be nonempty floating (B,Q)")
    mask = (
        torch.ones_like(scores, dtype=torch.bool)
        if candidate_mask is None
        else torch.as_tensor(candidate_mask, device=scores.device, dtype=torch.bool)
    )
    if tuple(mask.shape) != tuple(scores.shape) or bool((~mask.any(dim=1)).any().item()):
        raise ValueError(f"{name} candidate mask is invalid")
    if not bool(torch.isfinite(scores[mask]).all().item()):
        raise ValueError(f"valid {name} entries must be finite")
    return mask


def image_expression_global_max(
    candidate_score: Tensor,
    candidate_mask: Optional[Tensor] = None,
    *,
    name: str = "candidate_score",
) -> Tensor:
    mask = _validate_candidate_scores(candidate_score, candidate_mask, name=name)
    return candidate_score.float().masked_fill(~mask, -torch.inf).max(dim=1).values


def exact_tpr_operating_threshold(
    positive_global_score: Tensor, *, target_tpr: float = 0.95
) -> Tensor:
    score = positive_global_score.float().reshape(-1)
    if not score.numel() or not bool(torch.isfinite(score).all().item()):
        raise ValueError("positive scores must be nonempty and finite")
    if not 0.0 < float(target_tpr) <= 1.0:
        raise ValueError("target_tpr must be in (0,1]")
    accepted = max(1, int(math.ceil(float(target_tpr) * int(score.numel()))))
    return torch.sort(score, stable=True).values[int(score.numel()) - accepted]


@dataclass(frozen=True)
class DetachedRecentQ05TrustOutput:
    loss: Tensor
    negative_loss: Tensor
    positive_trust_loss: Tensor
    positive_threshold: Tensor
    current_positive_threshold: Tensor
    positive_global_score: Tensor
    negative_global_score: Tensor
    local_positive_global_score: Tensor
    local_negative_global_score: Tensor
    exact_tpr: Tensor
    exact_fpr: Tensor
    current_exact_tpr: Tensor
    current_exact_fpr: Tensor


def detached_recent_q05_trust_surrogate(
    positive_candidate_score: Tensor,
    negative_candidate_score: Tensor,
    positive_gate: Tensor,
    negative_gate: Tensor,
    *,
    positive_candidate_mask: Optional[Tensor] = None,
    negative_candidate_mask: Optional[Tensor] = None,
    positive_history: Optional[Tensor] = None,
    temperature: float = 0.1,
    margin: float = 0.0,
    target_tpr: float = 0.95,
    positive_trust_margin: float = 0.02,
    positive_trust_weight: float = 1.0,
    paired_margin_weight: float = 0.0,
    paired_margin: float = 0.0,
    positive_score_trust: bool = False,
) -> DetachedRecentQ05TrustOutput:
    """Suppress negative maxima against a detached recent positive q05."""
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    if min(float(positive_trust_margin), float(positive_trust_weight), float(paired_margin_weight)) < 0.0:
        raise ValueError("trust margins and weights must be non-negative")
    if not isinstance(positive_score_trust, bool):
        raise TypeError("positive_score_trust must be a bool")
    positive_global = image_expression_global_max(
        positive_candidate_score, positive_candidate_mask, name="positive_candidate_score"
    )
    negative_global = image_expression_global_max(
        negative_candidate_score, negative_candidate_mask, name="negative_candidate_score"
    )
    if positive_global.shape != negative_global.shape:
        raise ValueError("positive and negative score batches must align")
    for name, gate, expected in (
        ("positive_gate", positive_gate, positive_global),
        ("negative_gate", negative_gate, negative_global),
    ):
        if (
            not torch.is_tensor(gate)
            or gate.dim() != 1
            or not gate.is_floating_point()
            or tuple(gate.shape) != tuple(expected.shape)
            or gate.device != expected.device
            or not bool(torch.isfinite(gate).all().item())
        ):
            raise ValueError(f"{name} must be a finite floating (B,) tensor")

    current_threshold = exact_tpr_operating_threshold(
        positive_global, target_tpr=float(target_tpr)
    ).detach()
    threshold_bank = positive_global.detach()
    if positive_history is not None:
        threshold_bank = torch.as_tensor(
            positive_history, device=positive_global.device, dtype=torch.float32
        ).detach().reshape(-1)
        if not threshold_bank.numel() or not bool(torch.isfinite(threshold_bank).all().item()):
            raise ValueError("positive_history must be nonempty and finite")
    bank_threshold = exact_tpr_operating_threshold(
        threshold_bank, target_tpr=float(target_tpr)
    ).detach()
    translation = positive_global.mean() if positive_score_trust else positive_gate.mean()
    surrogate_threshold = bank_threshold + translation - translation.detach()
    tau = float(temperature)
    negative_loss = tau * F.softplus(
        (negative_global - surrogate_threshold + float(margin)) / tau
    ).mean()
    if positive_score_trust:
        trust_violation = F.relu(
            bank_threshold - float(positive_trust_margin) - positive_global
        )
    else:
        trust_violation = F.relu(-float(positive_trust_margin) - positive_gate)
    positive_trust_loss = trust_violation.mean()
    paired_loss = _graph_zero(negative_global)
    if float(paired_margin_weight) > 0.0:
        paired_loss = tau * F.softplus(
            (negative_global - positive_global + float(paired_margin)) / tau
        ).mean()
    loss = (
        negative_loss
        + float(positive_trust_weight) * positive_trust_loss
        + float(paired_margin_weight) * paired_loss
    )
    with torch.no_grad():
        exact_tpr = (threshold_bank >= bank_threshold).float().mean()
        exact_fpr = (negative_global >= bank_threshold).float().mean()
        current_exact_tpr = (positive_global >= current_threshold).float().mean()
        current_exact_fpr = (negative_global >= current_threshold).float().mean()
    return DetachedRecentQ05TrustOutput(
        loss=loss,
        negative_loss=negative_loss,
        positive_trust_loss=positive_trust_loss,
        positive_threshold=bank_threshold,
        current_positive_threshold=current_threshold,
        positive_global_score=positive_global,
        negative_global_score=negative_global,
        local_positive_global_score=positive_global,
        local_negative_global_score=negative_global,
        exact_tpr=exact_tpr,
        exact_fpr=exact_fpr,
        current_exact_tpr=current_exact_tpr,
        current_exact_fpr=current_exact_fpr,
    )


__all__ = [
    "BaselinePreservingRankOutput",
    "DetachedRecentQ05TrustOutput",
    "baseline_preserving_top1_rank_loss",
    "detached_recent_q05_trust_surrogate",
    "exact_tpr_operating_threshold",
    "image_expression_global_max",
]
