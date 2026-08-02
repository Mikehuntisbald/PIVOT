"""Frozen-R100, query-conditioned rank residual for the deployed Gap-2 gate."""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .stage_b_u0_gate_aligned_d11 import (
    _primary_candidate_iou,
    _validate_targets,
)


STAGE_B_U0_GATE_ALIGNED_D12_CONTRACT_VERSION = 12
STAGE_B_U0_GATE_ALIGNED_D12_LOSS = "loss_stage_b_u0_gate_aligned_d12"


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    if bool(mask.any().item()):
        return value[mask].mean()
    return value.sum() * 0.0


def _standardize(values: Tensor, mask: Tensor, *, clip: float = 5.0) -> Tensor:
    count = mask.sum(dim=1).clamp_min(1).float()
    safe = values.float().masked_fill(~mask, 0.0)
    mean = safe.sum(dim=1) / count
    centered = (values.float() - mean[:, None]).masked_fill(~mask, 0.0)
    std = (centered.square().sum(dim=1) / count).clamp_min(1e-6).sqrt()
    return (centered / std[:, None]).clamp(min=-float(clip), max=float(clip))


class StageBU0GateAlignedD12RankResidual(nn.Module):
    """Zero-initialized nonlinear residual over frozen R100 rank features."""

    def __init__(
        self,
        *,
        feature_dim: int = 128,
        hidden_dim: int = 64,
        residual_limit: float = 0.1,
    ) -> None:
        super().__init__()
        if int(feature_dim) <= 0 or int(hidden_dim) <= 0:
            raise ValueError("D12 feature dimensions must be positive")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.residual_limit = _finite_float(
            residual_limit, name="D12 residual limit"
        )
        if self.residual_limit <= 0.0:
            raise ValueError("D12 residual limit must be positive")
        self.feature_norm = nn.LayerNorm(self.feature_dim)
        self.trunk = nn.Sequential(
            nn.Linear(self.feature_dim + 1, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        # A bias would add the same scalar to every query and is therefore
        # unidentifiable under within-image ranking.
        self.output = nn.Linear(self.hidden_dim, 1, bias=False)
        nn.init.zeros_(self.output.weight)
        self.register_buffer(
            "contract_version",
            torch.as_tensor(
                STAGE_B_U0_GATE_ALIGNED_D12_CONTRACT_VERSION,
                dtype=torch.int64,
            ),
            persistent=True,
        )
        self.register_buffer(
            "contract_residual_limit",
            torch.as_tensor(self.residual_limit, dtype=torch.float32),
            persistent=True,
        )

    def trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.feature_norm.parameters()) + tuple(
            self.trunk.parameters()
        ) + tuple(self.output.parameters())

    def forward(
        self,
        rank_feature: Tensor,
        teacher_rank_score: Tensor,
        candidate_mask: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        if (
            rank_feature.dim() != 3
            or not rank_feature.is_floating_point()
            or int(rank_feature.shape[-1]) != self.feature_dim
        ):
            raise ValueError("D12 rank_feature must be floating (B,Q,D)")
        if (
            teacher_rank_score.dim() != 2
            or not teacher_rank_score.is_floating_point()
            or tuple(rank_feature.shape[:2]) != tuple(teacher_rank_score.shape)
        ):
            raise ValueError("D12 teacher score must align with rank features")
        if candidate_mask is None:
            mask = torch.ones_like(teacher_rank_score, dtype=torch.bool)
        else:
            mask = torch.as_tensor(
                candidate_mask,
                device=teacher_rank_score.device,
                dtype=torch.bool,
            )
        if (
            tuple(mask.shape) != tuple(teacher_rank_score.shape)
            or bool((~mask.any(dim=1)).any().item())
        ):
            raise ValueError("D12 candidate mask must be aligned and nonempty")
        feature = rank_feature.detach()
        teacher = teacher_rank_score.detach().to(device=feature.device)
        teacher_normalized = _standardize(teacher, mask).to(dtype=feature.dtype)
        inputs = torch.cat(
            (self.feature_norm(feature), teacher_normalized.unsqueeze(-1)),
            dim=-1,
        )
        raw = self.output(self.trunk(inputs)).squeeze(-1).float()
        limit = self.residual_limit
        residual = (limit * torch.tanh(raw / limit)).to(dtype=teacher.dtype)
        residual = residual.masked_fill(~mask, 0.0)
        rank_score = teacher + residual
        return {
            "teacher_rank_score": teacher,
            "rank_residual": residual,
            "rank_score": rank_score,
            "candidate_mask": mask,
        }


class StageBU0GateAlignedD12Criterion(nn.Module):
    """Fix frozen-R100 errors while preserving its correct deployed margins."""

    def __init__(
        self,
        *,
        weight: float = 1.0,
        positive_iou_threshold: float = 0.5,
        fix_margin: float = 0.05,
        preserve_tolerance: float = 0.01,
        preserve_floor: float = 0.005,
        temperature: float = 0.05,
        fix_weight: float = 1.0,
        preserve_weight: float = 1.0,
        residual_weight: float = 0.01,
    ) -> None:
        super().__init__()
        self.weight = _finite_float(weight, name="D12 loss weight")
        self.positive_iou_threshold = _finite_float(
            positive_iou_threshold, name="D12 positive IoU threshold"
        )
        self.fix_margin = _finite_float(fix_margin, name="D12 fix margin")
        self.preserve_tolerance = _finite_float(
            preserve_tolerance, name="D12 preserve tolerance"
        )
        self.preserve_floor = _finite_float(
            preserve_floor, name="D12 preserve floor"
        )
        self.temperature = _finite_float(
            temperature, name="D12 temperature"
        )
        self.fix_weight = _finite_float(fix_weight, name="D12 fix weight")
        self.preserve_weight = _finite_float(
            preserve_weight, name="D12 preserve weight"
        )
        self.residual_weight = _finite_float(
            residual_weight, name="D12 residual weight"
        )
        if (
            self.weight <= 0.0
            or not 0.0 < self.positive_iou_threshold <= 1.0
            or self.fix_margin <= 0.0
            or not 0.0 <= self.preserve_floor <= self.fix_margin
            or self.preserve_tolerance < 0.0
            or self.temperature <= 0.0
            or self.fix_weight <= 0.0
            or self.preserve_weight < 0.0
            or self.residual_weight < 0.0
        ):
            raise ValueError("D12 criterion geometry is invalid")
        self.weight_dict = {STAGE_B_U0_GATE_ALIGNED_D12_LOSS: self.weight}
        self.register_buffer(
            "criterion_contract_version",
            torch.as_tensor(
                STAGE_B_U0_GATE_ALIGNED_D12_CONTRACT_VERSION,
                dtype=torch.int64,
            ),
            persistent=True,
        )

    def forward(
        self,
        outputs: Mapping[str, Any],
        targets: Sequence[Mapping[str, Any]],
    ) -> dict[str, Tensor]:
        adapted = outputs.get("stage_b_u0_d12_rank_score")
        teacher = outputs.get("stage_b_u0_d12_teacher_rank_score")
        residual = outputs.get("stage_b_u0_d12_rank_residual")
        eligible = outputs.get("stage_b_u0_category_gate_eligible_mask")
        boxes = outputs.get("pred_boxes")
        if not all(torch.is_tensor(x) for x in (adapted, teacher, residual)):
            raise KeyError("D12 requires adapted, teacher, and residual rank scores")
        if (
            adapted.dim() != 2
            or not adapted.is_floating_point()
            or tuple(teacher.shape) != tuple(adapted.shape)
            or tuple(residual.shape) != tuple(adapted.shape)
            or not bool(torch.isfinite(adapted).all().item())
            or not bool(torch.isfinite(teacher).all().item())
            or not bool(torch.isfinite(residual).all().item())
        ):
            raise ValueError("D12 rank tensors must be finite aligned (B,Q)")
        if (
            not torch.is_tensor(eligible)
            or eligible.dtype != torch.bool
            or tuple(eligible.shape) != tuple(adapted.shape)
            or bool((~eligible.any(dim=1)).any().item())
        ):
            raise ValueError("D12 requires the exact nonempty hard-gate mask")
        if (
            not torch.is_tensor(boxes)
            or boxes.dim() != 3
            or tuple(boxes.shape[:2]) != tuple(adapted.shape)
            or int(boxes.shape[-1]) != 4
        ):
            raise ValueError("D12 boxes must align with rank scores")
        _validate_targets(targets, batch_size=int(adapted.shape[0]))

        primary_iou = _primary_candidate_iou(boxes, targets)
        positive = eligible & (
            primary_iou >= self.positive_iou_threshold
        )
        negative = eligible & (~positive)
        reachable = positive.any(dim=1)
        valid = reachable & negative.any(dim=1)

        def gap(score: Tensor) -> Tensor:
            pos = score.masked_fill(~positive, -torch.inf).amax(dim=1)
            neg = score.masked_fill(~negative, -torch.inf).amax(dim=1)
            return torch.where(valid, pos - neg, torch.zeros_like(pos))

        teacher_gap = gap(teacher.detach())
        adapted_gap = gap(adapted)
        teacher_correct = valid & (teacher_gap > 0.0)
        fix_rows = valid & (~teacher_correct)
        preserve_rows = teacher_correct

        tau = self.temperature
        fix_row_loss = tau * F.softplus(
            (self.fix_margin - adapted_gap) / tau
        )
        preserve_target = torch.maximum(
            teacher_gap - self.preserve_tolerance,
            teacher_gap.new_full(teacher_gap.shape, self.preserve_floor),
        )
        preserve_row_loss = F.relu(preserve_target - adapted_gap)
        fix_loss = _masked_mean(fix_row_loss, fix_rows)
        preserve_loss = _masked_mean(preserve_row_loss, preserve_rows)
        residual_l2 = residual[eligible].float().square().mean()
        loss = (
            self.fix_weight * fix_loss
            + self.preserve_weight * preserve_loss
            + self.residual_weight * residual_l2
        )
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("D12 loss is not finite")

        adapted_correct = valid & (adapted_gap.detach() > 0.0)
        metric = lambda value: adapted.new_tensor(float(value)).detach()
        return {
            STAGE_B_U0_GATE_ALIGNED_D12_LOSS: loss,
            "stage_b_u0_gate_aligned_d12_fix_loss": fix_loss.detach(),
            "stage_b_u0_gate_aligned_d12_preserve_loss": preserve_loss.detach(),
            "stage_b_u0_gate_aligned_d12_residual_l2": residual_l2.detach(),
            "stage_b_u0_gate_aligned_d12_rows": metric(int(adapted.shape[0])),
            "stage_b_u0_gate_aligned_d12_reachable_rows": metric(
                int(reachable.sum().item())
            ),
            "stage_b_u0_gate_aligned_d12_unreachable_rows": metric(
                int((~reachable).sum().item())
            ),
            "stage_b_u0_gate_aligned_d12_valid_rows": metric(int(valid.sum().item())),
            "stage_b_u0_gate_aligned_d12_fix_rows": metric(int(fix_rows.sum().item())),
            "stage_b_u0_gate_aligned_d12_preserve_rows": metric(
                int(preserve_rows.sum().item())
            ),
            "stage_b_u0_gate_aligned_d12_wrong_fixed": metric(
                int((fix_rows & adapted_correct).sum().item())
            ),
            "stage_b_u0_gate_aligned_d12_correct_regressed": metric(
                int((preserve_rows & (~adapted_correct)).sum().item())
            ),
            "stage_b_u0_gate_aligned_d12_teacher_gap_mean": (
                teacher_gap[valid].mean().detach()
                if bool(valid.any().item())
                else loss.detach()
            ),
            "stage_b_u0_gate_aligned_d12_adapted_gap_mean": (
                adapted_gap[valid].mean().detach()
                if bool(valid.any().item())
                else loss.detach()
            ),
        }


__all__ = [
    "STAGE_B_U0_GATE_ALIGNED_D12_CONTRACT_VERSION",
    "STAGE_B_U0_GATE_ALIGNED_D12_LOSS",
    "StageBU0GateAlignedD12Criterion",
    "StageBU0GateAlignedD12RankResidual",
]
