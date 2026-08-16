"""Post-category-gate rank residual for the U2-v2 deployment contract."""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .stage_b_u0_gate_aligned_d11 import _primary_candidate_iou, _validate_targets
from .stage_b_u0_gate_aligned_d12 import _masked_mean, _standardize


STAGE_B_U2V2_CONTRACT_VERSION = 1
STAGE_B_U2V2_LOSS = "loss_stage_b_u2v2_rank_residual"


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


class StageBU2V2RankResidual(nn.Module):
    """Seven-tensor, zero-init residual acting only on eligible queries."""

    def __init__(
        self, *, feature_dim: int = 128, hidden_dim: int = 64,
        residual_limit: float = 0.1,
    ) -> None:
        super().__init__()
        if int(feature_dim) <= 0 or int(hidden_dim) <= 0:
            raise ValueError("U2-v2 feature dimensions must be positive")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.residual_limit = _finite_float(
            residual_limit, name="U2-v2 residual limit"
        )
        if self.residual_limit <= 0.0:
            raise ValueError("U2-v2 residual limit must be positive")
        self.feature_norm = nn.LayerNorm(self.feature_dim)
        self.trunk = nn.Sequential(
            nn.Linear(self.feature_dim + 1, self.hidden_dim), nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.GELU(),
        )
        self.output = nn.Linear(self.hidden_dim, 1, bias=False)
        nn.init.zeros_(self.output.weight)
        self.register_buffer(
            "contract_version",
            torch.as_tensor(STAGE_B_U2V2_CONTRACT_VERSION, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "contract_residual_limit",
            torch.as_tensor(self.residual_limit, dtype=torch.float32),
            persistent=True,
        )

    def trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        return (
            tuple(self.feature_norm.parameters())
            + tuple(self.trunk.parameters())
            + tuple(self.output.parameters())
        )

    def forward(
        self, rank_feature: Tensor, teacher_rank_score: Tensor,
        eligible_mask: Optional[Tensor],
    ) -> dict[str, Tensor]:
        if (
            rank_feature.dim() != 3 or not rank_feature.is_floating_point()
            or int(rank_feature.shape[-1]) != self.feature_dim
        ):
            raise ValueError("U2-v2 rank_feature must be floating (B,Q,D)")
        if (
            teacher_rank_score.dim() != 2
            or not teacher_rank_score.is_floating_point()
            or tuple(rank_feature.shape[:2]) != tuple(teacher_rank_score.shape)
        ):
            raise ValueError("U2-v2 teacher score must align with rank features")
        if eligible_mask is None:
            raise ValueError("U2-v2 requires the deployed eligible mask")
        eligible = torch.as_tensor(
            eligible_mask, device=teacher_rank_score.device, dtype=torch.bool
        )
        if (
            tuple(eligible.shape) != tuple(teacher_rank_score.shape)
            or bool((~eligible.any(dim=1)).any().item())
        ):
            raise ValueError("U2-v2 eligible mask must be aligned and nonempty")
        feature = rank_feature.detach()
        teacher = teacher_rank_score.detach().to(device=feature.device)
        normalized = _standardize(teacher, eligible).to(dtype=feature.dtype)
        inputs = torch.cat(
            (self.feature_norm(feature), normalized.unsqueeze(-1)), dim=-1
        )
        raw = self.output(self.trunk(inputs)).squeeze(-1).float()
        limit = self.residual_limit
        residual = (limit * torch.tanh(raw / limit)).to(dtype=teacher.dtype)
        residual = residual.masked_fill(~eligible, 0.0)
        return {
            "teacher_rank_score": teacher,
            "rank_residual": residual,
            "pre_demotion_rank_score": teacher + residual,
            "eligible_mask": eligible,
        }


class StageBU2V2RankResidualCriterion(nn.Module):
    """Fix eligible teacher errors while preserving correct teacher margins."""

    def __init__(
        self, *, weight: float = 1.0, positive_iou_threshold: float = 0.5,
        fix_margin: float = 0.05, preserve_tolerance: float = 0.01,
        preserve_floor: float = 0.005, temperature: float = 0.05,
        fix_weight: float = 1.0, preserve_weight: float = 2.0,
        residual_weight: float = 0.05,
    ) -> None:
        super().__init__()
        values = {
            "weight": weight, "positive_iou_threshold": positive_iou_threshold,
            "fix_margin": fix_margin, "preserve_tolerance": preserve_tolerance,
            "preserve_floor": preserve_floor, "temperature": temperature,
            "fix_weight": fix_weight, "preserve_weight": preserve_weight,
            "residual_weight": residual_weight,
        }
        for name, value in values.items():
            setattr(self, name, _finite_float(value, name=f"U2-v2 {name}"))
        if (
            self.weight <= 0 or not 0 < self.positive_iou_threshold <= 1
            or self.fix_margin <= 0 or not 0 <= self.preserve_floor <= self.fix_margin
            or self.preserve_tolerance < 0 or self.temperature <= 0
            or self.fix_weight <= 0 or self.preserve_weight < 0
            or self.residual_weight < 0
        ):
            raise ValueError("U2-v2 criterion geometry is invalid")
        self.weight_dict = {STAGE_B_U2V2_LOSS: self.weight}

    def forward(
        self, outputs: Mapping[str, Any], targets: Sequence[Mapping[str, Any]]
    ) -> dict[str, Tensor]:
        adapted = outputs.get("stage_b_u2v2_rank_score")
        teacher = outputs.get("stage_b_u2v2_teacher_rank_score")
        residual = outputs.get("stage_b_u2v2_rank_residual")
        eligible = outputs.get("stage_b_u2v2_eligible_mask")
        boxes = outputs.get("pred_boxes")
        if not all(torch.is_tensor(x) for x in (adapted, teacher, residual, eligible, boxes)):
            raise KeyError("U2-v2 requires deployed rank, teacher, residual, mask, boxes")
        if (
            adapted.dim() != 2 or tuple(teacher.shape) != tuple(adapted.shape)
            or tuple(residual.shape) != tuple(adapted.shape)
            or eligible.dtype != torch.bool or tuple(eligible.shape) != tuple(adapted.shape)
            or bool((~eligible.any(dim=1)).any().item())
            or boxes.dim() != 3 or tuple(boxes.shape[:2]) != tuple(adapted.shape)
            or int(boxes.shape[-1]) != 4
        ):
            raise ValueError("U2-v2 tensors are not aligned")
        if not all(bool(torch.isfinite(x).all().item()) for x in (adapted, teacher, residual)):
            raise ValueError("U2-v2 rank tensors must be finite")
        _validate_targets(targets, batch_size=int(adapted.shape[0]))
        iou = _primary_candidate_iou(boxes, targets)
        positive = eligible & (iou >= self.positive_iou_threshold)
        negative = eligible & (~positive)
        reachable = positive.any(dim=1)
        valid = reachable & negative.any(dim=1)

        def gap(score: Tensor) -> Tensor:
            pos = score.masked_fill(~positive, -torch.inf).amax(dim=1)
            neg = score.masked_fill(~negative, -torch.inf).amax(dim=1)
            return torch.where(valid, pos - neg, torch.zeros_like(pos))

        teacher_gap = gap(teacher.detach())
        adapted_gap = gap(adapted)
        preserve_rows = valid & (teacher_gap > 0)
        fix_rows = valid & (~preserve_rows)
        tau = self.temperature
        fix = _masked_mean(
            tau * F.softplus((self.fix_margin - adapted_gap) / tau), fix_rows
        )
        preserve_target = torch.maximum(
            teacher_gap - self.preserve_tolerance,
            teacher_gap.new_full(teacher_gap.shape, self.preserve_floor),
        )
        preserve = _masked_mean(F.relu(preserve_target - adapted_gap), preserve_rows)
        l2 = residual[eligible].float().square().mean()
        loss = self.fix_weight * fix + self.preserve_weight * preserve + self.residual_weight * l2
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("U2-v2 loss is not finite")
        metric = lambda value: adapted.new_tensor(float(value)).detach()
        return {
            STAGE_B_U2V2_LOSS: loss,
            "stage_b_u2v2_fix_loss": fix.detach(),
            "stage_b_u2v2_preserve_loss": preserve.detach(),
            "stage_b_u2v2_residual_l2": l2.detach(),
            "stage_b_u2v2_valid_rows": metric(int(valid.sum().item())),
            "stage_b_u2v2_fix_rows": metric(int(fix_rows.sum().item())),
            "stage_b_u2v2_preserve_rows": metric(int(preserve_rows.sum().item())),
        }


__all__ = [
    "STAGE_B_U2V2_CONTRACT_VERSION", "STAGE_B_U2V2_LOSS",
    "StageBU2V2RankResidual", "StageBU2V2RankResidualCriterion",
]
