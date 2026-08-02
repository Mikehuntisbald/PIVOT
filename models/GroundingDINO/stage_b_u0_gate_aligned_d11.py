"""Gap-2-eligible text-rank continuation for the data-only composite.

D11 keeps the sealed D9 patch gate, P50 confidence branch, U0 module, and b58
feature generator frozen.  It updates only the R100 rank output layer using
primary-instance Stage-B ground truth among the queries that the exact Gap-2
deployment gate admits.  Rows whose primary object is absent from the admitted
set are measured as patch-unreachable and deliberately produce no text-rank
gradient.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from groundingdino.util import box_ops


STAGE_B_U0_GATE_ALIGNED_D11_CONTRACT_VERSION = 11
STAGE_B_U0_GATE_ALIGNED_D11_MARKER = "stage_b_native_patch_category_d2"
STAGE_B_U0_GATE_ALIGNED_D11_LOSS = "loss_stage_b_u0_gate_aligned_d11"


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_targets(
    targets: Sequence[Mapping[str, Any]], *, batch_size: int
) -> None:
    if (
        not isinstance(targets, Sequence)
        or isinstance(targets, (str, bytes))
        or len(targets) != int(batch_size)
    ):
        raise ValueError("D11 targets must match the output batch")
    for row_index, target in enumerate(targets):
        if not isinstance(target, Mapping):
            raise ValueError(f"D11 target {row_index} must be a mapping")
        boxes = target.get("boxes")
        labels = target.get("labels")
        support_class = target.get("support_class")
        primary = target.get("primary_instance_mask")
        marker = target.get(STAGE_B_U0_GATE_ALIGNED_D11_MARKER)
        if (
            not torch.is_tensor(boxes)
            or not boxes.is_floating_point()
            or boxes.dim() != 2
            or tuple(boxes.shape[1:]) != (4,)
            or int(boxes.shape[0]) < 1
            or not bool(torch.isfinite(boxes).all().item())
        ):
            raise ValueError(f"D11 target {row_index} boxes must be finite (N,4)")
        instance_count = int(boxes.shape[0])
        if (
            not torch.is_tensor(labels)
            or labels.dtype != torch.int64
            or tuple(labels.shape) != (instance_count,)
        ):
            raise ValueError(f"D11 target {row_index} labels must be int64 (N,)")
        if (
            not torch.is_tensor(support_class)
            or support_class.dtype != torch.int64
            or support_class.numel() != 1
            or not bool((labels == support_class.reshape(-1)[0]).all().item())
        ):
            raise ValueError(
                f"D11 target {row_index} must retain every and only same-class GT"
            )
        if (
            not torch.is_tensor(primary)
            or primary.dtype != torch.bool
            or tuple(primary.shape) != (instance_count,)
            or int(primary.sum().item()) != 1
        ):
            raise ValueError(f"D11 target {row_index} requires one primary instance")
        if not (
            torch.is_tensor(marker)
            and marker.dtype == torch.bool
            and tuple(marker.shape) == (1,)
            and bool(marker[0].item())
        ):
            raise ValueError("D11 requires the exact D2 category-complete marker")


def _primary_candidate_iou(
    candidate_boxes: Tensor,
    targets: Sequence[Mapping[str, Any]],
) -> Tensor:
    candidate_xyxy = box_ops.box_cxcywh_to_xyxy(
        candidate_boxes.detach().float()
    )
    result = candidate_xyxy.new_zeros(candidate_xyxy.shape[:2])
    for row_index, target in enumerate(targets):
        primary_box = target["boxes"][target["primary_instance_mask"]]
        primary_xyxy = box_ops.box_cxcywh_to_xyxy(
            primary_box.detach().to(
                device=result.device, dtype=torch.float32
            )
        )
        iou, _ = box_ops.box_iou(candidate_xyxy[row_index], primary_xyxy)
        result[row_index] = iou[:, 0]
    return result


def _masked_mean(value: Tensor, mask: Tensor) -> Tensor:
    if bool(mask.any().item()):
        return value[mask].mean()
    return value.sum() * 0.0


class StageBU0GateAlignedD11Criterion(nn.Module):
    """Repair R100 ordering only inside the exact deployed Gap-2 set."""

    def __init__(
        self,
        *,
        weight: float = 1.0,
        positive_iou_threshold: float = 0.5,
        fix_margin: float = 0.05,
        preserve_margin: float = 0.02,
        temperature: float = 0.05,
        fix_weight: float = 1.0,
        preserve_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.weight = _finite_float(weight, name="D11 loss weight")
        self.positive_iou_threshold = _finite_float(
            positive_iou_threshold, name="D11 positive IoU threshold"
        )
        self.fix_margin = _finite_float(fix_margin, name="D11 fix margin")
        self.preserve_margin = _finite_float(
            preserve_margin, name="D11 preserve margin"
        )
        self.temperature = _finite_float(
            temperature, name="D11 temperature"
        )
        self.fix_weight = _finite_float(fix_weight, name="D11 fix weight")
        self.preserve_weight = _finite_float(
            preserve_weight, name="D11 preserve weight"
        )
        if (
            self.weight <= 0.0
            or not 0.0 < self.positive_iou_threshold <= 1.0
            or self.fix_margin <= 0.0
            or not 0.0 <= self.preserve_margin <= self.fix_margin
            or self.temperature <= 0.0
            or self.fix_weight <= 0.0
            or self.preserve_weight < 0.0
        ):
            raise ValueError("D11 criterion geometry is invalid")
        self.weight_dict = {STAGE_B_U0_GATE_ALIGNED_D11_LOSS: self.weight}
        self.register_buffer(
            "criterion_contract_version",
            torch.as_tensor(
                STAGE_B_U0_GATE_ALIGNED_D11_CONTRACT_VERSION,
                dtype=torch.int64,
            ),
            persistent=True,
        )

    def forward(
        self,
        outputs: Mapping[str, Any],
        targets: Sequence[Mapping[str, Any]],
    ) -> dict[str, Tensor]:
        if not isinstance(outputs, Mapping):
            raise ValueError("D11 outputs must be a mapping")
        rank_score = outputs.get("stage_b_gdino_rank_score")
        boxes = outputs.get("pred_boxes")
        eligible = outputs.get("stage_b_u0_category_gate_eligible_mask")
        if (
            not torch.is_tensor(rank_score)
            or not rank_score.is_floating_point()
            or rank_score.dim() != 2
            or not bool(torch.isfinite(rank_score).all().item())
        ):
            raise ValueError("D11 requires finite R100 rank scores with shape (B,Q)")
        if (
            not torch.is_tensor(boxes)
            or not boxes.is_floating_point()
            or boxes.dim() != 3
            or tuple(boxes.shape[:2]) != tuple(rank_score.shape)
            or int(boxes.shape[-1]) != 4
            or not bool(torch.isfinite(boxes).all().item())
        ):
            raise ValueError("D11 pred_boxes must be finite and rank-aligned")
        if (
            not torch.is_tensor(eligible)
            or eligible.dtype != torch.bool
            or tuple(eligible.shape) != tuple(rank_score.shape)
            or eligible.device != rank_score.device
            or bool((~eligible.any(dim=1)).any().item())
        ):
            raise ValueError("D11 requires the exact nonempty hard-gate mask")
        if boxes.device != rank_score.device:
            raise ValueError("D11 output tensors must share one device")

        batch_size = int(rank_score.shape[0])
        _validate_targets(targets, batch_size=batch_size)
        primary_iou = _primary_candidate_iou(boxes, targets)
        positive = eligible & (
            primary_iou >= self.positive_iou_threshold
        )
        negative = eligible & (~positive)
        reachable = positive.any(dim=1)
        valid = reachable & negative.any(dim=1)

        positive_best = rank_score.masked_fill(~positive, -torch.inf).amax(dim=1)
        negative_best = rank_score.masked_fill(~negative, -torch.inf).amax(dim=1)
        gap = torch.where(
            valid,
            positive_best - negative_best,
            torch.zeros_like(positive_best),
        )
        # The deployed winner is correct exactly when the best positive score
        # is above the best negative score.  Detaching only this routing choice
        # keeps the score margin fully differentiable.
        fix_rows = valid & (gap.detach() <= 0.0)
        preserve_rows = valid & (gap.detach() > 0.0)
        tau = self.temperature
        fix_row_loss = tau * F.softplus(
            (self.fix_margin - gap) / tau
        )
        preserve_row_loss = F.relu(self.preserve_margin - gap)
        fix_loss = _masked_mean(fix_row_loss, fix_rows)
        preserve_loss = _masked_mean(preserve_row_loss, preserve_rows)
        loss = (
            self.fix_weight * fix_loss
            + self.preserve_weight * preserve_loss
        )
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("D11 loss is not finite")

        metric = lambda value: rank_score.new_tensor(float(value)).detach()
        adapted_correct = valid & (gap.detach() > 0.0)
        return {
            STAGE_B_U0_GATE_ALIGNED_D11_LOSS: loss,
            "stage_b_u0_gate_aligned_d11_fix_loss": fix_loss.detach(),
            "stage_b_u0_gate_aligned_d11_preserve_loss": preserve_loss.detach(),
            "stage_b_u0_gate_aligned_d11_rows": metric(batch_size),
            "stage_b_u0_gate_aligned_d11_reachable_rows": metric(
                int(reachable.sum().item())
            ),
            "stage_b_u0_gate_aligned_d11_unreachable_rows": metric(
                int((~reachable).sum().item())
            ),
            "stage_b_u0_gate_aligned_d11_valid_rows": metric(
                int(valid.sum().item())
            ),
            "stage_b_u0_gate_aligned_d11_single_class_rows": metric(
                int((reachable & (~negative.any(dim=1))).sum().item())
            ),
            "stage_b_u0_gate_aligned_d11_fix_rows": metric(
                int(fix_rows.sum().item())
            ),
            "stage_b_u0_gate_aligned_d11_preserve_rows": metric(
                int(preserve_rows.sum().item())
            ),
            "stage_b_u0_gate_aligned_d11_adapted_correct_rows": metric(
                int(adapted_correct.sum().item())
            ),
            "stage_b_u0_gate_aligned_d11_gap_mean": (
                gap[valid].mean().detach() if bool(valid.any().item()) else loss.detach()
            ),
        }


__all__ = [
    "STAGE_B_U0_GATE_ALIGNED_D11_CONTRACT_VERSION",
    "STAGE_B_U0_GATE_ALIGNED_D11_LOSS",
    "STAGE_B_U0_GATE_ALIGNED_D11_MARKER",
    "StageBU0GateAlignedD11Criterion",
]
