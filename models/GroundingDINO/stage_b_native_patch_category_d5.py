"""Active-tail positive protection for native patch-category training.

D5 keeps D3's deployment-critical negative-winner correction unchanged.  It
replaces the positive-winner loss averaged over every positive row with a
thresholded active mean: only positive native winners whose detached patch gap
has entered the warning band receive the positive barrier.  This concentrates
the protection budget on rows that can approach the deployment Gap-3 cutoff
without suppressing critical-negative learning on every batch.

Native text scores and boxes are detached selectors.  Only the patch branch
can receive gradients.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from groundingdino.util import box_ops

from .stage_b_gdino_score_adapter import aggregate_gdino_full_expression_score
from .stage_b_native_patch_category_d2 import (
    _expression_mask,
    _finite_float,
    _validate_targets,
    gate_aligned_standardized_patch_score,
)
from .stage_b_native_patch_category_d3 import NATIVE_PATCH_CATEGORY_D3_MARKER


NATIVE_PATCH_CATEGORY_D5_CONTRACT_VERSION = 5
# D5 consumes the same audited category-complete examples as D2-D4.
NATIVE_PATCH_CATEGORY_D5_MARKER = NATIVE_PATCH_CATEGORY_D3_MARKER
NATIVE_PATCH_CATEGORY_D5_LOSS = "loss_stage_b_native_patch_category_d5"


class StageBNativePatchCategoryD5Criterion(nn.Module):
    """Correct critical negatives while protecting only at-risk positives."""

    def __init__(
        self,
        *,
        weight: float = 1.0,
        positive_iou_threshold: float = 0.5,
        negative_iou_threshold: float = 0.3,
        gate_max_gap: float = 3.0,
        patch_score_clip: float = 5.0,
        keep_gap: float = 2.75,
        separation_gap: float = 3.25,
        temperature: float = 0.25,
        critical_weight: float = 2.0,
        critical_keep_weight: float = 1.0,
        active_gap: float = 2.0,
        target_gap: float = 2.5,
        positive_barrier_weight: float = 2.0,
    ) -> None:
        super().__init__()
        self.weight = _finite_float(weight, name="D5 loss weight")
        self.positive_iou_threshold = _finite_float(
            positive_iou_threshold, name="D5 positive IoU threshold"
        )
        self.negative_iou_threshold = _finite_float(
            negative_iou_threshold, name="D5 negative IoU threshold"
        )
        self.gate_max_gap = _finite_float(
            gate_max_gap, name="D5 gate max gap"
        )
        self.patch_score_clip = _finite_float(
            patch_score_clip, name="D5 patch-score clip"
        )
        self.keep_gap = _finite_float(keep_gap, name="D5 critical keep gap")
        self.separation_gap = _finite_float(
            separation_gap, name="D5 separation gap"
        )
        self.temperature = _finite_float(
            temperature, name="D5 temperature"
        )
        self.critical_weight = _finite_float(
            critical_weight, name="D5 critical weight"
        )
        self.critical_keep_weight = _finite_float(
            critical_keep_weight, name="D5 critical keep weight"
        )
        self.active_gap = _finite_float(
            active_gap, name="D5 positive active gap"
        )
        self.target_gap = _finite_float(
            target_gap, name="D5 positive target gap"
        )
        self.positive_barrier_weight = _finite_float(
            positive_barrier_weight, name="D5 positive barrier weight"
        )
        if (
            self.weight <= 0.0
            or not 0.0
            <= self.negative_iou_threshold
            < self.positive_iou_threshold
            <= 1.0
            or self.gate_max_gap <= 0.0
            or self.patch_score_clip <= self.gate_max_gap
            or not 0.0
            <= self.keep_gap
            < self.gate_max_gap
            < self.separation_gap
            < 2.0 * self.patch_score_clip
            or not 0.0
            <= self.active_gap
            < self.target_gap
            < self.gate_max_gap
            or self.temperature <= 0.0
            or min(self.critical_weight, self.critical_keep_weight) < 0.0
            or self.critical_weight + self.critical_keep_weight <= 0.0
            or self.positive_barrier_weight <= 0.0
        ):
            raise ValueError("D5 criterion geometry is invalid")
        self.weight_dict = {NATIVE_PATCH_CATEGORY_D5_LOSS: self.weight}

    def forward(
        self,
        outputs: Mapping[str, Any],
        targets: Sequence[Mapping[str, Any]],
        cap_list: Optional[Sequence[Sequence[str]]] = None,
        captions: Optional[Sequence[str]] = None,
    ) -> dict[str, Tensor]:
        del cap_list, captions
        if not isinstance(outputs, Mapping):
            raise ValueError("D5 outputs must be a mapping")
        patch_score = outputs.get("pred_logits_patch")
        if torch.is_tensor(patch_score) and patch_score.dim() == 3:
            if int(patch_score.shape[-1]) != 1:
                raise ValueError("D5 requires exactly one support-patch slot")
            patch_score = patch_score[..., 0]
        if (
            not torch.is_tensor(patch_score)
            or not patch_score.is_floating_point()
            or patch_score.dim() != 2
            or not bool(torch.isfinite(patch_score).all().item())
        ):
            raise ValueError("D5 pred_logits_patch must be finite floating (B,Q)")

        boxes = outputs.get("pred_boxes")
        token_logits = outputs.get("pred_logits_text")
        if (
            not torch.is_tensor(boxes)
            or not boxes.is_floating_point()
            or boxes.dim() != 3
            or tuple(boxes.shape[:2]) != tuple(patch_score.shape)
            or int(boxes.shape[-1]) != 4
            or not bool(torch.isfinite(boxes).all().item())
        ):
            raise ValueError("D5 pred_boxes must be finite and patch-aligned")
        if (
            not torch.is_tensor(token_logits)
            or not token_logits.is_floating_point()
            or token_logits.dim() != 3
            or tuple(token_logits.shape[:2]) != tuple(patch_score.shape)
        ):
            raise ValueError("D5 pred_logits_text must be floating and patch-aligned")
        if len({patch_score.device, boxes.device, token_logits.device}) != 1:
            raise ValueError("D5 output tensors must share one device")

        batch_size, query_count = patch_score.shape
        _validate_targets(targets, batch_size=int(batch_size))
        expression_mask = _expression_mask(outputs, token_logits)
        scored_text_mask = expression_mask[:, None, :].expand_as(token_logits)
        if not bool(torch.isfinite(token_logits[scored_text_mask]).all().item()):
            raise ValueError("D5 full-expression token logits must be finite")
        native_score = aggregate_gdino_full_expression_score(
            token_logits, expression_mask
        ).detach()
        if not bool(torch.isfinite(native_score).all().item()):
            raise RuntimeError("D5 native full-expression score is not finite")

        candidate_mask = torch.ones(
            (batch_size, query_count), dtype=torch.bool, device=patch_score.device
        )
        standardized = gate_aligned_standardized_patch_score(
            patch_score, candidate_mask, clip=self.patch_score_clip
        )
        row_best = standardized.amax(dim=1).detach()

        critical_separation_losses: list[Tensor] = []
        critical_keep_losses: list[Tensor] = []
        positive_barrier_losses: list[Tensor] = []
        separation_compliant = 0
        q_p_keep = 0
        positive_native_rows = 0
        positive_active_rows = 0
        positive_deployment_keep = 0
        positive_target_keep = 0

        candidates_xyxy = box_ops.box_cxcywh_to_xyxy(boxes.detach().float())
        for row_index, target in enumerate(targets):
            target_boxes = box_ops.box_cxcywh_to_xyxy(
                target["boxes"].detach().to(
                    device=boxes.device, dtype=torch.float32
                )
            )
            iou, _ = box_ops.box_iou(candidates_xyxy[row_index], target_boxes)
            max_iou = iou.amax(dim=1)
            category_positive = max_iou >= self.positive_iou_threshold
            category_negative = max_iou < self.negative_iou_threshold
            row_native = native_score[row_index]
            row_standardized = standardized[row_index]

            with torch.no_grad():
                native_winner = int(row_native.argmax().item())
                winner_is_positive = bool(category_positive[native_winner].item())
                winner_is_negative = bool(category_negative[native_winner].item())
                has_positive = bool(category_positive.any().item())

                positive_query = None
                if winner_is_negative and has_positive:
                    positive_query = int(
                        row_native.masked_fill(~category_positive, -torch.inf)
                        .argmax()
                        .item()
                    )

            if positive_query is not None:
                z_positive = row_standardized[positive_query]
                z_negative = row_standardized[native_winner]
                separation = z_positive - z_negative
                positive_gap = row_best[row_index] - z_positive
                critical_separation_losses.append(
                    F.softplus(
                        (self.separation_gap - separation) / self.temperature
                    )
                )
                critical_keep_losses.append(
                    F.softplus(
                        (positive_gap - self.keep_gap) / self.temperature
                    )
                )
                with torch.no_grad():
                    separation_compliant += int(
                        bool((separation >= self.separation_gap).item())
                    )
                    q_p_keep += int(
                        bool((positive_gap <= self.keep_gap).item())
                    )
            elif winner_is_positive:
                positive_gap = (
                    row_best[row_index] - row_standardized[native_winner]
                )
                with torch.no_grad():
                    active = bool((positive_gap > self.active_gap).item())
                    positive_native_rows += 1
                    positive_active_rows += int(active)
                    positive_deployment_keep += int(
                        bool((positive_gap <= self.gate_max_gap).item())
                    )
                    positive_target_keep += int(
                        bool((positive_gap <= self.target_gap).item())
                    )
                if active:
                    positive_barrier_losses.append(
                        F.softplus(
                            (positive_gap - self.target_gap) / self.temperature
                        )
                    )

        zero = standardized.sum() * 0.0
        critical_separation_loss = (
            torch.stack(critical_separation_losses).mean()
            if critical_separation_losses
            else zero
        )
        critical_keep_loss = (
            torch.stack(critical_keep_losses).mean()
            if critical_keep_losses
            else zero
        )
        positive_barrier_loss = (
            torch.stack(positive_barrier_losses).mean()
            if positive_barrier_losses
            else zero
        )
        loss = (
            self.critical_weight * critical_separation_loss
            + self.critical_keep_weight * critical_keep_loss
            + self.positive_barrier_weight * positive_barrier_loss
        )
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("D5 loss is not finite")

        metric = lambda value: patch_score.new_tensor(float(value)).detach()
        return {
            NATIVE_PATCH_CATEGORY_D5_LOSS: loss,
            "stage_b_native_patch_category_d5_critical_rows": metric(
                len(critical_separation_losses)
            ),
            "stage_b_native_patch_category_d5_separation_compliant": metric(
                separation_compliant
            ),
            "stage_b_native_patch_category_d5_q_p_keep": metric(q_p_keep),
            "stage_b_native_patch_category_d5_positive_native_rows": metric(
                positive_native_rows
            ),
            "stage_b_native_patch_category_d5_positive_active_rows": metric(
                positive_active_rows
            ),
            "stage_b_native_patch_category_d5_positive_deployment_keep": metric(
                positive_deployment_keep
            ),
            "stage_b_native_patch_category_d5_positive_target_keep": metric(
                positive_target_keep
            ),
            "stage_b_native_patch_category_d5_critical_separation_loss": (
                critical_separation_loss.detach()
            ),
            "stage_b_native_patch_category_d5_critical_keep_loss": (
                critical_keep_loss.detach()
            ),
            "stage_b_native_patch_category_d5_positive_barrier_loss": (
                positive_barrier_loss.detach()
            ),
        }


__all__ = [
    "NATIVE_PATCH_CATEGORY_D5_CONTRACT_VERSION",
    "NATIVE_PATCH_CATEGORY_D5_LOSS",
    "NATIVE_PATCH_CATEGORY_D5_MARKER",
    "StageBNativePatchCategoryD5Criterion",
]
