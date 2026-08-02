"""Critical-winner Stage-B patch-category supervision.

D3 is an objective-only revision over the D2 category-complete corpus.  It
trains the deployment-standardized patch score on the native full-text winner
that can actually change RefCOCO output:

* if the native winner is category-negative, separate it from the strongest
  native-ranked category-positive query;
* keep that positive query inside the deployment eligibility band; and
* when the native winner is category-positive, keep it inside the same band.

The three terms are averaged over their own row sets.  In particular, a
critical native winner is never diluted into a top-k negative-query pool.
Native text scores and boxes are detached selectors; only the patch branch can
receive gradients.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from groundingdino.util import box_ops

from .stage_b_gdino_score_adapter import aggregate_gdino_full_expression_score
from .stage_b_native_patch_category_d2 import (
    NATIVE_PATCH_CATEGORY_D2_MARKER,
    _expression_mask,
    _finite_float,
    _validate_targets,
    gate_aligned_standardized_patch_score,
)


NATIVE_PATCH_CATEGORY_D3_CONTRACT_VERSION = 3
# D3 deliberately consumes the audited D2 category-complete corpus unchanged.
NATIVE_PATCH_CATEGORY_D3_MARKER = NATIVE_PATCH_CATEGORY_D2_MARKER
NATIVE_PATCH_CATEGORY_D3_LOSS = "loss_stage_b_native_patch_category_d3"


class StageBNativePatchCategoryD3Criterion(nn.Module):
    """Directly correct category-negative native winners in standardized z."""

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
        positive_keep_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.weight = _finite_float(weight, name="D3 loss weight")
        self.positive_iou_threshold = _finite_float(
            positive_iou_threshold, name="D3 positive IoU threshold"
        )
        self.negative_iou_threshold = _finite_float(
            negative_iou_threshold, name="D3 negative IoU threshold"
        )
        self.gate_max_gap = _finite_float(
            gate_max_gap, name="D3 gate max gap"
        )
        self.patch_score_clip = _finite_float(
            patch_score_clip, name="D3 patch-score clip"
        )
        self.keep_gap = _finite_float(keep_gap, name="D3 keep gap")
        self.separation_gap = _finite_float(
            separation_gap, name="D3 separation gap"
        )
        self.temperature = _finite_float(temperature, name="D3 temperature")
        self.critical_weight = _finite_float(
            critical_weight, name="D3 critical weight"
        )
        self.critical_keep_weight = _finite_float(
            critical_keep_weight, name="D3 critical keep weight"
        )
        self.positive_keep_weight = _finite_float(
            positive_keep_weight, name="D3 positive keep weight"
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
            or self.temperature <= 0.0
            or min(
                self.critical_weight,
                self.critical_keep_weight,
                self.positive_keep_weight,
            )
            < 0.0
            or self.critical_weight + self.critical_keep_weight <= 0.0
            or self.positive_keep_weight <= 0.0
        ):
            raise ValueError("D3 criterion geometry is invalid")
        self.weight_dict = {NATIVE_PATCH_CATEGORY_D3_LOSS: self.weight}

    def forward(
        self,
        outputs: Mapping[str, Any],
        targets: Sequence[Mapping[str, Any]],
        cap_list: Optional[Sequence[Sequence[str]]] = None,
        captions: Optional[Sequence[str]] = None,
    ) -> dict[str, Tensor]:
        del cap_list, captions
        if not isinstance(outputs, Mapping):
            raise ValueError("D3 outputs must be a mapping")
        patch_score = outputs.get("pred_logits_patch")
        if torch.is_tensor(patch_score) and patch_score.dim() == 3:
            if int(patch_score.shape[-1]) != 1:
                raise ValueError("D3 requires exactly one support-patch slot")
            patch_score = patch_score[..., 0]
        if (
            not torch.is_tensor(patch_score)
            or not patch_score.is_floating_point()
            or patch_score.dim() != 2
            or not bool(torch.isfinite(patch_score).all().item())
        ):
            raise ValueError("D3 pred_logits_patch must be finite floating (B,Q)")

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
            raise ValueError("D3 pred_boxes must be finite and patch-aligned")
        if (
            not torch.is_tensor(token_logits)
            or not token_logits.is_floating_point()
            or token_logits.dim() != 3
            or tuple(token_logits.shape[:2]) != tuple(patch_score.shape)
        ):
            raise ValueError("D3 pred_logits_text must be floating and patch-aligned")
        if len({patch_score.device, boxes.device, token_logits.device}) != 1:
            raise ValueError("D3 output tensors must share one device")

        batch_size, query_count = patch_score.shape
        _validate_targets(targets, batch_size=int(batch_size))
        expression_mask = _expression_mask(outputs, token_logits)
        scored_text_mask = expression_mask[:, None, :].expand_as(token_logits)
        if not bool(torch.isfinite(token_logits[scored_text_mask]).all().item()):
            raise ValueError("D3 full-expression token logits must be finite")
        native_score = aggregate_gdino_full_expression_score(
            token_logits, expression_mask
        ).detach()
        if not bool(torch.isfinite(native_score).all().item()):
            raise RuntimeError("D3 native full-expression score is not finite")

        candidate_mask = torch.ones(
            (batch_size, query_count), dtype=torch.bool, device=patch_score.device
        )
        standardized = gate_aligned_standardized_patch_score(
            patch_score, candidate_mask, clip=self.patch_score_clip
        )
        row_best = standardized.amax(dim=1).detach()

        critical_separation_losses: list[Tensor] = []
        critical_keep_losses: list[Tensor] = []
        positive_keep_losses: list[Tensor] = []
        separation_compliant = 0
        q_p_keep = 0
        positive_native_keep = 0

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
                positive_keep_losses.append(
                    F.softplus(
                        (positive_gap - self.keep_gap) / self.temperature
                    )
                )
                with torch.no_grad():
                    positive_native_keep += int(
                        bool((positive_gap <= self.keep_gap).item())
                    )

        if not critical_separation_losses and not positive_keep_losses:
            raise RuntimeError(
                "D3 batch has no critical-negative or positive-native winner supervision"
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
        positive_keep_loss = (
            torch.stack(positive_keep_losses).mean()
            if positive_keep_losses
            else zero
        )
        loss = (
            self.critical_weight * critical_separation_loss
            + self.critical_keep_weight * critical_keep_loss
            + self.positive_keep_weight * positive_keep_loss
        )
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("D3 loss is not finite")

        metric = lambda value: patch_score.new_tensor(float(value)).detach()
        return {
            NATIVE_PATCH_CATEGORY_D3_LOSS: loss,
            "stage_b_native_patch_category_d3_critical_rows": metric(
                len(critical_separation_losses)
            ),
            "stage_b_native_patch_category_d3_separation_compliant": metric(
                separation_compliant
            ),
            "stage_b_native_patch_category_d3_q_p_keep": metric(q_p_keep),
            "stage_b_native_patch_category_d3_positive_native_rows": metric(
                len(positive_keep_losses)
            ),
            "stage_b_native_patch_category_d3_positive_native_keep": metric(
                positive_native_keep
            ),
            "stage_b_native_patch_category_d3_critical_separation_loss": (
                critical_separation_loss.detach()
            ),
            "stage_b_native_patch_category_d3_critical_keep_loss": (
                critical_keep_loss.detach()
            ),
            "stage_b_native_patch_category_d3_positive_keep_loss": (
                positive_keep_loss.detach()
            ),
        }


__all__ = [
    "NATIVE_PATCH_CATEGORY_D3_CONTRACT_VERSION",
    "NATIVE_PATCH_CATEGORY_D3_LOSS",
    "NATIVE_PATCH_CATEGORY_D3_MARKER",
    "StageBNativePatchCategoryD3Criterion",
]
