"""Direct deployment-gap supervision for native patch-category training.

D6 replaces D5's pairwise critical-winner separation with a barrier on the
native negative winner's own deployment gap.  Native text scores, predicted
boxes, category labels, and active-set membership are detached selectors.
Only the deployment-standardized patch score receives gradients.
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
    gate_aligned_standardized_patch_score,
)
from .stage_b_native_patch_category_d3 import NATIVE_PATCH_CATEGORY_D3_MARKER


NATIVE_PATCH_CATEGORY_D6_CONTRACT_VERSION = 6
# D6 consumes the same audited category-complete examples as D2-D5.
NATIVE_PATCH_CATEGORY_D6_MARKER = NATIVE_PATCH_CATEGORY_D3_MARKER
NATIVE_PATCH_CATEGORY_D6_LOSS = "loss_stage_b_native_patch_category_d6"


def _validate_d6_targets(
    targets: Sequence[Mapping[str, Any]], *, batch_size: int
) -> None:
    """Validate category-complete rows without consulting primary identity."""
    if (
        not isinstance(targets, Sequence)
        or isinstance(targets, (str, bytes))
        or len(targets) != int(batch_size)
    ):
        raise ValueError("D6 targets must match the output batch")
    for row_index, target in enumerate(targets):
        if not isinstance(target, Mapping):
            raise ValueError(f"D6 target {row_index} must be a mapping")
        boxes = target.get("boxes")
        labels = target.get("labels")
        support_class = target.get("support_class")
        marker = target.get(NATIVE_PATCH_CATEGORY_D6_MARKER)
        if (
            not torch.is_tensor(boxes)
            or not boxes.is_floating_point()
            or boxes.dim() != 2
            or tuple(boxes.shape[1:]) != (4,)
            or int(boxes.shape[0]) < 1
            or not bool(torch.isfinite(boxes).all().item())
        ):
            raise ValueError(f"D6 target {row_index} boxes must be finite (N,4)")
        instance_count = int(boxes.shape[0])
        if (
            not torch.is_tensor(labels)
            or labels.dtype != torch.int64
            or tuple(labels.shape) != (instance_count,)
        ):
            raise ValueError(f"D6 target {row_index} labels must be int64 (N,)")
        if (
            not torch.is_tensor(support_class)
            or support_class.dtype != torch.int64
            or support_class.numel() != 1
        ):
            raise ValueError(f"D6 target {row_index} requires one support class")
        if not bool((labels == support_class.reshape(-1)[0]).any().item()):
            raise ValueError(f"D6 target {row_index} has no same-class GT")
        if not (
            torch.is_tensor(marker)
            and marker.dtype == torch.bool
            and tuple(marker.shape) == (1,)
            and bool(marker[0].item())
        ):
            raise ValueError("D6 target requires the exact category marker")
        if len({boxes.device, labels.device, support_class.device, marker.device}) != 1:
            raise ValueError(f"D6 target {row_index} tensors must share one device")


class StageBNativePatchCategoryD6Criterion(nn.Module):
    """Correct native critical negatives through their deployed patch gap."""

    def __init__(
        self,
        *,
        weight: float = 1.0,
        positive_iou_threshold: float = 0.5,
        negative_iou_threshold: float = 0.3,
        gate_max_gap: float = 3.0,
        patch_score_clip: float = 5.0,
        keep_gap: float = 2.75,
        drop_gap: float = 3.25,
        drop_active_gap: float = 3.75,
        temperature: float = 0.25,
        drop_weight: float = 2.0,
        critical_keep_weight: float = 1.0,
        positive_active_gap: float = 2.0,
        positive_target_gap: float = 2.5,
        positive_barrier_weight: float = 2.0,
    ) -> None:
        super().__init__()
        self.weight = _finite_float(weight, name="D6 loss weight")
        self.positive_iou_threshold = _finite_float(
            positive_iou_threshold, name="D6 positive IoU threshold"
        )
        self.negative_iou_threshold = _finite_float(
            negative_iou_threshold, name="D6 negative IoU threshold"
        )
        self.gate_max_gap = _finite_float(
            gate_max_gap, name="D6 gate max gap"
        )
        self.patch_score_clip = _finite_float(
            patch_score_clip, name="D6 patch-score clip"
        )
        self.keep_gap = _finite_float(keep_gap, name="D6 critical keep gap")
        self.drop_gap = _finite_float(drop_gap, name="D6 drop gap")
        self.drop_active_gap = _finite_float(
            drop_active_gap, name="D6 drop active gap"
        )
        self.temperature = _finite_float(
            temperature, name="D6 temperature"
        )
        self.drop_weight = _finite_float(
            drop_weight, name="D6 drop weight"
        )
        self.critical_keep_weight = _finite_float(
            critical_keep_weight, name="D6 critical keep weight"
        )
        self.positive_active_gap = _finite_float(
            positive_active_gap, name="D6 positive active gap"
        )
        self.positive_target_gap = _finite_float(
            positive_target_gap, name="D6 positive target gap"
        )
        self.positive_barrier_weight = _finite_float(
            positive_barrier_weight, name="D6 positive barrier weight"
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
            < self.drop_gap
            < self.drop_active_gap
            < 2.0 * self.patch_score_clip
            or not 0.0
            <= self.positive_active_gap
            < self.positive_target_gap
            < self.gate_max_gap
            or self.temperature <= 0.0
            or min(
                self.drop_weight,
                self.critical_keep_weight,
                self.positive_barrier_weight,
            )
            <= 0.0
        ):
            raise ValueError("D6 criterion geometry is invalid")
        self.weight_dict = {NATIVE_PATCH_CATEGORY_D6_LOSS: self.weight}

    def _standardize_patch_score(
        self, patch_score: Tensor, candidate_mask: Tensor
    ) -> Tensor:
        """Return the deployment-standardized score used by the loss."""
        return gate_aligned_standardized_patch_score(
            patch_score, candidate_mask, clip=self.patch_score_clip
        )

    def forward(
        self,
        outputs: Mapping[str, Any],
        targets: Sequence[Mapping[str, Any]],
        cap_list: Optional[Sequence[Sequence[str]]] = None,
        captions: Optional[Sequence[str]] = None,
    ) -> dict[str, Tensor]:
        del cap_list, captions
        if not isinstance(outputs, Mapping):
            raise ValueError("D6 outputs must be a mapping")
        patch_score = outputs.get("pred_logits_patch")
        if torch.is_tensor(patch_score) and patch_score.dim() == 3:
            if int(patch_score.shape[-1]) != 1:
                raise ValueError("D6 requires exactly one support-patch slot")
            patch_score = patch_score[..., 0]
        if (
            not torch.is_tensor(patch_score)
            or not patch_score.is_floating_point()
            or patch_score.dim() != 2
            or not bool(torch.isfinite(patch_score).all().item())
        ):
            raise ValueError("D6 pred_logits_patch must be finite floating (B,Q)")

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
            raise ValueError("D6 pred_boxes must be finite and patch-aligned")
        if (
            not torch.is_tensor(token_logits)
            or not token_logits.is_floating_point()
            or token_logits.dim() != 3
            or tuple(token_logits.shape[:2]) != tuple(patch_score.shape)
        ):
            raise ValueError("D6 pred_logits_text must be floating and patch-aligned")
        if len({patch_score.device, boxes.device, token_logits.device}) != 1:
            raise ValueError("D6 output tensors must share one device")

        batch_size, query_count = patch_score.shape
        _validate_d6_targets(targets, batch_size=int(batch_size))
        expression_mask = _expression_mask(outputs, token_logits)
        scored_text_mask = expression_mask[:, None, :].expand_as(token_logits)
        if not bool(torch.isfinite(token_logits[scored_text_mask]).all().item()):
            raise ValueError("D6 full-expression token logits must be finite")
        native_score = aggregate_gdino_full_expression_score(
            token_logits, expression_mask
        ).detach()
        if not bool(torch.isfinite(native_score).all().item()):
            raise RuntimeError("D6 native full-expression score is not finite")

        candidate_mask = torch.ones(
            (batch_size, query_count), dtype=torch.bool, device=patch_score.device
        )
        standardized = self._standardize_patch_score(
            patch_score, candidate_mask
        )
        best = standardized.amax(dim=1).detach()
        gap = best[:, None] - standardized

        drop_losses: list[Tensor] = []
        critical_keep_losses: list[Tensor] = []
        positive_barrier_losses: list[Tensor] = []
        negative_deployment_rejected = 0
        negative_drop_target_met = 0
        q_p_deployment_keep = 0
        q_p_target_keep = 0
        positive_native_rows = 0
        positive_deployment_keep = 0
        positive_target_keep = 0

        candidates_xyxy = box_ops.box_cxcywh_to_xyxy(boxes.detach().float())
        for row_index, target in enumerate(targets):
            same_class = target["labels"] == target["support_class"].reshape(-1)[0]
            target_boxes = box_ops.box_cxcywh_to_xyxy(
                target["boxes"][same_class]
                .detach()
                .to(device=boxes.device, dtype=torch.float32)
            )
            iou, _ = box_ops.box_iou(candidates_xyxy[row_index], target_boxes)
            max_iou = iou.amax(dim=1)
            category_positive = max_iou >= self.positive_iou_threshold
            category_negative = max_iou < self.negative_iou_threshold
            row_native = native_score[row_index]
            row_gap = gap[row_index]

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
                negative_gap = row_gap[native_winner]
                positive_gap = row_gap[positive_query]
                if bool((negative_gap.detach() < self.drop_active_gap).item()):
                    drop_losses.append(
                        F.softplus(
                            (self.drop_gap - negative_gap) / self.temperature
                        )
                    )
                critical_keep_losses.append(
                    F.softplus(
                        (positive_gap - self.keep_gap) / self.temperature
                    )
                )
                with torch.no_grad():
                    negative_deployment_rejected += int(
                        bool((negative_gap > self.gate_max_gap).item())
                    )
                    negative_drop_target_met += int(
                        bool((negative_gap >= self.drop_gap).item())
                    )
                    q_p_deployment_keep += int(
                        bool((positive_gap <= self.gate_max_gap).item())
                    )
                    q_p_target_keep += int(
                        bool((positive_gap <= self.keep_gap).item())
                    )
            elif winner_is_positive:
                positive_gap = row_gap[native_winner]
                with torch.no_grad():
                    positive_native_rows += 1
                    active = bool(
                        (positive_gap.detach() > self.positive_active_gap).item()
                    )
                    positive_deployment_keep += int(
                        bool((positive_gap <= self.gate_max_gap).item())
                    )
                    positive_target_keep += int(
                        bool((positive_gap <= self.positive_target_gap).item())
                    )
                if active:
                    positive_barrier_losses.append(
                        F.softplus(
                            (positive_gap - self.positive_target_gap)
                            / self.temperature
                        )
                    )

        zero = standardized.sum() * 0.0
        drop_loss = (
            torch.stack(drop_losses).mean() if drop_losses else zero
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
            self.drop_weight * drop_loss
            + self.critical_keep_weight * critical_keep_loss
            + self.positive_barrier_weight * positive_barrier_loss
        )
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("D6 loss is not finite")

        metric = lambda value: patch_score.new_tensor(float(value)).detach()
        return {
            NATIVE_PATCH_CATEGORY_D6_LOSS: loss,
            "stage_b_native_patch_category_d6_critical_rows": metric(
                len(critical_keep_losses)
            ),
            "stage_b_native_patch_category_d6_drop_active_rows": metric(
                len(drop_losses)
            ),
            "stage_b_native_patch_category_d6_negative_deployment_rejected": metric(
                negative_deployment_rejected
            ),
            "stage_b_native_patch_category_d6_negative_drop_target_met": metric(
                negative_drop_target_met
            ),
            "stage_b_native_patch_category_d6_q_p_deployment_keep": metric(
                q_p_deployment_keep
            ),
            "stage_b_native_patch_category_d6_q_p_target_keep": metric(
                q_p_target_keep
            ),
            "stage_b_native_patch_category_d6_positive_native_rows": metric(
                positive_native_rows
            ),
            "stage_b_native_patch_category_d6_positive_active_rows": metric(
                len(positive_barrier_losses)
            ),
            "stage_b_native_patch_category_d6_positive_deployment_keep": metric(
                positive_deployment_keep
            ),
            "stage_b_native_patch_category_d6_positive_target_keep": metric(
                positive_target_keep
            ),
            "stage_b_native_patch_category_d6_drop_loss": drop_loss.detach(),
            "stage_b_native_patch_category_d6_critical_keep_loss": (
                critical_keep_loss.detach()
            ),
            "stage_b_native_patch_category_d6_positive_barrier_loss": (
                positive_barrier_loss.detach()
            ),
        }


__all__ = [
    "NATIVE_PATCH_CATEGORY_D6_CONTRACT_VERSION",
    "NATIVE_PATCH_CATEGORY_D6_LOSS",
    "NATIVE_PATCH_CATEGORY_D6_MARKER",
    "StageBNativePatchCategoryD6Criterion",
]
