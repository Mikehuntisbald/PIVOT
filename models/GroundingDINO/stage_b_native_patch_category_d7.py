"""All-state positive-anchor supervision for native patch-category training.

D7 keeps D6's direct negative-winner drop, critical-positive keep, and
positive-winner barrier unchanged.  It adds one independent active mean over
the native-best category-positive query in every row that has such a query,
regardless of whether the native winner is negative, neutral, or positive.
All selectors are detached; only the standardized patch score is optimized.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from groundingdino.util import box_ops

from .stage_b_gdino_score_adapter import aggregate_gdino_full_expression_score
from .stage_b_native_patch_category_d2 import (
    _expression_mask,
    _finite_float,
    gate_aligned_standardized_patch_score,
)
from .stage_b_native_patch_category_d6 import (
    NATIVE_PATCH_CATEGORY_D6_LOSS,
    NATIVE_PATCH_CATEGORY_D6_MARKER,
    StageBNativePatchCategoryD6Criterion,
)


NATIVE_PATCH_CATEGORY_D7_CONTRACT_VERSION = 7
NATIVE_PATCH_CATEGORY_D7_MARKER = NATIVE_PATCH_CATEGORY_D6_MARKER
NATIVE_PATCH_CATEGORY_D7_LOSS = "loss_stage_b_native_patch_category_d7"

_D6_TELEMETRY_PREFIX = "stage_b_native_patch_category_d6_"
_D7_TELEMETRY_PREFIX = "stage_b_native_patch_category_d7_"


class StageBNativePatchCategoryD7Criterion(
    StageBNativePatchCategoryD6Criterion
):
    """D6 deployment-gap supervision plus an all-state positive anchor."""

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
        anchor_active_gap: float = 2.0,
        anchor_target_gap: float = 2.5,
        anchor_weight: float = 2.0,
    ) -> None:
        super().__init__(
            weight=weight,
            positive_iou_threshold=positive_iou_threshold,
            negative_iou_threshold=negative_iou_threshold,
            gate_max_gap=gate_max_gap,
            patch_score_clip=patch_score_clip,
            keep_gap=keep_gap,
            drop_gap=drop_gap,
            drop_active_gap=drop_active_gap,
            temperature=temperature,
            drop_weight=drop_weight,
            critical_keep_weight=critical_keep_weight,
            positive_active_gap=positive_active_gap,
            positive_target_gap=positive_target_gap,
            positive_barrier_weight=positive_barrier_weight,
        )
        self.anchor_active_gap = _finite_float(
            anchor_active_gap, name="D7 anchor active gap"
        )
        self.anchor_target_gap = _finite_float(
            anchor_target_gap, name="D7 anchor target gap"
        )
        self.anchor_weight = _finite_float(
            anchor_weight, name="D7 anchor weight"
        )
        if (
            not 0.0
            <= self.anchor_active_gap
            < self.anchor_target_gap
            < self.gate_max_gap
            or self.anchor_weight <= 0.0
        ):
            raise ValueError("D7 anchor geometry is invalid")
        self.weight_dict = {NATIVE_PATCH_CATEGORY_D7_LOSS: self.weight}

    def forward(
        self,
        outputs: Mapping[str, Any],
        targets: Sequence[Mapping[str, Any]],
        cap_list: Optional[Sequence[Sequence[str]]] = None,
        captions: Optional[Sequence[str]] = None,
    ) -> dict[str, Tensor]:
        d6_outputs = super().forward(
            outputs,
            targets,
            cap_list=cap_list,
            captions=captions,
        )
        d6_loss = d6_outputs[NATIVE_PATCH_CATEGORY_D6_LOSS]

        patch_score = outputs["pred_logits_patch"]
        if patch_score.dim() == 3:
            patch_score = patch_score[..., 0]
        boxes = outputs["pred_boxes"]
        token_logits = outputs["pred_logits_text"]
        expression_mask = _expression_mask(outputs, token_logits)
        native_score = aggregate_gdino_full_expression_score(
            token_logits, expression_mask
        ).detach()
        batch_size, query_count = patch_score.shape
        candidate_mask = torch.ones(
            (batch_size, query_count), dtype=torch.bool, device=patch_score.device
        )
        standardized = gate_aligned_standardized_patch_score(
            patch_score, candidate_mask, clip=self.patch_score_clip
        )
        best = standardized.amax(dim=1).detach()
        gap = best[:, None] - standardized

        anchor_losses: list[Tensor] = []
        anchor_rows = 0
        anchor_deployment_keep = 0
        anchor_target_keep = 0
        candidates_xyxy = box_ops.box_cxcywh_to_xyxy(boxes.detach().float())
        for row_index, target in enumerate(targets):
            same_class = target["labels"] == target["support_class"].reshape(-1)[0]
            target_boxes = box_ops.box_cxcywh_to_xyxy(
                target["boxes"][same_class]
                .detach()
                .to(device=boxes.device, dtype=torch.float32)
            )
            iou, _ = box_ops.box_iou(candidates_xyxy[row_index], target_boxes)
            category_positive = (
                iou.amax(dim=1) >= self.positive_iou_threshold
            )
            if not bool(category_positive.any().item()):
                continue

            with torch.no_grad():
                positive_query = int(
                    native_score[row_index]
                    .masked_fill(~category_positive, -torch.inf)
                    .argmax()
                    .item()
                )
            anchor_gap = gap[row_index, positive_query]
            with torch.no_grad():
                anchor_rows += 1
                active = bool(
                    (anchor_gap.detach() > self.anchor_active_gap).item()
                )
                anchor_deployment_keep += int(
                    bool((anchor_gap <= self.gate_max_gap).item())
                )
                anchor_target_keep += int(
                    bool((anchor_gap <= self.anchor_target_gap).item())
                )
            if active:
                anchor_losses.append(
                    F.softplus(
                        (anchor_gap - self.anchor_target_gap)
                        / self.temperature
                    )
                )

        zero = standardized.sum() * 0.0
        anchor_loss = (
            torch.stack(anchor_losses).mean() if anchor_losses else zero
        )
        loss = d6_loss + self.anchor_weight * anchor_loss
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("D7 loss is not finite")

        renamed: dict[str, Tensor] = {NATIVE_PATCH_CATEGORY_D7_LOSS: loss}
        for name, value in d6_outputs.items():
            if name == NATIVE_PATCH_CATEGORY_D6_LOSS:
                continue
            if not name.startswith(_D6_TELEMETRY_PREFIX):
                raise RuntimeError(f"D7 received an unknown D6 output: {name!r}")
            renamed[
                _D7_TELEMETRY_PREFIX
                + name.removeprefix(_D6_TELEMETRY_PREFIX)
            ] = value

        metric = lambda value: patch_score.new_tensor(float(value)).detach()
        renamed.update(
            {
                "stage_b_native_patch_category_d7_anchor_rows": metric(
                    anchor_rows
                ),
                "stage_b_native_patch_category_d7_anchor_active_rows": metric(
                    len(anchor_losses)
                ),
                "stage_b_native_patch_category_d7_anchor_deployment_keep": metric(
                    anchor_deployment_keep
                ),
                "stage_b_native_patch_category_d7_anchor_target_keep": metric(
                    anchor_target_keep
                ),
                "stage_b_native_patch_category_d7_anchor_loss": (
                    anchor_loss.detach()
                ),
            }
        )
        return renamed


__all__ = [
    "NATIVE_PATCH_CATEGORY_D7_CONTRACT_VERSION",
    "NATIVE_PATCH_CATEGORY_D7_LOSS",
    "NATIVE_PATCH_CATEGORY_D7_MARKER",
    "StageBNativePatchCategoryD7Criterion",
]
