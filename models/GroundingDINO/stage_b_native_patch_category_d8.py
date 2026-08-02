"""State-and-class macro positive anchors for patch-category training.

D8 reuses D6's three deployment-gap terms exactly and adds positive anchors
whose aggregation is fixed by native-winner state.  Within each state, active
rows are averaged inside support class before active classes are averaged.
Empty states contribute differentiable zero and never renormalize the fixed
negative, neutral, and positive state weights.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from groundingdino.util import box_ops

from .stage_b_gdino_score_adapter import aggregate_gdino_full_expression_score
from .stage_b_native_patch_category_d2 import (
    _expression_mask,
    _finite_float,
    gate_aligned_standardized_patch_score,  # compatibility audit hook
)
from .stage_b_native_patch_category_d6 import (
    NATIVE_PATCH_CATEGORY_D6_LOSS,
    NATIVE_PATCH_CATEGORY_D6_MARKER,
    StageBNativePatchCategoryD6Criterion,
)


NATIVE_PATCH_CATEGORY_D8_CONTRACT_VERSION = 8
NATIVE_PATCH_CATEGORY_D8_MARKER = NATIVE_PATCH_CATEGORY_D6_MARKER
NATIVE_PATCH_CATEGORY_D8_LOSS = "loss_stage_b_native_patch_category_d8"

_D6_TELEMETRY_PREFIX = "stage_b_native_patch_category_d6_"
_D8_TELEMETRY_PREFIX = "stage_b_native_patch_category_d8_"
_NATIVE_STATES = ("negative", "neutral", "positive")


class StageBNativePatchCategoryD8Criterion(
    StageBNativePatchCategoryD6Criterion
):
    """D6 plus fixed-weight state and support-class macro anchors."""

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
        negative_weight: float = 1.0,
        neutral_weight: float = 2.0,
        positive_weight: float = 4.0,
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
            anchor_active_gap, name="D8 anchor active gap"
        )
        self.anchor_target_gap = _finite_float(
            anchor_target_gap, name="D8 anchor target gap"
        )
        self.negative_weight = _finite_float(
            negative_weight, name="D8 negative anchor weight"
        )
        self.neutral_weight = _finite_float(
            neutral_weight, name="D8 neutral anchor weight"
        )
        self.positive_weight = _finite_float(
            positive_weight, name="D8 positive anchor weight"
        )
        if (
            not 0.0
            <= self.anchor_active_gap
            < self.anchor_target_gap
            < self.gate_max_gap
            or min(
                self.negative_weight,
                self.neutral_weight,
                self.positive_weight,
            )
            <= 0.0
        ):
            raise ValueError("D8 anchor geometry is invalid")
        self.weight_dict = {NATIVE_PATCH_CATEGORY_D8_LOSS: self.weight}

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
        standardized = self._standardize_patch_score(
            patch_score, candidate_mask
        )
        best = standardized.amax(dim=1).detach()
        gap = best[:, None] - standardized

        state_class_losses: dict[str, dict[int, list[Tensor]]] = {
            state: defaultdict(list) for state in _NATIVE_STATES
        }
        anchor_rows = {state: 0 for state in _NATIVE_STATES}
        active_rows = {state: 0 for state in _NATIVE_STATES}
        deployment_keep = {state: 0 for state in _NATIVE_STATES}
        target_keep = {state: 0 for state in _NATIVE_STATES}

        candidates_xyxy = box_ops.box_cxcywh_to_xyxy(boxes.detach().float())
        for row_index, target in enumerate(targets):
            support_class = target["support_class"].reshape(-1)[0]
            same_class = target["labels"] == support_class
            target_boxes = box_ops.box_cxcywh_to_xyxy(
                target["boxes"][same_class]
                .detach()
                .to(device=boxes.device, dtype=torch.float32)
            )
            iou, _ = box_ops.box_iou(candidates_xyxy[row_index], target_boxes)
            max_iou = iou.amax(dim=1)
            category_positive = max_iou >= self.positive_iou_threshold
            category_negative = max_iou < self.negative_iou_threshold
            if not bool(category_positive.any().item()):
                continue

            with torch.no_grad():
                row_native = native_score[row_index]
                native_winner = int(row_native.argmax().item())
                if bool(category_positive[native_winner].item()):
                    native_state = "positive"
                elif bool(category_negative[native_winner].item()):
                    native_state = "negative"
                else:
                    native_state = "neutral"
                positive_query = int(
                    row_native.masked_fill(~category_positive, -torch.inf)
                    .argmax()
                    .item()
                )
                support_class_id = int(support_class.detach().item())

            anchor_gap = gap[row_index, positive_query]
            with torch.no_grad():
                anchor_rows[native_state] += 1
                active = bool(
                    (anchor_gap.detach() > self.anchor_active_gap).item()
                )
                deployment_keep[native_state] += int(
                    bool((anchor_gap <= self.gate_max_gap).item())
                )
                target_keep[native_state] += int(
                    bool((anchor_gap <= self.anchor_target_gap).item())
                )
            if active:
                active_rows[native_state] += 1
                state_class_losses[native_state][support_class_id].append(
                    F.softplus(
                        (anchor_gap - self.anchor_target_gap)
                        / self.temperature
                    )
                )

        zero = standardized.sum() * 0.0
        state_losses: dict[str, Tensor] = {}
        for state in _NATIVE_STATES:
            class_means = [
                torch.stack(losses).mean()
                for losses in state_class_losses[state].values()
            ]
            state_losses[state] = (
                torch.stack(class_means).mean() if class_means else zero
            )

        loss = (
            d6_loss
            + self.negative_weight * state_losses["negative"]
            + self.neutral_weight * state_losses["neutral"]
            + self.positive_weight * state_losses["positive"]
        )
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("D8 loss is not finite")

        renamed: dict[str, Tensor] = {NATIVE_PATCH_CATEGORY_D8_LOSS: loss}
        for name, value in d6_outputs.items():
            if name == NATIVE_PATCH_CATEGORY_D6_LOSS:
                continue
            if not name.startswith(_D6_TELEMETRY_PREFIX):
                raise RuntimeError(f"D8 received an unknown D6 output: {name!r}")
            renamed[
                _D8_TELEMETRY_PREFIX
                + name.removeprefix(_D6_TELEMETRY_PREFIX)
            ] = value

        metric = lambda value: patch_score.new_tensor(float(value)).detach()
        for state in _NATIVE_STATES:
            prefix = f"stage_b_native_patch_category_d8_anchor_{state}_"
            renamed.update(
                {
                    prefix + "rows": metric(anchor_rows[state]),
                    prefix + "active_rows": metric(active_rows[state]),
                    prefix + "active_classes": metric(
                        len(state_class_losses[state])
                    ),
                    prefix + "deployment_keep": metric(
                        deployment_keep[state]
                    ),
                    prefix + "target_keep": metric(target_keep[state]),
                    prefix + "loss": state_losses[state].detach(),
                }
            )
        return renamed


__all__ = [
    "NATIVE_PATCH_CATEGORY_D8_CONTRACT_VERSION",
    "NATIVE_PATCH_CATEGORY_D8_LOSS",
    "NATIVE_PATCH_CATEGORY_D8_MARKER",
    "StageBNativePatchCategoryD8Criterion",
]
