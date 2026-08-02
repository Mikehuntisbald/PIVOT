"""Gate-aligned Stage-B patch-category supervision.

D2 keeps the frozen b58 full-expression score as the within-image rank.  The
only learned surface is the patch projection used to decide which queries are
category eligible.  Native text scores are detached and used only to select
the category mistakes that matter at deployment; their values are never a
regression target.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from groundingdino.util import box_ops

from .stage_b_gdino_score_adapter import aggregate_gdino_full_expression_score
from .stage_b_data_driven_score import data_driven_category_gate_mask


NATIVE_PATCH_CATEGORY_D2_CONTRACT_VERSION = 2
NATIVE_PATCH_CATEGORY_D2_MARKER = "stage_b_native_patch_category_d2"
NATIVE_PATCH_CATEGORY_D2_LOSS = "loss_stage_b_native_patch_category_d2"


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def gate_aligned_standardized_patch_score(
    patch_score: Tensor,
    candidate_mask: Tensor,
    *,
    clip: float,
) -> Tensor:
    """Match deployment standardization while using an STE through clipping."""
    if patch_score.dim() == 3 and int(patch_score.shape[-1]) == 1:
        patch_score = patch_score[..., 0]
    if patch_score.dim() != 2 or not patch_score.is_floating_point():
        raise ValueError("D2 patch score must be a floating (B,Q) tensor")
    if (
        not torch.is_tensor(candidate_mask)
        or candidate_mask.dtype != torch.bool
        or tuple(candidate_mask.shape) != tuple(patch_score.shape)
        or candidate_mask.device != patch_score.device
    ):
        raise ValueError("D2 candidate mask must be boolean and patch-aligned")
    if bool((~candidate_mask.any(dim=1)).any().item()):
        raise ValueError("every D2 row requires at least one candidate")
    if not bool(torch.isfinite(patch_score).all().item()):
        raise ValueError("D2 patch score must contain only finite values")
    clip = _finite_float(clip, name="D2 patch-score clip")
    if clip <= 0.0:
        raise ValueError("D2 patch-score clip must be positive")

    count = candidate_mask.sum(dim=1).clamp_min(1).float()
    score = patch_score.float()
    safe = score.masked_fill(~candidate_mask, 0.0)
    mean = safe.sum(dim=1) / count
    centered = (score - mean[:, None]).masked_fill(~candidate_mask, 0.0)
    std = (centered.square().sum(dim=1) / count).clamp_min(1e-6).sqrt()
    unbounded = centered / std[:, None]
    clipped = unbounded.clamp(min=-clip, max=clip)
    standardized = unbounded + (clipped - unbounded).detach()
    return standardized.masked_fill(~candidate_mask, -clip)


def _validate_targets(
    targets: Sequence[Mapping[str, Any]], *, batch_size: int
) -> None:
    if (
        not isinstance(targets, Sequence)
        or isinstance(targets, (str, bytes))
        or len(targets) != int(batch_size)
    ):
        raise ValueError("D2 targets must match the output batch")
    for row_index, target in enumerate(targets):
        if not isinstance(target, Mapping):
            raise ValueError(f"D2 target {row_index} must be a mapping")
        boxes = target.get("boxes")
        labels = target.get("labels")
        primary = target.get("primary_instance_mask")
        support_class = target.get("support_class")
        marker = target.get(NATIVE_PATCH_CATEGORY_D2_MARKER)
        if (
            not torch.is_tensor(boxes)
            or not boxes.is_floating_point()
            or boxes.dim() != 2
            or tuple(boxes.shape[1:]) != (4,)
            or int(boxes.shape[0]) < 1
            or not bool(torch.isfinite(boxes).all().item())
        ):
            raise ValueError(f"D2 target {row_index} boxes must be finite (N,4)")
        instance_count = int(boxes.shape[0])
        if (
            not torch.is_tensor(labels)
            or labels.dtype != torch.int64
            or tuple(labels.shape) != (instance_count,)
        ):
            raise ValueError(f"D2 target {row_index} labels must be int64 (N,)")
        if (
            not torch.is_tensor(primary)
            or primary.dtype != torch.bool
            or tuple(primary.shape) != (instance_count,)
            or int(primary.sum().item()) != 1
        ):
            raise ValueError(f"D2 target {row_index} requires one primary instance")
        if (
            not torch.is_tensor(support_class)
            or support_class.dtype != torch.int64
            or support_class.numel() != 1
        ):
            raise ValueError(f"D2 target {row_index} requires one support class")
        if not bool((labels == support_class.reshape(-1)[0]).all().item()):
            raise ValueError(f"D2 target {row_index} support class does not match GT")
        if not (
            torch.is_tensor(marker)
            and marker.dtype == torch.bool
            and tuple(marker.shape) == (1,)
            and bool(marker[0].item())
        ):
            raise ValueError("D2 target requires the exact D2 marker")
        devices = {boxes.device, labels.device, primary.device, support_class.device, marker.device}
        if len(devices) != 1:
            raise ValueError(f"D2 target {row_index} tensors must share one device")


def _expression_mask(outputs: Mapping[str, Any], token_logits: Tensor) -> Tensor:
    mask = outputs.get("phrase_to_token_mask")
    if not torch.is_tensor(mask) or mask.dtype != torch.bool:
        raise ValueError("D2 outputs require a boolean phrase_to_token_mask")
    if mask.dim() != 3 or tuple(mask.shape) != (
        int(token_logits.shape[0]),
        1,
        int(token_logits.shape[2]),
    ):
        raise ValueError("D2 requires one full-expression token mask per row")
    if mask.device != token_logits.device or bool((~mask.any(dim=-1)).any().item()):
        raise ValueError("D2 expression masks must be non-empty and device-aligned")
    return mask[:, 0]


def _smooth_hinge(value: Tensor, *, temperature: float) -> Tensor:
    return float(temperature) * F.softplus(value / float(temperature))


class StageBNativePatchCategoryD2Criterion(nn.Module):
    """Train deployment Gap-3 eligibility around native-critical queries."""

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
        temperature: float = 0.25,
        native_hard_negatives: int = 16,
        patch_hard_negatives: int = 4,
        keep_weight: float = 2.0,
        drop_weight: float = 1.0,
        coverage_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self.weight = _finite_float(weight, name="D2 loss weight")
        self.positive_iou_threshold = _finite_float(
            positive_iou_threshold, name="D2 positive IoU threshold"
        )
        self.negative_iou_threshold = _finite_float(
            negative_iou_threshold, name="D2 negative IoU threshold"
        )
        self.gate_max_gap = _finite_float(gate_max_gap, name="D2 gate max gap")
        self.patch_score_clip = _finite_float(
            patch_score_clip, name="D2 patch-score clip"
        )
        self.keep_gap = _finite_float(keep_gap, name="D2 keep gap")
        self.drop_gap = _finite_float(drop_gap, name="D2 drop gap")
        self.temperature = _finite_float(temperature, name="D2 temperature")
        self.keep_weight = _finite_float(keep_weight, name="D2 keep weight")
        self.drop_weight = _finite_float(drop_weight, name="D2 drop weight")
        self.coverage_weight = _finite_float(
            coverage_weight, name="D2 coverage weight"
        )
        if (
            self.weight <= 0.0
            or not 0.0 <= self.negative_iou_threshold < self.positive_iou_threshold <= 1.0
            or self.gate_max_gap <= 0.0
            or self.patch_score_clip <= self.gate_max_gap
            or not 0.0 <= self.keep_gap < self.gate_max_gap < self.drop_gap
            or self.drop_gap >= 2.0 * self.patch_score_clip
            or self.temperature <= 0.0
            or min(self.keep_weight, self.drop_weight, self.coverage_weight) < 0.0
        ):
            raise ValueError("D2 criterion geometry is invalid")
        for name, value in (
            ("native hard negatives", native_hard_negatives),
            ("patch hard negatives", patch_hard_negatives),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"D2 {name} must be a positive integer")
        self.native_hard_negatives = int(native_hard_negatives)
        self.patch_hard_negatives = int(patch_hard_negatives)
        self.weight_dict = {NATIVE_PATCH_CATEGORY_D2_LOSS: self.weight}

    def forward(
        self,
        outputs: Mapping[str, Any],
        targets: Sequence[Mapping[str, Any]],
        cap_list: Optional[Sequence[Sequence[str]]] = None,
        captions: Optional[Sequence[str]] = None,
    ) -> dict[str, Tensor]:
        del cap_list, captions
        if not isinstance(outputs, Mapping):
            raise ValueError("D2 outputs must be a mapping")
        patch_score = outputs.get("pred_logits_patch")
        if torch.is_tensor(patch_score) and patch_score.dim() == 3:
            if int(patch_score.shape[-1]) != 1:
                raise ValueError("D2 requires exactly one support-patch slot")
            patch_score = patch_score[..., 0]
        if (
            not torch.is_tensor(patch_score)
            or not patch_score.is_floating_point()
            or patch_score.dim() != 2
            or not bool(torch.isfinite(patch_score).all().item())
        ):
            raise ValueError("D2 pred_logits_patch must be finite floating (B,Q)")
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
            raise ValueError("D2 pred_boxes must be finite and patch-aligned")
        if (
            not torch.is_tensor(token_logits)
            or not token_logits.is_floating_point()
            or token_logits.dim() != 3
            or tuple(token_logits.shape[:2]) != tuple(patch_score.shape)
        ):
            raise ValueError("D2 pred_logits_text must be floating and patch-aligned")
        if len({patch_score.device, boxes.device, token_logits.device}) != 1:
            raise ValueError("D2 output tensors must share one device")

        batch_size, query_count = patch_score.shape
        _validate_targets(targets, batch_size=int(batch_size))
        expression_mask = _expression_mask(outputs, token_logits)
        scored_text_mask = expression_mask[:, None, :].expand_as(token_logits)
        if not bool(torch.isfinite(token_logits[scored_text_mask]).all().item()):
            raise ValueError("D2 full-expression token logits must be finite")
        native_score = aggregate_gdino_full_expression_score(
            token_logits, expression_mask
        ).detach()
        if not bool(torch.isfinite(native_score).all().item()):
            raise RuntimeError("D2 native full-expression score is not finite")
        candidate_mask = torch.ones(
            (batch_size, query_count), dtype=torch.bool, device=patch_score.device
        )
        standardized = gate_aligned_standardized_patch_score(
            patch_score, candidate_mask, clip=self.patch_score_clip
        )
        row_best = standardized.amax(dim=1).detach()
        deployment_eligible, _ = data_driven_category_gate_mask(
            patch_score.detach(),
            candidate_mask,
            max_gap=self.gate_max_gap,
            clip=self.patch_score_clip,
        )

        row_losses: list[Tensor] = []
        reachable_instances = 0
        skipped_instances = 0
        selected_native_negatives = 0
        selected_patch_negatives = 0
        selected_drop_queries = 0
        keep_compliant = 0
        drop_compliant = 0
        category_best_positive = 0
        native_positive_winners = 0
        native_positive_winners_eligible = 0
        native_negative_winners = 0
        native_negative_winners_rejected = 0

        candidates_xyxy = box_ops.box_cxcywh_to_xyxy(boxes.detach().float())
        for row_index, target in enumerate(targets):
            target_boxes = box_ops.box_cxcywh_to_xyxy(
                target["boxes"].detach().to(
                    device=boxes.device, dtype=torch.float32
                )
            )
            iou, _ = box_ops.box_iou(candidates_xyxy[row_index], target_boxes)
            positives = iou >= self.positive_iou_threshold
            category_positive = positives.any(dim=1)
            category_negative = iou.amax(dim=1) < self.negative_iou_threshold
            reachable = positives.any(dim=0)
            reachable_instances += int(reachable.sum().item())
            skipped_instances += int((~reachable).sum().item())
            if not bool(reachable.any().item()) or not bool(category_negative.any().item()):
                continue

            row_native = native_score[row_index]
            row_standardized = standardized[row_index]
            with torch.no_grad():
                positive_native = row_native[:, None].masked_fill(~positives, -torch.inf)
                keep_queries = positive_native[:, reachable].argmax(dim=0)

                negative_count = int(category_negative.sum().item())
                native_k = min(self.native_hard_negatives, negative_count)
                patch_k = min(self.patch_hard_negatives, negative_count)
                native_negative_queries = row_native.masked_fill(
                    ~category_negative, -torch.inf
                ).topk(native_k).indices
                patch_negative_queries = row_standardized.detach().masked_fill(
                    ~category_negative, -torch.inf
                ).topk(patch_k).indices
                drop_queries = torch.unique(
                    torch.cat((native_negative_queries, patch_negative_queries)),
                    sorted=True,
                )

                native_winner = int(row_native.argmax().item())
                if bool(category_positive[native_winner].item()):
                    native_positive_winners += 1
                    native_positive_winners_eligible += int(
                        deployment_eligible[row_index, native_winner].item()
                    )
                elif bool(category_negative[native_winner].item()):
                    native_negative_winners += 1
                    native_negative_winners_rejected += int(
                        not deployment_eligible[row_index, native_winner].item()
                    )

            gap = row_best[row_index] - row_standardized
            keep_gap = gap[keep_queries]
            drop_gap = gap[drop_queries]
            keep_loss = _smooth_hinge(
                keep_gap - self.keep_gap, temperature=self.temperature
            ).mean()
            drop_loss = _smooth_hinge(
                self.drop_gap - drop_gap, temperature=self.temperature
            ).mean()

            reachable_positives = positives[:, reachable]
            positive_best = row_standardized[:, None].masked_fill(
                ~reachable_positives, -torch.inf
            ).amax(dim=0)
            negative_best = row_standardized.masked_fill(
                ~category_negative, -torch.inf
            ).amax()
            coverage_loss = _smooth_hinge(
                self.drop_gap - (positive_best - negative_best),
                temperature=self.temperature,
            ).mean()
            row_losses.append(
                self.keep_weight * keep_loss
                + self.drop_weight * drop_loss
                + self.coverage_weight * coverage_loss
            )

            with torch.no_grad():
                selected_native_negatives += int(native_negative_queries.numel())
                selected_patch_negatives += int(patch_negative_queries.numel())
                selected_drop_queries += int(drop_queries.numel())
                keep_compliant += int((keep_gap <= self.keep_gap).sum().item())
                drop_compliant += int((drop_gap >= self.drop_gap).sum().item())
                category_best_positive += int(
                    category_positive[int(row_standardized.argmax().item())].item()
                )

        if not row_losses:
            raise RuntimeError("D2 batch has no reachable positive/negative supervision")
        loss = torch.stack(row_losses).mean()
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("D2 loss is not finite")
        metric = lambda value: patch_score.new_tensor(float(value)).detach()
        return {
            NATIVE_PATCH_CATEGORY_D2_LOSS: loss,
            "stage_b_native_patch_category_d2_valid_rows": metric(len(row_losses)),
            "stage_b_native_patch_category_d2_reachable_instances": metric(
                reachable_instances
            ),
            "stage_b_native_patch_category_d2_skipped_instances": metric(
                skipped_instances
            ),
            "stage_b_native_patch_category_d2_native_hard_negatives": metric(
                selected_native_negatives
            ),
            "stage_b_native_patch_category_d2_patch_hard_negatives": metric(
                selected_patch_negatives
            ),
            "stage_b_native_patch_category_d2_drop_queries": metric(
                selected_drop_queries
            ),
            "stage_b_native_patch_category_d2_keep_compliant": metric(
                keep_compliant
            ),
            "stage_b_native_patch_category_d2_drop_compliant": metric(
                drop_compliant
            ),
            "stage_b_native_patch_category_d2_category_best_positive": metric(
                category_best_positive
            ),
            "stage_b_native_patch_category_d2_native_positive_winners": metric(
                native_positive_winners
            ),
            "stage_b_native_patch_category_d2_native_positive_winners_eligible": metric(
                native_positive_winners_eligible
            ),
            "stage_b_native_patch_category_d2_native_negative_winners": metric(
                native_negative_winners
            ),
            "stage_b_native_patch_category_d2_native_negative_winners_rejected": metric(
                native_negative_winners_rejected
            ),
        }


__all__ = [
    "NATIVE_PATCH_CATEGORY_D2_CONTRACT_VERSION",
    "NATIVE_PATCH_CATEGORY_D2_LOSS",
    "NATIVE_PATCH_CATEGORY_D2_MARKER",
    "StageBNativePatchCategoryD2Criterion",
    "gate_aligned_standardized_patch_score",
]
