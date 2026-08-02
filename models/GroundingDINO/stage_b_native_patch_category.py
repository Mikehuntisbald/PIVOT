"""Data-only patch-category supervision for frozen native full-text ranking."""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn

from .stage_b_data_driven_score import (
    _instance_complete_patch_margin_loss,
    data_driven_category_gate_mask,
)


NATIVE_PATCH_CATEGORY_CONTRACT_VERSION = 1
NATIVE_PATCH_CATEGORY_COMPLETE_MARKER = "stage_b_native_patch_category_d1"
NATIVE_PATCH_CATEGORY_LOSS = "loss_stage_b_native_patch_category"


def _require_finite_floating_tensor(
    value: Any,
    *,
    name: str,
    dimensions: int,
) -> Tensor:
    if (
        not torch.is_tensor(value)
        or not value.is_floating_point()
        or value.dim() != int(dimensions)
    ):
        raise ValueError(f"{name} must be a floating {dimensions}D tensor")
    if not bool(torch.isfinite(value).all().item()):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _validate_category_complete_targets(
    targets: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
) -> None:
    if not isinstance(targets, Sequence) or isinstance(targets, (str, bytes)):
        raise ValueError("native patch-category targets must be a sequence")
    if len(targets) != int(batch_size):
        raise ValueError(
            "native patch-category target batch does not match model outputs"
        )
    for row_index, target in enumerate(targets):
        if not isinstance(target, Mapping):
            raise ValueError(
                f"native patch-category target {row_index} must be a mapping"
            )
        boxes = _require_finite_floating_tensor(
            target.get("boxes"),
            name=f"native patch-category target {row_index} boxes",
            dimensions=2,
        )
        if int(boxes.shape[0]) < 1 or int(boxes.shape[1]) != 4:
            raise ValueError(
                f"native patch-category target {row_index} boxes must have shape (N,4)"
            )
        instance_count = int(boxes.shape[0])

        labels = target.get("labels")
        if (
            not torch.is_tensor(labels)
            or labels.dtype != torch.int64
            or tuple(labels.shape) != (instance_count,)
        ):
            raise ValueError(
                f"native patch-category target {row_index} labels must be int64 (N,)"
            )
        primary = target.get("primary_instance_mask")
        if (
            not torch.is_tensor(primary)
            or primary.dtype != torch.bool
            or tuple(primary.shape) != (instance_count,)
            or int(primary.sum().item()) != 1
        ):
            raise ValueError(
                f"native patch-category target {row_index} requires one exact primary instance"
            )
        marker = target.get(NATIVE_PATCH_CATEGORY_COMPLETE_MARKER)
        if not (
            torch.is_tensor(marker)
            and marker.dtype == torch.bool
            and tuple(marker.shape) == (1,)
            and bool(marker[0].item())
        ):
            raise ValueError(
                "native patch-category target requires the exact category-complete marker"
            )
        if not (
            labels.device == boxes.device
            and primary.device == boxes.device
            and marker.device == boxes.device
        ):
            raise ValueError(
                f"native patch-category target {row_index} tensors must share a device"
            )
        primary_label = labels[primary][0]
        if not bool((labels == primary_label).all().item()):
            raise ValueError(
                f"native patch-category target {row_index} contains multiple categories"
            )


def _validate_fulltext_batch_context(
    cap_list: Optional[Sequence[Sequence[str]]],
    captions: Optional[Sequence[str]],
    *,
    batch_size: int,
) -> None:
    if cap_list is None and captions is None:
        return
    if cap_list is None or captions is None:
        raise ValueError(
            "native patch-category caption context requires both cap_list and captions"
        )
    if (
        not isinstance(cap_list, Sequence)
        or isinstance(cap_list, (str, bytes))
        or not isinstance(captions, Sequence)
        or isinstance(captions, (str, bytes))
    ):
        raise ValueError("native patch-category caption context must be batched")
    if len(cap_list) != int(batch_size) or len(captions) != int(batch_size):
        raise ValueError("native patch-category caption context has the wrong batch size")
    for row_index, (expressions, caption) in enumerate(zip(cap_list, captions)):
        if (
            not isinstance(expressions, (list, tuple))
            or len(expressions) != 1
            or not isinstance(expressions[0], str)
            or not expressions[0].strip()
        ):
            raise ValueError(
                "native patch-category training requires one full expression per row; "
                f"row {row_index} is invalid"
            )
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError(
                f"native patch-category caption {row_index} must be non-empty"
            )


def apply_native_patch_category_gate(
    native_score: Tensor,
    patch_score: Tensor,
    candidate_mask: Tensor,
    *,
    max_gap: float,
    clip: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply a patch eligibility gate while preserving native rank exactly."""
    native_score = _require_finite_floating_tensor(
        native_score, name="native score", dimensions=2
    )
    if not torch.is_tensor(patch_score):
        raise ValueError("patch score must be a tensor")
    patch_shape = (
        tuple(patch_score.shape[:2])
        if patch_score.dim() == 3 and int(patch_score.shape[-1]) == 1
        else tuple(patch_score.shape)
    )
    if patch_shape != tuple(native_score.shape):
        raise ValueError("patch score must align with native score")
    if not torch.is_tensor(candidate_mask):
        raise ValueError("candidate mask must be a tensor")
    if tuple(candidate_mask.shape) != tuple(native_score.shape):
        raise ValueError("candidate mask must align with native score")
    if (
        native_score.device != patch_score.device
        or native_score.device != candidate_mask.device
    ):
        raise ValueError("native score, patch score, and mask must share a device")

    eligible, standardized_patch = data_driven_category_gate_mask(
        patch_score,
        candidate_mask,
        max_gap=max_gap,
        clip=clip,
    )
    # Eligibility and lexicographic demotion are inference decisions. Detaching
    # this branch also avoids exposing torch.nextafter's undefined derivative.
    with torch.no_grad():
        native_min = native_score.masked_fill(~candidate_mask, torch.inf).amin(
            dim=1, keepdim=True
        )
        native_max = native_score.masked_fill(~candidate_mask, -torch.inf).amax(
            dim=1, keepdim=True
        )
        below_native_min = torch.nextafter(
            native_min, torch.full_like(native_min, -torch.inf)
        )
        if not bool(torch.isfinite(below_native_min).all().item()):
            raise RuntimeError("cannot construct a finite category-demotion score")

        native_delta = (
            torch.where(candidate_mask, native_score, native_max) - native_max
        )
        demoted_score = below_native_min + native_delta
        if not bool(torch.isfinite(demoted_score).all().item()):
            raise RuntimeError("native score range cannot be demoted without overflow")
    rank_score = torch.where(eligible, native_score, demoted_score)
    if not torch.equal(rank_score[eligible], native_score[eligible]):
        raise RuntimeError("category gate changed an eligible native score")

    return rank_score, eligible, standardized_patch


class StageBNativePatchCategoryCriterion(nn.Module):
    """Train only patch category evidence against category-complete GT boxes."""

    def __init__(
        self,
        *,
        patch_weight: float = 1.0,
        positive_iou_threshold: float = 0.5,
        negative_iou_threshold: float = 0.3,
        margin: float = 0.1,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_weight = float(patch_weight)
        self.positive_iou_threshold = float(positive_iou_threshold)
        self.negative_iou_threshold = float(negative_iou_threshold)
        self.margin = float(margin)
        self.temperature = float(temperature)
        if not math.isfinite(self.patch_weight) or self.patch_weight <= 0.0:
            raise ValueError("native patch-category weight must be finite and positive")
        if not 0.0 < self.positive_iou_threshold <= 1.0:
            raise ValueError("native patch-category positive IoU threshold is invalid")
        if not (
            0.0
            <= self.negative_iou_threshold
            < self.positive_iou_threshold
        ):
            raise ValueError("native patch-category negative IoU threshold is invalid")
        if not math.isfinite(self.margin) or self.margin < 0.0:
            raise ValueError("native patch-category margin must be finite and nonnegative")
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError(
                "native patch-category temperature must be finite and positive"
            )
        self.weight_dict = {NATIVE_PATCH_CATEGORY_LOSS: self.patch_weight}

    def forward(
        self,
        outputs: Mapping[str, Any],
        targets: Sequence[Mapping[str, Any]],
        cap_list: Optional[Sequence[Sequence[str]]] = None,
        captions: Optional[Sequence[str]] = None,
    ) -> dict[str, Tensor]:
        if not isinstance(outputs, Mapping):
            raise ValueError("native patch-category outputs must be a mapping")
        patch_score = outputs.get("pred_logits_patch")
        if not torch.is_tensor(patch_score):
            raise ValueError("native patch-category outputs require pred_logits_patch")
        if patch_score.dim() == 3:
            if int(patch_score.shape[-1]) != 1:
                raise ValueError(
                    "native patch-category pred_logits_patch requires one support slot"
                )
            flat_patch_score = patch_score[..., 0]
        else:
            flat_patch_score = patch_score
        flat_patch_score = _require_finite_floating_tensor(
            flat_patch_score,
            name="native patch-category pred_logits_patch",
            dimensions=2,
        )
        candidate_boxes = _require_finite_floating_tensor(
            outputs.get("pred_boxes"),
            name="native patch-category pred_boxes",
            dimensions=3,
        )
        if (
            int(candidate_boxes.shape[0]) < 1
            or int(candidate_boxes.shape[1]) < 1
            or int(candidate_boxes.shape[2]) != 4
        ):
            raise ValueError(
                "native patch-category pred_boxes must have shape (B,Q,4)"
            )
        if tuple(flat_patch_score.shape) != tuple(candidate_boxes.shape[:2]):
            raise ValueError(
                "native patch-category patch scores and candidate boxes must align"
            )
        if flat_patch_score.device != candidate_boxes.device:
            raise ValueError(
                "native patch-category patch scores and boxes must share a device"
            )

        batch_size = int(candidate_boxes.shape[0])
        _validate_category_complete_targets(targets, batch_size=batch_size)
        _validate_fulltext_batch_context(
            cap_list, captions, batch_size=batch_size
        )
        loss, valid_instances, skipped_instances = (
            _instance_complete_patch_margin_loss(
                flat_patch_score,
                candidate_boxes,
                targets,
                positive_iou_threshold=self.positive_iou_threshold,
                negative_iou_threshold=self.negative_iou_threshold,
                margin=self.margin,
                temperature=self.temperature,
            )
        )
        if int(valid_instances.item()) <= 0:
            raise RuntimeError(
                "native patch-category batch has no valid positive/negative supervision"
            )
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("native patch-category loss is not finite")
        return {
            NATIVE_PATCH_CATEGORY_LOSS: loss,
            "stage_b_native_patch_category_valid_instances": valid_instances,
            "stage_b_native_patch_category_skipped_instances": skipped_instances,
        }


__all__ = [
    "NATIVE_PATCH_CATEGORY_COMPLETE_MARKER",
    "NATIVE_PATCH_CATEGORY_CONTRACT_VERSION",
    "NATIVE_PATCH_CATEGORY_LOSS",
    "StageBNativePatchCategoryCriterion",
    "apply_native_patch_category_gate",
]
