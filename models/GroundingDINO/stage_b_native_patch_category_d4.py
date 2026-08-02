"""Positive-protected critical-winner patch-category supervision.

D4 intentionally keeps the D3 selector, geometry, and loss formula as its
single source of truth.  Its only optimization change is a stronger default
weight on positive-native winner retention.  The wrapper gives that objective
an independent contract and telemetry namespace without duplicating D3 math.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from torch import Tensor

from .stage_b_native_patch_category_d3 import (
    NATIVE_PATCH_CATEGORY_D3_LOSS,
    NATIVE_PATCH_CATEGORY_D3_MARKER,
    StageBNativePatchCategoryD3Criterion,
)


NATIVE_PATCH_CATEGORY_D4_CONTRACT_VERSION = 4
# D4 consumes the same audited category-complete examples as D3 and D2.
NATIVE_PATCH_CATEGORY_D4_MARKER = NATIVE_PATCH_CATEGORY_D3_MARKER
NATIVE_PATCH_CATEGORY_D4_LOSS = "loss_stage_b_native_patch_category_d4"

_D3_TELEMETRY_PREFIX = "stage_b_native_patch_category_d3_"
_D4_TELEMETRY_PREFIX = "stage_b_native_patch_category_d4_"


class StageBNativePatchCategoryD4Criterion(
    StageBNativePatchCategoryD3Criterion
):
    """D3 critical-winner supervision with positive winners protected."""

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
        positive_keep_weight: float = 32.0,
    ) -> None:
        super().__init__(
            weight=weight,
            positive_iou_threshold=positive_iou_threshold,
            negative_iou_threshold=negative_iou_threshold,
            gate_max_gap=gate_max_gap,
            patch_score_clip=patch_score_clip,
            keep_gap=keep_gap,
            separation_gap=separation_gap,
            temperature=temperature,
            critical_weight=critical_weight,
            critical_keep_weight=critical_keep_weight,
            positive_keep_weight=positive_keep_weight,
        )
        self.weight_dict = {NATIVE_PATCH_CATEGORY_D4_LOSS: self.weight}

    def forward(
        self,
        outputs: Mapping[str, Any],
        targets: Sequence[Mapping[str, Any]],
        cap_list: Optional[Sequence[Sequence[str]]] = None,
        captions: Optional[Sequence[str]] = None,
    ) -> dict[str, Tensor]:
        d3_outputs = super().forward(
            outputs,
            targets,
            cap_list=cap_list,
            captions=captions,
        )
        renamed: dict[str, Tensor] = {}
        for name, value in d3_outputs.items():
            if name == NATIVE_PATCH_CATEGORY_D3_LOSS:
                renamed[NATIVE_PATCH_CATEGORY_D4_LOSS] = value
            elif name.startswith(_D3_TELEMETRY_PREFIX):
                renamed[
                    _D4_TELEMETRY_PREFIX
                    + name.removeprefix(_D3_TELEMETRY_PREFIX)
                ] = value
            else:
                raise RuntimeError(f"D4 received an unknown D3 output: {name!r}")
        return renamed


__all__ = [
    "NATIVE_PATCH_CATEGORY_D4_CONTRACT_VERSION",
    "NATIVE_PATCH_CATEGORY_D4_LOSS",
    "NATIVE_PATCH_CATEGORY_D4_MARKER",
    "StageBNativePatchCategoryD4Criterion",
]
