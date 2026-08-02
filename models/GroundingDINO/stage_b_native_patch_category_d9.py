"""Loss-gradient-localized patch-category training.

D9 keeps D8's deployment forward, examples, selectors, margins, reductions,
and fixed state weights unchanged.  Its only change is in the loss backward:
the per-row mean and standard deviation used to standardize patch scores are
detached.  Consequently a loss on one selected query no longer sends a dense
gradient through the row statistics to the other queries.  Deployment still
uses the original standardization and is numerically identical to D8.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch
from torch import Tensor

from .stage_b_native_patch_category_d2 import _finite_float
from .stage_b_native_patch_category_d8 import (
    NATIVE_PATCH_CATEGORY_D8_LOSS,
    NATIVE_PATCH_CATEGORY_D8_MARKER,
    StageBNativePatchCategoryD8Criterion,
)


NATIVE_PATCH_CATEGORY_D9_CONTRACT_VERSION = 9
NATIVE_PATCH_CATEGORY_D9_MARKER = NATIVE_PATCH_CATEGORY_D8_MARKER
NATIVE_PATCH_CATEGORY_D9_LOSS = "loss_stage_b_native_patch_category_d9"

_D8_TELEMETRY_PREFIX = "stage_b_native_patch_category_d8_"
_D9_TELEMETRY_PREFIX = "stage_b_native_patch_category_d9_"


def loss_gradient_localized_standardized_patch_score(
    patch_score: Tensor,
    candidate_mask: Tensor,
    *,
    clip: float,
) -> Tensor:
    """Match deployment values while detaching row statistics in backward."""
    if patch_score.dim() == 3 and int(patch_score.shape[-1]) == 1:
        patch_score = patch_score[..., 0]
    if patch_score.dim() != 2 or not patch_score.is_floating_point():
        raise ValueError("D9 patch score must be a floating (B,Q) tensor")
    if (
        not torch.is_tensor(candidate_mask)
        or candidate_mask.dtype != torch.bool
        or tuple(candidate_mask.shape) != tuple(patch_score.shape)
        or candidate_mask.device != patch_score.device
    ):
        raise ValueError("D9 candidate mask must be boolean and patch-aligned")
    if bool((~candidate_mask.any(dim=1)).any().item()):
        raise ValueError("every D9 row requires at least one candidate")
    if not bool(torch.isfinite(patch_score).all().item()):
        raise ValueError("D9 patch score must contain only finite values")
    clip = _finite_float(clip, name="D9 patch-score clip")
    if clip <= 0.0:
        raise ValueError("D9 patch-score clip must be positive")

    count = candidate_mask.sum(dim=1).clamp_min(1).float()
    score = patch_score.float()
    safe = score.masked_fill(~candidate_mask, 0.0)
    mean = safe.sum(dim=1) / count

    # This first centered view computes exactly the deployment standard
    # deviation.  It is used only as a detached numeric reference below.
    centered_for_stats = (score - mean[:, None]).masked_fill(
        ~candidate_mask, 0.0
    )
    std = (
        (centered_for_stats.square().sum(dim=1) / count)
        .clamp_min(1e-6)
        .sqrt()
    )

    # Repeating the same subtraction preserves the forward value exactly,
    # while detaching mean/std localizes d standardized[q] / d score[j] to
    # j == q.  Clipping keeps the same STE used by deployment-aligned D2-D8.
    centered = (score - mean.detach()[:, None]).masked_fill(
        ~candidate_mask, 0.0
    )
    unbounded = centered / std.detach()[:, None]
    clipped = unbounded.clamp(min=-clip, max=clip)
    standardized = unbounded + (clipped - unbounded).detach()
    return standardized.masked_fill(~candidate_mask, -clip)


class StageBNativePatchCategoryD9Criterion(
    StageBNativePatchCategoryD8Criterion
):
    """D8 with loss-only detached row standardization statistics."""

    def __init__(
        self,
        *,
        detach_row_stats: bool = True,
        **kwargs: Any,
    ) -> None:
        if detach_row_stats is not True:
            raise ValueError("D9 requires detach_row_stats=True")
        super().__init__(**kwargs)
        self.detach_row_stats = True
        self.weight_dict = {NATIVE_PATCH_CATEGORY_D9_LOSS: self.weight}

    def _standardize_patch_score(
        self, patch_score: Tensor, candidate_mask: Tensor
    ) -> Tensor:
        return loss_gradient_localized_standardized_patch_score(
            patch_score, candidate_mask, clip=self.patch_score_clip
        )

    def forward(
        self,
        outputs: Mapping[str, Any],
        targets: Sequence[Mapping[str, Any]],
        cap_list: Optional[Sequence[Sequence[str]]] = None,
        captions: Optional[Sequence[str]] = None,
    ) -> dict[str, Tensor]:
        d8_outputs = super().forward(
            outputs,
            targets,
            cap_list=cap_list,
            captions=captions,
        )
        renamed: dict[str, Tensor] = {}
        for name, value in d8_outputs.items():
            if name == NATIVE_PATCH_CATEGORY_D8_LOSS:
                renamed[NATIVE_PATCH_CATEGORY_D9_LOSS] = value
                continue
            if not name.startswith(_D8_TELEMETRY_PREFIX):
                raise RuntimeError(f"D9 received an unknown D8 output: {name!r}")
            renamed[
                _D9_TELEMETRY_PREFIX
                + name.removeprefix(_D8_TELEMETRY_PREFIX)
            ] = value
        if NATIVE_PATCH_CATEGORY_D9_LOSS not in renamed:
            raise RuntimeError("D9 did not receive the D8 base loss")
        return renamed


__all__ = [
    "NATIVE_PATCH_CATEGORY_D9_CONTRACT_VERSION",
    "NATIVE_PATCH_CATEGORY_D9_LOSS",
    "NATIVE_PATCH_CATEGORY_D9_MARKER",
    "StageBNativePatchCategoryD9Criterion",
    "loss_gradient_localized_standardized_patch_score",
]
