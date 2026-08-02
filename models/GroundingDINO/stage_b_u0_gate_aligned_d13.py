"""Frozen-composite, patch-only residual for the deployed Gap-2 gate.

D13 leaves D9, R100, P50, U0, the detector, and every shared projection
bitwise frozen.  A separate zero-initialized residual sees only patch-side
query/support interactions and the frozen standardized D9 patch score.  It may
change category eligibility, but it never changes text rank or confidence.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from groundingdino.util import box_ops

from .stage_b_u0_gate_aligned_d11 import _validate_targets


STAGE_B_U0_GATE_ALIGNED_D13_CONTRACT_VERSION = 13
STAGE_B_U0_GATE_ALIGNED_D13_LOSS = "loss_stage_b_u0_gate_aligned_d13"


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _hard_category_gate(
    patch_score: Tensor,
    rank_score: Tensor,
    candidate_mask: Tensor,
    *,
    max_gap: float,
) -> tuple[Tensor, Tensor]:
    """Apply the exact lexicographic U0 category gate."""
    best_patch = patch_score.masked_fill(~candidate_mask, -torch.inf).amax(
        dim=1, keepdim=True
    )
    eligible = candidate_mask & (
        best_patch - patch_score <= float(max_gap)
    )
    if bool((~eligible.any(dim=1)).any().item()):
        raise RuntimeError("D13 category gate produced an empty row")

    teacher_min = rank_score.masked_fill(~candidate_mask, torch.inf).amin(
        dim=1, keepdim=True
    )
    teacher_max = rank_score.masked_fill(~candidate_mask, -torch.inf).amax(
        dim=1, keepdim=True
    )
    below_teacher_min = torch.nextafter(
        teacher_min, torch.full_like(teacher_min, -torch.inf)
    )
    if not bool(torch.isfinite(below_teacher_min).all().item()):
        raise RuntimeError("D13 cannot construct a finite ineligible score")
    teacher_delta = (
        torch.where(candidate_mask, rank_score, teacher_max) - teacher_max
    )
    ineligible_score = below_teacher_min + teacher_delta
    return torch.where(eligible, rank_score, ineligible_score), eligible


class StageBU0GateAlignedD13PatchResidual(nn.Module):
    """Bounded category-score residual with no text-rank input."""

    def __init__(
        self,
        *,
        feature_dim: int = 256,
        hidden_dim: int = 64,
        residual_limit: float = 0.25,
        gate_max_gap: float = 2.0,
    ) -> None:
        super().__init__()
        if int(feature_dim) <= 0 or int(hidden_dim) <= 0:
            raise ValueError("D13 feature dimensions must be positive")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.residual_limit = _finite_float(
            residual_limit, name="D13 residual limit"
        )
        self.gate_max_gap = _finite_float(
            gate_max_gap, name="D13 gate max gap"
        )
        if self.residual_limit <= 0.0 or self.gate_max_gap <= 0.0:
            raise ValueError("D13 residual limit and gate gap must be positive")
        self.interaction_norm = nn.LayerNorm(self.feature_dim)
        self.trunk = nn.Sequential(
            nn.Linear(self.feature_dim + 1, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.output = nn.Linear(self.hidden_dim, 1, bias=False)
        nn.init.zeros_(self.output.weight)
        self.register_buffer(
            "contract_version",
            torch.as_tensor(
                STAGE_B_U0_GATE_ALIGNED_D13_CONTRACT_VERSION,
                dtype=torch.int64,
            ),
            persistent=True,
        )
        self.register_buffer(
            "contract_residual_limit",
            torch.as_tensor(self.residual_limit, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "contract_gate_max_gap",
            torch.as_tensor(self.gate_max_gap, dtype=torch.float32),
            persistent=True,
        )

    def trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.interaction_norm.parameters()) + tuple(
            self.trunk.parameters()
        ) + tuple(self.output.parameters())

    def forward(
        self,
        query_patch_feature: Tensor,
        support_patch_feature: Tensor,
        teacher_patch_score: Tensor,
        teacher_rank_score: Tensor,
        candidate_mask: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        if (
            query_patch_feature.dim() != 3
            or not query_patch_feature.is_floating_point()
            or int(query_patch_feature.shape[-1]) != self.feature_dim
        ):
            raise ValueError("D13 query patch feature must be floating (B,Q,D)")
        if (
            support_patch_feature.dim() != 2
            or not support_patch_feature.is_floating_point()
            or int(support_patch_feature.shape[-1]) != self.feature_dim
            or int(support_patch_feature.shape[0])
            != int(query_patch_feature.shape[0])
        ):
            raise ValueError("D13 support patch feature must be floating (B,D)")
        expected = tuple(query_patch_feature.shape[:2])
        if (
            teacher_patch_score.dim() != 2
            or teacher_rank_score.dim() != 2
            or not teacher_patch_score.is_floating_point()
            or not teacher_rank_score.is_floating_point()
            or tuple(teacher_patch_score.shape) != expected
            or tuple(teacher_rank_score.shape) != expected
        ):
            raise ValueError("D13 teacher scores must align with patch queries")
        if candidate_mask is None:
            mask = torch.ones_like(teacher_patch_score, dtype=torch.bool)
        else:
            mask = torch.as_tensor(
                candidate_mask,
                device=teacher_patch_score.device,
                dtype=torch.bool,
            )
        if tuple(mask.shape) != expected or bool((~mask.any(dim=1)).any().item()):
            raise ValueError("D13 candidate mask must be aligned and nonempty")
        tensors = (
            query_patch_feature,
            support_patch_feature,
            teacher_patch_score,
            teacher_rank_score,
        )
        if len({tensor.device for tensor in tensors}) != 1 or any(
            not bool(torch.isfinite(tensor).all().item()) for tensor in tensors
        ):
            raise ValueError("D13 inputs must be finite and share one device")

        query = query_patch_feature.detach()
        support = support_patch_feature.detach()
        teacher_patch = teacher_patch_score.detach().to(device=query.device)
        teacher_rank = teacher_rank_score.detach().to(device=query.device)
        interaction = query * support[:, None, :]
        features = torch.cat(
            (
                self.interaction_norm(interaction),
                teacher_patch.to(dtype=query.dtype).unsqueeze(-1),
            ),
            dim=-1,
        )
        raw = self.output(self.trunk(features)).squeeze(-1).float()
        limit = self.residual_limit
        residual = (limit * torch.tanh(raw / limit)).to(
            dtype=teacher_patch.dtype
        )
        residual = residual.masked_fill(~mask, 0.0)
        adapted_patch = teacher_patch + residual
        adapted_rank, eligible = _hard_category_gate(
            adapted_patch,
            teacher_rank,
            mask,
            max_gap=self.gate_max_gap,
        )
        teacher_gated_rank, teacher_eligible = _hard_category_gate(
            teacher_patch,
            teacher_rank,
            mask,
            max_gap=self.gate_max_gap,
        )
        return {
            "teacher_patch_score": teacher_patch,
            "patch_residual": residual,
            "patch_score": adapted_patch,
            "teacher_rank_score": teacher_rank,
            "teacher_gated_rank_score": teacher_gated_rank,
            "teacher_eligible_mask": teacher_eligible,
            "rank_score": adapted_rank,
            "eligible_mask": eligible,
            "candidate_mask": mask,
        }


def _candidate_category_masks(
    candidate_boxes: Tensor,
    targets: Sequence[Mapping[str, Any]],
    *,
    positive_iou_threshold: float,
    negative_iou_threshold: float,
) -> tuple[Tensor, Tensor]:
    candidates = box_ops.box_cxcywh_to_xyxy(candidate_boxes.detach().float())
    positive = torch.zeros(
        candidates.shape[:2], dtype=torch.bool, device=candidates.device
    )
    negative = torch.zeros_like(positive)
    for row_index, target in enumerate(targets):
        target_xyxy = box_ops.box_cxcywh_to_xyxy(
            target["boxes"].detach().to(
                device=candidates.device, dtype=torch.float32
            )
        )
        iou, _ = box_ops.box_iou(candidates[row_index], target_xyxy)
        best_iou = iou.amax(dim=1)
        positive[row_index] = best_iou >= float(positive_iou_threshold)
        negative[row_index] = best_iou < float(negative_iou_threshold)
    return positive, negative


class StageBU0GateAlignedD13Criterion(nn.Module):
    """Repair actionable category-gate errors and anchor correct teacher rows."""

    def __init__(
        self,
        *,
        weight: float = 1.0,
        positive_iou_threshold: float = 0.5,
        negative_iou_threshold: float = 0.3,
        gate_max_gap: float = 2.0,
        keep_gap: float = 1.95,
        drop_gap: float = 2.05,
        preserve_tolerance: float = 0.02,
        temperature: float = 0.05,
        keep_weight: float = 1.0,
        drop_weight: float = 1.0,
        preserve_weight: float = 4.0,
        residual_weight: float = 0.05,
    ) -> None:
        super().__init__()
        self.weight = _finite_float(weight, name="D13 loss weight")
        self.positive_iou_threshold = _finite_float(
            positive_iou_threshold, name="D13 positive IoU threshold"
        )
        self.negative_iou_threshold = _finite_float(
            negative_iou_threshold, name="D13 negative IoU threshold"
        )
        self.gate_max_gap = _finite_float(
            gate_max_gap, name="D13 gate max gap"
        )
        self.keep_gap = _finite_float(keep_gap, name="D13 keep gap")
        self.drop_gap = _finite_float(drop_gap, name="D13 drop gap")
        self.preserve_tolerance = _finite_float(
            preserve_tolerance, name="D13 preserve tolerance"
        )
        self.temperature = _finite_float(
            temperature, name="D13 temperature"
        )
        self.keep_weight = _finite_float(keep_weight, name="D13 keep weight")
        self.drop_weight = _finite_float(drop_weight, name="D13 drop weight")
        self.preserve_weight = _finite_float(
            preserve_weight, name="D13 preserve weight"
        )
        self.residual_weight = _finite_float(
            residual_weight, name="D13 residual weight"
        )
        if (
            self.weight <= 0.0
            or not 0.0
            <= self.negative_iou_threshold
            < self.positive_iou_threshold
            <= 1.0
            or not 0.0 <= self.keep_gap < self.gate_max_gap < self.drop_gap
            or self.preserve_tolerance < 0.0
            or self.temperature <= 0.0
            or min(self.keep_weight, self.drop_weight, self.preserve_weight)
            <= 0.0
            or self.residual_weight < 0.0
        ):
            raise ValueError("D13 criterion geometry is invalid")
        self.weight_dict = {STAGE_B_U0_GATE_ALIGNED_D13_LOSS: self.weight}
        self.register_buffer(
            "criterion_contract_version",
            torch.as_tensor(
                STAGE_B_U0_GATE_ALIGNED_D13_CONTRACT_VERSION,
                dtype=torch.int64,
            ),
            persistent=True,
        )

    def forward(
        self,
        outputs: Mapping[str, Any],
        targets: Sequence[Mapping[str, Any]],
    ) -> dict[str, Tensor]:
        adapted = outputs.get("stage_b_u0_d13_patch_score")
        teacher = outputs.get("stage_b_u0_d13_teacher_patch_score")
        residual = outputs.get("stage_b_u0_d13_patch_residual")
        rank = outputs.get("stage_b_u0_d13_teacher_rank_score")
        teacher_eligible = outputs.get("stage_b_u0_d13_teacher_eligible_mask")
        adapted_eligible = outputs.get("stage_b_u0_category_gate_eligible_mask")
        candidate_mask = outputs.get("stage_b_u0_candidate_mask")
        boxes = outputs.get("pred_boxes")
        score_tensors = (adapted, teacher, residual, rank)
        if not all(torch.is_tensor(value) for value in score_tensors):
            raise KeyError("D13 requires adapted/teacher patch and rank tensors")
        if (
            adapted.dim() != 2
            or not adapted.is_floating_point()
            or any(tuple(value.shape) != tuple(adapted.shape) for value in score_tensors)
            or any(not bool(torch.isfinite(value).all().item()) for value in score_tensors)
        ):
            raise ValueError("D13 score tensors must be finite aligned (B,Q)")
        masks = (teacher_eligible, adapted_eligible, candidate_mask)
        if any(
            not torch.is_tensor(mask)
            or mask.dtype != torch.bool
            or tuple(mask.shape) != tuple(adapted.shape)
            for mask in masks
        ):
            raise ValueError("D13 eligibility masks must be boolean and aligned")
        if bool((~candidate_mask.any(dim=1)).any().item()):
            raise ValueError("D13 candidate rows must be nonempty")
        if (
            not torch.is_tensor(boxes)
            or boxes.dim() != 3
            or tuple(boxes.shape[:2]) != tuple(adapted.shape)
            or int(boxes.shape[-1]) != 4
        ):
            raise ValueError("D13 boxes must align with patch scores")
        _validate_targets(targets, batch_size=int(adapted.shape[0]))

        positive, negative = _candidate_category_masks(
            boxes,
            targets,
            positive_iou_threshold=self.positive_iou_threshold,
            negative_iou_threshold=self.negative_iou_threshold,
        )
        positive &= candidate_mask
        negative &= candidate_mask
        adapted_best = adapted.masked_fill(~candidate_mask, -torch.inf).amax(
            dim=1
        )
        teacher_best = teacher.masked_fill(~candidate_mask, -torch.inf).amax(
            dim=1
        )
        adapted_gap = adapted_best[:, None] - adapted
        teacher_gap = teacher_best[:, None] - teacher

        keep_losses: list[Tensor] = []
        drop_losses: list[Tensor] = []
        preserve_losses: list[Tensor] = []
        valid_rows = 0
        teacher_correct_rows = 0
        actionable_rows = 0
        skipped_neutral_rows = 0
        selected_blockers = 0
        teacher_correct = torch.zeros(
            adapted.shape[0], dtype=torch.bool, device=adapted.device
        )
        adapted_correct = torch.zeros_like(teacher_correct)

        tau = self.temperature
        for row_index in range(int(adapted.shape[0])):
            if not bool(positive[row_index].any().item()):
                continue
            valid_rows += 1
            row_rank = rank[row_index]
            positive_query = int(
                row_rank.masked_fill(~positive[row_index], -torch.inf)
                .argmax()
                .item()
            )
            teacher_winner = int(
                row_rank.masked_fill(
                    ~teacher_eligible[row_index], -torch.inf
                )
                .argmax()
                .item()
            )
            adapted_winner = int(
                row_rank.masked_fill(
                    ~adapted_eligible[row_index], -torch.inf
                )
                .argmax()
                .item()
            )
            teacher_is_correct = bool(
                positive[row_index, teacher_winner].item()
            )
            adapted_is_correct = bool(
                positive[row_index, adapted_winner].item()
            )
            teacher_correct[row_index] = teacher_is_correct
            adapted_correct[row_index] = adapted_is_correct

            if teacher_is_correct:
                teacher_correct_rows += 1
                winner = teacher_winner
                keep_target = min(
                    float(teacher_gap[row_index, winner].detach().item())
                    + self.preserve_tolerance,
                    self.gate_max_gap,
                )
                preserve_losses.append(
                    F.relu(
                        adapted_gap[row_index, winner] - keep_target
                    )
                )
                higher_negative = negative[row_index] & (
                    row_rank > row_rank[winner]
                )
                for blocker in torch.nonzero(
                    higher_negative, as_tuple=False
                ).flatten():
                    blocker_index = int(blocker.item())
                    drop_target = max(
                        float(
                            teacher_gap[row_index, blocker_index]
                            .detach()
                            .item()
                        )
                        - self.preserve_tolerance,
                        self.gate_max_gap,
                    )
                    preserve_losses.append(
                        F.relu(
                            drop_target
                            - adapted_gap[row_index, blocker_index]
                        )
                    )
                continue

            positive_is_eligible = bool(
                (teacher_eligible[row_index] & positive[row_index])
                .any()
                .item()
            )
            eligible_blockers = (
                teacher_eligible[row_index]
                & negative[row_index]
                & (row_rank > row_rank[positive_query])
            )
            if positive_is_eligible and not bool(eligible_blockers.any().item()):
                skipped_neutral_rows += 1
                continue
            actionable_rows += 1
            keep_losses.append(
                tau
                * F.softplus(
                    (
                        adapted_gap[row_index, positive_query]
                        - self.keep_gap
                    )
                    / tau
                )
            )
            blocker_indices = torch.nonzero(
                eligible_blockers, as_tuple=False
            ).flatten()
            selected_blockers += int(blocker_indices.numel())
            for blocker in blocker_indices:
                drop_losses.append(
                    tau
                    * F.softplus(
                        (
                            self.drop_gap
                            - adapted_gap[row_index, int(blocker.item())]
                        )
                        / tau
                    )
                )

        zero = adapted[candidate_mask].sum() * 0.0
        keep_loss = torch.stack(keep_losses).mean() if keep_losses else zero
        drop_loss = torch.stack(drop_losses).mean() if drop_losses else zero
        preserve_loss = (
            torch.stack(preserve_losses).mean() if preserve_losses else zero
        )
        residual_l2 = residual[candidate_mask].float().square().mean()
        loss = (
            self.keep_weight * keep_loss
            + self.drop_weight * drop_loss
            + self.preserve_weight * preserve_loss
            + self.residual_weight * residual_l2
        )
        if valid_rows == 0 or not bool(torch.isfinite(loss).item()):
            raise RuntimeError("D13 batch has no valid row or a non-finite loss")

        metric = lambda value: adapted.new_tensor(float(value)).detach()
        return {
            STAGE_B_U0_GATE_ALIGNED_D13_LOSS: loss,
            "stage_b_u0_gate_aligned_d13_keep_loss": keep_loss.detach(),
            "stage_b_u0_gate_aligned_d13_drop_loss": drop_loss.detach(),
            "stage_b_u0_gate_aligned_d13_preserve_loss": preserve_loss.detach(),
            "stage_b_u0_gate_aligned_d13_residual_l2": residual_l2.detach(),
            "stage_b_u0_gate_aligned_d13_valid_rows": metric(valid_rows),
            "stage_b_u0_gate_aligned_d13_teacher_correct_rows": metric(
                teacher_correct_rows
            ),
            "stage_b_u0_gate_aligned_d13_actionable_rows": metric(
                actionable_rows
            ),
            "stage_b_u0_gate_aligned_d13_skipped_neutral_rows": metric(
                skipped_neutral_rows
            ),
            "stage_b_u0_gate_aligned_d13_selected_blockers": metric(
                selected_blockers
            ),
            "stage_b_u0_gate_aligned_d13_wrong_fixed": metric(
                int(((~teacher_correct) & adapted_correct).sum().item())
            ),
            "stage_b_u0_gate_aligned_d13_correct_regressed": metric(
                int((teacher_correct & (~adapted_correct)).sum().item())
            ),
        }


__all__ = [
    "STAGE_B_U0_GATE_ALIGNED_D13_CONTRACT_VERSION",
    "STAGE_B_U0_GATE_ALIGNED_D13_LOSS",
    "StageBU0GateAlignedD13Criterion",
    "StageBU0GateAlignedD13PatchResidual",
]
