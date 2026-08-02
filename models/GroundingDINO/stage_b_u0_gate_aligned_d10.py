"""Inference-aligned Stage-B category-gate training for the data-only composite.

D10 starts from the sealed D9 + R100/P50 composite and changes only the eight
patch-projection tensors.  R100 is a frozen selector, never a regression
target: it identifies category-negative queries that can win after the exact
Gap-2 patch eligibility rule.  Category-complete ground truth keeps the patch
branch responsible for category membership while R100 remains responsible for
within-category language ranking.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from groundingdino.util import box_ops

from .stage_b_data_driven_score import data_driven_category_gate_mask
from .stage_b_native_patch_category_d9 import (
    loss_gradient_localized_standardized_patch_score,
)


STAGE_B_U0_GATE_ALIGNED_D10_CONTRACT_VERSION = 10
# D10 deliberately reuses D9's audited, class-balanced, row-locked D2 corpus.
# This keeps data and support patches fixed so the only experimental variable
# is the frozen-R100/Gap-2-aligned objective.
STAGE_B_U0_GATE_ALIGNED_D10_MARKER = "stage_b_native_patch_category_d2"
STAGE_B_U0_GATE_ALIGNED_D10_LOSS = "loss_stage_b_u0_gate_aligned_d10"


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_targets(
    targets: Sequence[Mapping[str, Any]], *, batch_size: int
) -> None:
    if (
        not isinstance(targets, Sequence)
        or isinstance(targets, (str, bytes))
        or len(targets) != int(batch_size)
    ):
        raise ValueError("D10 targets must match the output batch")
    for row_index, target in enumerate(targets):
        if not isinstance(target, Mapping):
            raise ValueError(f"D10 target {row_index} must be a mapping")
        boxes = target.get("boxes")
        labels = target.get("labels")
        support_class = target.get("support_class")
        primary = target.get("primary_instance_mask")
        marker = target.get(STAGE_B_U0_GATE_ALIGNED_D10_MARKER)
        if (
            not torch.is_tensor(boxes)
            or not boxes.is_floating_point()
            or boxes.dim() != 2
            or tuple(boxes.shape[1:]) != (4,)
            or int(boxes.shape[0]) < 1
            or not bool(torch.isfinite(boxes).all().item())
        ):
            raise ValueError(f"D10 target {row_index} boxes must be finite (N,4)")
        instance_count = int(boxes.shape[0])
        if (
            not torch.is_tensor(labels)
            or labels.dtype != torch.int64
            or tuple(labels.shape) != (instance_count,)
        ):
            raise ValueError(f"D10 target {row_index} labels must be int64 (N,)")
        if (
            not torch.is_tensor(support_class)
            or support_class.dtype != torch.int64
            or support_class.numel() != 1
        ):
            raise ValueError(f"D10 target {row_index} requires one support class")
        if not bool((labels == support_class.reshape(-1)[0]).all().item()):
            raise ValueError(
                f"D10 target {row_index} must retain every and only same-class GT"
            )
        if (
            not torch.is_tensor(primary)
            or primary.dtype != torch.bool
            or tuple(primary.shape) != (instance_count,)
            or int(primary.sum().item()) != 1
        ):
            raise ValueError(f"D10 target {row_index} requires one primary instance")
        if not (
            torch.is_tensor(marker)
            and marker.dtype == torch.bool
            and tuple(marker.shape) == (1,)
            and bool(marker[0].item())
        ):
            raise ValueError("D10 requires the exact category-complete data marker")
        devices = {
            boxes.device,
            labels.device,
            support_class.device,
            primary.device,
            marker.device,
        }
        if len(devices) != 1:
            raise ValueError(f"D10 target {row_index} tensors must share one device")


def _active_softplus(value: Tensor, *, temperature: float) -> Tensor:
    return F.softplus(value / float(temperature))


class StageBU0GateAlignedD10Criterion(nn.Module):
    """Optimize the exact Gap-2 eligibility boundary under frozen R100 rank."""

    def __init__(
        self,
        *,
        weight: float = 1.0,
        positive_iou_threshold: float = 0.5,
        negative_iou_threshold: float = 0.3,
        gate_max_gap: float = 2.0,
        patch_score_clip: float = 5.0,
        keep_gap: float = 1.75,
        drop_gap: float = 2.25,
        drop_active_gap: float = 2.75,
        temperature: float = 0.25,
        max_rank_blockers: int = 4,
        drop_weight: float = 2.0,
        critical_keep_weight: float = 1.0,
        positive_active_gap: float = 1.25,
        positive_target_gap: float = 1.5,
        positive_barrier_weight: float = 2.0,
        instance_active_gap: float = 1.25,
        instance_target_gap: float = 1.5,
        instance_coverage_weight: float = 2.0,
    ) -> None:
        super().__init__()
        self.weight = _finite_float(weight, name="D10 loss weight")
        self.positive_iou_threshold = _finite_float(
            positive_iou_threshold, name="D10 positive IoU threshold"
        )
        self.negative_iou_threshold = _finite_float(
            negative_iou_threshold, name="D10 negative IoU threshold"
        )
        self.gate_max_gap = _finite_float(
            gate_max_gap, name="D10 gate max gap"
        )
        self.patch_score_clip = _finite_float(
            patch_score_clip, name="D10 patch-score clip"
        )
        self.keep_gap = _finite_float(keep_gap, name="D10 keep gap")
        self.drop_gap = _finite_float(drop_gap, name="D10 drop gap")
        self.drop_active_gap = _finite_float(
            drop_active_gap, name="D10 drop active gap"
        )
        self.temperature = _finite_float(temperature, name="D10 temperature")
        self.drop_weight = _finite_float(drop_weight, name="D10 drop weight")
        self.critical_keep_weight = _finite_float(
            critical_keep_weight, name="D10 critical keep weight"
        )
        self.positive_active_gap = _finite_float(
            positive_active_gap, name="D10 positive active gap"
        )
        self.positive_target_gap = _finite_float(
            positive_target_gap, name="D10 positive target gap"
        )
        self.positive_barrier_weight = _finite_float(
            positive_barrier_weight, name="D10 positive barrier weight"
        )
        self.instance_active_gap = _finite_float(
            instance_active_gap, name="D10 instance active gap"
        )
        self.instance_target_gap = _finite_float(
            instance_target_gap, name="D10 instance target gap"
        )
        self.instance_coverage_weight = _finite_float(
            instance_coverage_weight, name="D10 instance coverage weight"
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
            or not 0.0
            <= self.instance_active_gap
            < self.instance_target_gap
            < self.gate_max_gap
            or self.temperature <= 0.0
            or min(
                self.drop_weight,
                self.critical_keep_weight,
                self.positive_barrier_weight,
            )
            <= 0.0
            or self.instance_coverage_weight < 0.0
        ):
            raise ValueError("D10 criterion geometry is invalid")
        if (
            isinstance(max_rank_blockers, bool)
            or not isinstance(max_rank_blockers, int)
            or max_rank_blockers < 1
        ):
            raise ValueError("D10 max_rank_blockers must be a positive integer")
        self.max_rank_blockers = int(max_rank_blockers)
        self.weight_dict = {STAGE_B_U0_GATE_ALIGNED_D10_LOSS: self.weight}
        self.register_buffer(
            "criterion_contract_version",
            torch.as_tensor(
                STAGE_B_U0_GATE_ALIGNED_D10_CONTRACT_VERSION, dtype=torch.int64
            ),
            persistent=True,
        )

    def forward(
        self,
        outputs: Mapping[str, Any],
        targets: Sequence[Mapping[str, Any]],
    ) -> dict[str, Tensor]:
        if not isinstance(outputs, Mapping):
            raise ValueError("D10 outputs must be a mapping")
        patch_score = outputs.get("pred_logits_patch")
        if torch.is_tensor(patch_score) and patch_score.dim() == 3:
            if int(patch_score.shape[-1]) != 1:
                raise ValueError("D10 requires exactly one support-patch slot")
            patch_score = patch_score[..., 0]
        if (
            not torch.is_tensor(patch_score)
            or not patch_score.is_floating_point()
            or patch_score.dim() != 2
            or not bool(torch.isfinite(patch_score).all().item())
        ):
            raise ValueError("D10 pred_logits_patch must be finite floating (B,Q)")
        boxes = outputs.get("pred_boxes")
        rank_score = outputs.get("stage_b_u0_teacher_rank_score")
        candidate_mask = outputs.get("stage_b_u0_candidate_mask")
        if (
            not torch.is_tensor(boxes)
            or not boxes.is_floating_point()
            or boxes.dim() != 3
            or tuple(boxes.shape[:2]) != tuple(patch_score.shape)
            or int(boxes.shape[-1]) != 4
            or not bool(torch.isfinite(boxes).all().item())
        ):
            raise ValueError("D10 pred_boxes must be finite and patch-aligned")
        if (
            not torch.is_tensor(rank_score)
            or not rank_score.is_floating_point()
            or tuple(rank_score.shape) != tuple(patch_score.shape)
            or not bool(torch.isfinite(rank_score).all().item())
        ):
            raise ValueError("D10 requires finite patch-aligned frozen R100 scores")
        if (
            not torch.is_tensor(candidate_mask)
            or candidate_mask.dtype != torch.bool
            or tuple(candidate_mask.shape) != tuple(patch_score.shape)
            or candidate_mask.device != patch_score.device
            or bool((~candidate_mask.any(dim=1)).any().item())
        ):
            raise ValueError("D10 requires a valid boolean U0 candidate mask")
        if len({patch_score.device, boxes.device, rank_score.device}) != 1:
            raise ValueError("D10 output tensors must share one device")

        batch_size, _query_count = patch_score.shape
        _validate_targets(targets, batch_size=int(batch_size))
        rank = rank_score.detach().float()
        standardized = loss_gradient_localized_standardized_patch_score(
            patch_score, candidate_mask, clip=self.patch_score_clip
        )
        best = standardized.masked_fill(~candidate_mask, -torch.inf).amax(
            dim=1
        ).detach()
        gap = best[:, None] - standardized
        deployment_eligible, deployed_standardized = (
            data_driven_category_gate_mask(
                patch_score.detach(),
                candidate_mask,
                max_gap=self.gate_max_gap,
                clip=self.patch_score_clip,
            )
        )
        if not torch.equal(standardized.detach(), deployed_standardized):
            raise RuntimeError("D10 loss and deployment patch standardization drifted")

        drop_losses: list[Tensor] = []
        critical_keep_losses: list[Tensor] = []
        positive_barrier_losses: list[Tensor] = []
        instance_class_losses: dict[int, list[Tensor]] = defaultdict(list)
        valid_rows = 0
        deployed_positive_rows = 0
        deployed_negative_rows = 0
        deployed_neutral_rows = 0
        critical_rows = 0
        selected_blockers = 0
        blocker_target_met = 0
        critical_positive_kept = 0
        reachable_instances = 0
        skipped_instances = 0
        active_instances = 0
        instance_deployment_kept = 0
        instance_target_kept = 0

        candidates_xyxy = box_ops.box_cxcywh_to_xyxy(boxes.detach().float())
        for row_index, target in enumerate(targets):
            target_boxes = box_ops.box_cxcywh_to_xyxy(
                target["boxes"].detach().to(
                    device=boxes.device, dtype=torch.float32
                )
            )
            iou, _ = box_ops.box_iou(candidates_xyxy[row_index], target_boxes)
            positives_by_instance = iou >= self.positive_iou_threshold
            reachable = positives_by_instance.any(dim=0)
            category_positive = positives_by_instance.any(dim=1) & candidate_mask[
                row_index
            ]
            category_negative = (
                iou.amax(dim=1) < self.negative_iou_threshold
            ) & candidate_mask[row_index]
            reachable_instances += int(reachable.sum().item())
            skipped_instances += int((~reachable).sum().item())
            if not bool(category_positive.any().item()):
                continue
            valid_rows += 1

            row_rank = rank[row_index]
            row_gap = gap[row_index]
            row_eligible = deployment_eligible[row_index]
            with torch.no_grad():
                deployed_winner = int(
                    row_rank.masked_fill(~row_eligible, -torch.inf).argmax().item()
                )
                positive_query = int(
                    row_rank.masked_fill(~category_positive, -torch.inf)
                    .argmax()
                    .item()
                )
                winner_positive = bool(category_positive[deployed_winner].item())
                winner_negative = bool(category_negative[deployed_winner].item())
                if winner_positive:
                    deployed_positive_rows += 1
                elif winner_negative:
                    deployed_negative_rows += 1
                else:
                    deployed_neutral_rows += 1

                blocker_mask = (
                    category_negative
                    & (row_rank > row_rank[positive_query])
                    & (row_gap.detach() < self.drop_active_gap)
                )
                if winner_negative:
                    blocker_mask[deployed_winner] = True
                blocker_count = int(blocker_mask.sum().item())
                if blocker_count:
                    blocker_indices = row_rank.masked_fill(
                        ~blocker_mask, -torch.inf
                    ).topk(min(self.max_rank_blockers, blocker_count)).indices
                else:
                    blocker_indices = torch.empty(
                        (0,), dtype=torch.long, device=row_rank.device
                    )

            if int(blocker_indices.numel()):
                blocker_gaps = row_gap[blocker_indices]
                drop_losses.append(
                    _active_softplus(
                        self.drop_gap - blocker_gaps,
                        temperature=self.temperature,
                    ).mean()
                )
                selected_blockers += int(blocker_indices.numel())
                blocker_target_met += int(
                    (blocker_gaps.detach() >= self.drop_gap).sum().item()
                )
            if winner_negative or not bool(
                (row_eligible & category_positive).any().item()
            ):
                critical_rows += 1
                positive_gap = row_gap[positive_query]
                critical_keep_losses.append(
                    _active_softplus(
                        positive_gap - self.keep_gap,
                        temperature=self.temperature,
                    )
                )
                critical_positive_kept += int(
                    bool((positive_gap.detach() <= self.gate_max_gap).item())
                )
            elif winner_positive:
                positive_gap = row_gap[deployed_winner]
                if bool(
                    (positive_gap.detach() > self.positive_active_gap).item()
                ):
                    positive_barrier_losses.append(
                        _active_softplus(
                            positive_gap - self.positive_target_gap,
                            temperature=self.temperature,
                        )
                    )

            support_class_id = int(target["support_class"].reshape(-1)[0].item())
            for instance_index in torch.nonzero(
                reachable, as_tuple=False
            ).flatten():
                instance_positive = positives_by_instance[:, instance_index]
                with torch.no_grad():
                    instance_query = int(
                        row_rank.masked_fill(~instance_positive, -torch.inf)
                        .argmax()
                        .item()
                    )
                instance_gap = row_gap[instance_query]
                instance_deployment_kept += int(
                    bool((instance_gap.detach() <= self.gate_max_gap).item())
                )
                instance_target_kept += int(
                    bool((instance_gap.detach() <= self.instance_target_gap).item())
                )
                if bool(
                    (instance_gap.detach() > self.instance_active_gap).item()
                ):
                    active_instances += 1
                    instance_class_losses[support_class_id].append(
                        _active_softplus(
                            instance_gap - self.instance_target_gap,
                            temperature=self.temperature,
                        )
                    )

        zero = standardized[candidate_mask].sum() * 0.0
        drop_loss = torch.stack(drop_losses).mean() if drop_losses else zero
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
        class_means = [
            torch.stack(class_losses).mean()
            for class_losses in instance_class_losses.values()
        ]
        instance_coverage_loss = (
            torch.stack(class_means).mean() if class_means else zero
        )
        loss = (
            self.drop_weight * drop_loss
            + self.critical_keep_weight * critical_keep_loss
            + self.positive_barrier_weight * positive_barrier_loss
            + self.instance_coverage_weight * instance_coverage_loss
        )
        if valid_rows == 0:
            raise RuntimeError("D10 batch has no reachable category-positive query")
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError("D10 loss is not finite")

        metric = lambda value: patch_score.new_tensor(float(value)).detach()
        return {
            STAGE_B_U0_GATE_ALIGNED_D10_LOSS: loss,
            "stage_b_u0_gate_aligned_d10_valid_rows": metric(valid_rows),
            "stage_b_u0_gate_aligned_d10_deployed_positive_rows": metric(
                deployed_positive_rows
            ),
            "stage_b_u0_gate_aligned_d10_deployed_negative_rows": metric(
                deployed_negative_rows
            ),
            "stage_b_u0_gate_aligned_d10_deployed_neutral_rows": metric(
                deployed_neutral_rows
            ),
            "stage_b_u0_gate_aligned_d10_critical_rows": metric(critical_rows),
            "stage_b_u0_gate_aligned_d10_selected_blockers": metric(
                selected_blockers
            ),
            "stage_b_u0_gate_aligned_d10_blocker_target_met": metric(
                blocker_target_met
            ),
            "stage_b_u0_gate_aligned_d10_critical_positive_kept": metric(
                critical_positive_kept
            ),
            "stage_b_u0_gate_aligned_d10_reachable_instances": metric(
                reachable_instances
            ),
            "stage_b_u0_gate_aligned_d10_skipped_instances": metric(
                skipped_instances
            ),
            "stage_b_u0_gate_aligned_d10_active_instances": metric(
                active_instances
            ),
            "stage_b_u0_gate_aligned_d10_active_classes": metric(
                len(instance_class_losses)
            ),
            "stage_b_u0_gate_aligned_d10_instance_deployment_kept": metric(
                instance_deployment_kept
            ),
            "stage_b_u0_gate_aligned_d10_instance_target_kept": metric(
                instance_target_kept
            ),
            "stage_b_u0_gate_aligned_d10_drop_loss": drop_loss.detach(),
            "stage_b_u0_gate_aligned_d10_critical_keep_loss": (
                critical_keep_loss.detach()
            ),
            "stage_b_u0_gate_aligned_d10_positive_barrier_loss": (
                positive_barrier_loss.detach()
            ),
            "stage_b_u0_gate_aligned_d10_instance_coverage_loss": (
                instance_coverage_loss.detach()
            ),
        }


__all__ = [
    "STAGE_B_U0_GATE_ALIGNED_D10_CONTRACT_VERSION",
    "STAGE_B_U0_GATE_ALIGNED_D10_LOSS",
    "STAGE_B_U0_GATE_ALIGNED_D10_MARKER",
    "StageBU0GateAlignedD10Criterion",
]
