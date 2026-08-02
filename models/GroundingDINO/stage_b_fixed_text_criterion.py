from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import torch
import torch.distributed as dist
from torch.distributed.nn import functional as dist_nn_functional
from torch import nn
import torch.nn.functional as F

from groundingdino.util import box_ops


def candidate_max_iou(
    candidate_boxes: torch.Tensor,
    targets: Sequence[Dict[str, torch.Tensor]],
    *,
    box_format: str = "cxcywh",
) -> torch.Tensor:
    """Return each candidate's maximum IoU with any target box in its image."""
    if candidate_boxes.dim() != 3 or candidate_boxes.shape[-1] != 4:
        raise ValueError(
            "candidate_boxes must have shape (B, N, 4), "
            f"got {tuple(candidate_boxes.shape)}"
        )
    if len(targets) != int(candidate_boxes.shape[0]):
        raise ValueError(
            f"targets must have B={candidate_boxes.shape[0]} entries, got {len(targets)}"
        )
    box_format = str(box_format).lower().strip()
    if box_format not in {"cxcywh", "xyxy"}:
        raise ValueError(f"box_format must be 'cxcywh' or 'xyxy', got {box_format!r}")

    boxes = candidate_boxes.detach().float()
    if box_format == "cxcywh":
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
    result = torch.zeros(
        candidate_boxes.shape[:2],
        dtype=torch.float32,
        device=candidate_boxes.device,
    )
    with torch.no_grad():
        for batch_idx, target in enumerate(targets):
            target_boxes = target.get("boxes")
            if not torch.is_tensor(target_boxes) or target_boxes.numel() == 0:
                continue
            target_boxes = target_boxes.to(device=boxes.device, dtype=torch.float32).reshape(-1, 4)
            if box_format == "cxcywh":
                target_boxes = box_ops.box_cxcywh_to_xyxy(target_boxes)
            ious, _ = box_ops.box_iou(boxes[batch_idx], target_boxes)
            if ious.numel() > 0:
                result[batch_idx] = ious.max(dim=1).values
    return result


def multi_positive_candidate_listwise_loss(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_mask: torch.Tensor,
    *,
    temperature: float = 0.2,
) -> torch.Tensor:
    """Multi-positive InfoNCE over positive and unambiguous negative candidates."""
    if logits.dim() != 1:
        raise ValueError(f"logits must be one-dimensional, got {tuple(logits.shape)}")
    if positive_mask.shape != logits.shape or negative_mask.shape != logits.shape:
        raise ValueError("positive_mask and negative_mask must match logits")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")

    positive_mask = positive_mask.to(device=logits.device, dtype=torch.bool)
    negative_mask = negative_mask.to(device=logits.device, dtype=torch.bool) & (~positive_mask)
    finite = torch.isfinite(logits)
    positive_values = logits[positive_mask & finite].float()
    negative_values = logits[negative_mask & finite].float()
    if positive_values.numel() == 0 or negative_values.numel() == 0:
        return _graph_zero(logits)

    tau = float(temperature)
    positive_log_mass = torch.logsumexp(positive_values / tau, dim=0)
    negative_log_mass = torch.logsumexp(negative_values / tau, dim=0)
    return torch.logaddexp(positive_log_mass, negative_log_mass) - positive_log_mass


def _graph_zero(*tensors: Optional[torch.Tensor]) -> torch.Tensor:
    zero: Optional[torch.Tensor] = None
    for tensor in tensors:
        if not torch.is_tensor(tensor) or not tensor.is_floating_point():
            continue
        # Multiply before reduction. Summing many finite padding sentinels such
        # as finfo(float32).min first would overflow to -inf, and -inf * 0 is NaN.
        finite_tensor = torch.where(torch.isfinite(tensor), tensor, torch.zeros_like(tensor))
        term = (finite_tensor * 0.0).float().sum()
        zero = term if zero is None else zero + term
    if zero is None:
        return torch.tensor(0.0)
    return zero


def _coerce_candidate_mask(
    mask: Optional[torch.Tensor],
    *,
    shape: torch.Size,
    device: torch.device,
    name: str,
) -> torch.Tensor:
    if mask is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    mask = torch.as_tensor(mask, device=device, dtype=torch.bool)
    if mask.dim() == 0:
        return mask.expand(shape)
    if mask.dim() == 1 and mask.shape[0] == shape[0]:
        return mask[:, None].expand(shape)
    if mask.dim() == 2 and mask.shape == (shape[0], 1):
        return mask.expand(shape)
    if mask.shape != shape:
        raise ValueError(f"{name} must broadcast to {tuple(shape)}, got {tuple(mask.shape)}")
    return mask


def _coerce_batch_mask(
    mask: Optional[torch.Tensor],
    *,
    batch_size: int,
    device: torch.device,
    name: str,
) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    mask = torch.as_tensor(mask, device=device, dtype=torch.bool)
    if mask.dim() == 0:
        return mask.expand(batch_size)
    mask = mask.reshape(-1)
    if mask.numel() != int(batch_size):
        raise ValueError(f"{name} must contain B={batch_size} values, got {mask.numel()}")
    return mask


def _normalized_logsumexp(values: torch.Tensor, temperature: float) -> torch.Tensor:
    values = values.float().reshape(-1)
    if values.numel() == 0:
        raise ValueError("normalized logsumexp requires at least one value")
    tau = float(temperature)
    return tau * torch.logsumexp(values / tau, dim=0) - tau * math.log(float(values.numel()))


def _exact_lower_tail_operating_threshold(
    values: torch.Tensor,
    lower_tail_fraction: float,
) -> torch.Tensor:
    """Match the evaluator's exact ``score >= threshold`` order statistic."""
    values = values.float().reshape(-1)
    if values.numel() == 0 or not bool(torch.isfinite(values).all().item()):
        raise ValueError("tail threshold values must be non-empty and finite")
    if not 0.0 <= float(lower_tail_fraction) < 1.0:
        raise ValueError("lower_tail_fraction must be in [0, 1)")
    target_tpr = 1.0 - float(lower_tail_fraction)
    accepted = max(1, int(math.ceil(target_tpr * int(values.numel()))))
    # kthvalue is one-indexed. Ties are accepted by the evaluator's >= rule.
    kth = int(values.numel()) - accepted + 1
    return torch.kthvalue(values, kth).values


def _mean_or_zero(values: List[torch.Tensor], zero: torch.Tensor) -> torch.Tensor:
    if not values:
        return zero
    return torch.stack([value.reshape(()) for value in values]).mean()


def _top_quarter_cvar(values: torch.Tensor) -> torch.Tensor:
    """Mean the largest quarter while retaining gradients to selected values."""
    values = values.float().reshape(-1)
    if values.numel() == 0:
        raise ValueError("top-quarter CVaR requires at least one value")
    count = max(1, int(math.ceil(0.25 * int(values.numel()))))
    indices = torch.topk(values.detach(), count, sorted=False).indices
    return values.gather(0, indices).mean()


def _fpr95_negative_softplus_loss(
    current_negative: torch.Tensor,
    *,
    surrogate_threshold: torch.Tensor,
    operating_threshold: torch.Tensor,
    temperature: float,
    margin: float,
    reduction_contract: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reduce the FPR95 queue loss with an exact operating-point active set."""
    current_negative = current_negative.float().reshape(-1)
    if current_negative.numel() == 0:
        raise ValueError("FPR95 negative reduction requires at least one value")
    if not bool(torch.isfinite(current_negative).all().item()):
        raise ValueError("FPR95 negative values must be finite")
    if surrogate_threshold.numel() != 1 or operating_threshold.numel() != 1:
        raise ValueError("FPR95 thresholds must be scalar tensors")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("FPR95 temperature must be finite and positive")
    reduction_contract = str(reduction_contract).strip().lower()
    if reduction_contract not in {
        "all_mean_v1",
        "exact_fpr95_active_set_mean_v1",
    }:
        raise ValueError("unknown FPR95 negative reduction contract")

    tau = float(temperature)
    per_example = tau * F.softplus(
        (current_negative - surrogate_threshold + float(margin)) / tau
    )
    # Match the evaluator exactly: scores tied with the positive q05 threshold
    # are accepted and therefore remain false accepts.
    active = current_negative.detach().ge(operating_threshold.detach())
    selected = (
        active
        if reduction_contract == "exact_fpr95_active_set_mean_v1"
        else torch.ones_like(active)
    )
    if bool(selected.any().item()):
        loss = per_example[selected].mean()
    else:
        loss = per_example.sum() * 0.0
    return loss, active, selected


def binary_focal_with_logits(
    logits: torch.Tensor,
    target: float,
    *,
    gamma: float = 0.0,
) -> torch.Tensor:
    """Binary focal loss without alpha weighting, reduced by the caller."""
    logits = logits.float()
    targets = torch.full_like(logits, float(target))
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if float(gamma) > 0.0:
        probability = logits.sigmoid()
        target_probability = probability * targets + (1.0 - probability) * (1.0 - targets)
        loss = loss * (1.0 - target_probability).pow(float(gamma))
    return loss


def fixed_batch_tail_separation_loss(
    positive_scores: torch.Tensor,
    positive_valid_mask: torch.Tensor,
    negative_scores: torch.Tensor,
    negative_valid_mask: torch.Tensor,
    *,
    positive_quantile: float,
    negative_quantile: float,
    margin: float,
    ddp_global: bool = False,
) -> torch.Tensor:
    """Tail separation over fixed-size per-sample score buffers.

    When enabled under DDP, every rank gathers the same fixed ``(B, 4)``
    payload regardless of how many valid positive or TN scores it owns. The
    differentiable gather backward already sums contributions from the loss
    replica on every rank; DDP's subsequent parameter-gradient average then
    yields the gradient of the single global-batch loss, so no extra world-size
    scaling is applied here.
    """
    if positive_scores.dim() != 1 or negative_scores.dim() != 1:
        raise ValueError("positive_scores and negative_scores must be one-dimensional")
    if positive_scores.shape != negative_scores.shape:
        raise ValueError("positive_scores and negative_scores must have the same fixed B")
    if positive_valid_mask.shape != positive_scores.shape:
        raise ValueError("positive_valid_mask must match positive_scores")
    if negative_valid_mask.shape != negative_scores.shape:
        raise ValueError("negative_valid_mask must match negative_scores")
    if not 0.0 <= float(positive_quantile) <= 1.0:
        raise ValueError("positive_quantile must be in [0, 1]")
    if not 0.0 <= float(negative_quantile) <= 1.0:
        raise ValueError("negative_quantile must be in [0, 1]")

    device = positive_scores.device
    positive_scores = positive_scores.float()
    negative_scores = negative_scores.to(device=device).float()
    positive_valid_mask = positive_valid_mask.to(device=device, dtype=torch.bool)
    negative_valid_mask = negative_valid_mask.to(device=device, dtype=torch.bool)

    # Packing masks with the scores avoids backend-specific bool collectives and
    # guarantees one unconditional collective per rank on the enabled path.
    payload = torch.stack(
        (
            positive_scores,
            positive_valid_mask.float(),
            negative_scores,
            negative_valid_mask.float(),
        ),
        dim=-1,
    )
    if (
        bool(ddp_global)
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    ):
        payload = torch.cat(tuple(dist_nn_functional.all_gather(payload)), dim=0)

    positive_scores = payload[:, 0]
    positive_valid_mask = payload[:, 1] > 0.5
    negative_scores = payload[:, 2]
    negative_valid_mask = payload[:, 3] > 0.5
    positive_values = positive_scores[
        positive_valid_mask & torch.isfinite(positive_scores)
    ]
    negative_values = negative_scores[
        negative_valid_mask & torch.isfinite(negative_scores)
    ]
    if positive_values.numel() == 0 or negative_values.numel() == 0:
        return _graph_zero(positive_scores, negative_scores)

    # Keep quantiles out of lower-precision autocast even when scorer logits are
    # produced in fp16/bf16.
    positive_tail = torch.quantile(positive_values.float(), float(positive_quantile))
    negative_tail = torch.quantile(negative_values.float(), float(negative_quantile))
    return F.softplus(negative_tail - positive_tail + float(margin))


class StageBFixedTextCriterion(nn.Module):
    """Losses for a text scorer operating on frozen Stage-A candidates.

    ``candidate_logits`` and ``local_tn_logits`` are the candidate-ranking
    scores. Optional confidence tensors drive absolute classification and FPR
    tail objectives; omitting them aliases each confidence input to its ranking
    counterpart. Image-global TN losses require an explicit per-image
    ``global_tn_verified`` flag. Paper Table-B weak-scope negatives use the
    separate ``confidence_ablation_eligible`` mask and never acquire global
    verification status.
    """

    def __init__(
        self,
        *,
        positive_iou_threshold: float = 0.5,
        negative_iou_threshold: float = 0.3,
        listwise_temperature: float = 0.2,
        listwise_weight: float = 1.0,
        local_tn_rank_margin: float = 0.3,
        local_tn_rank_weight: float = 1.0,
        predicate_tn_rank_margin: float = 0.3,
        predicate_tn_rank_weight: float = 0.0,
        local_anchor_weight: float = 0.5,
        positive_anchor_logit: float = 0.5,
        negative_anchor_logit: float = -0.5,
        global_tn_negative_weight: float = 1.0,
        global_tn_tail_weight: float = 1.0,
        global_tn_tail_topk: int = 10,
        global_tn_tail_temperature: float = 0.2,
        global_tn_tail_target_logit: float = 0.0,
        batch_tail_separation_weight: float = 0.0,
        batch_positive_quantile: float = 0.05,
        batch_negative_quantile: float = 0.95,
        batch_tail_margin: float = 0.3,
        balance_local_anchor_classes: bool = False,
        batch_tail_ddp_global: bool = False,
        local_absolute_weight: float = 0.0,
        local_absolute_gamma: float = 0.0,
        deployed_global_absolute_weight: float = 0.0,
        deployed_global_absolute_gamma: float = 0.0,
        predicate_absolute_weight: float = 0.0,
        predicate_absolute_gamma: float = 0.0,
        tail_queue_weight: float = 0.0,
        tail_queue_size: int = 0,
        tail_queue_min_count: int = 0,
        tail_queue_positive_quantile: float = 0.05,
        tail_queue_negative_quantile: float = 0.95,
        tail_queue_temperature: float = 0.1,
        tail_queue_margin: float = 0.3,
        tail_queue_global_scores: bool = False,
        tail_queue_objective: str = "cvar",
        tail_queue_pair_weight: float = 0.0,
        tail_queue_pair_margin: float = 0.0,
        tail_queue_positive_trust_weight: float = 0.0,
        tail_queue_positive_trust_margin: float = 0.02,
        tail_queue_positive_trust_reduction_contract: str = "mean_v1",
        tail_queue_positive_gradient_contract: str = "mean_translation_v1",
        tail_queue_negative_reduction_contract: str = "all_mean_v1",
        token_objective: str = "off",
        token_weight: float = 0.0,
        token_positive_weight: float = 1.0,
        token_shared_weight: float = 0.25,
        token_edit_weight: float = 1.0,
        token_edit_query_scope: str = "target_iou_v1",
        token_focal_alpha: float = 0.25,
        token_focal_gamma: float = 2.0,
        allow_legacy_token_diff_fallback: bool = False,
        raw_veto_gate_weight: float = 0.0,
        raw_veto_positive_margin: float = 0.1,
        raw_veto_tn_margin: float = 0.1,
        raw_veto_query_scope: str = "target_iou_v1",
        raw_veto_tn_carrier_balance: float = 0.0,
        raw_veto_positive_carrier_balance: float = 0.0,
        raw_veto_carrier_pair_weight: float = 0.0,
        raw_veto_carrier_pair_margin: float = 0.25,
        raw_veto_carrier_pair_gradient_contract: str = "bidirectional_v1",
        raw_veto_gate_offset: float = 0.0,
        raw_veto_gate_scale: float = 1.0,
        raw_veto_tail_quantile: float = 0.95,
        raw_veto_tail_temperature: float = 0.1,
        raw_veto_tail_min_count: int = 256,
        deployed_veto_routing_weight: float = 0.0,
        deployed_veto_positive_max: float = 0.1,
        deployed_veto_tn_min: float = 0.9,
        deployed_veto_routing_reduction_contract: str = "balanced_mean_v1",
    ) -> None:
        super().__init__()
        if float(positive_iou_threshold) <= float(negative_iou_threshold):
            raise ValueError("positive_iou_threshold must exceed negative_iou_threshold")
        if float(listwise_temperature) <= 0.0:
            raise ValueError("listwise_temperature must be positive")
        if int(global_tn_tail_topk) <= 0:
            raise ValueError("global_tn_tail_topk must be positive")
        if float(global_tn_tail_temperature) <= 0.0:
            raise ValueError("global_tn_tail_temperature must be positive")
        if float(positive_anchor_logit) <= float(negative_anchor_logit):
            raise ValueError("positive_anchor_logit must exceed negative_anchor_logit")
        if not 0.0 <= float(batch_positive_quantile) <= 1.0:
            raise ValueError("batch_positive_quantile must be in [0, 1]")
        if not 0.0 <= float(batch_negative_quantile) <= 1.0:
            raise ValueError("batch_negative_quantile must be in [0, 1]")
        if (
            float(local_absolute_gamma) < 0.0
            or float(deployed_global_absolute_gamma) < 0.0
            or float(predicate_absolute_gamma) < 0.0
        ):
            raise ValueError("absolute focal gamma values must be non-negative")
        if float(deployed_global_absolute_weight) < 0.0:
            raise ValueError(
                "deployed_global_absolute_weight must be non-negative"
            )
        if int(tail_queue_size) < 0 or int(tail_queue_min_count) < 0:
            raise ValueError("tail queue sizes must be non-negative")
        if int(tail_queue_min_count) > int(tail_queue_size):
            raise ValueError("tail_queue_min_count cannot exceed tail_queue_size")
        if not 0.0 < float(tail_queue_positive_quantile) < 1.0:
            raise ValueError("tail_queue_positive_quantile must be in (0, 1)")
        if not 0.0 < float(tail_queue_negative_quantile) < 1.0:
            raise ValueError("tail_queue_negative_quantile must be in (0, 1)")
        if float(tail_queue_temperature) <= 0.0:
            raise ValueError("tail_queue_temperature must be positive")
        if float(tail_queue_pair_weight) < 0.0:
            raise ValueError("tail_queue_pair_weight must be non-negative")
        if float(tail_queue_positive_trust_weight) < 0.0:
            raise ValueError("tail_queue_positive_trust_weight must be non-negative")
        if float(tail_queue_positive_trust_margin) < 0.0:
            raise ValueError("tail_queue_positive_trust_margin must be non-negative")
        tail_queue_positive_trust_reduction_contract = str(
            tail_queue_positive_trust_reduction_contract
        ).strip().lower()
        if tail_queue_positive_trust_reduction_contract not in {
            "mean_v1",
            "top_quarter_cvar_v2",
        }:
            raise ValueError(
                "tail_queue_positive_trust_reduction_contract must be "
                "'mean_v1' or 'top_quarter_cvar_v2'"
            )
        tail_queue_positive_gradient_contract = str(
            tail_queue_positive_gradient_contract
        ).strip().lower()
        if tail_queue_positive_gradient_contract not in {
            "mean_translation_v1",
            "exact_batch_lower_tail_st_v2",
            "mean_plus_exact_lower_tail_st_v3",
            "mean_plus_quarter_exact_lower_tail_st_v4",
            "bounded_mean_plus_sixteenth_exact_lower_tail_st_v5",
            "elementwise_bounded_mean_plus_sixteenth_exact_lower_tail_st_v6",
        }:
            raise ValueError(
                "tail_queue_positive_gradient_contract must be "
                "'mean_translation_v1', 'exact_batch_lower_tail_st_v2', or "
                "'mean_plus_exact_lower_tail_st_v3', or "
                "'mean_plus_quarter_exact_lower_tail_st_v4', or "
                "'bounded_mean_plus_sixteenth_exact_lower_tail_st_v5', or "
                "'elementwise_bounded_mean_plus_sixteenth_exact_lower_tail_st_v6'"
            )
        tail_queue_negative_reduction_contract = str(
            tail_queue_negative_reduction_contract
        ).strip().lower()
        if tail_queue_negative_reduction_contract not in {
            "all_mean_v1",
            "exact_fpr95_active_set_mean_v1",
        }:
            raise ValueError(
                "tail_queue_negative_reduction_contract must be "
                "'all_mean_v1' or 'exact_fpr95_active_set_mean_v1'"
            )
        token_objective = str(token_objective).strip().lower().replace("-", "_")
        if token_objective not in {
            "off",
            "gdino_allquery_allneg_focal",
            "allquery_allneg_focal",
            "targetlocal_allneg_bce",
            "edit_bce",
            "edit_focal",
            "edit_bce_group_balanced",
            "targetlocal_allneg_focal",
        }:
            raise ValueError(
                "token_objective must be one of 'off', "
                "'gdino_allquery_allneg_focal', 'allquery_allneg_focal', "
                "'targetlocal_allneg_focal', 'targetlocal_allneg_bce', "
                "'edit_focal', 'edit_bce', or 'edit_bce_group_balanced'"
            )
        if any(
            float(value) < 0.0
            for value in (
                token_weight,
                token_positive_weight,
                token_shared_weight,
                token_edit_weight,
                token_focal_gamma,
            )
        ):
            raise ValueError("token loss weights and focal gamma must be non-negative")
        if not -1.0 <= float(token_focal_alpha) <= 1.0:
            raise ValueError("token_focal_alpha must be in [-1, 1]")
        token_edit_query_scope = str(token_edit_query_scope).strip().lower()
        if token_edit_query_scope not in {
            "target_iou_v1",
            "target_iou_union_detached_final_confidence_base_argmax_v2",
            "target_iou_union_detached_role_complete_confidence_base_argmax_v3",
        }:
            raise ValueError(
                "token_edit_query_scope must be 'target_iou_v1', "
                "'target_iou_union_detached_final_confidence_base_argmax_v2', "
                "or 'target_iou_union_detached_role_complete_"
                "confidence_base_argmax_v3'"
            )
        if (
            token_edit_query_scope != "target_iou_v1"
            and token_objective
            not in {"edit_bce", "edit_focal", "edit_bce_group_balanced"}
        ):
            raise ValueError(
                "global-carrier token supervision requires an edit-aware token "
                "objective"
            )
        if float(raw_veto_gate_weight) < 0.0:
            raise ValueError("raw_veto_gate_weight must be non-negative")
        if (
            float(raw_veto_positive_margin) <= 0.0
            or float(raw_veto_tn_margin) <= 0.0
        ):
            raise ValueError("raw veto margins must be positive")
        raw_veto_query_scope = str(raw_veto_query_scope).strip().lower()
        if raw_veto_query_scope not in {
            "target_iou_v1",
            "tn_all_admitted_positive_carrier_v2",
            "tn_all_admitted_carrier_balanced_positive_carrier_v3",
            "tn_all_admitted_carrier_balanced_positive_carrier_paired_v4",
            "tn_all_admitted_dual_carrier_balanced_paired_v5",
            "tn_all_admitted_tail_weighted_carrier_positive_carrier_paired_v6",
            "tn_all_admitted_tail_weighted_carrier_tail_paired_v7",
        }:
            raise ValueError(
                "raw_veto_query_scope must be 'target_iou_v1', "
                "'tn_all_admitted_positive_carrier_v2', or "
                "'tn_all_admitted_carrier_balanced_positive_carrier_v3', or "
                "'tn_all_admitted_carrier_balanced_positive_carrier_paired_v4', "
                "or 'tn_all_admitted_dual_carrier_balanced_paired_v5'"
                ", or 'tn_all_admitted_tail_weighted_carrier_positive_carrier_paired_v6'"
                ", or 'tn_all_admitted_tail_weighted_carrier_tail_paired_v7'"
            )
        if (
            not math.isfinite(float(raw_veto_tn_carrier_balance))
            or not 0.0 <= float(raw_veto_tn_carrier_balance) <= 1.0
        ):
            raise ValueError("raw_veto_tn_carrier_balance must be in [0, 1]")
        carrier_balanced_scopes = {
            "tn_all_admitted_carrier_balanced_positive_carrier_v3",
            "tn_all_admitted_carrier_balanced_positive_carrier_paired_v4",
            "tn_all_admitted_dual_carrier_balanced_paired_v5",
            "tn_all_admitted_tail_weighted_carrier_positive_carrier_paired_v6",
            "tn_all_admitted_tail_weighted_carrier_tail_paired_v7",
        }
        if (
            raw_veto_query_scope in carrier_balanced_scopes
            and not 0.0 < float(raw_veto_tn_carrier_balance) < 1.0
        ):
            raise ValueError(
                "carrier-balanced raw veto supervision requires a balance in (0, 1)"
            )
        if (
            not math.isfinite(float(raw_veto_positive_carrier_balance))
            or not 0.0 <= float(raw_veto_positive_carrier_balance) <= 1.0
        ):
            raise ValueError(
                "raw_veto_positive_carrier_balance must be in [0, 1]"
            )
        if raw_veto_query_scope == (
            "tn_all_admitted_dual_carrier_balanced_paired_v5"
        ):
            if not 0.0 < float(raw_veto_positive_carrier_balance) < 1.0:
                raise ValueError(
                    "dual-carrier raw veto supervision requires a positive "
                    "carrier balance in (0, 1)"
                )
        elif float(raw_veto_positive_carrier_balance) != 0.0:
            raise ValueError(
                "positive carrier balance requires the dual-carrier v5 query scope"
            )
        if (
            not math.isfinite(float(raw_veto_carrier_pair_weight))
            or float(raw_veto_carrier_pair_weight) < 0.0
        ):
            raise ValueError(
                "raw_veto_carrier_pair_weight must be finite and non-negative"
            )
        if (
            not math.isfinite(float(raw_veto_carrier_pair_margin))
            or float(raw_veto_carrier_pair_margin) <= 0.0
        ):
            raise ValueError("raw_veto_carrier_pair_margin must be finite and positive")
        raw_veto_carrier_pair_gradient_contract = str(
            raw_veto_carrier_pair_gradient_contract
        ).strip().lower()
        if raw_veto_carrier_pair_gradient_contract not in {
            "bidirectional_v1",
            "tn_only_positive_detached_v2",
        }:
            raise ValueError(
                "raw_veto_carrier_pair_gradient_contract must be "
                "'bidirectional_v1' or 'tn_only_positive_detached_v2'"
            )
        if (
            raw_veto_carrier_pair_gradient_contract
            == "tn_only_positive_detached_v2"
            and float(raw_veto_carrier_pair_weight) <= 0.0
        ):
            raise ValueError(
                "TN-only carrier-pair gradients require a positive pair weight"
            )
        if (
            float(raw_veto_carrier_pair_weight) > 0.0
            and raw_veto_query_scope
            not in {
                "tn_all_admitted_carrier_balanced_positive_carrier_paired_v4",
                "tn_all_admitted_dual_carrier_balanced_paired_v5",
                "tn_all_admitted_tail_weighted_carrier_positive_carrier_paired_v6",
                "tn_all_admitted_tail_weighted_carrier_tail_paired_v7",
            }
        ):
            raise ValueError(
                "raw veto carrier pair loss requires a paired carrier query scope"
            )
        if (
            not math.isfinite(float(raw_veto_gate_offset))
            or float(raw_veto_gate_offset) < 0.0
            or not math.isfinite(float(raw_veto_gate_scale))
            or float(raw_veto_gate_scale) <= 0.0
        ):
            raise ValueError("raw veto gate calibration is invalid")
        if (
            raw_veto_query_scope in carrier_balanced_scopes
            and float(raw_veto_gate_offset) + float(raw_veto_gate_scale)
            > float(raw_veto_tn_margin) + 1e-8
        ):
            raise ValueError("raw veto carrier gate must open by the TN margin")
        if not 0.0 < float(raw_veto_tail_quantile) < 1.0:
            raise ValueError("raw veto tail quantile must be in (0, 1)")
        if (
            not math.isfinite(float(raw_veto_tail_temperature))
            or float(raw_veto_tail_temperature) <= 0.0
        ):
            raise ValueError("raw veto tail temperature must be finite and positive")
        if int(raw_veto_tail_min_count) < 0:
            raise ValueError("raw veto tail min count must be non-negative")
        if (
            not math.isfinite(float(deployed_veto_routing_weight))
            or float(deployed_veto_routing_weight) < 0.0
        ):
            raise ValueError(
                "deployed_veto_routing_weight must be finite and non-negative"
            )
        if (
            not math.isfinite(float(deployed_veto_positive_max))
            or not 0.0 <= float(deployed_veto_positive_max) <= 1.0
        ):
            raise ValueError("deployed_veto_positive_max must be in [0, 1]")
        if (
            not math.isfinite(float(deployed_veto_tn_min))
            or not 0.0 <= float(deployed_veto_tn_min) <= 1.0
        ):
            raise ValueError("deployed_veto_tn_min must be in [0, 1]")
        if float(deployed_veto_positive_max) >= float(deployed_veto_tn_min):
            raise ValueError(
                "deployed_veto_positive_max must be below deployed_veto_tn_min"
            )
        deployed_veto_routing_reduction_contract = str(
            deployed_veto_routing_reduction_contract
        ).strip().lower()
        if deployed_veto_routing_reduction_contract not in {
            "balanced_mean_v1",
            "balanced_top_quarter_cvar_v2",
        }:
            raise ValueError(
                "deployed_veto_routing_reduction_contract must be "
                "'balanced_mean_v1' or 'balanced_top_quarter_cvar_v2'"
            )
        tail_queue_objective = str(tail_queue_objective).strip().lower()
        if tail_queue_objective not in {"cvar", "fpr95"}:
            raise ValueError("tail_queue_objective must be 'cvar' or 'fpr95'")

        self.positive_iou_threshold = float(positive_iou_threshold)
        self.negative_iou_threshold = float(negative_iou_threshold)
        self.listwise_temperature = float(listwise_temperature)
        self.local_tn_rank_margin = float(local_tn_rank_margin)
        self.predicate_tn_rank_margin = float(predicate_tn_rank_margin)
        self.positive_anchor_logit = float(positive_anchor_logit)
        self.negative_anchor_logit = float(negative_anchor_logit)
        self.global_tn_tail_topk = int(global_tn_tail_topk)
        self.global_tn_tail_temperature = float(global_tn_tail_temperature)
        self.global_tn_tail_target_logit = float(global_tn_tail_target_logit)
        self.batch_positive_quantile = float(batch_positive_quantile)
        self.batch_negative_quantile = float(batch_negative_quantile)
        self.batch_tail_margin = float(batch_tail_margin)
        self.balance_local_anchor_classes = bool(balance_local_anchor_classes)
        self.batch_tail_ddp_global = bool(batch_tail_ddp_global)
        self.local_absolute_gamma = float(local_absolute_gamma)
        self.deployed_global_absolute_gamma = float(
            deployed_global_absolute_gamma
        )
        self.predicate_absolute_gamma = float(predicate_absolute_gamma)
        self.tail_queue_size = int(tail_queue_size)
        self.tail_queue_min_count = int(tail_queue_min_count)
        self.tail_queue_positive_quantile = float(tail_queue_positive_quantile)
        self.tail_queue_negative_quantile = float(tail_queue_negative_quantile)
        self.tail_queue_temperature = float(tail_queue_temperature)
        self.tail_queue_margin = float(tail_queue_margin)
        self.tail_queue_global_scores = bool(tail_queue_global_scores)
        self.tail_queue_objective = tail_queue_objective
        self.tail_queue_pair_weight = float(tail_queue_pair_weight)
        self.tail_queue_pair_margin = float(tail_queue_pair_margin)
        self.tail_queue_positive_trust_weight = float(
            tail_queue_positive_trust_weight
        )
        self.tail_queue_positive_trust_margin = float(
            tail_queue_positive_trust_margin
        )
        self.tail_queue_positive_trust_reduction_contract = (
            tail_queue_positive_trust_reduction_contract
        )
        self.tail_queue_positive_gradient_contract = (
            tail_queue_positive_gradient_contract
        )
        self.tail_queue_negative_reduction_contract = (
            tail_queue_negative_reduction_contract
        )
        self.token_objective = token_objective
        self.token_positive_weight = float(token_positive_weight)
        self.token_shared_weight = float(token_shared_weight)
        self.token_edit_weight = float(token_edit_weight)
        self.token_edit_query_scope = token_edit_query_scope
        self.token_focal_alpha = float(token_focal_alpha)
        self.token_focal_gamma = float(token_focal_gamma)
        self.allow_legacy_token_diff_fallback = bool(
            allow_legacy_token_diff_fallback
        )
        self.raw_veto_gate_weight = float(raw_veto_gate_weight)
        self.raw_veto_positive_margin = float(raw_veto_positive_margin)
        self.raw_veto_tn_margin = float(raw_veto_tn_margin)
        self.raw_veto_query_scope = raw_veto_query_scope
        self.raw_veto_tn_carrier_balance = float(raw_veto_tn_carrier_balance)
        self.raw_veto_positive_carrier_balance = float(
            raw_veto_positive_carrier_balance
        )
        self.raw_veto_carrier_pair_weight = float(raw_veto_carrier_pair_weight)
        self.raw_veto_carrier_pair_margin = float(raw_veto_carrier_pair_margin)
        self.raw_veto_carrier_pair_gradient_contract = (
            raw_veto_carrier_pair_gradient_contract
        )
        self.raw_veto_gate_offset = float(raw_veto_gate_offset)
        self.raw_veto_gate_scale = float(raw_veto_gate_scale)
        self.raw_veto_tail_quantile = float(raw_veto_tail_quantile)
        self.raw_veto_tail_temperature = float(raw_veto_tail_temperature)
        self.raw_veto_tail_min_count = int(raw_veto_tail_min_count)
        self.deployed_veto_routing_weight = float(deployed_veto_routing_weight)
        self.deployed_veto_positive_max = float(deployed_veto_positive_max)
        self.deployed_veto_tn_min = float(deployed_veto_tn_min)
        self.deployed_veto_routing_reduction_contract = (
            deployed_veto_routing_reduction_contract
        )
        self._pending_tail_payload: Optional[torch.Tensor] = None
        self._deferred_tail_payloads: list[torch.Tensor] = []
        if self.tail_queue_size > 0:
            self.register_buffer(
                "tail_positive_queue",
                torch.zeros(self.tail_queue_size, dtype=torch.float32),
                persistent=True,
            )
            self.register_buffer(
                "tail_negative_queue",
                torch.zeros(self.tail_queue_size, dtype=torch.float32),
                persistent=True,
            )
            self.register_buffer("tail_positive_ptr", torch.zeros((), dtype=torch.long))
            self.register_buffer("tail_negative_ptr", torch.zeros((), dtype=torch.long))
            self.register_buffer("tail_positive_count", torch.zeros((), dtype=torch.long))
            self.register_buffer("tail_negative_count", torch.zeros((), dtype=torch.long))
        self.weight_dict = {
            "loss_fixed_text_listwise": float(listwise_weight),
            "loss_fixed_text_local_tn_rank": float(local_tn_rank_weight),
            "loss_fixed_text_predicate_tn_rank": float(predicate_tn_rank_weight),
            "loss_fixed_text_local_anchor": float(local_anchor_weight),
            "loss_fixed_text_global_tn_negative": float(global_tn_negative_weight),
            "loss_fixed_text_global_tn_tail": float(global_tn_tail_weight),
            "loss_fixed_text_batch_tail": float(batch_tail_separation_weight),
            "loss_fixed_text_local_absolute": float(local_absolute_weight),
            "loss_fixed_text_deployed_global_absolute": float(
                deployed_global_absolute_weight
            ),
            "loss_fixed_text_predicate_absolute": float(predicate_absolute_weight),
            "loss_fixed_text_tail_queue": float(tail_queue_weight),
            "loss_fixed_text_token": float(token_weight),
            "loss_fixed_text_raw_veto_gate": float(raw_veto_gate_weight),
            "loss_fixed_text_raw_veto_carrier_pair": float(
                raw_veto_carrier_pair_weight
            ),
            "loss_fixed_text_deployed_veto_routing": float(
                deployed_veto_routing_weight
            ),
        }

    @property
    def tail_queue_enabled(self) -> bool:
        return self.tail_queue_size > 0

    @torch.no_grad()
    def _enqueue_tail_values(self, queue_name: str, values: torch.Tensor) -> None:
        if not self.tail_queue_enabled or values.numel() == 0:
            return
        queue = getattr(self, f"tail_{queue_name}_queue")
        ptr = getattr(self, f"tail_{queue_name}_ptr")
        count = getattr(self, f"tail_{queue_name}_count")
        values = values.detach().to(device=queue.device, dtype=torch.float32).reshape(-1)
        values = values[torch.isfinite(values)]
        if values.numel() == 0:
            return
        if values.numel() >= self.tail_queue_size:
            values = values[-self.tail_queue_size :]
        start = int(ptr.item())
        first = min(int(values.numel()), self.tail_queue_size - start)
        queue[start : start + first].copy_(values[:first])
        remaining = int(values.numel()) - first
        if remaining > 0:
            queue[:remaining].copy_(values[first:])
        ptr.fill_((start + int(values.numel())) % self.tail_queue_size)
        count.fill_(min(self.tail_queue_size, int(count.item()) + int(values.numel())))

    @torch.no_grad()
    def defer_tail_queue_payload(self) -> None:
        """Hold one micro-batch payload until its accumulated optimizer step."""
        payload = self._pending_tail_payload
        self._pending_tail_payload = None
        if payload is not None:
            self._deferred_tail_payloads.append(payload)

    @torch.no_grad()
    def commit_tail_queue(self, step_succeeded: bool) -> None:
        """Commit the gathered score payload only after a successful optimizer step."""
        payloads = list(self._deferred_tail_payloads)
        if self._pending_tail_payload is not None:
            payloads.append(self._pending_tail_payload)
        payload = (
            payloads[0]
            if len(payloads) == 1
            else (torch.cat(payloads, dim=0) if payloads else None)
        )
        self._deferred_tail_payloads.clear()
        self._pending_tail_payload = None
        if payload is None:
            return
        if not bool(step_succeeded):
            return
        positive_values = payload[:, 0][payload[:, 1] > 0.5]
        negative_values = payload[:, 2][payload[:, 3] > 0.5]
        self._enqueue_tail_values("positive", positive_values)
        self._enqueue_tail_values("negative", negative_values)

    def _tail_queue_values(self, queue_name: str) -> torch.Tensor:
        queue = getattr(self, f"tail_{queue_name}_queue")
        count = int(getattr(self, f"tail_{queue_name}_count").item())
        return queue[:count] if count < self.tail_queue_size else queue

    def forward(
        self,
        candidate_logits: torch.Tensor,
        candidate_ious: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
        *,
        local_tn_logits: Optional[torch.Tensor] = None,
        confidence_logits: Optional[torch.Tensor] = None,
        sample_positive_confidence_logits: Optional[torch.Tensor] = None,
        sample_tn_confidence_logits: Optional[torch.Tensor] = None,
        positive_confidence_gate_logits: Optional[torch.Tensor] = None,
        local_tn_confidence_logits: Optional[torch.Tensor] = None,
        local_tn_mask: Optional[torch.Tensor] = None,
        positive_predicate_logits: Optional[torch.Tensor] = None,
        local_tn_predicate_logits: Optional[torch.Tensor] = None,
        predicate_pair_valid: Optional[torch.Tensor] = None,
        global_tn_logits: Optional[torch.Tensor] = None,
        global_tn_confidence_logits: Optional[torch.Tensor] = None,
        global_tn_verified: Optional[torch.Tensor] = None,
        confidence_ablation_eligible: Optional[torch.Tensor] = None,
        confidence_tn_train_eligible: Optional[torch.Tensor] = None,
        global_tn_candidate_mask: Optional[torch.Tensor] = None,
        token_edit_carrier_logits: Optional[torch.Tensor] = None,
        token_role_carrier_logits: Optional[torch.Tensor] = None,
        token_logits: Optional[torch.Tensor] = None,
        score_token_mask: Optional[torch.Tensor] = None,
        predicate_token_mask: Optional[torch.Tensor] = None,
        expression_valid_mask: Optional[torch.Tensor] = None,
        token_supervision_valid: Optional[torch.Tensor] = None,
        token_positive_mask: Optional[torch.Tensor] = None,
        token_shared_mask: Optional[torch.Tensor] = None,
        token_changed_mask: Optional[torch.Tensor] = None,
        token_direct_trace_valid: Optional[torch.Tensor] = None,
        token_residual_logits: Optional[torch.Tensor] = None,
        score_word_group_ids: Optional[torch.Tensor] = None,
        positive_reference_base_logits: Optional[torch.Tensor] = None,
        confidence_veto_carrier_indices: Optional[torch.Tensor] = None,
        confidence_mismatch_gate: Optional[torch.Tensor] = None,
        confidence_veto_coverage: Optional[torch.Tensor] = None,
        confidence_base_logits: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if candidate_logits.dim() != 2:
            raise ValueError(
                f"candidate_logits must have shape (B, N), got {tuple(candidate_logits.shape)}"
            )
        if candidate_ious.shape != candidate_logits.shape:
            raise ValueError(
                "candidate_ious must match candidate_logits, "
                f"got {tuple(candidate_ious.shape)} vs {tuple(candidate_logits.shape)}"
            )
        if local_tn_logits is not None and local_tn_logits.shape != candidate_logits.shape:
            raise ValueError("local_tn_logits must match candidate_logits")
        if confidence_logits is not None and confidence_logits.shape != candidate_logits.shape:
            raise ValueError("confidence_logits must match candidate_logits")
        sample_confidence_logits = {
            "sample_positive_confidence_logits": sample_positive_confidence_logits,
            "sample_tn_confidence_logits": sample_tn_confidence_logits,
        }
        supplied_sample_logits = [
            name for name, value in sample_confidence_logits.items() if value is not None
        ]
        if supplied_sample_logits and len(supplied_sample_logits) != len(
            sample_confidence_logits
        ):
            raise ValueError(
                "sample positive and TN confidence logits must be supplied together"
            )
        for name, value in sample_confidence_logits.items():
            if value is None:
                continue
            if (
                not torch.is_tensor(value)
                or not value.is_floating_point()
                or value.dim() != 1
                or int(value.numel()) != int(candidate_logits.shape[0])
            ):
                raise ValueError(
                    f"{name} must be a floating tensor with shape (B,)"
                )
            if value.device != candidate_logits.device:
                raise ValueError(f"{name} must share the candidate-logit device")
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError(f"{name} must be finite")
        if positive_confidence_gate_logits is not None:
            if (
                not torch.is_tensor(positive_confidence_gate_logits)
                or not positive_confidence_gate_logits.is_floating_point()
                or positive_confidence_gate_logits.dim() != 1
                or int(positive_confidence_gate_logits.numel())
                != int(candidate_logits.shape[0])
            ):
                raise ValueError(
                    "positive_confidence_gate_logits must be a floating tensor "
                    "with shape (B,)"
                )
            if positive_confidence_gate_logits.device != candidate_logits.device:
                raise ValueError(
                    "positive_confidence_gate_logits must share the candidate-logit device"
                )
            if not bool(torch.isfinite(positive_confidence_gate_logits).all().item()):
                raise ValueError("positive_confidence_gate_logits must be finite")
        if (
            self.tail_queue_objective == "fpr95"
            and (
                self.tail_queue_positive_trust_weight > 0.0
                or self.tail_queue_positive_gradient_contract
                in {
                    "exact_batch_lower_tail_st_v2",
                    "mean_plus_exact_lower_tail_st_v3",
                    "mean_plus_quarter_exact_lower_tail_st_v4",
                    "bounded_mean_plus_sixteenth_exact_lower_tail_st_v5",
                    "elementwise_bounded_mean_plus_sixteenth_exact_lower_tail_st_v6",
                }
            )
            and positive_confidence_gate_logits is None
        ):
            raise ValueError(
                "positive_confidence_gate_logits is required when "
                "positive trust or lower-tail gradient routing is enabled"
            )
        if (
            local_tn_confidence_logits is not None
            and local_tn_confidence_logits.shape != candidate_logits.shape
        ):
            raise ValueError("local_tn_confidence_logits must match candidate_logits")
        if (positive_predicate_logits is None) != (local_tn_predicate_logits is None):
            raise ValueError(
                "positive_predicate_logits and local_tn_predicate_logits must be supplied together"
            )
        if positive_predicate_logits is not None:
            if positive_predicate_logits.shape != candidate_logits.shape:
                raise ValueError("positive_predicate_logits must match candidate_logits")
            if local_tn_predicate_logits.shape != candidate_logits.shape:
                raise ValueError("local_tn_predicate_logits must match candidate_logits")
            if predicate_pair_valid is None:
                raise ValueError(
                    "predicate_pair_valid is required with predicate token logits"
                )
        if global_tn_logits is not None and global_tn_logits.shape != candidate_logits.shape:
            raise ValueError("global_tn_logits must match candidate_logits")
        if (
            global_tn_confidence_logits is not None
            and global_tn_confidence_logits.shape != candidate_logits.shape
        ):
            raise ValueError("global_tn_confidence_logits must match candidate_logits")
        if token_edit_carrier_logits is not None:
            if (
                not torch.is_tensor(token_edit_carrier_logits)
                or not token_edit_carrier_logits.is_floating_point()
                or token_edit_carrier_logits.shape != candidate_logits.shape
            ):
                raise ValueError(
                    "token_edit_carrier_logits must be a floating tensor matching "
                    "candidate_logits"
                )
            if token_edit_carrier_logits.device != candidate_logits.device:
                raise ValueError(
                    "token_edit_carrier_logits must share the candidate-logit device"
                )
        if token_role_carrier_logits is not None:
            if (
                not torch.is_tensor(token_role_carrier_logits)
                or not token_role_carrier_logits.is_floating_point()
                or tuple(token_role_carrier_logits.shape)
                != (*tuple(candidate_logits.shape), 2)
            ):
                raise ValueError(
                    "token_role_carrier_logits must be a floating tensor with "
                    "shape (B,N,2) aligned to candidate_logits"
                )
            if token_role_carrier_logits.device != candidate_logits.device:
                raise ValueError(
                    "token_role_carrier_logits must share the candidate-logit device"
                )
        if (
            global_tn_logits is not None or global_tn_confidence_logits is not None
        ) and global_tn_verified is None:
            raise ValueError(
                "global_tn_verified is required when global TN logits are supplied; "
                "local counterfactuals must not be treated as image-global negatives"
            )
        deployed_veto_routing_enabled = self.deployed_veto_routing_weight > 0.0
        if deployed_veto_routing_enabled:
            routing_inputs = {
                "confidence_mismatch_gate": confidence_mismatch_gate,
                "confidence_veto_coverage": confidence_veto_coverage,
                "confidence_base_logits": confidence_base_logits,
                "candidate_mask": candidate_mask,
                "local_tn_mask": local_tn_mask,
                "expression_valid_mask": expression_valid_mask,
                "confidence_tn_train_eligible": confidence_tn_train_eligible,
            }
            missing = [name for name, value in routing_inputs.items() if value is None]
            if missing:
                raise ValueError(
                    "deployed veto routing supervision requires " + ", ".join(missing)
                )
            expected_query_shape = (*tuple(candidate_logits.shape), 2)
            for name, value in (
                ("confidence_mismatch_gate", confidence_mismatch_gate),
                ("confidence_base_logits", confidence_base_logits),
            ):
                if not torch.is_tensor(value) or not value.is_floating_point():
                    raise TypeError(f"{name} must be a floating tensor")
                if tuple(value.shape) != expected_query_shape:
                    raise ValueError(
                        f"{name} must have shape (B,N,2) aligned to candidate_logits"
                    )
                if value.device != candidate_logits.device:
                    raise ValueError(f"{name} must share the candidate-logit device")
                if not bool(torch.isfinite(value).all().item()):
                    raise ValueError(f"{name} must be finite")
            if (
                not torch.is_tensor(confidence_veto_coverage)
                or not confidence_veto_coverage.is_floating_point()
            ):
                raise TypeError("confidence_veto_coverage must be a floating tensor")
            if tuple(confidence_veto_coverage.shape) != (
                int(candidate_logits.shape[0]),
                2,
            ):
                raise ValueError("confidence_veto_coverage must have shape (B,2)")
            if confidence_veto_coverage.device != candidate_logits.device:
                raise ValueError(
                    "confidence_veto_coverage must share the candidate-logit device"
                )
            if not bool(torch.isfinite(confidence_veto_coverage).all().item()):
                raise ValueError("confidence_veto_coverage must be finite")
            if bool(
                (
                    (confidence_mismatch_gate < 0.0)
                    | (confidence_mismatch_gate > 1.0)
                )
                .any()
                .item()
            ):
                raise ValueError(
                    "confidence_mismatch_gate must be bounded in [0, 1]"
                )
            # Coverage is a non-negative convex sum of exact hard gates.
            # Float32 softmax reduction can round only its upper bound a few
            # ULPs above one, so retain a strict lower bound and a dtype-sized
            # upper tolerance.
            coverage_roundoff_tolerance = min(
                1.0e-5,
                8.0 * torch.finfo(confidence_veto_coverage.dtype).eps,
            )
            coverage_min = float(confidence_veto_coverage.detach().amin().item())
            coverage_max = float(confidence_veto_coverage.detach().amax().item())
            if (
                coverage_min < 0.0
                or coverage_max > 1.0 + coverage_roundoff_tolerance
            ):
                raise ValueError(
                    "confidence_veto_coverage must be bounded in [0, 1] "
                    "within reduction tolerance; "
                    f"min={coverage_min}, max={coverage_max}, "
                    f"tolerance={coverage_roundoff_tolerance}"
                )
            bounded_confidence_veto_coverage = confidence_veto_coverage.float()
            bounded_confidence_veto_coverage = (
                bounded_confidence_veto_coverage
                + (
                    bounded_confidence_veto_coverage.clamp(0.0, 1.0)
                    - bounded_confidence_veto_coverage
                ).detach()
            )
            expected_mask_shape = tuple(candidate_logits.shape)
            if tuple(torch.as_tensor(candidate_mask).shape) != expected_mask_shape:
                raise ValueError("candidate_mask must have exact shape (B,N)")
            if tuple(torch.as_tensor(local_tn_mask).shape) != expected_mask_shape:
                raise ValueError("local_tn_mask must have exact shape (B,N)")
            if tuple(torch.as_tensor(expression_valid_mask).shape) != (
                int(candidate_logits.shape[0]),
                2,
            ):
                raise ValueError("expression_valid_mask must have exact shape (B,2)")
            if tuple(torch.as_tensor(confidence_tn_train_eligible).shape) != (
                int(candidate_logits.shape[0]),
            ):
                raise ValueError(
                    "confidence_tn_train_eligible must have exact shape (B,)"
                )
        if confidence_ablation_eligible is not None and (
            global_tn_logits is None and global_tn_confidence_logits is None
        ):
            raise ValueError(
                "confidence_ablation_eligible requires global TN confidence logits"
            )
        if self.token_objective != "off":
            token_inputs = {
                "token_logits": token_logits,
                "score_token_mask": score_token_mask,
                "predicate_token_mask": predicate_token_mask,
                "expression_valid_mask": expression_valid_mask,
            }
            missing = [name for name, value in token_inputs.items() if value is None]
            if missing:
                raise ValueError(
                    f"token objective {self.token_objective!r} requires "
                    + ", ".join(missing)
                )
            edit_aware_objective = self.token_objective in {
                "edit_bce",
                "edit_focal",
                "edit_bce_group_balanced",
            }
            direct_role_inputs = {
                "token_positive_mask": token_positive_mask,
                "token_shared_mask": token_shared_mask,
                "token_changed_mask": token_changed_mask,
                "token_direct_trace_valid": token_direct_trace_valid,
            }
            direct_roles_supplied = any(
                value is not None for value in direct_role_inputs.values()
            )
            if direct_roles_supplied:
                missing_direct = [
                    name
                    for name, value in direct_role_inputs.items()
                    if value is None
                ]
                if missing_direct:
                    raise ValueError(
                        "direct trace token roles are incomplete: "
                        + ", ".join(missing_direct)
                    )
            if (
                edit_aware_objective
                and token_supervision_valid is None
                and not self.allow_legacy_token_diff_fallback
            ):
                raise ValueError(
                    f"{self.token_objective} requires token_supervision_valid provenance; "
                    "legacy token-diff fallback is disabled"
                )
            if (
                edit_aware_objective
                and not direct_roles_supplied
                and not self.allow_legacy_token_diff_fallback
            ):
                raise ValueError(
                    f"{self.token_objective} requires explicit direct trace token "
                    "roles; legacy token-diff fallback is disabled"
                )
            if edit_aware_objective and predicate_pair_valid is None:
                raise ValueError(
                    f"{self.token_objective} requires predicate_pair_valid for "
                    "fail-closed token alignment"
                )
            expected_prefix = (*candidate_logits.shape, 2)
            if token_logits.dim() != 4 or tuple(token_logits.shape[:3]) != expected_prefix:
                raise ValueError(
                    "token_logits must have shape (B,N,2,T) aligned with candidate_logits, "
                    f"got {tuple(token_logits.shape)}"
                )
            token_width = int(token_logits.shape[-1])
            expected_token_mask_shape = (
                int(candidate_logits.shape[0]),
                2,
                token_width,
            )
            if tuple(score_token_mask.shape) != expected_token_mask_shape:
                raise ValueError(
                    "score_token_mask must have shape (B,2,T), got "
                    f"{tuple(score_token_mask.shape)}"
                )
            if tuple(predicate_token_mask.shape) != expected_token_mask_shape:
                raise ValueError(
                    "predicate_token_mask must have shape (B,2,T), got "
                    f"{tuple(predicate_token_mask.shape)}"
                )
            if tuple(expression_valid_mask.shape) != (
                int(candidate_logits.shape[0]),
                2,
            ):
                raise ValueError(
                    "expression_valid_mask must have shape (B,2), got "
                    f"{tuple(expression_valid_mask.shape)}"
                )
            if direct_roles_supplied:
                for name, value in (
                    ("token_positive_mask", token_positive_mask),
                    ("token_shared_mask", token_shared_mask),
                    ("token_changed_mask", token_changed_mask),
                ):
                    if tuple(value.shape) != expected_token_mask_shape:
                        raise ValueError(
                            f"{name} must have shape (B,2,T), got "
                            f"{tuple(value.shape)}"
                        )
        raw_veto_supervision_enabled = (
            self.raw_veto_gate_weight > 0.0
            or self.raw_veto_carrier_pair_weight > 0.0
        )
        if raw_veto_supervision_enabled:
            raw_veto_inputs = {
                "token_residual_logits": token_residual_logits,
                "score_word_group_ids": score_word_group_ids,
                "score_token_mask": score_token_mask,
                "token_changed_mask": token_changed_mask,
                "token_supervision_valid": token_supervision_valid,
                "token_direct_trace_valid": token_direct_trace_valid,
                "confidence_tn_train_eligible": confidence_tn_train_eligible,
                "local_tn_mask": local_tn_mask,
            }
            missing = [
                name for name, value in raw_veto_inputs.items() if value is None
            ]
            if missing:
                raise ValueError(
                    "raw word-veto gate supervision requires " + ", ".join(missing)
                )
            if self.token_objective not in {
                "edit_bce",
                "edit_focal",
                "edit_bce_group_balanced",
            }:
                raise ValueError("raw word-veto gate supervision requires edit-aware tokens")
            if (
                token_residual_logits.dim() != 4
                or tuple(token_residual_logits.shape[:3])
                != (*candidate_logits.shape, 2)
            ):
                raise ValueError(
                    "token_residual_logits must have shape (B,N,2,T)"
                )
            raw_token_shape = (
                int(candidate_logits.shape[0]),
                2,
                int(token_residual_logits.shape[-1]),
            )
            if tuple(score_word_group_ids.shape) != raw_token_shape:
                raise ValueError("score_word_group_ids must have shape (B,2,T)")
            if score_word_group_ids.dtype != torch.long:
                raise TypeError("score_word_group_ids must be int64")
            if tuple(score_token_mask.shape) != raw_token_shape:
                raise ValueError("raw veto score_token_mask shape drifted")
            if tuple(token_changed_mask.shape) != raw_token_shape:
                raise ValueError("raw veto token_changed_mask shape drifted")
            if not bool(torch.isfinite(token_residual_logits).all().item()):
                raise ValueError("token_residual_logits must be finite")
            if self.raw_veto_query_scope == "tn_all_admitted_positive_carrier_v2":
                if positive_reference_base_logits is None:
                    raise ValueError(
                        "all-admitted raw veto supervision requires "
                        "positive_reference_base_logits"
                    )
                if tuple(positive_reference_base_logits.shape) != tuple(
                    candidate_logits.shape
                ):
                    raise ValueError(
                        "positive_reference_base_logits must match candidate_logits"
                    )
            if self.raw_veto_query_scope in {
                "tn_all_admitted_carrier_balanced_positive_carrier_v3",
                "tn_all_admitted_carrier_balanced_positive_carrier_paired_v4",
                "tn_all_admitted_dual_carrier_balanced_paired_v5",
                "tn_all_admitted_tail_weighted_carrier_positive_carrier_paired_v6",
                "tn_all_admitted_tail_weighted_carrier_tail_paired_v7",
            }:
                if confidence_veto_carrier_indices is None:
                    raise ValueError(
                        "carrier-balanced raw veto supervision requires "
                        "confidence_veto_carrier_indices"
                    )
                if tuple(confidence_veto_carrier_indices.shape) != (
                    int(candidate_logits.shape[0]),
                    2,
                ):
                    raise ValueError(
                        "confidence_veto_carrier_indices must have shape (B, 2)"
                    )
                if confidence_veto_carrier_indices.dtype != torch.long:
                    raise TypeError("confidence_veto_carrier_indices must be int64")
                if confidence_veto_carrier_indices.device != candidate_logits.device:
                    raise ValueError(
                        "confidence_veto_carrier_indices must share the candidate device"
                    )
                if bool(
                    (
                        (confidence_veto_carrier_indices < -1)
                        | (confidence_veto_carrier_indices >= candidate_logits.shape[1])
                    ).any().item()
                ):
                    raise ValueError("confidence veto carrier index is out of range")
        if self.tail_queue_enabled and self._pending_tail_payload is not None:
            raise RuntimeError(
                "tail queue payload from the previous forward was not committed or discarded"
            )

        device = candidate_logits.device
        batch_size, num_candidates = candidate_logits.shape
        shape = candidate_logits.shape
        confidence_logits = (
            candidate_logits if confidence_logits is None else confidence_logits
        )
        local_tn_confidence_logits = (
            local_tn_logits
            if local_tn_confidence_logits is None
            else local_tn_confidence_logits
        )
        global_tn_confidence_logits = (
            global_tn_logits
            if global_tn_confidence_logits is None
            else global_tn_confidence_logits
        )
        carrier_edit_scope = (
            self.token_edit_query_scope
            == "target_iou_union_detached_final_confidence_base_argmax_v2"
        )
        role_complete_carrier_scope = (
            self.token_edit_query_scope
            == "target_iou_union_detached_role_complete_confidence_base_argmax_v3"
        )
        if carrier_edit_scope and token_edit_carrier_logits is None:
            raise ValueError(
                "carrier changed-token supervision requires query-specific "
                "token_edit_carrier_logits"
            )
        if role_complete_carrier_scope and token_role_carrier_logits is None:
            raise ValueError(
                "role-complete carrier supervision requires query-specific "
                "token_role_carrier_logits"
            )
        zero = _graph_zero(
            candidate_logits,
            local_tn_logits,
            positive_predicate_logits,
            local_tn_predicate_logits,
            global_tn_logits,
        )
        token_zero = (
            _graph_zero(token_logits)
            if self.token_objective != "off"
            else zero
        )
        raw_veto_zero = (
            _graph_zero(token_residual_logits)
            if raw_veto_supervision_enabled
            else zero
        )
        confidence_zero = _graph_zero(
            confidence_logits,
            local_tn_confidence_logits,
            global_tn_confidence_logits,
            positive_confidence_gate_logits,
            sample_positive_confidence_logits,
            sample_tn_confidence_logits,
        )
        deployed_veto_zero = (
            _graph_zero(confidence_mismatch_gate, confidence_veto_coverage)
            if deployed_veto_routing_enabled
            else zero
        )
        loss_deployed_veto_routing = deployed_veto_zero
        deployed_veto_winner_loss = deployed_veto_zero
        deployed_veto_coverage_loss = deployed_veto_zero
        empty_deployed_veto_values = deployed_veto_zero.reshape(1)[:0]
        deployed_veto_positive_winner_gates = empty_deployed_veto_values
        deployed_veto_tn_winner_gates = empty_deployed_veto_values
        deployed_veto_positive_coverages = empty_deployed_veto_values
        deployed_veto_tn_coverages = empty_deployed_veto_values
        deployed_veto_positive_winner_hinges = empty_deployed_veto_values
        deployed_veto_tn_winner_hinges = empty_deployed_veto_values
        deployed_veto_positive_coverage_hinges = empty_deployed_veto_values
        deployed_veto_tn_coverage_hinges = empty_deployed_veto_values
        admitted = _coerce_candidate_mask(
            candidate_mask,
            shape=shape,
            device=device,
            name="candidate_mask",
        )
        candidate_ious = candidate_ious.to(device=device, dtype=torch.float32)
        positive_mask = (
            admitted
            & torch.isfinite(candidate_logits)
            & torch.isfinite(candidate_ious)
            & (candidate_ious >= self.positive_iou_threshold)
        )
        negative_mask = (
            admitted
            & torch.isfinite(candidate_logits)
            & torch.isfinite(candidate_ious)
            & (candidate_ious <= self.negative_iou_threshold)
        )
        confidence_positive_mask = (
            admitted
            & torch.isfinite(confidence_logits)
            & torch.isfinite(candidate_ious)
            & (candidate_ious >= self.positive_iou_threshold)
        )

        listwise_losses: List[torch.Tensor] = []
        local_rank_losses: List[torch.Tensor] = []
        predicate_rank_losses: List[torch.Tensor] = []
        predicate_score_gaps: List[torch.Tensor] = []
        predicate_win_rates: List[torch.Tensor] = []
        local_anchor_losses: List[torch.Tensor] = []
        positive_anchor_losses: List[torch.Tensor] = []
        tn_anchor_losses: List[torch.Tensor] = []
        absolute_positive_losses: List[torch.Tensor] = []
        absolute_tn_losses: List[torch.Tensor] = []
        deployed_global_absolute_positive_losses: List[torch.Tensor] = []
        deployed_global_absolute_tn_losses: List[torch.Tensor] = []
        predicate_absolute_positive_losses: List[torch.Tensor] = []
        predicate_absolute_tn_losses: List[torch.Tensor] = []
        global_negative_losses: List[torch.Tensor] = []
        global_tail_losses: List[torch.Tensor] = []
        batch_positive_scores: List[torch.Tensor] = []
        batch_negative_scores: List[torch.Tensor] = []
        fixed_positive_scores: List[torch.Tensor] = [confidence_zero] * batch_size
        fixed_negative_scores: List[torch.Tensor] = [confidence_zero] * batch_size
        fixed_positive_valid = [False] * batch_size
        fixed_negative_valid = [False] * batch_size

        valid_listwise_count = 0
        local_pair_sample_count = 0
        local_pair_query_count = 0
        predicate_pair_sample_count = 0
        predicate_pair_query_count = 0
        local_anchor_sample_count = 0
        local_anchor_paired_sample_count = 0
        local_anchor_positive_query_count = 0
        local_anchor_tn_query_count = 0
        global_sample_count = 0
        global_candidate_count = 0
        local_absolute_positive_sample_count = 0
        local_absolute_tn_sample_count = 0
        predicate_absolute_sample_count = 0

        local_valid = None
        if local_tn_logits is not None:
            local_valid = _coerce_candidate_mask(
                local_tn_mask,
                shape=shape,
                device=device,
                name="local_tn_mask",
            ) & torch.isfinite(local_tn_logits)

        local_confidence_valid = None
        if local_tn_confidence_logits is not None:
            local_confidence_valid = _coerce_candidate_mask(
                local_tn_mask,
                shape=shape,
                device=device,
                name="local_tn_mask",
            ) & torch.isfinite(local_tn_confidence_logits)

        predicate_valid = _coerce_batch_mask(
            predicate_pair_valid,
            batch_size=batch_size,
            device=device,
            name="predicate_pair_valid",
        )

        verified = _coerce_batch_mask(
            global_tn_verified,
            batch_size=batch_size,
            device=device,
            name="global_tn_verified",
        )
        ablation_eligible = _coerce_batch_mask(
            confidence_ablation_eligible,
            batch_size=batch_size,
            device=device,
            name="confidence_ablation_eligible",
        )
        tn_train_eligible = _coerce_batch_mask(
            confidence_tn_train_eligible,
            batch_size=batch_size,
            device=device,
            name="confidence_tn_train_eligible",
        )
        if (
            verified is not None
            and ablation_eligible is not None
            and bool((verified & ablation_eligible).any().item())
        ):
            raise ValueError(
                "global_tn_verified and confidence_ablation_eligible must be disjoint"
            )
        confidence_negative_eligible = verified
        if ablation_eligible is not None:
            confidence_negative_eligible = (
                ablation_eligible
                if confidence_negative_eligible is None
                else confidence_negative_eligible | ablation_eligible
            )
        confidence_negative_source_eligible = confidence_negative_eligible
        if tn_train_eligible is not None:
            confidence_negative_eligible = (
                tn_train_eligible
                if confidence_negative_eligible is None
                else confidence_negative_eligible & tn_train_eligible
            )
            if local_confidence_valid is not None:
                local_confidence_valid &= tn_train_eligible[:, None]
        if deployed_veto_routing_enabled:
            routing_expression_valid = expression_valid_mask.to(
                device=device, dtype=torch.bool
            )
            routing_tn_admitted = _coerce_candidate_mask(
                local_tn_mask,
                shape=shape,
                device=device,
                name="local_tn_mask",
            )
            positive_has_candidate = admitted.any(dim=1)
            tn_has_candidate = routing_tn_admitted.any(dim=1)
            positive_routing_valid = (
                routing_expression_valid[:, 0] & positive_has_candidate
            )
            tn_routing_valid = (
                routing_expression_valid[:, 1]
                & tn_train_eligible
                & tn_has_candidate
            )

            detached_base = confidence_base_logits.detach().float()
            positive_winner_indices = detached_base[..., 0].masked_fill(
                ~admitted, float("-inf")
            ).argmax(dim=1)
            tn_winner_indices = detached_base[..., 1].masked_fill(
                ~routing_tn_admitted, float("-inf")
            ).argmax(dim=1)
            live_mismatch_gate = confidence_mismatch_gate.float()
            positive_winner_gates = live_mismatch_gate[..., 0].gather(
                1, positive_winner_indices[:, None]
            )[:, 0]
            tn_winner_gates = live_mismatch_gate[..., 1].gather(
                1, tn_winner_indices[:, None]
            )[:, 0]
            deployed_veto_positive_winner_gates = positive_winner_gates[
                positive_routing_valid
            ]
            deployed_veto_tn_winner_gates = tn_winner_gates[tn_routing_valid]
            deployed_veto_positive_coverages = bounded_confidence_veto_coverage[
                :, 0
            ][positive_routing_valid]
            deployed_veto_tn_coverages = bounded_confidence_veto_coverage[:, 1][
                tn_routing_valid
            ]

            deployed_veto_positive_winner_hinges = F.relu(
                deployed_veto_positive_winner_gates
                - self.deployed_veto_positive_max
            )
            deployed_veto_tn_winner_hinges = F.relu(
                self.deployed_veto_tn_min - deployed_veto_tn_winner_gates
            )
            deployed_veto_positive_coverage_hinges = F.relu(
                deployed_veto_positive_coverages
                - self.deployed_veto_positive_max
            )
            deployed_veto_tn_coverage_hinges = F.relu(
                self.deployed_veto_tn_min - deployed_veto_tn_coverages
            )

            def _reduce_deployed_hinges(values: torch.Tensor) -> torch.Tensor:
                if self.deployed_veto_routing_reduction_contract == (
                    "balanced_top_quarter_cvar_v2"
                ):
                    return _top_quarter_cvar(values)
                return values.mean()

            deployed_veto_components: List[torch.Tensor] = []
            if (
                deployed_veto_positive_winner_hinges.numel() > 0
                and deployed_veto_tn_winner_hinges.numel() > 0
            ):
                deployed_veto_winner_loss = (
                    0.5
                    * _reduce_deployed_hinges(
                        deployed_veto_positive_winner_hinges
                    )
                    + 0.5
                    * _reduce_deployed_hinges(deployed_veto_tn_winner_hinges)
                )
            elif deployed_veto_positive_winner_hinges.numel() > 0:
                deployed_veto_winner_loss = _reduce_deployed_hinges(
                    deployed_veto_positive_winner_hinges
                )
            elif deployed_veto_tn_winner_hinges.numel() > 0:
                deployed_veto_winner_loss = _reduce_deployed_hinges(
                    deployed_veto_tn_winner_hinges
                )
            else:
                deployed_veto_winner_loss = deployed_veto_zero
            if (
                deployed_veto_positive_winner_hinges.numel() > 0
                or deployed_veto_tn_winner_hinges.numel() > 0
            ):
                deployed_veto_components.append(deployed_veto_winner_loss)

            if (
                deployed_veto_positive_coverage_hinges.numel() > 0
                and deployed_veto_tn_coverage_hinges.numel() > 0
            ):
                deployed_veto_coverage_loss = (
                    0.5
                    * _reduce_deployed_hinges(
                        deployed_veto_positive_coverage_hinges
                    )
                    + 0.5
                    * _reduce_deployed_hinges(deployed_veto_tn_coverage_hinges)
                )
            elif deployed_veto_positive_coverage_hinges.numel() > 0:
                deployed_veto_coverage_loss = _reduce_deployed_hinges(
                    deployed_veto_positive_coverage_hinges
                )
            elif deployed_veto_tn_coverage_hinges.numel() > 0:
                deployed_veto_coverage_loss = _reduce_deployed_hinges(
                    deployed_veto_tn_coverage_hinges
                )
            else:
                deployed_veto_coverage_loss = deployed_veto_zero
            if (
                deployed_veto_positive_coverage_hinges.numel() > 0
                or deployed_veto_tn_coverage_hinges.numel() > 0
            ):
                deployed_veto_components.append(deployed_veto_coverage_loss)

            if deployed_veto_components:
                loss_deployed_veto_routing = torch.stack(
                    deployed_veto_components
                ).mean()
        token_provenance_valid = _coerce_batch_mask(
            token_supervision_valid,
            batch_size=batch_size,
            device=device,
            name="token_supervision_valid",
        )
        direct_trace_valid = _coerce_batch_mask(
            token_direct_trace_valid,
            batch_size=batch_size,
            device=device,
            name="token_direct_trace_valid",
        )
        carrier_edit_admitted = None
        carrier_edit_eligible = None
        if carrier_edit_scope:
            if verified is None or tn_train_eligible is None:
                raise ValueError(
                    "carrier changed-token supervision requires global_tn_verified "
                    "and confidence_tn_train_eligible"
                )
            if global_tn_candidate_mask is None:
                raise ValueError(
                    "carrier changed-token supervision requires the explicit "
                    "global_tn_candidate_mask"
                )
            carrier_edit_eligible = verified & tn_train_eligible
            carrier_edit_admitted = _coerce_candidate_mask(
                global_tn_candidate_mask,
                shape=shape,
                device=device,
                name="global_tn_candidate_mask",
            ) & torch.isfinite(token_edit_carrier_logits)
        role_carrier_positive_admitted = None
        role_carrier_tn_admitted = None
        role_carrier_eligible = None
        if role_complete_carrier_scope:
            if verified is None or tn_train_eligible is None:
                raise ValueError(
                    "role-complete carrier supervision requires global_tn_verified "
                    "and confidence_tn_train_eligible"
                )
            if global_tn_candidate_mask is None:
                raise ValueError(
                    "role-complete carrier supervision requires the explicit "
                    "global_tn_candidate_mask"
                )
            role_carrier_eligible = verified & tn_train_eligible
            role_carrier_positive_admitted = admitted & torch.isfinite(
                token_role_carrier_logits[..., 0]
            )
            role_carrier_tn_admitted = _coerce_candidate_mask(
                global_tn_candidate_mask,
                shape=shape,
                device=device,
                name="global_tn_candidate_mask",
            ) & torch.isfinite(token_role_carrier_logits[..., 1])

        loss_token = token_zero
        token_sample_count = 0
        token_target_query_count = 0
        token_positive_count = 0
        token_shared_count = 0
        token_edit_count = 0
        token_edit_query_count = 0
        token_edit_carrier_selected_count = 0
        token_edit_carrier_added_count = 0
        token_edit_carrier_target_overlap_count = 0
        token_role_carrier_pair_selected_count = 0
        token_role_carrier_positive_added_count = 0
        token_role_carrier_tn_added_count = 0
        token_role_carrier_positive_target_overlap_count = 0
        token_role_carrier_tn_target_overlap_count = 0
        token_all_negative_count = 0
        token_provenance_valid_count = (
            int(token_provenance_valid.sum().item())
            if token_provenance_valid is not None
            else 0
        )
        token_direct_trace_valid_count = (
            int(direct_trace_valid.sum().item())
            if direct_trace_valid is not None
            else 0
        )
        if self.token_objective != "off":
            token_logits = token_logits.float()
            score_token_mask = score_token_mask.to(device=device, dtype=torch.bool)
            predicate_token_mask = predicate_token_mask.to(
                device=device, dtype=torch.bool
            )
            expression_valid_mask = expression_valid_mask.to(
                device=device, dtype=torch.bool
            )
            direct_roles_supplied = token_positive_mask is not None
            if direct_roles_supplied:
                token_positive_mask = token_positive_mask.to(
                    device=device, dtype=torch.bool
                )
                token_shared_mask = token_shared_mask.to(
                    device=device, dtype=torch.bool
                )
                token_changed_mask = token_changed_mask.to(
                    device=device, dtype=torch.bool
                )
                if bool(
                    (
                        token_positive_mask
                        & (token_shared_mask | token_changed_mask)
                    ).any().item()
                ) or bool((token_shared_mask & token_changed_mask).any().item()):
                    raise ValueError("direct trace token roles must be disjoint")
                if bool(
                    (
                        (token_positive_mask | token_shared_mask | token_changed_mask)
                        & ~score_token_mask
                    ).any().item()
                ):
                    raise ValueError(
                        "direct trace token roles must be subsets of score_token_mask"
                    )
                if bool(token_positive_mask[:, 1].any().item()) or bool(
                    (token_shared_mask[:, 0] | token_changed_mask[:, 0]).any().item()
                ):
                    raise ValueError(
                        "direct trace roles must use positive slot 0 and TN slot 1"
                    )
                invalid_rows = ~direct_trace_valid
                if bool(
                    (
                        token_positive_mask[invalid_rows]
                        | token_shared_mask[invalid_rows]
                        | token_changed_mask[invalid_rows]
                    ).any().item()
                ):
                    raise ValueError("invalid direct trace rows must have empty roles")
                if bool(
                    (
                        direct_trace_valid
                        & (
                            ~token_positive_mask[:, 0].any(dim=-1)
                            | ~token_changed_mask[:, 1].any(dim=-1)
                        )
                    ).any().item()
                ):
                    raise ValueError(
                        "valid direct trace rows require positive and changed tokens"
                    )
            token_losses: List[torch.Tensor] = []
            flat_loss_sums: List[torch.Tensor] = []
            flat_element_count = 0
            group_balanced = self.token_objective == "edit_bce_group_balanced"
            gdino_reduction = (
                self.token_objective == "gdino_allquery_allneg_focal"
            )
            allquery_surface = self.token_objective in {
                "gdino_allquery_allneg_focal",
                "allquery_allneg_focal",
            }
            edit_aware = self.token_objective in {
                "edit_bce",
                "edit_focal",
                "edit_bce_group_balanced",
            }
            focal_family = self.token_objective in {
                "gdino_allquery_allneg_focal",
                "allquery_allneg_focal",
                "targetlocal_allneg_focal",
                "edit_focal",
            }
            target_queries = (
                admitted
                & torch.isfinite(candidate_ious)
                & (candidate_ious >= self.positive_iou_threshold)
            )
            token_target_query_count = int(target_queries.sum().item())

            for batch_idx in range(batch_size):
                sample_terms: List[torch.Tensor] = []
                sample_weights: List[float] = []
                sample_supervised_count = 0
                target_query = target_queries[batch_idx]
                all_query = admitted[batch_idx] & torch.isfinite(
                    candidate_ious[batch_idx]
                )
                pair_is_valid = bool(
                    expression_valid_mask[batch_idx].all().item()
                ) and (
                    self.allow_legacy_token_diff_fallback
                    or (
                        token_provenance_valid is not None
                        and bool(token_provenance_valid[batch_idx].item())
                    )
                )
                if direct_roles_supplied:
                    pair_is_valid = pair_is_valid and bool(
                        direct_trace_valid[batch_idx].item()
                    )
                else:
                    pair_is_valid = pair_is_valid and (
                        predicate_valid is None
                        or bool(predicate_valid[batch_idx].item())
                    )

                positive_supervision_query = target_query
                role_carrier_tn_query = None
                if (
                    role_carrier_eligible is not None
                    and pair_is_valid
                    and bool(role_carrier_eligible[batch_idx].item())
                ):
                    positive_carrier_mask = role_carrier_positive_admitted[
                        batch_idx
                    ]
                    tn_carrier_mask = role_carrier_tn_admitted[batch_idx]
                    # The v41 supervision surface is pair-complete: if either
                    # expression cannot identify its deployed carrier, neither
                    # side expands beyond the target-IoU queries.
                    if bool(positive_carrier_mask.any().item()) and bool(
                        tn_carrier_mask.any().item()
                    ):
                        pair_carrier_logits = token_role_carrier_logits[
                            batch_idx
                        ].detach().float()
                        positive_scores = pair_carrier_logits[:, 0].masked_fill(
                            ~positive_carrier_mask, torch.finfo(torch.float32).min
                        )
                        tn_scores = pair_carrier_logits[:, 1].masked_fill(
                            ~tn_carrier_mask, torch.finfo(torch.float32).min
                        )
                        positive_index = int(positive_scores.argmax().item())
                        tn_index = int(tn_scores.argmax().item())
                        positive_carrier_query = torch.zeros_like(target_query)
                        role_carrier_tn_query = torch.zeros_like(target_query)
                        positive_carrier_query[positive_index] = True
                        role_carrier_tn_query[tn_index] = True
                        positive_overlap = bool(target_query[positive_index].item())
                        tn_overlap = bool(target_query[tn_index].item())
                        token_role_carrier_pair_selected_count += 1
                        token_role_carrier_positive_added_count += int(
                            not positive_overlap
                        )
                        token_role_carrier_tn_added_count += int(not tn_overlap)
                        token_role_carrier_positive_target_overlap_count += int(
                            positive_overlap
                        )
                        token_role_carrier_tn_target_overlap_count += int(tn_overlap)
                        token_edit_carrier_selected_count += 1
                        token_edit_carrier_added_count += int(not tn_overlap)
                        token_edit_carrier_target_overlap_count += int(tn_overlap)
                        positive_supervision_query = (
                            target_query | positive_carrier_query
                        )

                def add_group(
                    group_logits: torch.Tensor,
                    group_targets: torch.Tensor,
                    group_mask: torch.Tensor,
                    group_weight: float,
                    *,
                    focal: bool,
                ) -> int:
                    nonlocal flat_element_count, sample_supervised_count
                    count = int(group_mask.sum().item())
                    if count == 0:
                        return count
                    sample_supervised_count += count
                    if not group_balanced:
                        # A zero coefficient removes that token's contribution,
                        # but it does not renormalize the remaining roles upward.
                        flat_element_count += count
                    if float(group_weight) <= 0.0:
                        return count
                    selected_logits = group_logits[group_mask].float()
                    selected_targets = group_targets[group_mask].float()
                    group_loss = F.binary_cross_entropy_with_logits(
                        selected_logits, selected_targets, reduction="none"
                    )
                    if focal and self.token_focal_gamma > 0.0:
                        probability = selected_logits.sigmoid()
                        target_probability = (
                            probability * selected_targets
                            + (1.0 - probability) * (1.0 - selected_targets)
                        )
                        group_loss = group_loss * (1.0 - target_probability).pow(
                            self.token_focal_gamma
                        )
                    if focal and self.token_focal_alpha >= 0.0:
                        alpha_t = (
                            self.token_focal_alpha * selected_targets
                            + (1.0 - self.token_focal_alpha)
                            * (1.0 - selected_targets)
                        )
                        group_loss = group_loss * alpha_t
                    if group_balanced:
                        sample_terms.append(
                            group_loss.mean() * float(group_weight)
                        )
                        sample_weights.append(float(group_weight))
                    else:
                        flat_loss_sums.append(
                            group_loss.sum() * float(group_weight)
                        )
                    return count

                if allquery_surface:
                    positive_mask_tokens = (
                        all_query[:, None]
                        & score_token_mask[batch_idx, 0][None, :]
                        & expression_valid_mask[batch_idx, 0]
                    )
                    positive_targets = target_query[:, None].expand_as(
                        positive_mask_tokens
                    )
                else:
                    positive_tokens = score_token_mask[batch_idx, 0]
                    if direct_roles_supplied and bool(
                        direct_trace_valid[batch_idx].item()
                    ):
                        positive_tokens = token_positive_mask[batch_idx, 0]
                    positive_mask_tokens = (
                        positive_supervision_query[:, None]
                        & positive_tokens[None, :]
                        & expression_valid_mask[batch_idx, 0]
                    )
                    positive_targets = torch.ones_like(
                        positive_mask_tokens, dtype=token_logits.dtype
                    )
                token_positive_count += add_group(
                    token_logits[batch_idx, :, 0],
                    positive_targets,
                    positive_mask_tokens,
                    self.token_positive_weight,
                    focal=focal_family,
                )

                if edit_aware and pair_is_valid:
                    shared_query = target_query
                    if local_valid is not None:
                        shared_query = shared_query & local_valid[batch_idx]
                    edit_query = shared_query
                    if role_carrier_tn_query is not None:
                        shared_query = shared_query | role_carrier_tn_query
                        edit_query = edit_query | role_carrier_tn_query
                    elif (
                        carrier_edit_admitted is not None
                        and bool(carrier_edit_eligible[batch_idx].item())
                    ):
                        carrier_mask = carrier_edit_admitted[batch_idx]
                        if bool(carrier_mask.any().item()):
                            carrier_scores = token_edit_carrier_logits[
                                batch_idx
                            ].detach().float().masked_fill(
                                ~carrier_mask, torch.finfo(torch.float32).min
                            )
                            carrier_index = int(carrier_scores.argmax().item())
                            carrier_query = torch.zeros_like(edit_query)
                            carrier_query[carrier_index] = True
                            carrier_overlaps_target = bool(
                                edit_query[carrier_index].item()
                            )
                            token_edit_carrier_selected_count += 1
                            token_edit_carrier_added_count += int(
                                not carrier_overlaps_target
                            )
                            token_edit_carrier_target_overlap_count += int(
                                carrier_overlaps_target
                            )
                            edit_query = edit_query | carrier_query
                    token_edit_query_count += int(edit_query.sum().item())
                    if direct_roles_supplied:
                        changed_tokens = token_changed_mask[batch_idx, 1]
                        shared_tokens = token_shared_mask[batch_idx, 1]
                    else:
                        changed_tokens = (
                            score_token_mask[batch_idx, 1]
                            & predicate_token_mask[batch_idx, 1]
                        )
                        shared_tokens = (
                            score_token_mask[batch_idx, 1] & ~changed_tokens
                        )
                    shared_mask = shared_query[:, None] & shared_tokens[None, :]
                    edit_mask = edit_query[:, None] & changed_tokens[None, :]
                    token_shared_count += add_group(
                        token_logits[batch_idx, :, 1],
                        torch.ones_like(shared_mask, dtype=token_logits.dtype),
                        shared_mask,
                        self.token_shared_weight,
                        focal=focal_family,
                    )
                    token_edit_count += add_group(
                        token_logits[batch_idx, :, 1],
                        torch.zeros_like(edit_mask, dtype=token_logits.dtype),
                        edit_mask,
                        self.token_edit_weight,
                        focal=focal_family,
                    )
                elif self.token_objective in {
                    "gdino_allquery_allneg_focal",
                    "targetlocal_allneg_focal",
                    "targetlocal_allneg_bce",
                    "allquery_allneg_focal",
                } and bool(expression_valid_mask[batch_idx, 1].item()):
                    tn_query = (
                        all_query
                        if allquery_surface
                        else target_query
                    )
                    if local_valid is not None:
                        tn_query = tn_query & local_valid[batch_idx]
                    tn_mask = (
                        tn_query[:, None]
                        & score_token_mask[batch_idx, 1][None, :]
                    )
                    token_all_negative_count += add_group(
                        token_logits[batch_idx, :, 1],
                        torch.zeros_like(tn_mask, dtype=token_logits.dtype),
                        tn_mask,
                        self.token_edit_weight,
                        focal=focal_family,
                    )

                if sample_terms:
                    token_losses.append(
                        torch.stack(sample_terms).sum() / sum(sample_weights)
                    )
                if sample_supervised_count > 0:
                    token_sample_count += 1
            if group_balanced:
                loss_token = _mean_or_zero(token_losses, token_zero)
            elif flat_loss_sums:
                denominator = (
                    max(token_target_query_count, 1)
                    if gdino_reduction
                    else max(flat_element_count, 1)
                )
                loss_token = torch.stack(flat_loss_sums).sum() / float(
                    denominator
                )

        loss_raw_veto_gate = raw_veto_zero
        loss_raw_veto_carrier_pair = raw_veto_zero
        tn_raw_loss = raw_veto_zero
        raw_veto_positive_losses: List[torch.Tensor] = []
        raw_veto_tn_losses: List[torch.Tensor] = []
        raw_veto_positive_all_hinges: List[torch.Tensor] = []
        raw_veto_positive_carrier_hinges: List[torch.Tensor] = []
        raw_veto_tn_all_hinges: List[torch.Tensor] = []
        raw_veto_tn_carrier_hinges: List[torch.Tensor] = []
        raw_veto_positive_sources: List[torch.Tensor] = []
        raw_veto_tn_sources: List[torch.Tensor] = []
        raw_veto_positive_carrier_sources: List[torch.Tensor] = []
        raw_veto_tn_carrier_sources: List[torch.Tensor] = []
        raw_veto_tn_carrier_changed_gates: List[torch.Tensor] = []
        raw_veto_tn_carrier_tail_scores: List[torch.Tensor] = []
        raw_veto_tn_carrier_tail_batch_indices: List[int] = []
        raw_veto_carrier_pair_gaps: List[torch.Tensor] = []
        raw_veto_carrier_pair_hinges: List[torch.Tensor] = []
        raw_veto_carrier_pair_batch_indices: List[int] = []
        raw_veto_positive_query_count = 0
        raw_veto_tn_query_count = 0
        raw_veto_positive_sample_count = 0
        raw_veto_tn_sample_count = 0
        raw_veto_positive_violation_count = 0
        raw_veto_tn_violation_count = 0
        raw_veto_positive_carrier_sample_count = 0
        raw_veto_positive_carrier_violation_count = 0
        raw_veto_tn_carrier_sample_count = 0
        raw_veto_tn_carrier_violation_count = 0
        raw_veto_tn_carrier_full_open_count = 0
        raw_veto_carrier_pair_violation_count = 0
        raw_veto_tn_tail_threshold = 0.0
        raw_veto_tn_tail_weight_mean = 0.0
        raw_veto_tn_tail_effective_sample_count = 0.0
        raw_veto_tn_tail_carrier_hinge_mean = 0.0
        raw_veto_tn_tail_weights: Optional[torch.Tensor] = None
        raw_veto_tn_tail_weight_sum: Optional[torch.Tensor] = None
        raw_veto_tail_pair_gap_mean = 0.0
        raw_veto_tail_pair_hinge_mean = 0.0
        raw_veto_tail_pair_violation_rate = 0.0
        raw_veto_tail_pair_effective_sample_count = 0.0
        carrier_scope = self.raw_veto_query_scope in {
            "tn_all_admitted_positive_carrier_v2",
            "tn_all_admitted_carrier_balanced_positive_carrier_v3",
            "tn_all_admitted_carrier_balanced_positive_carrier_paired_v4",
            "tn_all_admitted_dual_carrier_balanced_paired_v5",
            "tn_all_admitted_tail_weighted_carrier_positive_carrier_paired_v6",
            "tn_all_admitted_tail_weighted_carrier_tail_paired_v7",
        }
        carrier_balanced_scope = self.raw_veto_query_scope in {
            "tn_all_admitted_carrier_balanced_positive_carrier_v3",
            "tn_all_admitted_carrier_balanced_positive_carrier_paired_v4",
            "tn_all_admitted_dual_carrier_balanced_paired_v5",
            "tn_all_admitted_tail_weighted_carrier_positive_carrier_paired_v6",
            "tn_all_admitted_tail_weighted_carrier_tail_paired_v7",
        }
        carrier_pair_scope = self.raw_veto_query_scope in {
            "tn_all_admitted_carrier_balanced_positive_carrier_paired_v4",
            "tn_all_admitted_dual_carrier_balanced_paired_v5",
            "tn_all_admitted_tail_weighted_carrier_positive_carrier_paired_v6",
            "tn_all_admitted_tail_weighted_carrier_tail_paired_v7",
        }
        dual_carrier_scope = self.raw_veto_query_scope == (
            "tn_all_admitted_dual_carrier_balanced_paired_v5"
        )
        tail_weighted_carrier_scope = self.raw_veto_query_scope in {
            "tn_all_admitted_tail_weighted_carrier_positive_carrier_paired_v6",
            "tn_all_admitted_tail_weighted_carrier_tail_paired_v7",
        }
        tail_weighted_pair_scope = self.raw_veto_query_scope == (
            "tn_all_admitted_tail_weighted_carrier_tail_paired_v7"
        )
        if raw_veto_supervision_enabled:
            residual = token_residual_logits.float()
            groups = score_word_group_ids.to(device=device, dtype=torch.long)
            score_words = score_token_mask.to(device=device, dtype=torch.bool)
            changed = token_changed_mask.to(device=device, dtype=torch.bool)
            if bool((score_words & groups.lt(0)).any().item()):
                raise ValueError("raw veto score tokens require lexical word groups")

            def word_residuals(
                sample_residual: torch.Tensor,
                sample_mask: torch.Tensor,
                sample_groups: torch.Tensor,
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                member = sample_mask & sample_groups.ge(0)
                if not bool(member.any().item()):
                    empty = sample_residual.new_zeros(
                        (int(sample_residual.shape[0]), 1)
                    )
                    return (
                        empty,
                        torch.zeros(1, device=device, dtype=torch.bool),
                        sample_residual.new_zeros((int(sample_mask.numel()), 1)),
                    )
                group_count = int(sample_groups[member].max().item()) + 1
                membership = F.one_hot(
                    sample_groups.clamp_min(0), num_classes=group_count
                ).to(dtype=sample_residual.dtype)
                membership = membership * member[:, None].to(
                    dtype=membership.dtype
                )
                count = membership.sum(dim=0)
                values = torch.einsum(
                    "nt,tg->ng", sample_residual, membership
                ) / count.clamp_min(1.0)[None]
                return values, count.gt(0), membership

            for batch_idx in range(batch_size):
                positive_words, positive_word_valid, _positive_membership = (
                    word_residuals(
                        residual[batch_idx, :, 0],
                        score_words[batch_idx, 0],
                        groups[batch_idx, 0],
                    )
                )
                positive_query = positive_mask[batch_idx]
                positive_carrier_index = None
                positive_carrier_source_for_pair = None
                if carrier_scope:
                    if carrier_balanced_scope:
                        carrier_index = confidence_veto_carrier_indices[
                            batch_idx, 0
                        ]
                        if int(carrier_index.item()) >= 0:
                            if not bool(admitted[batch_idx, carrier_index].item()):
                                raise RuntimeError(
                                    "positive confidence carrier is not admitted"
                                )
                            positive_carrier_index = carrier_index
                    else:
                        carrier_valid = admitted[batch_idx] & torch.isfinite(
                            positive_reference_base_logits[batch_idx]
                        )
                        if bool(carrier_valid.any().item()):
                            carrier_index = positive_reference_base_logits[
                                batch_idx
                            ].float().masked_fill(
                                ~carrier_valid, -torch.inf
                            ).argmax()
                            positive_carrier_index = carrier_index
                    if positive_carrier_index is not None:
                        carrier_query = torch.zeros_like(admitted[batch_idx])
                        carrier_query[positive_carrier_index] = True
                        positive_query = positive_query | carrier_query
                if (
                    bool(expression_valid_mask[batch_idx, 0].item())
                    and bool(positive_word_valid.any().item())
                    and bool(positive_query.any().item())
                ):
                    positive_source_by_query = positive_words[
                        :, positive_word_valid
                    ].max(dim=-1).values
                    positive_source = positive_source_by_query[positive_query]
                    positive_hinge = F.relu(
                        positive_source + self.raw_veto_positive_margin
                    )
                    positive_sample_loss = positive_hinge.mean()
                    raw_veto_positive_all_hinges.append(positive_hinge.mean())
                    if carrier_balanced_scope:
                        if positive_carrier_index is None:
                            raise RuntimeError(
                                "carrier-balanced positive row has no confidence carrier"
                            )
                        positive_carrier_source = positive_source_by_query[
                            positive_carrier_index
                        ]
                        positive_carrier_hinge = F.relu(
                            positive_carrier_source
                            + self.raw_veto_positive_margin
                        )
                        raw_veto_positive_carrier_sources.append(
                            positive_carrier_source
                        )
                        if carrier_pair_scope:
                            positive_carrier_source_for_pair = positive_carrier_source
                        raw_veto_positive_carrier_hinges.append(
                            positive_carrier_hinge
                        )
                        if dual_carrier_scope:
                            balance = self.raw_veto_positive_carrier_balance
                            positive_sample_loss = (
                                (1.0 - balance) * positive_sample_loss
                                + balance * positive_carrier_hinge
                            )
                        raw_veto_positive_carrier_sample_count += 1
                        raw_veto_positive_carrier_violation_count += int(
                            positive_carrier_hinge.detach().gt(0).item()
                        )
                    raw_veto_positive_losses.append(positive_sample_loss)
                    raw_veto_positive_sources.append(positive_source.mean())
                    raw_veto_positive_sample_count += 1
                    raw_veto_positive_query_count += int(positive_source.numel())
                    raw_veto_positive_violation_count += int(
                        positive_hinge.detach().gt(0).sum().item()
                    )

                tn_words, tn_word_valid, tn_membership = word_residuals(
                    residual[batch_idx, :, 1],
                    score_words[batch_idx, 1],
                    groups[batch_idx, 1],
                )
                changed_word_valid = tn_word_valid & torch.einsum(
                    "t,tg->g",
                    changed[batch_idx, 1].to(dtype=tn_membership.dtype),
                    tn_membership,
                ).gt(0)
                tn_row_valid = bool(expression_valid_mask[batch_idx].all().item())
                tn_row_valid = tn_row_valid and bool(
                    token_provenance_valid[batch_idx].item()
                )
                tn_row_valid = tn_row_valid and bool(
                    direct_trace_valid[batch_idx].item()
                )
                tn_row_valid = tn_row_valid and bool(
                    tn_train_eligible[batch_idx].item()
                )
                if carrier_scope:
                    tn_query = (
                        local_valid[batch_idx]
                        if local_valid is not None
                        else admitted[batch_idx]
                    )
                else:
                    tn_query = positive_query
                    if local_valid is not None:
                        tn_query = tn_query & local_valid[batch_idx]
                if (
                    tn_row_valid
                    and bool(changed_word_valid.any().item())
                    and bool(tn_query.any().item())
                ):
                    tn_source_by_query = tn_words[:, changed_word_valid].max(
                        dim=-1
                    ).values
                    tn_source = tn_source_by_query[tn_query]
                    tn_hinge = F.relu(self.raw_veto_tn_margin - tn_source)
                    tn_sample_loss = tn_hinge.mean()
                    raw_veto_tn_all_hinges.append(tn_hinge.mean())
                    if carrier_balanced_scope:
                        tn_carrier_index = confidence_veto_carrier_indices[
                            batch_idx, 1
                        ]
                        if (
                            int(tn_carrier_index.item()) < 0
                            or not bool(tn_query[tn_carrier_index].item())
                        ):
                            raise RuntimeError(
                                "carrier-balanced TN row has no admitted confidence "
                                "carrier"
                            )
                        tn_carrier_source = tn_source_by_query[tn_carrier_index]
                        tn_carrier_hinge = F.relu(
                            self.raw_veto_tn_margin - tn_carrier_source
                        )
                        tn_carrier_changed_gate = (
                            (
                                tn_carrier_source.detach()
                                - self.raw_veto_gate_offset
                            )
                            / self.raw_veto_gate_scale
                        ).clamp(min=0.0, max=1.0)
                        balance = self.raw_veto_tn_carrier_balance
                        if not tail_weighted_carrier_scope:
                            tn_sample_loss = (
                                (1.0 - balance) * tn_sample_loss
                                + balance * tn_carrier_hinge
                            )
                        else:
                            tail_score_valid = tn_query & torch.isfinite(
                                local_tn_confidence_logits[batch_idx]
                            )
                            if not bool(tail_score_valid.any().item()):
                                raise RuntimeError(
                                    "tail-weighted TN carrier has no finite global score"
                                )
                            raw_veto_tn_carrier_tail_scores.append(
                                local_tn_confidence_logits[batch_idx]
                                .float()
                                .masked_fill(~tail_score_valid, -torch.inf)
                                .max()
                                .detach()
                            )
                            raw_veto_tn_carrier_tail_batch_indices.append(batch_idx)
                        raw_veto_tn_carrier_hinges.append(tn_carrier_hinge)
                        raw_veto_tn_carrier_changed_gates.append(
                            tn_carrier_changed_gate
                        )
                        raw_veto_tn_carrier_sources.append(tn_carrier_source)
                        if carrier_pair_scope:
                            if positive_carrier_source_for_pair is None:
                                raise RuntimeError(
                                    "paired v4 TN carrier has no same-sample positive "
                                    "carrier source"
                                )
                            if (
                                int(positive_carrier_source_for_pair.numel()) != 1
                                or int(tn_carrier_source.numel()) != 1
                            ):
                                raise RuntimeError(
                                    "paired v4 carrier sources must be one-to-one scalars"
                                )
                            positive_pair_anchor = positive_carrier_source_for_pair
                            if (
                                self.raw_veto_carrier_pair_gradient_contract
                                == "tn_only_positive_detached_v2"
                            ):
                                positive_pair_anchor = positive_pair_anchor.detach()
                            pair_gap = tn_carrier_source - positive_pair_anchor
                            pair_hinge = F.relu(
                                self.raw_veto_carrier_pair_margin - pair_gap
                            )
                            if not bool(
                                torch.isfinite(pair_gap).item()
                                and torch.isfinite(pair_hinge).item()
                            ):
                                raise ValueError(
                                    "paired v4 carrier gap and hinge must be finite"
                                )
                            raw_veto_carrier_pair_gaps.append(pair_gap)
                            raw_veto_carrier_pair_hinges.append(pair_hinge)
                            raw_veto_carrier_pair_batch_indices.append(batch_idx)
                            raw_veto_carrier_pair_violation_count += int(
                                pair_hinge.detach().gt(0).item()
                            )
                        raw_veto_tn_carrier_sample_count += 1
                        raw_veto_tn_carrier_violation_count += int(
                            tn_carrier_hinge.detach().gt(0).item()
                        )
                        raw_veto_tn_carrier_full_open_count += int(
                            tn_carrier_changed_gate.eq(1.0).item()
                        )
                    raw_veto_tn_losses.append(tn_sample_loss)
                    raw_veto_tn_sources.append(tn_source.mean())
                    raw_veto_tn_sample_count += 1
                    raw_veto_tn_query_count += int(tn_source.numel())
                    raw_veto_tn_violation_count += int(
                        tn_hinge.detach().gt(0).sum().item()
                    )

            positive_raw_loss = _mean_or_zero(
                raw_veto_positive_losses, raw_veto_zero
            )
            tn_raw_loss = _mean_or_zero(raw_veto_tn_losses, raw_veto_zero)
            if tail_weighted_carrier_scope and raw_veto_tn_carrier_hinges:
                tail_count = len(raw_veto_tn_carrier_hinges)
                if not (
                    len(raw_veto_tn_carrier_tail_scores) == tail_count
                    and len(raw_veto_tn_carrier_tail_batch_indices) == tail_count
                ):
                    raise RuntimeError(
                        "tail-weighted carrier scores, sample indices, and hinges "
                        "are not one-to-one"
                    )
                if len(raw_veto_tn_carrier_tail_batch_indices) != len(
                    set(raw_veto_tn_carrier_tail_batch_indices)
                ):
                    raise RuntimeError(
                        "tail-weighted carrier supervision produced duplicate samples"
                    )
                tail_scores = torch.stack(raw_veto_tn_carrier_tail_scores).float()
                tail_hinges = torch.stack(raw_veto_tn_carrier_hinges).float()
                history_ready = (
                    self.tail_queue_enabled
                    and int(self.tail_negative_count.item())
                    >= self.raw_veto_tail_min_count
                )
                if history_ready:
                    tail_threshold = torch.quantile(
                        self._tail_queue_values("negative").float(),
                        self.raw_veto_tail_quantile,
                    ).detach()
                else:
                    tail_threshold = torch.quantile(
                        tail_scores.detach(), self.raw_veto_tail_quantile
                    )
                tail_weights = torch.sigmoid(
                    (tail_scores.detach() - tail_threshold)
                    / self.raw_veto_tail_temperature
                )
                weight_sum = tail_weights.sum().clamp_min(1e-6)
                raw_veto_tn_tail_weights = tail_weights
                raw_veto_tn_tail_weight_sum = weight_sum
                tail_carrier_loss = (
                    tail_weights * tail_hinges
                ).sum() / weight_sum
                all_query_tn_loss = _mean_or_zero(
                    raw_veto_tn_all_hinges, raw_veto_zero
                )
                balance = self.raw_veto_tn_carrier_balance
                tn_raw_loss = (
                    (1.0 - balance) * all_query_tn_loss
                    + balance * tail_carrier_loss
                )
                raw_veto_tn_tail_threshold = float(tail_threshold.item())
                raw_veto_tn_tail_weight_mean = float(
                    tail_weights.mean().item()
                )
                raw_veto_tn_tail_effective_sample_count = float(
                    (tail_weights.sum().square() / tail_weights.square().sum().clamp_min(1e-12)).item()
                )
                raw_veto_tn_tail_carrier_hinge_mean = float(
                    tail_carrier_loss.detach().item()
                )
            if raw_veto_positive_losses and raw_veto_tn_losses:
                loss_raw_veto_gate = 0.5 * positive_raw_loss + 0.5 * tn_raw_loss
            elif raw_veto_positive_losses:
                loss_raw_veto_gate = positive_raw_loss
            elif raw_veto_tn_losses:
                loss_raw_veto_gate = tn_raw_loss
            if carrier_pair_scope:
                pair_count = len(raw_veto_carrier_pair_hinges)
                if not (
                    len(raw_veto_carrier_pair_batch_indices) == pair_count
                    and len(raw_veto_carrier_pair_gaps) == pair_count
                ):
                    raise RuntimeError(
                        "paired carrier sample indices, gaps, and hinges are not "
                        "one-to-one"
                    )
                if len(raw_veto_carrier_pair_batch_indices) != len(
                    set(raw_veto_carrier_pair_batch_indices)
                ):
                    raise RuntimeError(
                        "paired carrier supervision produced duplicate sample pairs"
                    )
                if tail_weighted_pair_scope and pair_count:
                    if not (
                        raw_veto_tn_tail_weights is not None
                        and raw_veto_tn_tail_weight_sum is not None
                        and len(raw_veto_tn_carrier_tail_batch_indices) == pair_count
                        and raw_veto_tn_carrier_tail_batch_indices
                        == raw_veto_carrier_pair_batch_indices
                        and int(raw_veto_tn_tail_weights.numel()) == pair_count
                    ):
                        raise RuntimeError(
                            "tail-weighted carrier scores and paired carriers are "
                            "not one-to-one in sample order"
                        )
                    pair_gaps = torch.stack(raw_veto_carrier_pair_gaps).float()
                    pair_hinges = torch.stack(raw_veto_carrier_pair_hinges).float()
                    pair_violation = pair_hinges.detach().gt(0).to(
                        dtype=raw_veto_tn_tail_weights.dtype
                    )
                    loss_raw_veto_carrier_pair = (
                        raw_veto_tn_tail_weights * pair_hinges
                    ).sum() / raw_veto_tn_tail_weight_sum
                    raw_veto_tail_pair_gap_mean = float(
                        (
                            (raw_veto_tn_tail_weights * pair_gaps).sum()
                            / raw_veto_tn_tail_weight_sum
                        )
                        .detach()
                        .item()
                    )
                    raw_veto_tail_pair_hinge_mean = float(
                        loss_raw_veto_carrier_pair.detach().item()
                    )
                    raw_veto_tail_pair_violation_rate = float(
                        (
                            (raw_veto_tn_tail_weights * pair_violation).sum()
                            / raw_veto_tn_tail_weight_sum
                        ).item()
                    )
                    raw_veto_tail_pair_effective_sample_count = float(
                        (
                            raw_veto_tn_tail_weights.sum().square()
                            / raw_veto_tn_tail_weights.square().sum().clamp_min(1e-12)
                        ).item()
                    )
                else:
                    loss_raw_veto_carrier_pair = _mean_or_zero(
                        raw_veto_carrier_pair_hinges, raw_veto_zero
                    )
        global_admitted = None
        if global_tn_confidence_logits is not None:
            global_admitted = _coerce_candidate_mask(
                global_tn_candidate_mask if global_tn_candidate_mask is not None else candidate_mask,
                shape=shape,
                device=device,
                name="global_tn_candidate_mask",
            ) & torch.isfinite(global_tn_confidence_logits)
        sample_global_admitted = None
        if sample_tn_confidence_logits is not None and global_tn_logits is not None:
            sample_global_admitted = _coerce_candidate_mask(
                global_tn_candidate_mask
                if global_tn_candidate_mask is not None
                else candidate_mask,
                shape=shape,
                device=device,
                name="global_tn_candidate_mask",
            ) & torch.isfinite(sample_tn_confidence_logits)[:, None]

        gathered_tail_payload = None
        if self.tail_queue_enabled:
            if sample_positive_confidence_logits is not None:
                best_positive = sample_positive_confidence_logits.float()
                best_positive_valid = (
                    admitted.any(dim=1) & torch.isfinite(best_positive)
                )
                if (
                    sample_tn_confidence_logits is not None
                    and local_confidence_valid is not None
                ):
                    negative_score_valid = admitted & local_confidence_valid
                    if confidence_negative_eligible is not None:
                        negative_score_valid &= confidence_negative_eligible[:, None]
                    best_negative = sample_tn_confidence_logits.float()
                    best_negative_valid = (
                        best_positive_valid
                        & negative_score_valid.any(dim=1)
                        & torch.isfinite(best_negative)
                    )
                else:
                    best_negative = torch.zeros_like(best_positive)
                    best_negative_valid = torch.zeros_like(best_positive_valid)
            elif self.tail_queue_global_scores:
                positive_score_valid = admitted & torch.isfinite(confidence_logits)
                positive_sentinel = torch.finfo(torch.float32).min
                best_positive = confidence_logits.float().masked_fill(
                    ~positive_score_valid, positive_sentinel
                ).max(dim=1).values
                # FPR@TPR scores every positive expression, including samples
                # whose fixed candidates miss IoU>=threshold. Match that global
                # metric exactly instead of silently selecting easy positives.
                best_positive_valid = positive_score_valid.any(dim=1)
                if (
                    local_tn_confidence_logits is not None
                    and local_confidence_valid is not None
                ):
                    negative_score_valid = admitted & local_confidence_valid
                    if confidence_negative_eligible is not None:
                        negative_score_valid &= confidence_negative_eligible[:, None]
                    best_negative = local_tn_confidence_logits.float().masked_fill(
                        ~negative_score_valid, positive_sentinel
                    ).max(dim=1).values
                    best_negative_valid = (
                        best_positive_valid & negative_score_valid.any(dim=1)
                    )
                else:
                    best_negative = torch.zeros_like(best_positive)
                    best_negative_valid = torch.zeros_like(best_positive_valid)
            else:
                tail_candidate_valid = (
                    admitted
                    & torch.isfinite(candidate_ious)
                    & torch.isfinite(confidence_logits)
                )
                best_iou = candidate_ious.masked_fill(~tail_candidate_valid, -1.0)
                best_idx = best_iou.argmax(dim=1, keepdim=True)
                best_iou_value = best_iou.gather(1, best_idx).squeeze(1)
                best_positive = confidence_logits.gather(1, best_idx).squeeze(1).float()
                best_positive_valid = (
                    tail_candidate_valid.gather(1, best_idx).squeeze(1)
                    & (best_iou_value >= self.positive_iou_threshold)
                )
                if (
                    local_tn_confidence_logits is not None
                    and local_confidence_valid is not None
                ):
                    best_negative = local_tn_confidence_logits.gather(
                        1, best_idx
                    ).squeeze(1).float()
                    best_negative_valid = (
                        best_positive_valid
                        & local_confidence_valid.gather(1, best_idx).squeeze(1)
                        & torch.isfinite(best_negative)
                    )
                else:
                    best_negative = torch.zeros_like(best_positive)
                    best_negative_valid = torch.zeros_like(best_positive_valid)
            positive_gate = (
                torch.zeros_like(best_positive)
                if positive_confidence_gate_logits is None
                else positive_confidence_gate_logits.float()
            )
            tail_payload = torch.stack(
                (
                    best_positive,
                    best_positive_valid.float(),
                    best_negative,
                    best_negative_valid.float(),
                    positive_gate,
                ),
                dim=-1,
            )
            if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                gathered_tail_payload = torch.cat(
                    tuple(dist_nn_functional.all_gather(tail_payload)), dim=0
                )
            else:
                gathered_tail_payload = tail_payload
            self._pending_tail_payload = gathered_tail_payload.detach()

        for batch_idx in range(batch_size):
            pos = positive_mask[batch_idx]
            neg = negative_mask[batch_idx]
            confidence_pos = confidence_positive_mask[batch_idx]
            positive_anchor = None
            if sample_positive_confidence_logits is not None:
                sample_positive_valid = bool(admitted[batch_idx].any().item())
                if sample_positive_valid:
                    sample_positive = sample_positive_confidence_logits[
                        batch_idx
                    ].float()
                    batch_positive_scores.append(sample_positive)
                    fixed_positive_scores[batch_idx] = sample_positive
                    fixed_positive_valid[batch_idx] = True
                    if (
                        self.weight_dict[
                            "loss_fixed_text_deployed_global_absolute"
                        ]
                        > 0.0
                    ):
                        deployed_global_absolute_positive_losses.append(
                            binary_focal_with_logits(
                                sample_positive,
                                1.0,
                                gamma=self.deployed_global_absolute_gamma,
                            )
                        )
                if (
                    sample_positive_valid
                    and sample_tn_confidence_logits is not None
                    and local_confidence_valid is not None
                ):
                    sample_negative_valid = admitted[batch_idx] & (
                        local_confidence_valid[batch_idx]
                    )
                    if confidence_negative_eligible is not None:
                        sample_negative_valid &= confidence_negative_eligible[
                            batch_idx
                        ]
                    if bool(sample_negative_valid.any().item()):
                        sample_negative = sample_tn_confidence_logits[
                            batch_idx
                        ].float()
                        batch_negative_scores.append(sample_negative)
                        fixed_negative_scores[batch_idx] = sample_negative
                        fixed_negative_valid[batch_idx] = True
                        if (
                            self.weight_dict[
                                "loss_fixed_text_deployed_global_absolute"
                            ]
                            > 0.0
                        ):
                            deployed_global_absolute_tn_losses.append(
                                binary_focal_with_logits(
                                    sample_negative,
                                    0.0,
                                    gamma=self.deployed_global_absolute_gamma,
                                )
                            )
            if bool(pos.any().item()):
                rank_positive_values = candidate_logits[batch_idx, pos]
                positive_anchor = F.softplus(
                    self.positive_anchor_logit - rank_positive_values.float()
                ).mean()
                positive_anchor_losses.append(positive_anchor)
                local_anchor_positive_query_count += int(pos.sum().item())
            if bool(confidence_pos.any().item()):
                confidence_positive_values = confidence_logits[
                    batch_idx, confidence_pos
                ]
                absolute_positive_losses.append(
                    binary_focal_with_logits(
                        confidence_positive_values,
                        1.0,
                        gamma=self.local_absolute_gamma,
                    ).mean()
                )
                local_absolute_positive_sample_count += 1
                if sample_positive_confidence_logits is None:
                    batch_positive_scores.append(
                        _normalized_logsumexp(
                            confidence_positive_values, self.listwise_temperature
                        )
                    )
                    fixed_positive_scores[batch_idx] = batch_positive_scores[-1]
                    fixed_positive_valid[batch_idx] = True
            if bool(pos.any().item()) and bool(neg.any().item()):
                listwise_losses.append(
                    multi_positive_candidate_listwise_loss(
                        candidate_logits[batch_idx],
                        pos,
                        neg,
                        temperature=self.listwise_temperature,
                    )
                )
                valid_listwise_count += 1

            if local_tn_logits is not None and local_valid is not None:
                pair_mask = pos & local_valid[batch_idx]
                if bool(pair_mask.any().item()):
                    positive_values = candidate_logits[batch_idx, pair_mask].float()
                    tn_values = local_tn_logits[batch_idx, pair_mask].float()
                    local_rank_losses.append(
                        F.softplus(tn_values - positive_values + self.local_tn_rank_margin).mean()
                    )
                    local_pair_sample_count += 1
                    local_pair_query_count += int(pair_mask.sum().item())
                    tn_anchor = F.softplus(
                        tn_values - self.negative_anchor_logit
                    ).mean()
                    tn_anchor_losses.append(tn_anchor)
                    local_anchor_losses.append(0.5 * positive_anchor + 0.5 * tn_anchor)
                    local_anchor_paired_sample_count += 1
                    local_anchor_tn_query_count += int(pair_mask.sum().item())

            if (
                local_tn_confidence_logits is not None
                and local_confidence_valid is not None
            ):
                confidence_pair_mask = (
                    confidence_pos & local_confidence_valid[batch_idx]
                )
                if bool(confidence_pair_mask.any().item()):
                    tn_confidence_values = local_tn_confidence_logits[
                        batch_idx, confidence_pair_mask
                    ].float()
                    absolute_tn_losses.append(
                        binary_focal_with_logits(
                            tn_confidence_values,
                            0.0,
                            gamma=self.local_absolute_gamma,
                        ).mean()
                    )
                    local_absolute_tn_sample_count += 1
                    if sample_tn_confidence_logits is None:
                        batch_negative_scores.append(
                            _normalized_logsumexp(
                                tn_confidence_values, self.listwise_temperature
                            )
                        )
                        fixed_negative_scores[batch_idx] = batch_negative_scores[-1]
                        fixed_negative_valid[batch_idx] = True

            if (
                positive_predicate_logits is not None
                and local_tn_predicate_logits is not None
                and predicate_valid is not None
                and bool(predicate_valid[batch_idx].item())
            ):
                predicate_mask = (
                    pos
                    & torch.isfinite(positive_predicate_logits[batch_idx])
                    & torch.isfinite(local_tn_predicate_logits[batch_idx])
                )
                if local_valid is not None:
                    predicate_mask &= local_valid[batch_idx]
                if bool(predicate_mask.any().item()):
                    positive_predicate_values = positive_predicate_logits[
                        batch_idx, predicate_mask
                    ].float()
                    tn_predicate_values = local_tn_predicate_logits[
                        batch_idx, predicate_mask
                    ].float()
                    predicate_rank_losses.append(
                        F.softplus(
                            tn_predicate_values
                            - positive_predicate_values
                            + self.predicate_tn_rank_margin
                        ).mean()
                    )
                    predicate_absolute_positive_losses.append(
                        binary_focal_with_logits(
                            positive_predicate_values,
                            1.0,
                            gamma=self.predicate_absolute_gamma,
                        ).mean()
                    )
                    predicate_absolute_tn_losses.append(
                        binary_focal_with_logits(
                            tn_predicate_values,
                            0.0,
                            gamma=self.predicate_absolute_gamma,
                        ).mean()
                    )
                    predicate_absolute_sample_count += 1
                    predicate_score_gaps.append(
                        (positive_predicate_values - tn_predicate_values).mean().detach()
                    )
                    predicate_win_rates.append(
                        (positive_predicate_values > tn_predicate_values)
                        .float()
                        .mean()
                        .detach()
                    )
                    predicate_pair_sample_count += 1
                    predicate_pair_query_count += int(predicate_mask.sum().item())

            if positive_anchor is not None:
                if not (
                    local_tn_logits is not None
                    and local_valid is not None
                    and bool((pos & local_valid[batch_idx]).any().item())
                ):
                    # Clean samples still anchor the absolute positive score.
                    local_anchor_losses.append(positive_anchor)
                local_anchor_sample_count += 1

            if (
                sample_tn_confidence_logits is not None
                and sample_global_admitted is not None
                and confidence_negative_eligible is not None
                and bool(confidence_negative_eligible[batch_idx].item())
            ):
                global_mask = sample_global_admitted[batch_idx]
                if bool(global_mask.any().item()):
                    tn_value = sample_tn_confidence_logits[batch_idx].float()
                    global_negative_losses.append(F.softplus(tn_value))
                    tail_score = tn_value
                    global_tail_losses.append(
                        F.softplus(
                            tail_score - self.global_tn_tail_target_logit
                        )
                    )
                    if not fixed_negative_valid[batch_idx]:
                        batch_negative_scores.append(tail_score)
                        fixed_negative_scores[batch_idx] = tail_score
                        fixed_negative_valid[batch_idx] = True
                    else:
                        fixed_negative_scores[batch_idx] = torch.maximum(
                            fixed_negative_scores[batch_idx], tail_score
                        )
                    global_sample_count += 1
                    global_candidate_count += int(global_mask.sum().item())
            elif (
                global_tn_confidence_logits is not None
                and global_admitted is not None
                and confidence_negative_eligible is not None
                and bool(confidence_negative_eligible[batch_idx].item())
            ):
                global_mask = global_admitted[batch_idx]
                if bool(global_mask.any().item()):
                    tn_values = global_tn_confidence_logits[
                        batch_idx, global_mask
                    ].float()
                    global_negative_losses.append(F.softplus(tn_values).mean())
                    topk = min(self.global_tn_tail_topk, int(tn_values.numel()))
                    hard_values = torch.topk(
                        tn_values, k=topk, largest=True, sorted=False
                    ).values
                    tail_score = _normalized_logsumexp(
                        hard_values, self.global_tn_tail_temperature
                    )
                    global_tail_losses.append(
                        F.softplus(tail_score - self.global_tn_tail_target_logit)
                    )
                    batch_negative_scores.append(tail_score)
                    if fixed_negative_valid[batch_idx]:
                        fixed_negative_scores[batch_idx] = torch.maximum(
                            fixed_negative_scores[batch_idx], tail_score
                        )
                    else:
                        fixed_negative_scores[batch_idx] = tail_score
                        fixed_negative_valid[batch_idx] = True
                    global_sample_count += 1
                    global_candidate_count += int(tn_values.numel())

        loss_listwise = _mean_or_zero(listwise_losses, zero)
        loss_local_rank = _mean_or_zero(local_rank_losses, zero)
        loss_predicate_rank = _mean_or_zero(predicate_rank_losses, zero)
        positive_absolute = _mean_or_zero(
            absolute_positive_losses, confidence_zero
        )
        tn_absolute = _mean_or_zero(absolute_tn_losses, confidence_zero)
        if absolute_positive_losses and absolute_tn_losses:
            loss_local_absolute = 0.5 * positive_absolute + 0.5 * tn_absolute
        elif absolute_positive_losses:
            loss_local_absolute = positive_absolute
        elif absolute_tn_losses:
            loss_local_absolute = tn_absolute
        else:
            loss_local_absolute = confidence_zero
        deployed_global_positive_absolute = _mean_or_zero(
            deployed_global_absolute_positive_losses, confidence_zero
        )
        deployed_global_tn_absolute = _mean_or_zero(
            deployed_global_absolute_tn_losses, confidence_zero
        )
        if (
            deployed_global_absolute_positive_losses
            and deployed_global_absolute_tn_losses
        ):
            loss_deployed_global_absolute = (
                0.5 * deployed_global_positive_absolute
                + 0.5 * deployed_global_tn_absolute
            )
        elif deployed_global_absolute_positive_losses:
            loss_deployed_global_absolute = deployed_global_positive_absolute
        elif deployed_global_absolute_tn_losses:
            loss_deployed_global_absolute = deployed_global_tn_absolute
        else:
            loss_deployed_global_absolute = confidence_zero
        predicate_positive_absolute = _mean_or_zero(
            predicate_absolute_positive_losses, zero
        )
        predicate_tn_absolute = _mean_or_zero(predicate_absolute_tn_losses, zero)
        if predicate_absolute_positive_losses and predicate_absolute_tn_losses:
            loss_predicate_absolute = (
                0.5 * predicate_positive_absolute + 0.5 * predicate_tn_absolute
            )
        elif predicate_absolute_positive_losses:
            loss_predicate_absolute = predicate_positive_absolute
        elif predicate_absolute_tn_losses:
            loss_predicate_absolute = predicate_tn_absolute
        else:
            loss_predicate_absolute = zero
        if self.balance_local_anchor_classes:
            positive_anchor_loss = _mean_or_zero(positive_anchor_losses, zero)
            tn_anchor_loss = _mean_or_zero(tn_anchor_losses, zero)
            if positive_anchor_losses and tn_anchor_losses:
                loss_local_anchor = 0.5 * positive_anchor_loss + 0.5 * tn_anchor_loss
            elif positive_anchor_losses:
                loss_local_anchor = positive_anchor_loss
            elif tn_anchor_losses:
                loss_local_anchor = tn_anchor_loss
            else:
                loss_local_anchor = zero
        else:
            loss_local_anchor = _mean_or_zero(local_anchor_losses, zero)
        loss_global_negative = _mean_or_zero(
            global_negative_losses, confidence_zero
        )
        loss_global_tail = _mean_or_zero(global_tail_losses, confidence_zero)

        loss_batch_tail = confidence_zero
        if self.batch_tail_ddp_global:
            loss_batch_tail = fixed_batch_tail_separation_loss(
                torch.stack(fixed_positive_scores),
                torch.as_tensor(fixed_positive_valid, device=device, dtype=torch.bool),
                torch.stack(fixed_negative_scores),
                torch.as_tensor(fixed_negative_valid, device=device, dtype=torch.bool),
                positive_quantile=self.batch_positive_quantile,
                negative_quantile=self.batch_negative_quantile,
                margin=self.batch_tail_margin,
                ddp_global=True,
            )
        elif batch_positive_scores and batch_negative_scores:
            positive_scores = torch.stack(batch_positive_scores).float()
            negative_scores = torch.stack(batch_negative_scores).float()
            positive_tail = torch.quantile(positive_scores, self.batch_positive_quantile)
            negative_tail = torch.quantile(negative_scores, self.batch_negative_quantile)
            loss_batch_tail = F.softplus(
                negative_tail - positive_tail + self.batch_tail_margin
            )

        loss_tail_queue = confidence_zero
        tail_queue_positive_threshold = 0.0
        tail_queue_negative_threshold = 0.0
        tail_queue_threshold_valid = 0
        tail_queue_pair_loss = confidence_zero
        tail_queue_positive_trust_loss = confidence_zero
        tail_queue_positive_trust_violation_rate = 0.0
        tail_queue_negative_loss = confidence_zero
        tail_queue_negative_total_count = 0
        tail_queue_negative_active_count = 0
        tail_queue_negative_selected_count = 0
        tail_queue_negative_active_fraction = 0.0
        tail_queue_negative_active_min_logit = 0.0
        tail_queue_negative_inactive_max_logit = 0.0
        if (
            self.tail_queue_enabled
            and gathered_tail_payload is not None
            and int(self.tail_positive_count.item()) > 0
            and int(self.tail_negative_count.item()) > 0
            and int(self.tail_positive_count.item()) >= self.tail_queue_min_count
            and int(self.tail_negative_count.item()) >= self.tail_queue_min_count
        ):
            history_positive = self._tail_queue_values("positive").float()
            history_negative = self._tail_queue_values("negative").float()
            if self.tail_queue_objective == "fpr95":
                positive_threshold = _exact_lower_tail_operating_threshold(
                    history_positive, self.tail_queue_positive_quantile
                ).detach()
            else:
                positive_threshold = torch.quantile(
                    history_positive, self.tail_queue_positive_quantile
                ).detach()
            negative_threshold = torch.quantile(
                history_negative, self.tail_queue_negative_quantile
            ).detach()
            tail_queue_positive_threshold = float(positive_threshold.item())
            tail_queue_negative_threshold = float(negative_threshold.item())
            tail_queue_threshold_valid = 1
            current_positive = gathered_tail_payload[:, 0][
                (gathered_tail_payload[:, 1] > 0.5)
                & torch.isfinite(gathered_tail_payload[:, 0])
            ].float()
            current_negative = gathered_tail_payload[:, 2][
                (gathered_tail_payload[:, 3] > 0.5)
                & torch.isfinite(gathered_tail_payload[:, 2])
            ].float()
            if self.tail_queue_objective == "fpr95":
                tau = self.tail_queue_temperature
                loss_tail_queue = confidence_zero
                current_positive_gate = gathered_tail_payload[:, 4][
                    (gathered_tail_payload[:, 1] > 0.5)
                    & torch.isfinite(gathered_tail_payload[:, 4])
                ].float()
                surrogate_threshold = positive_threshold
                if (
                    positive_confidence_gate_logits is not None
                    and current_positive_gate.numel() > 0
                ):
                    if self.tail_queue_positive_gradient_contract == (
                        "exact_batch_lower_tail_st_v2"
                    ):
                        # Preserve the exact historical q05 forward value while
                        # routing its straight-through gradient through the
                        # current batch's matching order statistic.  With the
                        # formal B=16 contract this selects the minimum (6.25%
                        # tail), instead of letting a high positive mean hide a
                        # collapsing low tail.
                        gate_translation = _exact_lower_tail_operating_threshold(
                            current_positive_gate,
                            self.tail_queue_positive_quantile,
                        )
                    elif self.tail_queue_positive_gradient_contract == (
                        "mean_plus_exact_lower_tail_st_v3"
                    ):
                        # Keep the mean carrier's useful logit scale while
                        # adding a second straight-through path through the
                        # exact current-batch lower tail.  The tail term is
                        # zero in forward mode, so the historical q05
                        # threshold and score calibration remain unchanged.
                        gate_mean = current_positive_gate.mean()
                        gate_tail = _exact_lower_tail_operating_threshold(
                            current_positive_gate,
                            self.tail_queue_positive_quantile,
                        )
                        gate_translation = gate_mean + gate_tail - gate_tail.detach()
                    elif self.tail_queue_positive_gradient_contract == (
                        "mean_plus_quarter_exact_lower_tail_st_v4"
                    ):
                        # A quarter-strength exact-tail path retains the
                        # low-tail direction without doubling the mean carrier
                        # gradient.  Its forward value is still exactly the
                        # mean carrier, hence the historical q05 threshold is
                        # unchanged.
                        gate_mean = current_positive_gate.mean()
                        gate_tail = _exact_lower_tail_operating_threshold(
                            current_positive_gate,
                            self.tail_queue_positive_quantile,
                        )
                        gate_translation = (
                            gate_mean
                            + 0.25 * (gate_tail - gate_tail.detach())
                        )
                    elif self.tail_queue_positive_gradient_contract == (
                        "bounded_mean_plus_sixteenth_exact_lower_tail_st_v5"
                    ):
                        # Bound the broad positive-scale carrier in FP32 while
                        # retaining a small exact lower-tail direction.  The
                        # zero-forward residual keeps the historical q05 value
                        # bit-for-bit unchanged; only its backward path differs.
                        gate_mean = current_positive_gate.mean().float()
                        gate_tail = _exact_lower_tail_operating_threshold(
                            current_positive_gate,
                            self.tail_queue_positive_quantile,
                        )
                        bounded_mean = torch.tanh(gate_mean)
                        gate_translation = bounded_mean + 0.0625 * (
                            gate_tail - gate_tail.detach()
                        )
                    elif self.tail_queue_positive_gradient_contract == (
                        "elementwise_bounded_mean_plus_sixteenth_exact_lower_tail_st_v6"
                    ):
                        # Bound each positive gate before averaging so a single
                        # high-logit outlier cannot dominate every sample's
                        # carrier gradient.  The exact-tail residual remains a
                        # small independent direction with unchanged forward.
                        gate_mean = torch.tanh(
                            current_positive_gate.float()
                        ).mean()
                        gate_tail = _exact_lower_tail_operating_threshold(
                            current_positive_gate,
                            self.tail_queue_positive_quantile,
                        )
                        gate_translation = gate_mean + 0.0625 * (
                            gate_tail - gate_tail.detach()
                        )
                    else:
                        gate_translation = current_positive_gate.mean()
                    surrogate_threshold = (
                        positive_threshold
                        + gate_translation
                        - gate_translation.detach()
                    )
                if current_negative.numel() > 0:
                    (
                        tail_queue_negative_loss,
                        tail_queue_negative_active,
                        tail_queue_negative_selected,
                    ) = _fpr95_negative_softplus_loss(
                        current_negative,
                        surrogate_threshold=surrogate_threshold,
                        operating_threshold=positive_threshold,
                        temperature=tau,
                        margin=self.tail_queue_margin,
                        reduction_contract=(
                            self.tail_queue_negative_reduction_contract
                        ),
                    )
                    loss_tail_queue = tail_queue_negative_loss
                    tail_queue_negative_total_count = int(
                        current_negative.numel()
                    )
                    tail_queue_negative_active_count = int(
                        tail_queue_negative_active.sum().item()
                    )
                    tail_queue_negative_selected_count = int(
                        tail_queue_negative_selected.sum().item()
                    )
                    tail_queue_negative_active_fraction = float(
                        tail_queue_negative_active.float().mean().item()
                    )
                    if tail_queue_negative_active_count > 0:
                        tail_queue_negative_active_min_logit = float(
                            current_negative[tail_queue_negative_active]
                            .detach()
                            .min()
                            .item()
                        )
                    tail_queue_negative_inactive = (
                        ~tail_queue_negative_active
                    )
                    if bool(tail_queue_negative_inactive.any().item()):
                        tail_queue_negative_inactive_max_logit = float(
                            current_negative[tail_queue_negative_inactive]
                            .detach()
                            .max()
                            .item()
                        )

                pair_valid = (
                    (gathered_tail_payload[:, 1] > 0.5)
                    & (gathered_tail_payload[:, 3] > 0.5)
                    & torch.isfinite(gathered_tail_payload[:, 0])
                    & torch.isfinite(gathered_tail_payload[:, 2])
                )
                if self.tail_queue_pair_weight > 0.0 and bool(pair_valid.any().item()):
                    pair_positive = gathered_tail_payload[:, 0][pair_valid].float()
                    pair_negative = gathered_tail_payload[:, 2][pair_valid].float()
                    tail_queue_pair_loss = tau * F.softplus(
                        (
                            pair_negative
                            - pair_positive
                            + self.tail_queue_pair_margin
                        )
                        / tau
                    ).mean()
                    loss_tail_queue = (
                        loss_tail_queue
                        + self.tail_queue_pair_weight * tail_queue_pair_loss
                    )

                if (
                    self.tail_queue_positive_trust_weight > 0.0
                    and current_positive_gate.numel() > 0
                ):
                    trust_violation = F.relu(
                        -self.tail_queue_positive_trust_margin
                        - current_positive_gate
                    )
                    if self.tail_queue_positive_trust_reduction_contract == (
                        "top_quarter_cvar_v2"
                    ):
                        tail_queue_positive_trust_loss = _top_quarter_cvar(
                            trust_violation
                        )
                    else:
                        tail_queue_positive_trust_loss = trust_violation.mean()
                    tail_queue_positive_trust_violation_rate = float(
                        (trust_violation > 0.0).float().mean().detach().item()
                    )
                    loss_tail_queue = (
                        loss_tail_queue
                        + self.tail_queue_positive_trust_weight
                        * tail_queue_positive_trust_loss
                    )
            elif current_positive.numel() > 0 and current_negative.numel() > 0:
                tau = self.tail_queue_temperature
                positive_tail_fraction = max(
                    self.tail_queue_positive_quantile, 1e-3
                )
                negative_tail_fraction = max(
                    1.0 - self.tail_queue_negative_quantile, 1e-3
                )
                lower_positive = positive_threshold - (
                    tau
                    * F.softplus((positive_threshold - current_positive) / tau).mean()
                    / positive_tail_fraction
                )
                upper_negative = negative_threshold + (
                    tau
                    * F.softplus((current_negative - negative_threshold) / tau).mean()
                    / negative_tail_fraction
                )
                loss_tail_queue = tau * F.softplus(
                    (upper_negative - lower_positive + self.tail_queue_margin) / tau
                )

        losses = {
            "loss_fixed_text_listwise": loss_listwise,
            "loss_fixed_text_local_tn_rank": loss_local_rank,
            "loss_fixed_text_predicate_tn_rank": loss_predicate_rank,
            "loss_fixed_text_local_anchor": loss_local_anchor,
            "loss_fixed_text_global_tn_negative": loss_global_negative,
            "loss_fixed_text_global_tn_tail": loss_global_tail,
            "loss_fixed_text_batch_tail": loss_batch_tail,
            "loss_fixed_text_local_absolute": loss_local_absolute,
            "loss_fixed_text_deployed_global_absolute": (
                loss_deployed_global_absolute
            ),
            "loss_fixed_text_predicate_absolute": loss_predicate_absolute,
            "loss_fixed_text_tail_queue": loss_tail_queue,
            "loss_fixed_text_token": loss_token,
            "loss_fixed_text_raw_veto_gate": loss_raw_veto_gate,
            "loss_fixed_text_raw_veto_carrier_pair": (
                loss_raw_veto_carrier_pair
            ),
            "loss_fixed_text_deployed_veto_routing": (
                loss_deployed_veto_routing
            ),
        }
        total = zero
        for name, loss in losses.items():
            total = total + loss * float(self.weight_dict[name])

        stats = {
            "fixed_text_valid_listwise_count": valid_listwise_count,
            "fixed_text_positive_query_count": int(positive_mask.sum().item()),
            "fixed_text_negative_query_count": int(negative_mask.sum().item()),
            "fixed_text_local_pair_sample_count": local_pair_sample_count,
            "fixed_text_local_pair_query_count": local_pair_query_count,
            "fixed_text_predicate_pair_sample_count": predicate_pair_sample_count,
            "fixed_text_predicate_pair_query_count": predicate_pair_query_count,
            "fixed_text_predicate_pair_score_gap": float(
                _mean_or_zero(predicate_score_gaps, zero).detach().item()
            ),
            "fixed_text_predicate_pair_win_rate": float(
                _mean_or_zero(predicate_win_rates, zero).detach().item()
            ),
            "fixed_text_local_anchor_sample_count": local_anchor_sample_count,
            "fixed_text_local_anchor_paired_sample_count": local_anchor_paired_sample_count,
            "fixed_text_local_anchor_positive_query_count": local_anchor_positive_query_count,
            "fixed_text_local_anchor_tn_query_count": local_anchor_tn_query_count,
            "fixed_text_global_tn_sample_count": global_sample_count,
            "fixed_text_global_tn_candidate_count": global_candidate_count,
            "fixed_text_batch_positive_count": len(batch_positive_scores),
            "fixed_text_batch_negative_count": len(batch_negative_scores),
            "fixed_text_local_absolute_positive_sample_count": local_absolute_positive_sample_count,
            "fixed_text_local_absolute_tn_sample_count": local_absolute_tn_sample_count,
            "fixed_text_deployed_global_absolute_positive_sample_count": len(
                deployed_global_absolute_positive_losses
            ),
            "fixed_text_deployed_global_absolute_tn_sample_count": len(
                deployed_global_absolute_tn_losses
            ),
            "fixed_text_deployed_global_absolute_positive_loss": float(
                deployed_global_positive_absolute.detach().item()
            ),
            "fixed_text_deployed_global_absolute_tn_loss": float(
                deployed_global_tn_absolute.detach().item()
            ),
            "fixed_text_predicate_absolute_sample_count": predicate_absolute_sample_count,
            "fixed_text_tail_queue_positive_count": (
                int(self.tail_positive_count.item()) if self.tail_queue_enabled else 0
            ),
            "fixed_text_tail_queue_negative_count": (
                int(self.tail_negative_count.item()) if self.tail_queue_enabled else 0
            ),
            "fixed_text_tail_queue_positive_threshold": tail_queue_positive_threshold,
            "fixed_text_tail_queue_negative_threshold": tail_queue_negative_threshold,
            "fixed_text_tail_queue_threshold_valid": tail_queue_threshold_valid,
            "fixed_text_tail_queue_pair_loss": float(
                tail_queue_pair_loss.detach().item()
            ),
            "fixed_text_tail_queue_positive_trust_loss": float(
                tail_queue_positive_trust_loss.detach().item()
            ),
            "fixed_text_tail_queue_positive_trust_violation_rate": (
                tail_queue_positive_trust_violation_rate
            ),
            "fixed_text_tail_queue_negative_loss": float(
                tail_queue_negative_loss.detach().item()
            ),
            "fixed_text_tail_queue_negative_total_count": (
                tail_queue_negative_total_count
            ),
            "fixed_text_tail_queue_negative_active_count": (
                tail_queue_negative_active_count
            ),
            "fixed_text_tail_queue_negative_selected_count": (
                tail_queue_negative_selected_count
            ),
            "fixed_text_tail_queue_negative_active_fraction": (
                tail_queue_negative_active_fraction
            ),
            "fixed_text_tail_queue_negative_active_min_logit": (
                tail_queue_negative_active_min_logit
            ),
            "fixed_text_tail_queue_negative_inactive_max_logit": (
                tail_queue_negative_inactive_max_logit
            ),
            "fixed_text_token_sample_count": token_sample_count,
            "fixed_text_token_target_query_count": token_target_query_count,
            "fixed_text_token_positive_count": token_positive_count,
            "fixed_text_token_shared_count": token_shared_count,
            "fixed_text_token_edit_count": token_edit_count,
            "fixed_text_token_edit_query_count": token_edit_query_count,
            "fixed_text_token_edit_carrier_selected_count": (
                token_edit_carrier_selected_count
            ),
            "fixed_text_token_edit_carrier_added_count": (
                token_edit_carrier_added_count
            ),
            "fixed_text_token_edit_carrier_target_overlap_count": (
                token_edit_carrier_target_overlap_count
            ),
            "fixed_text_token_role_carrier_pair_selected_count": (
                token_role_carrier_pair_selected_count
            ),
            "fixed_text_token_role_carrier_positive_added_count": (
                token_role_carrier_positive_added_count
            ),
            "fixed_text_token_role_carrier_tn_added_count": (
                token_role_carrier_tn_added_count
            ),
            "fixed_text_token_role_carrier_positive_target_overlap_count": (
                token_role_carrier_positive_target_overlap_count
            ),
            "fixed_text_token_role_carrier_tn_target_overlap_count": (
                token_role_carrier_tn_target_overlap_count
            ),
            "fixed_text_token_all_negative_count": token_all_negative_count,
            "fixed_text_token_provenance_valid_count": token_provenance_valid_count,
            "fixed_text_token_direct_trace_valid_count": (
                token_direct_trace_valid_count
            ),
            "fixed_text_raw_veto_positive_sample_count": (
                raw_veto_positive_sample_count
            ),
            "fixed_text_raw_veto_tn_sample_count": raw_veto_tn_sample_count,
            "fixed_text_raw_veto_positive_query_count": (
                raw_veto_positive_query_count
            ),
            "fixed_text_raw_veto_tn_query_count": raw_veto_tn_query_count,
            "fixed_text_raw_veto_positive_source_mean": float(
                _mean_or_zero(raw_veto_positive_sources, raw_veto_zero)
                .detach()
                .item()
            ),
            "fixed_text_raw_veto_tn_changed_source_mean": float(
                _mean_or_zero(raw_veto_tn_sources, raw_veto_zero).detach().item()
            ),
            "fixed_text_raw_veto_source_separation": float(
                (
                    _mean_or_zero(raw_veto_tn_sources, raw_veto_zero)
                    - _mean_or_zero(raw_veto_positive_sources, raw_veto_zero)
                )
                .detach()
                .item()
            ),
            "fixed_text_raw_veto_positive_violation_rate": (
                float(raw_veto_positive_violation_count)
                / float(max(raw_veto_positive_query_count, 1))
            ),
            "fixed_text_raw_veto_tn_violation_rate": (
                float(raw_veto_tn_violation_count)
                / float(max(raw_veto_tn_query_count, 1))
            ),
            "fixed_text_raw_veto_positive_all_hinge_mean": float(
                _mean_or_zero(raw_veto_positive_all_hinges, raw_veto_zero)
                .detach()
                .item()
            ),
            "fixed_text_raw_veto_positive_carrier_hinge_mean": float(
                _mean_or_zero(
                    raw_veto_positive_carrier_hinges, raw_veto_zero
                )
                .detach()
                .item()
            ),
            "fixed_text_raw_veto_positive_loss_mean": float(
                _mean_or_zero(raw_veto_positive_losses, raw_veto_zero)
                .detach()
                .item()
            ),
            "fixed_text_raw_veto_tn_all_hinge_mean": float(
                _mean_or_zero(raw_veto_tn_all_hinges, raw_veto_zero)
                .detach()
                .item()
            ),
            "fixed_text_raw_veto_tn_carrier_hinge_mean": float(
                _mean_or_zero(raw_veto_tn_carrier_hinges, raw_veto_zero)
                .detach()
                .item()
            ),
            "fixed_text_raw_veto_tn_balanced_loss_mean": float(
                tn_raw_loss.detach().item()
            ),
            "fixed_text_raw_veto_tn_tail_threshold": raw_veto_tn_tail_threshold,
            "fixed_text_raw_veto_tn_tail_weight_mean": (
                raw_veto_tn_tail_weight_mean
            ),
            "fixed_text_raw_veto_tn_tail_effective_sample_count": (
                raw_veto_tn_tail_effective_sample_count
            ),
            "fixed_text_raw_veto_tn_tail_carrier_hinge_mean": (
                raw_veto_tn_tail_carrier_hinge_mean
            ),
            "fixed_text_raw_veto_positive_carrier_sample_count": (
                raw_veto_positive_carrier_sample_count
            ),
            "fixed_text_raw_veto_positive_carrier_source_mean": float(
                _mean_or_zero(
                    raw_veto_positive_carrier_sources, raw_veto_zero
                )
                .detach()
                .item()
            ),
            "fixed_text_raw_veto_positive_carrier_violation_rate": (
                float(raw_veto_positive_carrier_violation_count)
                / float(max(raw_veto_positive_carrier_sample_count, 1))
            ),
            "fixed_text_raw_veto_tn_carrier_sample_count": (
                raw_veto_tn_carrier_sample_count
            ),
            "fixed_text_raw_veto_tn_carrier_source_mean": float(
                _mean_or_zero(raw_veto_tn_carrier_sources, raw_veto_zero)
                .detach()
                .item()
            ),
            "fixed_text_raw_veto_tn_carrier_violation_rate": (
                float(raw_veto_tn_carrier_violation_count)
                / float(max(raw_veto_tn_carrier_sample_count, 1))
            ),
            "fixed_text_raw_veto_tn_carrier_changed_gate_mean": float(
                _mean_or_zero(
                    raw_veto_tn_carrier_changed_gates, raw_veto_zero
                )
                .detach()
                .item()
            ),
            "fixed_text_raw_veto_tn_carrier_full_open_rate": (
                float(raw_veto_tn_carrier_full_open_count)
                / float(max(raw_veto_tn_carrier_sample_count, 1))
            ),
            "fixed_text_raw_veto_carrier_pair_sample_count": len(
                raw_veto_carrier_pair_hinges
            ),
            "fixed_text_raw_veto_carrier_pair_gap_mean": float(
                _mean_or_zero(raw_veto_carrier_pair_gaps, raw_veto_zero)
                .detach()
                .item()
            ),
            "fixed_text_raw_veto_carrier_pair_hinge_mean": float(
                _mean_or_zero(raw_veto_carrier_pair_hinges, raw_veto_zero)
                .detach()
                .item()
            ),
            "fixed_text_raw_veto_carrier_pair_violation_rate": (
                float(raw_veto_carrier_pair_violation_count)
                / float(max(len(raw_veto_carrier_pair_hinges), 1))
            ),
            "fixed_text_deployed_veto_routing_positive_sample_count": int(
                deployed_veto_positive_winner_gates.numel()
            ),
            "fixed_text_deployed_veto_routing_tn_sample_count": int(
                deployed_veto_tn_winner_gates.numel()
            ),
            "fixed_text_deployed_veto_routing_positive_winner_gate_mean": float(
                (
                    deployed_veto_positive_winner_gates.mean()
                    if deployed_veto_positive_winner_gates.numel() > 0
                    else deployed_veto_zero
                )
                .detach()
                .item()
            ),
            "fixed_text_deployed_veto_routing_tn_winner_gate_mean": float(
                (
                    deployed_veto_tn_winner_gates.mean()
                    if deployed_veto_tn_winner_gates.numel() > 0
                    else deployed_veto_zero
                )
                .detach()
                .item()
            ),
            "fixed_text_deployed_veto_routing_positive_coverage_mean": float(
                (
                    deployed_veto_positive_coverages.mean()
                    if deployed_veto_positive_coverages.numel() > 0
                    else deployed_veto_zero
                )
                .detach()
                .item()
            ),
            "fixed_text_deployed_veto_routing_tn_coverage_mean": float(
                (
                    deployed_veto_tn_coverages.mean()
                    if deployed_veto_tn_coverages.numel() > 0
                    else deployed_veto_zero
                )
                .detach()
                .item()
            ),
            "fixed_text_deployed_veto_routing_positive_winner_hinge_mean": float(
                (
                    deployed_veto_positive_winner_hinges.mean()
                    if deployed_veto_positive_winner_hinges.numel() > 0
                    else deployed_veto_zero
                )
                .detach()
                .item()
            ),
            "fixed_text_deployed_veto_routing_tn_winner_hinge_mean": float(
                (
                    deployed_veto_tn_winner_hinges.mean()
                    if deployed_veto_tn_winner_hinges.numel() > 0
                    else deployed_veto_zero
                )
                .detach()
                .item()
            ),
            "fixed_text_deployed_veto_routing_positive_coverage_hinge_mean": float(
                (
                    deployed_veto_positive_coverage_hinges.mean()
                    if deployed_veto_positive_coverage_hinges.numel() > 0
                    else deployed_veto_zero
                )
                .detach()
                .item()
            ),
            "fixed_text_deployed_veto_routing_tn_coverage_hinge_mean": float(
                (
                    deployed_veto_tn_coverage_hinges.mean()
                    if deployed_veto_tn_coverage_hinges.numel() > 0
                    else deployed_veto_zero
                )
                .detach()
                .item()
            ),
            "fixed_text_deployed_veto_routing_positive_winner_violation_rate": (
                float(
                    (deployed_veto_positive_winner_hinges > 0.0)
                    .float()
                    .mean()
                    .detach()
                    .item()
                )
                if deployed_veto_positive_winner_hinges.numel() > 0
                else 0.0
            ),
            "fixed_text_deployed_veto_routing_tn_winner_violation_rate": (
                float(
                    (deployed_veto_tn_winner_hinges > 0.0)
                    .float()
                    .mean()
                    .detach()
                    .item()
                )
                if deployed_veto_tn_winner_hinges.numel() > 0
                else 0.0
            ),
            "fixed_text_deployed_veto_routing_positive_coverage_violation_rate": (
                float(
                    (deployed_veto_positive_coverage_hinges > 0.0)
                    .float()
                    .mean()
                    .detach()
                    .item()
                )
                if deployed_veto_positive_coverage_hinges.numel() > 0
                else 0.0
            ),
            "fixed_text_deployed_veto_routing_tn_coverage_violation_rate": (
                float(
                    (deployed_veto_tn_coverage_hinges > 0.0)
                    .float()
                    .mean()
                    .detach()
                    .item()
                )
                if deployed_veto_tn_coverage_hinges.numel() > 0
                else 0.0
            ),
            "fixed_text_deployed_veto_routing_winner_loss_mean": float(
                deployed_veto_winner_loss.detach().item()
            ),
            "fixed_text_deployed_veto_routing_coverage_loss_mean": float(
                deployed_veto_coverage_loss.detach().item()
            ),
        }
        if tail_weighted_pair_scope:
            stats.update(
                {
                    "fixed_text_raw_veto_tail_pair_gap_mean": (
                        raw_veto_tail_pair_gap_mean
                    ),
                    "fixed_text_raw_veto_tail_pair_hinge_mean": (
                        raw_veto_tail_pair_hinge_mean
                    ),
                    "fixed_text_raw_veto_tail_pair_violation_rate": (
                        raw_veto_tail_pair_violation_rate
                    ),
                    "fixed_text_raw_veto_tail_pair_effective_sample_count": (
                        raw_veto_tail_pair_effective_sample_count
                    ),
                }
            )
        if tn_train_eligible is not None:
            source_eligible = (
                torch.zeros_like(tn_train_eligible)
                if confidence_negative_source_eligible is None
                else confidence_negative_source_eligible
            )
            stats.update(
                {
                    "fixed_text_confidence_tn_train_eligible_count": int(
                        (source_eligible & tn_train_eligible).sum().item()
                    ),
                    "fixed_text_confidence_tn_train_excluded_count": int(
                        (source_eligible & ~tn_train_eligible).sum().item()
                    ),
                }
            )
        result = dict(losses)
        result["loss_stage_b_fixed_text"] = total
        result.update(
            {
                name: torch.as_tensor(float(value), device=device, dtype=torch.float32)
                for name, value in stats.items()
            }
        )
        if ablation_eligible is not None:
            result["fixed_text_confidence_ablation_eligible_count"] = (
                torch.as_tensor(
                    float(ablation_eligible.sum().item()),
                    device=device,
                    dtype=torch.float32,
                )
            )
        return result


__all__ = [
    "StageBFixedTextCriterion",
    "candidate_max_iou",
    "fixed_batch_tail_separation_loss",
    "multi_positive_candidate_listwise_loss",
]
