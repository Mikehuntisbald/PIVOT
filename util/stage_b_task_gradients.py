"""Gradient diagnostics for the Stage-B rank/confidence ownership ablation."""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence, Tuple

import torch


RANK_LOSS_KEYS = (
    "loss_fixed_text_listwise",
    "loss_fixed_text_local_tn_rank",
    "loss_fixed_text_predicate_tn_rank",
    "loss_fixed_text_local_anchor",
    "loss_fixed_text_predicate_absolute",
    "loss_fixed_text_token",
)

CONFIDENCE_LOSS_KEYS = (
    "loss_fixed_text_global_tn_negative",
    "loss_fixed_text_global_tn_tail",
    "loss_fixed_text_batch_tail",
    "loss_fixed_text_local_absolute",
    "loss_fixed_text_deployed_global_absolute",
    "loss_fixed_text_tail_queue",
)


def _weighted_group_loss(
    loss_dict: Mapping[str, torch.Tensor],
    weight_dict: Mapping[str, float],
    keys: Sequence[str],
    *,
    label: str,
) -> torch.Tensor:
    terms = []
    for key in keys:
        weight = float(weight_dict.get(key, 0.0))
        if weight == 0.0 or key not in loss_dict:
            continue
        value = loss_dict[key]
        if not torch.is_tensor(value) or value.numel() != 1:
            raise TypeError(f"{label} loss {key!r} must be a scalar tensor")
        terms.append(value * weight)
    if not terms:
        raise RuntimeError(
            f"No active {label} losses were found; checked keys={tuple(keys)}"
        )
    result = sum(terms[1:], terms[0])
    if not result.requires_grad:
        raise RuntimeError(f"Active {label} loss has no autograd graph")
    return result


def weighted_stage_b_task_losses(
    loss_dict: Mapping[str, torch.Tensor],
    weight_dict: Mapping[str, float],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the two weighted task losses without changing the train objective."""
    return (
        _weighted_group_loss(
            loss_dict, weight_dict, RANK_LOSS_KEYS, label="rank"
        ),
        _weighted_group_loss(
            loss_dict,
            weight_dict,
            CONFIDENCE_LOSS_KEYS,
            label="confidence",
        ),
    )


def _task_gradients(
    loss: torch.Tensor,
    named_parameters: Sequence[Tuple[str, torch.nn.Parameter]],
    *,
    retain_graph: bool,
) -> Tuple[torch.Tensor | None, ...]:
    if not named_parameters:
        raise RuntimeError("Gradient diagnostic received no trainable parameters")
    return torch.autograd.grad(
        loss,
        tuple(parameter for _, parameter in named_parameters),
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )


def gradient_conflict_report(
    rank_loss: torch.Tensor,
    confidence_loss: torch.Tensor,
    named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
) -> dict:
    """Measure two-task gradients on every jointly connected trainable tensor.

    A parameter is shared only when both losses are structurally connected to
    it. The function fails closed for independent branches because reporting a
    zero cosine over an empty shared set would falsely imply no conflict.
    """
    named = tuple(
        (str(name), parameter)
        for name, parameter in named_parameters
        if parameter.requires_grad
    )
    rank_gradients = _task_gradients(rank_loss, named, retain_graph=True)
    confidence_gradients = _task_gradients(
        confidence_loss, named, retain_graph=True
    )
    shared = [
        (name, rank_gradient, confidence_gradient)
        for (name, _), rank_gradient, confidence_gradient in zip(
            named, rank_gradients, confidence_gradients
        )
        if rank_gradient is not None and confidence_gradient is not None
    ]
    if not shared:
        raise RuntimeError(
            "No shared trainable parameters connect both rank and confidence "
            "losses; use the branch-isolation diagnostic for independent branches"
        )

    dot = torch.zeros((), dtype=torch.float64, device=shared[0][1].device)
    rank_sq = torch.zeros_like(dot)
    confidence_sq = torch.zeros_like(dot)
    conflict_elements = 0
    finite_elements = 0
    tensor_conflicts = 0
    shared_elements = 0
    for _name, rank_gradient, confidence_gradient in shared:
        rank_value = rank_gradient.detach().to(dtype=torch.float64)
        confidence_value = confidence_gradient.detach().to(dtype=torch.float64)
        if rank_value.is_sparse:
            rank_value = rank_value.to_dense()
        if confidence_value.is_sparse:
            confidence_value = confidence_value.to_dense()
        finite = torch.isfinite(rank_value) & torch.isfinite(confidence_value)
        if not bool(finite.all().item()):
            rank_value = rank_value[finite]
            confidence_value = confidence_value[finite]
        product = rank_value * confidence_value
        tensor_dot = product.sum()
        dot = dot + tensor_dot
        rank_sq = rank_sq + rank_value.square().sum()
        confidence_sq = confidence_sq + confidence_value.square().sum()
        count = int(product.numel())
        shared_elements += int(rank_gradient.numel())
        finite_elements += count
        conflict_elements += int((product < 0).sum().item())
        tensor_conflicts += int(float(tensor_dot.item()) < 0.0)

    rank_norm = float(torch.sqrt(rank_sq).item())
    confidence_norm = float(torch.sqrt(confidence_sq).item())
    denominator = rank_norm * confidence_norm
    cosine_defined = math.isfinite(denominator) and denominator > 0.0
    cosine = float(dot.item() / denominator) if cosine_defined else 0.0
    return {
        "cosine": cosine,
        "cosine_defined": bool(cosine_defined),
        "rank_norm": rank_norm,
        "confidence_norm": confidence_norm,
        "element_conflict_fraction": (
            float(conflict_elements / finite_elements) if finite_elements else 0.0
        ),
        "tensor_conflict_fraction": float(tensor_conflicts / len(shared)),
        "shared_parameter_count": len(shared),
        "shared_element_count": shared_elements,
        "finite_element_count": finite_elements,
        "shared_parameter_names": tuple(name for name, _, _ in shared),
    }


def branch_isolation_report(
    rank_loss: torch.Tensor,
    confidence_loss: torch.Tensor,
    rank_named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
    confidence_named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
) -> dict:
    """Fail if either task is structurally connected to the other branch."""
    rank_named = tuple(
        (str(name), parameter)
        for name, parameter in rank_named_parameters
        if parameter.requires_grad
    )
    confidence_named = tuple(
        (str(name), parameter)
        for name, parameter in confidence_named_parameters
        if parameter.requires_grad
    )
    if not rank_named or not confidence_named:
        raise RuntimeError(
            "Branch-isolation diagnostic requires non-empty trainable rank and "
            "confidence parameter groups"
        )
    ids = {id(parameter) for _, parameter in rank_named}
    overlap = [name for name, parameter in confidence_named if id(parameter) in ids]
    if overlap:
        raise RuntimeError(
            f"Rank and confidence parameter groups overlap: {overlap[:8]}"
        )

    combined = rank_named + confidence_named
    rank_gradients = _task_gradients(rank_loss, combined, retain_graph=True)
    confidence_gradients = _task_gradients(
        confidence_loss, combined, retain_graph=True
    )
    split = len(rank_named)
    rank_own = rank_gradients[:split]
    rank_cross = rank_gradients[split:]
    confidence_cross = confidence_gradients[:split]
    confidence_own = confidence_gradients[split:]
    if not any(gradient is not None for gradient in rank_own):
        raise RuntimeError("Rank loss is not connected to the declared rank branch")
    if not any(gradient is not None for gradient in confidence_own):
        raise RuntimeError(
            "Confidence loss is not connected to the declared confidence branch"
        )
    rank_violations = [
        name
        for (name, _), gradient in zip(confidence_named, rank_cross)
        if gradient is not None
    ]
    confidence_violations = [
        name
        for (name, _), gradient in zip(rank_named, confidence_cross)
        if gradient is not None
    ]
    if rank_violations or confidence_violations:
        raise RuntimeError(
            "Stage-B branch isolation failed: "
            f"rank_loss_to_confidence={rank_violations[:8]}, "
            f"confidence_loss_to_rank={confidence_violations[:8]}"
        )
    return {
        "passed": True,
        "rank_parameter_count": len(rank_named),
        "confidence_parameter_count": len(confidence_named),
    }


__all__ = [
    "CONFIDENCE_LOSS_KEYS",
    "RANK_LOSS_KEYS",
    "branch_isolation_report",
    "gradient_conflict_report",
    "weighted_stage_b_task_losses",
]
