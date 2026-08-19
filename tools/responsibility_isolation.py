"""Baseline-agnostic heads for a frozen candidate model.

The module consumes cached candidate features and scores.  It never sends a
gradient back to the candidate model.  Two ownership layouts are provided for
the responsibility-isolation experiment:

``shared_trunk_two_heads``
    Ranking and absolute confidence both use the same capacity-matched pair of
    trainable feature towers.  Their losses therefore have a deliberate,
    auditable shared-gradient path.

``isolated_two_trunks``
    Ranking and absolute confidence have disjoint trainable towers.  Neither
    loss is structurally connected to the other task's parameters.

The confidence score is an absolute logit.  It is not added to, multiplied by,
or otherwise used to construct ``rank_score``.  In isolated mode a confidence
optimizer step consequently cannot change candidate ordering.
"""

from __future__ import annotations

import copy
import math
from typing import Iterable, Mapping

import torch
from torch import Tensor, nn


RESPONSIBILITY_OWNERSHIP_SHARED = "shared_trunk_two_heads"
RESPONSIBILITY_OWNERSHIP_ISOLATED = "isolated_two_trunks"
RESPONSIBILITY_OWNERSHIP_MODES = (
    RESPONSIBILITY_OWNERSHIP_SHARED,
    RESPONSIBILITY_OWNERSHIP_ISOLATED,
)


def normalize_responsibility_ownership(value: str) -> str:
    """Return a closed ownership label; aliases are intentionally rejected."""
    normalized = str(value).strip().lower()
    if normalized not in RESPONSIBILITY_OWNERSHIP_MODES:
        raise ValueError(
            "responsibility ownership must be one of "
            f"{RESPONSIBILITY_OWNERSHIP_MODES}, got {value!r}"
        )
    return normalized


def _positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _positive_float(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


class _CandidateTower(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, value: Tensor) -> Tensor:
        return self.layers(self.norm(value))


class FrozenCandidateResponsibilityHeads(nn.Module):
    """Ranking and rejection heads over a frozen candidate cache.

    Args:
        feature_dim: Last dimension of ``query_features``.
        hidden_dim: Private/shared tower width.
        ownership: ``shared_trunk_two_heads`` or ``isolated_two_trunks``.
        rank_residual_limit: Symmetric bound on the learned rank residual.

    ``query_features`` and ``native_score`` must be finite tensors with shapes
    ``(B, Q, feature_dim)`` and ``(B, Q)``.  ``candidate_mask`` is a boolean
    ``(B, Q)`` mask with at least one candidate per row.  Inputs are detached
    inside ``forward`` even when a caller accidentally supplies live tensors.
    """

    def __init__(
        self,
        *,
        feature_dim: int = 256,
        hidden_dim: int = 128,
        ownership: str = RESPONSIBILITY_OWNERSHIP_ISOLATED,
        rank_residual_limit: float = 0.1,
    ) -> None:
        super().__init__()
        self.feature_dim = _positive_int(feature_dim, name="feature_dim")
        self.hidden_dim = _positive_int(hidden_dim, name="hidden_dim")
        self.ownership = normalize_responsibility_ownership(ownership)
        self.rank_residual_limit = _positive_float(
            rank_residual_limit, name="rank_residual_limit"
        )
        input_dim = self.feature_dim + 1
        # Both causal arms instantiate the exact same tensors and execute both
        # towers.  The only change is graph ownership: the shared arm averages
        # both representations for both tasks, while the isolated arm assigns
        # one tower to each task.  This avoids confounding responsibility
        # isolation with trainable parameter count or tower FLOPs.
        self.rank_trunk = _CandidateTower(input_dim, self.hidden_dim)
        self.confidence_trunk = copy.deepcopy(self.rank_trunk)
        self.rank_head = nn.Linear(self.hidden_dim, 1)
        self.confidence_head = nn.Linear(self.hidden_dim, 1)
        # Native ranking is the exact U0 behavior; confidence starts as a
        # neutral absolute logit.  Both tasks can learn on their first update.
        nn.init.zeros_(self.rank_head.weight)
        nn.init.zeros_(self.rank_head.bias)
        nn.init.zeros_(self.confidence_head.weight)
        nn.init.zeros_(self.confidence_head.bias)

    def _named_submodule_parameters(
        self, prefix: str, module: nn.Module
    ) -> tuple[tuple[str, nn.Parameter], ...]:
        return tuple(
            (f"{prefix}.{name}", parameter)
            for name, parameter in module.named_parameters()
        )

    def named_task_parameters(
        self, task: str
    ) -> tuple[tuple[str, nn.Parameter], ...]:
        """Return the exact optimizer ownership for ``rank`` or ``confidence``."""
        task = str(task).strip().lower()
        if task not in ("rank", "confidence"):
            raise ValueError("task must be 'rank' or 'confidence'")
        head = self.rank_head if task == "rank" else self.confidence_head
        if self.ownership == RESPONSIBILITY_OWNERSHIP_SHARED:
            towers = (
                self._named_submodule_parameters("rank_trunk", self.rank_trunk)
                + self._named_submodule_parameters(
                    "confidence_trunk", self.confidence_trunk
                )
            )
        else:
            trunk_name = f"{task}_trunk"
            trunk = self.rank_trunk if task == "rank" else self.confidence_trunk
            towers = self._named_submodule_parameters(trunk_name, trunk)
        return towers + self._named_submodule_parameters(f"{task}_head", head)

    def task_parameters(self, task: str) -> tuple[nn.Parameter, ...]:
        return tuple(parameter for _, parameter in self.named_task_parameters(task))

    def _validate_inputs(
        self,
        query_features: Tensor,
        native_score: Tensor,
        candidate_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if (
            not torch.is_tensor(query_features)
            or query_features.dim() != 3
            or not query_features.is_floating_point()
            or int(query_features.shape[-1]) != self.feature_dim
            or int(query_features.shape[0]) <= 0
            or int(query_features.shape[1]) <= 0
        ):
            raise ValueError(
                "query_features must be a nonempty floating (B,Q,feature_dim) tensor"
            )
        if (
            not torch.is_tensor(native_score)
            or native_score.dim() != 2
            or not native_score.is_floating_point()
            or tuple(native_score.shape) != tuple(query_features.shape[:2])
        ):
            raise ValueError("native_score must be a floating (B,Q) tensor")
        if (
            not torch.is_tensor(candidate_mask)
            or candidate_mask.dtype != torch.bool
            or tuple(candidate_mask.shape) != tuple(native_score.shape)
        ):
            raise ValueError("candidate_mask must be a boolean (B,Q) tensor")
        if not (
            query_features.device
            == native_score.device
            == candidate_mask.device
        ):
            raise ValueError("candidate inputs must be on the same device")
        reference = self.rank_head.weight
        if query_features.device != reference.device:
            raise ValueError("candidate inputs and responsibility heads differ in device")
        if not bool(candidate_mask.any(dim=1).all().item()):
            raise ValueError("candidate_mask must contain at least one candidate per row")
        if not bool(torch.isfinite(query_features).all().item()):
            raise ValueError("query_features must be finite")
        if not bool(torch.isfinite(native_score).all().item()):
            raise ValueError("native_score must be finite")
        # Cached features may use a storage dtype different from the heads
        # (for example fp16 cache with fp32/AMP training).  Conversion remains
        # outside the candidate model's graph because detachment happens first.
        return (
            query_features.detach().to(dtype=reference.dtype),
            native_score.detach().to(dtype=reference.dtype),
            candidate_mask,
        )

    def forward(
        self,
        query_features: Tensor,
        native_score: Tensor,
        candidate_mask: Tensor,
    ) -> dict[str, Tensor]:
        features, native, mask = self._validate_inputs(
            query_features, native_score, candidate_mask
        )
        frozen_input = torch.cat((features, native.unsqueeze(-1)), dim=-1)
        rank_private = self.rank_trunk(frozen_input)
        confidence_private = self.confidence_trunk(frozen_input)
        if self.ownership == RESPONSIBILITY_OWNERSHIP_SHARED:
            # The towers start bitwise equal, so their average preserves the
            # isolated arm's initial function while connecting both losses to
            # both equal-capacity towers.
            shared_hidden = 0.5 * (rank_private + confidence_private)
            rank_hidden = confidence_hidden = shared_hidden
        else:
            rank_hidden = rank_private
            confidence_hidden = confidence_private

        raw_rank_residual = self.rank_head(rank_hidden).squeeze(-1)
        limit = self.rank_residual_limit
        rank_residual = limit * torch.tanh(raw_rank_residual / limit)
        rank_residual = rank_residual.masked_fill(~mask, 0.0)
        pre_mask_rank_score = native + rank_residual
        # A finite floor keeps audit/serialization tools finite while making an
        # invalid candidate strictly worse than any ordinary finite score.
        rank_floor = torch.finfo(pre_mask_rank_score.dtype).min
        if bool((pre_mask_rank_score[mask] == rank_floor).any().item()):
            raise ValueError("valid rank score equals the reserved candidate floor")
        rank_score = pre_mask_rank_score.masked_fill(~mask, rank_floor)

        # This is an absolute logit, not a residual on native/rank score.  Loss
        # code must select/mask valid candidates using ``candidate_mask``.
        confidence_score = self.confidence_head(confidence_hidden).squeeze(-1)
        if not all(
            bool(torch.isfinite(value).all().item())
            for value in (rank_residual, rank_score, confidence_score)
        ):
            raise RuntimeError("responsibility head produced a non-finite score")
        return {
            "native_score": native,
            "candidate_mask": mask,
            "rank_residual": rank_residual,
            "rank_score": rank_score,
            "confidence_score": confidence_score,
        }


def _canonical_parameter_map(
    module: FrozenCandidateResponsibilityHeads,
) -> dict[int, tuple[str, nn.Parameter]]:
    return {
        id(parameter): (name, parameter)
        for name, parameter in sorted(module.named_parameters())
        if parameter.requires_grad
    }


def responsibility_ownership_report(
    module: FrozenCandidateResponsibilityHeads,
) -> dict:
    """Return a deterministic, JSON-safe optimizer-ownership audit."""
    if not isinstance(module, FrozenCandidateResponsibilityHeads):
        raise TypeError("ownership audit requires FrozenCandidateResponsibilityHeads")
    canonical = _canonical_parameter_map(module)
    rank_ids = {id(parameter) for parameter in module.task_parameters("rank")}
    confidence_ids = {
        id(parameter) for parameter in module.task_parameters("confidence")
    }
    all_ids = set(canonical)
    unowned = all_ids - rank_ids - confidence_ids
    unknown = (rank_ids | confidence_ids) - all_ids
    if unowned or unknown:
        raise RuntimeError(
            "responsibility ownership is incomplete: "
            f"unowned={len(unowned)}, unknown={len(unknown)}"
        )

    def names(ids: Iterable[int]) -> tuple[str, ...]:
        return tuple(sorted(canonical[parameter_id][0] for parameter_id in ids))

    def elements(ids: Iterable[int]) -> int:
        return int(
            sum(canonical[parameter_id][1].numel() for parameter_id in ids)
        )

    shared_ids = rank_ids & confidence_ids
    report = {
        "ownership": module.ownership,
        "rank_parameter_names": names(rank_ids),
        "confidence_parameter_names": names(confidence_ids),
        "shared_parameter_names": names(shared_ids),
        "rank_tensor_count": len(rank_ids),
        "confidence_tensor_count": len(confidence_ids),
        "shared_tensor_count": len(shared_ids),
        "rank_element_count": elements(rank_ids),
        "confidence_element_count": elements(confidence_ids),
        "shared_element_count": elements(shared_ids),
        "all_trainable_tensor_count": len(all_ids),
        "all_trainable_element_count": elements(all_ids),
    }
    expected_shared = module.ownership == RESPONSIBILITY_OWNERSHIP_SHARED
    if bool(shared_ids) != expected_shared:
        raise RuntimeError(
            "ownership label does not match parameter topology: "
            f"ownership={module.ownership!r}, shared={names(shared_ids)}"
        )
    return report


def _validate_scalar_loss(loss: Tensor, *, name: str) -> None:
    if not torch.is_tensor(loss) or loss.numel() != 1:
        raise TypeError(f"{name} must be a scalar tensor")
    if not loss.requires_grad:
        raise RuntimeError(f"{name} has no autograd graph")
    if not bool(torch.isfinite(loss.detach()).item()):
        raise ValueError(f"{name} must be finite")


def responsibility_gradient_report(
    module: FrozenCandidateResponsibilityHeads,
    rank_loss: Tensor,
    confidence_loss: Tensor,
) -> dict:
    """Audit task-to-parameter paths without mutating ``.grad`` fields.

    A gradient tensor with all zeros still counts as a structural connection;
    this matters for zero-initialized heads.  ``None`` is the only indication
    that no autograd path exists.
    """
    _validate_scalar_loss(rank_loss, name="rank_loss")
    _validate_scalar_loss(confidence_loss, name="confidence_loss")
    ownership = responsibility_ownership_report(module)
    canonical = _canonical_parameter_map(module)
    ordered = tuple(canonical[parameter_id] for parameter_id in sorted(
        canonical, key=lambda value: canonical[value][0]
    ))
    parameters = tuple(parameter for _, parameter in ordered)
    rank_gradients = torch.autograd.grad(
        rank_loss,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    confidence_gradients = torch.autograd.grad(
        confidence_loss,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    rank_connected_ids = {
        id(parameter)
        for (_, parameter), gradient in zip(ordered, rank_gradients)
        if gradient is not None
    }
    confidence_connected_ids = {
        id(parameter)
        for (_, parameter), gradient in zip(ordered, confidence_gradients)
        if gradient is not None
    }
    rank_ids = {id(parameter) for parameter in module.task_parameters("rank")}
    confidence_ids = {
        id(parameter) for parameter in module.task_parameters("confidence")
    }
    jointly_connected_ids = rank_connected_ids & confidence_connected_ids
    rank_cross_ids = rank_connected_ids & (confidence_ids - rank_ids)
    confidence_cross_ids = confidence_connected_ids & (rank_ids - confidence_ids)

    finite = True
    for gradient in (*rank_gradients, *confidence_gradients):
        if gradient is not None and not bool(torch.isfinite(gradient).all().item()):
            finite = False
            break

    # Cosine is defined only over parameters connected to both tasks.
    rank_by_id = {
        id(parameter): gradient
        for (_, parameter), gradient in zip(ordered, rank_gradients)
        if gradient is not None
    }
    confidence_by_id = {
        id(parameter): gradient
        for (_, parameter), gradient in zip(ordered, confidence_gradients)
        if gradient is not None
    }
    dot = 0.0
    rank_sq = 0.0
    confidence_sq = 0.0
    for parameter_id in jointly_connected_ids:
        rank_value = rank_by_id[parameter_id].detach().double()
        confidence_value = confidence_by_id[parameter_id].detach().double()
        dot += float((rank_value * confidence_value).sum().item())
        rank_sq += float(rank_value.square().sum().item())
        confidence_sq += float(confidence_value.square().sum().item())
    denominator = math.sqrt(rank_sq) * math.sqrt(confidence_sq)
    cosine_defined = math.isfinite(denominator) and denominator > 0.0
    cosine = float(dot / denominator) if cosine_defined else None

    def names(ids: Iterable[int]) -> tuple[str, ...]:
        return tuple(sorted(canonical[parameter_id][0] for parameter_id in ids))

    structurally_isolated = (
        module.ownership == RESPONSIBILITY_OWNERSHIP_ISOLATED
        and not jointly_connected_ids
        and not rank_cross_ids
        and not confidence_cross_ids
    )
    return {
        **ownership,
        "rank_connected_parameter_names": names(rank_connected_ids),
        "confidence_connected_parameter_names": names(confidence_connected_ids),
        "jointly_connected_parameter_names": names(jointly_connected_ids),
        "rank_loss_to_confidence_only_parameter_names": names(rank_cross_ids),
        "confidence_loss_to_rank_only_parameter_names": names(
            confidence_cross_ids
        ),
        "gradient_finite": finite,
        "joint_gradient_cosine": cosine,
        "joint_gradient_cosine_defined": cosine_defined,
        "structurally_isolated": structurally_isolated,
    }


def assert_isolated_responsibility_gradients(
    module: FrozenCandidateResponsibilityHeads,
    rank_loss: Tensor,
    confidence_loss: Tensor,
) -> Mapping[str, object]:
    """Fail closed unless the isolated two-trunk graph is bidirectionally clean."""
    report = responsibility_gradient_report(module, rank_loss, confidence_loss)
    if not report["gradient_finite"]:
        raise RuntimeError("responsibility gradients contain non-finite values")
    if not report["structurally_isolated"]:
        raise RuntimeError(
            "responsibility gradient isolation failed: "
            f"joint={report['jointly_connected_parameter_names']}, "
            "rank_to_confidence="
            f"{report['rank_loss_to_confidence_only_parameter_names']}, "
            "confidence_to_rank="
            f"{report['confidence_loss_to_rank_only_parameter_names']}"
        )
    return report


__all__ = [
    "RESPONSIBILITY_OWNERSHIP_ISOLATED",
    "RESPONSIBILITY_OWNERSHIP_MODES",
    "RESPONSIBILITY_OWNERSHIP_SHARED",
    "FrozenCandidateResponsibilityHeads",
    "assert_isolated_responsibility_gradients",
    "normalize_responsibility_ownership",
    "responsibility_gradient_report",
    "responsibility_ownership_report",
]
