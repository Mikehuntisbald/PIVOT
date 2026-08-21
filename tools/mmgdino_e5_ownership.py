"""Capacity-controlled responsibility owners for the MM-GDINO e5 transfer.

The frozen detector supplies one 256-D query feature and one native score per
candidate.  This module owns only the two paper duties layered on top:

* a bounded residual for relative ranking; and
* an absolute confidence logit for rejection.

The three formal arms differ only in feature ownership.  Shared-128 provides
the same 128-D representation to both duties.  Shared-Wide is the capacity
control: a single 212->210 tower has effectively the same total parameters and
dual-output MAC count as two isolated 128-D towers.  Isolated gives each duty
its own 128-D tower.  Inputs are detached internally, so no detector graph can
be reached accidentally.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn


OWNERSHIP_SHARED_128 = "shared_128"
OWNERSHIP_SHARED_WIDE = "shared_wide"
OWNERSHIP_ISOLATED_128 = "isolated_128"
OWNERSHIP_MODES = (
    OWNERSHIP_SHARED_128,
    OWNERSHIP_SHARED_WIDE,
    OWNERSHIP_ISOLATED_128,
)


class OwnershipContractError(ValueError):
    """Raised when an ownership arm or its input violates the formal contract."""


def normalize_ownership(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in OWNERSHIP_MODES:
        raise OwnershipContractError(
            f"ownership must be one of {OWNERSHIP_MODES}, got {value!r}"
        )
    return normalized


def _positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OwnershipContractError(f"{name} must be a positive integer")
    return int(value)


def _positive_float(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise OwnershipContractError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise OwnershipContractError(f"{name} must be finite and positive")
    return result


class CandidateTower(nn.Module):
    """LayerNorm followed by a two-layer GELU MLP."""

    def __init__(self, input_dim: int, hidden_1: int, hidden_2: int) -> None:
        super().__init__()
        self.input_dim = _positive_int(input_dim, name="input_dim")
        self.hidden_1 = _positive_int(hidden_1, name="hidden_1")
        self.hidden_2 = _positive_int(hidden_2, name="hidden_2")
        self.norm = nn.LayerNorm(self.input_dim)
        self.linear_1 = nn.Linear(self.input_dim, self.hidden_1)
        self.linear_2 = nn.Linear(self.hidden_1, self.hidden_2)
        self.activation = nn.GELU()

    def forward(self, value: Tensor) -> Tensor:
        value = self.activation(self.linear_1(self.norm(value)))
        return self.activation(self.linear_2(value))

    @property
    def output_dim(self) -> int:
        return self.hidden_2

    def macs_per_query(self) -> int:
        return self.input_dim * self.hidden_1 + self.hidden_1 * self.hidden_2


@dataclass(frozen=True)
class OwnershipArchitectureReport:
    ownership: str
    trainable_parameters: int
    macs_per_query_both_outputs: int
    rank_representation_dim: int
    confidence_representation_dim: int
    shared_parameter_count: int
    rank_owned_parameter_count: int
    confidence_owned_parameter_count: int

    def as_dict(self) -> dict[str, int | str]:
        return {
            "ownership": self.ownership,
            "trainable_parameters": self.trainable_parameters,
            "macs_per_query_both_outputs": self.macs_per_query_both_outputs,
            "rank_representation_dim": self.rank_representation_dim,
            "confidence_representation_dim": self.confidence_representation_dim,
            "shared_parameter_count": self.shared_parameter_count,
            "rank_owned_parameter_count": self.rank_owned_parameter_count,
            "confidence_owned_parameter_count": self.confidence_owned_parameter_count,
        }


class MMGDinoE5ResponsibilityOwners(nn.Module):
    """Formal Shared-128, Shared-Wide, and Isolated ownership arms."""

    feature_dim = 256
    input_dim = 257

    def __init__(
        self,
        *,
        ownership: str,
        rank_residual_limit: float = 0.1,
    ) -> None:
        super().__init__()
        self.ownership = normalize_ownership(ownership)
        self.rank_residual_limit = _positive_float(
            rank_residual_limit, name="rank_residual_limit"
        )
        if self.ownership == OWNERSHIP_SHARED_128:
            self.shared_trunk = CandidateTower(self.input_dim, 128, 128)
            self.rank_trunk = None
            self.confidence_trunk = None
            rank_dim = confidence_dim = 128
        elif self.ownership == OWNERSHIP_SHARED_WIDE:
            self.shared_trunk = CandidateTower(self.input_dim, 212, 210)
            self.rank_trunk = None
            self.confidence_trunk = None
            rank_dim = confidence_dim = 210
        else:
            self.shared_trunk = None
            self.rank_trunk = CandidateTower(self.input_dim, 128, 128)
            self.confidence_trunk = CandidateTower(self.input_dim, 128, 128)
            rank_dim = confidence_dim = 128
        self.rank_head = nn.Linear(rank_dim, 1)
        self.confidence_head = nn.Linear(confidence_dim, 1)
        # All formal arms begin with exactly the native ranking and a neutral
        # absolute-confidence logit, independent of their tower dimensions.
        nn.init.zeros_(self.rank_head.weight)
        nn.init.zeros_(self.rank_head.bias)
        nn.init.zeros_(self.confidence_head.weight)
        nn.init.zeros_(self.confidence_head.bias)

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
            or tuple(query_features.shape[-1:]) != (self.feature_dim,)
            or int(query_features.shape[0]) <= 0
            or int(query_features.shape[1]) <= 0
        ):
            raise OwnershipContractError(
                "query_features must be nonempty floating (B,Q,256)"
            )
        if (
            not torch.is_tensor(native_score)
            or native_score.dim() != 2
            or not native_score.is_floating_point()
            or tuple(native_score.shape) != tuple(query_features.shape[:2])
        ):
            raise OwnershipContractError("native_score must be floating (B,Q)")
        if (
            not torch.is_tensor(candidate_mask)
            or candidate_mask.dtype != torch.bool
            or tuple(candidate_mask.shape) != tuple(native_score.shape)
        ):
            raise OwnershipContractError("candidate_mask must be boolean (B,Q)")
        if not (
            query_features.device == native_score.device == candidate_mask.device
        ):
            raise OwnershipContractError("all candidate inputs must share a device")
        reference = self.rank_head.weight
        if query_features.device != reference.device:
            raise OwnershipContractError("candidate inputs and owners differ in device")
        if not bool(candidate_mask.any(dim=1).all().item()):
            raise OwnershipContractError("every row must retain a candidate")
        if not bool(torch.isfinite(query_features).all().item()):
            raise OwnershipContractError("query_features must be finite")
        if not bool(torch.isfinite(native_score).all().item()):
            raise OwnershipContractError("native_score must be finite")
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
        if self.shared_trunk is not None:
            shared = self.shared_trunk(frozen_input)
            rank_hidden = confidence_hidden = shared
        else:
            assert self.rank_trunk is not None
            assert self.confidence_trunk is not None
            rank_hidden = self.rank_trunk(frozen_input)
            confidence_hidden = self.confidence_trunk(frozen_input)
        raw_residual = self.rank_head(rank_hidden).squeeze(-1)
        limit = self.rank_residual_limit
        rank_residual = limit * torch.tanh(raw_residual / limit)
        rank_residual = rank_residual.masked_fill(~mask, 0.0)
        pre_mask_rank = native + rank_residual
        rank_floor = torch.finfo(pre_mask_rank.dtype).min
        if bool((pre_mask_rank[mask] == rank_floor).any().item()):
            raise OwnershipContractError("valid rank score reached reserved floor")
        rank_score = pre_mask_rank.masked_fill(~mask, rank_floor)
        confidence_score = self.confidence_head(confidence_hidden).squeeze(-1)
        if not all(
            bool(torch.isfinite(value).all().item())
            for value in (rank_residual, rank_score, confidence_score)
        ):
            raise OwnershipContractError("owner output became non-finite")
        return {
            "native_score": native,
            "candidate_mask": mask,
            "rank_residual": rank_residual,
            "rank_score": rank_score,
            "confidence_score": confidence_score,
        }

    @staticmethod
    def _named(prefix: str, module: nn.Module) -> tuple[tuple[str, nn.Parameter], ...]:
        return tuple(
            (f"{prefix}.{name}", parameter)
            for name, parameter in module.named_parameters()
        )

    def named_task_parameters(
        self, task: str
    ) -> tuple[tuple[str, nn.Parameter], ...]:
        task = str(task).strip().lower()
        if task not in ("rank", "confidence"):
            raise OwnershipContractError("task must be rank or confidence")
        head = self.rank_head if task == "rank" else self.confidence_head
        if self.shared_trunk is not None:
            trunk_named = self._named("shared_trunk", self.shared_trunk)
        else:
            trunk = self.rank_trunk if task == "rank" else self.confidence_trunk
            assert trunk is not None
            trunk_named = self._named(f"{task}_trunk", trunk)
        return trunk_named + self._named(f"{task}_head", head)

    def task_parameters(self, task: str) -> tuple[nn.Parameter, ...]:
        return tuple(value for _, value in self.named_task_parameters(task))

    def shared_parameters(self) -> tuple[nn.Parameter, ...]:
        rank = {id(value): value for value in self.task_parameters("rank")}
        confidence = {
            id(value): value for value in self.task_parameters("confidence")
        }
        return tuple(rank[key] for key in sorted(rank.keys() & confidence.keys()))

    @staticmethod
    def _numel(values: Iterable[nn.Parameter]) -> int:
        return int(sum(parameter.numel() for parameter in values))

    def architecture_report(self) -> OwnershipArchitectureReport:
        all_parameters = tuple(self.parameters())
        rank_parameters = self.task_parameters("rank")
        confidence_parameters = self.task_parameters("confidence")
        shared_parameters = self.shared_parameters()
        if self.shared_trunk is not None:
            macs = (
                self.shared_trunk.macs_per_query()
                + self.rank_head.in_features
                + self.confidence_head.in_features
            )
            rank_dim = confidence_dim = self.shared_trunk.output_dim
        else:
            assert self.rank_trunk is not None
            assert self.confidence_trunk is not None
            macs = (
                self.rank_trunk.macs_per_query()
                + self.confidence_trunk.macs_per_query()
                + self.rank_head.in_features
                + self.confidence_head.in_features
            )
            rank_dim = self.rank_trunk.output_dim
            confidence_dim = self.confidence_trunk.output_dim
        report = OwnershipArchitectureReport(
            ownership=self.ownership,
            trainable_parameters=self._numel(all_parameters),
            macs_per_query_both_outputs=int(macs),
            rank_representation_dim=rank_dim,
            confidence_representation_dim=confidence_dim,
            shared_parameter_count=self._numel(shared_parameters),
            rank_owned_parameter_count=self._numel(rank_parameters),
            confidence_owned_parameter_count=self._numel(confidence_parameters),
        )
        expected = {
            OWNERSHIP_SHARED_128: (50308, 49536),
            OWNERSHIP_SHARED_WIDE: (100362, 99424),
            OWNERSHIP_ISOLATED_128: (100358, 98816),
        }[self.ownership]
        if (report.trainable_parameters, report.macs_per_query_both_outputs) != expected:
            raise RuntimeError(
                "formal ownership architecture drifted: "
                f"expected={expected}, actual="
                f"{(report.trainable_parameters, report.macs_per_query_both_outputs)}"
            )
        if self.ownership == OWNERSHIP_ISOLATED_128 and shared_parameters:
            raise RuntimeError("isolated owners unexpectedly share parameters")
        if self.ownership != OWNERSHIP_ISOLATED_128 and not shared_parameters:
            raise RuntimeError("shared arm lost its shared owner parameters")
        return report


def task_gradient_connection_report(
    module: MMGDinoE5ResponsibilityOwners,
    rank_loss: Tensor,
    confidence_loss: Tensor,
) -> dict[str, object]:
    """Audit which trainable tensors each task loss can reach."""
    if not isinstance(module, MMGDinoE5ResponsibilityOwners):
        raise TypeError("module must be MMGDinoE5ResponsibilityOwners")
    named = tuple(module.named_parameters())

    def connected(loss: Tensor) -> tuple[str, ...]:
        gradients = torch.autograd.grad(
            loss,
            tuple(value for _, value in named),
            allow_unused=True,
            retain_graph=True,
        )
        return tuple(
            name for (name, _), gradient in zip(named, gradients)
            if gradient is not None
        )

    rank_names = connected(rank_loss)
    confidence_names = connected(confidence_loss)
    intersection = tuple(sorted(set(rank_names) & set(confidence_names)))
    expected_shared = module.ownership != OWNERSHIP_ISOLATED_128
    if bool(intersection) != expected_shared:
        raise RuntimeError(
            "task-gradient topology drifted: "
            f"ownership={module.ownership}, intersection={intersection}"
        )
    return {
        "ownership": module.ownership,
        "rank_connected_parameter_names": rank_names,
        "confidence_connected_parameter_names": confidence_names,
        "cross_task_parameter_names": intersection,
        "structurally_isolated": not intersection,
    }


__all__ = [
    "CandidateTower",
    "MMGDinoE5ResponsibilityOwners",
    "OWNERSHIP_ISOLATED_128",
    "OWNERSHIP_MODES",
    "OWNERSHIP_SHARED_128",
    "OWNERSHIP_SHARED_WIDE",
    "OwnershipArchitectureReport",
    "OwnershipContractError",
    "normalize_ownership",
    "task_gradient_connection_report",
]
