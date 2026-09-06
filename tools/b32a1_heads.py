"""Frozen-positive-trunk abstention heads for the B32A1 FineCops experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import Tensor, nn

from tools.mmgdino_e5_ownership import (
    CandidateTower,
    MMGDinoE5ResponsibilityOwners,
    OWNERSHIP_ISOLATED_128,
    OWNERSHIP_SHARED_WIDE,
)


FACTORIZED_C_ONLY = "factorized_c_only"
COUPLED_SCALAR = "coupled_scalar"
SHARED_WIDE = OWNERSHIP_SHARED_WIDE
ISOLATED = OWNERSHIP_ISOLATED_128
B32A1_HEAD_MODES = (FACTORIZED_C_ONLY, COUPLED_SCALAR, SHARED_WIDE, ISOLATED)


class B32A1HeadContractError(ValueError):
    """Raised when an arm violates the frozen-candidate head contract."""


@dataclass(frozen=True)
class B32A1ArchitectureReport:
    mode: str
    trainable_parameters: int
    macs_per_query_both_duties: int
    rank_parameters: int
    confidence_parameters: int
    shared_parameters: int
    same_deployed_scalar: bool

    def as_dict(self) -> dict[str, int | str | bool]:
        return {
            "mode": self.mode,
            "trainable_parameters": self.trainable_parameters,
            "macs_per_query_both_duties": self.macs_per_query_both_duties,
            "rank_parameters": self.rank_parameters,
            "confidence_parameters": self.confidence_parameters,
            "shared_parameters": self.shared_parameters,
            "same_deployed_scalar": self.same_deployed_scalar,
        }


class B32A1AbstentionHead(nn.Module):
    """Coupled-scalar and capacity-matched factorized controls.

    Coupled Scalar has one bounded residual score per query.  The same
    ``native + residual`` scalar selects the box and supplies sample confidence
    through its maximum, matching the repository's established O0 meaning.
    Shared-Wide and Isolated delegate to the audited 100k-owner implementation.
    Every path detaches decoder features and native scores at its boundary.
    """

    feature_dim = 256
    input_dim = 257

    def __init__(self, *, mode: str, rank_residual_limit: float = 0.1) -> None:
        super().__init__()
        mode = str(mode).strip().lower()
        if mode not in B32A1_HEAD_MODES:
            raise B32A1HeadContractError(
                f"mode must be one of {B32A1_HEAD_MODES}, got {mode!r}"
            )
        if not isinstance(rank_residual_limit, (int, float)) or float(
            rank_residual_limit
        ) <= 0.0:
            raise B32A1HeadContractError("rank_residual_limit must be positive")
        self.mode = mode
        self.rank_residual_limit = float(rank_residual_limit)
        if mode in (FACTORIZED_C_ONLY, COUPLED_SCALAR):
            hidden1, hidden2 = (
                (128, 128) if mode == FACTORIZED_C_ONLY else (212, 210)
            )
            self.coupled_trunk: CandidateTower | None = CandidateTower(
                self.input_dim, hidden1, hidden2
            )
            self.coupled_output: nn.Linear | None = nn.Linear(hidden2, 1)
            nn.init.zeros_(self.coupled_output.weight)
            nn.init.zeros_(self.coupled_output.bias)
            self.factorized: MMGDinoE5ResponsibilityOwners | None = None
        else:
            self.coupled_trunk = None
            self.coupled_output = None
            self.factorized = MMGDinoE5ResponsibilityOwners(
                ownership=mode, rank_residual_limit=self.rank_residual_limit
            )

    def _coupled_forward(
        self, query_features: Tensor, native_score: Tensor, candidate_mask: Tensor
    ) -> dict[str, Tensor]:
        assert self.coupled_trunk is not None and self.coupled_output is not None
        reference = self.coupled_output.weight
        if (
            not torch.is_tensor(query_features)
            or query_features.dim() != 3
            or not query_features.is_floating_point()
            or tuple(query_features.shape[-1:]) != (self.feature_dim,)
            or int(query_features.shape[0]) <= 0
            or int(query_features.shape[1]) <= 0
        ):
            raise B32A1HeadContractError(
                "query_features must be nonempty floating (B,Q,256)"
            )
        if (
            not torch.is_tensor(native_score)
            or not native_score.is_floating_point()
            or tuple(native_score.shape) != tuple(query_features.shape[:2])
        ):
            raise B32A1HeadContractError("native_score must be floating (B,Q)")
        if (
            not torch.is_tensor(candidate_mask)
            or candidate_mask.dtype != torch.bool
            or tuple(candidate_mask.shape) != tuple(native_score.shape)
        ):
            raise B32A1HeadContractError("candidate_mask must be boolean (B,Q)")
        if not (
            query_features.device
            == native_score.device
            == candidate_mask.device
            == reference.device
        ):
            raise B32A1HeadContractError("all head tensors must share a device")
        if not bool(candidate_mask.any(dim=1).all().item()):
            raise B32A1HeadContractError("every row must retain a candidate")
        if not bool(torch.isfinite(query_features).all().item()) or not bool(
            torch.isfinite(native_score).all().item()
        ):
            raise B32A1HeadContractError("head inputs must be finite")
        features = query_features.detach().to(dtype=reference.dtype)
        native = native_score.detach().to(dtype=reference.dtype)
        hidden = self.coupled_trunk(
            torch.cat((features, native.unsqueeze(-1)), dim=-1)
        )
        raw_residual = self.coupled_output(hidden).squeeze(-1)
        floor = torch.finfo(native.dtype).min
        if self.mode == FACTORIZED_C_ONLY:
            residual = torch.zeros_like(native)
            rank_score = native.masked_fill(~candidate_mask, floor)
            confidence_score = (native + raw_residual).masked_fill(
                ~candidate_mask, floor
            )
        else:
            limit = self.rank_residual_limit
            residual = limit * torch.tanh(raw_residual / limit)
            residual = residual.masked_fill(~candidate_mask, 0.0)
            score = native + residual
            rank_score = score.masked_fill(~candidate_mask, floor)
            confidence_score = rank_score
        if not bool(torch.isfinite(residual).all().item()) or not bool(
            torch.isfinite(rank_score).all().item()
        ):
            raise B32A1HeadContractError("coupled scalar became non-finite")
        return {
            "native_score": native,
            "candidate_mask": candidate_mask,
            "rank_residual": residual,
            "rank_score": rank_score,
            "confidence_score": confidence_score,
        }

    def forward(
        self, query_features: Tensor, native_score: Tensor, candidate_mask: Tensor
    ) -> dict[str, Tensor]:
        if self.factorized is not None:
            return self.factorized(query_features, native_score, candidate_mask)
        return self._coupled_forward(query_features, native_score, candidate_mask)

    def named_task_parameters(self, task: str) -> tuple[tuple[str, nn.Parameter], ...]:
        task = str(task).strip().lower()
        if task not in {"rank", "confidence"}:
            raise B32A1HeadContractError("task must be rank or confidence")
        if self.factorized is not None:
            return tuple(
                (f"factorized.{name}", value)
                for name, value in self.factorized.named_task_parameters(task)
            )
        if self.mode == FACTORIZED_C_ONLY and task == "rank":
            return ()
        assert self.coupled_trunk is not None and self.coupled_output is not None
        return tuple(
            (f"coupled_trunk.{name}", value)
            for name, value in self.coupled_trunk.named_parameters()
        ) + tuple(
            (f"coupled_output.{name}", value)
            for name, value in self.coupled_output.named_parameters()
        )

    def task_parameters(self, task: str) -> tuple[nn.Parameter, ...]:
        return tuple(value for _, value in self.named_task_parameters(task))

    @staticmethod
    def _numel(values: Iterable[nn.Parameter]) -> int:
        identities: set[int] = set()
        total = 0
        for value in values:
            if id(value) not in identities:
                identities.add(id(value))
                total += value.numel()
        return int(total)

    def architecture_report(self) -> B32A1ArchitectureReport:
        rank = self.task_parameters("rank")
        confidence = self.task_parameters("confidence")
        shared_ids = {id(value) for value in rank} & {
            id(value) for value in confidence
        }
        shared = tuple(value for value in rank if id(value) in shared_ids)
        if self.mode == FACTORIZED_C_ONLY:
            assert self.coupled_trunk is not None and self.coupled_output is not None
            macs = self.coupled_trunk.macs_per_query() + 128
            expected = (50_179, 49_408)
        elif self.mode == COUPLED_SCALAR:
            assert self.coupled_trunk is not None and self.coupled_output is not None
            macs = self.coupled_trunk.macs_per_query() + 210
            expected = (100_151, 99_214)
        else:
            assert self.factorized is not None
            delegated = self.factorized.architecture_report()
            macs = delegated.macs_per_query_both_outputs
            expected = {
                SHARED_WIDE: (100_362, 99_424),
                ISOLATED: (100_358, 98_816),
            }[self.mode]
        report = B32A1ArchitectureReport(
            mode=self.mode,
            trainable_parameters=self._numel(self.parameters()),
            macs_per_query_both_duties=int(macs),
            rank_parameters=self._numel(rank),
            confidence_parameters=self._numel(confidence),
            shared_parameters=self._numel(shared),
            same_deployed_scalar=self.mode == COUPLED_SCALAR,
        )
        if (report.trainable_parameters, report.macs_per_query_both_duties) != expected:
            raise RuntimeError(
                f"B32A1 architecture drifted for {self.mode}: "
                f"{(report.trainable_parameters, report.macs_per_query_both_duties)} "
                f"!= {expected}"
            )
        if self.mode in (FACTORIZED_C_ONLY, ISOLATED) and report.shared_parameters:
            raise RuntimeError(f"{self.mode} unexpectedly shares task parameters")
        if self.mode in (COUPLED_SCALAR, SHARED_WIDE) and not report.shared_parameters:
            raise RuntimeError(f"{self.mode} unexpectedly lost task sharing")
        return report


def gradient_topology_report(
    module: B32A1AbstentionHead, rank_loss: Tensor, confidence_loss: Tensor
) -> dict[str, object]:
    """Report and enforce the structural task-gradient topology."""
    named = tuple(module.named_parameters())

    def connected(loss: Tensor) -> tuple[str, ...]:
        if not loss.requires_grad:
            return ()
        gradients = torch.autograd.grad(
            loss,
            tuple(value for _, value in named),
            allow_unused=True,
            retain_graph=True,
        )
        return tuple(
            name
            for (name, _), gradient in zip(named, gradients)
            if gradient is not None
        )

    rank_names = connected(rank_loss)
    confidence_names = connected(confidence_loss)
    cross = tuple(sorted(set(rank_names) & set(confidence_names)))
    expected_cross = module.mode in (COUPLED_SCALAR, SHARED_WIDE)
    if bool(cross) != expected_cross:
        raise RuntimeError(
            f"B32A1 gradient topology drifted for {module.mode}: {cross}"
        )
    return {
        "mode": module.mode,
        "rank_connected_parameter_names": rank_names,
        "confidence_connected_parameter_names": confidence_names,
        "cross_task_parameter_names": cross,
        "structurally_isolated": not cross,
    }


__all__ = [
    "B32A1AbstentionHead",
    "B32A1ArchitectureReport",
    "B32A1HeadContractError",
    "B32A1_HEAD_MODES",
    "COUPLED_SCALAR",
    "FACTORIZED_C_ONLY",
    "ISOLATED",
    "SHARED_WIDE",
    "gradient_topology_report",
]
