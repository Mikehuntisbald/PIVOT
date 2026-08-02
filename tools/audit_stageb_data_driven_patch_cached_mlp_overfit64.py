#!/usr/bin/env python3
"""Reproduce v19 and isolate scorer/objective changes on native O64 features."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from engine import _build_stage_b_data_driven_assignment_captions  # noqa: E402
from main import _make_grad_scaler  # noqa: E402
from models.GroundingDINO.stage_b_data_driven_patch_residual import (  # noqa: E402
    StageBDataDrivenPatchResidualMatcher,
)
from models.GroundingDINO.stage_b_data_driven_score import (  # noqa: E402
    DATA_DRIVEN_PATCH_DROP_ANCHOR_INSTANCE_BALANCED,
)
from tools.audit_stageb_data_driven_patch_free_table_overfit64 import (  # noqa: E402
    _complete_patch_checks,
)
from tools.audit_stageb_data_driven_patch_oracle_overfit64 import (  # noqa: E402
    _metrics,
    _patch_contract,
    _strict_witness_checks,
)
from tools.audit_stageb_data_driven_role_routed_coverage import (  # noqa: E402
    _build_runtime,
)
from tools.audit_stageb_data_driven_role_routed_real_model import (  # noqa: E402
    AUDIT_VARIANTS,
    EXPECTED_DATASET_CONFIG_SHA256,
    EXPECTED_QUERY_COUNT,
    RealModelAuditError,
    _move_criterion_target,
    _seed_everything,
    _sha256,
    _write_json_exclusive,
)
from tools.run_stageb_data_driven_role_routed_overfit64 import (  # noqa: E402
    _canonical_sha256,
    _select_rows,
    _tensor_sha256,
)
from util.misc import nested_tensor_from_tensor_list  # noqa: E402
from util.slconfig import SLConfig  # noqa: E402


REFERENCE_V19_RECEIPT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/data_driven_role_routed_20260727/"
    "overfit64_patch_residual128_raw_centered_v19_balanced_anchor_"
    "ranklr3e4_patchlr3e4_u100_seed42_v1/receipt.json"
)


class _CachedBaseGapResidualMatcher(nn.Module):
    """O64-only prototype that gives the MLP frozen base row context."""

    def __init__(
        self,
        *,
        feature_dim: int,
        hidden_dim: int,
        residual_limit: float,
        init_seed: int,
        context_mode: str,
    ) -> None:
        super().__init__()
        if context_mode not in {"base_gap_v1", "zero_context_control_v1"}:
            raise ValueError("cached base-gap context mode is invalid")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.residual_limit = float(residual_limit)
        self.init_seed = int(init_seed)
        self.context_mode = context_mode
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.init_seed)
            self.input = nn.Linear(4 * self.feature_dim + 2, self.hidden_dim)
            self.output = nn.Linear(self.hidden_dim, 1, bias=False)
        nn.init.zeros_(self.output.weight)

    def load_legacy_state(self, state: Mapping[str, torch.Tensor]) -> None:
        expected_prefix = 4 * self.feature_dim
        with torch.no_grad():
            self.input.weight[:, :expected_prefix].copy_(
                state["input.weight"]
            )
            self.input.bias.copy_(state["input.bias"])
            self.output.weight.copy_(state["output.weight"])

    def trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        parameters = tuple(self.parameters())
        if len(parameters) != 3:
            raise RuntimeError("cached base-gap matcher tensor set drifted")
        return parameters

    def forward(
        self,
        query: torch.Tensor,
        patch: torch.Tensor,
        base_score: torch.Tensor,
    ) -> torch.Tensor:
        if (
            query.dim() != 3
            or patch.dim() != 2
            or tuple(query.shape[:2]) != tuple(base_score.shape)
            or int(query.shape[0]) != int(patch.shape[0])
            or int(query.shape[-1]) != self.feature_dim
            or int(patch.shape[-1]) != self.feature_dim
        ):
            raise ValueError("cached base-gap inputs are misaligned")
        normalized_query = F.normalize(query.detach(), dim=-1)
        normalized_patch = F.normalize(patch.detach(), dim=-1)
        expanded_patch = normalized_patch[:, None, :].expand_as(
            normalized_query
        )
        pair_features = torch.cat(
            (
                normalized_query,
                expanded_patch,
                normalized_query * expanded_patch,
                (normalized_query - expanded_patch).abs(),
            ),
            dim=-1,
        )
        base = base_score.detach().float()
        centered = base - base.mean(dim=1, keepdim=True)
        std = centered.square().mean(dim=1, keepdim=True).clamp_min(1e-6).sqrt()
        standardized = (centered / std).clamp(-5.0, 5.0)
        gap = standardized.amax(dim=1, keepdim=True) - standardized
        # Keep the two bounded Gate3 coordinates on the same numeric scale as
        # normalized q/p features so the sealed AMP scale remains usable.
        context = torch.stack((standardized / 5.0, gap / 10.0), dim=-1)
        if self.context_mode == "zero_context_control_v1":
            context = torch.zeros_like(context)
        features = torch.cat((pair_features, context), dim=-1)
        raw = self.output(F.gelu(self.input(features))).squeeze(-1)
        raw = raw - raw.mean(dim=1, keepdim=True)
        limit = self.residual_limit
        return limit * torch.tanh(raw / limit)

    def architecture(self) -> dict[str, Any]:
        return {
            "contract": "cached_qp_mlp128_base_gap_context_tanh025_v1",
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "residual_limit": self.residual_limit,
            "init_seed": self.init_seed,
            "context_mode": self.context_mode,
            "context_features": (
                "base_standardized_div5",
                "base_gap_to_row_max_div10",
            ),
            "trainable_tensors": 3,
            "trainable_parameters": sum(
                parameter.numel() for parameter in self.parameters()
            ),
        }


class _CachedTopKSemanticContextResidualMatcher(nn.Module):
    """O64-only top-k DeepSets prototype over frozen patch-score rows."""

    def __init__(
        self,
        *,
        feature_dim: int,
        hidden_dim: int,
        context_dim: int,
        topk: int,
        residual_limit: float,
        init_seed: int,
        context_mode: str = "topk_semantic_v1",
    ) -> None:
        super().__init__()
        if (
            int(feature_dim) <= 0
            or int(hidden_dim) <= 0
            or int(context_dim) <= 0
            or int(topk) <= 0
            or not math.isfinite(float(residual_limit))
            or float(residual_limit) <= 0.0
            or context_mode not in {
                "topk_semantic_v1",
                "local_capacity_control_v1",
            }
        ):
            raise ValueError("cached top-k semantic context architecture is invalid")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.context_dim = int(context_dim)
        self.topk = int(topk)
        self.residual_limit = float(residual_limit)
        self.init_seed = int(init_seed)
        self.context_mode = str(context_mode)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.init_seed)
            self.input = nn.Linear(4 * self.feature_dim, self.hidden_dim)
            self.output = nn.Linear(self.hidden_dim, 1, bias=False)
            self.context_input = nn.Linear(
                2 * self.hidden_dim + 2, self.context_dim
            )
            self.context_output = nn.Linear(self.context_dim, 1, bias=False)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.context_output.weight)

    def load_legacy_state(self, state: Mapping[str, torch.Tensor]) -> None:
        expected = {
            "input.bias": self.input.bias,
            "input.weight": self.input.weight,
            "output.weight": self.output.weight,
        }
        if set(state) != set(expected):
            raise ValueError("cached top-k legacy state key set drifted")
        with torch.no_grad():
            for key, destination in expected.items():
                source = state[key]
                if (
                    not torch.is_tensor(source)
                    or source.dtype != destination.dtype
                    or tuple(source.shape) != tuple(destination.shape)
                ):
                    raise ValueError(
                        f"cached top-k legacy tensor drifted at {key}"
                    )
                destination.copy_(source)
        if not all(
            torch.equal(expected[key].detach(), state[key].detach())
            for key in expected
        ):
            raise RuntimeError("cached top-k legacy state migration was not exact")

    def trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        parameters = tuple(self.parameters())
        if len(parameters) != 6:
            raise RuntimeError("cached top-k matcher tensor set drifted")
        return parameters

    def _semantic_features(
        self,
        query: torch.Tensor,
        patch: torch.Tensor,
        base_score: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            query.dim() != 3
            or patch.dim() != 2
            or tuple(query.shape[:2]) != tuple(base_score.shape)
            or int(query.shape[0]) != int(patch.shape[0])
            or int(query.shape[-1]) != self.feature_dim
            or int(patch.shape[-1]) != self.feature_dim
            or int(query.shape[1]) <= 0
        ):
            raise ValueError("cached top-k semantic inputs are misaligned")
        normalized_query = F.normalize(query.detach(), dim=-1)
        normalized_patch = F.normalize(patch.detach(), dim=-1)
        expanded_patch = normalized_patch[:, None, :].expand_as(
            normalized_query
        )
        pair_features = torch.cat(
            (
                normalized_query,
                expanded_patch,
                normalized_query * expanded_patch,
                (normalized_query - expanded_patch).abs(),
            ),
            dim=-1,
        )
        hidden = F.gelu(self.input(pair_features))

        # Base-score statistics and the discrete top-k selection are inference
        # inputs only.  Keeping them in FP32 avoids the unscaled AMP failure of
        # the earlier BaseGap prototype without opening a gradient path upstream.
        base = base_score.detach().float()
        centered = base - base.mean(dim=1, keepdim=True)
        std = centered.square().mean(dim=1, keepdim=True).clamp_min(1e-6).sqrt()
        standardized = (centered / std).clamp(-5.0, 5.0)
        gap = standardized.amax(dim=1, keepdim=True) - standardized
        top_count = min(self.topk, int(base.shape[1]))
        top_indices = torch.topk(base, k=top_count, dim=1).indices
        gather_index = top_indices[:, :, None].expand(
            -1, -1, self.hidden_dim
        )
        top_hidden = torch.gather(hidden, dim=1, index=gather_index)
        prototype = top_hidden.float().mean(dim=1).to(dtype=hidden.dtype)
        expanded_prototype = prototype[:, None, :].expand_as(hidden)
        if self.context_mode == "local_capacity_control_v1":
            expanded_prototype = torch.zeros_like(expanded_prototype)
            standardized = torch.zeros_like(standardized)
            gap = torch.zeros_like(gap)
        scaled_score_context = torch.stack(
            (standardized / 5.0, gap / 10.0), dim=-1
        ).to(dtype=hidden.dtype)
        context = torch.cat(
            (
                hidden * expanded_prototype,
                (hidden - expanded_prototype).abs(),
                scaled_score_context,
            ),
            dim=-1,
        )
        context_hidden = F.gelu(self.context_input(context))
        return hidden, context_hidden

    def forward(
        self,
        query: torch.Tensor,
        patch: torch.Tensor,
        base_score: torch.Tensor,
    ) -> torch.Tensor:
        hidden, context_hidden = self._semantic_features(
            query, patch, base_score
        )
        raw = self.output(hidden).squeeze(-1)
        raw = raw + self.context_output(context_hidden).squeeze(-1)
        raw = raw - raw.mean(dim=1, keepdim=True)
        limit = self.residual_limit
        return limit * torch.tanh(raw / limit)

    def architecture(self) -> dict[str, Any]:
        return {
            "contract": "cached_qp_mlp128_topk10_semantic_context16_tanh025_v1",
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "context_dim": self.context_dim,
            "topk": self.topk,
            "residual_limit": self.residual_limit,
            "init_seed": self.init_seed,
            "context_mode": self.context_mode,
            "row_pool": "mean_of_frozen_base_score_topk_hidden_v1",
            "context_features": (
                "local_hidden_times_topk_hidden_mean",
                "absolute_local_hidden_minus_topk_hidden_mean",
                "base_standardized_div5",
                "base_gap_to_row_max_div10",
            ),
            "upstream_gradient_policy": "query_patch_and_base_detached",
            "trainable_tensors": 6,
            "trainable_parameters": sum(
                parameter.numel() for parameter in self.parameters()
            ),
        }


REGIMES: dict[str, dict[str, Any]] = {
    "mlp128_formal_amp_adamw_lr3e4_clip01_u100": {
        "hidden_dim": 128,
        "optimizer": "adamw",
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "clip_max_norm": 0.1,
        "steps": 100,
        "log_updates": (1, 10, 25, 50, 75, 100),
    },
    "mlp128_formal_amp_adamw_lr3e4_clip01_u300": {
        "hidden_dim": 128,
        "optimizer": "adamw",
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "clip_max_norm": 0.1,
        "steps": 300,
        "log_updates": (1, 10, 50, 100, 150, 200, 250, 300),
    },
    "mlp256_formal_amp_adamw_lr3e4_clip01_u300": {
        "hidden_dim": 256,
        "optimizer": "adamw",
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "clip_max_norm": 0.1,
        "steps": 300,
        "log_updates": (1, 10, 50, 100, 150, 200, 250, 300),
    },
    "mlp512_formal_amp_adamw_lr3e4_clip01_u300": {
        "hidden_dim": 512,
        "optimizer": "adamw",
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "clip_max_norm": 0.1,
        "steps": 300,
        "log_updates": (1, 10, 50, 100, 150, 200, 250, 300),
    },
    "mlp128_amp_adam_lr3e3_noclip_u300": {
        "hidden_dim": 128,
        "optimizer": "adam",
        "lr": 3e-3,
        "weight_decay": 0.0,
        "clip_max_norm": None,
        "steps": 300,
        "log_updates": (1, 10, 25, 50, 75, 100, 150, 200, 250, 300),
    },
    "mlp128_densetail1_formal_amp_adamw_lr3e4_clip01_u300": {
        "hidden_dim": 128,
        "optimizer": "adamw",
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "clip_max_norm": 0.1,
        "drop_dense_tail_weight": 1.0,
        "steps": 300,
        "log_updates": (1, 10, 50, 100, 150, 200, 250, 300),
    },
    "mlp128_densetail1_amp_adam_lr3e3_noclip_u300": {
        "hidden_dim": 128,
        "optimizer": "adam",
        "lr": 3e-3,
        "weight_decay": 0.0,
        "clip_max_norm": None,
        "drop_dense_tail_weight": 1.0,
        "steps": 300,
        "log_updates": (1, 10, 25, 50, 75, 100, 150, 200, 250, 300),
    },
    "topksemantic128_ctx16_k10_seed42_formal_amp_adamw_lr3e4_clip01_u300": {
        "hidden_dim": 128,
        "scorer_architecture": "topk_semantic_context_v1",
        "context_dim": 16,
        "context_topk": 10,
        "context_init_seed": 42,
        "optimizer": "adamw",
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "clip_max_norm": 0.1,
        "drop_dense_tail_weight": 0.0,
        "steps": 300,
        "log_updates": (1, 10, 50, 100, 150, 200, 250, 300),
    },
    "topksemantic128_ctx16_k10_seed42_formal_amp_adamw_lr3e4_clip01_u500": {
        "hidden_dim": 128,
        "scorer_architecture": "topk_semantic_context_v1",
        "context_dim": 16,
        "context_topk": 10,
        "context_init_seed": 42,
        "optimizer": "adamw",
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "clip_max_norm": 0.1,
        "drop_dense_tail_weight": 0.0,
        "steps": 500,
        "log_updates": (
            1, 10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500
        ),
    },
    "topklocal128_ctx16_k10_seed42_formal_amp_adamw_lr3e4_clip01_u300": {
        "hidden_dim": 128,
        "scorer_architecture": "topk_semantic_context_v1",
        "semantic_context_mode": "local_capacity_control_v1",
        "context_dim": 16,
        "context_topk": 10,
        "context_init_seed": 42,
        "optimizer": "adamw",
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "clip_max_norm": 0.1,
        "drop_dense_tail_weight": 0.0,
        "steps": 300,
        "log_updates": (1, 10, 50, 100, 150, 200, 250, 300),
    },
    "mlp128_amp_adam_lr1e2_noclip_u500": {
        "hidden_dim": 128,
        "optimizer": "adam",
        "lr": 1e-2,
        "weight_decay": 0.0,
        "clip_max_norm": None,
        "steps": 500,
        "log_updates": (1, 10, 25, 50, 100, 150, 200, 300, 400, 500),
    },
    "mlp128_amp_adam_lr3e3_noclip_u1000": {
        "hidden_dim": 128,
        "optimizer": "adam",
        "lr": 3e-3,
        "weight_decay": 0.0,
        "clip_max_norm": None,
        "steps": 1000,
        "log_updates": (1, 10, 50, 100, 200, 300, 500, 750, 1000),
    },
    "basegap128_seed42_formal_amp_adamw_lr3e4_clip01_u300": {
        "hidden_dim": 128,
        "context_mode": "base_gap_v1",
        "context_init_seed": 42,
        "optimizer": "adamw",
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "clip_max_norm": 0.1,
        "steps": 300,
        "log_updates": (1, 10, 50, 100, 150, 200, 250, 300),
    },
    "zerocontext128_seed42_formal_amp_adamw_lr3e4_clip01_u300": {
        "hidden_dim": 128,
        "context_mode": "zero_context_control_v1",
        "context_init_seed": 42,
        "optimizer": "adamw",
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "clip_max_norm": 0.1,
        "steps": 300,
        "log_updates": (1, 10, 50, 100, 150, 200, 250, 300),
    },
    "zerocontext128_seed42_formal_amp_adamw_lr3e4_clip01_u100": {
        "hidden_dim": 128,
        "context_mode": "zero_context_control_v1",
        "context_init_seed": 42,
        "optimizer": "adamw",
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "clip_max_norm": 0.1,
        "steps": 100,
        "log_updates": (1, 10, 25, 50, 75, 100),
    },
    "basegap128_seed42_amp_adam_lr3e3_noclip_u300": {
        "hidden_dim": 128,
        "context_mode": "base_gap_v1",
        "context_init_seed": 42,
        "optimizer": "adam",
        "lr": 3e-3,
        "weight_decay": 0.0,
        "clip_max_norm": None,
        "steps": 300,
        "log_updates": (1, 10, 25, 50, 75, 100, 150, 200, 250, 300),
    },
}


def _grad_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    gradients = [
        parameter.grad.detach().float()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not gradients:
        raise RealModelAuditError("cached MLP produced no gradients")
    return float(
        torch.linalg.vector_norm(
            torch.stack(
                [torch.linalg.vector_norm(gradient) for gradient in gradients]
            )
        ).item()
    )


def _gradient_abs_sum(parameters: Sequence[torch.nn.Parameter]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float()
        if not bool(torch.isfinite(gradient).all().item()):
            raise RealModelAuditError("cached MLP produced a non-finite gradient")
        total += float(gradient.abs().sum().item())
    return total


def _residual_metrics(
    residual: torch.Tensor,
    contract: Mapping[str, torch.Tensor],
    limit: float,
) -> dict[str, float]:
    detached = residual.detach().float()
    normalized = (detached / float(limit)).clamp(-0.999999, 0.999999)
    centered_raw = float(limit) * torch.atanh(normalized)
    positive = contract["category_positive_mask"]
    negative = contract["category_negative_mask"]
    neutral = contract["category_neutral_mask"]
    return {
        "abs_max": float(detached.abs().max().item()),
        "abs_mean": float(detached.abs().mean().item()),
        "centered_raw_rms": float(centered_raw.square().mean().sqrt().item()),
        "centered_raw_abs_max": float(centered_raw.abs().max().item()),
        "centered_raw_row_mean_abs_max": float(
            centered_raw.mean(dim=1).abs().max().item()
        ),
        "positive_mean": float(detached[positive].mean().item()),
        "negative_mean": float(detached[negative].mean().item()),
        "neutral_mean": float(detached[neutral].mean().item()),
        "positive_minus_negative_mean": float(
            detached[positive].mean().item()
            - detached[negative].mean().item()
        ),
        "saturation_fraction": float(
            (detached.abs() >= 0.95 * float(limit)).float().mean().item()
        ),
    }


def _forward_contract(
    matcher: nn.Module,
    query: torch.Tensor,
    patch: torch.Tensor,
    base: torch.Tensor,
    scale: torch.Tensor,
    boxes: torch.Tensor,
    targets: Sequence[Mapping[str, Any]],
    candidate: torch.Tensor,
    cfg: Any,
) -> tuple[torch.Tensor, torch.Tensor, Mapping[str, torch.Tensor]]:
    if isinstance(
        matcher,
        (
            _CachedBaseGapResidualMatcher,
            _CachedTopKSemanticContextResidualMatcher,
        ),
    ):
        residual = matcher(query, patch, base)
    else:
        residual = matcher(query, patch)
    score = base.detach() + scale.detach() * residual
    contract = _patch_contract(score, boxes, targets, candidate, cfg)
    return residual, score, contract


def _audit_topk_semantic_u0(
    matcher: _CachedTopKSemanticContextResidualMatcher,
    *,
    initial_state: Mapping[str, torch.Tensor],
    query: torch.Tensor,
    patch: torch.Tensor,
    base: torch.Tensor,
    scale: torch.Tensor,
) -> dict[str, Any]:
    legacy_parameters = {
        "input.bias": matcher.input.bias,
        "input.weight": matcher.input.weight,
        "output.weight": matcher.output.weight,
    }
    legacy_state_is_exact = all(
        torch.equal(parameter.detach(), initial_state[key].detach())
        for key, parameter in legacy_parameters.items()
    )
    context_output_is_exact_zero = bool(
        (matcher.context_output.weight.detach() == 0).all().item()
    )
    trainable_parameters = matcher.trainable_parameters()
    trainable_parameter_count = sum(
        parameter.numel() for parameter in trainable_parameters
    )
    permutation = torch.arange(
        int(query.shape[1]) - 1,
        -1,
        -1,
        device=query.device,
        dtype=torch.int64,
    )
    inverse = torch.argsort(permutation)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        residual = matcher(query, patch, base)
        score = base.detach() + scale.detach() * residual
        hidden, context_hidden = matcher._semantic_features(query, patch, base)
        permuted_hidden, permuted_context_hidden = matcher._semantic_features(
            query[:, permutation], patch, base[:, permutation]
        )
    hidden_permutation_max_abs = float(
        (
            hidden.float()
            - permuted_hidden[:, inverse].float()
        )
        .abs()
        .max()
        .item()
    )
    context_permutation_max_abs = float(
        (
            context_hidden.float()
            - permuted_context_hidden[:, inverse].float()
        )
        .abs()
        .max()
        .item()
    )
    permutation_tolerance = 1e-3
    checks = {
        "legacy_three_tensor_migration_is_exact": legacy_state_is_exact,
        "context_output_is_exactly_zero_initialized": (
            context_output_is_exact_zero
        ),
        "matcher_exposes_exactly_six_trainable_tensors": (
            len(trainable_parameters) == 6
        ),
        "matcher_has_expected_135488_trainable_parameters": (
            trainable_parameter_count == 135488
        ),
        "u0_residual_is_exactly_zero": bool((residual == 0).all().item()),
        "u0_score_is_bitwise_base_score": (
            tuple(score.shape) == tuple(base.shape)
            and score.dtype == base.dtype
            and torch.equal(score, base.detach())
        ),
        "local_hidden_is_query_permutation_equivariant": (
            hidden_permutation_max_abs <= permutation_tolerance
        ),
        "semantic_context_is_query_permutation_equivariant": (
            context_permutation_max_abs <= permutation_tolerance
        ),
    }
    if not all(checks.values()):
        raise RealModelAuditError(
            "cached top-k semantic U0 or permutation contract drifted"
        )
    return {
        "status": "passed",
        "checks": checks,
        "permutation_tolerance": permutation_tolerance,
        "trainable_tensors": len(trainable_parameters),
        "trainable_parameters": trainable_parameter_count,
        "local_hidden_permutation_max_abs_difference": (
            hidden_permutation_max_abs
        ),
        "semantic_context_permutation_max_abs_difference": (
            context_permutation_max_abs
        ),
    }


def _evaluate(
    matcher: nn.Module,
    *,
    query: torch.Tensor,
    patch: torch.Tensor,
    base: torch.Tensor,
    scale: torch.Tensor,
    boxes: torch.Tensor,
    targets: Sequence[Mapping[str, Any]],
    candidate: torch.Tensor,
    cfg: Any,
    baseline: Mapping[str, float],
) -> dict[str, Any]:
    was_training = matcher.training
    matcher.eval()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        residual, score, contract = _forward_contract(
            matcher,
            query,
            patch,
            base,
            scale,
            boxes,
            targets,
            candidate,
            cfg,
        )
    metrics, checks, derived = _complete_patch_checks(
        contract,
        score,
        boxes=boxes,
        targets=targets,
        candidate=candidate,
        cfg=cfg,
        baseline=baseline,
    )
    strict_checks = _strict_witness_checks(metrics)
    result = {
        "gate_status": "passed" if all(checks.values()) else "failed",
        "strict_witness_status": (
            "passed" if all(strict_checks.values()) else "failed"
        ),
        "checks": checks,
        "strict_witness_checks": strict_checks,
        "derived": derived,
        "metrics": metrics,
        "residual": _residual_metrics(
            residual,
            contract,
            float(cfg.stage_b_data_driven_patch_residual_limit),
        ),
    }
    matcher.train(was_training)
    return result


def _run_regime(
    name: str,
    spec: Mapping[str, Any],
    *,
    initial_state: Mapping[str, torch.Tensor],
    query: torch.Tensor,
    patch: torch.Tensor,
    base: torch.Tensor,
    scale: torch.Tensor,
    boxes: torch.Tensor,
    targets: Sequence[Mapping[str, Any]],
    candidate: torch.Tensor,
    cfg: Any,
    sealed_baseline: Mapping[str, float],
    progress_handle,
    artifact_path: Path,
    final_artifact_path: Path,
) -> dict[str, Any]:
    drop_dense_tail_weight = float(spec.get("drop_dense_tail_weight", 0.0))
    cfg.stage_b_data_driven_patch_drop_dense_tail_weight = (
        drop_dense_tail_weight
    )
    baseline_contract = _patch_contract(
        base, boxes, targets, candidate, cfg
    )
    baseline = _metrics(baseline_contract)
    if drop_dense_tail_weight == 0.0 and baseline != dict(sealed_baseline):
        raise RealModelAuditError(
            f"cached MLP {name!r} legacy U0 baseline drifted"
        )
    hidden_dim = int(spec["hidden_dim"])
    context_mode = spec.get("context_mode")
    scorer_architecture = str(spec.get("scorer_architecture", "pointwise_v19"))
    if scorer_architecture == "topk_semantic_context_v1":
        semantic_context_mode = str(
            spec.get("semantic_context_mode", "topk_semantic_v1")
        )
        if (
            context_mode is not None
            or hidden_dim != 128
            or drop_dense_tail_weight != 0.0
        ):
            raise RealModelAuditError(
                f"cached top-k semantic regime {name!r} changed its sealed surface"
            )
        matcher = _CachedTopKSemanticContextResidualMatcher(
            feature_dim=int(query.shape[-1]),
            hidden_dim=hidden_dim,
            context_dim=int(spec["context_dim"]),
            topk=int(spec["context_topk"]),
            residual_limit=float(cfg.stage_b_data_driven_patch_residual_limit),
            init_seed=int(spec["context_init_seed"]),
            context_mode=semantic_context_mode,
        ).to(query.device)
        matcher.load_legacy_state(initial_state)
    elif scorer_architecture != "pointwise_v19":
        raise RealModelAuditError(
            f"unknown cached scorer architecture in {name!r}"
        )
    elif context_mode is None:
        matcher = StageBDataDrivenPatchResidualMatcher(
            feature_dim=int(query.shape[-1]),
            hidden_dim=hidden_dim,
            residual_limit=float(cfg.stage_b_data_driven_patch_residual_limit),
            init_seed=int(cfg.stage_b_data_driven_patch_residual_init_seed),
            center_raw=True,
        ).to(query.device)
        if hidden_dim == 128:
            matcher.load_state_dict(initial_state, strict=True)
    else:
        matcher = _CachedBaseGapResidualMatcher(
            feature_dim=int(query.shape[-1]),
            hidden_dim=hidden_dim,
            residual_limit=float(cfg.stage_b_data_driven_patch_residual_limit),
            init_seed=int(spec["context_init_seed"]),
            context_mode=str(context_mode),
        ).to(query.device)
        matcher.load_legacy_state(initial_state)
    u0_audit = None
    if isinstance(matcher, _CachedTopKSemanticContextResidualMatcher):
        u0_audit = _audit_topk_semantic_u0(
            matcher,
            initial_state=initial_state,
            query=query,
            patch=patch,
            base=base,
            scale=scale,
        )
    optimizer_type = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }.get(str(spec["optimizer"]))
    if optimizer_type is None:
        raise RealModelAuditError(f"unknown cached MLP optimizer in {name!r}")
    parameters = list(matcher.trainable_parameters())
    optimizer = optimizer_type(
        [{"params": parameters, "lr": float(spec["lr"])}],
        lr=float(cfg.lr),
        weight_decay=float(spec["weight_decay"]),
    )
    scaler = _make_grad_scaler(enabled=True, init_scale=cfg.amp_init_scale)
    clip_max_norm = spec["clip_max_norm"]
    steps = int(spec["steps"])
    log_updates = {0, steps, *(int(value) for value in spec["log_updates"])}
    history: list[dict[str, Any]] = []
    sweep: list[dict[str, Any]] = []
    passing_full_records: list[dict[str, Any]] = []
    best_passing_state: dict[str, torch.Tensor] | None = None
    best_passing_key: tuple[float, float] | None = None
    bootstrap: dict[str, Any] = {}
    amp_skips = 0
    last_preclip = 0.0
    last_postclip = 0.0

    def capture_sweep(
        update: int,
        residual: torch.Tensor,
        score: torch.Tensor,
        contract: Mapping[str, torch.Tensor],
    ) -> None:
        nonlocal best_passing_key, best_passing_state
        metrics, checks, derived = _complete_patch_checks(
            contract,
            score,
            boxes=boxes,
            targets=targets,
            candidate=candidate,
            cfg=cfg,
            baseline=baseline,
        )
        compact = {
            "optimizer_updates": update,
            "gate_status": "passed" if all(checks.values()) else "failed",
            "passed_checks": sum(bool(value) for value in checks.values()),
            "total_checks": len(checks),
            "loss": metrics["loss"],
            "keep_safe_instances": metrics["keep_safe_instances"],
            "drop_safe_rows": metrics["drop_safe_rows"],
            "deployed_category_negative_queries": metrics[
                "deployed_category_negative_queries"
            ],
            "gate_retention": metrics["gate_retention"],
            "residual_abs_max": float(residual.detach().float().abs().max().item()),
        }
        sweep.append(compact)
        if all(checks.values()):
            full = {
                "optimizer_updates": update,
                "checks": checks,
                "derived": derived,
                "metrics": metrics,
                "model_tensor_sha256": _tensor_sha256(parameters),
            }
            passing_full_records.append(full)
            key = (
                metrics["deployed_category_negative_queries"],
                metrics["loss"],
            )
            if best_passing_key is None or key < best_passing_key:
                best_passing_key = key
                best_passing_state = {
                    key: value.detach().cpu().clone()
                    for key, value in matcher.state_dict().items()
                }

    def record(update: int) -> None:
        result = _evaluate(
            matcher,
            query=query,
            patch=patch,
            base=base,
            scale=scale,
            boxes=boxes,
            targets=targets,
            candidate=candidate,
            cfg=cfg,
            baseline=baseline,
        )
        result.update(
            optimizer_updates=update,
            grad_norm_preclip=last_preclip,
            grad_norm_postclip=last_postclip,
            amp_scale=float(scaler.get_scale()),
            amp_skips=amp_skips,
        )
        history.append(result)
        summary = {
            "regime": name,
            "optimizer_updates": update,
            "loss": result["metrics"]["loss"],
            "keep_safe": result["metrics"]["keep_safe_instances"],
            "drop_safe": result["metrics"]["drop_safe_rows"],
            "gated_negative": result["metrics"][
                "deployed_category_negative_queries"
            ],
            "gate_retention": result["metrics"]["gate_retention"],
            "residual_abs_max": result["residual"]["abs_max"],
            "gate_status": result["gate_status"],
        }
        print(json.dumps(summary, sort_keys=True), flush=True)
        progress_handle.write(json.dumps(summary, sort_keys=True) + "\n")
        progress_handle.flush()

    matcher.train()
    record(0)
    attempts = 0
    updates = 0
    while updates < steps:
        attempts += 1
        if attempts > steps + 20:
            raise RealModelAuditError(f"cached MLP {name!r} exceeded AMP retries")
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=True):
            residual, score, contract = _forward_contract(
                matcher,
                query,
                patch,
                base,
                scale,
                boxes,
                targets,
                candidate,
                cfg,
            )
            loss = contract["loss"]
        if updates > 0 and (
            not sweep or int(sweep[-1]["optimizer_updates"]) != updates
        ):
            capture_sweep(updates, residual, score, contract)
        if not bool(torch.isfinite(loss).item()):
            raise RealModelAuditError(f"cached MLP {name!r} loss is non-finite")
        scaler.scale(loss).backward()
        scale_before = float(scaler.get_scale())
        scaler.unscale_(optimizer)
        if updates == 0:
            output_grad = _gradient_abs_sum((matcher.output.weight,))
            input_grad = _gradient_abs_sum(tuple(matcher.input.parameters()))
            if isinstance(matcher, _CachedTopKSemanticContextResidualMatcher):
                context_output_grad = _gradient_abs_sum(
                    (matcher.context_output.weight,)
                )
                context_input_grad = _gradient_abs_sum(
                    tuple(matcher.context_input.parameters())
                )
                if not (
                    output_grad > 0.0
                    and context_output_grad > 0.0
                    and input_grad == 0.0
                    and context_input_grad == 0.0
                ):
                    raise RealModelAuditError(
                        "cached top-k six-tensor output bootstrap drifted"
                    )
                bootstrap.update(
                    first_update_output_grad_abs_sum=(
                        output_grad + context_output_grad
                    ),
                    first_update_input_grad_abs_sum=(
                        input_grad + context_input_grad
                    ),
                    first_update_legacy_output_grad_abs_sum=output_grad,
                    first_update_context_output_grad_abs_sum=(
                        context_output_grad
                    ),
                    first_update_legacy_input_grad_abs_sum=input_grad,
                    first_update_context_input_grad_abs_sum=context_input_grad,
                    first_update_only_two_output_layers_have_gradient=True,
                )
            else:
                if output_grad <= 0.0 or input_grad != 0.0:
                    raise RealModelAuditError(
                        "cached MLP zero-output bootstrap drifted"
                    )
                bootstrap.update(
                    first_update_output_grad_abs_sum=output_grad,
                    first_update_input_grad_abs_sum=input_grad,
                    first_update_only_output_layer_has_gradient=True,
                )
        elif updates == 1:
            input_grad = _gradient_abs_sum(tuple(matcher.input.parameters()))
            if isinstance(matcher, _CachedTopKSemanticContextResidualMatcher):
                context_input_grad = _gradient_abs_sum(
                    tuple(matcher.context_input.parameters())
                )
                if input_grad <= 0.0 or context_input_grad <= 0.0:
                    raise RealModelAuditError(
                        "cached top-k six-tensor trunks did not bootstrap"
                    )
                bootstrap.update(
                    second_update_input_grad_abs_sum=(
                        input_grad + context_input_grad
                    ),
                    second_update_legacy_input_grad_abs_sum=input_grad,
                    second_update_context_input_grad_abs_sum=context_input_grad,
                    second_update_both_trunks_have_gradient=True,
                    six_tensor_two_step_bootstrap=True,
                )
            else:
                if input_grad <= 0.0:
                    raise RealModelAuditError("cached MLP trunk did not bootstrap")
                bootstrap.update(
                    second_update_input_grad_abs_sum=input_grad,
                    second_update_trunk_has_gradient=True,
                )
        last_preclip = _grad_norm(parameters)
        if clip_max_norm is not None:
            torch.nn.utils.clip_grad_norm_(parameters, float(clip_max_norm))
        last_postclip = _grad_norm(parameters)
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        del residual, score, contract, loss
        if scale_after < scale_before:
            amp_skips += 1
            continue
        updates += 1
        if updates in log_updates:
            record(updates)
            matcher.train()

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        residual, score, contract = _forward_contract(
            matcher,
            query,
            patch,
            base,
            scale,
            boxes,
            targets,
            candidate,
            cfg,
        )
    capture_sweep(steps, residual, score, contract)
    passing_updates = [
        int(value["optimizer_updates"])
        for value in sweep
        if value["gate_status"] == "passed"
    ]
    late_window_size = min(25, steps)
    late_window = sweep[-late_window_size:]
    stable_late_window = (
        len(late_window) == late_window_size
        and all(value["gate_status"] == "passed" for value in late_window)
    )
    artifact = None
    if best_passing_state is not None:
        best_full = min(
            passing_full_records,
            key=lambda value: (
                value["metrics"]["deployed_category_negative_queries"],
                value["metrics"]["loss"],
            ),
        )
        with artifact_path.open("xb") as handle:
            torch.save(
                {
                    "schema": "pivot.stageb.data_driven.patch_cached_mlp_best/v1",
                    "regime": name,
                    "optimizer_updates": best_full["optimizer_updates"],
                    "model": best_passing_state,
                },
                handle,
            )
            handle.flush()
            os.fsync(handle.fileno())
        artifact = {
            "path": str(artifact_path.resolve()),
            "sha256": _sha256(artifact_path),
            "optimizer_updates": best_full["optimizer_updates"],
            "model_tensor_sha256": best_full["model_tensor_sha256"],
        }
    final = history[-1]
    final_state = {
        key: value.detach().cpu().clone()
        for key, value in matcher.state_dict().items()
    }
    with final_artifact_path.open("xb") as handle:
        torch.save(
            {
                "schema": "pivot.stageb.data_driven.patch_cached_mlp_final/v1",
                "regime": name,
                "optimizer_updates": steps,
                "model": final_state,
            },
            handle,
        )
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "status": (
            "passed" if passing_updates and amp_skips == 0 else "failed"
        ),
        "final_status": final["gate_status"],
        "stable_late_window_status": (
            "passed" if stable_late_window and amp_skips == 0 else "failed"
        ),
        "stable_late_window_updates": late_window_size,
        "spec": dict(spec),
        "objective": {
            "drop_dense_tail_weight": drop_dense_tail_weight,
            "u0_baseline": baseline,
        },
        "architecture": matcher.architecture(),
        "u0_audit": u0_audit,
        "attempts": attempts,
        "amp_skips": amp_skips,
        "bootstrap": bootstrap,
        "first_passing_update": passing_updates[0] if passing_updates else None,
        "last_passing_update": passing_updates[-1] if passing_updates else None,
        "passing_updates": passing_updates,
        "passing_full_records": passing_full_records,
        "best_passing_model_artifact": artifact,
        "final_model_artifact": {
            "path": str(final_artifact_path.resolve()),
            "sha256": _sha256(final_artifact_path),
            "optimizer_updates": steps,
            "model_tensor_sha256": _tensor_sha256(parameters),
        },
        "sweep": sweep,
        "history": history,
        "final_model_tensor_sha256": _tensor_sha256(parameters),
        "final": final,
    }


def _reference_comparison(regime: Mapping[str, Any]) -> dict[str, Any]:
    reference_payload = json.loads(REFERENCE_V19_RECEIPT.read_text(encoding="utf-8"))
    reference = reference_payload["final"]
    observed = regime["final"]
    mapping = {
        "loss": "loss_stage_b_data_driven_patch",
        "keep_component": "stage_b_data_driven_patch_keep_component",
        "drop_component": "stage_b_data_driven_patch_drop_component",
        "keep_safe_instances": "stage_b_data_driven_patch_keep_safe_instances",
        "keep_deployed_instances": (
            "stage_b_data_driven_patch_keep_deployed_instances"
        ),
        "drop_safe_rows": "stage_b_data_driven_patch_drop_safe_rows",
        "drop_deployed_rows": "stage_b_data_driven_patch_drop_deployed_rows",
        "deployed_category_negative_queries": (
            "stage_b_data_driven_patch_deployed_category_negative_queries"
        ),
    }
    differences = {
        metric: float(observed["metrics"][metric]) - float(reference[key])
        for metric, key in mapping.items()
    }
    differences["gate_retention"] = float(
        observed["metrics"]["gate_retention"]
    ) - float(reference["audit_gate_retention"])
    max_abs = max(abs(value) for value in differences.values())
    return {
        "reference_receipt_path": str(REFERENCE_V19_RECEIPT.resolve()),
        "reference_receipt_sha256": _sha256(REFERENCE_V19_RECEIPT),
        "metric_differences_observed_minus_reference": differences,
        "maximum_absolute_metric_difference": max_abs,
        "matches_within_1e_minus_5": max_abs <= 1e-5,
    }


def run_audit(
    *,
    device_name: str,
    seed: int,
    regime_names: Sequence[str],
    progress_path: Path,
) -> dict[str, Any]:
    if not device_name.startswith("cuda") or not torch.cuda.is_available():
        raise RealModelAuditError("cached MLP audit requires CUDA")
    unknown = sorted(set(regime_names) - set(REGIMES))
    if unknown or not regime_names or len(set(regime_names)) != len(regime_names):
        raise RealModelAuditError(f"invalid cached MLP regimes: {unknown}")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    _seed_everything(seed)
    binding = AUDIT_VARIANTS["raw_centered"]
    cfg = SLConfig.fromfile(str(binding["config"]))
    if not (
        math.isclose(float(cfg.stage_b_data_driven_patch_lr), 3e-4)
        and math.isclose(float(cfg.weight_decay), 1e-4)
        and math.isclose(float(cfg.clip_max_norm), 0.1)
        and math.isclose(
            float(cfg.stage_b_data_driven_patch_residual_limit), 0.25
        )
        and bool(cfg.stage_b_data_driven_patch_residual_center_raw)
        and cfg.stage_b_data_driven_patch_drop_positive_anchor_gradient_policy
        == DATA_DRIVEN_PATCH_DROP_ANCHOR_INSTANCE_BALANCED
    ):
        raise RealModelAuditError("sealed v19 cached MLP contract drifted")
    model, criterion, config_path, dataset_path, initializer_path = _build_runtime(
        cfg, device, binding=binding
    )
    del criterion
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    loaded, identities = _select_rows(cfg, payload["train"], seed=seed)
    raw_targets = [target for _image, target in loaded]
    canonical, expressions = _build_stage_b_data_driven_assignment_captions(
        raw_targets
    )
    samples = nested_tensor_from_tensor_list(
        [image for image, _target in loaded]
    ).to(device)
    patches = torch.stack(
        [target["patch"] for target in raw_targets], dim=0
    ).to(device)
    targets = [_move_criterion_target(target, device) for target in raw_targets]
    source_matcher = model.stage_b_data_driven_patch_residual
    if not isinstance(source_matcher, StageBDataDrivenPatchResidualMatcher):
        raise RealModelAuditError("cached MLP source matcher is missing")
    captured: list[tuple[torch.Tensor, torch.Tensor]] = []

    def capture_inputs(_module, inputs) -> None:
        if len(inputs) != 2:
            raise RealModelAuditError("cached MLP hook saw malformed inputs")
        captured.append(
            (inputs[0].detach().clone(), inputs[1].detach().clone())
        )

    hook = source_matcher.register_forward_pre_hook(capture_inputs)
    model.eval()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        outputs = model(
            samples,
            captions=canonical,
            patches=patches,
            stage_b_data_driven_expression_captions=expressions,
        )
    hook.remove()
    if len(captured) != 1:
        raise RealModelAuditError("cached MLP hook did not capture exactly one call")
    query, patch = captured[0]
    base = outputs["pred_logits_patch_base"].detach()
    adapted = outputs["pred_logits_patch"].detach()
    boxes = outputs["pred_boxes"].detach()
    paired_candidate = outputs["stage_b_data_driven_candidate_mask"].detach()
    if base.dim() == 3 and int(base.shape[-1]) == 1:
        base = base[..., 0]
        adapted = adapted[..., 0]
    if (
        tuple(base.shape) != (64, EXPECTED_QUERY_COUNT)
        or not torch.equal(base, adapted)
        or tuple(query.shape) != (64, EXPECTED_QUERY_COUNT, 256)
        or tuple(patch.shape) != (64, 256)
        or tuple(paired_candidate.shape) != (64, EXPECTED_QUERY_COUNT, 2)
        or not torch.equal(paired_candidate[..., 0], paired_candidate[..., 1])
        or not bool(paired_candidate.all().item())
    ):
        raise RealModelAuditError("cached MLP U0 native surface drifted")
    candidate = paired_candidate[..., 0]
    scale = model.patch_logit_scale.detach().exp().clamp(
        max=model.patch_logit_scale_max
    )
    initial_state = {
        key: value.detach().clone() for key, value in source_matcher.state_dict().items()
    }
    del outputs, source_matcher, model, samples, patches
    gc.collect()
    torch.cuda.empty_cache()

    baseline_contract = _patch_contract(base, boxes, targets, candidate, cfg)
    baseline = _metrics(baseline_contract)
    reference_payload = json.loads(
        REFERENCE_V19_RECEIPT.read_text(encoding="utf-8")
    )
    current_selection_sha256 = _canonical_sha256(identities)
    if (
        current_selection_sha256
        != reference_payload["selection"]["member_stream_sha256"]
        or _tensor_sha256([image for image, _target in loaded])
        != reference_payload["selection"]["fixed_image_tensor_stream_sha256"]
        or _tensor_sha256([target["patch"] for target in raw_targets])
        != reference_payload["selection"]["fixed_patch_tensor_stream_sha256"]
    ):
        raise RealModelAuditError("cached MLP O64 selection surface drifted")
    baseline_mapping = {
        "loss": "loss_stage_b_data_driven_patch",
        "keep_component": "stage_b_data_driven_patch_keep_component",
        "drop_component": "stage_b_data_driven_patch_drop_component",
        "keep_safe_instances": "stage_b_data_driven_patch_keep_safe_instances",
        "drop_safe_rows": "stage_b_data_driven_patch_drop_safe_rows",
        "deployed_category_negative_queries": (
            "stage_b_data_driven_patch_deployed_category_negative_queries"
        ),
    }
    if any(
        float(baseline[metric])
        != float(reference_payload["baseline"][reference_key])
        for metric, reference_key in baseline_mapping.items()
    ):
        raise RealModelAuditError("cached MLP O64 baseline surface drifted")
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    with progress_path.open("x", encoding="utf-8", newline="\n") as handle:
        regimes = {
            name: _run_regime(
                name,
                REGIMES[name],
                initial_state=initial_state,
                query=query,
                patch=patch,
                base=base,
                scale=scale,
                boxes=boxes,
                targets=targets,
                candidate=candidate,
                cfg=cfg,
                sealed_baseline=baseline,
                progress_handle=handle,
                artifact_path=progress_path.with_name(
                    f"{progress_path.stem}.{name}.best_model.pth"
                ),
                final_artifact_path=progress_path.with_name(
                    f"{progress_path.stem}.{name}.final_model.pth"
                ),
            )
            for name in regime_names
        }

    reproduction_name = "mlp128_formal_amp_adamw_lr3e4_clip01_u100"
    reproduction = (
        _reference_comparison(regimes[reproduction_name])
        if reproduction_name in regimes
        else None
    )
    if reproduction is not None and not reproduction["matches_within_1e_minus_5"]:
        raise RealModelAuditError("cached formal U100 no longer reproduces full v19")
    return {
        "schema": "pivot.stageb.data_driven.patch_cached_mlp_overfit64/v7",
        "status": "completed",
        "interpretation": {
            "cached_formal_u100_reproduces_full_v19": (
                None
                if reproduction is None
                else reproduction["matches_within_1e_minus_5"]
            ),
            "tested_schedules_found_shared_patch_scorer_witness": any(
                result["status"] == "passed" for result in regimes.values()
            ),
            "tested_schedules_found_stable_late_window_witness": any(
                result["stable_late_window_status"] == "passed"
                for result in regimes.values()
            ),
            "held_out_generalization_is_not_tested_here": True,
        },
        "bindings": {
            "audit_script_sha256": _sha256(Path(__file__).resolve()),
            "config_path": str(config_path),
            "config_sha256": binding["config_sha256"],
            "dataset_config_path": str(dataset_path),
            "dataset_config_sha256": EXPECTED_DATASET_CONFIG_SHA256,
            "initializer_path": str(initializer_path),
            "initializer_sha256": binding["initializer_sha256"],
            "selection_member_stream_sha256": current_selection_sha256,
            "fixed_image_tensor_stream_sha256": _tensor_sha256(
                [image for image, _target in loaded]
            ),
            "fixed_patch_tensor_stream_sha256": _tensor_sha256(
                [target["patch"] for target in raw_targets]
            ),
            "cached_query_tensor_sha256": _tensor_sha256([query]),
            "cached_patch_feature_tensor_sha256": _tensor_sha256([patch]),
            "cached_base_score_tensor_sha256": _tensor_sha256([base]),
        },
        "invariants": {
            "same_64_unique_images_as_strict_o64": True,
            "all_900_queries_are_candidates": True,
            "cached_inputs_are_native_autocast_residual_module_inputs": True,
            "base_score_and_logit_scale_keep_native_forward_dtype": True,
            "formal_regime_uses_amp_gradscaler_adamw_weight_decay_and_branch_clip": True,
            "formal_regime_preserves_zero_output_two_step_bootstrap": True,
            "only_shared_residual_scorer_parameters_are_trainable": True,
            "exact_deployment_gate_category_patch_loss_is_used": True,
            "each_regime_recomputes_its_own_u0_objective_baseline": True,
            "gate_retention_is_part_of_patch_status": True,
            "no_teacher_or_winner_score_is_used": True,
            "topk_context_is_cached_audit_only": True,
            "topk_context_uses_only_detached_query_patch_and_base_score": True,
            "topk_u0_and_permutation_checks_are_fail_closed_when_selected": True,
        },
        "selection": {"seed": seed, "members": identities},
        "cached_dtype": {
            "query": str(query.dtype),
            "patch": str(patch.dtype),
            "base": str(base.dtype),
            "scale": str(scale.dtype),
        },
        "base_patch_logit_scale": float(scale.float().item()),
        "baseline": baseline,
        "reference_u100_reproduction": reproduction,
        "regimes": regimes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--regimes",
        nargs="+",
        choices=tuple(REGIMES),
        default=("mlp128_formal_amp_adamw_lr3e4_clip01_u100",),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    output_json = args.output_json.expanduser()
    if not output_json.is_absolute():
        output_json = Path.cwd() / output_json
    output_json.parent.mkdir(parents=True, exist_ok=True)
    progress_path = output_json.with_suffix(".progress.jsonl")
    result = run_audit(
        device_name=args.device,
        seed=args.seed,
        regime_names=args.regimes,
        progress_path=progress_path,
    )
    _write_json_exclusive(output_json, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "interpretation": result["interpretation"],
                "output_json": str(output_json),
                "progress_jsonl": str(progress_path),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
