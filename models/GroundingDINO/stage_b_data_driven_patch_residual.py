"""Nonlinear residual matcher for the data-driven category patch score."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn


DATA_DRIVEN_PATCH_RESIDUAL_INITIALIZER_SCHEMA = (
    "pivot.stageb.data_driven.patch_residual_initializer/v1"
)
DATA_DRIVEN_PATCH_TOPK_SEMANTIC_INITIALIZER_SCHEMA = (
    "pivot.stageb.data_driven.patch_residual_initializer/v2"
)
DATA_DRIVEN_PATCH_RESIDUAL_CONTRACT = "detached_qp_mlp128_tanh025_v1"
DATA_DRIVEN_PATCH_RESIDUAL_RAW_CENTERED_CONTRACT = (
    "detached_qp_mlp128_query_raw_centered_tanh025_v2"
)
DATA_DRIVEN_PATCH_RESIDUAL_TOPK_SEMANTIC_CONTRACT = (
    "detached_qp_base_topk10_semantic_context16_"
    "query_raw_centered_tanh025_v3"
)


def _tensor_state_sha256(
    state: Mapping[str, Tensor], keys: list[str]
) -> str:
    digest = hashlib.sha256()
    for key in sorted(keys):
        value = state[key].detach().cpu().contiguous()
        header = json.dumps(
            [key, str(value.dtype), list(value.shape)],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        if value.numel():
            digest.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()


class StageBDataDrivenPatchResidualMatcher(nn.Module):
    """Score detached normalized query/support pairs with a bounded residual."""

    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 128,
        residual_limit: float = 0.25,
        init_seed: int = 42,
        center_raw: bool = False,
    ) -> None:
        super().__init__()
        if (
            isinstance(feature_dim, bool)
            or int(feature_dim) <= 0
            or isinstance(hidden_dim, bool)
            or int(hidden_dim) <= 0
            or not math.isfinite(float(residual_limit))
            or float(residual_limit) <= 0.0
            or isinstance(init_seed, bool)
            or not isinstance(center_raw, bool)
        ):
            raise ValueError("patch residual architecture is invalid")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.residual_limit = float(residual_limit)
        self.init_seed = int(init_seed)
        self.center_raw = center_raw
        self.contract = (
            DATA_DRIVEN_PATCH_RESIDUAL_RAW_CENTERED_CONTRACT
            if center_raw
            else DATA_DRIVEN_PATCH_RESIDUAL_CONTRACT
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.init_seed)
            self.input = nn.Linear(4 * self.feature_dim, self.hidden_dim)
            self.output = nn.Linear(self.hidden_dim, 1, bias=False)
        nn.init.zeros_(self.output.weight)

    def trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        parameters = tuple(self.parameters())
        if len(parameters) != 3:
            raise RuntimeError("patch residual must expose exactly three tensors")
        return parameters

    def _pair_features(self, query: Tensor, patch: Tensor) -> Tensor:
        if query.dim() != 3 or int(query.shape[-1]) != self.feature_dim:
            raise ValueError("patch residual query must have shape (B,Q,D)")
        if patch.dim() not in (2, 3) or int(patch.shape[-1]) != self.feature_dim:
            raise ValueError("patch residual support must have shape (B,D) or (B,K,D)")
        if int(query.shape[0]) != int(patch.shape[0]):
            raise ValueError("patch residual query/support batches differ")
        if not query.is_floating_point() or not patch.is_floating_point():
            raise TypeError("patch residual inputs must be floating point")
        if query.device != patch.device:
            raise ValueError("patch residual inputs must share a device")
        query = F.normalize(query.detach(), dim=-1)
        patch = F.normalize(patch.detach(), dim=-1)
        if patch.dim() == 2:
            expanded_query = query
            expanded_patch = patch[:, None, :].expand_as(query)
        else:
            expanded_query = query[:, :, None, :].expand(
                -1, -1, int(patch.shape[1]), -1
            )
            expanded_patch = patch[:, None, :, :].expand(
                -1, int(query.shape[1]), -1, -1
            )
        return torch.cat(
            (
                expanded_query,
                expanded_patch,
                expanded_query * expanded_patch,
                (expanded_query - expanded_patch).abs(),
            ),
            dim=-1,
        )

    def forward(self, query: Tensor, patch: Tensor) -> Tensor:
        features = self._pair_features(query, patch)
        raw = self.output(F.gelu(self.input(features))).squeeze(-1)
        if self.center_raw:
            raw = raw - raw.mean(dim=1, keepdim=True)
        limit = self.residual_limit
        return limit * torch.tanh(raw / limit)

    def architecture(self) -> dict[str, Any]:
        architecture = {
            "contract": self.contract,
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "residual_limit": self.residual_limit,
            "init_seed": self.init_seed,
            "trainable_tensors": 3,
            "trainable_parameters": sum(
                parameter.numel() for parameter in self.parameters()
            ),
        }
        if self.center_raw:
            architecture["query_centering"] = "raw_mean_before_tanh_v1"
        return architecture


class StageBDataDrivenTopKPatchResidualMatcher(nn.Module):
    """Condition a bounded patch residual on the row's top competitors."""

    requires_base_score = True

    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 128,
        context_dim: int = 16,
        topk: int = 10,
        residual_limit: float = 0.25,
        init_seed: int = 42,
    ) -> None:
        super().__init__()
        if (
            isinstance(feature_dim, bool)
            or int(feature_dim) <= 0
            or isinstance(hidden_dim, bool)
            or int(hidden_dim) <= 0
            or isinstance(context_dim, bool)
            or int(context_dim) <= 0
            or isinstance(topk, bool)
            or int(topk) <= 0
            or not math.isfinite(float(residual_limit))
            or float(residual_limit) <= 0.0
            or isinstance(init_seed, bool)
        ):
            raise ValueError("top-k patch residual architecture is invalid")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.context_dim = int(context_dim)
        self.topk = int(topk)
        self.residual_limit = float(residual_limit)
        self.init_seed = int(init_seed)
        self.contract = DATA_DRIVEN_PATCH_RESIDUAL_TOPK_SEMANTIC_CONTRACT
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

    def trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        parameters = tuple(self.parameters())
        if len(parameters) != 6:
            raise RuntimeError("top-k patch residual must expose six tensors")
        return parameters

    def _semantic_features(
        self,
        query: Tensor,
        patch: Tensor,
        base_score: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if query.dim() != 3 or int(query.shape[-1]) != self.feature_dim:
            raise ValueError("top-k patch residual query must have shape (B,Q,D)")
        if patch.dim() not in (2, 3) or int(patch.shape[-1]) != self.feature_dim:
            raise ValueError(
                "top-k patch residual support must have shape (B,D) or (B,K,D)"
            )
        expected_base_shape = tuple(query.shape[:2]) + tuple(patch.shape[1:-1])
        if (
            int(query.shape[0]) != int(patch.shape[0])
            or tuple(base_score.shape) != expected_base_shape
        ):
            raise ValueError("top-k patch residual inputs are misaligned")
        if not all(
            value.is_floating_point() for value in (query, patch, base_score)
        ):
            raise TypeError("top-k patch residual inputs must be floating point")
        if query.device != patch.device or query.device != base_score.device:
            raise ValueError("top-k patch residual inputs must share a device")
        if not bool(torch.isfinite(base_score).all().item()):
            raise ValueError("top-k patch residual base score must be finite")

        normalized_query = F.normalize(query.detach(), dim=-1)
        normalized_patch = F.normalize(patch.detach(), dim=-1)
        if patch.dim() == 2:
            expanded_query = normalized_query
            expanded_patch = normalized_patch[:, None, :].expand_as(
                normalized_query
            )
        else:
            expanded_query = normalized_query[:, :, None, :].expand(
                -1, -1, int(patch.shape[1]), -1
            )
            expanded_patch = normalized_patch[:, None, :, :].expand(
                -1, int(query.shape[1]), -1, -1
            )
        pair_features = torch.cat(
            (
                expanded_query,
                expanded_patch,
                expanded_query * expanded_patch,
                (expanded_query - expanded_patch).abs(),
            ),
            dim=-1,
        )
        hidden = F.gelu(self.input(pair_features))

        base = base_score.detach().float()
        centered = base - base.mean(dim=1, keepdim=True)
        std = centered.square().mean(dim=1, keepdim=True).clamp_min(1e-6).sqrt()
        standardized = (centered / std).clamp(-5.0, 5.0)
        gap = standardized.amax(dim=1, keepdim=True) - standardized
        top_count = min(self.topk, int(query.shape[1]))
        top_indices = torch.topk(base, k=top_count, dim=1).indices
        gather_index = top_indices.unsqueeze(-1).expand(
            *top_indices.shape, self.hidden_dim
        )
        top_hidden = torch.gather(hidden, dim=1, index=gather_index)
        prototype = top_hidden.float().mean(dim=1).to(dtype=hidden.dtype)
        expanded_prototype = prototype.unsqueeze(1).expand_as(hidden)
        score_context = torch.stack(
            (standardized / 5.0, gap / 10.0), dim=-1
        ).to(dtype=hidden.dtype)
        context = torch.cat(
            (
                hidden * expanded_prototype,
                (hidden - expanded_prototype).abs(),
                score_context,
            ),
            dim=-1,
        )
        return hidden, F.gelu(self.context_input(context))

    def forward(
        self, query: Tensor, patch: Tensor, base_score: Tensor
    ) -> Tensor:
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
            "contract": self.contract,
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "context_dim": self.context_dim,
            "topk": self.topk,
            "residual_limit": self.residual_limit,
            "init_seed": self.init_seed,
            "query_centering": "raw_mean_before_tanh_v1",
            "row_pool": "mean_of_detached_base_score_topk_hidden_v1",
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


def validate_data_driven_patch_residual_initializer_payload(
    expected_model: nn.Module,
    payload: Mapping[str, Any],
    *,
    checkpoint_label: str,
    expected_source_checkpoint_sha256: str,
    expected_a0_initializer_sha256: str,
    expected_source_optimizer_updates: int,
) -> None:
    """Fail closed on the additive, model-only residual initializer."""
    if not isinstance(payload, Mapping) or set(payload) != {
        "model",
        "data_driven_patch_residual_initializer",
    }:
        raise ValueError(
            f"{checkpoint_label}: patch residual initializer keys drifted"
        )
    state = payload.get("model")
    contract = payload.get("data_driven_patch_residual_initializer")
    if not isinstance(state, Mapping) or not isinstance(contract, Mapping):
        raise ValueError(f"{checkpoint_label}: patch residual initializer malformed")
    residual = getattr(expected_model, "stage_b_data_driven_patch_residual", None)
    expected_schema = (
        DATA_DRIVEN_PATCH_TOPK_SEMANTIC_INITIALIZER_SCHEMA
        if isinstance(residual, StageBDataDrivenTopKPatchResidualMatcher)
        else DATA_DRIVEN_PATCH_RESIDUAL_INITIALIZER_SCHEMA
    )
    if contract.get("schema") != expected_schema:
        raise ValueError(f"{checkpoint_label}: patch residual schema drifted")

    expected = expected_model.state_dict()
    if set(state) != set(expected):
        raise ValueError(
            f"{checkpoint_label}: patch residual model coverage drifted"
        )
    for key, wanted in expected.items():
        value = state.get(key)
        if (
            not torch.is_tensor(value)
            or value.dtype != wanted.dtype
            or tuple(value.shape) != tuple(wanted.shape)
        ):
            raise ValueError(
                f"{checkpoint_label}: residual tensor shape/dtype drift at {key}"
            )

    if not isinstance(
        residual,
        (
            StageBDataDrivenPatchResidualMatcher,
            StageBDataDrivenTopKPatchResidualMatcher,
        ),
    ):
        raise ValueError(f"{checkpoint_label}: expected model has no patch residual")
    architecture = residual.architecture()
    if contract.get("architecture") != architecture:
        raise ValueError(f"{checkpoint_label}: residual architecture drifted")

    source = contract.get("source_role_routed_initializer")
    source_checkpoint = contract.get("source_checkpoint")
    source_a0 = contract.get("source_a0_initializer")
    if not (
        isinstance(source, Mapping)
        and re.fullmatch(r"[0-9a-f]{64}", str(source.get("sha256", "")))
        and isinstance(source_checkpoint, Mapping)
        and source_checkpoint.get("sha256") == expected_source_checkpoint_sha256
        and isinstance(source_a0, Mapping)
        and source_a0.get("sha256") == expected_a0_initializer_sha256
        and contract.get("source_optimizer_updates")
        == expected_source_optimizer_updates
    ):
        raise ValueError(f"{checkpoint_label}: residual source lineage drifted")

    residual_keys = sorted(
        key
        for key in state
        if key.startswith("stage_b_data_driven_patch_residual.")
    )
    base_keys = sorted(set(state) - set(residual_keys))
    expected_residual_keys = sorted(
        "stage_b_data_driven_patch_residual." + key
        for key in residual.state_dict()
    )
    if residual_keys != expected_residual_keys:
        raise ValueError(f"{checkpoint_label}: residual key set drifted")
    zero_output_keys = [
        "stage_b_data_driven_patch_residual.output.weight"
    ]
    if isinstance(residual, StageBDataDrivenTopKPatchResidualMatcher):
        zero_output_keys.append(
            "stage_b_data_driven_patch_residual.context_output.weight"
        )
    if not (
        contract.get("residual_keys") == residual_keys
        and contract.get("base_key_count") == len(base_keys)
        and contract.get("residual_tensor_sha256")
        == _tensor_state_sha256(state, residual_keys)
        and contract.get("base_tensor_sha256")
        == _tensor_state_sha256(state, base_keys)
        and contract.get("full_model_tensor_sha256")
        == _tensor_state_sha256(state, sorted(state))
        and all(
            bool((state[key] == 0).all().item()) for key in zero_output_keys
        )
    ):
        raise ValueError(f"{checkpoint_label}: residual tensor binding drifted")
    required_invariants = {
        "base_model_tensors_are_bitwise_source_copy",
        "residual_output_is_exactly_zero_initialized",
        "initializer_contains_no_optimizer_criterion_scaler_or_rng",
        "no_teacher_or_old_winner_tensor_added",
        "formal_load_requires_exact_model_key_coverage",
    }
    if isinstance(residual, StageBDataDrivenTopKPatchResidualMatcher):
        required_invariants.update(
            {
                "context_output_is_exactly_zero_initialized",
                "topk_context_uses_only_inference_available_detached_inputs",
                "single_and_multi_patch_share_the_same_query_set_contract",
            }
        )
    invariants = contract.get("invariants")
    if not isinstance(invariants, Mapping) or any(
        invariants.get(key) is not True for key in required_invariants
    ):
        raise ValueError(f"{checkpoint_label}: residual invariants drifted")


__all__ = [
    "DATA_DRIVEN_PATCH_RESIDUAL_CONTRACT",
    "DATA_DRIVEN_PATCH_RESIDUAL_RAW_CENTERED_CONTRACT",
    "DATA_DRIVEN_PATCH_RESIDUAL_TOPK_SEMANTIC_CONTRACT",
    "DATA_DRIVEN_PATCH_RESIDUAL_INITIALIZER_SCHEMA",
    "DATA_DRIVEN_PATCH_TOPK_SEMANTIC_INITIALIZER_SCHEMA",
    "StageBDataDrivenPatchResidualMatcher",
    "StageBDataDrivenTopKPatchResidualMatcher",
    "validate_data_driven_patch_residual_initializer_payload",
]
