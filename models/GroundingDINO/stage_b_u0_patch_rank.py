"""Patch-category rank residual on top of the sealed R100/P50 all-query model."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import Tensor, nn

from groundingdino.util import box_ops

from .stage_b_gdino_score_adapter import baseline_preserving_top1_rank_loss


U0_PATCH_RANK_CONTRACT_VERSION = 1
U1_DIRECT_PATCH_CONTRACT_VERSION = 2
U0_INITIALIZER_SCHEMA = "pivot.stageb.u0_initializer/v2"
U1_DIRECT_PATCH_INITIALIZER_SCHEMA = "pivot.stageb.u1_direct_patch_initializer/v1"
U0_PATCH_BACKBONE_PREFIX = "patch_encoder.backbone."
U1_DIRECT_PATCH_ADDED_KEYS = frozenset(
    {
        "stage_b_u0_patch_rank_adapter.direct_patch_gain",
        "stage_b_u0_patch_rank_adapter._contract_direct_patch_gain_limit",
    }
)
U1_DIRECT_PATCH_REPLACED_KEYS = frozenset(
    {"stage_b_u0_patch_rank_adapter._contract_version"}
)
U0_SEALED_TEACHER_ARCHITECTURE_FIELDS = (
    "modelname",
    "hidden_dim",
    "num_queries",
    "stage_b_gdino_score_adapter",
    "stage_b_gdino_adapter_dim",
    "stage_b_gdino_gate_hidden_dim",
    "stage_b_gdino_gate_pool_temperature",
    "stage_b_gdino_gate_topk",
    "patch_only",
    "stage_b",
    "stage_b_v7",
    "stage_b_v11_fixed_text",
    "stage_b_legacy_global_gate",
    "enable_patch_branch",
)
U0_TEACHER_FUNCTIONAL_FIELDS = (
    "rank_feature",
    "rank_residual",
    "rank_score",
    "confidence_feature",
    "confidence_gate",
    "confidence_score",
)
U0_PATCH_SOURCE_KEYS = frozenset(
    {
        "patch_encoder.input_proj.0.weight",
        "patch_encoder.input_proj.0.bias",
        "patch_encoder.input_proj.1.weight",
        "patch_encoder.input_proj.1.bias",
        "patch_encoder.norm.weight",
        "patch_encoder.norm.bias",
        "query_proj_for_patch.weight",
        "query_proj_for_patch.bias",
        "patch_logit_scale",
    }
)


def stage_b_u0_tensor_state_sha256(
    state: Mapping[str, Any],
    keys: Sequence[str],
) -> str:
    selected = sorted(set(str(key) for key in keys))
    if not selected:
        raise ValueError("cannot hash an empty U0 tensor-state selection")
    digest = hashlib.sha256()
    for key in selected:
        value = state.get(key)
        if not torch.is_tensor(value):
            raise ValueError(f"U0 model state {key!r} is missing or is not a tensor")
        header = json.dumps(
            [key, str(value.dtype), list(value.shape)],
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        tensor = value.detach().cpu().contiguous()
        if tensor.numel():
            digest.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()


def validate_stage_b_u0_initializer_payload(
    expected_model_or_state: Any,
    payload: Mapping[str, Any],
    *,
    checkpoint_label: str,
) -> None:
    """Require the sealed, full-model U0 initializer contract."""
    expected_model = (
        expected_model_or_state
        if isinstance(expected_model_or_state, nn.Module)
        else None
    )
    expected_state = (
        expected_model.state_dict()
        if expected_model is not None
        else expected_model_or_state
    )
    if not isinstance(expected_state, Mapping):
        raise TypeError(f"{checkpoint_label}: expected U0 state is not a mapping")
    if not isinstance(payload, Mapping) or set(payload) != {
        "model",
        "u0_initializer",
    }:
        raise ValueError(f"{checkpoint_label}: U0 initializer top-level keys drifted")
    state = payload.get("model")
    contract = payload.get("u0_initializer")
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"{checkpoint_label}: U0 initializer model state is missing")
    if not isinstance(contract, Mapping) or contract.get("schema") != U0_INITIALIZER_SCHEMA:
        raise ValueError(f"{checkpoint_label}: U0 initializer contract is missing")
    teacher_architecture = contract.get("sealed_teacher_architecture")
    u0_architecture = contract.get("u0_architecture")
    architecture_fields = set(U0_SEALED_TEACHER_ARCHITECTURE_FIELDS)
    if (
        not isinstance(teacher_architecture, Mapping)
        or set(teacher_architecture) != architecture_fields
        or not isinstance(u0_architecture, Mapping)
        or set(u0_architecture) != architecture_fields
    ):
        raise ValueError(f"{checkpoint_label}: U0 teacher architecture contract drifted")
    for key in architecture_fields - {"enable_patch_branch"}:
        if u0_architecture.get(key) != teacher_architecture.get(key):
            raise ValueError(
                f"{checkpoint_label}: U0 architecture differs from teacher at {key}"
            )
    if teacher_architecture.get("enable_patch_branch") is not False or (
        u0_architecture.get("enable_patch_branch") is not True
    ):
        raise ValueError(f"{checkpoint_label}: U0 patch architecture transition drifted")
    functional = contract.get("teacher_functional_bitwise")
    if not isinstance(functional, Mapping) or set(functional) != set(
        U0_TEACHER_FUNCTIONAL_FIELDS
    ) or any(functional.get(key) is not True for key in U0_TEACHER_FUNCTIONAL_FIELDS):
        raise ValueError(f"{checkpoint_label}: U0 teacher functional contract drifted")
    if expected_model is not None:
        root = (
            expected_model.module
            if hasattr(expected_model, "module")
            else expected_model
        )
        adapter = getattr(root, "stage_b_gdino_score_adapter", None)
        if adapter is None:
            raise ValueError(f"{checkpoint_label}: U0 model has no sealed score adapter")
        runtime_architecture = {
            "hidden_dim": int(root.hidden_dim),
            "num_queries": int(root.num_queries),
            "stage_b_gdino_score_adapter": True,
            "stage_b_gdino_adapter_dim": int(adapter.adapter_dim),
            "stage_b_gdino_gate_hidden_dim": int(
                adapter.confidence_gate[0].out_features
            ),
            "stage_b_gdino_gate_pool_temperature": float(
                adapter.gate_pool_temperature
            ),
            "stage_b_gdino_gate_topk": int(adapter.gate_topk),
            "patch_only": bool(root.patch_only),
            "enable_patch_branch": bool(root.enable_patch_branch),
        }
        for key, observed in runtime_architecture.items():
            if u0_architecture.get(key) != observed:
                raise ValueError(
                    f"{checkpoint_label}: runtime U0 architecture drift at {key}: "
                    f"contract={u0_architecture.get(key)!r}, runtime={observed!r}"
                )
    expected_keys = set(str(key) for key in expected_state)
    state_keys = set(str(key) for key in state)
    if state_keys != expected_keys:
        raise ValueError(
            f"{checkpoint_label}: U0 initializer full-model key coverage drifted "
            f"(missing={sorted(expected_keys - state_keys)[:8]}, "
            f"unexpected={sorted(state_keys - expected_keys)[:8]})"
        )
    for key in sorted(expected_keys):
        value = state.get(key)
        wanted = expected_state.get(key)
        if not torch.is_tensor(value) or not torch.is_tensor(wanted):
            raise ValueError(f"{checkpoint_label}: non-tensor U0 model state at {key}")
        if value.dtype != wanted.dtype or tuple(value.shape) != tuple(wanted.shape):
            raise ValueError(
                f"{checkpoint_label}: U0 initializer shape/dtype drift at {key}"
            )

    roles = contract.get("role_keys")
    expected_roles = {
        "merged",
        "shared_backbone_alias",
        "stagea_patch",
        "u0_zero",
    }
    if not isinstance(roles, Mapping) or set(roles) != expected_roles:
        raise ValueError(f"{checkpoint_label}: U0 initializer role contract drifted")
    normalized_roles = {}
    for role in sorted(expected_roles):
        values = roles.get(role)
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"{checkpoint_label}: U0 role {role} is empty or malformed")
        normalized = [str(key) for key in values]
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"{checkpoint_label}: U0 role {role} contains duplicates")
        normalized_roles[role] = normalized
    flattened = [key for values in normalized_roles.values() for key in values]
    if len(set(flattened)) != len(flattened) or set(flattened) != state_keys:
        raise ValueError(f"{checkpoint_label}: U0 role coverage is not an exact partition")
    if set(normalized_roles["stagea_patch"]) != set(U0_PATCH_SOURCE_KEYS):
        raise ValueError(f"{checkpoint_label}: U0 Stage-A patch role drifted")
    if not all(
        key.startswith(U0_PATCH_BACKBONE_PREFIX)
        for key in normalized_roles["shared_backbone_alias"]
    ):
        raise ValueError(f"{checkpoint_label}: U0 shared-backbone role drifted")
    if not all(
        key.startswith("stage_b_u0_patch_rank_adapter.")
        for key in normalized_roles["u0_zero"]
    ):
        raise ValueError(f"{checkpoint_label}: U0 residual role drifted")

    hash_contract = (
        ("full_model_tensor_sha256", sorted(state_keys)),
        ("merged_teacher_tensor_sha256", normalized_roles["merged"]),
        ("stagea_patch_tensor_sha256", normalized_roles["stagea_patch"]),
        (
            "shared_backbone_alias_tensor_sha256",
            normalized_roles["shared_backbone_alias"],
        ),
        ("u0_zero_tensor_sha256", normalized_roles["u0_zero"]),
    )
    for field, keys in hash_contract:
        if contract.get(field) != stage_b_u0_tensor_state_sha256(state, keys):
            raise ValueError(f"{checkpoint_label}: U0 initializer {field} drifted")
    for alias in normalized_roles["shared_backbone_alias"]:
        base_key = alias.removeprefix("patch_encoder.")
        if base_key not in state or not torch.equal(state[alias], state[base_key]):
            raise ValueError(f"{checkpoint_label}: U0 shared backbone alias differs: {alias}")
    for key in (
        "stage_b_u0_patch_rank_adapter.output.weight",
        "stage_b_u0_patch_rank_adapter.output.bias",
    ):
        if key not in state or int(torch.count_nonzero(state[key]).item()):
            raise ValueError(f"{checkpoint_label}: serialized U0 residual is not zero")
    invariants = contract.get("invariants")
    required_invariants = {
        "merged_teacher_copied_bitwise",
        "stagea_patch_specific_keys_only",
        "stagea_patch_backbone_imported",
        "patch_backbone_aliases_source_b58",
        "u0_output_exactly_zero",
        "u0_rank_equals_r100_at_initialization",
        "p50_confidence_unchanged",
    }
    if not isinstance(invariants, Mapping) or any(
        invariants.get(key) is not (key != "stagea_patch_backbone_imported")
        for key in required_invariants
    ):
        raise ValueError(f"{checkpoint_label}: U0 initializer invariants drifted")


def validate_stage_b_u1_direct_patch_initializer_payload(
    expected_model_or_state: Any,
    payload: Mapping[str, Any],
    *,
    checkpoint_label: str,
) -> None:
    expected_model = (
        expected_model_or_state
        if isinstance(expected_model_or_state, nn.Module)
        else None
    )
    expected_state = (
        expected_model.state_dict()
        if expected_model is not None
        else expected_model_or_state
    )
    if not isinstance(expected_state, Mapping):
        raise TypeError(f"{checkpoint_label}: expected U1 state is not a mapping")
    if not isinstance(payload, Mapping) or set(payload) != {
        "model",
        "u1_initializer",
    }:
        raise ValueError(f"{checkpoint_label}: U1 initializer top-level keys drifted")
    state = payload.get("model")
    contract = payload.get("u1_initializer")
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"{checkpoint_label}: U1 initializer model state is missing")
    if (
        not isinstance(contract, Mapping)
        or contract.get("schema") != U1_DIRECT_PATCH_INITIALIZER_SCHEMA
    ):
        raise ValueError(f"{checkpoint_label}: U1 initializer contract is missing")
    expected_keys = set(str(key) for key in expected_state)
    state_keys = set(str(key) for key in state)
    if state_keys != expected_keys:
        raise ValueError(
            f"{checkpoint_label}: U1 full-model key coverage drifted "
            f"(missing={sorted(expected_keys - state_keys)[:8]}, "
            f"unexpected={sorted(state_keys - expected_keys)[:8]})"
        )
    for key in sorted(expected_keys):
        value = state.get(key)
        wanted = expected_state.get(key)
        if not torch.is_tensor(value) or not torch.is_tensor(wanted):
            raise ValueError(f"{checkpoint_label}: non-tensor U1 model state at {key}")
        if value.dtype != wanted.dtype or tuple(value.shape) != tuple(wanted.shape):
            raise ValueError(f"{checkpoint_label}: U1 shape/dtype drift at {key}")
    roles = contract.get("role_keys")
    expected_roles = {"source_preserved", "u1_added", "u1_replaced"}
    if not isinstance(roles, Mapping) or set(roles) != expected_roles:
        raise ValueError(f"{checkpoint_label}: U1 role contract drifted")
    normalized = {}
    for role in expected_roles:
        values = roles.get(role)
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"{checkpoint_label}: U1 role {role} is malformed")
        normalized[role] = [str(key) for key in values]
    flat = [key for values in normalized.values() for key in values]
    if len(flat) != len(set(flat)) or set(flat) != state_keys:
        raise ValueError(f"{checkpoint_label}: U1 roles are not an exact partition")
    if set(normalized["u1_added"]) != set(U1_DIRECT_PATCH_ADDED_KEYS):
        raise ValueError(f"{checkpoint_label}: U1 added-key contract drifted")
    if set(normalized["u1_replaced"]) != set(U1_DIRECT_PATCH_REPLACED_KEYS):
        raise ValueError(f"{checkpoint_label}: U1 replaced-key contract drifted")
    for field, keys in (
        ("full_model_tensor_sha256", sorted(state_keys)),
        ("source_preserved_tensor_sha256", normalized["source_preserved"]),
        ("u1_added_tensor_sha256", normalized["u1_added"]),
        ("u1_replaced_tensor_sha256", normalized["u1_replaced"]),
    ):
        if contract.get(field) != stage_b_u0_tensor_state_sha256(state, keys):
            raise ValueError(f"{checkpoint_label}: U1 {field} drifted")
    gain = state.get("stage_b_u0_patch_rank_adapter.direct_patch_gain")
    version = state.get("stage_b_u0_patch_rank_adapter._contract_version")
    if not torch.is_tensor(gain) or int(torch.count_nonzero(gain).item()):
        raise ValueError(f"{checkpoint_label}: U1 direct patch gain is not zero")
    if not torch.is_tensor(version) or int(version.item()) != U1_DIRECT_PATCH_CONTRACT_VERSION:
        raise ValueError(f"{checkpoint_label}: U1 contract version drifted")
    functional = contract.get("u100_functional_bitwise")
    if not isinstance(functional, Mapping) or set(functional) != {
        "teacher_rank_score",
        "patch_rank_residual",
        "rank_score",
    } or any(value is not True for value in functional.values()):
        raise ValueError(f"{checkpoint_label}: U1 source functional contract drifted")
    invariants = contract.get("invariants")
    required = {
        "u100_source_preserved_bitwise": True,
        "direct_patch_gain_zero": True,
        "u1_rank_equals_u100_at_initialization": True,
        "r100_p50_b58_frozen_source_unchanged": True,
    }
    if not isinstance(invariants, Mapping) or any(
        invariants.get(key) is not wanted for key, wanted in required.items()
    ):
        raise ValueError(f"{checkpoint_label}: U1 initializer invariants drifted")
    if expected_model is not None:
        root = expected_model.module if hasattr(expected_model, "module") else expected_model
        adapter = getattr(root, "stage_b_u0_patch_rank_adapter", None)
        if adapter is None or adapter.direct_patch_gain is None:
            raise ValueError(f"{checkpoint_label}: runtime model has no U1 direct patch skip")
        recorded_limit = state.get(
            "stage_b_u0_patch_rank_adapter._contract_direct_patch_gain_limit"
        )
        if not torch.is_tensor(recorded_limit) or not torch.equal(
            recorded_limit.detach().cpu(),
            torch.as_tensor(adapter.direct_patch_gain_limit, dtype=recorded_limit.dtype),
        ):
            raise ValueError(f"{checkpoint_label}: runtime U1 gain limit drifted")


def validate_stage_b_u0_patch_rank_checkpoint(
    model: nn.Module,
    state_dict: Mapping[str, Any],
    *,
    checkpoint_label: str,
) -> None:
    root = model.module if hasattr(model, "module") else model
    adapter = getattr(root, "stage_b_u0_patch_rank_adapter", None)
    if adapter is None:
        raise ValueError(f"{checkpoint_label}: model has no Stage-B U0 patch-rank adapter")
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"{checkpoint_label}: checkpoint model state must be a mapping")
    prefix = "stage_b_u0_patch_rank_adapter."
    expected = {prefix + key: value for key, value in adapter.state_dict().items()}
    provided = {
        str(key): value
        for key, value in state_dict.items()
        if str(key).startswith(prefix)
    }
    missing = sorted(set(expected).difference(provided))
    unexpected = sorted(set(provided).difference(expected))
    shape_mismatches = []
    contract_mismatches = []
    for key in sorted(set(expected).intersection(provided)):
        value = provided[key]
        wanted = expected[key]
        if not torch.is_tensor(value) or tuple(value.shape) != tuple(wanted.shape):
            shape_mismatches.append(
                (
                    key,
                    tuple(wanted.shape),
                    tuple(value.shape) if torch.is_tensor(value) else type(value).__name__,
                )
            )
            continue
        if key.startswith(prefix + "_contract_") and not torch.equal(
            value.detach().to(device="cpu", dtype=wanted.dtype),
            wanted.detach().to(device="cpu"),
        ):
            contract_mismatches.append(key)
    if missing or unexpected or shape_mismatches or contract_mismatches:
        raise ValueError(
            f"{checkpoint_label}: incompatible Stage-B U0 patch-rank state "
            f"(missing={missing[:8]}, unexpected={unexpected[:8]}, "
            f"shape_mismatches={shape_mismatches[:8]}, "
            f"contract_mismatches={contract_mismatches[:8]})"
        )


class StageBU0PatchRankAdapter(nn.Module):
    """Learn a patch correction, with an optional inference-only category gate."""

    input_dim = 3

    def __init__(
        self,
        *,
        query_count: int = 900,
        hidden_dim: int = 64,
        score_clip: float = 5.0,
        direct_patch_skip: bool = False,
        direct_patch_gain_limit: float = 0.5,
        category_preserving_gate: bool = False,
        category_gate_max_gap: float = 1.0,
        detach_teacher: bool = True,
    ) -> None:
        super().__init__()
        if int(query_count) <= 0 or int(hidden_dim) <= 0:
            raise ValueError("query_count and hidden_dim must be positive")
        if float(score_clip) <= 0.0:
            raise ValueError("score_clip must be positive")
        if bool(direct_patch_skip) and float(direct_patch_gain_limit) <= 0.0:
            raise ValueError("direct_patch_gain_limit must be positive")
        if float(category_gate_max_gap) < 0.0:
            raise ValueError("category_gate_max_gap must be non-negative")
        self.query_count = int(query_count)
        self.score_clip = float(score_clip)
        self.direct_patch_skip = bool(direct_patch_skip)
        self.direct_patch_gain_limit = float(direct_patch_gain_limit)
        self.category_preserving_gate = bool(category_preserving_gate)
        self.category_gate_max_gap = float(category_gate_max_gap)
        self.detach_teacher = bool(detach_teacher)
        self.trunk = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, int(hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
        )
        self.output = nn.Linear(int(hidden_dim), 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        if self.direct_patch_skip:
            self.direct_patch_gain = nn.Parameter(torch.zeros((), dtype=torch.float32))
            self.register_buffer(
                "_contract_direct_patch_gain_limit",
                torch.as_tensor(self.direct_patch_gain_limit, dtype=torch.float32),
                persistent=True,
            )
        else:
            self.register_parameter("direct_patch_gain", None)
        self.register_buffer(
            "_contract_version",
            torch.as_tensor(
                U1_DIRECT_PATCH_CONTRACT_VERSION
                if self.direct_patch_skip
                else U0_PATCH_RANK_CONTRACT_VERSION,
                dtype=torch.int64,
            ),
            persistent=True,
        )
        self.register_buffer(
            "_contract_query_count",
            torch.as_tensor(self.query_count, dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "_contract_score_clip",
            torch.as_tensor(self.score_clip, dtype=torch.float32),
            persistent=True,
        )

    def trainable_parameters(self) -> tuple[nn.Parameter, ...]:
        parameters = tuple(self.trunk.parameters()) + tuple(self.output.parameters())
        if self.direct_patch_gain is not None:
            parameters += (self.direct_patch_gain,)
        return parameters

    def direct_patch_gain_value(self) -> Optional[Tensor]:
        if self.direct_patch_gain is None:
            return None
        limit = float(self.direct_patch_gain_limit)
        return limit * torch.tanh(self.direct_patch_gain / limit)

    def _apply_category_preserving_gate(
        self,
        patch_normalized: Tensor,
        teacher: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return a lexicographic (patch eligibility, teacher rank) score."""
        if self.training:
            raise RuntimeError(
                "category-preserving patch gate is inference-only; call eval()"
            )
        best_patch = patch_normalized.masked_fill(~mask, -torch.inf).amax(
            dim=1, keepdim=True
        )
        eligible = mask & (
            best_patch - patch_normalized <= self.category_gate_max_gap
        )
        if bool((~eligible.any(dim=1)).any().item()):
            raise RuntimeError("category-preserving patch gate produced an empty row")

        teacher_min = teacher.masked_fill(~mask, torch.inf).amin(
            dim=1, keepdim=True
        )
        teacher_max = teacher.masked_fill(~mask, -torch.inf).amax(
            dim=1, keepdim=True
        )
        below_teacher_min = torch.nextafter(
            teacher_min, torch.full_like(teacher_min, -torch.inf)
        )
        if not bool(torch.isfinite(below_teacher_min).all().item()):
            raise RuntimeError(
                "cannot construct a finite score below the teacher minimum"
            )
        # Masked-out queries use zero delta. Every non-eligible score is at or
        # below the representable predecessor of the lowest eligible teacher
        # score, while eligible scores remain bitwise equal to the teacher.
        teacher_delta = torch.where(mask, teacher, teacher_max) - teacher_max
        ineligible_score = below_teacher_min + teacher_delta
        rank_score = torch.where(eligible, teacher, ineligible_score)
        return rank_score, eligible

    def apply_category_preserving_gate(
        self, patch_normalized: Tensor, teacher_rank_score: Tensor,
        candidate_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Public, inference-identical gate used by post-gate score adapters."""
        if not self.category_preserving_gate:
            raise RuntimeError("category-preserving patch gate is disabled")
        return self._apply_category_preserving_gate(
            patch_normalized, teacher_rank_score, candidate_mask
        )

    @staticmethod
    def _standardize(
        values: Tensor,
        mask: Tensor,
        *,
        clip: float,
    ) -> Tensor:
        count = mask.sum(dim=1).clamp_min(1).float()
        safe = values.float().masked_fill(~mask, 0.0)
        mean = safe.sum(dim=1) / count
        centered = (values.float() - mean[:, None]).masked_fill(~mask, 0.0)
        std = (centered.square().sum(dim=1) / count).clamp_min(1e-6).sqrt()
        return (centered / std[:, None]).clamp(min=-float(clip), max=float(clip))

    def forward(
        self,
        patch_score: Tensor,
        teacher_rank_score: Tensor,
        candidate_mask: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        if patch_score.dim() != 2 or teacher_rank_score.dim() != 2:
            raise ValueError("patch_score and teacher_rank_score must have shape (B,Q)")
        if tuple(patch_score.shape) != tuple(teacher_rank_score.shape):
            raise ValueError("patch_score and teacher_rank_score must have identical shape")
        if int(patch_score.shape[1]) != self.query_count:
            raise ValueError(
                f"U0 requires exactly {self.query_count} queries, got {patch_score.shape[1]}"
            )
        if not patch_score.is_floating_point() or not teacher_rank_score.is_floating_point():
            raise TypeError("patch_score and teacher_rank_score must be floating point")
        if patch_score.device != teacher_rank_score.device:
            raise ValueError("patch_score and teacher_rank_score must share a device")
        if candidate_mask is None:
            mask = torch.ones_like(teacher_rank_score, dtype=torch.bool)
        else:
            mask = torch.as_tensor(
                candidate_mask,
                device=teacher_rank_score.device,
                dtype=torch.bool,
            )
            if tuple(mask.shape) != tuple(teacher_rank_score.shape):
                raise ValueError("candidate_mask must match the score tensors")
        if bool((~mask.any(dim=1)).any().item()):
            raise ValueError("every U0 row must contain at least one candidate")
        patch = patch_score
        teacher = (
            teacher_rank_score.detach()
            if self.detach_teacher else teacher_rank_score
        )
        if not bool(torch.isfinite(patch[mask]).all().item()):
            raise ValueError("valid patch scores must be finite")
        if not bool(torch.isfinite(teacher[mask]).all().item()):
            raise ValueError("valid teacher rank scores must be finite")

        patch_normalized = self._standardize(
            patch, mask, clip=self.score_clip
        )
        teacher_normalized = self._standardize(
            teacher, mask, clip=self.score_clip
        )
        features = torch.stack(
            (
                patch_normalized,
                teacher_normalized,
                patch_normalized * teacher_normalized,
            ),
            dim=-1,
        )
        learned_residual = self.output(self.trunk(features)).squeeze(-1)
        direct_gain = self.direct_patch_gain_value()
        if direct_gain is None:
            direct_residual = torch.zeros_like(learned_residual)
        else:
            direct_residual = (direct_gain * patch_normalized).to(
                dtype=learned_residual.dtype
            )
        continuous_residual = learned_residual + direct_residual
        continuous_residual = continuous_residual.to(
            dtype=teacher.dtype
        ).masked_fill(~mask, 0.0)
        learned_residual = learned_residual.to(dtype=teacher.dtype).masked_fill(
            ~mask, 0.0
        )
        direct_residual = direct_residual.to(dtype=teacher.dtype).masked_fill(
            ~mask, 0.0
        )
        continuous_rank_score = teacher + continuous_residual
        if self.category_preserving_gate:
            rank_score, eligible = self._apply_category_preserving_gate(
                patch_normalized, teacher, mask
            )
            residual = rank_score - teacher
        else:
            rank_score = continuous_rank_score
            residual = continuous_residual
            eligible = mask
        result = {
            "teacher_rank_score": teacher,
            "admission_standardized_score": patch_normalized,
            "patch_rank_residual": residual,
            "learned_patch_rank_residual": learned_residual,
            "direct_patch_rank_residual": direct_residual,
            "direct_patch_gain": (
                direct_gain
                if direct_gain is not None
                else teacher.new_zeros(())
            ),
            "rank_score": rank_score,
            "candidate_mask": mask,
        }
        if self.category_preserving_gate:
            result.update(
                {
                    "pre_category_gate_rank_score": continuous_rank_score,
                    "category_gate_eligible_mask": eligible,
                    "category_gate_patch_score": patch_normalized,
                }
            )
        return result


def _candidate_max_iou(
    candidate_boxes: Tensor,
    targets: Sequence[Mapping[str, Any]],
    *,
    primary_only: bool = False,
) -> Tensor:
    if candidate_boxes.dim() != 3 or int(candidate_boxes.shape[-1]) != 4:
        raise ValueError("candidate boxes must have shape (B,Q,4)")
    if len(targets) != int(candidate_boxes.shape[0]):
        raise ValueError("targets must align with candidate boxes")
    candidates = box_ops.box_cxcywh_to_xyxy(candidate_boxes.detach().float())
    result = candidates.new_zeros(candidates.shape[:2])
    for index, target in enumerate(targets):
        boxes = target.get("boxes")
        if not torch.is_tensor(boxes) or boxes.numel() == 0:
            continue
        if primary_only:
            primary = target.get("primary_instance_mask")
            if not torch.is_tensor(primary) or primary.dtype != torch.bool:
                raise ValueError(
                    "category-complete supervision requires boolean primary_instance_mask"
                )
            primary = primary.reshape(-1)
            if int(primary.numel()) != int(boxes.shape[0]):
                raise ValueError("primary_instance_mask must align with target boxes")
            boxes = boxes[primary]
            if boxes.numel() == 0:
                continue
            if int(boxes.shape[0]) != 1:
                raise ValueError("each row must retain at most one primary instance")
        target_xyxy = box_ops.box_cxcywh_to_xyxy(
            boxes.detach().to(device=result.device, dtype=torch.float32).reshape(-1, 4)
        )
        iou, _ = box_ops.box_iou(candidates[index], target_xyxy)
        if iou.numel():
            result[index] = iou.max(dim=1).values
    return result


def _category_complete_patch_margin_loss(
    patch_score: Tensor,
    candidate_boxes: Tensor,
    targets: Sequence[Mapping[str, Any]],
    *,
    positive_iou_threshold: float,
    negative_iou_threshold: float,
    margin: float,
    temperature: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Require one high patch score per same-category instance."""
    if patch_score.dim() == 3 and int(patch_score.shape[-1]) == 1:
        patch_score = patch_score[..., 0]
    if patch_score.dim() != 2 or tuple(patch_score.shape) != tuple(candidate_boxes.shape[:2]):
        raise ValueError("category-complete patch scores must align with candidate boxes")
    candidates = box_ops.box_cxcywh_to_xyxy(candidate_boxes.detach().float())
    row_losses = []
    valid_instance_count = 0
    skipped_instance_count = 0
    tau = float(temperature)
    for row_index, target in enumerate(targets):
        marker = target.get("stage_b_u2_category_complete")
        if not (
            torch.is_tensor(marker)
            and marker.dtype == torch.bool
            and marker.numel() == 1
            and bool(marker.reshape(-1)[0].item())
        ):
            raise ValueError(
                "category-complete patch loss requires an exact true dataset marker"
            )
        boxes = target.get("boxes")
        if not torch.is_tensor(boxes) or boxes.numel() == 0:
            continue
        target_xyxy = box_ops.box_cxcywh_to_xyxy(
            boxes.detach().to(device=candidates.device, dtype=torch.float32).reshape(-1, 4)
        )
        iou, _ = box_ops.box_iou(candidates[row_index], target_xyxy)
        category_max_iou = iou.max(dim=1).values
        negative_mask = category_max_iou < float(negative_iou_threshold)
        if not bool(negative_mask.any().item()):
            skipped_instance_count += int(target_xyxy.shape[0])
            continue
        scores = patch_score[row_index].float()
        hard_negative = scores.masked_fill(~negative_mask, -torch.inf).max()
        positive_mask = iou >= float(positive_iou_threshold)
        positive_counts = positive_mask.sum(dim=0)
        valid_instances = positive_counts > 0
        valid_count = int(valid_instances.sum().item())
        valid_instance_count += valid_count
        skipped_instance_count += int(target_xyxy.shape[0]) - valid_count
        if valid_count:
            valid_positive_mask = positive_mask[:, valid_instances]
            valid_positive_counts = positive_counts[valid_instances]
            smooth_positive = tau * (
                torch.logsumexp(
                    (scores[:, None] / tau).masked_fill(
                        ~valid_positive_mask, -torch.inf
                    ),
                    dim=0,
                )
                - valid_positive_counts.float().log()
            )
            instance_loss = tau * torch.nn.functional.softplus(
                (float(margin) - smooth_positive + hard_negative) / tau
            )
            row_losses.append(instance_loss.mean())
    if row_losses:
        loss = torch.stack(row_losses).mean()
    else:
        loss = patch_score.float().sum() * 0.0
    return (
        loss,
        patch_score.new_tensor(float(valid_instance_count)).detach(),
        patch_score.new_tensor(float(skipped_instance_count)).detach(),
    )


class StageBU0PatchRankCriterion(nn.Module):
    """Improve U0 over the sealed R100 teacher without touching confidence."""

    def __init__(
        self,
        *,
        weight: float = 1.0,
        iou_threshold: float = 0.5,
        fix_margin: float = 0.05,
        preserve_margin: float = 0.02,
        temperature: float = 0.1,
        residual_weight: float = 1e-3,
        category_complete_supervision: bool = False,
        category_loss_weight: float = 0.0,
        category_negative_iou_threshold: float = 0.3,
        category_margin: float = 0.1,
        target_preserve_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if float(weight) <= 0.0:
            raise ValueError("U0 criterion weight must be positive")
        self.weight_dict = {"loss_stage_b_u0_patch_rank": float(weight)}
        self.iou_threshold = float(iou_threshold)
        self.fix_margin = float(fix_margin)
        self.preserve_margin = float(preserve_margin)
        self.temperature = float(temperature)
        self.residual_weight = float(residual_weight)
        self.category_complete_supervision = bool(category_complete_supervision)
        self.category_loss_weight = float(category_loss_weight)
        self.category_negative_iou_threshold = float(
            category_negative_iou_threshold
        )
        self.category_margin = float(category_margin)
        self.target_preserve_weight = float(target_preserve_weight)
        if self.category_complete_supervision and self.category_loss_weight <= 0.0:
            raise ValueError("category-complete supervision requires a positive loss weight")
        if self.target_preserve_weight < 0.0:
            raise ValueError("target preserve weight must be non-negative")
        if not 0.0 <= self.category_negative_iou_threshold < self.iou_threshold:
            raise ValueError(
                "category negative IoU threshold must be non-negative and below the positive threshold"
            )
        self.register_buffer(
            "criterion_contract_version",
            torch.as_tensor(U0_PATCH_RANK_CONTRACT_VERSION, dtype=torch.int64),
            persistent=True,
        )

    def forward(
        self,
        outputs: Mapping[str, Any],
        targets: Sequence[Mapping[str, Any]],
    ) -> dict[str, Tensor]:
        rank_score = outputs.get("stage_b_u0_rank_score")
        teacher_score = outputs.get("stage_b_u0_teacher_rank_score")
        residual = outputs.get("stage_b_u0_patch_rank_residual")
        boxes = outputs.get("pred_boxes")
        if not all(
            torch.is_tensor(value)
            for value in (rank_score, teacher_score, residual, boxes)
        ):
            raise KeyError("U0 rank training requires rank/teacher/residual/box outputs")
        primary_iou = _candidate_max_iou(
            boxes, targets, primary_only=self.category_complete_supervision
        )
        primary_result = baseline_preserving_top1_rank_loss(
            rank_score,
            teacher_score,
            residual,
            primary_iou,
            iou_threshold=self.iou_threshold,
            fix_margin=self.fix_margin,
            preserve_margin=self.preserve_margin,
            temperature=self.temperature,
            residual_weight=self.residual_weight,
        )
        category_result = primary_result
        total_loss = primary_result.loss
        category_loss = rank_score.new_zeros(())
        category_valid_instances = rank_score.new_zeros(())
        category_skipped_instances = rank_score.new_zeros(())
        if self.category_complete_supervision:
            category_iou = _candidate_max_iou(boxes, targets)
            category_result = baseline_preserving_top1_rank_loss(
                rank_score,
                teacher_score,
                residual,
                category_iou,
                iou_threshold=self.iou_threshold,
                fix_margin=self.fix_margin,
                preserve_margin=self.preserve_margin,
                temperature=self.temperature,
                residual_weight=self.residual_weight,
            )
            patch_score = outputs.get("pred_logits_patch")
            if not torch.is_tensor(patch_score):
                raise KeyError(
                    "category-complete U2 training requires raw pred_logits_patch"
                )
            (
                category_loss,
                category_valid_instances,
                category_skipped_instances,
            ) = _category_complete_patch_margin_loss(
                patch_score,
                boxes,
                targets,
                positive_iou_threshold=self.iou_threshold,
                negative_iou_threshold=self.category_negative_iou_threshold,
                margin=self.category_margin,
                temperature=self.temperature,
            )
            total_loss = (
                category_result.margin_loss
                + self.residual_weight * category_result.residual_loss
                + self.target_preserve_weight * primary_result.preserve_loss
                + self.category_loss_weight * category_loss
            )
        losses = {
            "loss_stage_b_u0_patch_rank": total_loss,
            "stage_b_u0_rank_margin_loss": primary_result.margin_loss.detach(),
            "stage_b_u0_rank_fix_loss": primary_result.fix_loss.detach(),
            "stage_b_u0_rank_preserve_loss": primary_result.preserve_loss.detach(),
            "stage_b_u0_rank_residual_l2": primary_result.residual_loss.detach(),
            "stage_b_u0_valid_rank_rows": primary_result.valid_rows,
            "stage_b_u0_rank_fix_rows": primary_result.fix_rows,
            "stage_b_u0_rank_preserve_rows": primary_result.preserve_rows,
            "stage_b_u0_rank_rows_no_positive": primary_result.rows_no_positive,
            "stage_b_u0_teacher_correct": primary_result.base_correct,
            "stage_b_u0_adapted_correct": primary_result.adapted_correct,
            "stage_b_u0_wrong_fixed": primary_result.wrong_fixed,
            "stage_b_u0_correct_regressed": primary_result.correct_regressed,
            "stage_b_u2_category_patch_loss": category_loss.detach(),
            "stage_b_u2_category_valid_instances": category_valid_instances,
            "stage_b_u2_category_skipped_instances": category_skipped_instances,
            "stage_b_u2_category_rank_margin_loss": category_result.margin_loss.detach(),
            "stage_b_u2_category_rank_fix_loss": category_result.fix_loss.detach(),
            "stage_b_u2_category_rank_preserve_loss": category_result.preserve_loss.detach(),
            "stage_b_u2_category_teacher_correct": category_result.base_correct,
            "stage_b_u2_category_adapted_correct": category_result.adapted_correct,
            "stage_b_u2_category_wrong_fixed": category_result.wrong_fixed,
            "stage_b_u2_category_correct_regressed": category_result.correct_regressed,
            "stage_b_u2_target_preserve_loss": primary_result.preserve_loss.detach(),
        }
        direct_gain = outputs.get("stage_b_u1_direct_patch_gain")
        if torch.is_tensor(direct_gain):
            losses["stage_b_u1_direct_patch_gain"] = direct_gain.detach().float()
        return losses


__all__ = [
    "StageBU0PatchRankAdapter",
    "StageBU0PatchRankCriterion",
    "U0_PATCH_RANK_CONTRACT_VERSION",
    "U1_DIRECT_PATCH_CONTRACT_VERSION",
    "U0_INITIALIZER_SCHEMA",
    "U1_DIRECT_PATCH_INITIALIZER_SCHEMA",
    "U1_DIRECT_PATCH_ADDED_KEYS",
    "U1_DIRECT_PATCH_REPLACED_KEYS",
    "U0_PATCH_BACKBONE_PREFIX",
    "U0_PATCH_SOURCE_KEYS",
    "U0_SEALED_TEACHER_ARCHITECTURE_FIELDS",
    "U0_TEACHER_FUNCTIONAL_FIELDS",
    "stage_b_u0_tensor_state_sha256",
    "validate_stage_b_u0_initializer_payload",
    "validate_stage_b_u1_direct_patch_initializer_payload",
    "validate_stage_b_u0_patch_rank_checkpoint",
]
