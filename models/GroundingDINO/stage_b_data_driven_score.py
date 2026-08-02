"""Teacher-output-free Stage-B scoring heads and direct data losses.

The frozen GroundingDINO base receives only a canonical category caption and
produces candidate boxes/query features.  Full expressions are encoded on a
separate frozen text path.  A selectable rank branch handles within-image
ordering while a parameter-disjoint absolute token head handles cross-sample
confidence. Patch scores are trained only as category evidence and are fused
with text rank by an inference-only, parameter-free eligibility gate.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from difflib import SequenceMatcher
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from groundingdino.util import box_ops

from .patch_only_criterion import _sigmoid_focal_loss_no_reduce
from .stage_b_gdino_score_adapter import (
    aggregate_gdino_full_expression_score,
    fpr95_global_max_surrogate,
    multi_positive_listwise_rank_loss,
)


DATA_DRIVEN_SCORE_CONTRACT_VERSION = 1
DATA_DRIVEN_RELATIONAL_SCORE_CONTRACT_VERSION = 3
DATA_DRIVEN_INITIALIZER_SCHEMA = "pivot.stageb.data_driven_initializer/v1"
DATA_DRIVEN_RELATIONAL_INITIALIZER_SCHEMA = (
    "pivot.stageb.data_driven_relational_initializer/v2"
)
DATA_DRIVEN_CONFIDENCE_INITIALIZER_SCHEMA = (
    "pivot.stageb.data_driven_confidence_initializer/v1"
)
DATA_DRIVEN_ROLE_ROUTED_INITIALIZER_SCHEMA = (
    "pivot.stageb.data_driven.role_routed_model_initializer/v1"
)
DATA_DRIVEN_TRAIN_MODES = ("rank_patch_only", "confidence_pair")
DATA_DRIVEN_RANK_ARCHITECTURES = ("absolute_token", "relational_v1")
DATA_DRIVEN_RANK_SUPERVISION_ALL_NONPOSITIVE = "all_nonpositive_negative_v1"
DATA_DRIVEN_RANK_SUPERVISION_SAME_CATEGORY = (
    "primary_vs_same_category_aux_v1"
)
DATA_DRIVEN_RANK_SUPERVISION_SAME_CATEGORY_GAP3_COVERAGE = (
    "primary_vs_same_category_aux_plus_gap3_coverage_v1"
)
DATA_DRIVEN_RANK_SUPERVISION_OFFICIAL_ASSIGNMENT = (
    "official_same_image_same_category_assignment_v1"
)
DATA_DRIVEN_RANK_SUPERVISION_ROLE_ROUTED_ASSIGNMENT = (
    "role_routed_official_assignment_top1_v1"
)
DATA_DRIVEN_RANK_SUPERVISION_ROLE_ROUTED_ALL_NONOWNED = (
    "role_routed_official_assignment_all_exclusive_nonowned_v2"
)
DATA_DRIVEN_RANK_SUPERVISION_MODES = (
    DATA_DRIVEN_RANK_SUPERVISION_ALL_NONPOSITIVE,
    DATA_DRIVEN_RANK_SUPERVISION_SAME_CATEGORY,
    DATA_DRIVEN_RANK_SUPERVISION_SAME_CATEGORY_GAP3_COVERAGE,
    DATA_DRIVEN_RANK_SUPERVISION_OFFICIAL_ASSIGNMENT,
    DATA_DRIVEN_RANK_SUPERVISION_ROLE_ROUTED_ASSIGNMENT,
    DATA_DRIVEN_RANK_SUPERVISION_ROLE_ROUTED_ALL_NONOWNED,
)
_DATA_DRIVEN_ROLE_ROUTED_SUPERVISIONS = frozenset(
    {
        DATA_DRIVEN_RANK_SUPERVISION_ROLE_ROUTED_ASSIGNMENT,
        DATA_DRIVEN_RANK_SUPERVISION_ROLE_ROUTED_ALL_NONOWNED,
    }
)
DATA_DRIVEN_CRITERION_CONTRACT_VERSION = 4
DATA_DRIVEN_ROLE_ROUTED_CRITERION_CONTRACT_VERSION = 18
DATA_DRIVEN_PATCH_DROP_ANCHOR_GLOBAL_MAX = "global_max_positive_v1"
DATA_DRIVEN_PATCH_DROP_ANCHOR_INSTANCE_BALANCED = (
    "reachable_instance_best_mean_straight_through_v1"
)
DATA_DRIVEN_PATCH_DROP_ANCHOR_POLICIES = (
    DATA_DRIVEN_PATCH_DROP_ANCHOR_GLOBAL_MAX,
    DATA_DRIVEN_PATCH_DROP_ANCHOR_INSTANCE_BALANCED,
)
DATA_DRIVEN_GLOBAL_TN_SCOPE = "image_global_topk_verified"
_DATA_DRIVEN_CURRENT_CRITERION_CONTRACTS = {
    DATA_DRIVEN_RANK_SUPERVISION_ALL_NONPOSITIVE: (4, 1),
    DATA_DRIVEN_RANK_SUPERVISION_SAME_CATEGORY: (4, 2),
    DATA_DRIVEN_RANK_SUPERVISION_SAME_CATEGORY_GAP3_COVERAGE: (4, 3),
    DATA_DRIVEN_RANK_SUPERVISION_OFFICIAL_ASSIGNMENT: (4, 4),
    DATA_DRIVEN_RANK_SUPERVISION_ROLE_ROUTED_ASSIGNMENT: (13, 5),
    DATA_DRIVEN_RANK_SUPERVISION_ROLE_ROUTED_ALL_NONOWNED: (18, 6),
}
_DATA_DRIVEN_EVAL_CRITERION_CONTRACTS = {
    DATA_DRIVEN_RANK_SUPERVISION_ALL_NONPOSITIVE: {(1, None), (4, 1)},
    DATA_DRIVEN_RANK_SUPERVISION_SAME_CATEGORY: {(2, 2), (4, 2)},
    DATA_DRIVEN_RANK_SUPERVISION_SAME_CATEGORY_GAP3_COVERAGE: {
        (2, 3),
        (4, 3),
    },
    DATA_DRIVEN_RANK_SUPERVISION_OFFICIAL_ASSIGNMENT: {(4, 4)},
    DATA_DRIVEN_RANK_SUPERVISION_ROLE_ROUTED_ASSIGNMENT: {
        (9, 5),
        (10, 5),
        (11, 5),
        (12, 5),
        (13, 5),
    },
    DATA_DRIVEN_RANK_SUPERVISION_ROLE_ROUTED_ALL_NONOWNED: {
        (14, 6),
        (15, 6),
        (16, 6),
        (17, 6),
        (18, 6),
    },
}
DATA_DRIVEN_RELATIONAL_IMAGE_POOL_POLICY = (
    "valid_extent_masked_adaptive_avg_v1"
)
_TRACE_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _is_device_autocast_enabled(device_type: str) -> bool:
    """Query device-specific autocast across supported PyTorch APIs."""
    try:
        return bool(torch.is_autocast_enabled(device_type))
    except TypeError:
        if device_type == "cuda":
            return bool(torch.is_autocast_enabled())
        if device_type == "cpu" and hasattr(torch, "is_autocast_cpu_enabled"):
            return bool(torch.is_autocast_cpu_enabled())
        return False


def groundingdino_raw_dot_phrase_geometry(
    query_hs: Tensor,
    encoded_text: Tensor,
    text_token_mask: Tensor,
    expression_token_mask: Tensor,
    *,
    max_text_len: int,
) -> dict[str, Tensor]:
    """Return fixed-query GroundingDINO dot-product phrase scores.

    This reproduces :class:`ContrastiveEmbed` token geometry, then applies the
    authoritative full-expression mean-sigmoid aggregation.  It does not imply
    equivalence to a full-expression GroundingDINO forward: that also depends on
    which caption conditioned the image encoder and decoder query features.
    """
    if not torch.is_tensor(query_hs) or not torch.is_tensor(encoded_text):
        raise TypeError("query_hs and encoded_text must be tensors")
    if query_hs.dim() != 3 or encoded_text.dim() != 3:
        raise ValueError("query_hs and encoded_text must have shape (B,N,D)")
    if not query_hs.is_floating_point() or not encoded_text.is_floating_point():
        raise TypeError("query_hs and encoded_text must be floating point")
    if query_hs.device != encoded_text.device:
        raise ValueError("query_hs and encoded_text must share a device")
    if query_hs.dtype != encoded_text.dtype and not _is_device_autocast_enabled(
        query_hs.device.type
    ):
        raise ValueError(
            "mixed query/text dtypes require autocast on their shared device"
        )
    batch_size, query_count, hidden_dim = map(int, query_hs.shape)
    text_batch, token_count, text_hidden_dim = map(int, encoded_text.shape)
    if (
        batch_size <= 0
        or query_count <= 0
        or hidden_dim <= 0
        or token_count <= 0
    ):
        raise ValueError("query/text geometry inputs must be non-empty")
    if text_batch != batch_size or text_hidden_dim != hidden_dim:
        raise ValueError("query_hs and encoded_text batches/dimensions must align")
    if isinstance(max_text_len, bool) or not isinstance(max_text_len, int):
        raise TypeError("max_text_len must be an integer")
    if max_text_len < token_count:
        raise ValueError("max_text_len cannot be shorter than encoded_text")
    if not bool(torch.isfinite(query_hs).all().item()) or not bool(
        torch.isfinite(encoded_text).all().item()
    ):
        raise ValueError("query_hs and encoded_text must contain only finite values")

    for name, mask in (
        ("text_token_mask", text_token_mask),
        ("expression_token_mask", expression_token_mask),
    ):
        if not torch.is_tensor(mask) or mask.dtype != torch.bool:
            raise TypeError(f"{name} must be a boolean tensor")
        if mask.device != encoded_text.device:
            raise ValueError(f"{name} must share encoded_text's device")
        if tuple(mask.shape) != (batch_size, token_count):
            raise ValueError(f"{name} must have shape (B,T)")
    if bool((~text_token_mask.any(dim=1)).any().item()):
        raise ValueError("every text row must contain a valid token")
    if bool((~expression_token_mask.any(dim=1)).any().item()):
        raise ValueError("every expression must contain a scored token")
    if bool((expression_token_mask & ~text_token_mask).any().item()):
        raise ValueError("expression_token_mask must be a subset of text_token_mask")

    raw_logits = query_hs @ encoded_text.transpose(-1, -2)
    if not bool(torch.isfinite(raw_logits).all().item()):
        raise ValueError("raw query-text dot products must be finite")
    raw_logits = raw_logits.masked_fill(
        ~text_token_mask[:, None, :], float("-inf")
    )
    # ContrastiveEmbed allocates this padded carrier in the default floating
    # dtype, including when the raw matmul itself is autocast.
    token_logits = torch.full(
        (batch_size, query_count, max_text_len),
        float("-inf"),
        device=raw_logits.device,
    )
    token_logits[..., :token_count] = raw_logits
    padded_expression_mask = torch.zeros(
        (batch_size, max_text_len),
        device=encoded_text.device,
        dtype=torch.bool,
    )
    padded_expression_mask[:, :token_count] = expression_token_mask
    score = aggregate_gdino_full_expression_score(
        token_logits, padded_expression_mask
    )
    if tuple(score.shape) != (batch_size, query_count) or not bool(
        torch.isfinite(score).all().item()
    ):
        raise RuntimeError("GroundingDINO phrase aggregation produced invalid scores")
    return {
        "token_logits": token_logits,
        "expression_token_mask": padded_expression_mask,
        "score": score,
    }


def _data_driven_standardized_patch_score(
    patch_score: Tensor,
    candidate_mask: Tensor,
    *,
    clip: float,
    detach_statistics: bool,
    straight_through_clip: bool,
) -> Tensor:
    count = candidate_mask.sum(dim=1).clamp_min(1).float()
    score = patch_score.float()
    safe = score.masked_fill(~candidate_mask, 0.0)
    mean = safe.sum(dim=1) / count
    centered_for_std = (score - mean[:, None]).masked_fill(
        ~candidate_mask, 0.0
    )
    std = (
        centered_for_std.square().sum(dim=1) / count
    ).clamp_min(1e-6).sqrt()
    if detach_statistics:
        mean = mean.detach()
        std = std.detach()
    centered = (score - mean[:, None]).masked_fill(~candidate_mask, 0.0)
    raw_standardized = centered / std[:, None]
    clipped = raw_standardized.clamp(
        min=-float(clip), max=float(clip)
    )
    if straight_through_clip:
        clipped = _ExactForwardStraightThroughClamp.apply(
            raw_standardized,
            -float(clip),
            float(clip),
        )
    return clipped


class _ExactForwardStraightThroughClamp(torch.autograd.Function):
    """Use the deployment clamp bit-for-bit while differentiating as identity."""

    @staticmethod
    def forward(ctx, value: Tensor, minimum: float, maximum: float) -> Tensor:
        del ctx
        return value.clamp(min=float(minimum), max=float(maximum))

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        del ctx
        return grad_output, None, None


def data_driven_category_gate_mask(
    patch_score: Tensor,
    candidate_mask: Tensor,
    *,
    max_gap: float,
    clip: float,
) -> tuple[Tensor, Tensor]:
    """Return the exact inference Gap mask and standardized patch score."""
    if patch_score.dim() == 3:
        if int(patch_score.shape[-1]) != 1:
            raise ValueError("patch score must have a singleton slot dimension")
        patch_score = patch_score[..., 0]
    if patch_score.dim() != 2 or not patch_score.is_floating_point():
        raise ValueError("patch score must be a floating (B,Q) tensor")
    if (
        candidate_mask.dtype != torch.bool
        or tuple(candidate_mask.shape) != tuple(patch_score.shape)
    ):
        raise ValueError("candidate mask must be boolean and patch-aligned")
    if not bool(torch.isfinite(patch_score).all().item()):
        raise ValueError("patch score must contain only finite values")
    if bool((~candidate_mask.any(dim=1)).any().item()):
        raise ValueError("every category-gate row requires a candidate")
    if (
        not math.isfinite(float(max_gap))
        or float(max_gap) < 0.0
        or not math.isfinite(float(clip))
        or float(clip) <= 0.0
    ):
        raise ValueError("category-gate gap/clip must be finite and valid")

    standardized = _data_driven_standardized_patch_score(
        patch_score,
        candidate_mask,
        clip=float(clip),
        detach_statistics=False,
        straight_through_clip=False,
    )
    best = standardized.masked_fill(~candidate_mask, -torch.inf).amax(
        dim=1, keepdim=True
    )
    eligible = candidate_mask & (best - standardized <= float(max_gap))
    if bool((~eligible.any(dim=1)).any().item()):
        raise RuntimeError("data-driven category gate produced an empty row")
    return eligible, standardized


def data_driven_tensor_state_sha256(
    state: Mapping[str, Any], keys: Sequence[str]
) -> str:
    selected = sorted(set(str(key) for key in keys))
    if not selected:
        raise ValueError("cannot hash an empty data-driven tensor selection")
    digest = hashlib.sha256()
    for key in selected:
        value = state.get(key)
        if not torch.is_tensor(value):
            raise ValueError(f"data-driven model state {key!r} is missing")
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


def validate_data_driven_initializer_payload(
    expected_model: nn.Module,
    payload: Mapping[str, Any],
    *,
    checkpoint_label: str,
) -> None:
    """Validate the deterministic b58-only initializer and its role partition."""
    if not isinstance(payload, Mapping) or set(payload) != {
        "model",
        "data_driven_initializer",
    }:
        raise ValueError(
            f"{checkpoint_label}: data-driven initializer top-level keys drifted"
        )
    state = payload.get("model")
    contract = payload.get("data_driven_initializer")
    if not isinstance(state, Mapping) or not isinstance(contract, Mapping):
        raise ValueError(f"{checkpoint_label}: initializer payload is malformed")
    if contract.get("schema") != DATA_DRIVEN_INITIALIZER_SCHEMA:
        raise ValueError(f"{checkpoint_label}: initializer schema drifted")
    expected = expected_model.state_dict()
    if set(state) != set(expected):
        raise ValueError(
            f"{checkpoint_label}: initializer model key coverage drifted"
        )
    for key, wanted in expected.items():
        value = state.get(key)
        if (
            not torch.is_tensor(value)
            or value.dtype != wanted.dtype
            or tuple(value.shape) != tuple(wanted.shape)
        ):
            raise ValueError(
                f"{checkpoint_label}: initializer tensor shape/dtype drift at {key}"
            )
    roles = contract.get("role_keys")
    expected_roles = {
        "b58_base",
        "shared_backbone_alias",
        "random_patch_projection",
        "random_absolute_heads",
    }
    if not isinstance(roles, Mapping) or set(roles) != expected_roles:
        raise ValueError(f"{checkpoint_label}: initializer roles drifted")
    normalized = {}
    for role in expected_roles:
        values = roles.get(role)
        if not isinstance(values, list) or not values:
            raise ValueError(f"{checkpoint_label}: initializer role {role} is empty")
        normalized[role] = [str(key) for key in values]
    flattened = [key for keys in normalized.values() for key in keys]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(state):
        raise ValueError(
            f"{checkpoint_label}: initializer roles are not an exact partition"
        )
    if not all(
        key.startswith("patch_encoder.backbone.")
        for key in normalized["shared_backbone_alias"]
    ):
        raise ValueError(f"{checkpoint_label}: shared backbone role drifted")
    if not all(
        key.startswith("stage_b_data_driven_score_heads.")
        for key in normalized["random_absolute_heads"]
    ):
        raise ValueError(f"{checkpoint_label}: absolute-head role drifted")
    for role in expected_roles:
        expected_hash = contract.get(f"{role}_tensor_sha256")
        observed_hash = data_driven_tensor_state_sha256(state, normalized[role])
        if expected_hash != observed_hash:
            raise ValueError(
                f"{checkpoint_label}: initializer {role} tensor hash drifted"
            )
    if contract.get("full_model_tensor_sha256") != data_driven_tensor_state_sha256(
        state, sorted(state)
    ):
        raise ValueError(f"{checkpoint_label}: full-model tensor hash drifted")
    invariants = contract.get("invariants")
    required = {
        "b58_is_only_checkpoint_source",
        "no_r100_p50_u0_or_stagea_tensor_source",
        "canonical_query_and_full_text_heads_are_separate",
        "rank_and_confidence_parameters_are_disjoint",
        "patch_backbone_aliases_b58",
    }
    if not isinstance(invariants, Mapping) or any(
        invariants.get(key) is not True for key in required
    ):
        raise ValueError(f"{checkpoint_label}: initializer invariants drifted")


def validate_data_driven_role_routed_initializer_payload(
    expected_model: nn.Module,
    payload: Mapping[str, Any],
    *,
    checkpoint_label: str,
    expected_source_checkpoint_sha256: str,
    expected_a0_initializer_sha256: str,
    expected_source_optimizer_updates: int,
) -> None:
    """Validate the model-only clean-DD1 continuation used by role routing."""
    if isinstance(payload, Mapping) and set(payload) == {
        "model",
        "data_driven_patch_residual_initializer",
    }:
        from .stage_b_data_driven_patch_residual import (
            validate_data_driven_patch_residual_initializer_payload,
        )

        validate_data_driven_patch_residual_initializer_payload(
            expected_model,
            payload,
            checkpoint_label=checkpoint_label,
            expected_source_checkpoint_sha256=(
                expected_source_checkpoint_sha256
            ),
            expected_a0_initializer_sha256=expected_a0_initializer_sha256,
            expected_source_optimizer_updates=expected_source_optimizer_updates,
        )
        return
    if not isinstance(payload, Mapping) or set(payload) != {
        "model",
        "data_driven_role_routed_initializer",
    }:
        raise ValueError(
            f"{checkpoint_label}: role-routed initializer top-level keys drifted"
        )
    state = payload.get("model")
    contract = payload.get("data_driven_role_routed_initializer")
    if not isinstance(state, Mapping) or not isinstance(contract, Mapping):
        raise ValueError(f"{checkpoint_label}: role-routed initializer is malformed")
    if contract.get("schema") != DATA_DRIVEN_ROLE_ROUTED_INITIALIZER_SCHEMA:
        raise ValueError(f"{checkpoint_label}: role-routed initializer schema drifted")

    expected = expected_model.state_dict()
    if set(state) != set(expected):
        raise ValueError(
            f"{checkpoint_label}: role-routed initializer model coverage drifted"
        )
    for key, wanted in expected.items():
        value = state.get(key)
        if (
            not torch.is_tensor(value)
            or value.dtype != wanted.dtype
            or tuple(value.shape) != tuple(wanted.shape)
        ):
            raise ValueError(
                f"{checkpoint_label}: role-routed tensor shape/dtype drift at {key}"
            )

    heads = getattr(expected_model, "stage_b_data_driven_score_heads", None)
    if not (
        heads is not None
        and getattr(heads, "rank_architecture", None) == "absolute_token"
        and contract.get("architecture")
        == {
            "rank_architecture": "absolute_token",
            "head_init_seed": int(heads.head_init_seed),
            "enable_patch_branch": bool(
                getattr(expected_model, "enable_patch_branch", False)
            ),
        }
    ):
        raise ValueError(
            f"{checkpoint_label}: role-routed initializer architecture drifted"
        )
    source_checkpoint = contract.get("source_checkpoint")
    source_a0 = contract.get("source_a0_initializer")
    if not (
        isinstance(expected_source_checkpoint_sha256, str)
        and re.fullmatch(r"[0-9a-f]{64}", expected_source_checkpoint_sha256)
        and isinstance(expected_a0_initializer_sha256, str)
        and re.fullmatch(r"[0-9a-f]{64}", expected_a0_initializer_sha256)
        and isinstance(expected_source_optimizer_updates, int)
        and not isinstance(expected_source_optimizer_updates, bool)
        and expected_source_optimizer_updates > 0
        and isinstance(source_checkpoint, Mapping)
        and source_checkpoint.get("sha256")
        == expected_source_checkpoint_sha256
        and isinstance(source_a0, Mapping)
        and source_a0.get("sha256") == expected_a0_initializer_sha256
        and contract.get("source_optimizer_updates")
        == expected_source_optimizer_updates
        and contract.get("source_checkpoint_reason") == "max_train_iters"
        and contract.get("source_criterion_contract_version") == 4
        and contract.get("source_rank_supervision_contract_id") == 1
        and contract.get("source_rank_supervision")
        == DATA_DRIVEN_RANK_SUPERVISION_ALL_NONPOSITIVE
    ):
        raise ValueError(
            f"{checkpoint_label}: role-routed source lineage drifted"
        )

    roles = contract.get("role_keys")
    counts = contract.get("role_key_counts")
    expected_roles = {
        "source_changed_rank",
        "source_changed_patch",
        "source_frozen_confidence",
        "source_unchanged_other",
    }
    if not (
        isinstance(roles, Mapping)
        and set(roles) == expected_roles
        and isinstance(counts, Mapping)
        and set(counts) == expected_roles
    ):
        raise ValueError(f"{checkpoint_label}: role-routed roles drifted")
    normalized: dict[str, list[str]] = {}
    for role in expected_roles:
        keys = roles.get(role)
        if not isinstance(keys, list) or not keys or counts.get(role) != len(keys):
            raise ValueError(
                f"{checkpoint_label}: role-routed role {role} drifted"
            )
        normalized[role] = [str(key) for key in keys]
    flattened = [key for keys in normalized.values() for key in keys]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(state):
        raise ValueError(
            f"{checkpoint_label}: role-routed roles are not an exact partition"
        )
    if not all(
        key.startswith("stage_b_data_driven_score_heads.rank_branch.")
        for key in normalized["source_changed_rank"]
    ):
        raise ValueError(f"{checkpoint_label}: role-routed rank role drifted")
    expected_patch_keys = {
        "patch_logit_scale",
        "patch_encoder.input_proj.0.weight",
        "patch_encoder.input_proj.0.bias",
        "patch_encoder.input_proj.1.weight",
        "patch_encoder.input_proj.1.bias",
        "patch_encoder.norm.weight",
        "patch_encoder.norm.bias",
        "query_proj_for_patch.weight",
        "query_proj_for_patch.bias",
    }
    if set(normalized["source_changed_patch"]) != expected_patch_keys:
        raise ValueError(f"{checkpoint_label}: role-routed patch role drifted")
    if not all(
        key.startswith(
            (
                "stage_b_data_driven_score_heads.confidence_branch.",
                "stage_b_data_driven_score_heads.confidence_gate.",
            )
        )
        for key in normalized["source_frozen_confidence"]
    ):
        raise ValueError(
            f"{checkpoint_label}: role-routed confidence role drifted"
        )
    for role, keys in normalized.items():
        if contract.get(f"{role}_tensor_sha256") != (
            data_driven_tensor_state_sha256(state, keys)
        ):
            raise ValueError(
                f"{checkpoint_label}: role-routed role hash drifted for {role}"
            )
    if contract.get("full_model_tensor_sha256") != data_driven_tensor_state_sha256(
        state, sorted(state)
    ):
        raise ValueError(
            f"{checkpoint_label}: role-routed full-model tensor hash drifted"
        )
    invariants = contract.get("invariants")
    required_invariants = {
        "source_is_clean_DD1_stage_b_data_only_u1000",
        "source_has_no_teacher_adapter_or_old_winner_tensor_route",
        "output_copies_only_source_model_tensors",
        "optimizer_scheduler_scaler_rng_and_old_criterion_are_excluded",
        "only_rank_patch_and_deployment_inert_scale_changed_from_A0",
        "confidence_and_backbone_remain_bitwise_A0",
        "role_routed_training_starts_with_fresh_optimizer_and_v5_criterion",
    }
    if not isinstance(invariants, Mapping) or any(
        invariants.get(key) is not True for key in required_invariants
    ):
        raise ValueError(
            f"{checkpoint_label}: role-routed initializer invariants drifted"
        )


def validate_data_driven_relational_initializer_payload(
    expected_model: nn.Module,
    payload: Mapping[str, Any],
    *,
    checkpoint_label: str,
) -> None:
    """Validate a b58-only relational-rank initializer and its exact roles."""
    if not isinstance(payload, Mapping) or set(payload) != {
        "model",
        "data_driven_relational_initializer",
    }:
        raise ValueError(
            f"{checkpoint_label}: relational initializer top-level keys drifted"
        )
    state = payload.get("model")
    contract = payload.get("data_driven_relational_initializer")
    if not isinstance(state, Mapping) or not isinstance(contract, Mapping):
        raise ValueError(f"{checkpoint_label}: relational initializer is malformed")
    if contract.get("schema") != DATA_DRIVEN_RELATIONAL_INITIALIZER_SCHEMA:
        raise ValueError(f"{checkpoint_label}: relational initializer schema drifted")

    expected = expected_model.state_dict()
    if set(state) != set(expected):
        raise ValueError(
            f"{checkpoint_label}: relational initializer model coverage drifted"
        )
    for key, wanted in expected.items():
        value = state.get(key)
        if (
            not torch.is_tensor(value)
            or value.dtype != wanted.dtype
            or tuple(value.shape) != tuple(wanted.shape)
        ):
            raise ValueError(
                f"{checkpoint_label}: relational tensor shape/dtype drift at {key}"
            )

    heads = getattr(expected_model, "stage_b_data_driven_score_heads", None)
    if heads is None or getattr(heads, "rank_architecture", None) != "relational_v1":
        raise ValueError(
            f"{checkpoint_label}: expected model is not relational_v1"
        )
    rank = heads.rank_branch
    expected_architecture = {
        "hidden_dim": int(heads.hidden_dim),
        "num_queries": int(getattr(expected_model, "num_queries")),
        "rank_architecture": "relational_v1",
        "rank_dim": int(heads.rank_dim),
        "rank_num_heads": int(rank.num_heads),
        "rank_image_level_policy": str(rank.image_level_policy),
        "rank_image_levels": int(rank.image_levels),
        "rank_image_pool_size": int(rank.image_pool_size),
        "rank_image_pool_policy": str(rank.image_pool_policy),
        "rank_box_fourier_bands": int(rank.box_fourier_bands),
        "rank_ffn_dim": int(rank.ffn_dim),
        "rank_dropout": float(rank.dropout),
        "head_init_seed": int(heads.head_init_seed),
        "confidence_dim": int(heads.confidence_dim),
        "enable_patch_branch": bool(
            getattr(expected_model, "enable_patch_branch", False)
        ),
    }
    if contract.get("architecture") != expected_architecture:
        raise ValueError(
            f"{checkpoint_label}: relational architecture contract drifted"
        )
    seed = contract.get("seed")
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed != int(heads.head_init_seed)
    ):
        raise ValueError(f"{checkpoint_label}: relational initializer seed drifted")

    expected_roles = {
        "b58_base",
        "shared_backbone_alias",
        "random_patch_projection",
        "random_relational_rank",
        "random_absolute_confidence",
        "score_contract_buffer",
    }
    roles = contract.get("role_keys")
    counts = contract.get("role_key_counts")
    if (
        not isinstance(roles, Mapping)
        or set(roles) != expected_roles
        or not isinstance(counts, Mapping)
        or set(counts) != expected_roles
    ):
        raise ValueError(f"{checkpoint_label}: relational initializer roles drifted")
    normalized: dict[str, list[str]] = {}
    for role in expected_roles:
        keys = roles.get(role)
        if not isinstance(keys, list) or not keys or counts.get(role) != len(keys):
            raise ValueError(
                f"{checkpoint_label}: relational initializer role {role} drifted"
            )
        normalized[role] = [str(key) for key in keys]
    flattened = [key for keys in normalized.values() for key in keys]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(state):
        raise ValueError(
            f"{checkpoint_label}: relational roles are not an exact partition"
        )
    prefix_contracts = {
        "shared_backbone_alias": ("patch_encoder.backbone.",),
        "random_relational_rank": (
            "stage_b_data_driven_score_heads.rank_branch.",
        ),
        "random_absolute_confidence": (
            "stage_b_data_driven_score_heads.confidence_branch.",
            "stage_b_data_driven_score_heads.confidence_gate.",
        ),
    }
    for role, prefixes in prefix_contracts.items():
        if not all(key.startswith(prefixes) for key in normalized[role]):
            raise ValueError(
                f"{checkpoint_label}: relational role prefix drifted for {role}"
            )
    if normalized["score_contract_buffer"] != [
        "stage_b_data_driven_score_heads._contract_version"
    ]:
        raise ValueError(f"{checkpoint_label}: relational score contract drifted")
    for role, keys in normalized.items():
        if contract.get(f"{role}_tensor_sha256") != data_driven_tensor_state_sha256(
            state, keys
        ):
            raise ValueError(
                f"{checkpoint_label}: relational role hash drifted for {role}"
            )
    if contract.get("full_model_tensor_sha256") != data_driven_tensor_state_sha256(
        state, sorted(state)
    ):
        raise ValueError(f"{checkpoint_label}: relational full-model hash drifted")
    invariants = contract.get("invariants")
    required = {
        "b58_is_only_checkpoint_source",
        "no_u1000_u5020_tensor_source",
        "no_teacher_adapter_tensor_source",
        "canonical_query_and_full_text_paths_are_separate",
        "rank_and_confidence_parameters_are_disjoint",
        "patch_backbone_aliases_b58",
        "patch_initialization_matches_absolute_a0",
        "confidence_initialization_matches_absolute_a0",
        "image_pooling_is_padding_invariant",
    }
    if not isinstance(invariants, Mapping) or any(
        invariants.get(key) is not True for key in required
    ):
        raise ValueError(
            f"{checkpoint_label}: relational initializer invariants drifted"
        )


def validate_stage_b_data_driven_score_checkpoint(
    expected_model: nn.Module,
    state: Mapping[str, Any],
    *,
    checkpoint_label: str,
) -> None:
    """Fail closed on incomplete or architecture-incompatible trained states."""
    if not isinstance(state, Mapping):
        raise ValueError(f"{checkpoint_label}: model state is malformed")
    expected = expected_model.state_dict()
    missing = sorted(set(expected).difference(state))
    unexpected = sorted(set(state).difference(expected))
    if missing or unexpected:
        raise ValueError(
            f"{checkpoint_label}: exact model key coverage drifted; "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    for key, wanted in expected.items():
        value = state.get(key)
        if (
            not torch.is_tensor(value)
            or value.dtype != wanted.dtype
            or tuple(value.shape) != tuple(wanted.shape)
        ):
            raise ValueError(
                f"{checkpoint_label}: tensor shape/dtype drift at {key}"
            )
    contract_key = "stage_b_data_driven_score_heads._contract_version"
    contract = state.get(contract_key)
    expected_contract = expected.get(contract_key)
    if (
        not torch.is_tensor(contract)
        or not torch.is_tensor(expected_contract)
        or contract.numel() != 1
        or expected_contract.numel() != 1
        or int(contract.detach().cpu().item())
        != int(expected_contract.detach().cpu().item())
    ):
        raise ValueError(
            f"{checkpoint_label}: score-head contract version drifted"
        )


def validate_data_driven_confidence_initializer_payload(
    expected_model: nn.Module,
    payload: Mapping[str, Any],
    *,
    checkpoint_label: str,
    expected_scope: Optional[str] = None,
    minimum_source_optimizer_updates: Optional[int] = None,
) -> None:
    """Validate an exact DD1 model-only handoff into DD2/DD3."""
    if not isinstance(payload, Mapping) or set(payload) != {
        "model",
        "data_driven_confidence_initializer",
    }:
        raise ValueError(
            f"{checkpoint_label}: confidence initializer top-level keys drifted"
        )
    state = payload.get("model")
    contract = payload.get("data_driven_confidence_initializer")
    if not isinstance(state, Mapping) or not isinstance(contract, Mapping):
        raise ValueError(f"{checkpoint_label}: confidence initializer is malformed")
    validate_stage_b_data_driven_score_checkpoint(
        expected_model, state, checkpoint_label=checkpoint_label
    )
    if contract.get("schema") != DATA_DRIVEN_CONFIDENCE_INITIALIZER_SCHEMA:
        raise ValueError(f"{checkpoint_label}: confidence initializer schema drifted")
    if contract.get("scope") not in {"smoke", "formal"}:
        raise ValueError(f"{checkpoint_label}: confidence initializer scope drifted")
    if expected_scope is not None and contract.get("scope") != str(expected_scope):
        raise ValueError(
            f"{checkpoint_label}: expected scope {expected_scope!r}, got "
            f"{contract.get('scope')!r}"
        )
    exact_fields = {
        "training_initializer": True,
        "resumable": False,
        "optimizer_state_carried": False,
        "criterion_state_carried": False,
        "scheduler_scaler_rng_carried": False,
    }
    if any(contract.get(key) is not value for key, value in exact_fields.items()):
        raise ValueError(
            f"{checkpoint_label}: confidence initializer state boundary drifted"
        )
    source_args = contract.get("source_args_contract")
    required_args = {
        "stage_b_data_driven_experiment_id": "DD1",
        "stage_b_data_driven_train_mode": "rank_patch_only",
        "stage_b_data_driven_category_complete": True,
        "seed": 42,
    }
    if not isinstance(source_args, Mapping) or any(
        source_args.get(key) != value for key, value in required_args.items()
    ):
        raise ValueError(f"{checkpoint_label}: DD1 source args contract drifted")
    if contract.get("scope") == "formal" and source_args.get(
        "stage_b_data_driven_confidence_trained"
    ) is not False:
        raise ValueError(
            f"{checkpoint_label}: formal DD1 source did not seal untrained confidence"
        )
    source_provenance = contract.get("source_training_provenance")
    required_provenance_fields = {
        "schema",
        "code_files",
        "dataset_asset_files",
        "support_patch_pool_content",
        "allocator_environment",
        "required_allocator",
        "software",
    }
    if (
        not isinstance(source_provenance, Mapping)
        or source_provenance.get("schema")
        != "pivot.stageb.data_driven_training_provenance/v1"
        or not required_provenance_fields.issubset(source_provenance)
    ):
        raise ValueError(
            f"{checkpoint_label}: DD1 source provenance is incomplete"
        )
    updates = contract.get("source_optimizer_updates")
    minimum = contract.get("minimum_source_optimizer_updates")
    if (
        not isinstance(updates, int)
        or isinstance(updates, bool)
        or not isinstance(minimum, int)
        or isinstance(minimum, bool)
        or minimum <= 0
        or updates < minimum
    ):
        raise ValueError(f"{checkpoint_label}: DD1 training-position contract drifted")
    if (
        minimum_source_optimizer_updates is not None
        and updates < int(minimum_source_optimizer_updates)
    ):
        raise ValueError(
            f"{checkpoint_label}: DD1 source has {updates} updates, below the "
            f"required {int(minimum_source_optimizer_updates)}"
        )
    for record_key in ("source_checkpoint", "base_initializer"):
        record = contract.get(record_key)
        if not (
            isinstance(record, Mapping)
            and isinstance(record.get("path"), str)
            and isinstance(record.get("sha256"), str)
            and len(record["sha256"]) == 64
        ):
            raise ValueError(
                f"{checkpoint_label}: {record_key} file record drifted"
            )
    expected_roles = {
        "frozen_b58_base",
        "frozen_shared_backbone_alias",
        "dd1_trained_rank",
        "dd1_trained_patch",
        "untouched_random_confidence",
        "score_contract_buffer",
    }
    roles = contract.get("role_keys")
    counts = contract.get("role_key_counts")
    if (
        not isinstance(roles, Mapping)
        or set(roles) != expected_roles
        or not isinstance(counts, Mapping)
        or set(counts) != expected_roles
    ):
        raise ValueError(f"{checkpoint_label}: confidence initializer roles drifted")
    normalized: dict[str, list[str]] = {}
    for role in expected_roles:
        keys = roles.get(role)
        if (
            not isinstance(keys, list)
            or not keys
            or counts.get(role) != len(keys)
        ):
            raise ValueError(
                f"{checkpoint_label}: confidence initializer role {role} drifted"
            )
        normalized[role] = [str(key) for key in keys]
    flattened = [key for keys in normalized.values() for key in keys]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(state):
        raise ValueError(
            f"{checkpoint_label}: confidence roles are not an exact state partition"
        )
    for role, keys in normalized.items():
        observed = data_driven_tensor_state_sha256(state, keys)
        if contract.get(f"{role}_tensor_sha256") != observed:
            raise ValueError(
                f"{checkpoint_label}: confidence role hash drifted for {role}"
            )
    if contract.get("full_model_tensor_sha256") != data_driven_tensor_state_sha256(
        state, sorted(state)
    ):
        raise ValueError(f"{checkpoint_label}: confidence full-model hash drifted")
    invariants = contract.get("invariants")
    required_invariants = {
        "dd1_model_preserved_bitwise",
        "rank_changed_from_base_initializer",
        "patch_changed_from_base_initializer",
        "confidence_unchanged_from_base_initializer",
        "frozen_b58_and_alias_unchanged_from_base_initializer",
        "new_confidence_optimizer_required",
        "dd2_dd3_share_this_initializer",
    }
    if not isinstance(invariants, Mapping) or any(
        invariants.get(key) is not True for key in required_invariants
    ):
        raise ValueError(
            f"{checkpoint_label}: confidence initializer invariants drifted"
        )


def validate_data_driven_criterion_checkpoint_state(
    state: Mapping[str, Any],
    *,
    expected_rank_supervision: str,
    checkpoint_label: str,
    allow_legacy_eval_contract: bool,
    require_cold_rank_queue: bool,
) -> tuple[int, Optional[int]]:
    """Validate criterion buffer *values* before ``load_state_dict`` mutates it."""
    if not isinstance(state, Mapping):
        raise ValueError(f"{checkpoint_label}: criterion state is not a mapping")
    supervision = normalize_data_driven_rank_supervision(
        expected_rank_supervision
    )

    def _exact_int64_scalar(key: str, *, optional: bool = False) -> Optional[int]:
        value = state.get(key)
        if optional and value is None:
            return None
        if (
            not torch.is_tensor(value)
            or value.dtype != torch.int64
            or value.numel() != 1
        ):
            raise ValueError(
                f"{checkpoint_label}: criterion {key} must be one int64 scalar"
            )
        return int(value.detach().reshape(-1)[0].item())

    version = _exact_int64_scalar("criterion_contract_version")
    contract_id = _exact_int64_scalar(
        "rank_supervision_contract_id", optional=True
    )
    observed = (int(version), contract_id)
    if allow_legacy_eval_contract:
        allowed = _DATA_DRIVEN_EVAL_CRITERION_CONTRACTS[supervision]
    else:
        allowed = {_DATA_DRIVEN_CURRENT_CRITERION_CONTRACTS[supervision]}
    if observed not in allowed:
        raise ValueError(
            f"{checkpoint_label}: criterion contract {observed} does not "
            f"match supervision {supervision!r}; allowed={sorted(allowed, key=str)}"
        )

    if require_cold_rank_queue:
        queue = state.get("fpr_positive_queue")
        count = state.get("fpr_positive_queue_count")
        cursor = state.get("fpr_positive_queue_cursor")
        if (
            not torch.is_tensor(queue)
            or not queue.is_floating_point()
            or not bool(torch.isfinite(queue).all().item())
            or bool((queue != 0).any().item())
            or _exact_int64_scalar("fpr_positive_queue_count") != 0
            or _exact_int64_scalar("fpr_positive_queue_cursor") != 0
            or count is None
            or cursor is None
        ):
            raise ValueError(
                f"{checkpoint_label}: rank/patch criterion queue is not cold"
            )
    return observed


def validate_data_driven_trained_checkpoint_payload(
    expected_model: nn.Module,
    payload: Mapping[str, Any],
    *,
    checkpoint_label: str,
    expected_experiment_id: str,
    expected_confidence_trained: bool,
    expected_token_weight: Optional[float] = None,
    expected_confidence_initializer_sha256: Optional[str] = None,
    expected_optimizer_updates: Optional[int] = None,
    expected_variant_id: Optional[str] = None,
    expected_rank_supervision: Optional[str] = None,
    expected_rank_negative_iou_threshold: Optional[float] = None,
    expected_assignment_weight: Optional[float] = None,
    expected_deployment_weight: Optional[float] = None,
    allow_legacy_criterion_contract: bool = True,
) -> Mapping[str, Any]:
    """Bind evaluation behavior to the training phase recorded in a checkpoint."""
    if not isinstance(payload, Mapping):
        raise ValueError(f"{checkpoint_label}: trained checkpoint is not a mapping")
    state = payload.get("model")
    validate_stage_b_data_driven_score_checkpoint(
        expected_model, state, checkpoint_label=checkpoint_label
    )
    saved_args = payload.get("args")
    if not isinstance(saved_args, Mapping):
        raise ValueError(f"{checkpoint_label}: trained checkpoint has no saved args")
    experiment_id = str(expected_experiment_id)
    if experiment_id not in {"DD0", "DD1", "DD2", "DD3"}:
        raise ValueError(f"{checkpoint_label}: unknown data-driven experiment ID")
    expected_mode = (
        "confidence_pair" if experiment_id in {"DD2", "DD3"} else "rank_patch_only"
    )
    required = {
        "stage_b_data_driven_score": True,
        "stage_b_data_driven_experiment_id": experiment_id,
        "stage_b_data_driven_train_mode": expected_mode,
        "stage_b_data_driven_category_complete": experiment_id != "DD0",
        "stage_b_data_driven_confidence_trained": bool(
            expected_confidence_trained
        ),
        "seed": 42,
    }
    drift = {
        key: (saved_args.get(key), value)
        for key, value in required.items()
        if saved_args.get(key) != value
    }
    if drift:
        raise ValueError(
            f"{checkpoint_label}: trained data-driven phase metadata drifted: {drift}"
        )
    if expected_variant_id is not None:
        wanted_variant = str(expected_variant_id).strip()
        if not wanted_variant or saved_args.get(
            "stage_b_data_driven_variant_id"
        ) != wanted_variant:
            raise ValueError(
                f"{checkpoint_label}: trained data-driven variant attribution drifted"
            )
    observed_supervision = normalize_data_driven_rank_supervision(
        saved_args.get(
            "stage_b_data_driven_rank_supervision",
            DATA_DRIVEN_RANK_SUPERVISION_ALL_NONPOSITIVE,
        )
    )
    if expected_rank_supervision is not None:
        wanted_supervision = normalize_data_driven_rank_supervision(
            expected_rank_supervision
        )
        if observed_supervision != wanted_supervision:
            raise ValueError(
                f"{checkpoint_label}: trained rank supervision attribution drifted"
            )
    criterion_state = payload.get("criterion")
    if not isinstance(criterion_state, Mapping):
        raise ValueError(
            f"{checkpoint_label}: trained checkpoint has no criterion state"
        )
    validate_data_driven_criterion_checkpoint_state(
        criterion_state,
        expected_rank_supervision=observed_supervision,
        checkpoint_label=checkpoint_label,
        allow_legacy_eval_contract=bool(
            allow_legacy_criterion_contract
        ),
        require_cold_rank_queue=expected_mode == "rank_patch_only",
    )
    if expected_rank_negative_iou_threshold is not None:
        observed_threshold = saved_args.get(
            "stage_b_data_driven_rank_negative_iou_threshold", 0.3
        )
        if (
            not isinstance(observed_threshold, (int, float))
            or isinstance(observed_threshold, bool)
            or not math.isfinite(float(observed_threshold))
            or float(observed_threshold)
            != float(expected_rank_negative_iou_threshold)
        ):
            raise ValueError(
                f"{checkpoint_label}: trained rank negative-IoU threshold drifted"
            )
    for argument, expected_weight, label in (
        (
            "stage_b_data_driven_assignment_weight",
            expected_assignment_weight,
            "assignment objective",
        ),
        (
            "stage_b_data_driven_deployment_weight",
            expected_deployment_weight,
            "deployment objective",
        ),
    ):
        if expected_weight is None:
            continue
        if (
            not isinstance(expected_weight, (int, float))
            or isinstance(expected_weight, bool)
            or not math.isfinite(float(expected_weight))
            or float(expected_weight) < 0.0
        ):
            raise ValueError(
                f"expected_{argument.removeprefix('stage_b_data_driven_')} "
                "must be a finite nonnegative number"
            )
        observed_weight = saved_args.get(argument)
        if (
            not isinstance(observed_weight, (int, float))
            or isinstance(observed_weight, bool)
            or not math.isfinite(float(observed_weight))
            or float(observed_weight) != float(expected_weight)
        ):
            raise ValueError(
                f"{checkpoint_label}: trained {label} weight drifted"
            )
    heads = getattr(expected_model, "stage_b_data_driven_score_heads", None)
    expected_rank_architecture = normalize_data_driven_rank_architecture(
        getattr(heads, "rank_architecture", "absolute_token")
    )
    observed_rank_architecture = normalize_data_driven_rank_architecture(
        saved_args.get("stage_b_data_driven_rank_architecture", "absolute_token")
    )
    if observed_rank_architecture != expected_rank_architecture:
        raise ValueError(
            f"{checkpoint_label}: trained rank architecture attribution drifted"
        )
    if expected_rank_architecture == "relational_v1":
        rank = heads.rank_branch
        expected_relational = {
            "stage_b_data_driven_rank_dim": int(heads.rank_dim),
            "stage_b_data_driven_rank_num_heads": int(rank.num_heads),
            "stage_b_data_driven_rank_image_level_policy": str(
                rank.image_level_policy
            ),
            "stage_b_data_driven_rank_image_levels": int(rank.image_levels),
            "stage_b_data_driven_rank_image_pool_size": int(
                rank.image_pool_size
            ),
            "stage_b_data_driven_rank_image_pool_policy": str(
                rank.image_pool_policy
            ),
            "stage_b_data_driven_rank_box_fourier_bands": int(
                rank.box_fourier_bands
            ),
            "stage_b_data_driven_rank_ffn_dim": int(rank.ffn_dim),
            "stage_b_data_driven_rank_dropout": float(rank.dropout),
            "stage_b_data_driven_head_init_seed": int(heads.head_init_seed),
        }
        relational_drift = {
            key: (saved_args.get(key), value)
            for key, value in expected_relational.items()
            if saved_args.get(key) != value
        }
        if relational_drift:
            raise ValueError(
                f"{checkpoint_label}: trained relational rank metadata drifted: "
                f"{relational_drift}"
            )
    updates = payload.get("optimizer_updates")
    if not isinstance(updates, int) or isinstance(updates, bool) or updates <= 0:
        raise ValueError(
            f"{checkpoint_label}: trained checkpoint has no positive optimizer update"
        )
    if expected_optimizer_updates is not None:
        if (
            isinstance(expected_optimizer_updates, bool)
            or not isinstance(expected_optimizer_updates, int)
            or expected_optimizer_updates <= 0
        ):
            raise ValueError(
                "expected_optimizer_updates must be a positive exact integer"
            )
        if updates != expected_optimizer_updates:
            raise ValueError(
                f"{checkpoint_label}: expected exactly {expected_optimizer_updates} "
                f"successful optimizer updates, observed {updates}"
            )
        if payload.get("checkpoint_reason") != "max_train_iters":
            raise ValueError(
                f"{checkpoint_label}: exact-update evaluation requires a "
                "max_train_iters terminal checkpoint"
            )
        if saved_args.get("max_train_iters") != expected_optimizer_updates:
            raise ValueError(
                f"{checkpoint_label}: saved max_train_iters does not bind the "
                f"exact evaluation target {expected_optimizer_updates}"
            )
    if expected_token_weight is not None:
        observed_token_weight = saved_args.get(
            "stage_b_data_driven_token_weight"
        )
        if (
            not isinstance(observed_token_weight, (int, float))
            or isinstance(observed_token_weight, bool)
            or not math.isfinite(float(observed_token_weight))
            or float(observed_token_weight) != float(expected_token_weight)
        ):
            raise ValueError(
                f"{checkpoint_label}: DD2/DD3 token treatment drifted"
            )
    if expected_confidence_initializer_sha256 is not None and saved_args.get(
        "stage_b_data_driven_confidence_initializer_sha256"
    ) != str(expected_confidence_initializer_sha256):
        raise ValueError(
            f"{checkpoint_label}: confidence phase initializer attribution drifted"
        )
    return saved_args


def normalize_data_driven_train_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    if mode not in DATA_DRIVEN_TRAIN_MODES:
        raise ValueError(
            "stage_b_data_driven_train_mode must be one of "
            f"{DATA_DRIVEN_TRAIN_MODES}, got {mode!r}"
        )
    return mode


def normalize_data_driven_rank_architecture(value: Any) -> str:
    architecture = str(value or "absolute_token").strip().lower()
    if architecture not in DATA_DRIVEN_RANK_ARCHITECTURES:
        raise ValueError(
            "stage_b_data_driven_rank_architecture must be one of "
            f"{DATA_DRIVEN_RANK_ARCHITECTURES}, got {architecture!r}"
        )
    return architecture


def normalize_data_driven_rank_supervision(value: Any) -> str:
    supervision = str(
        value or DATA_DRIVEN_RANK_SUPERVISION_ALL_NONPOSITIVE
    ).strip().lower()
    if supervision not in DATA_DRIVEN_RANK_SUPERVISION_MODES:
        raise ValueError(
            "stage_b_data_driven_rank_supervision must be one of "
            f"{DATA_DRIVEN_RANK_SUPERVISION_MODES}, got {supervision!r}"
        )
    return supervision


class RelationalRankAdapter(nn.Module):
    """Rank all frozen b58 queries against full text and pooled image context."""

    def __init__(
        self,
        hidden_dim: int,
        rank_dim: int,
        *,
        num_heads: int = 4,
        image_level_policy: str = "last",
        image_levels: int = 2,
        image_pool_size: int = 8,
        image_pool_policy: str = DATA_DRIVEN_RELATIONAL_IMAGE_POOL_POLICY,
        box_fourier_bands: int = 16,
        ffn_dim: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        values = (
            hidden_dim,
            rank_dim,
            num_heads,
            image_levels,
            image_pool_size,
            box_fourier_bands,
            ffn_dim,
        )
        if any(int(value) <= 0 for value in values):
            raise ValueError("relational rank dimensions must be positive")
        if int(rank_dim) % int(num_heads) != 0:
            raise ValueError("relational rank_dim must be divisible by num_heads")
        if not math.isfinite(float(dropout)) or not 0.0 <= float(dropout) < 1.0:
            raise ValueError("relational rank dropout must be in [0,1)")
        if str(image_level_policy) != "last":
            raise ValueError("relational rank image_level_policy must be 'last'")
        if str(image_pool_policy) != DATA_DRIVEN_RELATIONAL_IMAGE_POOL_POLICY:
            raise ValueError(
                "relational rank image_pool_policy must be "
                f"{DATA_DRIVEN_RELATIONAL_IMAGE_POOL_POLICY!r}"
            )
        self.hidden_dim = int(hidden_dim)
        self.rank_dim = int(rank_dim)
        self.num_heads = int(num_heads)
        self.image_level_policy = str(image_level_policy)
        self.image_levels = int(image_levels)
        self.image_pool_size = int(image_pool_size)
        self.image_pool_policy = str(image_pool_policy)
        self.box_fourier_bands = int(box_fourier_bands)
        self.ffn_dim = int(ffn_dim)
        self.dropout = float(dropout)

        self.query_norm = nn.LayerNorm(self.hidden_dim)
        self.query_proj = nn.Linear(self.hidden_dim, self.rank_dim)
        self.box_proj = nn.Linear(
            4 * 2 * self.box_fourier_bands, self.rank_dim
        )
        self.text_norm = nn.LayerNorm(self.hidden_dim)
        self.text_proj = nn.Linear(self.hidden_dim, self.rank_dim)
        self.image_norm = nn.LayerNorm(self.hidden_dim)
        self.image_proj = nn.Linear(self.hidden_dim, self.rank_dim)
        self.image_pos_proj = nn.Linear(
            2 * 2 * self.box_fourier_bands, self.rank_dim
        )
        self.image_level_embed = nn.Parameter(
            torch.empty(self.image_levels, self.rank_dim)
        )
        nn.init.normal_(self.image_level_embed, std=0.02)

        self.text_query_norm = nn.LayerNorm(self.rank_dim)
        self.text_attention = nn.MultiheadAttention(
            self.rank_dim,
            self.num_heads,
            dropout=self.dropout,
            batch_first=True,
        )
        self.image_query_norm = nn.LayerNorm(self.rank_dim)
        self.image_attention = nn.MultiheadAttention(
            self.rank_dim,
            self.num_heads,
            dropout=self.dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(self.rank_dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.rank_dim, self.ffn_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.ffn_dim, self.rank_dim),
            nn.Dropout(self.dropout),
        )
        self.output_norm = nn.LayerNorm(self.rank_dim)
        self.score_head = nn.Linear(self.rank_dim, 1)

        frequencies = math.pi * torch.pow(
            2.0, torch.arange(self.box_fourier_bands, dtype=torch.float32)
        )
        self.register_buffer("fourier_frequencies", frequencies, persistent=True)

    @staticmethod
    def _fourier(values: Tensor, frequencies: Tensor) -> Tensor:
        angles = values.float().unsqueeze(-1) * frequencies.float()
        return torch.cat((angles.sin(), angles.cos()), dim=-1).flatten(-2)

    def _pool_image_memory(
        self,
        image_features: Sequence[Tensor],
        image_masks: Sequence[Tensor],
    ) -> tuple[Tensor, Tensor]:
        if len(image_features) < self.image_levels or len(image_masks) != len(
            image_features
        ):
            raise ValueError("relational rank image levels are incomplete")
        selected_features = image_features[-self.image_levels :]
        selected_masks = image_masks[-self.image_levels :]
        batch = int(selected_features[0].shape[0])
        pooled_rows = []
        pooled_masks = []
        for level, (feature, mask) in enumerate(
            zip(selected_features, selected_masks)
        ):
            if feature.dim() != 4 or int(feature.shape[0]) != batch or int(
                feature.shape[1]
            ) != self.hidden_dim:
                raise ValueError("relational image features must have shape (B,D,H,W)")
            if tuple(mask.shape) != (
                batch,
                int(feature.shape[-2]),
                int(feature.shape[-1]),
            ):
                raise ValueError("relational image masks must align with image features")
            valid_bool = ~mask.detach().to(device=feature.device, dtype=torch.bool)
            row_valid = valid_bool.any(dim=2)
            column_valid = valid_bool.any(dim=1)
            if bool((~row_valid.any(dim=1) | ~column_valid.any(dim=1)).any().item()):
                raise ValueError("relational image feature contains an empty valid region")

            height, width = map(int, feature.shape[-2:])
            row_indices = torch.arange(height, device=feature.device)[None]
            column_indices = torch.arange(width, device=feature.device)[None]
            y0 = torch.where(row_valid, row_indices, height).amin(dim=1)
            y1 = torch.where(row_valid, row_indices + 1, 0).amax(dim=1)
            x0 = torch.where(column_valid, column_indices, width).amin(dim=1)
            x1 = torch.where(column_valid, column_indices + 1, 0).amax(dim=1)
            extents = torch.stack((y0, y1, x0, x1), dim=1).detach().cpu().tolist()

            groups: dict[tuple[int, int, int, int], list[int]] = {}
            for sample_index, raw_extent in enumerate(extents):
                extent = tuple(int(value) for value in raw_extent)
                groups.setdefault(extent, []).append(sample_index)

            level_memory: list[Optional[Tensor]] = [None] * batch
            level_masks: list[Optional[Tensor]] = [None] * batch
            for (top, bottom, left, right), sample_indices in groups.items():
                crop_height = bottom - top
                crop_width = right - left
                if crop_height <= 0 or crop_width <= 0:
                    raise ValueError("relational image feature has invalid valid extents")
                group_indices = torch.as_tensor(
                    sample_indices, device=feature.device, dtype=torch.long
                )
                cropped_feature = feature.detach().index_select(0, group_indices)[
                    :, :, top:bottom, left:right
                ].float()
                cropped_valid = valid_bool.index_select(0, group_indices)[
                    :, top:bottom, left:right
                ][:, None].float()

                y_coordinates = (
                    torch.arange(crop_height, device=feature.device, dtype=torch.float32)
                    + 0.5
                ) / float(crop_height)
                x_coordinates = (
                    torch.arange(crop_width, device=feature.device, dtype=torch.float32)
                    + 0.5
                ) / float(crop_width)
                yy, xx = torch.meshgrid(y_coordinates, x_coordinates, indexing="ij")
                coordinates = torch.stack((xx, yy), dim=0)[None].expand(
                    len(sample_indices), -1, -1, -1
                )
                pooled_components = F.adaptive_avg_pool2d(
                    torch.cat(
                        (
                            cropped_feature * cropped_valid,
                            cropped_valid,
                            coordinates * cropped_valid,
                        ),
                        dim=1,
                    ),
                    self.image_pool_size,
                )
                valid_fraction = pooled_components[
                    :, self.hidden_dim : self.hidden_dim + 1
                ]
                denominator = valid_fraction.clamp_min(1e-6)
                pooled_feature = (
                    pooled_components[:, : self.hidden_dim] / denominator
                ).flatten(2).transpose(1, 2)
                pooled_coordinates = (
                    pooled_components[:, self.hidden_dim + 1 :] / denominator
                ).flatten(2).transpose(1, 2)
                pooled = self.image_proj(self.image_norm(pooled_feature))
                spatial = self.image_pos_proj(
                    self._fourier(pooled_coordinates, self.fourier_frequencies)
                ).to(dtype=pooled.dtype)
                pooled = (
                    pooled
                    + spatial
                    + self.image_level_embed[level].to(dtype=pooled.dtype)[None, None]
                )
                group_masks = valid_fraction[:, 0].flatten(1) <= 0.0
                for group_row, sample_index in enumerate(sample_indices):
                    level_memory[sample_index] = pooled[group_row]
                    level_masks[sample_index] = group_masks[group_row]

            if any(value is None for value in level_memory) or any(
                value is None for value in level_masks
            ):
                raise RuntimeError("relational image pooling lost a batch row")
            pooled_rows.append(torch.stack(level_memory, dim=0))
            pooled_masks.append(torch.stack(level_masks, dim=0))
        memory = torch.cat(pooled_rows, dim=1)
        memory_mask = torch.cat(pooled_masks, dim=1)
        if bool(memory_mask.all(dim=1).any().item()):
            raise ValueError("relational image memory contains an empty row")
        return memory, memory_mask

    def forward(
        self,
        query_hs: Tensor,
        encoded_text: Tensor,
        expression_token_mask: Tensor,
        *,
        query_boxes: Tensor,
        image_features: Sequence[Tensor],
        image_masks: Sequence[Tensor],
        image_owner_indices: Tensor,
    ) -> dict[str, Tensor]:
        if query_hs.dim() != 3 or encoded_text.dim() != 3:
            raise ValueError("relational query/text inputs must have shape (B,N,D)")
        scorer_batch, query_count = map(int, query_hs.shape[:2])
        if tuple(encoded_text.shape[:1]) != (scorer_batch,) or int(
            query_hs.shape[-1]
        ) != self.hidden_dim or int(encoded_text.shape[-1]) != self.hidden_dim:
            raise ValueError("relational query/text batches or dimensions drifted")
        token_mask = torch.as_tensor(
            expression_token_mask, device=encoded_text.device, dtype=torch.bool
        )
        if tuple(token_mask.shape) != tuple(encoded_text.shape[:2]) or bool(
            (~token_mask.any(dim=1)).any().item()
        ):
            raise ValueError("every relational expression requires a scored token")
        owners = torch.as_tensor(
            image_owner_indices, device=query_hs.device, dtype=torch.long
        )
        if tuple(owners.shape) != (scorer_batch,):
            raise ValueError("image_owner_indices must have shape (scorer_batch,)")
        if query_boxes.dim() != 3 or int(query_boxes.shape[1]) != query_count or int(
            query_boxes.shape[-1]
        ) != 4:
            raise ValueError("relational query boxes must have shape (B,Q,4)")
        image_batch = int(query_boxes.shape[0])
        if int(owners.min().item()) < 0 or int(owners.max().item()) >= image_batch:
            raise ValueError("image_owner_indices are outside the image batch")

        boxes = query_boxes.detach().index_select(0, owners)
        query = self.query_proj(self.query_norm(query_hs.detach()))
        query = query + self.box_proj(
            self._fourier(boxes, self.fourier_frequencies)
        ).to(dtype=query.dtype)
        text = self.text_proj(self.text_norm(encoded_text.detach()))
        text_update = self.text_attention(
            self.text_query_norm(query),
            text,
            text,
            key_padding_mask=~token_mask,
            need_weights=False,
        )[0]
        query = query + text_update

        image_memory, image_mask = self._pool_image_memory(
            image_features, image_masks
        )
        if int(image_memory.shape[0]) != image_batch:
            raise ValueError("relational image memory and query boxes do not align")
        image_memory = image_memory.index_select(0, owners)
        image_mask = image_mask.index_select(0, owners)
        image_update = self.image_attention(
            self.image_query_norm(query),
            image_memory,
            image_memory,
            key_padding_mask=image_mask,
            need_weights=False,
        )[0]
        query = query + image_update
        query = query + self.ffn(self.ffn_norm(query))
        query = self.output_norm(query)
        score = self.score_head(query).squeeze(-1).float()
        if tuple(score.shape) != (scorer_batch, query_count) or not bool(
            torch.isfinite(score).all().item()
        ):
            raise RuntimeError("relational rank produced invalid scores")

        with torch.no_grad():
            diagnostic_query = F.normalize(query.detach().float(), dim=-1)
            diagnostic_text = F.normalize(text.detach().float(), dim=-1)
            token_logits = torch.einsum(
                "bqd,btd->bqt", diagnostic_query, diagnostic_text
            )
            token_logits.masked_fill_(~token_mask[:, None], 0.0)
        return {
            "token_logits": token_logits,
            "score": score,
            "query_feature": F.normalize(query.float(), dim=-1),
            "token_mask": token_mask,
            "logit_scale": score.new_tensor(1.0),
        }


class AbsoluteTokenScoreBranch(nn.Module):
    """Produce absolute query-token compatibility from detached base features."""

    def __init__(self, hidden_dim: int, score_dim: int, *, temperature: float) -> None:
        super().__init__()
        if int(hidden_dim) <= 0 or int(score_dim) <= 0:
            raise ValueError("hidden_dim and score_dim must be positive")
        if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
            raise ValueError("token score temperature must be finite and positive")
        self.hidden_dim = int(hidden_dim)
        self.score_dim = int(score_dim)
        self.query_norm = nn.LayerNorm(self.hidden_dim)
        self.text_norm = nn.LayerNorm(self.hidden_dim)
        self.query_proj = nn.Linear(self.hidden_dim, self.score_dim)
        self.text_proj = nn.Linear(self.hidden_dim, self.score_dim)
        self.logit_scale = nn.Parameter(
            torch.as_tensor(math.log(1.0 / float(temperature)), dtype=torch.float32)
        )

    def forward(
        self,
        query_hs: Tensor,
        encoded_text: Tensor,
        expression_token_mask: Tensor,
    ) -> dict[str, Tensor]:
        if query_hs.dim() != 3 or encoded_text.dim() != 3:
            raise ValueError("query_hs and encoded_text must have shape (B,N,D)")
        if int(query_hs.shape[0]) != int(encoded_text.shape[0]):
            raise ValueError("query and expression batches must align")
        if int(query_hs.shape[-1]) != self.hidden_dim or int(
            encoded_text.shape[-1]
        ) != self.hidden_dim:
            raise ValueError("query/text hidden dimensions do not match the scorer")
        token_mask = torch.as_tensor(
            expression_token_mask,
            device=encoded_text.device,
            dtype=torch.bool,
        )
        if tuple(token_mask.shape) != tuple(encoded_text.shape[:2]):
            raise ValueError("expression_token_mask must have shape (B,T)")
        if bool((~token_mask.any(dim=1)).any().item()):
            raise ValueError("every full expression must contain a scored token")

        query = F.normalize(
            self.query_proj(self.query_norm(query_hs.detach())).float(), dim=-1
        )
        text = F.normalize(
            self.text_proj(self.text_norm(encoded_text.detach())).float(), dim=-1
        )
        scale = self.logit_scale.float().clamp(max=math.log(100.0)).exp()
        token_logits = scale * torch.einsum("bqd,btd->bqt", query, text)
        token_logits = token_logits.masked_fill(~token_mask[:, None, :], 0.0)
        token_count = token_mask.sum(dim=1).clamp_min(1).float()
        score = (
            token_logits.sigmoid().masked_fill(~token_mask[:, None, :], 0.0).sum(dim=-1)
            / token_count[:, None]
        )
        return {
            "token_logits": token_logits,
            "score": score,
            "query_feature": query,
            "token_mask": token_mask,
            "logit_scale": scale,
        }


class StageBDataDrivenScoreHeads(nn.Module):
    """Independent rank/confidence heads plus a patch category gate."""

    score_feature_dim = 6

    def __init__(
        self,
        hidden_dim: int,
        *,
        rank_dim: int = 128,
        rank_architecture: str = "absolute_token",
        rank_num_heads: int = 4,
        rank_image_level_policy: str = "last",
        rank_image_levels: int = 2,
        rank_image_pool_size: int = 8,
        rank_image_pool_policy: str = DATA_DRIVEN_RELATIONAL_IMAGE_POOL_POLICY,
        rank_box_fourier_bands: int = 16,
        rank_ffn_dim: int = 512,
        rank_dropout: float = 0.0,
        head_init_seed: int = 42,
        confidence_dim: int = 128,
        token_temperature: float = 0.07,
        gate_hidden_dim: int = 128,
        gate_pool_temperature: float = 0.1,
        gate_topk: int = 10,
        category_gate: bool = False,
        category_gate_max_gap: float = 3.0,
        patch_score_clip: float = 5.0,
    ) -> None:
        super().__init__()
        if int(gate_hidden_dim) <= 0 or int(gate_topk) <= 0:
            raise ValueError("confidence gate dimensions must be positive")
        if float(gate_pool_temperature) <= 0.0:
            raise ValueError("gate_pool_temperature must be positive")
        if float(category_gate_max_gap) < 0.0 or float(patch_score_clip) <= 0.0:
            raise ValueError("category gate gap/clip are invalid")
        self.hidden_dim = int(hidden_dim)
        self.rank_dim = int(rank_dim)
        self.rank_architecture = normalize_data_driven_rank_architecture(
            rank_architecture
        )
        self.head_init_seed = int(head_init_seed)
        self.confidence_dim = int(confidence_dim)
        self.gate_pool_temperature = float(gate_pool_temperature)
        self.gate_topk = int(gate_topk)
        self.category_gate = bool(category_gate)
        self.category_gate_max_gap = float(category_gate_max_gap)
        self.patch_score_clip = float(patch_score_clip)

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.head_init_seed + 1009)
            if self.rank_architecture == "absolute_token":
                self.rank_branch = AbsoluteTokenScoreBranch(
                    self.hidden_dim, self.rank_dim, temperature=token_temperature
                )
            else:
                self.rank_branch = RelationalRankAdapter(
                    self.hidden_dim,
                    self.rank_dim,
                    num_heads=rank_num_heads,
                    image_level_policy=rank_image_level_policy,
                    image_levels=rank_image_levels,
                    image_pool_size=rank_image_pool_size,
                    image_pool_policy=rank_image_pool_policy,
                    box_fourier_bands=rank_box_fourier_bands,
                    ffn_dim=rank_ffn_dim,
                    dropout=rank_dropout,
                )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.head_init_seed + 2017)
            self.confidence_branch = AbsoluteTokenScoreBranch(
                self.hidden_dim, self.confidence_dim, temperature=token_temperature
            )
            self.confidence_gate = nn.Sequential(
                nn.Linear(
                    self.confidence_dim + self.score_feature_dim,
                    int(gate_hidden_dim),
                ),
                nn.GELU(),
                nn.Linear(int(gate_hidden_dim), int(gate_hidden_dim)),
                nn.GELU(),
                nn.Linear(int(gate_hidden_dim), 1),
            )
            nn.init.zeros_(self.confidence_gate[-1].weight)
            nn.init.zeros_(self.confidence_gate[-1].bias)
        contract_version = (
            DATA_DRIVEN_SCORE_CONTRACT_VERSION
            if self.rank_architecture == "absolute_token"
            else DATA_DRIVEN_RELATIONAL_SCORE_CONTRACT_VERSION
        )
        self.register_buffer(
            "_contract_version",
            torch.as_tensor(contract_version, dtype=torch.int64),
            persistent=True,
        )

    def rank_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.rank_branch.parameters())

    def confidence_parameters(self) -> tuple[nn.Parameter, ...]:
        return tuple(self.confidence_branch.parameters()) + tuple(
            self.confidence_gate.parameters()
        )

    def _confidence_gate_inputs(
        self,
        query_feature: Tensor,
        score: Tensor,
        candidate_mask: Tensor,
    ) -> Tensor:
        masked_score = score.float().masked_fill(~candidate_mask, -torch.inf)
        valid_count = candidate_mask.sum(dim=1)
        weights = torch.softmax(
            masked_score / self.gate_pool_temperature, dim=1
        ).masked_fill(~candidate_mask, 0.0)
        pooled = torch.einsum("bq,bqd->bd", weights, query_feature.float())
        score_max = masked_score.max(dim=1).values
        count = valid_count.float()
        score_mean = score.masked_fill(~candidate_mask, 0.0).sum(dim=1) / count
        centered = (score - score_mean[:, None]).masked_fill(~candidate_mask, 0.0)
        score_std = (centered.square().sum(dim=1) / count).clamp_min(0.0).sqrt()
        top_count = min(self.gate_topk, int(score.shape[1]))
        top_values = torch.topk(masked_score, k=top_count, dim=1).values
        top_valid = torch.arange(top_count, device=score.device)[None, :] < torch.minimum(
            valid_count, valid_count.new_full(valid_count.shape, top_count)
        )[:, None]
        score_top_mean = top_values.masked_fill(~top_valid, 0.0).sum(dim=1) / top_valid.sum(
            dim=1
        ).clamp_min(1)
        if int(score.shape[1]) > 1:
            top_two = torch.topk(masked_score, k=2, dim=1).values
            score_margin = torch.where(
                valid_count > 1,
                top_two[:, 0] - top_two[:, 1],
                torch.zeros_like(score_max),
            )
        else:
            score_margin = torch.zeros_like(score_max)
        entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=1)
        score_entropy = torch.where(
            valid_count > 1,
            entropy / count.log().clamp_min(1e-8),
            torch.zeros_like(entropy),
        )
        stats = torch.stack(
            (
                score_max,
                score_top_mean,
                score_mean,
                score_std,
                score_margin,
                score_entropy,
            ),
            dim=-1,
        )
        return torch.cat((pooled, stats), dim=-1)

    def _apply_category_gate(
        self,
        patch_score: Tensor,
        text_rank_score: Tensor,
        candidate_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.training:
            raise RuntimeError("the data-driven category gate is inference-only")
        eligible, patch = data_driven_category_gate_mask(
            patch_score,
            candidate_mask,
            max_gap=self.category_gate_max_gap,
            clip=self.patch_score_clip,
        )
        text_min = text_rank_score.masked_fill(~candidate_mask, torch.inf).amin(
            dim=1, keepdim=True
        )
        text_max = text_rank_score.masked_fill(~candidate_mask, -torch.inf).amax(
            dim=1, keepdim=True
        )
        below_min = torch.nextafter(
            text_min, torch.full_like(text_min, -torch.inf)
        )
        if not bool(torch.isfinite(below_min).all().item()):
            raise RuntimeError("cannot construct a finite category-demotion score")
        delta = torch.where(candidate_mask, text_rank_score, text_max) - text_max
        ineligible_score = below_min + delta
        return torch.where(eligible, text_rank_score, ineligible_score), eligible, patch

    def forward(
        self,
        query_hs: Tensor,
        encoded_expression: Tensor,
        expression_token_mask: Tensor,
        *,
        patch_score: Optional[Tensor] = None,
        candidate_mask: Optional[Tensor] = None,
        query_boxes: Optional[Tensor] = None,
        image_features: Optional[Sequence[Tensor]] = None,
        image_masks: Optional[Sequence[Tensor]] = None,
        image_owner_indices: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        if self.rank_architecture == "absolute_token":
            rank = self.rank_branch(
                query_hs, encoded_expression, expression_token_mask
            )
        else:
            if (
                query_boxes is None
                or image_features is None
                or image_masks is None
                or image_owner_indices is None
            ):
                raise ValueError(
                    "relational_v1 rank requires boxes, image features/masks, "
                    "and image owners"
                )
            rank = self.rank_branch(
                query_hs,
                encoded_expression,
                expression_token_mask,
                query_boxes=query_boxes,
                image_features=image_features,
                image_masks=image_masks,
                image_owner_indices=image_owner_indices,
            )
        confidence = self.confidence_branch(
            query_hs, encoded_expression, expression_token_mask
        )
        if candidate_mask is None:
            mask = torch.ones_like(rank["score"], dtype=torch.bool)
        else:
            mask = torch.as_tensor(
                candidate_mask, device=rank["score"].device, dtype=torch.bool
            )
            if tuple(mask.shape) != tuple(rank["score"].shape):
                raise ValueError("candidate_mask must align with query scores")
        if bool((~mask.any(dim=1)).any().item()):
            raise ValueError("every data-driven row requires a candidate")

        gate_input = self._confidence_gate_inputs(
            confidence["query_feature"], confidence["score"], mask
        )
        gate = self.confidence_gate(gate_input).squeeze(-1)
        confidence_score = confidence["score"] + gate[:, None]
        rank_score = rank["score"]
        eligible = mask
        normalized_patch = rank_score.new_zeros(rank_score.shape)
        if patch_score is not None:
            if patch_score.dim() == 3 and int(patch_score.shape[-1]) == 1:
                patch_score = patch_score[..., 0]
            if tuple(patch_score.shape) != tuple(rank_score.shape):
                raise ValueError("patch_score must align with data-driven rank scores")
            if self.category_gate:
                rank_score, eligible, normalized_patch = self._apply_category_gate(
                    patch_score, rank["score"], mask
                )

        return {
            "text_rank_token_logits": rank["token_logits"],
            "text_rank_score": rank["score"],
            "rank_score": rank_score,
            "confidence_token_logits": confidence["token_logits"],
            "confidence_base_score": confidence["score"],
            "confidence_gate": gate,
            "confidence_score": confidence_score,
            "expression_token_mask": rank["token_mask"],
            "candidate_mask": mask,
            "category_gate_eligible_mask": eligible,
            "category_gate_patch_score": normalized_patch,
            "rank_logit_scale": rank["logit_scale"],
            "confidence_logit_scale": confidence["logit_scale"],
        }


def _candidate_max_iou(
    candidate_boxes: Tensor,
    targets: Sequence[Mapping[str, Any]],
    *,
    primary_only: bool,
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
                raise ValueError("category-complete rank rows require primary_instance_mask")
            primary = primary.reshape(-1)
            if int(primary.numel()) != int(boxes.shape[0]) or int(
                primary.sum().item()
            ) != 1:
                raise ValueError("each category-complete row requires one primary instance")
            boxes = boxes[primary]
        target_xyxy = box_ops.box_cxcywh_to_xyxy(
            boxes.detach().to(device=result.device, dtype=torch.float32).reshape(-1, 4)
        )
        iou, _ = box_ops.box_iou(candidates[index], target_xyxy)
        if iou.numel():
            result[index] = iou.max(dim=1).values
    return result


def _candidate_primary_auxiliary_iou(
    candidate_boxes: Tensor,
    targets: Sequence[Mapping[str, Any]],
) -> tuple[Tensor, Tensor]:
    """Return detached candidate IoU to one primary and all same-class auxiliaries."""
    if candidate_boxes.dim() != 3 or int(candidate_boxes.shape[-1]) != 4:
        raise ValueError("candidate boxes must have shape (B,Q,4)")
    if len(targets) != int(candidate_boxes.shape[0]):
        raise ValueError("targets must align with candidate boxes")
    candidates = box_ops.box_cxcywh_to_xyxy(candidate_boxes.detach().float())
    primary_result = candidates.new_zeros(candidates.shape[:2])
    auxiliary_result = candidates.new_zeros(candidates.shape[:2])
    for index, target in enumerate(targets):
        boxes = target.get("boxes")
        primary = target.get("primary_instance_mask")
        if (
            not torch.is_tensor(boxes)
            or boxes.dim() != 2
            or int(boxes.shape[-1]) != 4
            or not torch.is_tensor(primary)
            or primary.dtype != torch.bool
        ):
            raise ValueError(
                "same-category rank rows require boxes and a boolean primary mask"
            )
        primary = primary.reshape(-1)
        if (
            int(primary.numel()) != int(boxes.shape[0])
            or int(primary.sum().item()) != 1
        ):
            raise ValueError(
                "same-category rank rows require one exact primary instance"
            )
        auxiliary = ~primary
        if not bool(auxiliary.any().item()):
            raise ValueError(
                "same-category rank rows require at least one auxiliary instance"
            )
        target_xyxy = box_ops.box_cxcywh_to_xyxy(
            boxes.detach()
            .to(device=candidates.device, dtype=torch.float32)
            .reshape(-1, 4)
        )
        iou, _ = box_ops.box_iou(candidates[index], target_xyxy)
        primary_result[index] = iou[:, primary].amax(dim=1)
        auxiliary_result[index] = iou[:, auxiliary].amax(dim=1)
    return primary_result, auxiliary_result


def _candidate_official_assignment_iou(
    candidate_boxes: Tensor,
    targets: Sequence[Mapping[str, Any]],
) -> tuple[Tensor, Tensor]:
    """Return query IoU to two explicitly assigned official referents."""
    if candidate_boxes.dim() != 3 or int(candidate_boxes.shape[-1]) != 4:
        raise ValueError("candidate boxes must have shape (B,Q,4)")
    if len(targets) != int(candidate_boxes.shape[0]):
        raise ValueError("assignment targets must align with candidate boxes")
    candidates = box_ops.box_cxcywh_to_xyxy(candidate_boxes.detach().float())
    result = candidates.new_zeros((*candidates.shape[:2], 2))
    valid = torch.zeros(
        (int(candidate_boxes.shape[0]),),
        device=candidates.device,
        dtype=torch.bool,
    )
    for index, target in enumerate(targets):
        boxes = target.get("boxes")
        roles = target.get("stage_b_data_driven_assignment_role")
        pair_valid = target.get("stage_b_data_driven_assignment_valid")
        if (
            not torch.is_tensor(boxes)
            or boxes.dim() != 2
            or int(boxes.shape[-1]) != 4
            or not torch.is_tensor(roles)
            or roles.dtype != torch.int64
            or tuple(roles.reshape(-1).shape) != (int(boxes.shape[0]),)
            or not torch.is_tensor(pair_valid)
            or pair_valid.dtype != torch.bool
            or pair_valid.numel() != 1
        ):
            raise ValueError(
                "official assignment rows require aligned boxes, int64 roles, "
                "and one exact boolean validity flag"
            )
        is_valid = bool(pair_valid.detach().reshape(-1)[0].item())
        if not is_valid:
            if bool(((roles < -1) | (roles > 1)).any().item()):
                raise ValueError("invalid assignment rows contain an unknown role")
            continue
        flat_roles = roles.reshape(-1)
        if any(int((flat_roles == role).sum().item()) != 1 for role in (0, 1)):
            raise ValueError(
                "every valid official assignment row requires one role-0 and "
                "one role-1 instance"
            )
        if bool(((flat_roles < -1) | (flat_roles > 1)).any().item()):
            raise ValueError("official assignment rows contain an unknown role")
        target_xyxy = box_ops.box_cxcywh_to_xyxy(
            boxes.detach()
            .to(device=candidates.device, dtype=torch.float32)
            .reshape(-1, 4)
        )
        iou, _ = box_ops.box_iou(candidates[index], target_xyxy)
        for role in (0, 1):
            result[index, :, role] = iou[:, flat_roles == role].squeeze(1)
        valid[index] = True
    return result, valid


def _candidate_official_assignment_role_iou(
    candidate_boxes: Tensor,
    targets: Sequence[Mapping[str, Any]],
) -> tuple[Tensor, Tensor, Tensor]:
    """Return assigned-pair IoU plus max IoU to every unassigned instance.

    The third value is deliberately separate from the two official roles.
    Queries that also overlap an unassigned same-category instance are neutral
    for the text objective rather than silently becoming positives or
    negatives for either assigned expression.
    """
    assignment_iou, pair_valid = _candidate_official_assignment_iou(
        candidate_boxes, targets
    )
    candidates = box_ops.box_cxcywh_to_xyxy(candidate_boxes.detach().float())
    other_iou = candidates.new_zeros(candidates.shape[:2])
    for index, target in enumerate(targets):
        if not bool(pair_valid[index].item()):
            continue
        boxes = target["boxes"]
        roles = target["stage_b_data_driven_assignment_role"].reshape(-1)
        other = roles == -1
        if not bool(other.any().item()):
            continue
        target_xyxy = box_ops.box_cxcywh_to_xyxy(
            boxes.detach()
            .to(device=candidates.device, dtype=torch.float32)
            .reshape(-1, 4)
        )
        iou, _ = box_ops.box_iou(candidates[index], target_xyxy)
        other_iou[index] = iou[:, other].amax(dim=1)
    return assignment_iou, other_iou, pair_valid


def official_assignment_delta_loss(
    rank_score: Tensor,
    assignment_iou: Tensor,
    pair_valid: Tensor,
    candidate_mask: Tensor,
    patch_score: Tensor,
    *,
    positive_iou_threshold: float,
    negative_iou_threshold: float,
    category_gate_max_gap: float,
    patch_score_clip: float,
    temperature: float,
) -> dict[str, Tensor]:
    """Train the deployed top-1 decision for two official referents.

    Detached GT IoU and patch scores select one representative query per
    referent. Each expression receives its own within-expression pairwise loss;
    all other queries are ignored by that assignment objective. A separately
    returned deployment-hard loss compares the selected correct query with the
    highest-scoring incorrect query under the exact Gap3 deployment rule.
    """
    if (
        rank_score.dim() != 3
        or int(rank_score.shape[-1]) != 2
        or not rank_score.is_floating_point()
    ):
        raise ValueError("official assignment rank scores must have shape (B,Q,2)")
    batch_size, query_count, _ = map(int, rank_score.shape)
    if tuple(assignment_iou.shape) != (batch_size, query_count, 2):
        raise ValueError("official assignment IoU must have shape (B,Q,2)")
    if (
        pair_valid.dtype != torch.bool
        or tuple(pair_valid.reshape(-1).shape) != (batch_size,)
    ):
        raise ValueError("official assignment validity must have shape (B,)")
    if candidate_mask.dtype != torch.bool:
        raise ValueError("official assignment candidate mask must be boolean")
    if tuple(candidate_mask.shape) == (batch_size, query_count, 2):
        if not torch.equal(candidate_mask[..., 0], candidate_mask[..., 1]):
            raise ValueError("paired expressions changed the candidate query set")
        candidate = candidate_mask[..., 0]
    elif tuple(candidate_mask.shape) == (batch_size, query_count):
        candidate = candidate_mask
    else:
        raise ValueError("official assignment candidate mask is misaligned")
    if not bool(torch.isfinite(rank_score).all().item()):
        raise ValueError("official assignment rank scores must all be finite")
    if not bool(torch.isfinite(assignment_iou).all().item()):
        raise ValueError("official assignment IoUs must all be finite")
    if (
        not 0.0 <= float(negative_iou_threshold) < float(positive_iou_threshold)
        or not float(positive_iou_threshold) <= 1.0
        or not math.isfinite(float(temperature))
        or float(temperature) <= 0.0
    ):
        raise ValueError("official assignment thresholds/temperature are invalid")

    gap3, _normalized_patch = data_driven_category_gate_mask(
        patch_score.detach(),
        candidate,
        max_gap=float(category_gate_max_gap),
        clip=float(patch_score_clip),
    )
    owned0 = (
        gap3
        & (assignment_iou[..., 0] >= float(positive_iou_threshold))
        & (assignment_iou[..., 1] < float(negative_iou_threshold))
    )
    owned1 = (
        gap3
        & (assignment_iou[..., 1] >= float(positive_iou_threshold))
        & (assignment_iou[..., 0] < float(negative_iou_threshold))
    )

    def select_query(own_iou: Tensor, cross_iou: Tensor, mask: Tensor) -> Tensor:
        # Lexicographic GT-only choice: max own IoU, min cross IoU, min index.
        own_max = own_iou.masked_fill(~mask, -torch.inf).amax(dim=1)
        own_tie = mask & (own_iou == own_max[:, None])
        cross_min = cross_iou.masked_fill(~own_tie, torch.inf).amin(dim=1)
        final_tie = own_tie & (cross_iou == cross_min[:, None])
        return final_tie.to(dtype=torch.int64).argmax(dim=1)

    query0 = select_query(assignment_iou[..., 0], assignment_iou[..., 1], owned0)
    query1 = select_query(assignment_iou[..., 1], assignment_iou[..., 0], owned1)
    reachable0 = owned0.any(dim=1)
    reachable1 = owned1.any(dim=1)
    pair_valid = pair_valid.reshape(-1).to(device=rank_score.device)
    query_collision = pair_valid & reachable0 & reachable1 & (query0 == query1)
    runtime_valid = pair_valid & reachable0 & reachable1 & ~query_collision
    rows = torch.arange(batch_size, device=rank_score.device)

    score00 = rank_score[rows, query0, 0]
    score01 = rank_score[rows, query1, 0]
    score10 = rank_score[rows, query0, 1]
    score11 = rank_score[rows, query1, 1]
    direction_delta = torch.stack((score00 - score01, score11 - score10), dim=1)
    delta = direction_delta.sum(dim=1)
    direction_loss = float(temperature) * F.softplus(
        (float(temperature) - direction_delta.float()) / float(temperature)
    )
    if bool(runtime_valid.any().item()):
        loss = direction_loss[runtime_valid].mean()
    else:
        loss = rank_score.sum() * 0.0

    correct_query = gap3[:, :, None] & (
        assignment_iou >= float(positive_iou_threshold)
    ) & (
        assignment_iou.flip(-1) < float(negative_iou_threshold)
    )
    wrong_query = gap3[:, :, None] & ~correct_query
    has_wrong_query = wrong_query.any(dim=1)
    hard_wrong_score, hard_wrong_query = rank_score.masked_fill(
        ~wrong_query, -torch.inf
    ).max(dim=1)
    selected_correct_score = torch.stack((score00, score11), dim=1)
    hard_wrong_score = torch.where(
        has_wrong_query, hard_wrong_score, selected_correct_score.detach()
    )
    deployment_hard_delta = selected_correct_score - hard_wrong_score
    deployment_hard_valid = runtime_valid[:, None] & has_wrong_query
    deployment_hard_direction_loss = float(temperature) * F.softplus(
        (
            float(temperature)
            - deployment_hard_delta.float()
        )
        / float(temperature)
    )
    if bool(deployment_hard_valid.any().item()):
        deployment_hard_loss = deployment_hard_direction_loss[
            deployment_hard_valid
        ].mean()
    else:
        deployment_hard_loss = rank_score.sum() * 0.0

    deployment_top = rank_score.masked_fill(~gap3[:, :, None], -torch.inf).argmax(
        dim=1
    )
    deployment_iou = assignment_iou.gather(
        1, deployment_top[:, None, :]
    ).squeeze(1)
    deployment_cross_iou = assignment_iou.flip(-1).gather(
        1, deployment_top[:, None, :]
    ).squeeze(1)
    deployment_correct_direction = (
        (deployment_iou >= float(positive_iou_threshold))
        & (deployment_cross_iou < float(negative_iou_threshold))
        & runtime_valid[:, None]
    )
    selected_own_iou = torch.stack(
        (
            assignment_iou[rows, query0, 0],
            assignment_iou[rows, query1, 1],
        ),
        dim=1,
    )
    selected_cross_iou = torch.stack(
        (
            assignment_iou[rows, query0, 1],
            assignment_iou[rows, query1, 0],
        ),
        dim=1,
    )
    return {
        "loss": loss,
        "deployment_hard_loss": deployment_hard_loss,
        "delta": delta.detach(),
        "direction_delta": direction_delta.detach(),
        "deployment_hard_delta": deployment_hard_delta.detach(),
        "deployment_hard_valid": deployment_hard_valid.detach(),
        "deployment_hard_query": hard_wrong_query.detach(),
        "runtime_valid": runtime_valid.detach(),
        "data_valid": pair_valid.detach(),
        "role0_reachable": reachable0.detach(),
        "role1_reachable": reachable1.detach(),
        "query_collision": query_collision.detach(),
        "owned_role0_queries": owned0.float().sum().detach(),
        "owned_role1_queries": owned1.float().sum().detach(),
        "gap3_queries": gap3.float().sum().detach(),
        "selected_query0": query0.detach(),
        "selected_query1": query1.detach(),
        "selected_own_iou": selected_own_iou.detach(),
        "selected_cross_iou": selected_cross_iou.detach(),
        "deployment_correct_direction": deployment_correct_direction.detach(),
    }


def role_routed_official_assignment_top1_loss(
    rank_score: Tensor,
    assignment_iou: Tensor,
    other_same_category_iou: Tensor,
    pair_valid: Tensor,
    candidate_mask: Tensor,
    patch_score: Tensor,
    *,
    positive_iou_threshold: float,
    negative_iou_threshold: float,
    category_gate_max_gap: float,
    patch_score_clip: float,
    margin: float,
    temperature: float,
    include_all_exclusive_nonowned: bool = False,
) -> dict[str, Tensor]:
    """Supervise only unambiguous within-category assignment decisions.

    For expression ``e``, a text positive must overlap its own assigned
    referent, while being disjoint from the paired referent and every
    unassigned same-category instance.  V1 uses only the reciprocal paired
    referent as a text negative.  V2 also uses a query that exclusively hits
    any other category-complete same-category instance: it is a provably wrong
    referent for the expression without teaching category/background rejection
    to the text branch.  Background/category negatives, 0.3--0.5 overlaps, and
    double overlaps remain neutral.  The patch gate is the exact inference
    gate and is detached so text supervision cannot alter category scoring.
    """
    if (
        rank_score.dim() != 3
        or int(rank_score.shape[-1]) != 2
        or not rank_score.is_floating_point()
    ):
        raise ValueError("role-routed rank scores must have shape (B,Q,2)")
    batch_size, query_count, _ = map(int, rank_score.shape)
    if tuple(assignment_iou.shape) != (batch_size, query_count, 2):
        raise ValueError("role-routed assignment IoU must have shape (B,Q,2)")
    if tuple(other_same_category_iou.shape) != (batch_size, query_count):
        raise ValueError("role-routed unassigned IoU must have shape (B,Q)")
    if (
        pair_valid.dtype != torch.bool
        or tuple(pair_valid.reshape(-1).shape) != (batch_size,)
    ):
        raise ValueError("role-routed assignment validity must have shape (B,)")
    if candidate_mask.dtype != torch.bool:
        raise ValueError("role-routed candidate mask must be boolean")
    if tuple(candidate_mask.shape) == (batch_size, query_count, 2):
        if not torch.equal(candidate_mask[..., 0], candidate_mask[..., 1]):
            raise ValueError("paired expressions changed the candidate query set")
        candidate = candidate_mask[..., 0]
    elif tuple(candidate_mask.shape) == (batch_size, query_count):
        candidate = candidate_mask
    else:
        raise ValueError("role-routed candidate mask is misaligned")
    if not bool(torch.isfinite(rank_score).all().item()):
        raise ValueError("role-routed rank scores must all be finite")
    if not bool(torch.isfinite(assignment_iou).all().item()) or not bool(
        torch.isfinite(other_same_category_iou).all().item()
    ):
        raise ValueError("role-routed IoUs must all be finite")
    if (
        not 0.0 <= float(negative_iou_threshold) < float(positive_iou_threshold)
        or not float(positive_iou_threshold) <= 1.0
        or not math.isfinite(float(margin))
        or float(margin) < 0.0
        or not math.isfinite(float(temperature))
        or float(temperature) <= 0.0
    ):
        raise ValueError("role-routed thresholds/margin/temperature are invalid")

    gate, _normalized_patch = data_driven_category_gate_mask(
        patch_score.detach(),
        candidate,
        max_gap=float(category_gate_max_gap),
        clip=float(patch_score_clip),
    )
    separated_from_other = (
        other_same_category_iou < float(negative_iou_threshold)
    )
    role0 = (
        gate
        & (assignment_iou[..., 0] >= float(positive_iou_threshold))
        & (assignment_iou[..., 1] < float(negative_iou_threshold))
        & separated_from_other
    )
    role1 = (
        gate
        & (assignment_iou[..., 1] >= float(positive_iou_threshold))
        & (assignment_iou[..., 0] < float(negative_iou_threshold))
        & separated_from_other
    )
    owned = torch.stack((role0, role1), dim=-1)
    paired_sibling = torch.stack((role1, role0), dim=-1)
    if include_all_exclusive_nonowned:
        safe_nonowned0 = (
            gate
            & (assignment_iou[..., 0] < float(negative_iou_threshold))
            & (
                (assignment_iou[..., 1] >= float(positive_iou_threshold))
                | (
                    other_same_category_iou
                    >= float(positive_iou_threshold)
                )
            )
        )
        safe_nonowned1 = (
            gate
            & (assignment_iou[..., 1] < float(negative_iou_threshold))
            & (
                (assignment_iou[..., 0] >= float(positive_iou_threshold))
                | (
                    other_same_category_iou
                    >= float(positive_iou_threshold)
                )
            )
        )
        safe_nonowned = torch.stack(
            (safe_nonowned0, safe_nonowned1), dim=-1
        )
    else:
        safe_nonowned = paired_sibling
    has_owned = owned.any(dim=1)
    has_nonowned = safe_nonowned.any(dim=1)
    data_valid = pair_valid.reshape(-1).to(device=rank_score.device)
    runtime_valid_direction = (
        data_valid[:, None] & has_owned & has_nonowned
    )

    positive_score, positive_query = rank_score.masked_fill(
        ~owned, -torch.inf
    ).max(dim=1)
    negative_score, negative_query = rank_score.masked_fill(
        ~safe_nonowned, -torch.inf
    ).max(dim=1)
    positive_score = torch.where(
        has_owned, positive_score, torch.zeros_like(positive_score)
    )
    negative_score = torch.where(
        has_nonowned, negative_score, torch.zeros_like(negative_score)
    )
    direction_delta = positive_score - negative_score
    direction_loss = float(temperature) * F.softplus(
        (
            float(margin)
            - direction_delta.float()
        )
        / float(temperature)
    )
    if bool(runtime_valid_direction.any().item()):
        loss = direction_loss[runtime_valid_direction].mean()
    else:
        loss = rank_score.sum() * 0.0

    category_max_iou = torch.maximum(
        assignment_iou.amax(dim=-1), other_same_category_iou
    )
    category_negative = (
        gate & (category_max_iou < float(negative_iou_threshold))
    )
    expanded_category_negative = category_negative[:, :, None].expand(
        -1, -1, 2
    )
    neutral = gate[:, :, None] & ~(
        owned | safe_nonowned | expanded_category_negative
    )
    deployment_top = rank_score.masked_fill(
        ~gate[:, :, None], -torch.inf
    ).argmax(dim=1)
    winner_owned = owned.gather(
        1, deployment_top[:, None, :]
    ).squeeze(1)
    winner_sibling = paired_sibling.gather(
        1, deployment_top[:, None, :]
    ).squeeze(1)
    winner_nonowned = safe_nonowned.gather(
        1, deployment_top[:, None, :]
    ).squeeze(1)
    winner_unassigned = winner_nonowned & ~winner_sibling
    winner_category_negative = expanded_category_negative.gather(
        1, deployment_top[:, None, :]
    ).squeeze(1)
    winner_neutral = neutral.gather(
        1, deployment_top[:, None, :]
    ).squeeze(1)
    return {
        "loss": loss,
        "data_valid": data_valid.detach(),
        "runtime_valid_direction": runtime_valid_direction.detach(),
        "runtime_valid": runtime_valid_direction.all(dim=1).detach(),
        "direction_delta": direction_delta.detach(),
        "owned_mask": owned.detach(),
        "paired_sibling_mask": paired_sibling.detach(),
        "safe_sibling_mask": safe_nonowned.detach(),
        "safe_nonowned_mask": safe_nonowned.detach(),
        "category_negative_mask": category_negative.detach(),
        "neutral_mask": neutral.detach(),
        "positive_query": positive_query.detach(),
        "negative_query": negative_query.detach(),
        "deployment_top": deployment_top.detach(),
        "deployment_winner_owned": winner_owned.detach(),
        "deployment_winner_sibling": winner_sibling.detach(),
        "deployment_winner_nonowned": winner_nonowned.detach(),
        "deployment_winner_unassigned": winner_unassigned.detach(),
        "deployment_winner_category_negative": (
            winner_category_negative.detach()
        ),
        "deployment_winner_neutral": winner_neutral.detach(),
        "gap_queries": gate.float().sum().detach(),
    }


def _validate_rank_patch_targets(
    targets: Sequence[Mapping[str, Any]],
    *,
    category_complete: bool,
) -> None:
    for index, target in enumerate(targets):
        boxes = target.get("boxes")
        labels = target.get("labels")
        primary = target.get("primary_instance_mask")
        marker = target.get("stage_b_u2_category_complete")
        if (
            not torch.is_tensor(boxes)
            or boxes.dim() != 2
            or int(boxes.shape[-1]) != 4
            or int(boxes.shape[0]) < 1
        ):
            raise ValueError(f"data-driven target {index} has invalid boxes")
        if (
            not torch.is_tensor(primary)
            or primary.dtype != torch.bool
            or tuple(primary.reshape(-1).shape) != (int(boxes.shape[0]),)
            or int(primary.sum().item()) != 1
        ):
            raise ValueError(
                f"data-driven target {index} requires one exact primary instance"
            )
        if not (
            torch.is_tensor(marker)
            and marker.dtype == torch.bool
            and marker.numel() == 1
            and bool(marker.reshape(-1)[0].item()) is bool(category_complete)
        ):
            raise ValueError(
                "data-driven target variant does not match the configured "
                f"category_complete={bool(category_complete)}"
            )
        if not category_complete and int(boxes.shape[0]) != 1:
            raise ValueError("DD0 ordinary-primary supervision requires exactly one box")
        if category_complete:
            if (
                not torch.is_tensor(labels)
                or labels.numel() != int(boxes.shape[0])
            ):
                raise ValueError("DD1 category-complete labels do not align with boxes")
            flat_labels = labels.reshape(-1)
            primary_label = flat_labels[primary.reshape(-1)][0]
            if not bool((flat_labels == primary_label).all().item()):
                raise ValueError("DD1 contains an instance outside the primary category")


def _instance_complete_patch_margin_loss(
    patch_score: Tensor,
    candidate_boxes: Tensor,
    targets: Sequence[Mapping[str, Any]],
    *,
    positive_iou_threshold: float,
    negative_iou_threshold: float,
    margin: float,
    temperature: float,
) -> tuple[Tensor, Tensor, Tensor]:
    if patch_score.dim() == 3 and int(patch_score.shape[-1]) == 1:
        patch_score = patch_score[..., 0]
    if patch_score.dim() != 2 or tuple(patch_score.shape) != tuple(
        candidate_boxes.shape[:2]
    ):
        raise ValueError("patch scores must align with candidate boxes")
    candidates = box_ops.box_cxcywh_to_xyxy(candidate_boxes.detach().float())
    row_losses = []
    valid_instances = 0
    skipped_instances = 0
    tau = float(temperature)
    for row_index, target in enumerate(targets):
        boxes = target.get("boxes")
        if not torch.is_tensor(boxes) or boxes.numel() == 0:
            continue
        targets_xyxy = box_ops.box_cxcywh_to_xyxy(
            boxes.detach().to(device=candidates.device, dtype=torch.float32).reshape(-1, 4)
        )
        iou, _ = box_ops.box_iou(candidates[row_index], targets_xyxy)
        category_max_iou = iou.max(dim=1).values
        negative = category_max_iou < float(negative_iou_threshold)
        if not bool(negative.any().item()):
            skipped_instances += int(targets_xyxy.shape[0])
            continue
        scores = patch_score[row_index].float()
        hard_negative = scores.masked_fill(~negative, -torch.inf).max()
        positives = iou >= float(positive_iou_threshold)
        positive_counts = positives.sum(dim=0)
        reachable = positive_counts > 0
        reachable_count = int(reachable.sum().item())
        valid_instances += reachable_count
        skipped_instances += int(targets_xyxy.shape[0]) - reachable_count
        if reachable_count:
            reachable_positive = positives[:, reachable]
            counts = positive_counts[reachable]
            smooth_positive = tau * (
                torch.logsumexp(
                    (scores[:, None] / tau).masked_fill(
                        ~reachable_positive, -torch.inf
                    ),
                    dim=0,
                )
                - counts.float().log()
            )
            instance_loss = tau * F.softplus(
                (float(margin) - smooth_positive + hard_negative) / tau
            )
            row_losses.append(instance_loss.mean())
    loss = (
        torch.stack(row_losses).mean()
        if row_losses
        else patch_score.float().sum() * 0.0
    )
    return (
        loss,
        patch_score.new_tensor(float(valid_instances)).detach(),
        patch_score.new_tensor(float(skipped_instances)).detach(),
    )


def _active_unsafe_row_fraction_loss(
    violations: Tensor,
    *,
    temperature: float,
) -> tuple[Tensor, Tensor]:
    """Approximate an unsafe fraction while focusing gradient at its boundary."""
    if violations.dim() != 1 or not violations.is_floating_point():
        raise ValueError("active-unsafe violations must be one floating vector")
    active = violations.detach() > 0.0
    if int(violations.numel()) == 0:
        return violations.sum() * 0.0, active
    boundary_occupancy = float(temperature) * (
        2.0 * torch.sigmoid(violations / float(temperature)) - 1.0
    )
    loss = torch.where(
        active,
        boundary_occupancy,
        torch.zeros_like(boundary_occupancy),
    ).mean()
    return loss, active


def _active_unsafe_fixed_denominator_severity_loss(
    violations: Tensor,
    *,
    temperature: float,
) -> tuple[Tensor, Tensor]:
    """Penalize every unsafe item without saturating its tail gradient."""
    if violations.dim() != 1 or not violations.is_floating_point():
        raise ValueError("active-unsafe violations must be one floating vector")
    active = violations.detach() > 0.0
    if int(violations.numel()) == 0:
        return violations.sum() * 0.0, active
    tau = float(temperature)
    active_severity = tau * (
        F.softplus(violations / tau) - math.log(2.0)
    )
    loss = torch.where(
        active,
        active_severity,
        torch.zeros_like(active_severity),
    ).mean()
    return loss, active


def _dense_fixed_denominator_softplus_loss(
    violations: Tensor,
    *,
    temperature: float,
) -> Tensor:
    """Smoothly penalize every negative near or beyond the Gate3 boundary."""
    if violations.dim() != 1 or not violations.is_floating_point():
        raise ValueError("dense-tail violations must be one floating vector")
    if int(violations.numel()) == 0:
        return violations.sum() * 0.0
    tau = float(temperature)
    return (tau * F.softplus(violations / tau)).mean()


def deployment_gate_category_patch_loss(
    patch_score: Tensor,
    candidate_boxes: Tensor,
    targets: Sequence[Mapping[str, Any]],
    candidate_mask: Tensor,
    *,
    positive_iou_threshold: float,
    negative_iou_threshold: float,
    category_gate_max_gap: float,
    patch_score_clip: float,
    boundary_margin: float,
    temperature: float,
    active_unsafe_auxiliary_weight: float = 1.0,
    drop_dense_tail_weight: float = 0.0,
    dense_category_focal_weight: float = 1.0,
    dense_category_focal_alpha: float = 0.25,
    dense_category_focal_gamma: float = 2.0,
    dense_category_focal_negative_weight: float = 1.0,
    role_exclusive_keep: bool = False,
    drop_positive_anchor_gradient_policy: str = (
        DATA_DRIVEN_PATCH_DROP_ANCHOR_GLOBAL_MAX
    ),
) -> dict[str, Tensor]:
    """Optimize the exact standardized category gate used at inference.

    Each reachable same-category GT contributes one keep constraint: at least
    one of its IoU-positive queries must remain within ``Gap - eps`` of the
    current row maximum.  Each row with a certain category negative contributes
    one drop constraint: the best category-positive query must exceed the
    hardest certain negative by ``Gap + eps``.  In role-routed training, each
    assigned referent also contributes a keep constraint over queries that are
    positive for it and below the negative threshold for every other annotated
    same-category instance.  The worst keep barrier is augmented with a
    row-normalized unsafe-fraction surrogate.  The worst drop barrier is
    augmented with fixed-denominator shifted-softplus severity over every
    active category negative.  The v17 drop contrast sends its positive gradient
    only through the global category-positive maximum.  The v18 policy keeps that
    exact forward anchor but uses a straight-through mean of every reachable
    instance's best query for its gradient.  Both policies are exactly zero-sum
    per row and prevent a deployment-inert raw-score shift shortcut; v18 also
    aligns rejection pressure with per-instance retention.  The raw-logit focal
    components remain available as diagnostics, but their optional weight is zero
    in the affine-invariant contracts.  Ambiguous queries receive no direct class
    target.
    """
    if patch_score.dim() == 3 and int(patch_score.shape[-1]) == 1:
        flat_patch_score = patch_score[..., 0]
    else:
        flat_patch_score = patch_score
    if (
        flat_patch_score.dim() != 2
        or not flat_patch_score.is_floating_point()
        or tuple(flat_patch_score.shape) != tuple(candidate_boxes.shape[:2])
    ):
        raise ValueError("deployment-gate patch scores must align with boxes")
    if (
        candidate_boxes.dim() != 3
        or int(candidate_boxes.shape[-1]) != 4
        or len(targets) != int(candidate_boxes.shape[0])
    ):
        raise ValueError("deployment-gate candidate boxes/targets are misaligned")
    if (
        candidate_mask.dtype != torch.bool
        or tuple(candidate_mask.shape) != tuple(flat_patch_score.shape)
    ):
        raise ValueError("deployment-gate candidate mask is misaligned")
    if (
        not 0.0 <= float(negative_iou_threshold) < float(positive_iou_threshold)
        or not float(positive_iou_threshold) <= 1.0
        or not math.isfinite(float(category_gate_max_gap))
        or float(category_gate_max_gap) <= 0.0
        or not math.isfinite(float(boundary_margin))
        or not 0.0 <= float(boundary_margin) < float(category_gate_max_gap)
        or (
            float(category_gate_max_gap) + float(boundary_margin)
            >= 2.0 * float(patch_score_clip)
        )
        or not math.isfinite(float(temperature))
        or float(temperature) <= 0.0
        or not math.isfinite(float(active_unsafe_auxiliary_weight))
        or not 0.0 < float(active_unsafe_auxiliary_weight) <= 1.0
        or not math.isfinite(float(drop_dense_tail_weight))
        or float(drop_dense_tail_weight) < 0.0
        or not math.isfinite(float(dense_category_focal_weight))
        or float(dense_category_focal_weight) < 0.0
        or not 0.0 <= float(dense_category_focal_alpha) <= 1.0
        or not math.isfinite(float(dense_category_focal_gamma))
        or float(dense_category_focal_gamma) < 0.0
        or not math.isfinite(float(dense_category_focal_negative_weight))
        or float(dense_category_focal_negative_weight) < 0.0
        or drop_positive_anchor_gradient_policy
        not in DATA_DRIVEN_PATCH_DROP_ANCHOR_POLICIES
    ):
        raise ValueError("deployment-gate patch thresholds are invalid")

    deployed_gate, deployed_standardized = data_driven_category_gate_mask(
        flat_patch_score,
        candidate_mask,
        max_gap=float(category_gate_max_gap),
        clip=float(patch_score_clip),
    )
    # Forward values exactly match deployment.  Detached row statistics keep
    # neutral candidates from receiving a dense normalization gradient, while
    # the straight-through clamp prevents already-saturated selected queries
    # from becoming impossible to correct.
    standardized = _data_driven_standardized_patch_score(
        flat_patch_score,
        candidate_mask,
        clip=float(patch_score_clip),
        detach_statistics=True,
        straight_through_clip=True,
    )
    if not torch.equal(
        standardized.detach(), deployed_standardized.detach()
    ):
        raise RuntimeError(
            "training patch standardization drifted from deployment values"
        )
    candidates = box_ops.box_cxcywh_to_xyxy(candidate_boxes.detach().float())
    category_positive_mask = torch.zeros_like(candidate_mask)
    category_negative_mask = torch.zeros_like(candidate_mask)
    category_neutral_mask = torch.zeros_like(candidate_mask)
    role_exclusive_positive_mask = torch.zeros(
        (*candidate_mask.shape, 2),
        device=candidate_mask.device,
        dtype=torch.bool,
    )
    row_losses: list[Tensor] = []
    row_keep_losses: list[Tensor] = []
    row_keep_barrier_losses: list[Tensor] = []
    row_keep_active_unsafe_losses: list[Tensor] = []
    row_generic_keep_active_unsafe_losses: list[Tensor] = []
    row_role_exclusive_keep_active_unsafe_losses: list[Tensor] = []
    row_keep_mean_losses: list[Tensor] = []
    row_role_exclusive_keep_losses: list[Tensor] = []
    row_drop_losses: list[Tensor] = []
    row_drop_barrier_losses: list[Tensor] = []
    row_drop_active_unsafe_losses: list[Tensor] = []
    row_drop_dense_tail_losses: list[Tensor] = []
    row_dense_category_focal_losses: list[Tensor] = []
    row_dense_category_positive_focal_losses: list[Tensor] = []
    row_dense_category_negative_focal_losses: list[Tensor] = []
    valid_instances = 0
    skipped_instances = 0
    keep_safe_instances = 0
    keep_deployed_instances = 0
    role_exclusive_reachable_instances = 0
    role_exclusive_unreachable_instances = 0
    role_exclusive_keep_safe_instances = 0
    role_exclusive_keep_deployed_instances = 0
    active_unsafe_generic_keep_constraints = 0
    active_unsafe_role_exclusive_keep_constraints = 0
    active_unsafe_keep_rows = 0
    valid_drop_rows = 0
    drop_safe_rows = 0
    drop_deployed_rows = 0
    active_unsafe_drop_queries = 0
    active_unsafe_drop_rows = 0
    tau = float(temperature)
    keep_limit = float(category_gate_max_gap) - float(boundary_margin)
    drop_limit = float(category_gate_max_gap) + float(boundary_margin)

    for row_index, target in enumerate(targets):
        boxes = target.get("boxes")
        if (
            not torch.is_tensor(boxes)
            or boxes.dim() != 2
            or int(boxes.shape[-1]) != 4
            or int(boxes.shape[0]) <= 0
        ):
            raise ValueError(
                f"deployment-gate target {row_index} has invalid boxes"
            )
        targets_xyxy = box_ops.box_cxcywh_to_xyxy(
            boxes.detach()
            .to(device=candidates.device, dtype=torch.float32)
            .reshape(-1, 4)
        )
        iou, _ = box_ops.box_iou(candidates[row_index], targets_xyxy)
        role_exclusive_by_role: dict[int, Tensor] = {}
        if role_exclusive_keep:
            roles = target.get("stage_b_data_driven_assignment_role")
            pair_valid = target.get("stage_b_data_driven_assignment_valid")
            if (
                not torch.is_tensor(roles)
                or roles.dtype != torch.int64
                or tuple(roles.reshape(-1).shape)
                != (int(targets_xyxy.shape[0]),)
                or not torch.is_tensor(pair_valid)
                or pair_valid.dtype != torch.bool
                or pair_valid.numel() != 1
            ):
                raise ValueError(
                    "role-exclusive patch keep requires aligned int64 roles "
                    "and one exact boolean validity flag"
                )
            flat_roles = roles.reshape(-1).to(device=iou.device)
            if bool(((flat_roles < -1) | (flat_roles > 1)).any().item()):
                raise ValueError("role-exclusive patch keep found an unknown role")
            is_pair_valid = bool(pair_valid.detach().reshape(-1)[0].item())
            if is_pair_valid and any(
                int((flat_roles == role).sum().item()) != 1
                for role in (0, 1)
            ):
                raise ValueError(
                    "role-exclusive patch keep requires one role-0 and one "
                    "role-1 instance in every valid row"
                )
        row_candidate = candidate_mask[row_index]
        positive_by_instance = (
            (iou >= float(positive_iou_threshold))
            & row_candidate[:, None]
        )
        category_max_iou = iou.amax(dim=1)
        category_positive = positive_by_instance.any(dim=1)
        category_negative = (
            (category_max_iou < float(negative_iou_threshold))
            & row_candidate
        )
        category_neutral = row_candidate & ~(
            category_positive | category_negative
        )
        category_positive_mask[row_index] = category_positive
        category_negative_mask[row_index] = category_negative
        category_neutral_mask[row_index] = category_neutral

        if role_exclusive_keep and is_pair_valid:
            target_count = int(targets_xyxy.shape[0])
            for role in (0, 1):
                instance_index = int(
                    torch.nonzero(
                        flat_roles == role, as_tuple=False
                    ).reshape(-1)[0].item()
                )
                other_instance = torch.ones(
                    (target_count,), device=iou.device, dtype=torch.bool
                )
                other_instance[instance_index] = False
                separated_from_every_other = (
                    iou[:, other_instance] < float(negative_iou_threshold)
                ).all(dim=1)
                exclusive_positive = (
                    positive_by_instance[:, instance_index]
                    & separated_from_every_other
                )
                role_exclusive_positive_mask[
                    row_index, :, role
                ] = exclusive_positive
                if bool(exclusive_positive.any().item()):
                    role_exclusive_by_role[role] = exclusive_positive
                    role_exclusive_reachable_instances += 1
                else:
                    role_exclusive_unreachable_instances += 1

        reachable = positive_by_instance.any(dim=0)
        reachable_count = int(reachable.sum().item())
        valid_instances += reachable_count
        skipped_instances += int(targets_xyxy.shape[0]) - reachable_count
        if reachable_count <= 0:
            continue

        score = standardized[row_index].float()
        raw_score = flat_patch_score[row_index].float()
        positive_focal_loss = _sigmoid_focal_loss_no_reduce(
            raw_score[category_positive],
            torch.ones_like(raw_score[category_positive]),
            alpha=float(dense_category_focal_alpha),
            gamma=float(dense_category_focal_gamma),
        ).mean()
        if bool(category_negative.any().item()):
            negative_focal_loss = _sigmoid_focal_loss_no_reduce(
                raw_score[category_negative],
                torch.zeros_like(raw_score[category_negative]),
                alpha=float(dense_category_focal_alpha),
                gamma=float(dense_category_focal_gamma),
            ).mean()
        else:
            negative_focal_loss = raw_score.sum() * 0.0
        dense_category_focal_loss = positive_focal_loss + float(
            dense_category_focal_negative_weight
        ) * negative_focal_loss
        row_dense_category_focal_losses.append(dense_category_focal_loss)
        row_dense_category_positive_focal_losses.append(positive_focal_loss)
        row_dense_category_negative_focal_losses.append(negative_focal_loss)
        global_best = score.masked_fill(~row_candidate, -torch.inf).amax()
        generic_keep_terms: list[Tensor] = []
        generic_keep_violations: list[Tensor] = []
        reachable_instance_bests: list[Tensor] = []
        for instance_index in torch.nonzero(
            reachable, as_tuple=False
        ).reshape(-1).tolist():
            instance_positive = positive_by_instance[:, int(instance_index)]
            instance_best = score.masked_fill(
                ~instance_positive, -torch.inf
            ).amax()
            reachable_instance_bests.append(instance_best)
            deployed_distance = global_best - instance_best
            keep_violation = deployed_distance - keep_limit
            generic_keep_violations.append(keep_violation)
            generic_keep_terms.append(
                tau * F.softplus(keep_violation / tau)
            )
            keep_safe_instances += int(
                bool((deployed_distance.detach() <= keep_limit).item())
            )
            keep_deployed_instances += int(
                bool(
                    (
                        deployed_distance.detach()
                        <= float(category_gate_max_gap)
                    ).item()
                )
            )

        optimization_keep_terms = list(generic_keep_terms)
        optimization_keep_violations = list(generic_keep_violations)
        role_exclusive_terms: list[Tensor] = []
        role_exclusive_violations: list[Tensor] = []
        for exclusive_positive in role_exclusive_by_role.values():
            exclusive_best = score.masked_fill(
                ~exclusive_positive, -torch.inf
            ).amax()
            exclusive_distance = global_best - exclusive_best
            exclusive_violation = exclusive_distance - keep_limit
            exclusive_term = tau * F.softplus(exclusive_violation / tau)
            role_exclusive_terms.append(exclusive_term)
            role_exclusive_violations.append(exclusive_violation)
            optimization_keep_terms.append(exclusive_term)
            optimization_keep_violations.append(exclusive_violation)
            role_exclusive_keep_safe_instances += int(
                bool((exclusive_distance.detach() <= keep_limit).item())
            )
            role_exclusive_keep_deployed_instances += int(
                bool(
                    (
                        exclusive_distance.detach()
                        <= float(category_gate_max_gap)
                    ).item()
                )
            )

        keep_mean_loss = torch.stack(generic_keep_terms).mean()
        keep_barrier_loss = torch.stack(optimization_keep_terms).amax()
        generic_keep_active_unsafe_loss, generic_active_keep = (
            _active_unsafe_row_fraction_loss(
                torch.stack(generic_keep_violations),
                temperature=tau,
            )
        )
        row_generic_keep_active_unsafe_losses.append(
            generic_keep_active_unsafe_loss
        )
        active_unsafe_generic_keep_constraints += int(
            generic_active_keep.sum().item()
        )
        if role_exclusive_violations:
            (
                role_exclusive_keep_active_unsafe_loss,
                exclusive_active_keep,
            ) = _active_unsafe_row_fraction_loss(
                torch.stack(role_exclusive_violations),
                temperature=tau,
            )
            row_role_exclusive_keep_active_unsafe_losses.append(
                role_exclusive_keep_active_unsafe_loss
            )
            active_unsafe_role_exclusive_keep_constraints += int(
                exclusive_active_keep.sum().item()
            )
            keep_active_unsafe_loss = 0.5 * (
                generic_keep_active_unsafe_loss
                + role_exclusive_keep_active_unsafe_loss
            )
            active_keep = generic_active_keep.any() | (
                exclusive_active_keep.any()
            )
        else:
            keep_active_unsafe_loss = generic_keep_active_unsafe_loss
            active_keep = generic_active_keep.any()
        if bool(active_keep.item()):
            active_unsafe_keep_rows += 1
        # Keep the original minimax barrier intact.  The additive unsafe
        # fraction densifies its gradient without changing the boundary.
        keep_loss = keep_barrier_loss + float(
            active_unsafe_auxiliary_weight
        ) * keep_active_unsafe_loss
        row_keep_losses.append(keep_loss)
        row_keep_barrier_losses.append(keep_barrier_loss)
        row_keep_active_unsafe_losses.append(keep_active_unsafe_loss)
        row_keep_mean_losses.append(keep_mean_loss)
        if role_exclusive_terms:
            row_role_exclusive_keep_losses.append(
                torch.stack(role_exclusive_terms).amax()
            )
        if bool(category_negative.any().item()):
            forward_positive_anchor = score.masked_fill(
                ~category_positive, -torch.inf
            ).amax()
            if (
                drop_positive_anchor_gradient_policy
                == DATA_DRIVEN_PATCH_DROP_ANCHOR_INSTANCE_BALANCED
            ):
                gradient_anchor = torch.stack(reachable_instance_bests).mean()
                positive_anchor = forward_positive_anchor.detach() + (
                    gradient_anchor - gradient_anchor.detach()
                )
            else:
                positive_anchor = forward_positive_anchor
            hardest_negative = score.masked_fill(
                ~category_negative, -torch.inf
            ).amax()
            positive_negative_gap = positive_anchor - hardest_negative
            negative_scores = score[category_negative]
            positive_negative_gaps = positive_anchor - negative_scores
            drop_violations = drop_limit - positive_negative_gaps
            drop_barrier_loss = tau * F.softplus(
                (drop_limit - positive_negative_gap) / tau
            )
            drop_active_unsafe_loss, active_drop = (
                _active_unsafe_fixed_denominator_severity_loss(
                    drop_violations,
                    temperature=tau,
                )
            )
            drop_dense_tail_loss = _dense_fixed_denominator_softplus_loss(
                drop_violations,
                temperature=tau,
            )
            if bool(active_drop.any().item()):
                active_unsafe_drop_rows += 1
            active_unsafe_drop_queries += int(active_drop.sum().item())
            drop_loss = drop_barrier_loss + float(
                active_unsafe_auxiliary_weight
            ) * drop_active_unsafe_loss
            if float(drop_dense_tail_weight) > 0.0:
                drop_loss = drop_loss + float(
                    drop_dense_tail_weight
                ) * drop_dense_tail_loss
            row_drop_losses.append(drop_loss)
            row_drop_barrier_losses.append(drop_barrier_loss)
            row_drop_active_unsafe_losses.append(drop_active_unsafe_loss)
            row_drop_dense_tail_losses.append(drop_dense_tail_loss)
            # Give category retention and rejection equal row-level influence.
            # Averaging every instance term with one drop term makes rejection
            # vanish on crowded images (for n instances: weight 1/(n+1)).
            row_losses.append(
                0.5 * (keep_loss + drop_loss)
                + float(dense_category_focal_weight)
                * dense_category_focal_loss
            )
            valid_drop_rows += 1
            drop_safe_rows += int(
                bool((positive_negative_gap.detach() >= drop_limit).item())
            )
            drop_deployed_rows += int(
                bool(
                    (
                        positive_negative_gap.detach()
                        > float(category_gate_max_gap)
                    ).item()
                )
            )
        else:
            row_losses.append(
                keep_loss
                + float(dense_category_focal_weight)
                * dense_category_focal_loss
            )

    zero = flat_patch_score.float().sum() * 0.0
    loss = torch.stack(row_losses).mean() if row_losses else zero
    keep_objective_component = (
        torch.stack(row_keep_losses).mean() if row_keep_losses else zero
    )
    keep_component = (
        torch.stack(row_keep_barrier_losses).mean()
        if row_keep_barrier_losses
        else zero
    )
    keep_active_unsafe_component = (
        torch.stack(row_keep_active_unsafe_losses).mean()
        if row_keep_active_unsafe_losses
        else zero
    )
    generic_keep_active_unsafe_component = (
        torch.stack(row_generic_keep_active_unsafe_losses).mean()
        if row_generic_keep_active_unsafe_losses
        else zero
    )
    role_exclusive_keep_active_unsafe_component = (
        torch.stack(row_role_exclusive_keep_active_unsafe_losses).mean()
        if row_role_exclusive_keep_active_unsafe_losses
        else zero
    )
    keep_mean_component = (
        torch.stack(row_keep_mean_losses).mean()
        if row_keep_mean_losses
        else zero
    )
    role_exclusive_keep_component = (
        torch.stack(row_role_exclusive_keep_losses).mean()
        if row_role_exclusive_keep_losses
        else zero
    )
    drop_objective_component = (
        torch.stack(row_drop_losses).mean() if row_drop_losses else zero
    )
    drop_component = (
        torch.stack(row_drop_barrier_losses).mean()
        if row_drop_barrier_losses
        else zero
    )
    drop_active_unsafe_component = (
        torch.stack(row_drop_active_unsafe_losses).mean()
        if row_drop_active_unsafe_losses
        else zero
    )
    drop_dense_tail_component = (
        torch.stack(row_drop_dense_tail_losses).mean()
        if row_drop_dense_tail_losses
        else zero
    )
    dense_category_focal_component = (
        torch.stack(row_dense_category_focal_losses).mean()
        if row_dense_category_focal_losses
        else zero
    )
    dense_category_positive_focal_component = (
        torch.stack(row_dense_category_positive_focal_losses).mean()
        if row_dense_category_positive_focal_losses
        else zero
    )
    dense_category_negative_focal_component = (
        torch.stack(row_dense_category_negative_focal_losses).mean()
        if row_dense_category_negative_focal_losses
        else zero
    )
    scalar = flat_patch_score.new_tensor
    return {
        "loss": loss,
        "keep_component": keep_component,
        "keep_objective_component": keep_objective_component,
        "keep_active_unsafe_component": keep_active_unsafe_component,
        "generic_keep_active_unsafe_component": (
            generic_keep_active_unsafe_component
        ),
        "role_exclusive_keep_active_unsafe_component": (
            role_exclusive_keep_active_unsafe_component
        ),
        "keep_mean_component": keep_mean_component,
        "role_exclusive_keep_component": role_exclusive_keep_component,
        "drop_component": drop_component,
        "drop_objective_component": drop_objective_component,
        "drop_active_unsafe_component": drop_active_unsafe_component,
        "drop_dense_tail_component": drop_dense_tail_component,
        "dense_category_focal_component": dense_category_focal_component,
        "dense_category_positive_focal_component": (
            dense_category_positive_focal_component
        ),
        "dense_category_negative_focal_component": (
            dense_category_negative_focal_component
        ),
        "deployed_gate": deployed_gate.detach(),
        "standardized_patch_score": deployed_standardized.detach(),
        "category_positive_mask": category_positive_mask.detach(),
        "category_negative_mask": category_negative_mask.detach(),
        "category_neutral_mask": category_neutral_mask.detach(),
        "role_exclusive_positive_mask": (
            role_exclusive_positive_mask.detach()
        ),
        "valid_instances": scalar(float(valid_instances)).detach(),
        "skipped_instances": scalar(float(skipped_instances)).detach(),
        "keep_safe_instances": scalar(float(keep_safe_instances)).detach(),
        "keep_deployed_instances": scalar(
            float(keep_deployed_instances)
        ).detach(),
        "role_exclusive_reachable_instances": scalar(
            float(role_exclusive_reachable_instances)
        ).detach(),
        "role_exclusive_unreachable_instances": scalar(
            float(role_exclusive_unreachable_instances)
        ).detach(),
        "role_exclusive_keep_safe_instances": scalar(
            float(role_exclusive_keep_safe_instances)
        ).detach(),
        "role_exclusive_keep_deployed_instances": scalar(
            float(role_exclusive_keep_deployed_instances)
        ).detach(),
        "active_unsafe_generic_keep_constraints": scalar(
            float(active_unsafe_generic_keep_constraints)
        ).detach(),
        "active_unsafe_role_exclusive_keep_constraints": scalar(
            float(active_unsafe_role_exclusive_keep_constraints)
        ).detach(),
        "active_unsafe_keep_rows": scalar(
            float(active_unsafe_keep_rows)
        ).detach(),
        "valid_drop_rows": scalar(float(valid_drop_rows)).detach(),
        "drop_safe_rows": scalar(float(drop_safe_rows)).detach(),
        "drop_deployed_rows": scalar(float(drop_deployed_rows)).detach(),
        "active_unsafe_drop_queries": scalar(
            float(active_unsafe_drop_queries)
        ).detach(),
        "active_unsafe_drop_rows": scalar(
            float(active_unsafe_drop_rows)
        ).detach(),
    }


def _trace_lexical_tokens(value: Any) -> list[dict[str, Any]]:
    text = str(value or "")
    return [
        {
            "text": match.group(0),
            "norm": match.group(0).lower(),
            "start": int(match.start()),
            "end": int(match.end()),
        }
        for match in _TRACE_TOKEN_RE.finditer(text)
    ]


def _strict_false_scalar(value: Any) -> bool:
    return bool(
        torch.is_tensor(value)
        and value.dtype == torch.bool
        and value.numel() == 1
        and not bool(value.detach().reshape(-1)[0].item())
    )


def _strict_true_scalar(value: Any) -> bool:
    return bool(
        torch.is_tensor(value)
        and value.dtype == torch.bool
        and value.numel() == 1
        and bool(value.detach().reshape(-1)[0].item())
    )


def _validate_exact_trace_reconstruction(
    positive_expression: str,
    negative_expression: str,
    trace: Mapping[str, Any],
    *,
    target_index: int,
) -> None:
    positive = [token["norm"] for token in _trace_lexical_tokens(
        positive_expression
    )]
    negative = [token["norm"] for token in _trace_lexical_tokens(
        negative_expression
    )]
    replace_from = [
        token["norm"] for token in _trace_lexical_tokens(trace["replace_from"])
    ]
    replace_to = [
        token["norm"] for token in _trace_lexical_tokens(trace["replace_to"])
    ]
    span = trace["replace_span"]
    start, end = int(span[0]), int(span[1])
    if (
        not positive
        or not negative
        or not replace_from
        or not replace_to
        or end > len(positive)
        or positive[start:end] != replace_from
        or positive[:start] + replace_to + positive[end:] != negative
    ):
        raise ValueError(
            "data-driven TN trace does not exactly reconstruct the negative "
            f"expression at target {target_index}"
        )


def _validate_confidence_pair_targets(
    targets: Sequence[Mapping[str, Any]],
) -> list[list[str]]:
    pairs: list[list[str]] = []
    for index, target in enumerate(targets):
        expressions = target.get("stage_b_data_driven_expression_captions")
        trace = target.get("stage_b_data_driven_trace")
        if not (
            isinstance(expressions, (list, tuple))
            and len(expressions) == 2
            and all(
                isinstance(expression, str) and bool(expression.strip())
                for expression in expressions
            )
        ):
            raise ValueError(
                f"data-driven confidence target {index} lost its expression pair"
            )
        if target.get("tn_scope") != DATA_DRIVEN_GLOBAL_TN_SCOPE:
            raise ValueError(
                "global-max confidence requires an image-global verified TN; "
                f"target {index} has scope={target.get('tn_scope')!r}"
            )
        if not _strict_true_scalar(target.get("global_tn_verified")):
            raise ValueError(
                "global-max confidence requires exact "
                f"global_tn_verified=true at target {index}"
            )
        if not isinstance(trace, Mapping):
            raise ValueError(
                f"data-driven confidence target {index} has no immutable edit trace"
            )
        required_text = ("category", "replace_from", "replace_to")
        if any(
            not isinstance(trace.get(key), str) or not trace[key].strip()
            for key in required_text
        ):
            raise ValueError(
                f"data-driven confidence target {index} has a malformed edit trace"
            )
        span = trace.get("replace_span")
        if not (
            isinstance(span, list)
            and len(span) == 2
            and all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in span
            )
            and 0 <= span[0] < span[1]
        ):
            raise ValueError(
                f"data-driven confidence target {index} has an invalid trace span"
            )
        _validate_exact_trace_reconstruction(
            str(expressions[0]),
            str(expressions[1]),
            trace,
            target_index=index,
        )
        pairs.append([str(expressions[0]), str(expressions[1])])
    return pairs


def build_direct_trace_token_roles(
    tokenizer: Any,
    expression_pairs: Sequence[Sequence[str]],
    traces: Sequence[Mapping[str, Any]],
    expression_input_ids: Tensor,
    expression_token_mask: Tensor,
    *,
    max_text_len: int,
    allow_incidental_edits: bool = False,
) -> dict[str, Tensor]:
    """Map exact single-edit traces to positive/shared/changed BERT tokens.

    By default a row is token-valid only when applying the declared lexical
    span and replacement reconstructs the complete TN lexical sequence.  With
    ``allow_incidental_edits=True``, tokens changed outside the declared trace
    are ignored rather than mislabeled as either shared-positive or edited.
    """
    batch_size = len(expression_pairs)
    if batch_size <= 0 or len(traces) != batch_size:
        raise ValueError("direct-trace pairs and traces must be non-empty and aligned")
    if any(len(pair) != 2 for pair in expression_pairs):
        raise ValueError("direct-trace expressions must have exactly two slots")
    if expression_input_ids.dim() != 3 or tuple(
        expression_input_ids.shape[:2]
    ) != (batch_size, 2):
        raise ValueError("direct-trace input IDs must have shape (B,2,T)")
    if tuple(expression_token_mask.shape) != tuple(expression_input_ids.shape):
        raise ValueError("direct-trace token mask must align with input IDs")
    if type(allow_incidental_edits) is not bool:
        raise TypeError("allow_incidental_edits must be a boolean")

    flat_expressions = [expression for pair in expression_pairs for expression in pair]
    try:
        tokenized = tokenizer(
            flat_expressions,
            padding="longest",
            return_tensors="pt",
            return_offsets_mapping=True,
        )
    except (TypeError, NotImplementedError) as error:
        raise RuntimeError(
            "DD3 direct-trace supervision requires a fast tokenizer with offsets"
        ) from error
    input_ids = tokenized["input_ids"][:, : int(max_text_len)].reshape(
        batch_size, 2, -1
    )
    expected_ids = expression_input_ids.detach().to(device="cpu", dtype=input_ids.dtype)
    if tuple(input_ids.shape) != tuple(expected_ids.shape) or not torch.equal(
        input_ids, expected_ids
    ):
        raise RuntimeError(
            "criterion tokenization drifted from the model expression tokenization"
        )
    offsets = tokenized["offset_mapping"][:, : input_ids.shape[-1]].reshape(
        batch_size, 2, input_ids.shape[-1], 2
    )
    model_mask = expression_token_mask.detach().to(device="cpu", dtype=torch.bool)
    positive = torch.zeros_like(model_mask)
    shared = torch.zeros_like(model_mask)
    changed = torch.zeros_like(model_mask)
    valid = torch.zeros((batch_size,), dtype=torch.bool)

    for row_index, (pair, trace) in enumerate(zip(expression_pairs, traces)):
        if not isinstance(trace, Mapping):
            continue
        positive_tokens = _trace_lexical_tokens(pair[0])
        negative_tokens = _trace_lexical_tokens(pair[1])
        from_tokens = _trace_lexical_tokens(trace.get("replace_from"))
        to_tokens = _trace_lexical_tokens(trace.get("replace_to"))
        span = trace.get("replace_span")
        if not (
            isinstance(span, list)
            and len(span) == 2
            and all(isinstance(value, int) and not isinstance(value, bool) for value in span)
        ):
            continue
        declared_start, declared_end = int(span[0]), int(span[1])
        positive_norm = [token["norm"] for token in positive_tokens]
        negative_norm = [token["norm"] for token in negative_tokens]
        from_norm = [token["norm"] for token in from_tokens]
        to_norm = [token["norm"] for token in to_tokens]
        if not from_norm or not to_norm:
            continue
        source_spans = []
        reconstructing_spans = []
        for candidate_start in range(0, len(positive_norm) - len(from_norm) + 1):
            candidate_end = candidate_start + len(from_norm)
            if positive_norm[candidate_start:candidate_end] != from_norm:
                continue
            source_spans.append((candidate_start, candidate_end))
            reconstructed = (
                positive_norm[:candidate_start]
                + to_norm
                + positive_norm[candidate_end:]
            )
            if reconstructed == negative_norm:
                reconstructing_spans.append((candidate_start, candidate_end))
        declared_span = (declared_start, declared_end)
        if declared_span in reconstructing_spans:
            start, end = declared_span
        elif len(reconstructing_spans) == 1:
            # Some sealed rows have a stale span width. Recover only when the
            # immutable replace_from text identifies one exact reconstruction.
            start, end = reconstructing_spans[0]
        elif allow_incidental_edits and declared_span in source_spans:
            start, end = declared_span
        elif allow_incidental_edits and len(source_spans) == 1:
            start, end = source_spans[0]
        else:
            continue
        changed_to_indices: list[int] = []
        for tag, _i1, _i2, j1, j2 in SequenceMatcher(
            None, from_norm, to_norm
        ).get_opcodes():
            if tag in {"replace", "insert"}:
                changed_to_indices.extend(range(int(j1), int(j2)))
        if not changed_to_indices:
            continue

        expected_negative = (
            positive_norm[:start] + to_norm + positive_norm[end:]
        )
        expected_to_indices = list(range(start, start + len(to_norm)))
        expected_to_negative: dict[int, int] = {}
        shared_negative_indices: set[int] = set()
        for tag, i1, i2, j1, _j2 in SequenceMatcher(
            None, expected_negative, negative_norm, autojunk=False
        ).get_opcodes():
            if tag != "equal":
                continue
            for offset in range(int(i2) - int(i1)):
                expected_index = int(i1) + offset
                negative_index = int(j1) + offset
                shared_negative_indices.add(negative_index)
                if expected_index in expected_to_indices:
                    expected_to_negative[expected_index] = negative_index
        mapped_to_indices = [
            expected_to_negative.get(expected_index)
            for expected_index in expected_to_indices
        ]
        if any(index is None for index in mapped_to_indices):
            continue
        mapped_to_indices = [int(index) for index in mapped_to_indices]
        if mapped_to_indices != list(
            range(mapped_to_indices[0], mapped_to_indices[0] + len(to_norm))
        ):
            continue
        if [negative_norm[index] for index in mapped_to_indices] != to_norm:
            continue
        if not allow_incidental_edits and expected_negative != negative_norm:
            continue

        negative_offsets = offsets[row_index, 1]
        all_changed_tokens_covered = True
        changed_mask = torch.zeros_like(model_mask[row_index, 1])
        for local_index in changed_to_indices:
            lexical_index = mapped_to_indices[int(local_index)]
            if lexical_index >= len(negative_tokens):
                all_changed_tokens_covered = False
                break
            lexical = negative_tokens[lexical_index]
            overlap = (
                (negative_offsets[:, 1] > int(lexical["start"]))
                & (negative_offsets[:, 0] < int(lexical["end"]))
                & model_mask[row_index, 1]
            )
            if not bool(overlap.any().item()):
                all_changed_tokens_covered = False
                break
            changed_mask |= overlap
        if not all_changed_tokens_covered or not bool(changed_mask.any().item()):
            continue
        positive_mask = model_mask[row_index, 0]
        if not bool(positive_mask.any().item()):
            continue
        shared_mask = torch.zeros_like(model_mask[row_index, 1])
        for lexical_index in shared_negative_indices:
            lexical = negative_tokens[lexical_index]
            shared_mask |= (
                (negative_offsets[:, 1] > int(lexical["start"]))
                & (negative_offsets[:, 0] < int(lexical["end"]))
                & model_mask[row_index, 1]
            )
        positive[row_index, 0] = positive_mask
        changed[row_index, 1] = changed_mask
        shared[row_index, 1] = shared_mask & ~changed_mask
        valid[row_index] = True

    device = expression_token_mask.device
    return {
        "positive": positive.to(device=device),
        "shared": shared.to(device=device),
        "changed": changed.to(device=device),
        "valid": valid.to(device=device),
    }


def _distributed_weighted_token_mean(
    local_numerator: Tensor, local_denominator: Tensor
) -> Tensor:
    """Return a global weighted mean with DDP-correct local gradients."""
    if local_numerator.numel() != 1 or local_denominator.numel() != 1:
        raise ValueError("token numerator and denominator must be scalar")
    if not (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
    ):
        return torch.where(
            local_denominator > 0,
            local_numerator / local_denominator.clamp_min(1.0),
            local_numerator * 0.0,
        )
    world_size = torch.distributed.get_world_size()
    global_denominator = local_denominator.detach().clone()
    global_numerator = local_numerator.detach().clone()
    torch.distributed.all_reduce(global_denominator, op=torch.distributed.ReduceOp.SUM)
    torch.distributed.all_reduce(global_numerator, op=torch.distributed.ReduceOp.SUM)
    if not bool((global_denominator > 0).item()):
        return local_numerator * 0.0
    reported = global_numerator / global_denominator
    local_ratio = local_numerator / global_denominator
    return reported.detach() + float(world_size) * (
        local_ratio - local_ratio.detach()
    )


class StageBDataDrivenCriterion(nn.Module):
    """Direct GT supervision with no frozen score or teacher target."""

    def __init__(
        self,
        *,
        train_mode: str,
        category_complete: bool,
        rank_supervision: str = DATA_DRIVEN_RANK_SUPERVISION_ALL_NONPOSITIVE,
        tokenizer: Any = None,
        max_text_len: int = 256,
        rank_weight: float = 1.0,
        assignment_weight: float = 0.0,
        deployment_weight: float = 0.0,
        patch_weight: float = 1.0,
        confidence_weight: float = 1.0,
        token_weight: float = 0.0,
        shared_token_weight: float = 0.25,
        positive_iou_threshold: float = 0.5,
        rank_negative_iou_threshold: float = 0.3,
        patch_negative_iou_threshold: float = 0.3,
        temperature: float = 0.1,
        rank_margin: float = 0.1,
        category_margin: float = 0.1,
        category_gate_max_gap: float = 3.0,
        category_gate_boundary_margin: float = 0.25,
        patch_active_unsafe_auxiliary_weight: float = 1.0,
        patch_dense_category_focal_weight: float = 1.0,
        patch_dense_category_focal_alpha: float = 0.25,
        patch_dense_category_focal_gamma: float = 2.0,
        patch_dense_category_focal_negative_weight: float = 1.0,
        patch_drop_positive_anchor_gradient_policy: str = (
            DATA_DRIVEN_PATCH_DROP_ANCHOR_GLOBAL_MAX
        ),
        patch_score_clip: float = 5.0,
        fpr_temperature: float = 0.1,
        fpr_margin: float = 0.0,
        target_tpr: float = 0.95,
        positive_queue_size: int = 4096,
    ) -> None:
        super().__init__()
        self.train_mode = normalize_data_driven_train_mode(train_mode)
        self.category_complete = bool(category_complete)
        self.rank_supervision = normalize_data_driven_rank_supervision(
            rank_supervision
        )
        self.tokenizer = tokenizer
        self.max_text_len = int(max_text_len)
        self.positive_iou_threshold = float(positive_iou_threshold)
        self.rank_negative_iou_threshold = float(
            rank_negative_iou_threshold
        )
        self.patch_negative_iou_threshold = float(patch_negative_iou_threshold)
        self.temperature = float(temperature)
        self.rank_margin = float(rank_margin)
        self.category_margin = float(category_margin)
        self.category_gate_max_gap = float(category_gate_max_gap)
        self.category_gate_boundary_margin = float(
            category_gate_boundary_margin
        )
        self.patch_active_unsafe_auxiliary_weight = float(
            patch_active_unsafe_auxiliary_weight
        )
        self.patch_dense_category_focal_weight = float(
            patch_dense_category_focal_weight
        )
        self.patch_dense_category_focal_alpha = float(
            patch_dense_category_focal_alpha
        )
        self.patch_dense_category_focal_gamma = float(
            patch_dense_category_focal_gamma
        )
        self.patch_dense_category_focal_negative_weight = float(
            patch_dense_category_focal_negative_weight
        )
        self.patch_drop_positive_anchor_gradient_policy = str(
            patch_drop_positive_anchor_gradient_policy
        )
        self.patch_score_clip = float(patch_score_clip)
        self.token_weight = float(token_weight)
        self.assignment_weight = float(assignment_weight)
        self.deployment_weight = float(deployment_weight)
        self.shared_token_weight = float(shared_token_weight)
        self.fpr_temperature = float(fpr_temperature)
        self.fpr_margin = float(fpr_margin)
        self.target_tpr = float(target_tpr)
        self.positive_queue_size = int(positive_queue_size)
        if not 0.0 < self.positive_iou_threshold <= 1.0:
            raise ValueError("positive_iou_threshold must be in (0,1]")
        if not (
            0.0
            <= self.rank_negative_iou_threshold
            < self.positive_iou_threshold
        ):
            raise ValueError("rank negative IoU threshold is invalid")
        if not 0.0 <= self.patch_negative_iou_threshold < self.positive_iou_threshold:
            raise ValueError("patch negative IoU threshold is invalid")
        if (
            self.rank_supervision in _DATA_DRIVEN_ROLE_ROUTED_SUPERVISIONS
            and self.patch_negative_iou_threshold
            != self.rank_negative_iou_threshold
        ):
            raise ValueError(
                "role-routed patch and rank negative IoU thresholds must match"
            )
        if (
            self.temperature <= 0.0
            or not math.isfinite(self.rank_margin)
            or self.rank_margin < 0.0
            or self.category_margin < 0.0
            or not math.isfinite(self.category_gate_max_gap)
            or self.category_gate_max_gap <= 0.0
            or not math.isfinite(self.category_gate_boundary_margin)
            or not (
                0.0
                <= self.category_gate_boundary_margin
                < self.category_gate_max_gap
            )
            or not math.isfinite(self.patch_score_clip)
            or self.patch_score_clip <= 0.0
            or not math.isfinite(
                self.patch_active_unsafe_auxiliary_weight
            )
            or not (
                0.0 < self.patch_active_unsafe_auxiliary_weight <= 1.0
            )
            or not math.isfinite(self.patch_dense_category_focal_weight)
            or self.patch_dense_category_focal_weight < 0.0
            or not (
                0.0 <= self.patch_dense_category_focal_alpha <= 1.0
            )
            or not math.isfinite(self.patch_dense_category_focal_gamma)
            or self.patch_dense_category_focal_gamma < 0.0
            or not math.isfinite(
                self.patch_dense_category_focal_negative_weight
            )
            or self.patch_dense_category_focal_negative_weight < 0.0
            or self.patch_drop_positive_anchor_gradient_policy
            not in DATA_DRIVEN_PATCH_DROP_ANCHOR_POLICIES
            or (
                self.category_gate_max_gap
                + self.category_gate_boundary_margin
                >= 2.0 * self.patch_score_clip
            )
        ):
            raise ValueError("temperature/category margin is invalid")
        if (
            self.max_text_len <= 0
            or self.fpr_temperature <= 0.0
            or not 0.0 < self.target_tpr <= 1.0
            or self.positive_queue_size < 0
            or self.token_weight < 0.0
            or not math.isfinite(self.assignment_weight)
            or self.assignment_weight < 0.0
            or not 0.0 <= self.shared_token_weight <= 1.0
        ):
            raise ValueError("data-driven confidence/token settings are invalid")
        if (
            not math.isfinite(self.deployment_weight)
            or self.deployment_weight < 0.0
        ):
            raise ValueError("deployment_weight must be finite and nonnegative")
        if self.train_mode == "rank_patch_only":
            if (
                self.rank_supervision
                in {
                    DATA_DRIVEN_RANK_SUPERVISION_SAME_CATEGORY,
                    DATA_DRIVEN_RANK_SUPERVISION_SAME_CATEGORY_GAP3_COVERAGE,
                    DATA_DRIVEN_RANK_SUPERVISION_OFFICIAL_ASSIGNMENT,
                    DATA_DRIVEN_RANK_SUPERVISION_ROLE_ROUTED_ASSIGNMENT,
                    DATA_DRIVEN_RANK_SUPERVISION_ROLE_ROUTED_ALL_NONOWNED,
                }
                and not self.category_complete
            ):
                raise ValueError(
                    "same-category rank supervision requires category-complete rows"
                )
            if self.rank_supervision in {
                DATA_DRIVEN_RANK_SUPERVISION_OFFICIAL_ASSIGNMENT,
                DATA_DRIVEN_RANK_SUPERVISION_ROLE_ROUTED_ASSIGNMENT,
                DATA_DRIVEN_RANK_SUPERVISION_ROLE_ROUTED_ALL_NONOWNED,
            }:
                if self.assignment_weight <= 0.0:
                    raise ValueError(
                        "official assignment supervision requires a positive "
                        "assignment_weight"
                    )
                if (
                    self.rank_supervision
                    in _DATA_DRIVEN_ROLE_ROUTED_SUPERVISIONS
                    and self.deployment_weight != 0.0
                ):
                    raise ValueError(
                        "role-routed assignment forbids the legacy all-other "
                        "deployment-hard objective"
                    )
                rank_loss_key = (
                    "loss_stage_b_data_driven_role_routed_rank"
                    if self.rank_supervision
                    in _DATA_DRIVEN_ROLE_ROUTED_SUPERVISIONS
                    else "loss_stage_b_data_driven_assignment"
                )
                self.weight_dict = {
                    rank_loss_key: self.assignment_weight,
                    "loss_stage_b_data_driven_patch": float(patch_weight),
                }
                if self.deployment_weight > 0.0:
                    self.weight_dict[
                        "loss_stage_b_data_driven_deployment_hard"
                    ] = self.deployment_weight
            else:
                self.weight_dict = {
                    "loss_stage_b_data_driven_rank": float(rank_weight),
                    "loss_stage_b_data_driven_patch": float(patch_weight),
                }
        else:
            if not self.category_complete:
                raise ValueError(
                    "confidence_pair must start from the category-complete DD1 phase"
                )
            if self.token_weight > 0.0 and self.tokenizer is None:
                raise ValueError("DD3 token supervision requires the model tokenizer")
            self.weight_dict = {
                "loss_stage_b_data_driven_confidence": float(confidence_weight),
                "loss_stage_b_data_driven_token": self.token_weight,
            }
        self.register_buffer(
            "fpr_positive_queue",
            torch.zeros((self.positive_queue_size,), dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "fpr_positive_queue_count",
            torch.zeros((), dtype=torch.int64),
            persistent=True,
        )
        self.register_buffer(
            "fpr_positive_queue_cursor",
            torch.zeros((), dtype=torch.int64),
            persistent=True,
        )
        criterion_contract_version = _DATA_DRIVEN_CURRENT_CRITERION_CONTRACTS[
            self.rank_supervision
        ][0]
        if (
            self.rank_supervision
            == DATA_DRIVEN_RANK_SUPERVISION_ROLE_ROUTED_ALL_NONOWNED
        ):
            criterion_contract_version = (
                18
                if self.patch_drop_positive_anchor_gradient_policy
                == DATA_DRIVEN_PATCH_DROP_ANCHOR_INSTANCE_BALANCED
                else 17
            )
        elif (
            self.patch_drop_positive_anchor_gradient_policy
            != DATA_DRIVEN_PATCH_DROP_ANCHOR_GLOBAL_MAX
        ):
            raise ValueError(
                "instance-balanced drop anchors require role-routed all-nonowned "
                "supervision"
            )
        self.register_buffer(
            "criterion_contract_version",
            torch.as_tensor(
                criterion_contract_version,
                dtype=torch.int64,
            ),
            persistent=True,
        )
        self.register_buffer(
            "rank_supervision_contract_id",
            torch.as_tensor(
                {
                    DATA_DRIVEN_RANK_SUPERVISION_ALL_NONPOSITIVE: 1,
                    DATA_DRIVEN_RANK_SUPERVISION_SAME_CATEGORY: 2,
                    DATA_DRIVEN_RANK_SUPERVISION_SAME_CATEGORY_GAP3_COVERAGE: 3,
                    DATA_DRIVEN_RANK_SUPERVISION_OFFICIAL_ASSIGNMENT: 4,
                    DATA_DRIVEN_RANK_SUPERVISION_ROLE_ROUTED_ASSIGNMENT: 5,
                    DATA_DRIVEN_RANK_SUPERVISION_ROLE_ROUTED_ALL_NONOWNED: 6,
                }[self.rank_supervision],
                dtype=torch.int64,
            ),
            persistent=True,
        )
        self._pending_queue_payload: Optional[Tensor] = None
        self._deferred_queue_payloads: list[Tensor] = []

    def _positive_history(self) -> Optional[Tensor]:
        count = int(self.fpr_positive_queue_count.item())
        if count <= 0:
            return None
        return self.fpr_positive_queue[:count].detach()

    @torch.no_grad()
    def _commit_positive_history(self, values: Tensor) -> None:
        size = self.positive_queue_size
        if size <= 0:
            return
        values = values.detach().to(
            device=self.fpr_positive_queue.device, dtype=torch.float32
        ).reshape(-1)
        if values.numel() == 0:
            return
        if int(values.numel()) >= size:
            self.fpr_positive_queue.copy_(values[-size:])
            self.fpr_positive_queue_count.fill_(size)
            self.fpr_positive_queue_cursor.zero_()
            return
        cursor = int(self.fpr_positive_queue_cursor.item())
        first = min(int(values.numel()), size - cursor)
        self.fpr_positive_queue[cursor : cursor + first].copy_(values[:first])
        remaining = int(values.numel()) - first
        if remaining:
            self.fpr_positive_queue[:remaining].copy_(values[first:])
        self.fpr_positive_queue_cursor.fill_((cursor + int(values.numel())) % size)
        self.fpr_positive_queue_count.fill_(
            min(size, int(self.fpr_positive_queue_count.item()) + int(values.numel()))
        )

    @torch.no_grad()
    def defer_tail_queue_payload(self) -> None:
        payload = self._pending_queue_payload
        self._pending_queue_payload = None
        if payload is not None:
            self._deferred_queue_payloads.append(payload)

    @torch.no_grad()
    def commit_tail_queue(self, step_succeeded: bool) -> None:
        payloads = list(self._deferred_queue_payloads)
        if self._pending_queue_payload is not None:
            payloads.append(self._pending_queue_payload)
        self._deferred_queue_payloads.clear()
        self._pending_queue_payload = None
        if not bool(step_succeeded) or not payloads:
            return
        payload = payloads[0] if len(payloads) == 1 else torch.cat(payloads)
        self._commit_positive_history(payload)

    def _rank_patch_forward(
        self,
        outputs: Mapping[str, Any],
        targets: Sequence[Mapping[str, Any]],
    ) -> dict[str, Tensor]:
        raw_rank_score = outputs.get("stage_b_data_driven_text_rank_score")
        patch_score = outputs.get("pred_logits_patch")
        boxes = outputs.get("pred_boxes")
        if not all(
            torch.is_tensor(value)
            for value in (raw_rank_score, patch_score, boxes)
        ):
            raise KeyError("data-driven rank/patch training outputs are incomplete")
        role_routed_assignment_mode = (
            self.rank_supervision in _DATA_DRIVEN_ROLE_ROUTED_SUPERVISIONS
        )
        assignment_mode = self.rank_supervision in {
            DATA_DRIVEN_RANK_SUPERVISION_OFFICIAL_ASSIGNMENT,
            DATA_DRIVEN_RANK_SUPERVISION_ROLE_ROUTED_ASSIGNMENT,
            DATA_DRIVEN_RANK_SUPERVISION_ROLE_ROUTED_ALL_NONOWNED,
        }
        legacy_assignment_mode = (
            self.rank_supervision
            == DATA_DRIVEN_RANK_SUPERVISION_OFFICIAL_ASSIGNMENT
        )
        paired_rank_score: Optional[Tensor] = None
        if assignment_mode:
            if (
                raw_rank_score.dim() != 3
                or int(raw_rank_score.shape[-1]) != 2
                or not raw_rank_score.is_floating_point()
            ):
                raise ValueError(
                    "official assignment rank scores must have shape (B,Q,2)"
                )
            paired_rank_score = raw_rank_score
            rank_score = raw_rank_score[..., 0]
        else:
            if raw_rank_score.dim() != 2 or not raw_rank_score.is_floating_point():
                raise ValueError("data-driven rank scores must have shape (B,Q)")
            rank_score = raw_rank_score
        if not bool(torch.isfinite(raw_rank_score).all().item()):
            raise ValueError("data-driven rank scores must all be finite")
        _validate_rank_patch_targets(
            targets, category_complete=self.category_complete
        )
        primary_iou = _candidate_max_iou(boxes, targets, primary_only=True)
        positive = primary_iou >= self.positive_iou_threshold
        candidate_mask = torch.ones_like(positive)
        eligible = torch.ones_like(positive)
        hard_negative = ~positive
        coverage_negative = torch.zeros_like(positive)
        gap3_mask = torch.zeros_like(positive)
        category_complete_supervision = self.rank_supervision in {
            DATA_DRIVEN_RANK_SUPERVISION_SAME_CATEGORY,
            DATA_DRIVEN_RANK_SUPERVISION_SAME_CATEGORY_GAP3_COVERAGE,
            DATA_DRIVEN_RANK_SUPERVISION_OFFICIAL_ASSIGNMENT,
            DATA_DRIVEN_RANK_SUPERVISION_ROLE_ROUTED_ASSIGNMENT,
            DATA_DRIVEN_RANK_SUPERVISION_ROLE_ROUTED_ALL_NONOWNED,
        }
        if category_complete_supervision:
            raw_candidate_mask = outputs.get(
                "stage_b_data_driven_candidate_mask"
            )
            expected_candidate_shape = (
                tuple(raw_rank_score.shape)
                if assignment_mode
                else tuple(rank_score.shape)
            )
            if (
                not torch.is_tensor(raw_candidate_mask)
                or raw_candidate_mask.dtype != torch.bool
                or tuple(raw_candidate_mask.shape) != expected_candidate_shape
            ):
                raise ValueError(
                    "same-category rank supervision requires an aligned boolean "
                    "candidate mask"
                )
            if assignment_mode:
                if not torch.equal(
                    raw_candidate_mask[..., 0], raw_candidate_mask[..., 1]
                ):
                    raise ValueError(
                        "paired official expressions changed the candidate set"
                    )
                candidate_mask = raw_candidate_mask[..., 0]
            else:
                candidate_mask = raw_candidate_mask
            primary_iou, auxiliary_iou = _candidate_primary_auxiliary_iou(
                boxes, targets
            )
            positive = (
                primary_iou >= self.positive_iou_threshold
            ) & candidate_mask
            hard_negative = (
                (auxiliary_iou >= self.positive_iou_threshold)
                & (primary_iou < self.rank_negative_iou_threshold)
                & candidate_mask
            )
            if (
                self.rank_supervision
                == DATA_DRIVEN_RANK_SUPERVISION_SAME_CATEGORY_GAP3_COVERAGE
            ):
                gap3_mask, _normalized_patch = data_driven_category_gate_mask(
                    patch_score.detach(),
                    candidate_mask,
                    max_gap=self.category_gate_max_gap,
                    clip=self.patch_score_clip,
                )
                coverage_negative = gap3_mask & ~(positive | hard_negative)
            if assignment_mode:
                # PairTop1 is the sole rank objective. Keep the DD1-H masks as
                # diagnostics, but give their legacy objective no gradient.
                rank_negative = hard_negative
                eligible = positive | rank_negative
                rank_loss = raw_rank_score.sum() * 0.0
            else:
                rank_negative = hard_negative | coverage_negative
                eligible = positive | rank_negative
                rank_loss = multi_positive_listwise_rank_loss(
                    rank_score,
                    positive,
                    eligible_mask=eligible,
                    temperature=self.temperature,
                )
        else:
            rank_negative = hard_negative
            rank_loss = multi_positive_listwise_rank_loss(
                rank_score,
                positive,
                temperature=self.temperature,
            )
        patch_contract: Optional[dict[str, Tensor]] = None
        if role_routed_assignment_mode:
            patch_contract = deployment_gate_category_patch_loss(
                patch_score,
                boxes,
                targets,
                candidate_mask,
                positive_iou_threshold=self.positive_iou_threshold,
                negative_iou_threshold=self.patch_negative_iou_threshold,
                category_gate_max_gap=self.category_gate_max_gap,
                patch_score_clip=self.patch_score_clip,
                boundary_margin=self.category_gate_boundary_margin,
                temperature=self.temperature,
                active_unsafe_auxiliary_weight=(
                    self.patch_active_unsafe_auxiliary_weight
                ),
                dense_category_focal_weight=(
                    self.patch_dense_category_focal_weight
                ),
                dense_category_focal_alpha=(
                    self.patch_dense_category_focal_alpha
                ),
                dense_category_focal_gamma=(
                    self.patch_dense_category_focal_gamma
                ),
                dense_category_focal_negative_weight=(
                    self.patch_dense_category_focal_negative_weight
                ),
                role_exclusive_keep=True,
                drop_positive_anchor_gradient_policy=(
                    self.patch_drop_positive_anchor_gradient_policy
                ),
            )
            patch_loss = patch_contract["loss"]
            valid_instances = patch_contract["valid_instances"]
            skipped_instances = patch_contract["skipped_instances"]
        else:
            patch_loss, valid_instances, skipped_instances = (
                _instance_complete_patch_margin_loss(
                    patch_score,
                    boxes,
                    targets,
                    positive_iou_threshold=self.positive_iou_threshold,
                    negative_iou_threshold=self.patch_negative_iou_threshold,
                    margin=self.category_margin,
                    temperature=self.temperature,
                )
            )
        with torch.no_grad():
            top = rank_score.masked_fill(~candidate_mask, -torch.inf).argmax(
                dim=1
            )
            correct = positive.gather(1, top[:, None]).squeeze(1) & positive.any(dim=1)
            valid_listwise = positive.any(dim=1) & rank_negative.any(dim=1)
            eligible_top = rank_score.masked_fill(~eligible, -torch.inf).argmax(dim=1)
            eligible_correct = (
                positive.gather(1, eligible_top[:, None]).squeeze(1)
                & valid_listwise
            )
            ignored = candidate_mask & ~eligible
            ignored_winner = (
                ignored.gather(1, top[:, None]).squeeze(1)
                & positive.any(dim=1)
            )
            if (
                self.rank_supervision
                == DATA_DRIVEN_RANK_SUPERVISION_SAME_CATEGORY_GAP3_COVERAGE
            ):
                gap3_top = rank_score.masked_fill(
                    ~gap3_mask, -torch.inf
                ).argmax(dim=1)
                gap3_correct = positive.gather(
                    1, gap3_top[:, None]
                ).squeeze(1) & (positive & gap3_mask).any(dim=1)
                gap3_coverage_winner = coverage_negative.gather(
                    1, gap3_top[:, None]
                ).squeeze(1)
            else:
                gap3_correct = torch.zeros_like(correct)
                gap3_coverage_winner = torch.zeros_like(correct)
        result = {
            "loss_stage_b_data_driven_rank": rank_loss,
            "loss_stage_b_data_driven_patch": patch_loss,
            "stage_b_data_driven_rank_correct": correct.float().sum(),
            "stage_b_data_driven_rank_rows": positive.any(dim=1).float().sum(),
            "stage_b_data_driven_rank_listwise_rows": valid_listwise.float().sum(),
            "stage_b_data_driven_rank_eligible_correct": eligible_correct.float().sum(),
            "stage_b_data_driven_rank_positive_queries": positive.float().sum(),
            "stage_b_data_driven_rank_hard_negative_queries": hard_negative.float().sum(),
            "stage_b_data_driven_rank_total_negative_queries": rank_negative.float().sum(),
            "stage_b_data_driven_rank_ignored_queries": ignored.float().sum(),
            "stage_b_data_driven_rank_skipped_no_positive_rows": (
                ~positive.any(dim=1)
            ).float().sum(),
            "stage_b_data_driven_rank_skipped_no_hard_negative_rows": (
                ~rank_negative.any(dim=1)
            ).float().sum(),
            "stage_b_data_driven_rank_ignored_winner_rows": ignored_winner.float().sum(),
            "stage_b_data_driven_patch_valid_instances": valid_instances,
            "stage_b_data_driven_patch_skipped_instances": skipped_instances,
        }
        if patch_contract is not None:
            result.update(
                {
                    "stage_b_data_driven_patch_keep_safe_instances": (
                        patch_contract["keep_safe_instances"]
                    ),
                    "stage_b_data_driven_patch_keep_component": (
                        patch_contract["keep_component"]
                    ),
                    "stage_b_data_driven_patch_keep_objective_component": (
                        patch_contract["keep_objective_component"]
                    ),
                    "stage_b_data_driven_patch_keep_active_unsafe_component": (
                        patch_contract["keep_active_unsafe_component"]
                    ),
                    "stage_b_data_driven_patch_generic_keep_active_unsafe_component": (
                        patch_contract[
                            "generic_keep_active_unsafe_component"
                        ]
                    ),
                    "stage_b_data_driven_patch_role_exclusive_keep_active_unsafe_component": (
                        patch_contract[
                            "role_exclusive_keep_active_unsafe_component"
                        ]
                    ),
                    "stage_b_data_driven_patch_keep_mean_component": (
                        patch_contract["keep_mean_component"]
                    ),
                    "stage_b_data_driven_patch_role_exclusive_keep_component": (
                        patch_contract["role_exclusive_keep_component"]
                    ),
                    "stage_b_data_driven_patch_drop_component": (
                        patch_contract["drop_component"]
                    ),
                    "stage_b_data_driven_patch_drop_objective_component": (
                        patch_contract["drop_objective_component"]
                    ),
                    "stage_b_data_driven_patch_drop_active_unsafe_component": (
                        patch_contract["drop_active_unsafe_component"]
                    ),
                    "stage_b_data_driven_patch_dense_category_focal_component": (
                        patch_contract["dense_category_focal_component"]
                    ),
                    "stage_b_data_driven_patch_dense_category_positive_focal_component": (
                        patch_contract[
                            "dense_category_positive_focal_component"
                        ]
                    ),
                    "stage_b_data_driven_patch_dense_category_negative_focal_component": (
                        patch_contract[
                            "dense_category_negative_focal_component"
                        ]
                    ),
                    "stage_b_data_driven_patch_keep_deployed_instances": (
                        patch_contract["keep_deployed_instances"]
                    ),
                    "stage_b_data_driven_patch_role_exclusive_reachable_instances": (
                        patch_contract[
                            "role_exclusive_reachable_instances"
                        ]
                    ),
                    "stage_b_data_driven_patch_role_exclusive_unreachable_instances": (
                        patch_contract[
                            "role_exclusive_unreachable_instances"
                        ]
                    ),
                    "stage_b_data_driven_patch_role_exclusive_keep_safe_instances": (
                        patch_contract[
                            "role_exclusive_keep_safe_instances"
                        ]
                    ),
                    "stage_b_data_driven_patch_role_exclusive_keep_deployed_instances": (
                        patch_contract[
                            "role_exclusive_keep_deployed_instances"
                        ]
                    ),
                    "stage_b_data_driven_patch_active_unsafe_generic_keep_constraints": (
                        patch_contract[
                            "active_unsafe_generic_keep_constraints"
                        ]
                    ),
                    "stage_b_data_driven_patch_active_unsafe_role_exclusive_keep_constraints": (
                        patch_contract[
                            "active_unsafe_role_exclusive_keep_constraints"
                        ]
                    ),
                    "stage_b_data_driven_patch_active_unsafe_keep_rows": (
                        patch_contract["active_unsafe_keep_rows"]
                    ),
                    "stage_b_data_driven_patch_valid_drop_rows": (
                        patch_contract["valid_drop_rows"]
                    ),
                    "stage_b_data_driven_patch_drop_safe_rows": (
                        patch_contract["drop_safe_rows"]
                    ),
                    "stage_b_data_driven_patch_drop_deployed_rows": (
                        patch_contract["drop_deployed_rows"]
                    ),
                    "stage_b_data_driven_patch_active_unsafe_drop_queries": (
                        patch_contract["active_unsafe_drop_queries"]
                    ),
                    "stage_b_data_driven_patch_active_unsafe_drop_rows": (
                        patch_contract["active_unsafe_drop_rows"]
                    ),
                    "stage_b_data_driven_patch_category_positive_queries": (
                        patch_contract["category_positive_mask"].float().sum()
                    ),
                    "stage_b_data_driven_patch_category_negative_queries": (
                        patch_contract["category_negative_mask"].float().sum()
                    ),
                    "stage_b_data_driven_patch_category_neutral_queries": (
                        patch_contract["category_neutral_mask"].float().sum()
                    ),
                    "stage_b_data_driven_patch_deployed_gate_queries": (
                        patch_contract["deployed_gate"].float().sum()
                    ),
                    "stage_b_data_driven_patch_deployed_category_negative_queries": (
                        (
                            patch_contract["deployed_gate"]
                            & patch_contract["category_negative_mask"]
                        )
                        .float()
                        .sum()
                    ),
                }
            )
        if role_routed_assignment_mode:
            if paired_rank_score is None:
                raise RuntimeError("role-routed assignment scores were not preserved")
            (
                assignment_iou,
                other_same_category_iou,
                pair_valid,
            ) = _candidate_official_assignment_role_iou(boxes, targets)
            assignment = role_routed_official_assignment_top1_loss(
                paired_rank_score,
                assignment_iou,
                other_same_category_iou,
                pair_valid,
                raw_candidate_mask,
                patch_score,
                positive_iou_threshold=self.positive_iou_threshold,
                negative_iou_threshold=self.rank_negative_iou_threshold,
                category_gate_max_gap=self.category_gate_max_gap,
                patch_score_clip=self.patch_score_clip,
                margin=self.rank_margin,
                temperature=self.temperature,
                include_all_exclusive_nonowned=(
                    self.rank_supervision
                    == DATA_DRIVEN_RANK_SUPERVISION_ROLE_ROUTED_ALL_NONOWNED
                ),
            )
            expected_owned = (
                patch_contract["role_exclusive_positive_mask"]
                & patch_contract["deployed_gate"][:, :, None]
            )
            if not torch.equal(expected_owned, assignment["owned_mask"]):
                raise RuntimeError(
                    "role-exclusive patch keep geometry drifted from the "
                    "deployed rank-owned mask"
                )
            runtime_direction = assignment["runtime_valid_direction"]
            runtime_valid = assignment["runtime_valid"]
            direction_delta = assignment["direction_delta"]
            direction_correct = (direction_delta > 0.0) & runtime_direction
            direction_margin = (
                direction_delta >= self.rank_margin
            ) & runtime_direction
            valid_delta = direction_delta[runtime_direction]
            winner_owned = (
                assignment["deployment_winner_owned"] & runtime_direction
            )
            winner_sibling = (
                assignment["deployment_winner_sibling"] & runtime_direction
            )
            winner_unassigned = (
                assignment["deployment_winner_unassigned"]
                & runtime_direction
            )
            winner_category_negative = (
                assignment["deployment_winner_category_negative"]
                & runtime_direction
            )
            winner_neutral = (
                assignment["deployment_winner_neutral"] & runtime_direction
            )
            result.update(
                {
                    "loss_stage_b_data_driven_role_routed_rank": assignment[
                        "loss"
                    ],
                    "stage_b_data_driven_assignment_data_rows": (
                        assignment["data_valid"].float().sum()
                    ),
                    "stage_b_data_driven_assignment_runtime_rows": (
                        runtime_valid.float().sum()
                    ),
                    "stage_b_data_driven_assignment_runtime_directions": (
                        runtime_direction.float().sum()
                    ),
                    "stage_b_data_driven_assignment_unreachable_rows": (
                        (
                            assignment["data_valid"] & ~runtime_valid
                        ).float().sum()
                    ),
                    "stage_b_data_driven_assignment_correct_rows": (
                        (direction_correct.all(dim=1) & runtime_valid)
                        .float()
                        .sum()
                    ),
                    "stage_b_data_driven_assignment_margin_rows": (
                        (direction_margin.all(dim=1) & runtime_valid)
                        .float()
                        .sum()
                    ),
                    "stage_b_data_driven_assignment_correct_directions": (
                        direction_correct.float().sum()
                    ),
                    "stage_b_data_driven_assignment_margin_directions": (
                        direction_margin.float().sum()
                    ),
                    "stage_b_data_driven_assignment_deployment_correct_rows": (
                        (winner_owned.all(dim=1) & runtime_valid)
                        .float()
                        .sum()
                    ),
                    "stage_b_data_driven_assignment_deployment_correct_directions": (
                        winner_owned.float().sum()
                    ),
                    "stage_b_data_driven_assignment_deployment_sibling_winners": (
                        winner_sibling.float().sum()
                    ),
                    "stage_b_data_driven_assignment_deployment_unassigned_winners": (
                        winner_unassigned.float().sum()
                    ),
                    "stage_b_data_driven_assignment_deployment_category_negative_winners": (
                        winner_category_negative.float().sum()
                    ),
                    "stage_b_data_driven_assignment_deployment_neutral_winners": (
                        winner_neutral.float().sum()
                    ),
                    "stage_b_data_driven_assignment_direction_delta_mean": (
                        valid_delta.mean()
                        if int(valid_delta.numel()) > 0
                        else direction_delta.sum() * 0.0
                    ),
                    "stage_b_data_driven_assignment_role0_queries": (
                        assignment["owned_mask"][..., 0].float().sum()
                    ),
                    "stage_b_data_driven_assignment_role1_queries": (
                        assignment["owned_mask"][..., 1].float().sum()
                    ),
                    "stage_b_data_driven_assignment_safe_sibling_queries": (
                        assignment["safe_sibling_mask"].float().sum()
                    ),
                    "stage_b_data_driven_assignment_paired_sibling_queries": (
                        assignment["paired_sibling_mask"].float().sum()
                    ),
                    "stage_b_data_driven_assignment_category_negative_queries": (
                        assignment["category_negative_mask"].float().sum()
                    ),
                    "stage_b_data_driven_assignment_neutral_queries": (
                        assignment["neutral_mask"].float().sum()
                    ),
                    "stage_b_data_driven_assignment_gap3_queries": assignment[
                        "gap_queries"
                    ],
                }
            )
        if legacy_assignment_mode:
            if paired_rank_score is None:
                raise RuntimeError("official assignment scores were not preserved")
            assignment_iou, pair_valid = _candidate_official_assignment_iou(
                boxes, targets
            )
            assignment = official_assignment_delta_loss(
                paired_rank_score,
                assignment_iou,
                pair_valid,
                raw_candidate_mask,
                patch_score,
                positive_iou_threshold=self.positive_iou_threshold,
                negative_iou_threshold=self.rank_negative_iou_threshold,
                category_gate_max_gap=self.category_gate_max_gap,
                patch_score_clip=self.patch_score_clip,
                temperature=self.temperature,
            )
            runtime_valid = assignment["runtime_valid"]
            delta = assignment["delta"]
            direction_delta = assignment["direction_delta"]
            valid_delta = delta[runtime_valid]
            valid_direction_delta = direction_delta[runtime_valid]
            direction_correct = (direction_delta > 0.0) & runtime_valid[:, None]
            direction_margin = (
                direction_delta >= self.temperature
            ) & runtime_valid[:, None]
            reciprocal_correct = direction_correct.all(dim=1) & runtime_valid
            reciprocal_margin = direction_margin.all(dim=1) & runtime_valid
            deployment_direction = assignment[
                "deployment_correct_direction"
            ]
            deployment_reciprocal = deployment_direction.all(dim=1) & runtime_valid
            result.update(
                {
                    "loss_stage_b_data_driven_assignment": assignment["loss"],
                    "stage_b_data_driven_assignment_data_rows": assignment[
                        "data_valid"
                    ].float().sum(),
                    "stage_b_data_driven_assignment_runtime_rows": (
                        runtime_valid.float().sum()
                    ),
                    "stage_b_data_driven_assignment_unreachable_rows": (
                        (
                            assignment["data_valid"]
                            & ~runtime_valid
                        ).float().sum()
                    ),
                    "stage_b_data_driven_assignment_correct_rows": (
                        reciprocal_correct.float().sum()
                    ),
                    "stage_b_data_driven_assignment_margin_rows": (
                        reciprocal_margin.float().sum()
                    ),
                    "stage_b_data_driven_assignment_correct_directions": (
                        direction_correct.float().sum()
                    ),
                    "stage_b_data_driven_assignment_margin_directions": (
                        direction_margin.float().sum()
                    ),
                    "stage_b_data_driven_assignment_deployment_correct_rows": (
                        deployment_reciprocal.float().sum()
                    ),
                    "stage_b_data_driven_assignment_deployment_correct_directions": (
                        deployment_direction.float().sum()
                    ),
                    "stage_b_data_driven_assignment_delta_mean": (
                        valid_delta.mean()
                        if int(valid_delta.numel()) > 0
                        else delta.sum() * 0.0
                    ),
                    "stage_b_data_driven_assignment_direction_delta_mean": (
                        valid_direction_delta.mean()
                        if int(valid_direction_delta.numel()) > 0
                        else direction_delta.sum() * 0.0
                    ),
                    "stage_b_data_driven_assignment_selected_own_iou_mean": (
                        assignment["selected_own_iou"][runtime_valid].mean()
                        if bool(runtime_valid.any().item())
                        else assignment["selected_own_iou"].sum() * 0.0
                    ),
                    "stage_b_data_driven_assignment_selected_cross_iou_mean": (
                        assignment["selected_cross_iou"][runtime_valid].mean()
                        if bool(runtime_valid.any().item())
                        else assignment["selected_cross_iou"].sum() * 0.0
                    ),
                    "stage_b_data_driven_assignment_query_collision_rows": (
                        assignment["query_collision"].float().sum()
                    ),
                    "stage_b_data_driven_assignment_role0_queries": assignment[
                        "owned_role0_queries"
                    ],
                    "stage_b_data_driven_assignment_role1_queries": assignment[
                        "owned_role1_queries"
                    ],
                    "stage_b_data_driven_assignment_gap3_queries": assignment[
                        "gap3_queries"
                    ],
                }
            )
            if self.deployment_weight > 0.0:
                deployment_hard_valid = assignment[
                    "deployment_hard_valid"
                ]
                deployment_hard_delta = assignment[
                    "deployment_hard_delta"
                ]
                valid_deployment_hard_delta = deployment_hard_delta[
                    deployment_hard_valid
                ]
                result.update(
                    {
                        "loss_stage_b_data_driven_deployment_hard": assignment[
                            "deployment_hard_loss"
                        ],
                        "stage_b_data_driven_deployment_hard_valid_directions": (
                            deployment_hard_valid.float().sum()
                        ),
                        "stage_b_data_driven_deployment_hard_margin_directions": (
                            (
                                deployment_hard_delta >= self.temperature
                            )
                            & deployment_hard_valid
                        ).float().sum(),
                        "stage_b_data_driven_deployment_hard_delta_mean": (
                            valid_deployment_hard_delta.mean()
                            if int(valid_deployment_hard_delta.numel()) > 0
                            else deployment_hard_delta.sum() * 0.0
                        ),
                    }
                )
        if (
            self.rank_supervision
            == DATA_DRIVEN_RANK_SUPERVISION_SAME_CATEGORY_GAP3_COVERAGE
        ):
            result.update(
                {
                    "stage_b_data_driven_rank_coverage_negative_queries": (
                        coverage_negative.float().sum()
                    ),
                    "stage_b_data_driven_rank_gap3_eligible_queries": (
                        gap3_mask.float().sum()
                    ),
                    "stage_b_data_driven_rank_gap3_positive_queries": (
                        (positive & gap3_mask).float().sum()
                    ),
                    "stage_b_data_driven_rank_gap3_oracle_rows": (
                        (positive & gap3_mask).any(dim=1).float().sum()
                    ),
                    "stage_b_data_driven_rank_gap3_correct": (
                        gap3_correct.float().sum()
                    ),
                    "stage_b_data_driven_rank_gap3_coverage_winner_rows": (
                        gap3_coverage_winner.float().sum()
                    ),
                    "stage_b_data_driven_rank_gap3_ambiguous_coverage_queries": (
                        (
                            coverage_negative
                            & (primary_iou >= self.rank_negative_iou_threshold)
                        )
                        .float()
                        .sum()
                    ),
                    "stage_b_data_driven_rank_gap3_negative_coverage_queries": (
                        (
                            coverage_negative
                            & (primary_iou < self.rank_negative_iou_threshold)
                        )
                        .float()
                        .sum()
                    ),
                    "stage_b_data_driven_rank_semantic_negatives_in_gap3": (
                        (hard_negative & gap3_mask).float().sum()
                    ),
                }
            )
        return result

    def _confidence_pair_forward(
        self,
        outputs: Mapping[str, Any],
        targets: Sequence[Mapping[str, Any]],
    ) -> dict[str, Tensor]:
        confidence_score = outputs.get("stage_b_data_driven_confidence_score")
        confidence_token_logits = outputs.get(
            "stage_b_data_driven_confidence_token_logits"
        )
        expression_mask = outputs.get(
            "stage_b_data_driven_expression_token_mask"
        )
        expression_input_ids = outputs.get(
            "stage_b_data_driven_expression_input_ids"
        )
        candidate_mask = outputs.get("stage_b_data_driven_candidate_mask")
        boxes = outputs.get("pred_boxes")
        required = (
            confidence_score,
            confidence_token_logits,
            expression_mask,
            expression_input_ids,
            candidate_mask,
            boxes,
        )
        if not all(torch.is_tensor(value) for value in required):
            raise KeyError("data-driven confidence training outputs are incomplete")
        batch_size, query_count = boxes.shape[:2]
        if tuple(confidence_score.shape) != (batch_size, query_count, 2):
            raise ValueError("confidence_pair scores must have shape (B,Q,2)")
        token_count = int(expression_mask.shape[-1])
        if tuple(confidence_token_logits.shape) != (
            batch_size,
            query_count,
            2,
            token_count,
        ):
            raise ValueError("confidence token logits do not align with paired scores")
        if tuple(expression_mask.shape) != (batch_size, 2, token_count):
            raise ValueError("paired expression mask must have shape (B,2,T)")
        if tuple(expression_input_ids.shape) != (batch_size, 2, token_count):
            raise ValueError("paired expression IDs must have shape (B,2,T)")
        if tuple(candidate_mask.shape) != tuple(confidence_score.shape):
            raise ValueError("paired candidate mask does not align with confidence")
        expression_pairs = _validate_confidence_pair_targets(targets)

        fpr_output = fpr95_global_max_surrogate(
            confidence_score[..., 0],
            confidence_score[..., 1],
            positive_candidate_mask=candidate_mask[..., 0],
            negative_candidate_mask=candidate_mask[..., 1],
            positive_history=self._positive_history(),
            temperature=self.fpr_temperature,
            margin=self.fpr_margin,
            target_tpr=self.target_tpr,
        )
        token_loss = confidence_token_logits.sum() * 0.0
        trace_valid = confidence_score.new_zeros(())
        supervised_queries = confidence_score.new_zeros(())
        if self.token_weight > 0.0:
            roles = build_direct_trace_token_roles(
                self.tokenizer,
                expression_pairs,
                [target["stage_b_data_driven_trace"] for target in targets],
                expression_input_ids,
                expression_mask,
                max_text_len=self.max_text_len,
            )
            positive_query = _candidate_max_iou(
                boxes, targets, primary_only=False
            ) >= self.positive_iou_threshold
            valid_query = positive_query & roles["valid"][:, None]
            role_weight = confidence_token_logits.new_zeros(
                (batch_size, query_count, 2, token_count)
            )
            role_target = torch.zeros_like(role_weight)
            role_weight[:, :, 0] = (
                valid_query[:, :, None] & roles["positive"][:, None, 0]
            ).float()
            role_target[:, :, 0] = roles["positive"][:, None, 0].float()
            role_weight[:, :, 1] = (
                valid_query[:, :, None] & roles["changed"][:, None, 1]
            ).float()
            shared_weight = (
                valid_query[:, :, None] & roles["shared"][:, None, 1]
            ).float() * self.shared_token_weight
            role_weight[:, :, 1] += shared_weight
            role_target[:, :, 1] = roles["shared"][:, None, 1].float()
            element_loss = F.binary_cross_entropy_with_logits(
                confidence_token_logits.float(), role_target, reduction="none"
            )
            token_loss = _distributed_weighted_token_mean(
                (element_loss * role_weight).sum(),
                role_weight.sum(),
            )
            trace_valid = roles["valid"].float().sum().detach()
            supervised_queries = valid_query.float().sum().detach()

        if self.training:
            self._pending_queue_payload = (
                fpr_output.positive_global_score.detach().reshape(-1)
            )
        return {
            "loss_stage_b_data_driven_confidence": fpr_output.loss,
            "loss_stage_b_data_driven_token": token_loss,
            "stage_b_data_driven_positive_q05": fpr_output.positive_threshold.detach(),
            "stage_b_data_driven_current_positive_q05": (
                fpr_output.current_positive_threshold.detach()
            ),
            "stage_b_data_driven_exact_tpr": fpr_output.exact_tpr.detach(),
            "stage_b_data_driven_exact_fpr": fpr_output.exact_fpr.detach(),
            "stage_b_data_driven_positive_global_mean": (
                fpr_output.positive_global_score.detach().mean()
            ),
            "stage_b_data_driven_negative_global_mean": (
                fpr_output.negative_global_score.detach().mean()
            ),
            "stage_b_data_driven_trace_valid_rows": trace_valid,
            "stage_b_data_driven_token_supervised_queries": supervised_queries,
        }

    def forward(
        self,
        outputs: Mapping[str, Any],
        targets: Sequence[Mapping[str, Any]],
    ) -> dict[str, Tensor]:
        if self.train_mode == "rank_patch_only":
            return self._rank_patch_forward(outputs, targets)
        return self._confidence_pair_forward(outputs, targets)


__all__ = [
    "AbsoluteTokenScoreBranch",
    "DATA_DRIVEN_CONFIDENCE_INITIALIZER_SCHEMA",
    "DATA_DRIVEN_CRITERION_CONTRACT_VERSION",
    "DATA_DRIVEN_PATCH_DROP_ANCHOR_GLOBAL_MAX",
    "DATA_DRIVEN_PATCH_DROP_ANCHOR_INSTANCE_BALANCED",
    "DATA_DRIVEN_PATCH_DROP_ANCHOR_POLICIES",
    "DATA_DRIVEN_RANK_ARCHITECTURES",
    "DATA_DRIVEN_RANK_SUPERVISION_ALL_NONPOSITIVE",
    "DATA_DRIVEN_RANK_SUPERVISION_MODES",
    "DATA_DRIVEN_RANK_SUPERVISION_OFFICIAL_ASSIGNMENT",
    "DATA_DRIVEN_RANK_SUPERVISION_ROLE_ROUTED_ASSIGNMENT",
    "DATA_DRIVEN_RANK_SUPERVISION_ROLE_ROUTED_ALL_NONOWNED",
    "DATA_DRIVEN_RANK_SUPERVISION_SAME_CATEGORY",
    "DATA_DRIVEN_RANK_SUPERVISION_SAME_CATEGORY_GAP3_COVERAGE",
    "DATA_DRIVEN_ROLE_ROUTED_CRITERION_CONTRACT_VERSION",
    "DATA_DRIVEN_RELATIONAL_IMAGE_POOL_POLICY",
    "DATA_DRIVEN_RELATIONAL_INITIALIZER_SCHEMA",
    "DATA_DRIVEN_RELATIONAL_SCORE_CONTRACT_VERSION",
    "DATA_DRIVEN_ROLE_ROUTED_INITIALIZER_SCHEMA",
    "DATA_DRIVEN_SCORE_CONTRACT_VERSION",
    "DATA_DRIVEN_TRAIN_MODES",
    "RelationalRankAdapter",
    "StageBDataDrivenCriterion",
    "StageBDataDrivenScoreHeads",
    "build_direct_trace_token_roles",
    "data_driven_category_gate_mask",
    "data_driven_tensor_state_sha256",
    "deployment_gate_category_patch_loss",
    "groundingdino_raw_dot_phrase_geometry",
    "normalize_data_driven_rank_architecture",
    "normalize_data_driven_rank_supervision",
    "normalize_data_driven_train_mode",
    "official_assignment_delta_loss",
    "role_routed_official_assignment_top1_loss",
    "validate_data_driven_confidence_initializer_payload",
    "validate_data_driven_criterion_checkpoint_state",
    "validate_data_driven_relational_initializer_payload",
    "validate_data_driven_role_routed_initializer_payload",
    "validate_data_driven_trained_checkpoint_payload",
    "validate_stage_b_data_driven_score_checkpoint",
]
