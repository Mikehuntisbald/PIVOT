"""Fail-closed lineage checks for D2 continuation from the formal D1 model."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor, nn

from tools.build_stageb_native_patch_category_initializer import (
    RANDOM_TRAINABLE_PATCH_KEYS,
)
from util import misc as utils


D2_SOURCE_AUDIT_SCHEMA = "pivot.stageb.native_patch_category_d2_source_audit/v1"


class NativePatchCategoryD2ContractError(RuntimeError):
    pass


def _model_state(payload: Mapping[str, Any], *, label: str) -> dict[str, Tensor]:
    if not isinstance(payload, Mapping):
        raise NativePatchCategoryD2ContractError(f"{label} must be a mapping")
    value = payload.get("model")
    if not isinstance(value, Mapping) or not value:
        raise NativePatchCategoryD2ContractError(f"{label} has no model state")
    state = dict(utils.clean_state_dict(value))
    invalid = [
        key
        for key, tensor in state.items()
        if not isinstance(key, str) or not torch.is_tensor(tensor)
    ]
    if invalid:
        raise NativePatchCategoryD2ContractError(
            f"{label} contains non-tensor state: {invalid[:8]}"
        )
    return state


def audit_d2_source_transition(
    expected_model: nn.Module | Mapping[str, Tensor],
    initializer_payload: Mapping[str, Any],
    source_payload: Mapping[str, Any],
    *,
    expected_optimizer_updates: int = 500,
) -> dict[str, Any]:
    """Require a D1 source that changed exactly the eight patch projections."""
    expected_state = (
        expected_model.state_dict()
        if isinstance(expected_model, nn.Module)
        else dict(expected_model)
    )
    initializer_state = _model_state(initializer_payload, label="D2 base initializer")
    source_state = _model_state(source_payload, label="D2 D1 source")
    expected_keys = set(expected_state)
    if set(initializer_state) != expected_keys or set(source_state) != expected_keys:
        raise NativePatchCategoryD2ContractError(
            "D2 initializer/source model coverage differs from the built model"
        )

    changed: list[str] = []
    for key, template in expected_state.items():
        initial = initializer_state[key]
        source = source_state[key]
        if (
            not torch.is_tensor(template)
            or initial.dtype != template.dtype
            or source.dtype != template.dtype
            or tuple(initial.shape) != tuple(template.shape)
            or tuple(source.shape) != tuple(template.shape)
        ):
            raise NativePatchCategoryD2ContractError(
                f"D2 tensor shape/dtype drifted at {key}"
            )
        if source.is_floating_point() or source.is_complex():
            if not bool(torch.isfinite(source).all().item()):
                raise NativePatchCategoryD2ContractError(
                    f"D2 source tensor is non-finite at {key}"
                )
        if not torch.equal(source, initial):
            changed.append(key)

    if set(changed) != set(RANDOM_TRAINABLE_PATCH_KEYS):
        raise NativePatchCategoryD2ContractError(
            "D2 source must differ from the b58-only initializer at exactly "
            f"the eight patch projections; observed={sorted(changed)}"
        )
    forbidden = sorted(
        key
        for key in source_state
        if key.startswith(
            (
                "stage_b_gdino_score_adapter.",
                "stage_b_u0_patch_rank_adapter.",
                "stage_b_data_driven_score_heads.",
            )
        )
    )
    if forbidden:
        raise NativePatchCategoryD2ContractError(
            f"D2 source contains a forbidden old adapter: {forbidden[:8]}"
        )
    updates = source_payload.get("optimizer_updates")
    if type(updates) is not int or updates != int(expected_optimizer_updates):
        raise NativePatchCategoryD2ContractError(
            "D2 source optimizer-update count drifted"
        )
    if (
        source_payload.get("checkpoint_reason") != "max_train_iters"
        or source_payload.get("iteration") != 2 * int(expected_optimizer_updates)
        or source_payload.get("epoch") != 0
        or source_payload.get("epoch_finished") is not False
    ):
        raise NativePatchCategoryD2ContractError(
            "D2 source is not the exact formal D1 iteration checkpoint"
        )
    saved_args = source_payload.get("args")
    if hasattr(saved_args, "__dict__"):
        saved_args = vars(saved_args)
    required_args = {
        "max_train_iters": int(expected_optimizer_updates),
        "gradient_accumulation_steps": 2,
        "batch_size": 36,
        "seed": 42,
        "stage_b_native_patch_execution_scope": "native_patch_category_d1_u500_v1",
    }
    if not isinstance(saved_args, Mapping) or any(
        type(saved_args.get(key)) is not type(value)
        or saved_args.get(key) != value
        for key, value in required_args.items()
    ):
        raise NativePatchCategoryD2ContractError(
            "D2 source saved-argument contract drifted"
        )
    return {
        "schema": D2_SOURCE_AUDIT_SCHEMA,
        "optimizer_updates": updates,
        "model_tensor_count": len(source_state),
        "changed_tensor_count": len(changed),
        "changed_tensor_keys": sorted(changed),
        "all_frozen_tensors_bitwise_equal_b58_initializer": True,
        "no_teacher_u2_r100_p50_stagea_adapter_tensors": True,
    }


__all__ = [
    "D2_SOURCE_AUDIT_SCHEMA",
    "NativePatchCategoryD2ContractError",
    "audit_d2_source_transition",
]
