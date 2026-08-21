"""Fail-closed runtime validation for the B58 capacity-control block."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from models.GroundingDINO.stage_b_u0_patch_rank import (
    stage_b_u0_tensor_state_sha256,
)


SCHEMA = "arrow.stageb.b58_capacity_control_checkpoint/v1"
ROW_SCHEMA = "arrow.stageb.b58_capacity_control_row/v1"

ROWS = {
    "B58_SHARED_WIDE": {
        "config": "config/ablations/cfg_stageb_b58_capacity_shared_wide.py",
        "ownership": "shared_wide_two_heads",
        "capacity": {
            "trainable_parameters": 352138,
            "score_owner_parameters": 83971,
            "score_macs_per_query_and_output": 83007,
            "representation_dim": 163,
            "gate_hidden_dim": 62,
        },
    },
    "B58_ISOLATED_REPLAY": {
        "config": "config/ablations/cfg_stageb_b58_capacity_isolated_replay.py",
        "ownership": "isolated_heads",
        "capacity": {
            "trainable_parameters": 352136,
            "score_owner_parameters": 83969,
            "score_macs_per_query_and_output": 82944,
            "representation_dim": 128,
            "gate_hidden_dim": 128,
        },
    },
}


class CapacityControlContractError(RuntimeError):
    pass


def _state(payload: Mapping[str, Any], label: str) -> Mapping[str, torch.Tensor]:
    state = payload.get("model")
    if not isinstance(state, Mapping) or not state:
        raise CapacityControlContractError(f"{label} lacks model state")
    if not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise CapacityControlContractError(f"{label} model state is malformed")
    return state


def validate_b58_capacity_runtime_payload(
    model: torch.nn.Module,
    payload: Mapping[str, Any],
    *,
    row_id: str,
    checkpoint_label: str,
) -> dict[str, Any]:
    expected = ROWS.get(row_id)
    if expected is None:
        raise CapacityControlContractError(f"unknown capacity row {row_id!r}")
    contract = payload.get("u2v5_ownership")
    if not isinstance(contract, Mapping) or contract.get("schema") != SCHEMA:
        raise CapacityControlContractError(
            f"{checkpoint_label} lacks B58 capacity-control provenance"
        )
    expected_row = {
        "schema": ROW_SCHEMA,
        "row_id": row_id,
        "config": expected["config"],
        "ownership": expected["ownership"],
        "updates": 150,
        "batch_size": 56,
        "parent": "clean_initializer",
    }
    if contract.get("row") != expected_row:
        raise CapacityControlContractError(
            f"{checkpoint_label} capacity row binding drifted"
        )
    state = _state(payload, checkpoint_label)
    frozen = contract.get("frozen_keys")
    trainable = contract.get("trainable_keys")
    if not (
        isinstance(frozen, list)
        and isinstance(trainable, list)
        and set(frozen).isdisjoint(trainable)
        and set(frozen) | set(trainable) == set(state)
        and set(state) == set(model.state_dict())
    ):
        raise CapacityControlContractError(
            f"{checkpoint_label} capacity ownership partition drifted"
        )
    if contract.get("frozen_tensor_sha256") != stage_b_u0_tensor_state_sha256(
        state, frozen
    ):
        raise CapacityControlContractError(
            f"{checkpoint_label} capacity frozen hash drifted"
        )
    runtime = contract.get("runtime_audit")
    optimizer = contract.get("optimizer_ownership")
    accounting = contract.get("parameter_accounting")
    if not (
        contract.get("c100_confidence_imported") is False
        and contract.get("exposure") == {"admission": 100, "confidence": 50}
        and isinstance(runtime, Mapping)
        and runtime.get("successful_optimizer_steps") == 150
        and runtime.get("task_successful_steps")
        == {"admission": 100, "confidence": 50}
        and runtime.get("amp_skipped_optimizer_steps") == 0
        and runtime.get("nonfinite_gradient_boundaries") == 0
        and isinstance(optimizer, Mapping)
        and optimizer.get("task_specific_states") is True
        and optimizer.get("weight_decay") == 0.0
        and isinstance(accounting, Mapping)
        and accounting.get("capacity_control") == expected["capacity"]
        and accounting.get("trainable")
        == expected["capacity"]["trainable_parameters"]
    ):
        raise CapacityControlContractError(
            f"{checkpoint_label} capacity runtime contract drifted"
        )
    gradient = contract.get("gradient_audit")
    if row_id == "B58_ISOLATED_REPLAY" and not (
        isinstance(gradient, Mapping)
        and gradient.get("structural_isolation_checks") == 150
        and gradient.get("structural_cross_gradients") == 0
    ):
        raise CapacityControlContractError(
            f"{checkpoint_label} failed structural isolation audit"
        )
    if row_id == "B58_SHARED_WIDE" and not (
        isinstance(gradient, Mapping)
        and gradient.get("diagnostic_pairs", 0) > 0
    ):
        raise CapacityControlContractError(
            f"{checkpoint_label} lacks shared-gradient diagnostics"
        )
    return dict(contract)


__all__ = [
    "CapacityControlContractError",
    "SCHEMA",
    "validate_b58_capacity_runtime_payload",
]
