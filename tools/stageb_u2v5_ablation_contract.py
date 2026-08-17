#!/usr/bin/env python3
"""Fail-closed ownership/provenance contracts for U2-v5 ablation rows."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from models.GroundingDINO.stage_b_u0_patch_rank import stage_b_u0_tensor_state_sha256
from tools.build_stageb_u2v5_clean_initializer import validate_initializer_payload
from tools.stageb_u2v4_legacy_training_contract import (
    AUXILIARY_RESIDUAL_KEYS,
    SURFACE_PARAMETER_KEYS,
)
from tools.stageb_u2v5_ablation_registry import get_row


SCHEMA = "pivot.stageb.u2v5_ablation_checkpoint/v1"
RUNTIME_SCHEMA = "pivot.stageb.u2v5_ablation_runtime/v1"


class U2V5AblationContractError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state(payload: Mapping[str, Any], label: str) -> Mapping[str, torch.Tensor]:
    state = payload.get("model")
    if not isinstance(state, Mapping) or not state or any(
        not isinstance(key, str) or not torch.is_tensor(value)
        for key, value in state.items()
    ):
        raise U2V5AblationContractError(f"{label} lacks tensor-only model state")
    return state


def admission_trainable_keys(roles: Sequence[str]) -> tuple[str, ...]:
    roles = tuple(str(role) for role in roles)
    if not roles or len(roles) != len(set(roles)) or set(roles) - {
        "surface8", "auxiliary8"
    }:
        raise U2V5AblationContractError(f"invalid admission roles: {roles}")
    keys: list[str] = []
    if "surface8" in roles:
        keys.extend(SURFACE_PARAMETER_KEYS)
    if "auxiliary8" in roles:
        keys.extend(AUXILIARY_RESIDUAL_KEYS)
    return tuple(keys)


def build_admission_contract(
    initializer_payload: Mapping[str, Any], *, initializer_path: Path,
    initializer_sha256: str, row_id: str, roles: Sequence[str],
    category_loss_weight: float, preserve_weight: float,
) -> dict[str, Any]:
    row = get_row(row_id)
    if row.phase != "admission":
        raise U2V5AblationContractError(f"row {row_id} is not an admission row")
    keys = admission_trainable_keys(roles)
    if tuple(roles) != row.trainable_roles or set(keys) == set():
        raise U2V5AblationContractError(f"row {row_id} role registry drifted")
    initializer_path = initializer_path.resolve(strict=True)
    observed_sha = _sha256(initializer_path)
    if observed_sha != str(initializer_sha256):
        raise U2V5AblationContractError("clean initializer SHA256 mismatch")
    initializer_contract = validate_initializer_payload(initializer_payload)
    state = _state(initializer_payload, "clean initializer")
    frozen = sorted(set(state) - set(keys))
    return {
        "schema": SCHEMA,
        "row": row.payload(),
        "phase": "admission",
        "initializer": {
            "path": str(initializer_path),
            "sha256": observed_sha,
            "schema": initializer_contract["schema"],
        },
        "trainable_keys": list(keys),
        "trainable_tensor_count": len(keys),
        "frozen_keys": frozen,
        "frozen_tensor_sha256": stage_b_u0_tensor_state_sha256(state, frozen),
        "initial_trainable_tensor_sha256": stage_b_u0_tensor_state_sha256(
            state, keys
        ),
        "loss_contract": {
            "category_loss_weight": float(category_loss_weight),
            "target_preserve_weight": float(preserve_weight),
        },
        "c100_confidence_imported": False,
    }


def validate_admission_runtime_payload(
    model: torch.nn.Module, payload: Mapping[str, Any], *, checkpoint_label: str,
    initializer_path: Path, initializer_sha256: str,
) -> dict[str, Any]:
    contract = payload.get("u2v5_ablation")
    if not isinstance(contract, Mapping) or contract.get("schema") != SCHEMA:
        raise U2V5AblationContractError(
            f"{checkpoint_label} lacks U2-v5 ablation provenance"
        )
    row_payload = contract.get("row")
    if not isinstance(row_payload, Mapping):
        raise U2V5AblationContractError(f"{checkpoint_label} lacks row binding")
    row_id = str(row_payload.get("row_id", ""))
    roles = tuple(row_payload.get("trainable_roles", ()))
    loss = contract.get("loss_contract")
    if not isinstance(loss, Mapping):
        raise U2V5AblationContractError(f"{checkpoint_label} lacks loss binding")
    initializer_path = initializer_path.resolve(strict=True)
    initializer_payload = torch.load(
        initializer_path, map_location="cpu", weights_only=False
    )
    expected = build_admission_contract(
        initializer_payload,
        initializer_path=initializer_path,
        initializer_sha256=initializer_sha256,
        row_id=row_id,
        roles=roles,
        category_loss_weight=float(loss["category_loss_weight"]),
        preserve_weight=float(loss["target_preserve_weight"]),
    )
    for key in (
        "schema", "row", "phase", "initializer", "trainable_keys",
        "trainable_tensor_count", "frozen_keys", "frozen_tensor_sha256",
        "initial_trainable_tensor_sha256", "loss_contract",
        "c100_confidence_imported",
    ):
        if contract.get(key) != expected[key]:
            raise U2V5AblationContractError(
                f"{checkpoint_label} contract drifted at {key}"
            )
    state = _state(payload, checkpoint_label)
    if set(state) != set(model.state_dict()):
        raise U2V5AblationContractError(f"{checkpoint_label} model keys drifted")
    if stage_b_u0_tensor_state_sha256(
        state, expected["frozen_keys"]
    ) != expected["frozen_tensor_sha256"]:
        raise U2V5AblationContractError(
            f"{checkpoint_label} changed a frozen tensor"
        )
    saved_args = payload.get("args")
    runtime = (
        saved_args.get("stage_b_u2v5_ablation_runtime_audit")
        if isinstance(saved_args, Mapping) else None
    )
    if not isinstance(runtime, Mapping) or runtime.get("schema") != RUNTIME_SCHEMA:
        raise U2V5AblationContractError(
            f"{checkpoint_label} lacks ablation runtime audit"
        )
    return dict(contract)


def validate_ownership_runtime_payload(
    model: torch.nn.Module, payload: Mapping[str, Any], *, checkpoint_label: str,
) -> dict[str, Any]:
    contract = payload.get("u2v5_ownership")
    if not isinstance(contract, Mapping) or contract.get("schema") != (
        "pivot.stageb.u2v5_ownership_checkpoint/v1"
    ):
        raise U2V5AblationContractError(
            f"{checkpoint_label} lacks ownership provenance"
        )
    row_payload = contract.get("row")
    if not isinstance(row_payload, Mapping):
        raise U2V5AblationContractError(f"{checkpoint_label} lacks row binding")
    row = get_row(str(row_payload.get("row_id", "")))
    if row.phase != "ownership" or row.payload() != dict(row_payload):
        raise U2V5AblationContractError(f"{checkpoint_label} row binding drifted")
    state = _state(payload, checkpoint_label)
    frozen = contract.get("frozen_keys")
    trainable = contract.get("trainable_keys")
    if not (
        isinstance(frozen, list) and isinstance(trainable, list)
        and set(frozen).isdisjoint(trainable)
        and set(frozen) | set(trainable) == set(state)
        and set(state) == set(model.state_dict())
    ):
        raise U2V5AblationContractError(
            f"{checkpoint_label} ownership partition drifted"
        )
    if contract.get("frozen_tensor_sha256") != stage_b_u0_tensor_state_sha256(
        state, frozen
    ):
        raise U2V5AblationContractError(f"{checkpoint_label} frozen hash drifted")
    runtime = contract.get("runtime_audit")
    expected_exposure = {"admission": 100, "confidence": 50}
    if not (
        contract.get("c100_confidence_imported") is False
        and contract.get("exposure") == expected_exposure
        and isinstance(runtime, Mapping)
        and runtime.get("task_successful_steps") == expected_exposure
        and runtime.get("amp_skipped_optimizer_steps") == 0
        and runtime.get("nonfinite_gradient_boundaries") == 0
    ):
        raise U2V5AblationContractError(
            f"{checkpoint_label} ownership runtime drifted"
        )
    return dict(contract)


__all__ = [
    "RUNTIME_SCHEMA", "SCHEMA", "U2V5AblationContractError",
    "admission_trainable_keys", "build_admission_contract",
    "validate_admission_runtime_payload", "validate_ownership_runtime_payload",
]
