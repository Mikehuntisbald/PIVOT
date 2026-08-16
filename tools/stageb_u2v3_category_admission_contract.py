"""Fail-closed ownership contract for U2-v3 category-admission training."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch

from models.GroundingDINO.stage_b_u0_patch_rank import (
    stage_b_u0_tensor_state_sha256,
)
from tools.build_stageb_u2v2_initializer import validate_initializer_payload


SCHEMA = "pivot.stageb.u2v3_category_admission/v1"
RUNTIME_SCHEMA = "pivot.stageb.u2v3_category_admission_runtime/v1"
TRAINABLE_KEYS = (
    "patch_encoder.input_proj.0.weight",
    "patch_encoder.input_proj.0.bias",
    "patch_encoder.input_proj.1.weight",
    "patch_encoder.input_proj.1.bias",
    "patch_encoder.norm.weight",
    "patch_encoder.norm.bias",
    "query_proj_for_patch.weight",
    "query_proj_for_patch.bias",
)


class U2V3ContractError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state(payload: Mapping[str, Any], *, label: str) -> Mapping[str, torch.Tensor]:
    state = payload.get("model")
    if not isinstance(state, Mapping) or not state:
        raise U2V3ContractError(f"{label} lacks model state")
    if not all(isinstance(key, str) and torch.is_tensor(value) for key, value in state.items()):
        raise U2V3ContractError(f"{label} model state is malformed")
    return state


def build_training_contract(
    initializer_payload: Mapping[str, Any], *, initializer_path: Path,
    initializer_sha256: str,
) -> dict[str, Any]:
    initializer_path = initializer_path.resolve(strict=True)
    observed_sha = _sha256_file(initializer_path)
    if observed_sha != str(initializer_sha256):
        raise U2V3ContractError("U2-v3 initializer SHA256 mismatch")
    initializer_contract = validate_initializer_payload(initializer_payload)
    state = _state(initializer_payload, label="U2-v3 initializer")
    if not set(TRAINABLE_KEYS).issubset(state):
        raise U2V3ContractError("U2-v3 initializer lacks admission tensors")
    frozen_keys = sorted(set(state) - set(TRAINABLE_KEYS))
    return {
        "schema": SCHEMA,
        "initializer": {
            "path": str(initializer_path),
            "sha256": observed_sha,
            "schema": initializer_contract["schema"],
        },
        "trainable_keys": list(TRAINABLE_KEYS),
        "trainable_tensor_count": len(TRAINABLE_KEYS),
        "frozen_tensor_count": len(frozen_keys),
        "frozen_tensor_sha256": stage_b_u0_tensor_state_sha256(state, frozen_keys),
        "initial_trainable_tensor_sha256": stage_b_u0_tensor_state_sha256(
            state, TRAINABLE_KEYS
        ),
        "model_state_keys": len(state),
        "ownership": {
            "trainable": "stage_a_category_admission_projection",
            "frozen": ["b58_trunk", "r100_rank8", "c100_confidence12", "u0_shell"],
        },
    }


def validate_runtime_payload(
    model: torch.nn.Module, payload: Mapping[str, Any], *,
    checkpoint_label: str, initializer_path: Path, initializer_sha256: str,
) -> dict[str, Any]:
    contract = payload.get("u2v3_category_admission")
    if not isinstance(contract, Mapping) or contract.get("schema") != SCHEMA:
        raise U2V3ContractError(f"{checkpoint_label} lacks U2-v3 provenance")
    initializer_path = initializer_path.resolve(strict=True)
    initializer_payload = torch.load(
        initializer_path, map_location="cpu", weights_only=False
    )
    expected = build_training_contract(
        initializer_payload,
        initializer_path=initializer_path,
        initializer_sha256=initializer_sha256,
    )
    for key in (
        "initializer", "trainable_keys", "trainable_tensor_count",
        "frozen_tensor_count", "frozen_tensor_sha256",
        "initial_trainable_tensor_sha256", "model_state_keys", "ownership",
    ):
        if contract.get(key) != expected[key]:
            raise U2V3ContractError(f"{checkpoint_label} U2-v3 contract drifted at {key}")
    state = _state(payload, label=checkpoint_label)
    initializer_state = _state(initializer_payload, label="U2-v3 initializer")
    if set(state) != set(initializer_state) or set(state) != set(model.state_dict()):
        raise U2V3ContractError(f"{checkpoint_label} model keys drifted")
    frozen_keys = sorted(set(state) - set(TRAINABLE_KEYS))
    if stage_b_u0_tensor_state_sha256(state, frozen_keys) != expected["frozen_tensor_sha256"]:
        raise U2V3ContractError(f"{checkpoint_label} changed frozen tensors")
    for key in TRAINABLE_KEYS:
        if state[key].shape != initializer_state[key].shape or not bool(
            torch.isfinite(state[key]).all().item()
        ):
            raise U2V3ContractError(f"{checkpoint_label} invalid admission tensor {key}")
    saved_args = payload.get("args")
    runtime = saved_args.get("stage_b_u2v3_runtime_audit") if isinstance(saved_args, Mapping) else None
    if not isinstance(runtime, Mapping) or runtime.get("schema") != RUNTIME_SCHEMA:
        raise U2V3ContractError(f"{checkpoint_label} lacks U2-v3 runtime audit")
    return dict(contract)


__all__ = [
    "RUNTIME_SCHEMA", "SCHEMA", "TRAINABLE_KEYS", "U2V3ContractError",
    "build_training_contract", "validate_runtime_payload",
]
