#!/usr/bin/env python3
"""Fail-closed contracts for ARROW Admission-input training."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import torch

from models.GroundingDINO.stage_b_u0_patch_rank import (
    stage_b_u0_tensor_state_sha256,
)
from tools.build_stageb_u2v5_clean_initializer import validate_initializer_payload
from tools.stageb_u2v4_legacy_training_contract import TRAINABLE_KEYS


SCHEMA = "arrow.stageb.admission_input_ablation/v1"
RUNTIME_SCHEMA = "arrow.stageb.admission_input_runtime/v1"
SENTINEL_SEED = 20260818
SOURCES = {
    "AR_A_PATCH": "support_patch",
    "AR_B_TEXT": "canonical_text",
    "AR_C_NULL": "learned_null",
}


class ArrowAdmissionContractError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def null_sentinel() -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SENTINEL_SEED)
    value = torch.randint(0, 2, (768,), generator=generator, dtype=torch.int64)
    return (value.to(torch.float32).mul_(2.0).sub_(1.0)) / 768**0.5


def null_sentinel_sha256() -> str:
    return hashlib.sha256(null_sentinel().contiguous().numpy().tobytes()).hexdigest()


def _state(payload: Mapping[str, Any], label: str) -> Mapping[str, torch.Tensor]:
    state = payload.get("model")
    if not isinstance(state, Mapping) or not state or any(
        not isinstance(key, str) or not torch.is_tensor(value)
        for key, value in state.items()
    ):
        raise ArrowAdmissionContractError(f"{label} lacks tensor-only model state")
    return state


def build_training_contract(
    initializer_payload: Mapping[str, Any], *, initializer_path: Path,
    initializer_sha256: str, row_id: str, source: str,
) -> dict[str, Any]:
    if row_id not in SOURCES or SOURCES[row_id] != source:
        raise ArrowAdmissionContractError("ARROW row/source registry mismatch")
    if row_id == "AR_A_PATCH":
        raise ArrowAdmissionContractError("sealed A is not a new training row")
    initializer_path = initializer_path.resolve(strict=True)
    observed = sha256_file(initializer_path)
    if observed != str(initializer_sha256):
        raise ArrowAdmissionContractError("clean initializer SHA256 mismatch")
    initializer = validate_initializer_payload(initializer_payload)
    state = _state(initializer_payload, "clean initializer")
    missing = sorted(set(TRAINABLE_KEYS) - set(state))
    if missing:
        raise ArrowAdmissionContractError(f"initializer lacks surface16: {missing}")
    frozen = sorted(set(state) - set(TRAINABLE_KEYS))
    return {
        "schema": SCHEMA,
        "row_id": row_id,
        "source": source,
        "initializer": {
            "path": str(initializer_path),
            "sha256": observed,
            "schema": initializer["schema"],
        },
        "trainable_keys": list(TRAINABLE_KEYS),
        "trainable_tensor_count": len(TRAINABLE_KEYS),
        "frozen_keys": frozen,
        "frozen_tensor_sha256": stage_b_u0_tensor_state_sha256(state, frozen),
        "initial_trainable_tensor_sha256": stage_b_u0_tensor_state_sha256(
            state, TRAINABLE_KEYS
        ),
        "shared_surface8_auxiliary8": True,
        "null_sentinel": {
            "seed": SENTINEL_SEED,
            "sha256": null_sentinel_sha256(),
            "trainable": False,
        },
        "legacy_u2v5_parent": True,
        "c100_confidence_imported": False,
    }


def validate_checkpoint_payload(
    model: torch.nn.Module, payload: Mapping[str, Any], *, row_id: str,
    source: str, state_from_payload: bool = False,
) -> dict[str, Any]:
    contract = payload.get("arrow_admission_input")
    if not isinstance(contract, Mapping) or contract.get("schema") != SCHEMA:
        raise ArrowAdmissionContractError("checkpoint lacks ARROW Admission contract")
    if contract.get("row_id") != row_id or contract.get("source") != source:
        raise ArrowAdmissionContractError("checkpoint ARROW row/source drifted")
    if tuple(contract.get("trainable_keys", ())) != tuple(TRAINABLE_KEYS):
        raise ArrowAdmissionContractError("checkpoint trainable ownership drifted")
    state = (
        _state(payload, "ARROW checkpoint")
        if state_from_payload else model.state_dict()
    )
    frozen = list(contract.get("frozen_keys", ()))
    if contract.get("frozen_tensor_sha256") != stage_b_u0_tensor_state_sha256(
        state, frozen
    ):
        raise ArrowAdmissionContractError("checkpoint changed frozen tensors")
    if contract.get("null_sentinel", {}).get("sha256") != null_sentinel_sha256():
        raise ArrowAdmissionContractError("null sentinel contract drifted")
    overlay = payload.get("arrow_confidence_overlay")
    if overlay is not None:
        if (
            not isinstance(overlay, Mapping)
            or overlay.get("schema") != "arrow.stageb.confidence_overlay/v1"
            or overlay.get("c100_confidence_imported") is not False
        ):
            raise ArrowAdmissionContractError("confidence overlay contract drifted")
        confidence_keys = list(overlay.get("confidence_keys", ()))
        if len(confidence_keys) != 12 or set(confidence_keys) & set(frozen):
            raise ArrowAdmissionContractError("confidence overlay ownership drifted")
        if overlay.get("confidence_tensor_sha256") != stage_b_u0_tensor_state_sha256(
            state, confidence_keys
        ):
            raise ArrowAdmissionContractError("confidence overlay tensor hash drifted")
    return dict(contract)


__all__ = [
    "ArrowAdmissionContractError", "RUNTIME_SCHEMA", "SCHEMA", "SENTINEL_SEED",
    "SOURCES", "build_training_contract", "null_sentinel",
    "null_sentinel_sha256", "sha256_file", "validate_checkpoint_payload",
]
