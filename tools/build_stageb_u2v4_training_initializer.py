#!/usr/bin/env python3
"""Restore the legacy U0 auxiliary initialization inside the C100 C0 stack."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.GroundingDINO.stage_b_u0_patch_rank import (  # noqa: E402
    stage_b_u0_tensor_state_sha256,
)
from tools.build_stageb_u0_initializer import verify_initializer as verify_u0  # noqa: E402
from tools.build_stageb_u2v2_initializer import validate_initializer_payload  # noqa: E402
from tools.stageb_gdino_adapter_probe_audit import file_record, load_checkpoint  # noqa: E402


SCHEMA = "pivot.stageb.u2v4_training_initializer/v1"
U0_PREFIX = "stage_b_u0_patch_rank_adapter."


class U2V4TrainingInitializerError(RuntimeError):
    pass


def _state(payload: Mapping[str, Any], *, label: str) -> Mapping[str, torch.Tensor]:
    state = payload.get("model")
    if not isinstance(state, Mapping) or not state:
        raise U2V4TrainingInitializerError(f"{label} lacks model state")
    if not all(isinstance(k, str) and torch.is_tensor(v) for k, v in state.items()):
        raise U2V4TrainingInitializerError(f"{label} model state is malformed")
    return state


def _source(path: Path, expected_sha256: str, *, label: str):
    path = path.resolve(strict=True)
    record = file_record(path)
    if record["sha256"] != str(expected_sha256).strip().lower():
        raise U2V4TrainingInitializerError(f"{label} SHA256 mismatch")
    payload = load_checkpoint(path)
    return record, payload, _state(payload, label=label)


def build_training_initializer_payload(
    *, c0_checkpoint: Path, c0_sha256: str,
    legacy_u0_initializer: Path, legacy_u0_sha256: str,
) -> dict[str, Any]:
    c0_record, c0_payload, c0_state = _source(
        c0_checkpoint, c0_sha256, label="C100 C0"
    )
    c0_contract = validate_initializer_payload(c0_payload)
    u0_record, u0_payload, u0_state = _source(
        legacy_u0_initializer, legacy_u0_sha256, label="legacy U0 initializer"
    )
    u0_verification = verify_u0(legacy_u0_initializer)
    if u0_verification["checkpoint"] != u0_record:
        raise U2V4TrainingInitializerError("legacy U0 verification record drifted")
    if len(c0_state) != 1165 or set(u0_state) != set(c0_state):
        raise U2V4TrainingInitializerError("C0/U0 must share the exact 1,165-key model")
    u0_keys = sorted(key for key in c0_state if key.startswith(U0_PREFIX))
    if len(u0_keys) != 11:
        raise U2V4TrainingInitializerError("legacy U0 auxiliary shell must have 11 tensors")
    output_keys = (U0_PREFIX + "output.weight", U0_PREFIX + "output.bias")
    if any(int(torch.count_nonzero(u0_state[key])) for key in output_keys):
        raise U2V4TrainingInitializerError("legacy U0 output must be exactly zero")
    trunk_weight_keys = [
        key for key in u0_keys
        if key.endswith("weight") and key not in output_keys
    ]
    if not trunk_weight_keys or any(
        int(torch.count_nonzero(u0_state[key])) == 0 for key in trunk_weight_keys
    ):
        raise U2V4TrainingInitializerError("legacy U0 trunk initialization is degenerate")

    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in c0_state.items():
        source = u0_state[key] if key in u0_keys else value
        if source.dtype != value.dtype or tuple(source.shape) != tuple(value.shape):
            raise U2V4TrainingInitializerError(f"U0 tensor shape/dtype mismatch: {key}")
        result[key] = source.detach().cpu().clone()
    frozen_c0_keys = sorted(set(result) - set(u0_keys))
    contract = {
        "schema": SCHEMA,
        "training_initializer": True,
        "resumable": False,
        "model_state_keys": len(result),
        "sources": {"c100_c0": c0_record, "legacy_u0": u0_record},
        "source_schemas": {
            "c100_c0": c0_contract["schema"],
            "legacy_u0": u0_verification["schema"],
        },
        "u0_auxiliary_keys": u0_keys,
        "u0_auxiliary_tensor_sha256": stage_b_u0_tensor_state_sha256(
            result, u0_keys
        ),
        "frozen_c0_tensor_count": len(frozen_c0_keys),
        "frozen_c0_tensor_sha256": stage_b_u0_tensor_state_sha256(
            result, frozen_c0_keys
        ),
        "full_model_tensor_sha256": stage_b_u0_tensor_state_sha256(
            result, result.keys()
        ),
        "invariants": {
            "c100_c0_1154_non_u0_tensors_bitwise": True,
            "legacy_u0_auxiliary11_bitwise": True,
            "legacy_u0_trunk_nonzero": True,
            "legacy_u0_output_zero": True,
        },
    }
    return {"model": result, "u2v4_training_initializer": contract}


def validate_training_initializer_payload(
    payload: Mapping[str, Any], *, verify_sources: bool = True,
) -> dict[str, Any]:
    if set(payload) != {"model", "u2v4_training_initializer"}:
        raise U2V4TrainingInitializerError("U2-v4 initializer top-level keys drifted")
    state = _state(payload, label="U2-v4 training initializer")
    contract = payload.get("u2v4_training_initializer")
    if not isinstance(contract, Mapping) or contract.get("schema") != SCHEMA:
        raise U2V4TrainingInitializerError("U2-v4 initializer schema drifted")
    u0_keys = sorted(key for key in state if key.startswith(U0_PREFIX))
    frozen_keys = sorted(set(state) - set(u0_keys))
    if len(state) != 1165 or contract.get("u0_auxiliary_keys") != u0_keys:
        raise U2V4TrainingInitializerError("U2-v4 initializer tensor contract drifted")
    output_keys = (U0_PREFIX + "output.weight", U0_PREFIX + "output.bias")
    if any(int(torch.count_nonzero(state[key])) for key in output_keys):
        raise U2V4TrainingInitializerError("U2-v4 initializer output is not zero")
    trunk_weight_keys = [
        key for key in u0_keys
        if key.endswith("weight") and key not in output_keys
    ]
    if not trunk_weight_keys or any(
        int(torch.count_nonzero(state[key])) == 0 for key in trunk_weight_keys
    ):
        raise U2V4TrainingInitializerError("U2-v4 initializer trunk is degenerate")
    checks = {
        "u0_auxiliary_tensor_sha256": stage_b_u0_tensor_state_sha256(state, u0_keys),
        "frozen_c0_tensor_sha256": stage_b_u0_tensor_state_sha256(state, frozen_keys),
        "full_model_tensor_sha256": stage_b_u0_tensor_state_sha256(state, state.keys()),
    }
    for key, observed in checks.items():
        if contract.get(key) != observed:
            raise U2V4TrainingInitializerError(f"U2-v4 initializer hash drifted at {key}")
    if verify_sources:
        sources = contract.get("sources")
        if not isinstance(sources, Mapping):
            raise U2V4TrainingInitializerError("U2-v4 initializer sources are missing")
        c0_record, u0_record = sources.get("c100_c0"), sources.get("legacy_u0")
        if not isinstance(c0_record, Mapping) or not isinstance(u0_record, Mapping):
            raise U2V4TrainingInitializerError("U2-v4 initializer source is malformed")
        c0_path = Path(str(c0_record.get("path", ""))).resolve(strict=True)
        u0_path = Path(str(u0_record.get("path", ""))).resolve(strict=True)
        if file_record(c0_path) != dict(c0_record) or file_record(u0_path) != dict(u0_record):
            raise U2V4TrainingInitializerError("U2-v4 initializer source changed")
        c0_payload = load_checkpoint(c0_path)
        validate_initializer_payload(c0_payload)
        verify_u0(u0_path)
        c0_state = _state(c0_payload, label="C100 C0 source")
        u0_state = _state(load_checkpoint(u0_path), label="legacy U0 source")
        for key in state:
            expected = u0_state[key] if key in u0_keys else c0_state[key]
            if not torch.equal(state[key], expected):
                raise U2V4TrainingInitializerError(f"source tensor drift at {key}")
    return dict(contract)


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    if path.exists():
        raise U2V4TrainingInitializerError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--output", required=True)
    create.add_argument("--c0-checkpoint", required=True)
    create.add_argument("--c0-sha256", required=True)
    create.add_argument("--legacy-u0-initializer", required=True)
    create.add_argument("--legacy-u0-sha256", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    if args.command == "create":
        payload = build_training_initializer_payload(
            c0_checkpoint=Path(args.c0_checkpoint), c0_sha256=args.c0_sha256,
            legacy_u0_initializer=Path(args.legacy_u0_initializer),
            legacy_u0_sha256=args.legacy_u0_sha256,
        )
        _write(Path(args.output), payload)
        checkpoint = Path(args.output).resolve(strict=True)
    else:
        checkpoint = Path(args.checkpoint).resolve(strict=True)
    contract = validate_training_initializer_payload(load_checkpoint(checkpoint))
    print(json.dumps({"status": "verified", "checkpoint": file_record(checkpoint), "contract": contract}, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, U2V4TrainingInitializerError) as error:
        raise SystemExit(f"[FAIL] {error}") from error
