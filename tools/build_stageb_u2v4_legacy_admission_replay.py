#!/usr/bin/env python3
"""Transplant the sealed legacy-U2 patch surface into the C100 C0 stack."""

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
from tools.build_stageb_u2v2_initializer import (  # noqa: E402
    validate_initializer_payload,
)
from tools.stageb_gdino_adapter_probe_audit import (  # noqa: E402
    file_record,
    load_checkpoint,
)


SCHEMA = "pivot.stageb.u2v4_legacy_admission_replay/v1"
SURFACE_KEYS = (
    "patch_encoder.input_proj.0.weight",
    "patch_encoder.input_proj.0.bias",
    "patch_encoder.input_proj.1.weight",
    "patch_encoder.input_proj.1.bias",
    "patch_encoder.norm.weight",
    "patch_encoder.norm.bias",
    "query_proj_for_patch.weight",
    "query_proj_for_patch.bias",
    "patch_logit_scale",
)


class U2V4ReplayError(RuntimeError):
    pass


def _state(payload: Mapping[str, Any], *, label: str) -> Mapping[str, torch.Tensor]:
    value = payload.get("model")
    if not isinstance(value, Mapping) or not value:
        raise U2V4ReplayError(f"{label} lacks model state")
    if not all(
        isinstance(key, str) and torch.is_tensor(tensor)
        for key, tensor in value.items()
    ):
        raise U2V4ReplayError(f"{label} model state is malformed")
    return value


def _source(path: Path, expected_sha256: str, *, label: str):
    path = path.resolve(strict=True)
    record = file_record(path)
    if record["sha256"] != str(expected_sha256).strip().lower():
        raise U2V4ReplayError(
            f"{label} SHA256 mismatch: expected {expected_sha256}, "
            f"got {record['sha256']}"
        )
    payload = load_checkpoint(path)
    return record, payload, _state(payload, label=label)


def _equal(left: torch.Tensor, right: torch.Tensor, *, key: str) -> None:
    if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
        raise U2V4ReplayError(f"shape/dtype drift at {key}")
    if not torch.equal(left, right):
        raise U2V4ReplayError(f"bitwise tensor drift at {key}")


def _validate_legacy_training_payload(payload: Mapping[str, Any]) -> None:
    args = payload.get("args")
    if (
        not isinstance(args, Mapping)
        or args.get("stage_b_u2_category_complete_supervision") is not True
        or args.get("stage_b_u0_patch_rank") is not True
        or int(args.get("batch_size", -1)) != 56
        or int(args.get("seed", -1)) != 42
        or int(args.get("max_train_iters", -1)) != 100
        or int(payload.get("optimizer_updates", -1)) != 100
        or payload.get("checkpoint_reason") != "max_train_iters"
    ):
        raise U2V4ReplayError("legacy U2 training provenance drifted")


def build_replay_payload(
    *, c0_checkpoint: Path, c0_sha256: str,
    legacy_u2_checkpoint: Path, legacy_u2_sha256: str,
) -> dict[str, Any]:
    c0_record, c0_payload, c0_state = _source(
        c0_checkpoint, c0_sha256, label="C100 C0"
    )
    c0_contract = validate_initializer_payload(c0_payload)
    legacy_record, legacy_payload, legacy_state = _source(
        legacy_u2_checkpoint, legacy_u2_sha256, label="legacy U2"
    )
    _validate_legacy_training_payload(legacy_payload)
    if len(c0_state) != 1165 or set(legacy_state) != set(c0_state):
        raise U2V4ReplayError("C0/legacy U2 must share the exact 1,165-key model")
    if set(SURFACE_KEYS) - set(c0_state):
        raise U2V4ReplayError("C0/legacy U2 lacks the nine patch-surface tensors")

    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, c0_value in c0_state.items():
        value = legacy_state[key] if key in SURFACE_KEYS else c0_value
        if value.dtype != c0_value.dtype or tuple(value.shape) != tuple(c0_value.shape):
            raise U2V4ReplayError(f"legacy surface shape/dtype mismatch: {key}")
        result[key] = value.detach().cpu().clone()

    frozen_keys = sorted(set(result) - set(SURFACE_KEYS))
    contract = {
        "schema": SCHEMA,
        "eval_only": True,
        "resumable": False,
        "model_state_keys": len(result),
        "sources": {
            "c100_c0": c0_record,
            "legacy_u2": legacy_record,
        },
        "source_schemas": {
            "c100_c0": c0_contract["schema"],
            "legacy_u2": "sealed_category_complete_u100_seed42_b56",
        },
        "surface_keys": list(SURFACE_KEYS),
        "surface_tensor_count": len(SURFACE_KEYS),
        "surface_tensor_sha256": stage_b_u0_tensor_state_sha256(
            result, SURFACE_KEYS
        ),
        "frozen_tensor_count": len(frozen_keys),
        "frozen_tensor_sha256": stage_b_u0_tensor_state_sha256(
            result, frozen_keys
        ),
        "full_model_tensor_sha256": stage_b_u0_tensor_state_sha256(
            result, result.keys()
        ),
        "ownership": {
            "surface9": "legacy_u2",
            "all_other_tensors": "c100_c0",
            "trunk": "frozen_b58",
            "rank": "frozen_r100",
            "confidence": "frozen_c100",
            "u0_auxiliary": "frozen_zero_c0_shell",
        },
        "routes": {
            "ref": "full_expression_legacy_surface_gap3_frozen_r100",
            "confidence": "frozen_c100_total_trust_v1",
            "b58_top1_guard": False,
        },
    }
    return {"model": result, "u2v4_legacy_admission_replay": contract}


def validate_replay_payload(
    payload: Mapping[str, Any], *, verify_sources: bool = True,
) -> dict[str, Any]:
    if set(payload) != {"model", "u2v4_legacy_admission_replay"}:
        raise U2V4ReplayError("U2-v4 replay top-level keys drifted")
    state = _state(payload, label="U2-v4 replay")
    contract = payload.get("u2v4_legacy_admission_replay")
    if not isinstance(contract, Mapping) or contract.get("schema") != SCHEMA:
        raise U2V4ReplayError("U2-v4 replay schema drifted")
    if (
        len(state) != 1165
        or contract.get("model_state_keys") != 1165
        or contract.get("surface_keys") != list(SURFACE_KEYS)
        or contract.get("surface_tensor_count") != len(SURFACE_KEYS)
    ):
        raise U2V4ReplayError("U2-v4 replay tensor contract drifted")
    frozen_keys = sorted(set(state) - set(SURFACE_KEYS))
    checks = {
        "surface_tensor_sha256": stage_b_u0_tensor_state_sha256(
            state, SURFACE_KEYS
        ),
        "frozen_tensor_sha256": stage_b_u0_tensor_state_sha256(
            state, frozen_keys
        ),
        "full_model_tensor_sha256": stage_b_u0_tensor_state_sha256(
            state, state.keys()
        ),
    }
    for key, observed in checks.items():
        if contract.get(key) != observed:
            raise U2V4ReplayError(f"U2-v4 replay hash drifted at {key}")
    if verify_sources:
        sources = contract.get("sources")
        if not isinstance(sources, Mapping):
            raise U2V4ReplayError("U2-v4 replay source records are missing")
        c0_record = sources.get("c100_c0")
        legacy_record = sources.get("legacy_u2")
        if not isinstance(c0_record, Mapping) or not isinstance(legacy_record, Mapping):
            raise U2V4ReplayError("U2-v4 replay source record is malformed")
        c0_path = Path(str(c0_record.get("path", ""))).resolve(strict=True)
        legacy_path = Path(str(legacy_record.get("path", ""))).resolve(strict=True)
        if file_record(c0_path) != dict(c0_record):
            raise U2V4ReplayError("C100 C0 source changed after replay build")
        if file_record(legacy_path) != dict(legacy_record):
            raise U2V4ReplayError("legacy U2 source changed after replay build")
        c0_payload = load_checkpoint(c0_path)
        validate_initializer_payload(c0_payload)
        c0_state = _state(c0_payload, label="C100 C0 source")
        legacy_payload = load_checkpoint(legacy_path)
        _validate_legacy_training_payload(legacy_payload)
        legacy_state = _state(legacy_payload, label="legacy U2 source")
        for key in state:
            expected = legacy_state[key] if key in SURFACE_KEYS else c0_state[key]
            _equal(state[key], expected, key=key)
    return dict(contract)


def validate_runtime_payload(
    model: torch.nn.Module, payload: Mapping[str, Any], *, checkpoint_label: str,
) -> dict[str, Any]:
    contract = validate_replay_payload(payload, verify_sources=True)
    state = _state(payload, label=checkpoint_label)
    if set(state) != set(model.state_dict()):
        raise U2V4ReplayError(f"{checkpoint_label} does not match runtime model")
    return contract


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    if path.exists():
        raise U2V4ReplayError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--output", required=True)
    create.add_argument("--c0-checkpoint", required=True)
    create.add_argument("--c0-sha256", required=True)
    create.add_argument("--legacy-u2-checkpoint", required=True)
    create.add_argument("--legacy-u2-sha256", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--checkpoint", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "create":
        payload = build_replay_payload(
            c0_checkpoint=Path(args.c0_checkpoint),
            c0_sha256=args.c0_sha256,
            legacy_u2_checkpoint=Path(args.legacy_u2_checkpoint),
            legacy_u2_sha256=args.legacy_u2_sha256,
        )
        _write(Path(args.output), payload)
        checkpoint = Path(args.output).resolve(strict=True)
    else:
        checkpoint = Path(args.checkpoint).resolve(strict=True)
    contract = validate_replay_payload(load_checkpoint(checkpoint))
    print(json.dumps({
        "status": "verified",
        "checkpoint": file_record(checkpoint),
        "contract": contract,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, U2V4ReplayError) as error:
        raise SystemExit(f"[FAIL] {error}") from error
