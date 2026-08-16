#!/usr/bin/env python3
"""Build and verify the frozen Stage-A/R100/C100 U2-v2 C0 checkpoint."""

from __future__ import annotations

import argparse
import gc
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

from main import build_model_main  # noqa: E402
from models.GroundingDINO.stage_b_u0_patch_rank import (  # noqa: E402
    stage_b_u0_tensor_state_sha256,
)
from tools.stageb_gdino_adapter_probe_audit import file_record, load_checkpoint  # noqa: E402
from util.slconfig import SLConfig  # noqa: E402


SCHEMA = "pivot.stageb.u2v2_initializer/v1"
DEFAULT_CONFIG = REPO_ROOT / "config/ablations/cfg_stageb_u2v2_c0.py"
U0_PREFIX = "stage_b_u0_patch_rank_adapter."
ADAPTER_PREFIX = "stage_b_gdino_score_adapter."
RANK_PREFIXES = (
    ADAPTER_PREFIX + "rank_norm.", ADAPTER_PREFIX + "rank_trunk.",
    ADAPTER_PREFIX + "rank_output.",
)
CONFIDENCE_PREFIXES = (
    ADAPTER_PREFIX + "confidence_norm.", ADAPTER_PREFIX + "confidence_trunk.",
    ADAPTER_PREFIX + "confidence_gate.",
)


class U2V2InitializerError(RuntimeError):
    pass


def _state(payload: Mapping[str, Any], *, label: str) -> Mapping[str, torch.Tensor]:
    value = payload.get("model")
    if not isinstance(value, Mapping) or not value:
        raise U2V2InitializerError(f"{label} has no model state")
    if any(not isinstance(k, str) or not torch.is_tensor(v) for k, v in value.items()):
        raise U2V2InitializerError(f"{label} model state is not tensor-only")
    return value


def _source(path: Path, expected_sha256: str, *, label: str):
    path = path.resolve(strict=True)
    record = file_record(path)
    if record["sha256"] != expected_sha256.strip().lower():
        raise U2V2InitializerError(
            f"{label} SHA256 mismatch: expected {expected_sha256}, got {record['sha256']}"
        )
    payload = load_checkpoint(path)
    return record, payload, _state(payload, label=label)


def _equal(left: torch.Tensor, right: torch.Tensor, *, key: str) -> None:
    if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
        raise U2V2InitializerError(f"shape/dtype drift at {key}")
    if not torch.equal(left, right):
        raise U2V2InitializerError(f"bitwise tensor drift at {key}")


def _audit_sources(
    stagea: Mapping[str, torch.Tensor], r100: Mapping[str, torch.Tensor],
    c100: Mapping[str, torch.Tensor],
) -> dict[str, list[str]]:
    if set(r100) != set(c100) or len(c100) != 958:
        raise U2V2InitializerError("R100/C100 must have the same 958-key state")
    rank = sorted(k for k in c100 if k.startswith(RANK_PREFIXES))
    confidence = sorted(k for k in c100 if k.startswith(CONFIDENCE_PREFIXES))
    trunk = sorted(set(c100) - set(rank) - set(confidence))
    if (len(trunk), len(rank), len(confidence)) != (938, 8, 12):
        raise U2V2InitializerError("C100 ownership must be trunk938/rank8/confidence12")
    changed = []
    for key in c100:
        if not torch.equal(c100[key], r100[key]):
            changed.append(key)
    if set(changed) != set(confidence):
        raise U2V2InitializerError("C100 relative to R100 must change confidence12 only")
    for key in trunk + rank:
        _equal(c100[key], r100[key], key=key)

    shared = sorted(set(stagea) & set(c100))
    patch = sorted(set(stagea) - set(c100))
    if len(shared) != 938 or set(shared) != set(trunk) or len(patch) != 196:
        raise U2V2InitializerError("Stage-A/C100 ownership must be shared938/patch196")
    for key in shared:
        _equal(stagea[key], c100[key], key=key)
    patch_backbone = sorted(k for k in patch if k.startswith("patch_encoder.backbone."))
    if len(patch_backbone) != 187:
        raise U2V2InitializerError("Stage-A patch backbone must contain 187 tensors")
    for key in patch_backbone:
        base = key.removeprefix("patch_encoder.")
        if base not in stagea:
            raise U2V2InitializerError(f"patch backbone alias has no main source: {key}")
        _equal(stagea[key], stagea[base], key=key)
    return {
        "trunk": trunk, "rank": rank, "confidence": confidence,
        "patch": patch, "patch_backbone": patch_backbone,
    }


def _template(config: Path):
    cfg = SLConfig.fromfile(str(config.resolve(strict=True)))
    cfg.device = "cpu"
    if not bool(getattr(cfg, "stage_b_u2v2", False)):
        raise U2V2InitializerError("config must enable stage_b_u2v2")
    if bool(getattr(cfg, "stage_b_u2v2_rank_residual", False)):
        raise U2V2InitializerError("initializer config must describe residual-free C0")
    torch.manual_seed(0)
    model, _criterion, _post = build_model_main(cfg)
    return model.eval(), cfg


def build_initializer_payload(
    *, stagea_checkpoint: Path, stagea_sha256: str,
    r100_checkpoint: Path, r100_sha256: str,
    c100_checkpoint: Path, c100_sha256: str,
    config: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    stagea_record, stagea_payload, stagea = _source(
        stagea_checkpoint, stagea_sha256, label="Stage-A"
    )
    r100_record, r100_payload, r100 = _source(
        r100_checkpoint, r100_sha256, label="R100"
    )
    c100_record, c100_payload, c100 = _source(
        c100_checkpoint, c100_sha256, label="C100"
    )
    roles = _audit_sources(stagea, r100, c100)
    model, _cfg = _template(config)
    template = model.state_dict()
    expected = 958 + 196 + 11
    if len(template) != expected:
        raise U2V2InitializerError(
            f"C0 template must contain {expected} tensors, got {len(template)}"
        )
    u0_parameter_keys = {
        name for name, _ in model.named_parameters() if name.startswith(U0_PREFIX)
    }
    u0_keys = sorted(k for k in template if k.startswith(U0_PREFIX))
    if len(u0_keys) != 11 or len(u0_parameter_keys) != 8:
        raise U2V2InitializerError("U0 compatibility shell must be 8 params + 3 buffers")

    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, wanted in template.items():
        if key in c100:
            value = c100[key]
        elif key in roles["patch"]:
            value = stagea[key]
        elif key in u0_keys:
            value = torch.zeros_like(wanted) if key in u0_parameter_keys else wanted
        else:
            raise U2V2InitializerError(f"unowned C0 tensor: {key}")
        if value.dtype != wanted.dtype or tuple(value.shape) != tuple(wanted.shape):
            raise U2V2InitializerError(f"C0 tensor shape/dtype mismatch: {key}")
        result[key] = value.detach().cpu().clone()
    load = model.load_state_dict(result, strict=True)
    if load.missing_keys or load.unexpected_keys:
        raise U2V2InitializerError("strict C0 model load drifted")
    if any(int(torch.count_nonzero(result[k])) for k in u0_parameter_keys):
        raise U2V2InitializerError("U0 compatibility parameters are not exactly zero")
    role_keys = {
        "trunk": roles["trunk"], "rank": roles["rank"],
        "confidence": roles["confidence"], "patch": roles["patch"],
        "u0_shell": u0_keys,
    }
    contract = {
        "schema": SCHEMA,
        "training_initializer": True,
        "resumable": False,
        "model_state_keys": len(result),
        "runtime_state_keys_with_residual": len(result) + 9,
        "sources": {
            "stagea": stagea_record, "r100": r100_record, "c100": c100_record,
        },
        "role_key_counts": {k: len(v) for k, v in role_keys.items()},
        "role_keys": role_keys,
        "role_tensor_sha256": {
            k: stage_b_u0_tensor_state_sha256(result, v)
            for k, v in role_keys.items()
        },
        "full_model_tensor_sha256": stage_b_u0_tensor_state_sha256(
            result, result.keys()
        ),
        "routes": {
            "ref": "full_expression_patch_gap_postgate_bounded_r100_v1",
            "confidence": "frozen_c100_total_trust_v1",
            "b58_top1_guard": False,
        },
        "invariants": {
            "stagea_c100_trunk938_bitwise": True,
            "c100_r100_rank8_bitwise": True,
            "c100_changes_confidence12_only": True,
            "patch196_from_stagea": True,
            "patch_backbone187_equals_main": True,
            "u0_shell_parameters_zero_and_frozen": True,
        },
    }
    del model, stagea_payload, r100_payload, c100_payload
    gc.collect()
    return {"model": result, "u2v2_initializer": contract}


def validate_initializer_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    if set(payload) != {"model", "u2v2_initializer"}:
        raise U2V2InitializerError("U2-v2 initializer top-level keys drifted")
    state = _state(payload, label="U2-v2 initializer")
    contract = payload.get("u2v2_initializer")
    if not isinstance(contract, Mapping) or contract.get("schema") != SCHEMA:
        raise U2V2InitializerError("U2-v2 initializer schema drifted")
    if len(state) != 1165 or contract.get("model_state_keys") != 1165:
        raise U2V2InitializerError("U2-v2 initializer must contain 1,165 tensors")
    roles = contract.get("role_keys")
    hashes = contract.get("role_tensor_sha256")
    if not isinstance(roles, Mapping) or not isinstance(hashes, Mapping):
        raise U2V2InitializerError("U2-v2 role contract is missing")
    all_role_keys = []
    for role, keys in roles.items():
        if not isinstance(keys, list) or hashes.get(role) != stage_b_u0_tensor_state_sha256(state, keys):
            raise U2V2InitializerError(f"U2-v2 {role} role hash drifted")
        all_role_keys.extend(keys)
    if len(all_role_keys) != len(set(all_role_keys)) or set(all_role_keys) != set(state):
        raise U2V2InitializerError("U2-v2 roles do not partition model state")
    if contract.get("full_model_tensor_sha256") != stage_b_u0_tensor_state_sha256(state, state.keys()):
        raise U2V2InitializerError("U2-v2 full state hash drifted")
    for key in roles["patch"]:
        if key.startswith("patch_encoder.backbone."):
            _equal(state[key], state[key.removeprefix("patch_encoder.")], key=key)
    for key in roles["u0_shell"]:
        if key.endswith(("weight", "bias")) and int(torch.count_nonzero(state[key])):
            raise U2V2InitializerError(f"U0 shell parameter is nonzero: {key}")
    return dict(contract)


def validate_runtime_payload(
    model: torch.nn.Module, payload: Mapping[str, Any], *, checkpoint_label: str,
) -> dict[str, Any]:
    state = _state(payload, label=checkpoint_label)
    contract = payload.get("u2v2_initializer")
    if not isinstance(contract, Mapping) or contract.get("schema") != SCHEMA:
        raise U2V2InitializerError(f"{checkpoint_label} lacks U2-v2 provenance")
    roles = contract.get("role_keys")
    hashes = contract.get("role_tensor_sha256")
    if not isinstance(roles, Mapping) or not isinstance(hashes, Mapping):
        raise U2V2InitializerError(f"{checkpoint_label} lacks U2-v2 roles")
    frozen_keys = set()
    for role, keys in roles.items():
        if hashes.get(role) != stage_b_u0_tensor_state_sha256(state, keys):
            raise U2V2InitializerError(f"{checkpoint_label} changed frozen {role}")
        frozen_keys.update(keys)
    residual_keys = {
        key for key in state if key.startswith("stage_b_u2v2_rank_residual.")
    }
    expected_residual = {
        key for key in model.state_dict()
        if key.startswith("stage_b_u2v2_rank_residual.")
    }
    if residual_keys != expected_residual or len(residual_keys) not in {0, 9}:
        raise U2V2InitializerError(f"{checkpoint_label} residual state drifted")
    if set(state) != frozen_keys | residual_keys:
        raise U2V2InitializerError(f"{checkpoint_label} has unowned tensors")
    if set(state) != set(model.state_dict()):
        raise U2V2InitializerError(f"{checkpoint_label} does not match runtime model")
    if residual_keys:
        version = state.get("stage_b_u2v2_rank_residual.contract_version")
        limit = state.get("stage_b_u2v2_rank_residual.contract_residual_limit")
        expected_state = model.state_dict()
        expected_limit = expected_state[
            "stage_b_u2v2_rank_residual.contract_residual_limit"
        ]
        if (
            not torch.is_tensor(version)
            or not torch.is_tensor(limit)
            or int(version.detach().cpu().item()) != 1
            or not torch.equal(
                limit.detach().cpu(), expected_limit.detach().cpu()
            )
        ):
            raise U2V2InitializerError(f"{checkpoint_label} residual buffers drifted")
    return dict(contract)


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    if path.exists():
        raise U2V2InitializerError(f"refusing to overwrite {path}")
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
    create.add_argument("--stagea-checkpoint", required=True)
    create.add_argument("--stagea-sha256", required=True)
    create.add_argument("--r100-checkpoint", required=True)
    create.add_argument("--r100-sha256", required=True)
    create.add_argument("--c100-checkpoint", required=True)
    create.add_argument("--c100-sha256", required=True)
    create.add_argument("--config", default=str(DEFAULT_CONFIG))
    verify = commands.add_parser("verify")
    verify.add_argument("--checkpoint", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "create":
        payload = build_initializer_payload(
            stagea_checkpoint=Path(args.stagea_checkpoint), stagea_sha256=args.stagea_sha256,
            r100_checkpoint=Path(args.r100_checkpoint), r100_sha256=args.r100_sha256,
            c100_checkpoint=Path(args.c100_checkpoint), c100_sha256=args.c100_sha256,
            config=Path(args.config),
        )
        _write(Path(args.output), payload)
        checkpoint = Path(args.output).resolve(strict=True)
    else:
        checkpoint = Path(args.checkpoint).resolve(strict=True)
    contract = validate_initializer_payload(load_checkpoint(checkpoint))
    print(json.dumps({"status": "verified", "checkpoint": file_record(checkpoint), "contract": contract}, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, U2V2InitializerError) as error:
        raise SystemExit(f"[FAIL] {error}") from error
