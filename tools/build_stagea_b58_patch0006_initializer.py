#!/usr/bin/env python3
"""Build the Stage-A B58-trunk / checkpoint0006-patch initializer.

The patch encoder shares the main image backbone by object identity.  Therefore
``patch_encoder.backbone.*`` must mirror B58 rather than checkpoint0006; loading
those keys from checkpoint0006 would silently overwrite the B58 trunk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


SCHEMA = "pivot.stagea.b58_trunk_patch0006_initializer/v1"
DEFAULT_B58 = Path(
    "/media/haoyi/T9/gdino/outputs/"
    "gdino_ft_stageb_from_gdino_ft_e1_with_tn_bs19_nopatchbranch/"
    "checkpoint0001.pth"
)
DEFAULT_PATCH0006 = Path(
    "/media/haoyi/T9/gdino/outputs/stageA_coco_multipatch/checkpoint0006.pth"
)
EXPECTED_B58_SHA256 = (
    "b58e5209dc07dbffb2e5ed3d792e0db70c3306ec2ed24389693b4aeeebab1157"
)
EXPECTED_PATCH0006_SHA256 = (
    "a4f153c8cbd9b408b9479901e27ec486a10f393013193d44b0da1dcd1888cb91"
)
PATCH_TRANSFER_PREFIXES = (
    "patch_encoder.input_proj.",
    "patch_encoder.norm.",
    "query_proj_for_patch.",
)
PATCH_TRANSFER_EXACT = frozenset({"patch_logit_scale"})
DISABLED_QUERY_KEYS = frozenset({"patch_dn_tgt"})
PATCH_BACKBONE_PREFIX = "patch_encoder.backbone."
MAIN_BACKBONE_PREFIX = "backbone."

_NUMPY_SAFE_GLOBALS = (
    np.ndarray,
    np._core.multiarray._reconstruct,
    np.dtype,
    type(np.dtype(np.uint32)),
)
_ALLOWED_CHECKPOINT_GLOBALS = frozenset(
    {
        "numpy.ndarray",
        "numpy._core.multiarray._reconstruct",
        "numpy.dtype",
    }
)


class StageAInitializerError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def safe_load(path: Path) -> Mapping[str, Any]:
    path = path.resolve(strict=True)
    scanner = getattr(torch.serialization, "get_unsafe_globals_in_checkpoint", None)
    if scanner is None:
        raise StageAInitializerError("PyTorch cannot inspect checkpoint globals")
    unexpected = sorted(set(scanner(path)).difference(_ALLOWED_CHECKPOINT_GLOBALS))
    if unexpected:
        raise StageAInitializerError(
            f"checkpoint requires unknown pickle globals: {unexpected}"
        )
    with torch.serialization.safe_globals(list(_NUMPY_SAFE_GLOBALS)):
        payload = torch.load(
            path,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
    if not isinstance(payload, Mapping):
        raise StageAInitializerError(f"checkpoint is not a mapping: {path}")
    return payload


def model_state(payload: Mapping[str, Any], *, label: str) -> Mapping[str, torch.Tensor]:
    state = payload.get("model")
    if not isinstance(state, Mapping) or not state:
        raise StageAInitializerError(f"{label} has no model state")
    if any(
        not isinstance(key, str) or not torch.is_tensor(value)
        for key, value in state.items()
    ):
        raise StageAInitializerError(f"{label} model state is not tensor-only")
    return state


def is_patch_transfer_key(key: str) -> bool:
    return key in PATCH_TRANSFER_EXACT or key.startswith(PATCH_TRANSFER_PREFIXES)


def _same_tensor_contract(
    value: torch.Tensor, template: torch.Tensor, *, key: str
) -> None:
    if value.dtype != template.dtype or tuple(value.shape) != tuple(template.shape):
        raise StageAInitializerError(
            f"shape/dtype mismatch at {key}: source={tuple(value.shape)}/{value.dtype}, "
            f"template={tuple(template.shape)}/{template.dtype}"
        )


def compose_model_state(
    b58_state: Mapping[str, torch.Tensor],
    patch0006_state: Mapping[str, torch.Tensor],
) -> tuple[OrderedDict[str, torch.Tensor], dict[str, list[str]]]:
    """Compose a no-DN Stage-A state without allowing 0006 to alter B58 trunk."""
    if not set(b58_state).issubset(patch0006_state):
        missing = sorted(set(b58_state).difference(patch0006_state))
        raise StageAInitializerError(
            f"checkpoint0006 does not cover the B58 architecture: {missing[:8]}"
        )

    roles = {
        "b58_trunk": [],
        "b58_shared_patch_backbone_mirror": [],
        "patch0006_transfer": [],
        "disabled_query_state": [],
    }
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, template in patch0006_state.items():
        if key in DISABLED_QUERY_KEYS:
            roles["disabled_query_state"].append(key)
            continue
        if key in b58_state:
            value = b58_state[key]
            role = "b58_trunk"
        elif key.startswith(PATCH_BACKBONE_PREFIX):
            mirror_key = MAIN_BACKBONE_PREFIX + key.removeprefix(PATCH_BACKBONE_PREFIX)
            if mirror_key not in b58_state:
                raise StageAInitializerError(
                    f"shared patch backbone key has no B58 mirror: {key}"
                )
            value = b58_state[mirror_key]
            role = "b58_shared_patch_backbone_mirror"
        elif is_patch_transfer_key(key):
            value = patch0006_state[key]
            role = "patch0006_transfer"
        else:
            raise StageAInitializerError(f"unowned checkpoint0006 tensor: {key}")
        _same_tensor_contract(value, template, key=key)
        result[key] = value.detach().cpu()
        roles[role].append(key)

    if set(roles["b58_trunk"]) != set(b58_state):
        raise StageAInitializerError("B58 trunk coverage is incomplete")
    expected_patch = {key for key in patch0006_state if is_patch_transfer_key(key)}
    if set(roles["patch0006_transfer"]) != expected_patch:
        raise StageAInitializerError("checkpoint0006 patch transfer coverage is incomplete")
    if set(roles["disabled_query_state"]) != DISABLED_QUERY_KEYS:
        raise StageAInitializerError("the no-DN query-state contract drifted")
    if not roles["b58_shared_patch_backbone_mirror"]:
        raise StageAInitializerError("shared patch backbone mirror is empty")
    return result, roles


def tensor_state_sha256(
    state: Mapping[str, torch.Tensor], keys: Sequence[str]
) -> str:
    digest = hashlib.sha256()
    for key in keys:
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def build_payload(
    *,
    b58_path: Path,
    patch0006_path: Path,
) -> dict[str, Any]:
    b58_record = file_record(b58_path)
    patch_record = file_record(patch0006_path)
    if b58_record["sha256"] != EXPECTED_B58_SHA256:
        raise StageAInitializerError("B58 checkpoint SHA256 mismatch")
    if patch_record["sha256"] != EXPECTED_PATCH0006_SHA256:
        raise StageAInitializerError("checkpoint0006 SHA256 mismatch")

    b58_payload = safe_load(b58_path)
    patch_payload = safe_load(patch0006_path)
    state, roles = compose_model_state(
        model_state(b58_payload, label="B58"),
        model_state(patch_payload, label="checkpoint0006"),
    )
    contract = {
        "schema": SCHEMA,
        "training_initializer": True,
        "resumable": False,
        "sources": {"b58_trunk": b58_record, "patch0006": patch_record},
        "model_state_keys": len(state),
        "role_keys": roles,
        "role_key_counts": {name: len(keys) for name, keys in roles.items()},
        "role_tensor_sha256": {
            name: tensor_state_sha256(state, keys)
            for name, keys in roles.items()
            if name != "disabled_query_state"
        },
        "full_model_tensor_sha256": tensor_state_sha256(state, list(state)),
        "invariants": {
            "b58_common_trunk_preserved_bitwise": True,
            "shared_patch_backbone_is_b58_bitwise": True,
            "patch_projection_from_checkpoint0006_bitwise": True,
            "decoder_encoder_text_box_and_main_backbone_frozen_by_training_config": True,
            "patch_dn_query_disabled": True,
        },
    }
    return {"model": state, "stage_a_b58_patch0006_initializer": contract}


def validate_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    state = model_state(payload, label="Stage-A initializer")
    contract = payload.get("stage_a_b58_patch0006_initializer")
    if not isinstance(contract, Mapping) or contract.get("schema") != SCHEMA:
        raise StageAInitializerError("initializer provenance schema is invalid")
    roles = contract.get("role_keys")
    if not isinstance(roles, Mapping):
        raise StageAInitializerError("initializer role map is missing")
    flattened = [key for name, keys in roles.items() if name != "disabled_query_state" for key in keys]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(state):
        raise StageAInitializerError("initializer tensor role coverage is invalid")
    if any(key in state for key in DISABLED_QUERY_KEYS):
        raise StageAInitializerError("disabled DN query state leaked into initializer")
    observed = tensor_state_sha256(state, list(state))
    if observed != contract.get("full_model_tensor_sha256"):
        raise StageAInitializerError("initializer full-model tensor digest mismatch")
    return {
        "schema": SCHEMA,
        "status": "verified",
        "model_state_keys": len(state),
        "role_key_counts": dict(contract.get("role_key_counts", {})),
        "full_model_tensor_sha256": observed,
    }


def write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    if path.exists():
        raise StageAInitializerError(f"refusing to overwrite initializer: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    if temporary.exists():
        raise StageAInitializerError(f"stale temporary initializer exists: {temporary}")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--b58", type=Path, default=DEFAULT_B58)
    build.add_argument("--patch0006", type=Path, default=DEFAULT_PATCH0006)
    build.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--checkpoint", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        payload = build_payload(b58_path=args.b58, patch0006_path=args.patch0006)
        write_atomic(args.output, payload)
        report = validate_payload(safe_load(args.output))
        report["checkpoint"] = file_record(args.output)
    else:
        report = validate_payload(safe_load(args.checkpoint))
        report["checkpoint"] = file_record(args.checkpoint)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
