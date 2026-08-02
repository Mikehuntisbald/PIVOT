#!/usr/bin/env python3
"""Convert a sealed U0-U100 checkpoint into a zero-gain U1 initializer."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import build_model_main  # noqa: E402
from models.GroundingDINO.stage_b_u0_patch_rank import (  # noqa: E402
    StageBU0PatchRankAdapter,
    U1_DIRECT_PATCH_ADDED_KEYS,
    U1_DIRECT_PATCH_INITIALIZER_SCHEMA,
    U1_DIRECT_PATCH_REPLACED_KEYS,
    stage_b_u0_tensor_state_sha256,
    validate_stage_b_u0_patch_rank_checkpoint,
    validate_stage_b_u1_direct_patch_initializer_payload,
)
from tools.stageb_dependency_audit import config_import_chain  # noqa: E402
from tools.stageb_gdino_adapter_probe_audit import file_record  # noqa: E402
from util.slconfig import SLConfig  # noqa: E402


DEFAULT_CONFIG = REPO_ROOT / "config/ablations/cfg_stageb_u1_direct_patch_rank.py"
U0_PREFIX = "stage_b_u0_patch_rank_adapter."
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


class U1InitializerError(RuntimeError):
    pass


def _load(path: Path) -> Mapping[str, Any]:
    scanner = getattr(torch.serialization, "get_unsafe_globals_in_checkpoint", None)
    if scanner is None:
        raise U1InitializerError("PyTorch cannot inspect checkpoint globals")
    observed = frozenset(scanner(path))
    unexpected = sorted(observed.difference(_ALLOWED_CHECKPOINT_GLOBALS))
    if unexpected:
        raise U1InitializerError(
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
        raise U1InitializerError(f"checkpoint is not a mapping: {path}")
    return payload


def _state(payload: Mapping[str, Any], *, label: str) -> Mapping[str, torch.Tensor]:
    state = payload.get("model")
    if not isinstance(state, Mapping) or not state:
        raise U1InitializerError(f"{label} has no model state")
    if any(not isinstance(key, str) or not torch.is_tensor(value) for key, value in state.items()):
        raise U1InitializerError(f"{label} model state is not tensor-only")
    return state


def compose_u1_model_state(
    template_state: Mapping[str, torch.Tensor],
    source_state: Mapping[str, torch.Tensor],
) -> tuple[OrderedDict[str, torch.Tensor], dict[str, list[str]]]:
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    roles = {"source_preserved": [], "u1_added": [], "u1_replaced": []}
    for key, wanted in template_state.items():
        if key in U1_DIRECT_PATCH_ADDED_KEYS:
            value = wanted
            role = "u1_added"
        elif key in U1_DIRECT_PATCH_REPLACED_KEYS:
            if key not in source_state:
                raise U1InitializerError(f"U1 source lacks replaced contract key: {key}")
            value = wanted
            role = "u1_replaced"
        elif key in source_state:
            value = source_state[key]
            role = "source_preserved"
        else:
            raise U1InitializerError(f"U1 template has unbound tensor: {key}")
        if value.dtype != wanted.dtype or tuple(value.shape) != tuple(wanted.shape):
            raise U1InitializerError(f"U1 shape/dtype mismatch at {key}")
        result[key] = value.detach().cpu().clone()
        roles[role].append(key)
    consumed_source = set(roles["source_preserved"]) | set(
        U1_DIRECT_PATCH_REPLACED_KEYS
    )
    if consumed_source != set(source_state):
        raise U1InitializerError(
            "U1 source key coverage drifted "
            f"(missing={sorted(set(source_state) - consumed_source)[:8]})"
        )
    if set(roles["u1_added"]) != set(U1_DIRECT_PATCH_ADDED_KEYS):
        raise U1InitializerError("U1 added-key role is incomplete")
    return result, roles


def _config_binding(config: Path) -> dict[str, Any]:
    return {
        "leaf": file_record(config),
        "import_chain": [
            file_record(path)
            for path in config_import_chain(config, root=REPO_ROOT)
        ],
    }


def _build_model(config: Path):
    cfg = SLConfig.fromfile(str(config))
    cfg.device = "cpu"
    if not bool(getattr(cfg, "stage_b_u1_direct_patch_skip", False)):
        raise U1InitializerError("config does not enable U1 direct patch skip")
    torch.manual_seed(0)
    model, _criterion, _postprocessors = build_model_main(cfg)
    return model.eval()


def _source_u0_adapter(source_state: Mapping[str, torch.Tensor]) -> StageBU0PatchRankAdapter:
    adapter = StageBU0PatchRankAdapter(query_count=900, hidden_dim=64, score_clip=5.0)
    adapter_state = {
        key.removeprefix(U0_PREFIX): value
        for key, value in source_state.items()
        if key.startswith(U0_PREFIX)
    }
    adapter.load_state_dict(adapter_state, strict=True)
    return adapter.eval()


def build_payload(
    *,
    source_checkpoint: Path,
    source_sha256: str,
    config: Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    source_checkpoint = source_checkpoint.resolve(strict=True)
    source_record = file_record(source_checkpoint)
    if source_record["sha256"] != str(source_sha256).strip().lower():
        raise U1InitializerError("U1 source checkpoint SHA256 mismatch")
    source_payload = _load(source_checkpoint)
    if int(source_payload.get("optimizer_updates", -1)) != 100:
        raise U1InitializerError("U1 source must be the sealed U0-U100 milestone")
    source_state = _state(source_payload, label="U1 source")
    source_adapter = _source_u0_adapter(source_state)
    model = _build_model(config.resolve())
    state, roles = compose_u1_model_state(model.state_dict(), source_state)
    model.load_state_dict(state, strict=True)
    validate_stage_b_u0_patch_rank_checkpoint(
        model, state, checkpoint_label="U1 initializer"
    )
    generator = torch.Generator(device="cpu").manual_seed(20_260_720)
    patch = torch.randn(2, 900, generator=generator)
    teacher = torch.randn(2, 900, generator=generator)
    with torch.inference_mode():
        expected = source_adapter(patch, teacher)
        observed = model.stage_b_u0_patch_rank_adapter(patch, teacher)
    functional = {
        key: bool(torch.equal(observed[key], expected[key]))
        for key in (
            "teacher_rank_score",
            "patch_rank_residual",
            "rank_score",
        )
    }
    if not all(functional.values()):
        raise U1InitializerError(f"zero-gain U1 differs from U100: {functional}")
    contract = {
        "schema": U1_DIRECT_PATCH_INITIALIZER_SCHEMA,
        "training_initializer": True,
        "resumable": False,
        "source": source_record,
        "source_optimizer_updates": 100,
        "config": _config_binding(config.resolve()),
        "model_state_keys": len(state),
        "role_keys": {key: list(value) for key, value in roles.items()},
        "role_key_counts": {key: len(value) for key, value in roles.items()},
        "source_preserved_tensor_sha256": stage_b_u0_tensor_state_sha256(
            state, roles["source_preserved"]
        ),
        "u1_added_tensor_sha256": stage_b_u0_tensor_state_sha256(
            state, roles["u1_added"]
        ),
        "u1_replaced_tensor_sha256": stage_b_u0_tensor_state_sha256(
            state, roles["u1_replaced"]
        ),
        "full_model_tensor_sha256": stage_b_u0_tensor_state_sha256(
            state, list(state)
        ),
        "u100_functional_bitwise": functional,
        "invariants": {
            "u100_source_preserved_bitwise": True,
            "direct_patch_gain_zero": True,
            "u1_rank_equals_u100_at_initialization": True,
            "r100_p50_b58_frozen_source_unchanged": True,
        },
    }
    payload = {"model": state, "u1_initializer": contract}
    validate_stage_b_u1_direct_patch_initializer_payload(
        model, payload, checkpoint_label="built U1 initializer"
    )
    del model, source_adapter, source_payload
    gc.collect()
    return payload


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.resolve()
    if path.exists():
        raise U1InitializerError(f"refusing to overwrite U1 initializer: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    if temporary.exists():
        raise U1InitializerError(f"stale U1 initializer temporary exists: {temporary}")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify(path: Path) -> dict[str, Any]:
    path = path.resolve(strict=True)
    payload = _load(path)
    state = _state(payload, label="serialized U1 initializer")
    validate_stage_b_u1_direct_patch_initializer_payload(
        state, payload, checkpoint_label="serialized U1 initializer"
    )
    return {
        "schema": U1_DIRECT_PATCH_INITIALIZER_SCHEMA,
        "status": "verified",
        "checkpoint": file_record(path),
        "contract": dict(payload["u1_initializer"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--source-checkpoint", required=True)
    create.add_argument("--source-sha256", required=True)
    create.add_argument("--config", default=str(DEFAULT_CONFIG))
    create.add_argument("--output", required=True)
    check = sub.add_parser("verify")
    check.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    if args.command == "create":
        output = Path(args.output)
        payload = build_payload(
            source_checkpoint=Path(args.source_checkpoint),
            source_sha256=args.source_sha256,
            config=Path(args.config),
        )
        _write(output, payload)
        result = verify(output)
    else:
        result = verify(Path(args.checkpoint))
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(f"[FAIL] {error}") from error
