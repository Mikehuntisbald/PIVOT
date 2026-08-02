#!/usr/bin/env python3
"""Build a strict, deterministic b58-only D1 patch-category initializer."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.stageb_dependency_audit import config_import_chain  # noqa: E402
from util import misc as utils  # noqa: E402
from util.slconfig import SLConfig  # noqa: E402


NATIVE_PATCH_CATEGORY_INITIALIZER_SCHEMA = (
    "pivot.stageb.native_patch_category_initializer/v1"
)
DEFAULT_CONFIG = (
    REPO_ROOT / "config/ablations/cfg_stageb_native_patch_category_d1.py"
)
DEFAULT_B58 = Path(
    "/media/haoyi/T9/gdino/outputs/"
    "gdino_ft_stageb_from_gdino_ft_e1_with_tn_bs19_nopatchbranch/"
    "checkpoint0001.pth"
)
EXPECTED_B58_SHA256 = (
    "b58e5209dc07dbffb2e5ed3d792e0db70c3306ec2ed24389693b4aeeebab1157"
)

PATCH_BACKBONE_PREFIX = "patch_encoder.backbone."
RANDOM_TRAINABLE_PATCH_KEYS = frozenset(
    {
        "patch_encoder.input_proj.0.weight",
        "patch_encoder.input_proj.0.bias",
        "patch_encoder.input_proj.1.weight",
        "patch_encoder.input_proj.1.bias",
        "patch_encoder.norm.weight",
        "patch_encoder.norm.bias",
        "query_proj_for_patch.weight",
        "query_proj_for_patch.bias",
    }
)
RANDOM_FROZEN_PATCH_KEYS = frozenset({"patch_logit_scale"})
RANDOM_TEMPLATE_KEYS = RANDOM_TRAINABLE_PATCH_KEYS | RANDOM_FROZEN_PATCH_KEYS
ROLE_NAMES = frozenset(
    {
        "b58_base",
        "shared_backbone_alias",
        "random_trainable_patch_projection",
        "random_frozen_patch_scale",
    }
)
FORBIDDEN_SOURCE_PREFIXES = (
    "patch_encoder.",
    "query_proj_for_patch.",
    "stage_b_gdino_score_adapter.",
    "stage_b_u0_patch_rank_adapter.",
    "stage_b_data_driven_score_heads.",
    "stage_b_native_patch_category.",
)
FORBIDDEN_SOURCE_KEYS = frozenset({"patch_logit_scale"})
FORBIDDEN_MODE_FLAGS = (
    "patch_only",
    "stage_b",
    "stage_b_legacy_score_head",
    "stage_b_gdino_score_adapter",
    "stage_b_u0_patch_rank",
    "stage_b_data_driven_score",
)
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


class NativePatchCategoryInitializerError(RuntimeError):
    """The requested artifact violates the D1 initialization contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_file_record(path: Path, *, label: str) -> dict[str, Any]:
    """Hash a regular file and reject concurrent replacement or mutation."""
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise NativePatchCategoryInitializerError(
            f"{label} cannot be resolved: {path}: {error}"
        ) from error
    if not resolved.is_file():
        raise NativePatchCategoryInitializerError(f"{label} is not a file: {resolved}")
    before = resolved.stat()
    digest = _sha256_file(resolved)
    after = resolved.stat()
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise NativePatchCategoryInitializerError(
            f"{label} changed while it was hashed"
        )
    return {
        "path": str(resolved),
        "size_bytes": int(after.st_size),
        "sha256": digest,
    }


def _safe_load_checkpoint(path: Path, *, label: str) -> MutableMapping[str, Any]:
    """Load tensor-only checkpoints without permitting arbitrary pickle code."""
    resolved = path.expanduser().resolve(strict=True)
    scanner = getattr(torch.serialization, "get_unsafe_globals_in_checkpoint", None)
    if scanner is None:
        raise NativePatchCategoryInitializerError(
            "installed PyTorch cannot statically inspect checkpoint globals"
        )
    try:
        observed_globals = frozenset(scanner(resolved))
    except Exception as error:
        raise NativePatchCategoryInitializerError(
            f"could not inspect globals in {label}: {error}"
        ) from error
    unexpected = sorted(observed_globals.difference(_ALLOWED_CHECKPOINT_GLOBALS))
    if unexpected:
        raise NativePatchCategoryInitializerError(
            f"{label} requires unsafe or unknown pickle globals: {unexpected}"
        )
    try:
        with torch.serialization.safe_globals(list(_NUMPY_SAFE_GLOBALS)):
            payload = torch.load(
                resolved,
                map_location="cpu",
                mmap=True,
                weights_only=True,
            )
    except Exception as error:
        raise NativePatchCategoryInitializerError(
            f"could not safely load {label}: {error}"
        ) from error
    if not isinstance(payload, MutableMapping):
        raise NativePatchCategoryInitializerError(f"{label} payload is not a mapping")
    return payload


def native_patch_category_tensor_state_sha256(
    state: Mapping[str, Any], keys: Sequence[str]
) -> str:
    """Hash selected tensors including their names, dtypes, and shapes."""
    selected = sorted(set(str(key) for key in keys))
    if not selected:
        raise ValueError("cannot hash an empty native patch-category tensor selection")
    digest = hashlib.sha256()
    for key in selected:
        value = state.get(key)
        if not torch.is_tensor(value):
            raise ValueError(f"native patch-category model state {key!r} is missing")
        header = json.dumps(
            [key, str(value.dtype), list(value.shape)],
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        tensor = value.detach().cpu().contiguous()
        if tensor.numel():
            digest.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()


def _exact_tensor(
    value: torch.Tensor, wanted: torch.Tensor, *, key: str
) -> None:
    if value.dtype != wanted.dtype or tuple(value.shape) != tuple(wanted.shape):
        raise NativePatchCategoryInitializerError(
            "initializer tensor shape/dtype mismatch at "
            f"{key}: {tuple(value.shape)}/{value.dtype} != "
            f"{tuple(wanted.shape)}/{wanted.dtype}"
        )


def extract_b58_source_state(
    payload: Mapping[str, Any], *, checkpoint_label: str = "b58 source checkpoint"
) -> Mapping[str, torch.Tensor]:
    """Return a clean, tensor-only b58 state and reject derived-model sources."""
    if not isinstance(payload, Mapping):
        raise NativePatchCategoryInitializerError(
            f"{checkpoint_label} payload is not a mapping"
        )
    state = payload.get("model")
    if not isinstance(state, Mapping) or not state:
        raise NativePatchCategoryInitializerError(
            f"{checkpoint_label} has no model mapping"
        )
    cleaned = utils.clean_state_dict(state)
    invalid = [
        key
        for key, value in cleaned.items()
        if not isinstance(key, str) or not torch.is_tensor(value)
    ]
    if invalid:
        raise NativePatchCategoryInitializerError(
            f"{checkpoint_label} contains non-tensor model values: {invalid[:8]}"
        )
    forbidden = sorted(
        key
        for key in cleaned
        if key in FORBIDDEN_SOURCE_KEYS
        or any(key.startswith(prefix) for prefix in FORBIDDEN_SOURCE_PREFIXES)
    )
    if forbidden:
        raise NativePatchCategoryInitializerError(
            f"{checkpoint_label} contains forbidden derived/patch tensors: "
            f"{forbidden[:8]}"
        )
    return cleaned


def _require_b58_source_record(record: Mapping[str, Any]) -> None:
    if not isinstance(record, Mapping) or set(record) != {
        "path",
        "size_bytes",
        "sha256",
    }:
        raise NativePatchCategoryInitializerError("b58 source record is malformed")
    if record.get("sha256") != EXPECTED_B58_SHA256:
        raise NativePatchCategoryInitializerError(
            "b58 source SHA256 mismatch: expected "
            f"{EXPECTED_B58_SHA256}, got {record.get('sha256')}"
        )


def compose_native_patch_category_state(
    template: Mapping[str, torch.Tensor],
    b58: Mapping[str, torch.Tensor],
) -> tuple[OrderedDict[str, torch.Tensor], dict[str, list[str]]]:
    """Bind every D1 tensor to exactly one approved initialization source."""
    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    roles = {role: [] for role in sorted(ROLE_NAMES)}
    for key, template_value in template.items():
        if not isinstance(key, str) or not torch.is_tensor(template_value):
            raise NativePatchCategoryInitializerError(
                f"D1 template contains a non-tensor entry at {key!r}"
            )
        if key in b58:
            value = b58[key]
            role = "b58_base"
        elif key.startswith(PATCH_BACKBONE_PREFIX):
            source_key = key.removeprefix("patch_encoder.")
            if source_key not in b58:
                raise NativePatchCategoryInitializerError(
                    f"patch backbone alias has no b58 source: {key} -> {source_key}"
                )
            value = b58[source_key]
            role = "shared_backbone_alias"
        elif key in RANDOM_TRAINABLE_PATCH_KEYS:
            value = template_value
            role = "random_trainable_patch_projection"
        elif key in RANDOM_FROZEN_PATCH_KEYS:
            value = template_value
            role = "random_frozen_patch_scale"
        else:
            raise NativePatchCategoryInitializerError(
                f"D1 template contains an unbound tensor: {key}"
            )
        _exact_tensor(value, template_value, key=key)
        result[key] = value.detach().cpu().clone()
        roles[role].append(key)

    if set(roles["b58_base"]) != set(b58):
        raise NativePatchCategoryInitializerError(
            "b58 keys do not map exactly into the D1 template: "
            f"missing={sorted(set(b58) - set(roles['b58_base']))[:8]}"
        )
    if set(roles["random_trainable_patch_projection"]) != set(
        RANDOM_TRAINABLE_PATCH_KEYS
    ):
        raise NativePatchCategoryInitializerError(
            "D1 trainable random-key contract is incomplete or drifted"
        )
    if set(roles["random_frozen_patch_scale"]) != set(
        RANDOM_FROZEN_PATCH_KEYS
    ):
        raise NativePatchCategoryInitializerError(
            "D1 frozen random-key contract is incomplete or drifted"
        )
    if not roles["b58_base"] or not roles["shared_backbone_alias"]:
        raise NativePatchCategoryInitializerError(
            "D1 b58/shared-backbone roles must be non-empty"
        )
    for alias in roles["shared_backbone_alias"]:
        source_key = alias.removeprefix("patch_encoder.")
        if source_key not in result or not torch.equal(
            result[alias], result[source_key]
        ):
            raise NativePatchCategoryInitializerError(
                f"shared patch backbone differs from b58 at {alias}"
            )
    return result, roles


def _expected_state(expected_model: nn.Module | Mapping[str, torch.Tensor]):
    if isinstance(expected_model, nn.Module):
        return expected_model.state_dict()
    if isinstance(expected_model, Mapping):
        return expected_model
    raise TypeError("expected_model must be a module or state mapping")


def build_native_patch_category_contract(
    *,
    state: Mapping[str, torch.Tensor],
    roles: Mapping[str, Sequence[str]],
    b58_record: Mapping[str, Any],
    config_binding: Mapping[str, Any],
    seed: int,
    architecture: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the manifest embedded in a D1 initializer payload."""
    _require_b58_source_record(b58_record)
    normalized = {role: list(roles[role]) for role in sorted(ROLE_NAMES)}
    contract: dict[str, Any] = {
        "schema": NATIVE_PATCH_CATEGORY_INITIALIZER_SCHEMA,
        "seed": int(seed),
        "sources": {"b58": dict(b58_record)},
        "random_initialization": {
            "source": "fresh_model_template",
            "seed": int(seed),
            "trainable_tensor_keys": sorted(RANDOM_TRAINABLE_PATCH_KEYS),
            "frozen_tensor_keys": sorted(RANDOM_FROZEN_PATCH_KEYS),
        },
        "config": dict(config_binding),
        "architecture": dict(architecture),
        "role_key_counts": {
            role: len(keys) for role, keys in normalized.items()
        },
        "role_keys": normalized,
        "full_model_tensor_sha256": native_patch_category_tensor_state_sha256(
            state, sorted(state)
        ),
        "invariants": {
            "b58_is_only_checkpoint_source": True,
            "no_u2_r100_p50_stagea_or_teacher_tensor_source": True,
            "b58_same_name_tensors_copied_bitwise": True,
            "patch_backbone_aliases_b58_bitwise": True,
            "only_declared_patch_tensors_use_random_template_state": True,
            "patch_logit_scale_is_frozen": True,
        },
    }
    for role, keys in normalized.items():
        contract[f"{role}_tensor_sha256"] = (
            native_patch_category_tensor_state_sha256(state, keys)
        )
    return contract


def validate_native_patch_category_initializer_payload(
    expected_model: nn.Module | Mapping[str, torch.Tensor],
    payload: Mapping[str, Any],
    *,
    checkpoint_label: str,
    expected_b58_state: Mapping[str, torch.Tensor] | None = None,
) -> None:
    """Validate D1 coverage, provenance, hashes, and optional b58 anchoring."""
    if not isinstance(payload, Mapping) or set(payload) != {
        "model",
        "native_patch_category_initializer",
    }:
        raise ValueError(
            f"{checkpoint_label}: native patch-category top-level keys drifted"
        )
    state = payload.get("model")
    contract = payload.get("native_patch_category_initializer")
    if not isinstance(state, Mapping) or not isinstance(contract, Mapping):
        raise ValueError(f"{checkpoint_label}: initializer payload is malformed")
    if contract.get("schema") != NATIVE_PATCH_CATEGORY_INITIALIZER_SCHEMA:
        raise ValueError(f"{checkpoint_label}: initializer schema drifted")

    expected = _expected_state(expected_model)
    if set(state) != set(expected):
        raise ValueError(f"{checkpoint_label}: initializer model key coverage drifted")
    for key, wanted in expected.items():
        value = state.get(key)
        if (
            not torch.is_tensor(value)
            or not torch.is_tensor(wanted)
            or value.dtype != wanted.dtype
            or tuple(value.shape) != tuple(wanted.shape)
        ):
            raise ValueError(
                f"{checkpoint_label}: initializer tensor shape/dtype drift at {key}"
            )

    roles = contract.get("role_keys")
    counts = contract.get("role_key_counts")
    if (
        not isinstance(roles, Mapping)
        or set(roles) != ROLE_NAMES
        or not isinstance(counts, Mapping)
        or set(counts) != ROLE_NAMES
    ):
        raise ValueError(f"{checkpoint_label}: initializer roles drifted")
    normalized: dict[str, list[str]] = {}
    for role in ROLE_NAMES:
        keys = roles.get(role)
        if (
            not isinstance(keys, list)
            or not keys
            or counts.get(role) != len(keys)
            or any(not isinstance(key, str) for key in keys)
        ):
            raise ValueError(f"{checkpoint_label}: initializer role {role} drifted")
        normalized[role] = list(keys)
    flattened = [key for keys in normalized.values() for key in keys]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(state):
        raise ValueError(
            f"{checkpoint_label}: initializer roles are not an exact partition"
        )
    if set(normalized["random_trainable_patch_projection"]) != set(
        RANDOM_TRAINABLE_PATCH_KEYS
    ):
        raise ValueError(f"{checkpoint_label}: trainable random-key role drifted")
    if set(normalized["random_frozen_patch_scale"]) != set(
        RANDOM_FROZEN_PATCH_KEYS
    ):
        raise ValueError(f"{checkpoint_label}: frozen random-key role drifted")
    if not all(
        key.startswith(PATCH_BACKBONE_PREFIX)
        for key in normalized["shared_backbone_alias"]
    ):
        raise ValueError(f"{checkpoint_label}: shared-backbone role drifted")
    remaining = set(state) - set(normalized["shared_backbone_alias"]) - set(
        RANDOM_TEMPLATE_KEYS
    )
    if set(normalized["b58_base"]) != remaining:
        raise ValueError(f"{checkpoint_label}: b58 base role drifted")
    for alias in normalized["shared_backbone_alias"]:
        source_key = alias.removeprefix("patch_encoder.")
        if source_key not in state or not torch.equal(state[alias], state[source_key]):
            raise ValueError(
                f"{checkpoint_label}: shared-backbone alias drifted at {alias}"
            )

    sources = contract.get("sources")
    if not isinstance(sources, Mapping) or set(sources) != {"b58"}:
        raise ValueError(f"{checkpoint_label}: checkpoint-source provenance drifted")
    try:
        _require_b58_source_record(sources["b58"])
    except NativePatchCategoryInitializerError as error:
        raise ValueError(f"{checkpoint_label}: {error}") from error
    random_initialization = contract.get("random_initialization")
    if not isinstance(random_initialization, Mapping) or random_initialization != {
        "source": "fresh_model_template",
        "seed": contract.get("seed"),
        "trainable_tensor_keys": sorted(RANDOM_TRAINABLE_PATCH_KEYS),
        "frozen_tensor_keys": sorted(RANDOM_FROZEN_PATCH_KEYS),
    }:
        raise ValueError(f"{checkpoint_label}: random provenance drifted")
    if isinstance(contract.get("seed"), bool) or not isinstance(
        contract.get("seed"), int
    ):
        raise ValueError(f"{checkpoint_label}: initializer seed drifted")

    for role, keys in normalized.items():
        wanted_hash = contract.get(f"{role}_tensor_sha256")
        observed_hash = native_patch_category_tensor_state_sha256(state, keys)
        if wanted_hash != observed_hash:
            raise ValueError(
                f"{checkpoint_label}: initializer {role} tensor hash drifted"
            )
    if contract.get(
        "full_model_tensor_sha256"
    ) != native_patch_category_tensor_state_sha256(state, sorted(state)):
        raise ValueError(f"{checkpoint_label}: full-model tensor hash drifted")

    invariants = contract.get("invariants")
    required_invariants = {
        "b58_is_only_checkpoint_source",
        "no_u2_r100_p50_stagea_or_teacher_tensor_source",
        "b58_same_name_tensors_copied_bitwise",
        "patch_backbone_aliases_b58_bitwise",
        "only_declared_patch_tensors_use_random_template_state",
        "patch_logit_scale_is_frozen",
    }
    if (
        not isinstance(invariants, Mapping)
        or set(invariants) != required_invariants
        or any(invariants.get(key) is not True for key in required_invariants)
    ):
        raise ValueError(f"{checkpoint_label}: initializer invariants drifted")

    if expected_b58_state is not None:
        if set(expected_b58_state) != set(normalized["b58_base"]):
            raise ValueError(f"{checkpoint_label}: external b58 key coverage drifted")
        for key in normalized["b58_base"]:
            source = expected_b58_state.get(key)
            if not torch.is_tensor(source) or not torch.equal(state[key], source):
                raise ValueError(
                    f"{checkpoint_label}: b58-anchored tensor drifted at {key}"
                )


def _build_template(config: Path, seed: int):
    from main import build_model_main

    cfg = SLConfig.fromfile(str(config))
    cfg.device = "cpu"
    if not bool(getattr(cfg, "stage_b_native_patch_category", False)):
        raise NativePatchCategoryInitializerError(
            "initializer config must enable stage_b_native_patch_category"
        )
    if not bool(getattr(cfg, "enable_patch_branch", False)):
        raise NativePatchCategoryInitializerError(
            "initializer config must enable the patch branch"
        )
    if bool(getattr(cfg, "patch_gate_with_text", True)):
        raise NativePatchCategoryInitializerError(
            "initializer config must disable patch_gate_with_text"
        )
    enabled_forbidden = [
        name for name in FORBIDDEN_MODE_FLAGS if bool(getattr(cfg, name, False))
    ]
    if enabled_forbidden:
        raise NativePatchCategoryInitializerError(
            f"initializer config enables incompatible modes: {enabled_forbidden}"
        )
    torch.manual_seed(int(seed))
    model, _criterion, _postprocessors = build_model_main(cfg)
    return model.eval(), cfg


def _config_binding(path: Path) -> dict[str, Any]:
    chain = config_import_chain(path.resolve(), root=REPO_ROOT)
    return {
        "leaf": stable_file_record(path, label="D1 initializer config"),
        "import_chain": [
            stable_file_record(item, label=f"config dependency {item.name}")
            for item in chain
        ],
    }


def build_payload(
    *, config: Path, b58_path: Path, seed: int
) -> tuple[dict[str, Any], nn.Module]:
    b58_record = stable_file_record(b58_path, label="b58 source checkpoint")
    _require_b58_source_record(b58_record)
    model, cfg = _build_template(config, seed)
    source_payload = _safe_load_checkpoint(b58_path, label="b58 source checkpoint")
    source_state = extract_b58_source_state(source_payload)
    state, roles = compose_native_patch_category_state(
        model.state_dict(), source_state
    )
    model.load_state_dict(state, strict=True)
    architecture = {
        "hidden_dim": int(cfg.hidden_dim),
        "num_queries": int(cfg.num_queries),
        "enable_patch_branch": bool(cfg.enable_patch_branch),
        "patch_gate_with_text": bool(cfg.patch_gate_with_text),
        "stage_b_native_patch_category": bool(
            cfg.stage_b_native_patch_category
        ),
        "trainable_tensor_keys": sorted(RANDOM_TRAINABLE_PATCH_KEYS),
        "frozen_tensor_keys": sorted(RANDOM_FROZEN_PATCH_KEYS),
    }
    contract = build_native_patch_category_contract(
        state=state,
        roles=roles,
        b58_record=b58_record,
        config_binding=_config_binding(config),
        seed=seed,
        architecture=architecture,
    )
    payload = {
        "model": state,
        "native_patch_category_initializer": contract,
    }
    validate_native_patch_category_initializer_payload(
        model,
        payload,
        checkpoint_label="in-memory D1 initializer",
        expected_b58_state=source_state,
    )
    del source_payload
    gc.collect()
    return payload, model


def _default_output(seed: int) -> Path:
    return (
        REPO_ROOT
        / "outputs/paper_cvpr_v1/native_patch_category_initializers"
        / f"d1_seed{int(seed)}"
        / "checkpoint_d1_init.pth"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--b58", type=Path, default=DEFAULT_B58)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    config = args.config.resolve(strict=True)
    b58_path = args.b58.resolve(strict=True)
    output = args.output or _default_output(args.seed)
    if output.exists():
        raise FileExistsError(f"initializer output must be fresh: {output}")
    payload, model = build_payload(
        config=config, b58_path=b58_path, seed=args.seed
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, output)
    reloaded = _safe_load_checkpoint(
        output, label="published D1 initializer"
    )
    source_payload = _safe_load_checkpoint(
        b58_path, label="b58 source checkpoint"
    )
    source_state = extract_b58_source_state(source_payload)
    validate_native_patch_category_initializer_payload(
        model,
        reloaded,
        checkpoint_label=f"published D1 initializer {output}",
        expected_b58_state=source_state,
    )
    record = stable_file_record(output, label="published D1 initializer")
    print(
        f"[OK] wrote {record['path']} size={record['size_bytes']} "
        f"sha256={record['sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
