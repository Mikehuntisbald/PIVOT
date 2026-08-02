#!/usr/bin/env python3
"""Seal the causal U50 PairTop1 versus HardGap3 probe comparison."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.stageb_dependency_audit import config_import_chain  # noqa: E402


SCHEMA = "pivot.stageb.data_driven.pairtop1_hardgap3_probe_receipt/v1"
PROVENANCE_SCHEMA = "pivot.stageb.data_driven_training_provenance/v1"
PAIRTOP1_VARIANT = "DD1-PairTop1"
HARDGAP3_VARIANT = "DD1-PairTop1-HardGap3"
PAIRTOP1_ROLE = "pairtop1"
HARDGAP3_ROLE = "hardgap3"
EXPECTED_SCOPE = "official_assignment_full_321327_v1"
EXPECTED_PROBE_SCOPE = "full_official_assignment_u50_v1"
EXPECTED_ROWS = 321327
EXPECTED_VALID_ROWS = 274582
EXPECTED_UPDATES = 50
EXPECTED_MODEL_TENSOR_COUNT = 1190
EXPECTED_RANK_TRAINABLE_COUNT = 39
EXPECTED_PATCH_PARAMETER_COUNT = 9
EXPECTED_CODE_FILE_COUNT = 93
EXPECTED_CRITERION_VERSION = 4
EXPECTED_RANK_SUPERVISION_ID = 4
EXPECTED_RANK_SUPERVISION = "official_same_image_same_category_assignment_v1"
EXPECTED_CHECKPOINT_KEYS = frozenset(
    {
        "model",
        "criterion",
        "optimizer",
        "lr_scheduler",
        "scaler",
        "epoch",
        "iteration",
        "optimizer_updates",
        "epoch_finished",
        "rng_state",
        "epoch_rng_state",
        "args",
        "stage_b_data_driven_sampling_state",
        "checkpoint_reason",
    }
)
EXPECTED_INITIALIZER_SHA256 = (
    "54e1a5d6cf080114162b66cc96abbf92f25753c2829eaea6cffc3753fa42e5ef"
)
EXPECTED_INITIALIZER_PAIR_SHA256 = (
    "e304d2e8439f5714facf1b510795ba3a9874ec456433110afeef91f2d1dc7d8d"
)
EXPECTED_DATASET_CONFIG_SHA256 = (
    "5c659bb2de76f32d644af330b4284550c104ca24a96f9aeef6a004a698acf89a"
)
EXPECTED_ASSIGNMENT_RECEIPT_SHA256 = (
    "7b9ce1c911a2e1f0b67464243df8290fc2baf0786a2a3b131ddc57a6a6d2ddaa"
)
EXPECTED_MANIFEST_SHA256 = {
    "refcoco_stageb_phrase_v1.jsonl": (
        "f253c8bec4d15e421b11c42d8114e17c41bc32ed28f2614e34fe341e4da32592"
    ),
    "refcocoplus_stageb_phrase_v1.jsonl": (
        "69039abbd5baeb1173c849c19c55128aea8053271ab79f0cc16fa679000deaa8"
    ),
    "refcocog_stageb_phrase_v1.jsonl": (
        "378c5e34899e4113cd5dca1fd60352b362924e7969a717ea729b8278ce97a553"
    ),
}
EXPECTED_INITIALIZER = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/data_driven_initializers/fair_v2_seed42/"
    "checkpoint_dd_a1_relational_v2_init.pth"
)
EXPECTED_INITIALIZER_PAIR = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/data_driven_initializers/fair_v2_seed42/"
    "a0_a1_v2_pair_receipt.json"
)
EXPECTED_DATASET_CONFIG = (
    REPO_ROOT / "config/datasets_stageb_data_driven_dd1_official_assignment_three_ref.json"
)
EXPECTED_ASSIGNMENT_ROOT = (
    REPO_ROOT / "data/ablations/stageb_data_driven_assignment_pairs_20260722"
)
EXPECTED_ASSIGNMENT_RECEIPT = EXPECTED_ASSIGNMENT_ROOT / "receipt.json"
EXPECTED_CONFIGS = {
    PAIRTOP1_ROLE: (
        REPO_ROOT
        / "config/ablations/cfg_stageb_data_driven_dd1_pairtop1_fair_v2.py"
    ),
    HARDGAP3_ROLE: (
        REPO_ROOT
        / "config/ablations/"
        "cfg_stageb_data_driven_dd1_pairtop1_hardgap3_fair_v2_probe_u50.py"
    ),
}
RANK_PREFIX = "stage_b_data_driven_score_heads.rank_branch."
RANK_FIXED_BUFFER = f"{RANK_PREFIX}fourier_frequencies"
PATCH_PARAMETER_NAMES = (
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
EXPECTED_SAMPLING_STATE = {
    "schema": "deterministic_epoch_ledger_v1",
    "epoch": 0,
    "dataset_size": EXPECTED_ROWS,
    "num_samples": EXPECTED_ROWS,
    "weighted": True,
    "replacement": True,
    "sampler_seed": 42,
    "sampler_epoch_seed": 42,
    "loader_seed": 1042,
    "loader_epoch_seed": 1042,
    "persistent_workers": False,
    "weights_sha256": (
        "257c1ab2f46a4328641ee49356e046f84db87c1deeaae15b8de10678080fed6a"
    ),
    "ledger_sha256": (
        "6ac698c5d20bbfd4917f2b366f2088eb047a8905b4af2090d167c1c97a0a5ba9"
    ),
}

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


class PairTop1HardGap3ProbeAuditError(RuntimeError):
    """The two U50 probes do not satisfy the sealed causal contract."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise PairTop1HardGap3ProbeAuditError(
            f"receipt is not canonical JSON: {error}"
        ) from error
    return rendered.encode("ascii")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_file(path: Path, *, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise PairTop1HardGap3ProbeAuditError(
            f"{label} cannot be resolved: {path}: {error}"
        ) from error
    if not resolved.is_file():
        raise PairTop1HardGap3ProbeAuditError(f"{label} is not a file: {resolved}")
    return resolved


def stable_file_record(path: Path, *, label: str) -> dict[str, Any]:
    resolved = _resolve_file(path, label=label)
    before = resolved.stat()
    digest = _sha256_file(resolved)
    after = resolved.stat()
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_identity != after_identity:
        raise PairTop1HardGap3ProbeAuditError(
            f"{label} changed while it was hashed"
        )
    return {
        "path": str(resolved),
        "size_bytes": int(after.st_size),
        "sha256": digest,
    }


def _safe_load_checkpoint(path: Path, *, label: str) -> MutableMapping[str, Any]:
    resolved = _resolve_file(path, label=label)
    scanner = getattr(torch.serialization, "get_unsafe_globals_in_checkpoint", None)
    if scanner is None:
        raise PairTop1HardGap3ProbeAuditError(
            "installed PyTorch cannot statically inspect checkpoint globals"
        )
    try:
        observed_globals = frozenset(scanner(resolved))
    except Exception as error:
        raise PairTop1HardGap3ProbeAuditError(
            f"could not inspect globals in {label}: {error}"
        ) from error
    unexpected = sorted(observed_globals.difference(_ALLOWED_CHECKPOINT_GLOBALS))
    if unexpected:
        raise PairTop1HardGap3ProbeAuditError(
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
        raise PairTop1HardGap3ProbeAuditError(
            f"safe weights-only load failed for {label}: {error}"
        ) from error
    if not isinstance(payload, MutableMapping):
        raise PairTop1HardGap3ProbeAuditError(f"{label} payload is not a mapping")
    return payload


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PairTop1HardGap3ProbeAuditError(f"{label} must be a mapping")
    return value


def _strict_equal(observed: Any, expected: Any) -> bool:
    if isinstance(expected, bool):
        return isinstance(observed, bool) and observed is expected
    if isinstance(expected, int):
        return type(observed) is int and observed == expected
    if isinstance(expected, float):
        return type(observed) is float and observed == expected
    return type(observed) is type(expected) and observed == expected


def _require_exact_fields(
    value: Mapping[str, Any], expected: Mapping[str, Any], *, label: str
) -> None:
    drift = {
        key: {"expected": wanted, "observed": value.get(key)}
        for key, wanted in expected.items()
        if not _strict_equal(value.get(key), wanted)
    }
    if drift:
        raise PairTop1HardGap3ProbeAuditError(
            f"{label} exact contract drifted: {drift}"
        )


def _resolve_saved_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise PairTop1HardGap3ProbeAuditError(f"{label} is not a non-empty path")
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    if not path.is_absolute():
        path = REPO_ROOT / path
    return _resolve_file(path, label=label)


def _validate_saved_file_record(
    value: Any,
    *,
    label: str,
    expected_path: Path | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    record = _require_mapping(value, label=label)
    path = _resolve_saved_path(record.get("path"), label=f"{label}.path")
    observed = stable_file_record(path, label=label)
    if expected_path is not None and path != expected_path.resolve(strict=True):
        raise PairTop1HardGap3ProbeAuditError(f"{label} path drifted")
    if record.get("sha256") != observed["sha256"]:
        raise PairTop1HardGap3ProbeAuditError(f"{label} saved SHA drifted")
    if "size_bytes" in record and record.get("size_bytes") != observed["size_bytes"]:
        raise PairTop1HardGap3ProbeAuditError(f"{label} saved size drifted")
    if expected_sha256 is not None and observed["sha256"] != expected_sha256:
        raise PairTop1HardGap3ProbeAuditError(f"{label} canonical SHA drifted")
    return observed


def _validate_scalar_training_contract(args: Mapping[str, Any], *, role: str) -> None:
    if role not in {PAIRTOP1_ROLE, HARDGAP3_ROLE}:
        raise PairTop1HardGap3ProbeAuditError(f"unsupported probe role: {role!r}")
    variant = PAIRTOP1_VARIANT if role == PAIRTOP1_ROLE else HARDGAP3_VARIANT
    deployment_weight = 0.0 if role == PAIRTOP1_ROLE else 1.0
    expected = {
        "stage_b_data_driven_score": True,
        "stage_b_data_driven_experiment_id": "DD1",
        "stage_b_data_driven_variant_id": variant,
        "stage_b_data_driven_train_mode": "rank_patch_only",
        "stage_b_data_driven_category_complete": True,
        "stage_b_data_driven_confidence_trained": False,
        "stage_b_data_driven_rank_supervision": EXPECTED_RANK_SUPERVISION,
        "stage_b_data_driven_rank_weight": 0.0,
        "stage_b_data_driven_assignment_weight": 1.0,
        "stage_b_data_driven_deployment_weight": deployment_weight,
        "stage_b_data_driven_patch_weight": 1.0,
        "stage_b_data_driven_strict_sample_identity": True,
        "stage_b_data_driven_category_gate": False,
        "stage_b_data_driven_category_gate_max_gap": 3.0,
        "stage_b_data_driven_patch_score_clip": 5.0,
        "stage_b_data_driven_assignment_dataset_scope": EXPECTED_SCOPE,
        "stage_b_data_driven_assignment_expected_rows": EXPECTED_ROWS,
        "stage_b_data_driven_assignment_expected_valid_rows": EXPECTED_VALID_ROWS,
        "stage_b_data_driven_assignment_dataset_config_sha256": (
            EXPECTED_DATASET_CONFIG_SHA256
        ),
        "stage_b_data_driven_assignment_receipt_sha256": (
            EXPECTED_ASSIGNMENT_RECEIPT_SHA256
        ),
        "stage_b_data_driven_assignment_manifest_sha256": (
            EXPECTED_MANIFEST_SHA256
        ),
        "stage_b_data_driven_base_initializer_sha256": EXPECTED_INITIALIZER_SHA256,
        "stage_b_data_driven_initializer_pair_receipt_sha256": (
            EXPECTED_INITIALIZER_PAIR_SHA256
        ),
        "stage_b_data_driven_no_teacher_contract": (
            "b58_only_random_independent_heads_v1"
        ),
        "stage_b_gdino_score_adapter": False,
        "stage_b_u0_patch_rank": False,
        "stage_b_v7": False,
        "stage_b_v11_fixed_text": False,
        "stage_b_legacy_global_gate": False,
        "seed": 42,
        "start_epoch": 0,
        "resume": "",
        "max_train_iters": EXPECTED_UPDATES,
        "iter_checkpoint_interval": EXPECTED_UPDATES,
        "num_workers": 4,
        "prefetch_factor": 1,
        "persistent_workers": False,
        "gradient_accumulation_steps": 1,
        "amp": True,
        "amp_init_scale": 8192.0,
        "save_log": True,
        "world_size": 1,
        "distributed": False,
        "batch_size": 64,
        "epochs": 1,
        "fix_size": True,
        "strong_aug": False,
        "data_aug_hflip_prob": 0.0,
    }
    if role == HARDGAP3_ROLE:
        expected.update(
            {
                "stage_b_data_driven_probe_scope": EXPECTED_PROBE_SCOPE,
                "stage_b_data_driven_probe_fresh_start": True,
                "stage_b_data_driven_probe_expected_max_train_iters": 50,
                "stage_b_data_driven_probe_expected_iter_checkpoint_interval": 50,
                "stage_b_data_driven_probe_expected_num_workers": 4,
                "stage_b_data_driven_probe_expected_prefetch_factor": 1,
                "stage_b_data_driven_probe_expected_gradient_accumulation_steps": 1,
                "stage_b_data_driven_probe_expected_amp": True,
                "stage_b_data_driven_probe_expected_save_log": True,
                "stage_b_data_driven_probe_expected_world_size": 1,
                "stage_b_data_driven_probe_expected_distributed": False,
            }
        )
    _require_exact_fields(args, expected, label=f"{role} saved args")
    expected_config = str(EXPECTED_CONFIGS[role].relative_to(REPO_ROOT))
    if args.get("config_file") != expected_config:
        raise PairTop1HardGap3ProbeAuditError(
            f"{role} config_file drifted: {args.get('config_file')!r}"
        )
    expected_initializer = str(EXPECTED_INITIALIZER.relative_to(REPO_ROOT))
    if args.get("pretrain_model_path") != expected_initializer:
        raise PairTop1HardGap3ProbeAuditError(
            f"{role} pretrain_model_path drifted"
        )


def _validate_config_chain(args: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    expected_config = EXPECTED_CONFIGS[role].resolve(strict=True)
    saved_config = _resolve_saved_path(args.get("config_file"), label=f"{role} config")
    if saved_config != expected_config:
        raise PairTop1HardGap3ProbeAuditError(f"{role} config path drifted")
    chain = config_import_chain(expected_config, root=REPO_ROOT)
    observed = [
        stable_file_record(path, label=f"{role} config dependency {path.name}")
        for path in chain
    ]
    compact = [
        {"path": item["path"], "sha256": item["sha256"]} for item in observed
    ]
    if args.get("stage_b_data_driven_config_import_chain") != compact:
        raise PairTop1HardGap3ProbeAuditError(
            f"{role} saved config import chain drifted"
        )
    return {
        "leaf": stable_file_record(expected_config, label=f"{role} config leaf"),
        "import_chain": observed,
    }


def _validate_data_and_initializer_bindings(
    args: Mapping[str, Any], *, role: str
) -> dict[str, Any]:
    base = _validate_saved_file_record(
        args.get("stage_b_data_driven_base_initializer"),
        label=f"{role} base initializer",
        expected_path=EXPECTED_INITIALIZER,
        expected_sha256=EXPECTED_INITIALIZER_SHA256,
    )
    pair = _validate_saved_file_record(
        args.get("stage_b_data_driven_initializer_pair_receipt"),
        label=f"{role} initializer pair receipt",
        expected_path=EXPECTED_INITIALIZER_PAIR,
        expected_sha256=EXPECTED_INITIALIZER_PAIR_SHA256,
    )
    pair_saved = _require_mapping(
        args.get("stage_b_data_driven_initializer_pair_receipt"),
        label=f"{role} initializer pair receipt",
    )
    if (
        pair_saved.get("schema") != "pivot.stageb.data_driven_initializer_pair/v1"
        or pair_saved.get("common_tensor_sha256")
        != "39d092a4a80180e554e6854e6d55aab60decbd51d0011ffe824d85506b2852e5"
    ):
        raise PairTop1HardGap3ProbeAuditError(
            f"{role} initializer pair contract drifted"
        )
    binding = _require_mapping(
        args.get("stage_b_data_driven_assignment_dataset_binding"),
        label=f"{role} assignment dataset binding",
    )
    _require_exact_fields(
        binding,
        {
            "scope": EXPECTED_SCOPE,
            "rows": EXPECTED_ROWS,
            "valid_rows": EXPECTED_VALID_ROWS,
        },
        label=f"{role} assignment dataset binding",
    )
    dataset = _validate_saved_file_record(
        binding.get("dataset_config"),
        label=f"{role} assignment dataset config",
        expected_path=EXPECTED_DATASET_CONFIG,
        expected_sha256=EXPECTED_DATASET_CONFIG_SHA256,
    )
    if args.get("stage_b_data_driven_dataset_config") != {
        "path": dataset["path"],
        "sha256": dataset["sha256"],
    }:
        raise PairTop1HardGap3ProbeAuditError(
            f"{role} direct dataset-config binding drifted"
        )
    receipt = _validate_saved_file_record(
        binding.get("receipt"),
        label=f"{role} official assignment receipt",
        expected_path=EXPECTED_ASSIGNMENT_RECEIPT,
        expected_sha256=EXPECTED_ASSIGNMENT_RECEIPT_SHA256,
    )
    receipt_saved = _require_mapping(
        binding.get("receipt"), label=f"{role} official assignment receipt"
    )
    if (
        receipt_saved.get("schema")
        != "pivot.stageb.data_driven.official_assignment_pair_receipt/v1"
    ):
        raise PairTop1HardGap3ProbeAuditError(
            f"{role} official assignment receipt schema drifted"
        )
    manifest_values = binding.get("manifests")
    if not isinstance(manifest_values, list) or len(manifest_values) != len(
        EXPECTED_MANIFEST_SHA256
    ):
        raise PairTop1HardGap3ProbeAuditError(
            f"{role} assignment manifest coverage drifted"
        )
    manifests = []
    for index, (name, expected_sha) in enumerate(EXPECTED_MANIFEST_SHA256.items()):
        manifests.append(
            _validate_saved_file_record(
                manifest_values[index],
                label=f"{role} assignment manifest {name}",
                expected_path=EXPECTED_ASSIGNMENT_ROOT / name,
                expected_sha256=expected_sha,
            )
        )
    return {
        "scope": EXPECTED_SCOPE,
        "rows": EXPECTED_ROWS,
        "valid_rows": EXPECTED_VALID_ROWS,
        "dataset_config": dataset,
        "assignment_receipt": receipt,
        "manifests": manifests,
        "initializer": base,
        "initializer_pair_receipt": pair,
    }


def _deep_equal(left: Any, right: Any) -> bool:
    if torch.is_tensor(left) or torch.is_tensor(right):
        return _tensor_bitwise_equal(left, right)
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and np.array_equal(left, right)
        )
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_deep_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_deep_equal(a, b) for a, b in zip(left, right))
        )
    return type(left) is type(right) and left == right


def _tensor_bitwise_equal(left: Any, right: Any) -> bool:
    if not torch.is_tensor(left) or not torch.is_tensor(right):
        return False
    if (
        left.dtype != right.dtype
        or left.layout != right.layout
        or tuple(left.shape) != tuple(right.shape)
        or tuple(left.stride()) != tuple(right.stride())
        or left.layout != torch.strided
    ):
        return False
    left_bytes = left.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    right_bytes = right.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return bool(torch.equal(left_bytes, right_bytes))


def _audit_finite_tree(value: Any, *, label: str) -> int:
    if torch.is_tensor(value):
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all().item()
        ):
            raise PairTop1HardGap3ProbeAuditError(f"{label} has non-finite tensor")
        return 1
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.inexact) and not bool(np.isfinite(value).all()):
            raise PairTop1HardGap3ProbeAuditError(f"{label} has non-finite array")
        return 1
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PairTop1HardGap3ProbeAuditError(f"{label} has non-finite float")
        return 0
    if isinstance(value, Mapping):
        return sum(
            _audit_finite_tree(item, label=f"{label}.{key}")
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return sum(
            _audit_finite_tree(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    return 0


def _audit_terminal_checkpoint(payload: Mapping[str, Any], *, label: str) -> None:
    if set(payload) != EXPECTED_CHECKPOINT_KEYS:
        raise PairTop1HardGap3ProbeAuditError(
            f"{label} top-level checkpoint schema drifted"
        )
    required_mappings = ("model", "criterion", "optimizer", "scaler", "args")
    for key in required_mappings:
        _require_mapping(payload.get(key), label=f"{label}.{key}")
    _require_exact_fields(
        payload,
        {
            "epoch": 0,
            "iteration": EXPECTED_UPDATES,
            "optimizer_updates": EXPECTED_UPDATES,
            "epoch_finished": False,
            "checkpoint_reason": "max_train_iters",
        },
        label=f"{label} terminal checkpoint",
    )


def _audit_model_pair(
    pair_state: Mapping[str, Any],
    hard_state: Mapping[str, Any],
    *,
    expected_model_tensor_count: int = EXPECTED_MODEL_TENSOR_COUNT,
    expected_rank_trainable_count: int = EXPECTED_RANK_TRAINABLE_COUNT,
) -> dict[str, Any]:
    if set(pair_state) != set(hard_state):
        raise PairTop1HardGap3ProbeAuditError("model state key coverage differs")
    if len(pair_state) != expected_model_tensor_count:
        raise PairTop1HardGap3ProbeAuditError(
            "model tensor count drifted: "
            f"expected={expected_model_tensor_count}, observed={len(pair_state)}"
        )
    non_tensors = [
        key
        for key in pair_state
        if not torch.is_tensor(pair_state[key]) or not torch.is_tensor(hard_state[key])
    ]
    if non_tensors:
        raise PairTop1HardGap3ProbeAuditError(
            f"model state contains non-tensors: {non_tensors[:8]}"
        )
    rank_keys = sorted(key for key in pair_state if key.startswith(RANK_PREFIX))
    if RANK_FIXED_BUFFER not in rank_keys:
        raise PairTop1HardGap3ProbeAuditError("rank fixed Fourier buffer is missing")
    rank_trainable = [key for key in rank_keys if key != RANK_FIXED_BUFFER]
    if len(rank_trainable) != expected_rank_trainable_count:
        raise PairTop1HardGap3ProbeAuditError(
            "rank trainable tensor count drifted: "
            f"expected={expected_rank_trainable_count}, observed={len(rank_trainable)}"
        )
    non_rank = sorted(key for key in pair_state if not key.startswith(RANK_PREFIX))
    non_rank_drift = [
        key
        for key in non_rank
        if not _tensor_bitwise_equal(pair_state[key], hard_state[key])
    ]
    if non_rank_drift:
        raise PairTop1HardGap3ProbeAuditError(
            f"non-rank model tensors differ: {non_rank_drift[:8]}"
        )
    if not _tensor_bitwise_equal(
        pair_state[RANK_FIXED_BUFFER], hard_state[RANK_FIXED_BUFFER]
    ):
        raise PairTop1HardGap3ProbeAuditError("rank fixed Fourier buffer differs")
    differing_rank = [
        key
        for key in rank_trainable
        if not _tensor_bitwise_equal(pair_state[key], hard_state[key])
    ]
    if len(differing_rank) != len(rank_trainable):
        equal_rank = sorted(set(rank_trainable).difference(differing_rank))
        raise PairTop1HardGap3ProbeAuditError(
            f"not every rank trainable tensor changed: {equal_rank[:8]}"
        )
    return {
        "model_tensor_count": len(pair_state),
        "non_rank_model_tensor_count": len(non_rank),
        "rank_trainable_tensor_count": len(rank_trainable),
        "rank_trainable_tensors_changed": len(differing_rank),
        "rank_fixed_buffer_count": 1,
        "rank_trainable_names": rank_trainable,
    }


def _as_exact_step(value: Any, *, label: str) -> int:
    if torch.is_tensor(value):
        if value.numel() != 1 or not bool(torch.isfinite(value).all().item()):
            raise PairTop1HardGap3ProbeAuditError(f"{label} is not a finite scalar")
        numeric = float(value.detach().cpu().item())
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
    else:
        raise PairTop1HardGap3ProbeAuditError(f"{label} is not numeric")
    if not math.isfinite(numeric) or numeric != EXPECTED_UPDATES:
        raise PairTop1HardGap3ProbeAuditError(
            f"{label} must equal {EXPECTED_UPDATES}, got {numeric}"
        )
    return int(numeric)


def _optimizer_roles(
    optimizer: Mapping[str, Any],
    model_state: Mapping[str, Any],
    *,
    label: str,
    expected_rank_trainable_count: int = EXPECTED_RANK_TRAINABLE_COUNT,
) -> dict[str, Any]:
    groups = optimizer.get("param_groups")
    states = optimizer.get("state")
    if not isinstance(groups, list) or len(groups) != 2 or not isinstance(states, Mapping):
        raise PairTop1HardGap3ProbeAuditError(
            f"{label} optimizer must contain exactly two groups and a state mapping"
        )
    if [group.get("stage_b_data_driven_branch") for group in groups] != [
        "rank",
        "patch",
    ]:
        raise PairTop1HardGap3ProbeAuditError(
            f"{label} optimizer branch labels/order drifted"
        )
    rank_names = [
        key
        for key in model_state
        if key.startswith(RANK_PREFIX) and key != RANK_FIXED_BUFFER
    ]
    if len(rank_names) != expected_rank_trainable_count:
        raise PairTop1HardGap3ProbeAuditError(
            f"{label} rank parameter-name coverage drifted"
        )
    if (
        len(PATCH_PARAMETER_NAMES) != EXPECTED_PATCH_PARAMETER_COUNT
        or any(name not in model_state for name in PATCH_PARAMETER_NAMES)
    ):
        raise PairTop1HardGap3ProbeAuditError(
            f"{label} patch parameter-name coverage drifted"
        )
    names_by_role = {"rank": rank_names, "patch": list(PATCH_PARAMETER_NAMES)}
    ids_by_role: dict[str, list[Any]] = {}
    normalized_states: dict[str, dict[str, Any]] = {}
    all_ids = []
    for group in groups:
        role = group["stage_b_data_driven_branch"]
        parameter_ids = group.get("params")
        if not isinstance(parameter_ids, list) or len(parameter_ids) != len(
            names_by_role[role]
        ):
            raise PairTop1HardGap3ProbeAuditError(
                f"{label} {role} optimizer parameter coverage drifted"
            )
        if len(set(parameter_ids)) != len(parameter_ids):
            raise PairTop1HardGap3ProbeAuditError(
                f"{label} {role} optimizer contains duplicate parameter ids"
            )
        ids_by_role[role] = list(parameter_ids)
        all_ids.extend(parameter_ids)
        normalized_states[role] = {}
        for name, parameter_id in zip(names_by_role[role], parameter_ids):
            state = states.get(parameter_id)
            if not isinstance(state, Mapping) or set(state) != {
                "step",
                "exp_avg",
                "exp_avg_sq",
            }:
                raise PairTop1HardGap3ProbeAuditError(
                    f"{label} optimizer state schema drifted at {name}"
                )
            _as_exact_step(state["step"], label=f"{label}.{role}.{name}.step")
            for moment in ("exp_avg", "exp_avg_sq"):
                value = state[moment]
                if (
                    not torch.is_tensor(value)
                    or tuple(value.shape) != tuple(model_state[name].shape)
                ):
                    raise PairTop1HardGap3ProbeAuditError(
                        f"{label} {moment} shape drifted at {name}"
                    )
            _audit_finite_tree(state, label=f"{label}.{role}.{name}")
            normalized_states[role][name] = state
    if len(set(all_ids)) != len(all_ids) or set(states) != set(all_ids):
        raise PairTop1HardGap3ProbeAuditError(
            f"{label} optimizer states do not exactly cover both branches"
        )
    expected_lrs = {"rank": 3e-5, "patch": 3e-4}
    for group in groups:
        role = group["stage_b_data_driven_branch"]
        expected_static = {
            "lr": expected_lrs[role],
            "stage_b_data_driven_branch": role,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": 0.0001,
            "amsgrad": False,
            "maximize": False,
            "foreach": None,
            "capturable": False,
            "differentiable": False,
            "fused": None,
            "decoupled_weight_decay": True,
            "initial_lr": expected_lrs[role],
        }
        observed_static = {
            key: value for key, value in group.items() if key != "params"
        }
        if not _deep_equal(observed_static, expected_static):
            raise PairTop1HardGap3ProbeAuditError(
                f"{label} {role} optimizer static options drifted"
            )
    static_groups = [
        {key: value for key, value in group.items() if key != "params"}
        for group in groups
    ]
    return {
        "states": normalized_states,
        "static_groups": static_groups,
        "state_count": len(states),
    }


def _audit_optimizer_pair(
    pair_optimizer: Mapping[str, Any],
    hard_optimizer: Mapping[str, Any],
    pair_model_state: Mapping[str, Any],
    hard_model_state: Mapping[str, Any],
    *,
    expected_rank_trainable_count: int = EXPECTED_RANK_TRAINABLE_COUNT,
) -> dict[str, Any]:
    pair = _optimizer_roles(
        pair_optimizer,
        pair_model_state,
        label=PAIRTOP1_ROLE,
        expected_rank_trainable_count=expected_rank_trainable_count,
    )
    hard = _optimizer_roles(
        hard_optimizer,
        hard_model_state,
        label=HARDGAP3_ROLE,
        expected_rank_trainable_count=expected_rank_trainable_count,
    )
    if not _deep_equal(pair["static_groups"], hard["static_groups"]):
        raise PairTop1HardGap3ProbeAuditError("optimizer static group options differ")
    patch_drift = [
        name
        for name, state in pair["states"]["patch"].items()
        if not _deep_equal(state, hard["states"]["patch"].get(name))
    ]
    if patch_drift:
        raise PairTop1HardGap3ProbeAuditError(
            f"patch optimizer states differ: {patch_drift[:8]}"
        )
    rank_differences = [
        name
        for name, state in pair["states"]["rank"].items()
        if not _deep_equal(state, hard["states"]["rank"].get(name))
    ]
    if len(rank_differences) != expected_rank_trainable_count:
        raise PairTop1HardGap3ProbeAuditError(
            "not every rank optimizer state changed: "
            f"observed={len(rank_differences)}, expected={expected_rank_trainable_count}"
        )
    return {
        "optimizer_parameter_states": pair["state_count"],
        "rank_optimizer_state_count": expected_rank_trainable_count,
        "rank_optimizer_states_changed": len(rank_differences),
        "patch_optimizer_state_count": len(pair["states"]["patch"]),
    }


def _criterion_scalar(criterion: Mapping[str, Any], key: str, *, label: str) -> int:
    value = criterion.get(key)
    if (
        not torch.is_tensor(value)
        or value.numel() != 1
        or value.dtype != torch.int64
    ):
        raise PairTop1HardGap3ProbeAuditError(f"{label}.{key} tensor drifted")
    return int(value.detach().cpu().item())


def _audit_criterion_state(criterion: Mapping[str, Any], *, label: str) -> None:
    expected_keys = {
        "fpr_positive_queue",
        "fpr_positive_queue_count",
        "fpr_positive_queue_cursor",
        "criterion_contract_version",
        "rank_supervision_contract_id",
    }
    if set(criterion) != expected_keys:
        raise PairTop1HardGap3ProbeAuditError(f"{label} criterion schema drifted")
    queue = criterion["fpr_positive_queue"]
    if (
        not torch.is_tensor(queue)
        or queue.dtype != torch.float32
        or tuple(queue.shape) != (4096,)
        or bool(torch.count_nonzero(queue).item())
    ):
        raise PairTop1HardGap3ProbeAuditError(f"{label} criterion queue drifted")
    expected_scalars = {
        "fpr_positive_queue_count": 0,
        "fpr_positive_queue_cursor": 0,
        "criterion_contract_version": EXPECTED_CRITERION_VERSION,
        "rank_supervision_contract_id": EXPECTED_RANK_SUPERVISION_ID,
    }
    for key, expected in expected_scalars.items():
        if _criterion_scalar(criterion, key, label=label) != expected:
            raise PairTop1HardGap3ProbeAuditError(
                f"{label}.{key} must equal {expected}"
            )


def _audit_scaler_state(scaler: Mapping[str, Any], *, label: str) -> None:
    expected = {
        "scale": 8192.0,
        "growth_factor": 2.0,
        "backoff_factor": 0.5,
        "growth_interval": 2000,
        "_growth_tracker": EXPECTED_UPDATES,
    }
    if not _deep_equal(scaler, expected):
        raise PairTop1HardGap3ProbeAuditError(
            f"{label} scaler contract or zero-skip evidence drifted"
        )


def _validate_provenance_pair(
    pair_args: Mapping[str, Any], hard_args: Mapping[str, Any]
) -> dict[str, Any]:
    pair = _require_mapping(
        pair_args.get("stage_b_data_driven_training_provenance"),
        label="PairTop1 training provenance",
    )
    hard = _require_mapping(
        hard_args.get("stage_b_data_driven_training_provenance"),
        label="HardGap3 training provenance",
    )
    if not _deep_equal(pair, hard):
        raise PairTop1HardGap3ProbeAuditError(
            "saved training provenance differs across the causal pair"
        )
    if pair.get("schema") != PROVENANCE_SCHEMA:
        raise PairTop1HardGap3ProbeAuditError("training provenance schema drifted")
    code_files = pair.get("code_files")
    if not isinstance(code_files, list) or len(code_files) != EXPECTED_CODE_FILE_COUNT:
        raise PairTop1HardGap3ProbeAuditError(
            "training code file coverage drifted: "
            f"expected={EXPECTED_CODE_FILE_COUNT}, "
            f"observed={len(code_files) if isinstance(code_files, list) else None}"
        )
    verified_code = []
    for index, value in enumerate(code_files):
        record = _require_mapping(value, label=f"training code file {index}")
        if (
            not isinstance(record.get("path"), str)
            or not Path(record["path"]).is_absolute()
            or type(record.get("size_bytes")) is not int
            or record["size_bytes"] <= 0
            or not isinstance(record.get("sha256"), str)
            or len(record["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in record["sha256"])
        ):
            raise PairTop1HardGap3ProbeAuditError(
                f"training code file {index} record is malformed"
            )
        verified_code.append(dict(record))
    if len({record["path"] for record in verified_code}) != len(verified_code):
        raise PairTop1HardGap3ProbeAuditError("training code manifest has duplicates")
    dataset_assets = pair.get("dataset_asset_files")
    if not isinstance(dataset_assets, list) or not dataset_assets:
        raise PairTop1HardGap3ProbeAuditError(
            "training provenance has no dataset asset files"
        )
    for index, value in enumerate(dataset_assets):
        record = _require_mapping(value, label=f"dataset asset file {index}")
        if (
            not isinstance(record.get("path"), str)
            or not Path(record["path"]).is_absolute()
            or type(record.get("size_bytes")) is not int
            or record["size_bytes"] <= 0
            or not isinstance(record.get("sha256"), str)
            or len(record["sha256"]) != 64
        ):
            raise PairTop1HardGap3ProbeAuditError(
                f"dataset asset file {index} record is malformed"
            )
    return {
        "training_code_file_count": len(code_files),
        "training_code_files_manifest_sha256": canonical_json_sha256(code_files),
        "saved_training_provenance_bitwise_equal": True,
    }


def build_receipt(
    *, pairtop1_checkpoint: Path, hardgap3_checkpoint: Path
) -> dict[str, Any]:
    pair_record = stable_file_record(pairtop1_checkpoint, label="PairTop1 U50 checkpoint")
    hard_record = stable_file_record(hardgap3_checkpoint, label="HardGap3 U50 checkpoint")
    if pair_record["sha256"] == hard_record["sha256"]:
        raise PairTop1HardGap3ProbeAuditError("the two probe checkpoints are identical")
    pair_payload = _safe_load_checkpoint(pairtop1_checkpoint, label="PairTop1 U50 checkpoint")
    hard_payload = _safe_load_checkpoint(hardgap3_checkpoint, label="HardGap3 U50 checkpoint")
    _audit_terminal_checkpoint(pair_payload, label=PAIRTOP1_ROLE)
    _audit_terminal_checkpoint(hard_payload, label=HARDGAP3_ROLE)
    pair_args = _require_mapping(pair_payload["args"], label="PairTop1 saved args")
    hard_args = _require_mapping(hard_payload["args"], label="HardGap3 saved args")
    _validate_scalar_training_contract(pair_args, role=PAIRTOP1_ROLE)
    _validate_scalar_training_contract(hard_args, role=HARDGAP3_ROLE)
    pair_config = _validate_config_chain(pair_args, role=PAIRTOP1_ROLE)
    hard_config = _validate_config_chain(hard_args, role=HARDGAP3_ROLE)
    pair_bindings = _validate_data_and_initializer_bindings(
        pair_args, role=PAIRTOP1_ROLE
    )
    hard_bindings = _validate_data_and_initializer_bindings(
        hard_args, role=HARDGAP3_ROLE
    )
    if not _deep_equal(pair_bindings, hard_bindings):
        raise PairTop1HardGap3ProbeAuditError(
            "data/initializer bindings differ across the causal pair"
        )
    provenance = _validate_provenance_pair(pair_args, hard_args)

    pair_model = _require_mapping(pair_payload["model"], label="PairTop1 model state")
    hard_model = _require_mapping(hard_payload["model"], label="HardGap3 model state")
    pair_criterion = _require_mapping(
        pair_payload["criterion"], label="PairTop1 criterion state"
    )
    hard_criterion = _require_mapping(
        hard_payload["criterion"], label="HardGap3 criterion state"
    )
    pair_optimizer = _require_mapping(
        pair_payload["optimizer"], label="PairTop1 optimizer state"
    )
    hard_optimizer = _require_mapping(
        hard_payload["optimizer"], label="HardGap3 optimizer state"
    )
    pair_scaler = _require_mapping(
        pair_payload["scaler"], label="PairTop1 scaler state"
    )
    hard_scaler = _require_mapping(
        hard_payload["scaler"], label="HardGap3 scaler state"
    )
    numerical_counts = {
        "pairtop1_model_tensors": _audit_finite_tree(
            pair_model, label="PairTop1 model"
        ),
        "hardgap3_model_tensors": _audit_finite_tree(
            hard_model, label="HardGap3 model"
        ),
        "pairtop1_optimizer_tensors": _audit_finite_tree(
            pair_optimizer, label="PairTop1 optimizer"
        ),
        "hardgap3_optimizer_tensors": _audit_finite_tree(
            hard_optimizer, label="HardGap3 optimizer"
        ),
        "pairtop1_criterion_tensors": _audit_finite_tree(
            pair_criterion, label="PairTop1 criterion"
        ),
        "hardgap3_criterion_tensors": _audit_finite_tree(
            hard_criterion, label="HardGap3 criterion"
        ),
        "pairtop1_scaler_tensors": _audit_finite_tree(
            pair_scaler, label="PairTop1 scaler"
        ),
        "hardgap3_scaler_tensors": _audit_finite_tree(
            hard_scaler, label="HardGap3 scaler"
        ),
    }
    _audit_criterion_state(pair_criterion, label=PAIRTOP1_ROLE)
    _audit_criterion_state(hard_criterion, label=HARDGAP3_ROLE)
    _audit_scaler_state(pair_scaler, label=PAIRTOP1_ROLE)
    _audit_scaler_state(hard_scaler, label=HARDGAP3_ROLE)
    if not _deep_equal(pair_criterion, hard_criterion):
        raise PairTop1HardGap3ProbeAuditError(
            "criterion persistent state differs across the causal pair"
        )
    model_audit = _audit_model_pair(pair_model, hard_model)
    optimizer_audit = _audit_optimizer_pair(
        pair_optimizer, hard_optimizer, pair_model, hard_model
    )
    pair_sampling = _require_mapping(
        pair_payload.get("stage_b_data_driven_sampling_state"),
        label="PairTop1 sampling state",
    )
    hard_sampling = _require_mapping(
        hard_payload.get("stage_b_data_driven_sampling_state"),
        label="HardGap3 sampling state",
    )
    if not _deep_equal(pair_sampling, EXPECTED_SAMPLING_STATE):
        raise PairTop1HardGap3ProbeAuditError("PairTop1 sampling contract drifted")
    if not _deep_equal(pair_sampling, hard_sampling):
        raise PairTop1HardGap3ProbeAuditError("sampling state differs across probes")
    equal_state_fields = ("lr_scheduler", "scaler", "rng_state", "epoch_rng_state")
    unequal_state_fields = [
        key
        for key in equal_state_fields
        if not _deep_equal(pair_payload.get(key), hard_payload.get(key))
    ]
    if unequal_state_fields:
        raise PairTop1HardGap3ProbeAuditError(
            f"scheduler/scaler/RNG state differs: {unequal_state_fields}"
        )

    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "passed",
        "scope": "memory_and_protocol_probe_only_do_not_resume_into_formal",
        "checkpoint": {
            "optimizer_updates": EXPECTED_UPDATES,
            "iteration": EXPECTED_UPDATES,
            "checkpoint_reason": "max_train_iters",
            "epoch": 0,
            "epoch_finished": False,
            "pairtop1": pair_record,
            "hardgap3": hard_record,
        },
        "configs": {
            "pairtop1": pair_config,
            "hardgap3": hard_config,
        },
        "criterion": {
            "criterion_contract_version": EXPECTED_CRITERION_VERSION,
            "rank_supervision_contract_id": EXPECTED_RANK_SUPERVISION_ID,
            "rank_supervision": EXPECTED_RANK_SUPERVISION,
            "assignment_weight": 1.0,
            "pairtop1_deployment_weight": 0.0,
            "deployment_weight": 1.0,
        },
        "data": {
            key: pair_bindings[key]
            for key in (
                "scope",
                "rows",
                "valid_rows",
                "dataset_config",
                "assignment_receipt",
                "manifests",
            )
        },
        "fresh_start": {
            "resume": "",
            "pretrain_model": pair_bindings["initializer"],
            "initializer_pair_receipt": pair_bindings[
                "initializer_pair_receipt"
            ],
            "seed": 42,
        },
        "source": provenance,
        "sampling": dict(pair_sampling),
        "numerical_audit": numerical_counts,
        "causal_audit": {
            **model_audit,
            **optimizer_audit,
            "all_non_rank_model_tensors_bitwise_equal_to_pairtop1": True,
            "patch_optimizer_state_bitwise_equal_to_pairtop1": True,
            "rank_trainable_tensors_differ_from_pairtop1": True,
            "rank_optimizer_states_differ_from_pairtop1": True,
            "criterion_scheduler_scaler_rng_and_sampling_state_equal": True,
        },
        "invariants": {
            "fifty_of_fifty_optimizer_updates_succeeded": True,
            "amp_step_skips_zero": True,
            "all_model_and_optimizer_tensors_are_finite": True,
            "all_criterion_and_scaler_state_is_finite": True,
            "official_assignment_full_data_and_initializer_are_identical": True,
            "training_code_manifest_is_identical": True,
            "only_rank_trainable_model_and_optimizer_state_changed": True,
            "no_teacher_logits_weights_or_loss_targets_are_used": True,
            "probe_checkpoint_is_forbidden_as_formal_resume_source": True,
        },
        "decision": (
            "pass_for_fresh_hardgap3_formal_u5020; this U50 receipt establishes "
            "protocol stability and causal isolation, not RefCOCO accuracy"
        ),
    }
    receipt["receipt_sha256"] = canonical_json_sha256(receipt)
    return receipt


def _atomic_publish_fresh_json(path: Path, payload: Mapping[str, Any]) -> None:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"probe receipt output must be fresh: {output}")
    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"
    temporary = output.parent / f".{output.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    descriptor: int | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _rename_noreplace(temporary, output)
        directory_fd = os.open(output.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish without replacing an existing destination on Linux."""
    at_fdcwd = -100
    rename_noreplace = 1
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        try:
            os.link(source, destination)
        except FileExistsError:
            raise
        source.unlink()
        return
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(destination),
        rename_noreplace,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error in (errno.ENOSYS, errno.EINVAL):
        try:
            os.link(source, destination)
        except FileExistsError:
            raise
        source.unlink()
        return
    raise OSError(error, os.strerror(error), destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairtop1-checkpoint", type=Path, required=True)
    parser.add_argument("--hardgap3-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_receipt(
        pairtop1_checkpoint=args.pairtop1_checkpoint.resolve(strict=True),
        hardgap3_checkpoint=args.hardgap3_checkpoint.resolve(strict=True),
    )
    _atomic_publish_fresh_json(args.output, receipt)
    output_record = stable_file_record(args.output, label="HardGap3 probe receipt")
    print(
        f"[OK] wrote {output_record['path']} size={output_record['size_bytes']} "
        f"sha256={output_record['sha256']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
