#!/usr/bin/env python3
"""Seal the reproducibility evidence for the formal U0 B56 training run.

The builder is intentionally fail closed.  It parses every declared training
data row, safely deserializes checkpoints with ``weights_only=True``, replays
the U0 transition audits, and publishes a self-hashed JSON receipt atomically
without replacing an existing file.
"""

from __future__ import annotations

import argparse
import ctypes
import csv
import errno
import gc
import hashlib
import json
import math
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.GroundingDINO.stage_b_u0_patch_rank import (  # noqa: E402
    U0_PATCH_SOURCE_KEYS,
    stage_b_u0_tensor_state_sha256,
    validate_stage_b_u0_initializer_payload,
)
from tools.audit_stageb_u0_transition import (  # noqa: E402
    FROZEN_PATCH_KEYS,
    U0_PREFIX,
    audit_u0_transition,
)
from tools.stageb_dependency_audit import (  # noqa: E402
    DependencyAuditError,
    config_import_chain,
)
from util.path_compat import default_data_root, remap_legacy_path  # noqa: E402


SCHEMA = "pivot.stageb.u0_training_receipt/v1"
TRANSITION_SCHEMA = "pivot.stageb.u0_transition_audit/v1"
EXPECTED_BATCH_SIZE = 56
EXPECTED_SEED = 42
MILESTONES = (("u50", 50), ("u100", 100))

CORE_SOURCE_PATHS = (
    "main.py",
    "engine.py",
    "datasets/__init__.py",
    "datasets/patch_episode.py",
    "models/__init__.py",
    "models/GroundingDINO/groundingdino.py",
    "models/GroundingDINO/stage_b_gdino_score_adapter.py",
    "models/GroundingDINO/stage_b_u0_patch_rank.py",
    "tools/audit_stageb_u0_transition.py",
    "tools/build_stageb_u0_initializer.py",
    "tools/build_stageb_u0_training_receipt.py",
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


class U0TrainingReceiptError(RuntimeError):
    """The requested U0 evidence does not satisfy the sealed-run contract."""


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        rendered = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise U0TrainingReceiptError(
            f"receipt value is not canonical JSON: {error}"
        ) from error
    return rendered.encode("ascii")


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _resolve_file(path: Path, *, label: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise U0TrainingReceiptError(
            f"{label} cannot be resolved: {path}: {error}"
        ) from error
    if not resolved.is_file():
        raise U0TrainingReceiptError(f"{label} is not a file: {resolved}")
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_file_record(path: Path, *, label: str) -> dict[str, Any]:
    resolved = _resolve_file(path, label=label)
    try:
        before = resolved.stat()
        digest = _sha256_file(resolved)
        after = resolved.stat()
    except OSError as error:
        raise U0TrainingReceiptError(f"could not hash {label}: {error}") from error
    if _stat_identity(before) != _stat_identity(after):
        raise U0TrainingReceiptError(f"{label} changed while it was hashed")
    return {
        "path": str(resolved),
        "size_bytes": int(after.st_size),
        "sha256": digest,
    }


def _strict_json_load(path: Path, *, label: str) -> Any:
    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise U0TrainingReceiptError(f"could not parse {label}: {error}") from error


def _json_safe(value: Any, *, label: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise U0TrainingReceiptError(f"{label} contains a non-finite float")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise U0TrainingReceiptError(
                    f"{label} contains non-string mapping key {key!r}"
                )
            result[key] = _json_safe(item, label=f"{label}.{key}")
        return result
    raise U0TrainingReceiptError(
        f"{label} contains unsupported {type(value).__name__}"
    )


def _safe_load_checkpoint(path: Path, *, label: str) -> MutableMapping[str, Any]:
    resolved = _resolve_file(path, label=label)
    scanner = getattr(torch.serialization, "get_unsafe_globals_in_checkpoint", None)
    if scanner is None:
        raise U0TrainingReceiptError(
            "installed PyTorch cannot statically inspect checkpoint globals"
        )
    try:
        observed_globals = frozenset(scanner(resolved))
    except Exception as error:
        raise U0TrainingReceiptError(
            f"could not inspect globals in {label}: {error}"
        ) from error
    unexpected = sorted(observed_globals.difference(_ALLOWED_CHECKPOINT_GLOBALS))
    if unexpected:
        raise U0TrainingReceiptError(
            f"{label} requires unsafe or unknown pickle globals: {unexpected}"
        )
    try:
        with torch.serialization.safe_globals(list(_NUMPY_SAFE_GLOBALS)):
            value = torch.load(
                resolved,
                map_location="cpu",
                mmap=True,
                weights_only=True,
            )
    except Exception as error:
        raise U0TrainingReceiptError(
            f"safe weights-only load failed for {label}: {error}"
        ) from error
    if not isinstance(value, MutableMapping):
        raise U0TrainingReceiptError(f"{label} payload is not a mapping")
    return value


def _resolve_runtime_path(
    value: Any,
    *,
    label: str,
    data_root: Path,
) -> Path:
    if not isinstance(value, (str, os.PathLike)) or not os.fspath(value):
        raise U0TrainingReceiptError(f"{label} is not a non-empty path")
    raw = os.fspath(value)
    expanded = raw.replace("${DATA_ROOT}", str(data_root)).replace(
        "$DATA_ROOT", str(data_root)
    )
    expanded = os.path.expandvars(os.path.expanduser(expanded))
    if "$" in expanded:
        raise U0TrainingReceiptError(f"{label} contains an unresolved variable: {raw}")
    path = remap_legacy_path(
        expanded,
        repo_root=REPO_ROOT,
        data_root=data_root,
    )
    if not path.is_absolute():
        path = REPO_ROOT / path
    return _resolve_file(path, label=label)


def _same_path(
    value: Any,
    expected: Path,
    *,
    label: str,
    data_root: Path,
) -> None:
    if not isinstance(value, (str, os.PathLike)) or not os.fspath(value):
        raise U0TrainingReceiptError(f"{label} is not a non-empty path")
    raw = os.fspath(value)
    expanded = raw.replace("${DATA_ROOT}", str(data_root)).replace(
        "$DATA_ROOT", str(data_root)
    )
    expanded = os.path.expandvars(os.path.expanduser(expanded))
    if "$" in expanded:
        raise U0TrainingReceiptError(f"{label} contains an unresolved variable: {raw}")
    observed = remap_legacy_path(
        expanded,
        repo_root=REPO_ROOT,
        data_root=data_root,
    )
    if not observed.is_absolute():
        observed = REPO_ROOT / observed
    try:
        observed = observed.resolve(strict=True)
    except OSError as error:
        raise U0TrainingReceiptError(
            f"{label} cannot be resolved: {observed}: {error}"
        ) from error
    if observed != expected:
        raise U0TrainingReceiptError(
            f"{label} lineage mismatch: expected {expected}, got {observed}"
        )


def _inspect_jsonl(path: Path, *, label: str) -> dict[str, Any]:
    record = stable_file_record(path, label=label)
    resolved = Path(record["path"])
    before = resolved.stat()
    rows = 0
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, start=1):
                if not raw.strip():
                    continue
                try:
                    value = json.loads(
                        raw,
                        parse_constant=lambda token: (_ for _ in ()).throw(
                            ValueError(f"non-finite JSON constant {token}")
                        ),
                    )
                except (json.JSONDecodeError, ValueError) as error:
                    raise U0TrainingReceiptError(
                        f"{label} line {line_number} is invalid JSON: {error}"
                    ) from error
                if not isinstance(value, Mapping):
                    raise U0TrainingReceiptError(
                        f"{label} line {line_number} is not a JSON object"
                    )
                rows += 1
    except (OSError, UnicodeError) as error:
        raise U0TrainingReceiptError(f"could not parse {label}: {error}") from error
    if rows <= 0:
        raise U0TrainingReceiptError(f"{label} contains no JSON rows")
    if _stat_identity(before) != _stat_identity(resolved.stat()):
        raise U0TrainingReceiptError(f"{label} changed while it was parsed")
    return {**record, "parsed_rows": rows, "all_rows_json_objects": True}


def _inspect_tsv(path: Path, *, label: str) -> dict[str, Any]:
    record = stable_file_record(path, label=label)
    resolved = Path(record["path"])
    before = resolved.stat()
    rows = 0
    bucket_counts: dict[str, int] = {}
    try:
        with resolved.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t", strict=True)
            header = next(reader, None)
            if not header or any(not value for value in header):
                raise U0TrainingReceiptError(f"{label} has an empty TSV header")
            if len(set(header)) != len(header):
                raise U0TrainingReceiptError(f"{label} has duplicate TSV columns")
            required = {"path", "class", "bucket"}
            if not required.issubset(header):
                raise U0TrainingReceiptError(
                    f"{label} is missing TSV columns {sorted(required - set(header))}"
                )
            bucket_index = header.index("bucket")
            path_index = header.index("path")
            class_index = header.index("class")
            for line_number, row in enumerate(reader, start=2):
                if len(row) != len(header):
                    raise U0TrainingReceiptError(
                        f"{label} line {line_number} has {len(row)} columns; "
                        f"expected {len(header)}"
                    )
                if not row[path_index] or not row[class_index] or not row[bucket_index]:
                    raise U0TrainingReceiptError(
                        f"{label} line {line_number} has an empty required field"
                    )
                bucket_counts[row[bucket_index]] = bucket_counts.get(row[bucket_index], 0) + 1
                rows += 1
    except (OSError, UnicodeError, csv.Error) as error:
        raise U0TrainingReceiptError(f"could not parse {label}: {error}") from error
    if rows <= 0:
        raise U0TrainingReceiptError(f"{label} contains no TSV rows")
    if _stat_identity(before) != _stat_identity(resolved.stat()):
        raise U0TrainingReceiptError(f"{label} changed while it was parsed")
    return {
        **record,
        "header": header,
        "parsed_rows": rows,
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "all_rows_rectangular": True,
    }


def _inspect_canonical_json(path: Path, *, label: str) -> dict[str, Any]:
    record = stable_file_record(path, label=label)
    resolved = Path(record["path"])
    before = resolved.stat()
    value = _strict_json_load(resolved, label=label)
    if not isinstance(value, list) or not value:
        raise U0TrainingReceiptError(f"{label} must be a non-empty JSON list")
    ids = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise U0TrainingReceiptError(f"{label}[{index}] is not an object")
        identifier = item.get("id")
        if isinstance(identifier, bool) or not isinstance(identifier, int):
            raise U0TrainingReceiptError(f"{label}[{index}].id is not an integer")
        ids.append(identifier)
    if len(set(ids)) != len(ids):
        raise U0TrainingReceiptError(f"{label} contains duplicate canonical ids")
    if _stat_identity(before) != _stat_identity(resolved.stat()):
        raise U0TrainingReceiptError(f"{label} changed while it was parsed")
    return {
        **record,
        "parsed_entries": len(value),
        "unique_ids": len(ids),
        "id_min": min(ids),
        "id_max": max(ids),
    }


def _dataset_binding(dataset_json: Path, *, data_root: Path) -> dict[str, Any]:
    dataset_record = stable_file_record(dataset_json, label="dataset JSON")
    dataset_path = Path(dataset_record["path"])
    value = _strict_json_load(dataset_path, label="dataset JSON")
    if not isinstance(value, Mapping) or set(value) != {"train", "val"}:
        raise U0TrainingReceiptError("dataset JSON must contain exactly train and val")
    train = value.get("train")
    if not isinstance(train, list) or len(train) != 3:
        raise U0TrainingReceiptError("U0 dataset JSON must declare exactly three train entries")
    if value.get("val") != []:
        raise U0TrainingReceiptError("formal U0 training dataset must have an empty val split")

    entries = []
    annotation_paths: set[Path] = set()
    patch_paths: set[Path] = set()
    canonical_paths: set[Path] = set()
    for index, item in enumerate(train):
        label = f"dataset train[{index}]"
        if not isinstance(item, Mapping):
            raise U0TrainingReceiptError(f"{label} is not an object")
        if item.get("dataset_mode") != "patch_episode":
            raise U0TrainingReceiptError(f"{label} is not patch_episode")
        if item.get("neg_episode_prob") != 0.0:
            raise U0TrainingReceiptError(f"{label} is not positive-only")
        if item.get("lvis_neg_category_only") is not False:
            raise U0TrainingReceiptError(f"{label} lvis_neg_category_only drifted")
        if item.get("mix_weight") != 2.0:
            raise U0TrainingReceiptError(f"{label} mix_weight is not 2.0")
        annotation = _resolve_runtime_path(
            item.get("anno"), label=f"{label} annotation", data_root=data_root
        )
        patch_tsv = _resolve_runtime_path(
            item.get("support_patch_tsv"),
            label=f"{label} support patch TSV",
            data_root=data_root,
        )
        canonical = _resolve_runtime_path(
            item.get("canonical_classes_json"),
            label=f"{label} canonical JSON",
            data_root=data_root,
        )
        if annotation in annotation_paths:
            raise U0TrainingReceiptError("U0 dataset repeats an annotation JSONL")
        annotation_paths.add(annotation)
        patch_paths.add(patch_tsv)
        canonical_paths.add(canonical)
        entries.append(
            {
                "index": index,
                "dataset_mode": "patch_episode",
                "mix_weight": 2.0,
                "support_patch_bucket": item.get("support_patch_bucket"),
                "keep_only_support_gt": item.get("keep_only_support_gt"),
                "annotation": _inspect_jsonl(
                    annotation, label=f"{label} annotation"
                ),
            }
        )
    if len(patch_paths) != 1 or len(canonical_paths) != 1:
        raise U0TrainingReceiptError(
            "all three U0 datasets must share one patch TSV and canonical JSON"
        )
    return {
        "file": dataset_record,
        "train_entry_count": 3,
        "val_entry_count": 0,
        "train": entries,
        "support_patch_tsv": _inspect_tsv(
            next(iter(patch_paths)), label="shared support patch TSV"
        ),
        "canonical_classes_json": _inspect_canonical_json(
            next(iter(canonical_paths)), label="shared canonical JSON"
        ),
    }


def _config_binding(config: Path) -> dict[str, Any]:
    resolved = _resolve_file(config, label="U0 config")
    try:
        chain = config_import_chain(resolved, root=REPO_ROOT)
    except DependencyAuditError as error:
        raise U0TrainingReceiptError(f"could not inspect config imports: {error}") from error
    if resolved not in chain:
        raise U0TrainingReceiptError("U0 config is absent from its import chain")
    return {
        "leaf": stable_file_record(resolved, label="U0 config"),
        "import_chain": [
            stable_file_record(path, label=f"config dependency {path}")
            for path in chain
        ],
    }


def _source_binding() -> list[dict[str, Any]]:
    records = []
    for relative in CORE_SOURCE_PATHS:
        path = REPO_ROOT / relative
        record = stable_file_record(path, label=f"core source {relative}")
        records.append({"relative_path": relative, **record})
    return records


def _checkpoint_args(payload: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    value = payload.get("args")
    if isinstance(value, Mapping):
        result = dict(value)
    elif value is not None and hasattr(value, "__dict__"):
        result = dict(vars(value))
    else:
        raise U0TrainingReceiptError(f"{label} has no training args mapping")
    return _json_safe(result, label=f"{label}.args")


def _strict_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise U0TrainingReceiptError(f"{label} is not an integer")
    return value


def _training_record(
    payload: Mapping[str, Any],
    *,
    label: str,
    target: int,
    checkpoint: Path,
    initializer: Path,
    u50: Path,
    config: Path,
    datasets: Path,
    experiment_root: Path,
    data_root: Path,
) -> dict[str, Any]:
    iteration = _strict_int(payload.get("iteration"), label=f"{label}.iteration")
    updates = _strict_int(
        payload.get("optimizer_updates"), label=f"{label}.optimizer_updates"
    )
    if iteration != target or updates != target:
        raise U0TrainingReceiptError(
            f"{label} must be iteration/update {target}, got {iteration}/{updates}"
        )
    if payload.get("checkpoint_reason") != "max_train_iters":
        raise U0TrainingReceiptError(f"{label} was not sealed by max_train_iters")
    args = _checkpoint_args(payload, label=label)
    for field, expected in (
        ("batch_size", EXPECTED_BATCH_SIZE),
        ("seed", EXPECTED_SEED),
        ("max_train_iters", target),
        ("gradient_accumulation_steps", 1),
        ("world_size", 1),
    ):
        if args.get(field) != expected or isinstance(args.get(field), bool):
            raise U0TrainingReceiptError(
                f"{label} args.{field} must be {expected!r}, got {args.get(field)!r}"
            )
    for field in ("amp", "stage_b_u0_patch_rank", "enable_patch_branch"):
        if args.get(field) is not True:
            raise U0TrainingReceiptError(f"{label} args.{field} must be true")
    if args.get("stage_b_gdino_score_adapter") is not True or (
        args.get("stage_b_gdino_adapter_train_mode") != "rank_only"
    ):
        raise U0TrainingReceiptError(f"{label} sealed teacher/U0 mode drifted")
    if args.get("distributed") is not False:
        raise U0TrainingReceiptError(f"{label} must be a single-process run")
    if not isinstance(payload.get("optimizer"), Mapping):
        raise U0TrainingReceiptError(f"{label} has no resumable optimizer state")
    if not isinstance(payload.get("scaler"), Mapping):
        raise U0TrainingReceiptError(f"{label} AMP scaler state is missing")
    _same_path(
        args.get("config_file"), config, label=f"{label} config", data_root=data_root
    )
    _same_path(
        args.get("datasets"), datasets, label=f"{label} datasets", data_root=data_root
    )
    _same_path(
        args.get("output_dir"),
        experiment_root,
        label=f"{label} output_dir",
        data_root=data_root,
    )
    if target == 50:
        if args.get("resume") not in (None, ""):
            raise U0TrainingReceiptError("U50 must start without a resume checkpoint")
        _same_path(
            args.get("pretrain_model_path"),
            initializer,
            label="U50 initializer",
            data_root=data_root,
        )
    else:
        _same_path(
            args.get("resume"), u50, label="U100 resume", data_root=data_root
        )
        if args.get("pretrain_model_path") not in (None, ""):
            raise U0TrainingReceiptError("U100 must resume U50 without another pretrain")
    model = payload.get("model")
    if not isinstance(model, Mapping) or not model:
        raise U0TrainingReceiptError(f"{label} has no model state")
    if any(not isinstance(key, str) or not torch.is_tensor(value) for key, value in model.items()):
        raise U0TrainingReceiptError(f"{label} model state is not tensor-only")
    return {
        "epoch": payload.get("epoch"),
        "iteration": iteration,
        "optimizer_updates": updates,
        "epoch_finished": payload.get("epoch_finished"),
        "checkpoint_reason": payload.get("checkpoint_reason"),
        "model_state_keys": len(model),
        "has_optimizer": isinstance(payload.get("optimizer"), Mapping),
        "has_lr_scheduler": isinstance(payload.get("lr_scheduler"), Mapping),
        "has_amp_scaler": isinstance(payload.get("scaler"), Mapping),
        "has_criterion": isinstance(payload.get("criterion"), Mapping),
        "args": args,
    }


def _transition_audit_path(checkpoint: Path) -> Path:
    experiment_root = checkpoint.parent.parent
    return experiment_root / "audits" / f"{checkpoint.stem}.transition.json"


def _transition_binding(
    *,
    initializer_payload: Mapping[str, Any],
    trained_payload: Mapping[str, Any],
    initializer_record: Mapping[str, Any],
    checkpoint_record: Mapping[str, Any],
    checkpoint: Path,
    target: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    audit_path = _transition_audit_path(checkpoint)
    audit_record = stable_file_record(audit_path, label=f"U{target} transition audit")
    observed = _strict_json_load(Path(audit_record["path"]), label=f"U{target} transition audit")
    if not isinstance(observed, Mapping):
        raise U0TrainingReceiptError(f"U{target} transition audit is not an object")
    try:
        recomputed = audit_u0_transition(initializer_payload, trained_payload)
    except (TypeError, ValueError, RuntimeError) as error:
        raise U0TrainingReceiptError(
            f"U{target} transition replay failed: {error}"
        ) from error
    recomputed["initializer"] = dict(initializer_record)
    recomputed["checkpoint"] = dict(checkpoint_record)
    if dict(observed) != recomputed:
        raise U0TrainingReceiptError(
            f"U{target} transition audit does not equal a fresh replay"
        )
    if recomputed.get("schema") != TRANSITION_SCHEMA or recomputed.get("status") != "verified":
        raise U0TrainingReceiptError(f"U{target} transition audit schema/status drifted")
    if recomputed.get("iteration") != target or recomputed.get("optimizer_updates") != target:
        raise U0TrainingReceiptError(f"U{target} transition audit update count drifted")
    summary = {
        "file": audit_record,
        "recomputed_equal": True,
        "schema": recomputed["schema"],
        "status": recomputed["status"],
        "iteration": recomputed["iteration"],
        "optimizer_updates": recomputed["optimizer_updates"],
        "changed_key_count": recomputed["changed_key_count"],
        "changed_keys": recomputed["changed_keys"],
        "frozen_key_count": recomputed["frozen_key_count"],
        "merged_teacher_tensor_sha256": recomputed[
            "merged_teacher_tensor_sha256"
        ],
        "shared_backbone_alias_tensor_sha256": recomputed[
            "shared_backbone_alias_tensor_sha256"
        ],
        "u0_trainable_tensor_sha256": recomputed[
            "u0_trainable_tensor_sha256"
        ],
    }
    return summary, recomputed


def _frozen_keys(initializer_payload: Mapping[str, Any]) -> list[str]:
    state = initializer_payload["model"]
    contract = initializer_payload["u0_initializer"]
    roles = contract["role_keys"]
    u0_trainable = {
        key
        for key in roles["u0_zero"]
        if not key.removeprefix(U0_PREFIX).startswith("_contract_")
    }
    patch_trainable = set(U0_PATCH_SOURCE_KEYS).difference(FROZEN_PATCH_KEYS)
    allowed = u0_trainable | patch_trainable
    frozen = sorted(set(state).difference(allowed))
    if not frozen:
        raise U0TrainingReceiptError("U0 frozen tensor set is empty")
    return frozen


def _initializer_binding(
    payload: Mapping[str, Any], record: Mapping[str, Any]
) -> tuple[dict[str, Any], list[str], str]:
    state = payload.get("model")
    try:
        validate_stage_b_u0_initializer_payload(
            state, payload, checkpoint_label="U0 training receipt initializer"
        )
    except (TypeError, ValueError, RuntimeError) as error:
        raise U0TrainingReceiptError(f"U0 initializer validation failed: {error}") from error
    contract = payload["u0_initializer"]
    frozen = _frozen_keys(payload)
    frozen_hash = stage_b_u0_tensor_state_sha256(state, frozen)
    summary_fields = (
        "schema",
        "model_state_keys",
        "role_key_counts",
        "merged_teacher_tensor_sha256",
        "stagea_patch_tensor_sha256",
        "shared_backbone_alias_tensor_sha256",
        "u0_zero_tensor_sha256",
        "full_model_tensor_sha256",
        "invariants",
    )
    return (
        {
            "file": dict(record),
            "contract_sha256": canonical_json_sha256(
                _json_safe(contract, label="initializer contract")
            ),
            "contract": {
                key: _json_safe(contract[key], label=f"initializer contract.{key}")
                for key in summary_fields
                if key in contract
            },
            "frozen_key_count": len(frozen),
            "frozen_tensor_sha256": frozen_hash,
            "validated": True,
        },
        frozen,
        frozen_hash,
    )


def build_receipt_payload(
    *,
    initializer: Path,
    u50: Path,
    u100: Path,
    config: Path,
    datasets: Path,
    data_root: Path | None = None,
) -> dict[str, Any]:
    data_root = (data_root or default_data_root()).expanduser().resolve()
    initializer = _resolve_file(initializer, label="U0 initializer")
    u50 = _resolve_file(u50, label="U50 milestone")
    u100 = _resolve_file(u100, label="U100 milestone")
    config = _resolve_file(config, label="U0 config")
    datasets = _resolve_file(datasets, label="U0 dataset JSON")
    if u50.parent != u100.parent or u50.parent.name != "milestones":
        raise U0TrainingReceiptError("U50/U100 are not in one milestones directory")
    experiment_root = u50.parent.parent
    if initializer.parent.parent != experiment_root:
        raise U0TrainingReceiptError(
            "initializer and U50/U100 do not share one experiment root"
        )

    config_record = _config_binding(config)
    dataset_record = _dataset_binding(datasets, data_root=data_root)
    source_records = _source_binding()
    initializer_record = stable_file_record(initializer, label="U0 initializer")
    u50_record = stable_file_record(u50, label="U50 milestone")
    u100_record = stable_file_record(u100, label="U100 milestone")

    initializer_payload = _safe_load_checkpoint(initializer, label="U0 initializer")
    initializer_summary, frozen_keys, initializer_frozen_hash = _initializer_binding(
        initializer_payload, initializer_record
    )
    checkpoints: dict[str, Any] = {}
    transition_replays = []
    for role, target, path, record in (
        ("u50", 50, u50, u50_record),
        ("u100", 100, u100, u100_record),
    ):
        trained_payload = _safe_load_checkpoint(path, label=f"U{target} milestone")
        training = _training_record(
            trained_payload,
            label=f"U{target}",
            target=target,
            checkpoint=path,
            initializer=initializer,
            u50=u50,
            config=config,
            datasets=datasets,
            experiment_root=experiment_root,
            data_root=data_root,
        )
        transition, replay = _transition_binding(
            initializer_payload=initializer_payload,
            trained_payload=trained_payload,
            initializer_record=initializer_record,
            checkpoint_record=record,
            checkpoint=path,
            target=target,
        )
        frozen_hash = stage_b_u0_tensor_state_sha256(
            trained_payload["model"], frozen_keys
        )
        if frozen_hash != initializer_frozen_hash:
            raise U0TrainingReceiptError(
                f"U{target} frozen tensor hash differs from the initializer"
            )
        if transition["frozen_key_count"] != len(frozen_keys):
            raise U0TrainingReceiptError(
                f"U{target} transition frozen-key count differs from receipt contract"
            )
        checkpoints[role] = {
            "file": record,
            "training": training,
            "transition_audit": transition,
            "frozen_key_count": len(frozen_keys),
            "frozen_tensor_sha256": frozen_hash,
        }
        transition_replays.append(replay)
        del trained_payload
        gc.collect()

    initializer_contract = initializer_payload["u0_initializer"]
    merged_hash = initializer_contract["merged_teacher_tensor_sha256"]
    alias_hash = initializer_contract["shared_backbone_alias_tensor_sha256"]
    for role in ("u50", "u100"):
        transition = checkpoints[role]["transition_audit"]
        if transition["merged_teacher_tensor_sha256"] != merged_hash:
            raise U0TrainingReceiptError(f"{role} sealed teacher hash drifted")
        if transition["shared_backbone_alias_tensor_sha256"] != alias_hash:
            raise U0TrainingReceiptError(f"{role} shared-backbone hash drifted")
    del initializer_payload
    gc.collect()

    return {
        "schema": SCHEMA,
        "repository_root": str(REPO_ROOT),
        "experiment_root": str(experiment_root),
        "data_root": str(data_root),
        "checkpoint_load_policy": {
            "weights_only": True,
            "mmap": True,
            "fallback_to_weights_only_false": False,
            "allowed_pickle_globals": sorted(_ALLOWED_CHECKPOINT_GLOBALS),
        },
        "initializer": initializer_summary,
        "checkpoints": checkpoints,
        "config": config_record,
        "datasets": dataset_record,
        "core_sources": source_records,
        "invariants": {
            "single_experiment_root": True,
            "batch_size": EXPECTED_BATCH_SIZE,
            "seed": EXPECTED_SEED,
            "amp": True,
            "optimizer_updates": [50, 100],
            "u50_starts_from_initializer": True,
            "u100_resumes_u50": True,
            "transition_audits_recomputed_equal": len(transition_replays) == 2,
            "frozen_tensor_hash_equal_across_initializer_u50_u100": True,
            "merged_r100_p50_teacher_frozen": True,
            "shared_patch_backbone_frozen": True,
            "three_annotation_jsonl_files_fully_parsed": True,
            "support_patch_tsv_fully_parsed": True,
            "canonical_json_fully_parsed": True,
        },
    }


def _seal_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["receipt_sha256"] = canonical_json_sha256(result)
    return result


def _atomic_publish_fresh_json(path: Path, payload: Mapping[str, Any]) -> None:
    output = path.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
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
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
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
    """Atomically publish ``source`` while preserving an existing destination."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is not None:
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = renameat2(
            -100,
            os.fsencode(source),
            -100,
            os.fsencode(destination),
            1,
        )
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise U0TrainingReceiptError(
                f"refusing to overwrite existing receipt: {destination}"
            )
        if error_number not in (errno.ENOSYS, errno.EINVAL):
            raise OSError(
                error_number,
                os.strerror(error_number),
                str(destination),
            )
    try:
        os.link(source, destination)
    except FileExistsError as error:
        raise U0TrainingReceiptError(
            f"refusing to overwrite existing receipt: {destination}"
        ) from error


def build_receipt(
    *,
    output: Path,
    initializer: Path,
    u50: Path,
    u100: Path,
    config: Path,
    datasets: Path,
    data_root: Path | None = None,
) -> dict[str, Any]:
    output = output.expanduser().resolve()
    if output.exists():
        raise U0TrainingReceiptError(
            f"refusing to overwrite existing receipt: {output}"
        )
    payload = build_receipt_payload(
        initializer=initializer,
        u50=u50,
        u100=u100,
        config=config,
        datasets=datasets,
        data_root=data_root,
    )
    receipt = _seal_payload(payload)
    _atomic_publish_fresh_json(output, receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initializer", required=True)
    parser.add_argument("--u50", required=True)
    parser.add_argument("--u100", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--data-root", default=str(default_data_root()))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = build_receipt(
            output=Path(args.output),
            initializer=Path(args.initializer),
            u50=Path(args.u50),
            u100=Path(args.u100),
            config=Path(args.config),
            datasets=Path(args.datasets),
            data_root=Path(args.data_root),
        )
        output_record = stable_file_record(Path(args.output), label="published receipt")
    except (OSError, U0TrainingReceiptError) as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "status": "built",
                "output": output_record,
                "receipt_sha256": receipt["receipt_sha256"],
                "invariants": receipt["invariants"],
            },
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
