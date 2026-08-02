#!/usr/bin/env python3
"""Build or verify the fixed b58 successful-update batch-slot receipt.

The historical checkpoints are trusted pickle inputs only after both their
canonical paths and content hashes have been checked.  Optimizer state is the
primary update-count evidence.  Multiplying successful optimizer updates by
the single-process global batch size measures only successful-update batch
slots.  It does not measure all consumed batches or FLOPs.  Epoch JSONL rows
and the two earlier checkpoints are auxiliary cross-checks and cannot override
that primary fact.
"""

from __future__ import annotations

import argparse
import ast
import ctypes
import errno
import hashlib
import json
import math
import numbers
import os
import sys
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "pivot.stageb.b58_exposure_derivation_receipt/v1"
ALGORITHM_VERSION = "b58_successful_optimizer_step_global_batch_slots/v1"

BASELINE_REPO_ROOT = Path("/media/haoyi/T9/gdino")
BASELINE_TRAIN_ROOT = (
    BASELINE_REPO_ROOT
    / "outputs/gdino_ft_stageb_from_gdino_ft_e1_with_tn_bs19_nopatchbranch"
)
CANONICAL_RECEIPT_PATH = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/attestations/"
    "b58_exposure_derivation_receipt.json"
)

LOCKED_SHA256: Mapping[str, str] = {
    "checkpoint0001": (
        "b58e5209dc07dbffb2e5ed3d792e0db70c3306ec2ed24389693b4aeeebab1157"
    ),
    "checkpoint0000": (
        "2ab614010baca3784b0a7ad3c778aca6d4f0f4d52e2cb5dff4ffc2dab3c31aef"
    ),
    "checkpoint_iter": (
        "3eb58702bfaddfe61461103e1070d51135d0189f735cc33b8524f9431310b0a2"
    ),
    "config_args": (
        "62ba05757267c250f5f5d95588daa03b067c5ad3598b38b79ac82e4d4ac3c931"
    ),
    "config_source": (
        "f0dc568c6f35225176712618d5f3449b253478ef32a2b65fa9e089da1ad8a05f"
    ),
    "epoch_records": (
        "c989fc76730e858828d3d0de405071e06deda088ac2c6e2e4b49706f52f0bc62"
    ),
    "training_main": (
        "71e64ba1fd69d1d84b95a5582effd5b22292c681d69545decd612d8296d9a043"
    ),
}

BASELINE_BATCH_SIZE = 19
BASELINE_OPTIMIZER_UPDATES = 49_539
BASELINE_OPTIMIZER_STATE_COUNT = 661
BASELINE_SUCCESSFUL_UPDATE_BATCH_SLOTS = 941_241
CANDIDATE_BATCH_SIZE = 40
CANDIDATE_OPTIMIZER_UPDATES = 23_532
CANDIDATE_SUCCESSFUL_UPDATE_BATCH_SLOTS = 941_280

CHECKPOINT_EXPECTATIONS: Mapping[str, Mapping[str, Any]] = {
    "checkpoint0001": {
        "epoch": 1,
        "iteration": 0,
        "epoch_finished": True,
        "args_batch_size": 19,
        "args_epochs": 2,
        "optimizer_state_count": 661,
        "optimizer_step": 49_539,
        "evidence_role": "primary_optimizer_update_truth",
    },
    "checkpoint0000": {
        "epoch": 0,
        "iteration": 0,
        "epoch_finished": True,
        "args_batch_size": 19,
        "args_epochs": 1,
        "optimizer_state_count": 661,
        "optimizer_step": 24_766,
        "evidence_role": "auxiliary_historical_cross_check",
    },
    "checkpoint_iter": {
        "epoch": 1,
        "iteration": 24_500,
        "epoch_finished": False,
        "args_batch_size": 19,
        "args_epochs": 2,
        "optimizer_state_count": 661,
        "optimizer_step": 49_257,
        "evidence_role": "auxiliary_content_addressed_historical_cross_check",
    },
}

EVIDENCE_KEYS = frozenset(
    {
        "checkpoint0001",
        "checkpoint0000",
        "checkpoint_iter",
        "config_args",
        "epoch_records",
        "training_main",
    }
)


class ExposureReceiptError(RuntimeError):
    """The b58 exposure derivation contract failed closed."""


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish source without replacing an existing receipt."""

    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as exc:  # pragma: no cover - required Linux runtime
        raise ExposureReceiptError(
            "atomic receipt publication requires Linux renameat2"
        ) from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,  # AT_FDCWD
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise ExposureReceiptError(
            f"receipt appeared concurrently and was not replaced: {destination}"
        )
    raise OSError(error_number, os.strerror(error_number), str(destination))


def _input_paths() -> dict[str, Path]:
    return {
        "checkpoint0001": BASELINE_TRAIN_ROOT / "checkpoint0001.pth",
        "checkpoint0000": BASELINE_TRAIN_ROOT / "checkpoint0000.pth",
        "checkpoint_iter": BASELINE_TRAIN_ROOT / "checkpoint_iter.pth",
        "config_args": BASELINE_TRAIN_ROOT / "config_args_all.json",
        "config_source": BASELINE_TRAIN_ROOT / "config_cfg.py",
        "epoch_records": BASELINE_TRAIN_ROOT / "log.txt",
        "training_main": BASELINE_REPO_ROOT / "main.py",
    }


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        rendered = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ExposureReceiptError(f"receipt is not canonical JSON: {exc}") from exc
    return rendered.encode("ascii")


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _strict_json_loads(data: bytes, *, label: str) -> Any:
    def _reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    try:
        return json.loads(
            data.decode("utf-8"),
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ExposureReceiptError(f"cannot parse {label}: {exc}") from exc


def _hash_file_stable(path: Path) -> tuple[str, int]:
    try:
        before = path.stat()
        if not path.is_file():
            raise ExposureReceiptError(f"locked input is not a file: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                digest.update(chunk)
        after = path.stat()
    except OSError as exc:
        raise ExposureReceiptError(f"cannot hash locked input {path}: {exc}") from exc
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ExposureReceiptError(f"locked input changed while hashing: {path}")
    return digest.hexdigest(), int(after.st_size)


def _locked_file_record(name: str, path: Path) -> dict[str, Any]:
    if name not in LOCKED_SHA256:
        raise ExposureReceiptError(f"no locked digest for {name}")
    expected_path = _input_paths().get(name)
    if expected_path is None:
        raise ExposureReceiptError(f"no canonical path for {name}")
    try:
        resolved = path.expanduser().resolve(strict=True)
        canonical = expected_path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise ExposureReceiptError(f"cannot resolve locked {name} path: {exc}") from exc
    if resolved != canonical:
        raise ExposureReceiptError(
            f"{name} path drifted: expected {canonical}, observed {resolved}"
        )
    observed_sha256, size_bytes = _hash_file_stable(resolved)
    expected_sha256 = LOCKED_SHA256[name]
    if observed_sha256 != expected_sha256:
        raise ExposureReceiptError(
            f"{name} sha256 drifted: expected {expected_sha256}, "
            f"observed {observed_sha256}"
        )
    return {
        "path": str(resolved),
        "sha256": observed_sha256,
        "size_bytes": size_bytes,
    }


def _read_locked_bytes(name: str, path: Path) -> tuple[bytes, dict[str, Any]]:
    record = _locked_file_record(name, path)
    resolved = Path(record["path"])
    try:
        before = resolved.stat()
        data = resolved.read_bytes()
        after = resolved.stat()
    except OSError as exc:
        raise ExposureReceiptError(f"cannot read locked input {resolved}: {exc}") from exc
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ExposureReceiptError(f"locked input changed while reading: {resolved}")
    if hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise ExposureReceiptError(f"locked input changed after hashing: {resolved}")
    return data, record


def _strict_integral(value: Any, *, label: str) -> int:
    if hasattr(value, "numel") and hasattr(value, "item"):
        try:
            numel = value.numel()
        except Exception as exc:  # pragma: no cover - defensive for tensor-likes
            raise ExposureReceiptError(f"{label} is not a readable scalar: {exc}") from exc
        if isinstance(numel, bool) or numel != 1:
            raise ExposureReceiptError(f"{label} is not a scalar")
        try:
            value = value.item()
        except Exception as exc:
            raise ExposureReceiptError(f"{label} cannot be converted to a scalar: {exc}") from exc
    if isinstance(value, bool):
        raise ExposureReceiptError(f"{label} is bool, not an integer")
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        rendered = float(value)
        if not math.isfinite(rendered):
            raise ExposureReceiptError(f"{label} is not finite")
        if not rendered.is_integer():
            raise ExposureReceiptError(f"{label} is not integral: {rendered!r}")
        return int(rendered)
    raise ExposureReceiptError(
        f"{label} has unsupported numeric type {type(value).__name__}"
    )


def _checkpoint_args(checkpoint: Mapping[str, Any], *, label: str) -> Mapping[str, Any]:
    value = checkpoint.get("args")
    if not isinstance(value, Mapping):
        raise ExposureReceiptError(f"{label}.args is not a mapping")
    return value


def _validate_single_process_args(value: Mapping[str, Any], *, label: str) -> None:
    world_size = _strict_integral(
        value.get("world_size"), label=f"{label}.world_size"
    )
    rank = _strict_integral(value.get("rank"), label=f"{label}.rank")
    local_rank = _strict_integral(
        value.get("local_rank"), label=f"{label}.local_rank"
    )
    distributed = value.get("distributed")
    if type(distributed) is not bool:
        raise ExposureReceiptError(f"{label}.distributed is not an exact bool")
    if (world_size, distributed, rank, local_rank) != (1, False, 0, 0):
        raise ExposureReceiptError(
            f"{label} is not the locked single-process run: observed "
            f"world_size={world_size}, distributed={distributed!r}, "
            f"rank={rank}, local_rank={local_rank}"
        )
    if "gpu" in value:
        gpu = _strict_integral(value["gpu"], label=f"{label}.gpu")
        if gpu != 0:
            raise ExposureReceiptError(
                f"{label}.gpu drifted: expected 0, observed {gpu}"
            )


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _hash_open_file(handle: Any) -> tuple[str, int]:
    handle.seek(0)
    digest = hashlib.sha256()
    size_bytes = 0
    for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
        digest.update(chunk)
        size_bytes += len(chunk)
    return digest.hexdigest(), size_bytes


def _inspect_checkpoint(name: str, path: Path) -> dict[str, Any]:
    expected = CHECKPOINT_EXPECTATIONS[name]
    if name not in LOCKED_SHA256:
        raise ExposureReceiptError(f"no locked digest for {name}")
    expected_path = _input_paths().get(name)
    if expected_path is None:
        raise ExposureReceiptError(f"no canonical path for {name}")
    try:
        resolved = path.expanduser().resolve(strict=True)
        canonical = expected_path.expanduser().resolve(strict=True)
    except OSError as exc:
        raise ExposureReceiptError(f"cannot resolve locked {name} path: {exc}") from exc
    if resolved != canonical:
        raise ExposureReceiptError(
            f"{name} path drifted: expected {canonical}, observed {resolved}"
        )

    try:
        import torch

        with resolved.open("rb") as handle:
            fd_before = os.fstat(handle.fileno())
            path_before = resolved.stat()
            if _stat_identity(fd_before) != _stat_identity(path_before):
                raise ExposureReceiptError(
                    f"{name} path changed while opening the trusted checkpoint"
                )
            first_sha256, first_size = _hash_open_file(handle)
            expected_sha256 = LOCKED_SHA256[name]
            if first_size != fd_before.st_size:
                raise ExposureReceiptError(
                    f"{name} size changed during the first descriptor hash"
                )
            if first_sha256 != expected_sha256:
                raise ExposureReceiptError(
                    f"{name} sha256 drifted: expected {expected_sha256}, "
                    f"observed {first_sha256}"
                )

            # Load the exact descriptor that was hashed.  Reopening the path here
            # would allow a replacement between the digest check and pickle load.
            handle.seek(0)
            checkpoint = torch.load(
                handle,
                map_location="cpu",
                weights_only=False,
            )
            try:
                if not isinstance(checkpoint, Mapping):
                    raise ExposureReceiptError(f"{name} checkpoint is not a mapping")
                epoch = _strict_integral(
                    checkpoint.get("epoch"), label=f"{name}.epoch"
                )
                iteration = _strict_integral(
                    checkpoint.get("iteration"), label=f"{name}.iteration"
                )
                epoch_finished = checkpoint.get("epoch_finished")
                if type(epoch_finished) is not bool:
                    raise ExposureReceiptError(
                        f"{name}.epoch_finished is not an exact bool"
                    )
                args = _checkpoint_args(checkpoint, label=name)
                args_batch_size = _strict_integral(
                    args.get("batch_size"), label=f"{name}.args.batch_size"
                )
                args_epochs = _strict_integral(
                    args.get("epochs"), label=f"{name}.args.epochs"
                )
                _validate_single_process_args(args, label=f"{name}.args")

                optimizer = checkpoint.get("optimizer")
                states = (
                    optimizer.get("state")
                    if isinstance(optimizer, Mapping)
                    else None
                )
                if not isinstance(states, Mapping) or not states:
                    raise ExposureReceiptError(f"{name} has no optimizer state mapping")
                expected_count = int(expected["optimizer_state_count"])
                if len(states) != expected_count:
                    raise ExposureReceiptError(
                        f"{name} optimizer state count drifted: expected {expected_count}, "
                        f"observed {len(states)}"
                    )
                observed_steps: set[int] = set()
                for state_key, state in states.items():
                    if not isinstance(state, Mapping):
                        raise ExposureReceiptError(
                            f"{name} optimizer state {state_key!r} is not a mapping"
                        )
                    if "step" not in state:
                        raise ExposureReceiptError(
                            f"{name} optimizer state {state_key!r} has no step"
                        )
                    observed_steps.add(
                        _strict_integral(
                            state["step"],
                            label=f"{name}.optimizer.state[{state_key!r}].step",
                        )
                    )
                if len(observed_steps) != 1:
                    raise ExposureReceiptError(
                        f"{name} optimizer states have mixed steps {sorted(observed_steps)}"
                    )
                optimizer_step = next(iter(observed_steps))

                observed_metadata = {
                    "epoch": epoch,
                    "iteration": iteration,
                    "epoch_finished": epoch_finished,
                    "args_batch_size": args_batch_size,
                    "args_epochs": args_epochs,
                    "optimizer_state_count": len(states),
                    "optimizer_step": optimizer_step,
                }
                for key, observed in observed_metadata.items():
                    if observed != expected[key] or type(observed) is not type(
                        expected[key]
                    ):
                        raise ExposureReceiptError(
                            f"{name} {key} drifted: expected {expected[key]!r}, "
                            f"observed {observed!r}"
                        )
            finally:
                del checkpoint

            second_sha256, second_size = _hash_open_file(handle)
            fd_after = os.fstat(handle.fileno())
            if (
                _stat_identity(fd_after) != _stat_identity(fd_before)
                or second_size != fd_after.st_size
                or second_sha256 != expected_sha256
                or second_sha256 != first_sha256
            ):
                raise ExposureReceiptError(
                    f"{name} changed across trusted checkpoint load"
                )
    except Exception as exc:
        if isinstance(exc, ExposureReceiptError):
            raise
        raise ExposureReceiptError(
            f"cannot load trusted locked checkpoint {name}: {exc}"
        ) from exc
    try:
        path_after = resolved.stat()
    except OSError as exc:
        raise ExposureReceiptError(
            f"cannot restat locked checkpoint path {resolved}: {exc}"
        ) from exc
    if _stat_identity(path_after) != _stat_identity(fd_after):
        raise ExposureReceiptError(
            f"{name} path no longer references the loaded checkpoint"
        )
    return {
        "path": str(resolved),
        "sha256": first_sha256,
        "size_bytes": first_size,
        "evidence_role": expected["evidence_role"],
        **observed_metadata,
        "pickle_policy": (
            "weights_only_false_after_exact_canonical_path_and_sha256_match"
        ),
    }


def _attribute_path(node: ast.AST) -> str | None:
    pieces: list[str] = []
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        pieces.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    pieces.append(current.id)
    return ".".join(reversed(pieces))


def _is_not_args_eval(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.Not)
        and _attribute_path(node.operand) == "args.eval"
    )


class _BatchSamplerVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.ancestors: list[ast.AST] = []
        self.matches: list[tuple[ast.Assign, ast.Call, bool]] = []

    def generic_visit(self, node: ast.AST) -> None:
        self.ancestors.append(node)
        super().generic_visit(node)
        self.ancestors.pop()

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        target_match = any(
            isinstance(target, ast.Name) and target.id == "batch_sampler_train"
            for target in node.targets
        )
        if target_match and isinstance(node.value, ast.Call):
            protected = any(
                isinstance(parent, ast.If) and _is_not_args_eval(parent.test)
                for parent in self.ancestors
            )
            self.matches.append((node, node.value, protected))
        self.generic_visit(node)


def _verify_training_main_ast(data: bytes) -> dict[str, Any]:
    try:
        tree = ast.parse(data.decode("utf-8"), filename="training_main")
    except (UnicodeDecodeError, SyntaxError) as exc:
        raise ExposureReceiptError(f"cannot parse training main.py AST: {exc}") from exc
    visitor = _BatchSamplerVisitor()
    visitor.visit(tree)
    if len(visitor.matches) != 1:
        raise ExposureReceiptError(
            "training main.py must have exactly one batch_sampler_train assignment"
        )
    assignment, call, protected = visitor.matches[0]
    if not protected:
        raise ExposureReceiptError(
            "batch_sampler_train is not guarded by exact `if not args.eval`"
        )
    if _attribute_path(call.func) != "torch.utils.data.BatchSampler":
        raise ExposureReceiptError("training BatchSampler constructor drifted")
    if len(call.args) != 2:
        raise ExposureReceiptError("training BatchSampler must have two positional args")
    if not isinstance(call.args[0], ast.Name) or call.args[0].id != "sampler_train":
        raise ExposureReceiptError("training BatchSampler sampler argument drifted")
    if _attribute_path(call.args[1]) != "args.batch_size":
        raise ExposureReceiptError("training BatchSampler batch-size argument drifted")
    if any(keyword.arg is None for keyword in call.keywords):
        raise ExposureReceiptError("training BatchSampler uses expanded kwargs")
    drop_keywords = [keyword for keyword in call.keywords if keyword.arg == "drop_last"]
    if len(drop_keywords) != 1 or len(call.keywords) != 1:
        raise ExposureReceiptError("training BatchSampler keyword contract drifted")
    drop_value = drop_keywords[0].value
    if not (
        isinstance(drop_value, ast.Constant)
        and type(drop_value.value) is bool
        and drop_value.value is True
    ):
        raise ExposureReceiptError("training BatchSampler drop_last is not exact True")
    return {
        "assignment_lineno": int(assignment.lineno),
        "call_lineno": int(call.lineno),
        "constructor": "torch.utils.data.BatchSampler",
        "sampler_argument": "sampler_train",
        "batch_size_argument": "args.batch_size",
        "guard": "if_not_args.eval",
        "drop_last": True,
        "drop_last_ast": "Constant(bool:true)",
    }


def _validate_config_args(data: bytes) -> dict[str, Any]:
    value = _strict_json_loads(data, label="config_args_all.json")
    if not isinstance(value, Mapping):
        raise ExposureReceiptError("config_args_all.json is not an object")
    _validate_single_process_args(value, label="config_args")
    batch_size = _strict_integral(
        value.get("batch_size"), label="config_args.batch_size"
    )
    epochs = _strict_integral(value.get("epochs"), label="config_args.epochs")
    options = value.get("options")
    if not isinstance(options, Mapping):
        raise ExposureReceiptError("config_args.options is not an object")
    option_batch_size = _strict_integral(
        options.get("batch_size"), label="config_args.options.batch_size"
    )
    option_epochs = _strict_integral(
        options.get("epochs"), label="config_args.options.epochs"
    )
    observed = (batch_size, epochs, option_batch_size, option_epochs)
    if observed != (19, 2, 19, 2):
        raise ExposureReceiptError(
            "config_args batch/epoch contract drifted: "
            f"observed {observed!r}"
        )
    return {
        "batch_size": batch_size,
        "epochs": epochs,
        "options_batch_size": option_batch_size,
        "options_epochs": option_epochs,
    }


def _reject_non_finite_json(value: Any, *, label: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, numbers.Real):
        if not math.isfinite(float(value)):
            raise ExposureReceiptError(f"{label} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ExposureReceiptError(f"{label} has a non-string object key")
            _reject_non_finite_json(child, label=f"{label}.{key}")
        return
    if isinstance(value, Sequence):
        for index, child in enumerate(value):
            _reject_non_finite_json(child, label=f"{label}[{index}]")
        return
    raise ExposureReceiptError(
        f"{label} has unsupported JSON value {type(value).__name__}"
    )


def _validate_epoch_records(data: bytes) -> dict[str, Any]:
    rows: list[Mapping[str, Any]] = []
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ExposureReceiptError(f"log.txt is not UTF-8: {exc}") from exc
    lines = text.splitlines()
    if len(lines) != 2 or any(not line.strip() for line in lines):
        raise ExposureReceiptError(
            f"log.txt must contain exactly two non-empty JSONL rows, got {len(lines)}"
        )
    for index, line in enumerate(lines):
        value = _strict_json_loads(line.encode("utf-8"), label=f"log.txt row {index}")
        if not isinstance(value, Mapping):
            raise ExposureReceiptError(f"log.txt row {index} is not an object")
        for required in ("train_lr", "train_loss", "now_time", "epoch_time"):
            if required not in value:
                raise ExposureReceiptError(
                    f"log.txt row {index} is missing {required}"
                )
        _reject_non_finite_json(value, label=f"log.txt row {index}")
        rows.append(value)
    return {
        "record_count": len(rows),
        "completed_epoch_record_indices": [0, 1],
        "role": "auxiliary_epoch_completion_evidence_not_update_truth",
    }


def _derive_receipt_payload() -> dict[str, Any]:
    paths = _input_paths()
    checkpoint_evidence = {
        name: _inspect_checkpoint(name, paths[name])
        for name in ("checkpoint0001", "checkpoint0000", "checkpoint_iter")
    }

    config_data, config_record = _read_locked_bytes(
        "config_args", paths["config_args"]
    )
    config_summary = _validate_config_args(config_data)
    config_source_data, config_source_record = _read_locked_bytes(
        "config_source", paths["config_source"]
    )
    if not config_source_data:
        raise ExposureReceiptError("locked config_cfg.py is empty")

    log_data, log_record = _read_locked_bytes(
        "epoch_records", paths["epoch_records"]
    )
    log_summary = _validate_epoch_records(log_data)

    main_data, main_record = _read_locked_bytes(
        "training_main", paths["training_main"]
    )
    sampler_proof = _verify_training_main_ast(main_data)

    baseline_successful_slots = BASELINE_OPTIMIZER_UPDATES * BASELINE_BATCH_SIZE
    candidate_updates_from_ceil = math.ceil(
        baseline_successful_slots / CANDIDATE_BATCH_SIZE
    )
    candidate_successful_slots = candidate_updates_from_ceil * CANDIDATE_BATCH_SIZE
    if (
        baseline_successful_slots != BASELINE_SUCCESSFUL_UPDATE_BATCH_SLOTS
        or candidate_updates_from_ceil != CANDIDATE_OPTIMIZER_UPDATES
        or candidate_successful_slots != CANDIDATE_SUCCESSFUL_UPDATE_BATCH_SLOTS
        or candidate_successful_slots - baseline_successful_slots != 39
    ):
        raise ExposureReceiptError(
            "locked successful-update batch-slot arithmetic is inconsistent"
        )

    evidence: dict[str, Any] = {
        **checkpoint_evidence,
        "config_args": {
            **config_record,
            **config_summary,
            "locked_config_source": config_source_record,
            "role": "auxiliary_batch_and_epoch_configuration_evidence",
        },
        "epoch_records": {**log_record, **log_summary},
        "training_main": {
            **main_record,
            "batch_sampler_ast_proof": sampler_proof,
            "role": "batch_slot_cardinality_proof",
        },
    }
    if set(evidence) != EVIDENCE_KEYS:
        raise ExposureReceiptError("internal evidence artifact set drifted")

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "algorithm_version": ALGORITHM_VERSION,
        "status": "verified",
        "derived_from_hashed_optimizer_states": True,
        "baseline": {
            "checkpoint_sha256": LOCKED_SHA256["checkpoint0001"],
            "batch_size": BASELINE_BATCH_SIZE,
            "optimizer_updates": BASELINE_OPTIMIZER_UPDATES,
            "optimizer_state_count": BASELINE_OPTIMIZER_STATE_COUNT,
            "successful_update_batch_slots": (
                BASELINE_SUCCESSFUL_UPDATE_BATCH_SLOTS
            ),
            "derivation": "checkpoint0001_optimizer_step49539_x_batch19",
            "drop_last": True,
            "matching_unit": "successful_optimizer_update_global_batch_slots",
        },
        "candidate": {
            "id": "M0",
            "architecture_objective": "S2F",
            "batch_size": CANDIDATE_BATCH_SIZE,
            "optimizer_updates": CANDIDATE_OPTIMIZER_UPDATES,
            "successful_update_batch_slots": (
                CANDIDATE_SUCCESSFUL_UPDATE_BATCH_SLOTS
            ),
            "derivation": "23532_optimizer_updates_x_batch40",
        },
        "candidate_minus_baseline_successful_update_batch_slots": 39,
        "successful_update_batch_slot_math": {
            "baseline": "49539_optimizer_updates_x_19_batch_slots=941241",
            "candidate_update_rule": "ceil(941241/40)=23532",
            "candidate": "23532_optimizer_updates_x_40_batch_slots=941280",
            "overshoot_batch_slots": 39,
        },
        "scope_limitations": {
            "total_consumed_batch_slots_derived": False,
            "flops_matched": False,
            "wall_clock_compute_matched": False,
            "statement": (
                "This receipt matches batch slots attached to successful optimizer "
                "updates. Optimizer state does not count AMP-skipped attempted "
                "batches, total consumed samples, FLOPs, or wall-clock compute."
            ),
        },
        "evidence_policy": {
            "primary_update_truth": "evidence.checkpoint0001.optimizer_step",
            "auxiliary_cannot_override_primary": [
                "evidence.checkpoint0000",
                "evidence.checkpoint_iter",
                "evidence.config_args",
                "evidence.epoch_records",
            ],
            "checkpoint_iter_mutability_policy": (
                "fail_closed_against_locked_historical_sha256"
            ),
            "pickle_policy": (
                "weights_only_false_only_for_exact_path_and_sha256_locked_inputs"
            ),
        },
        "evidence": evidence,
    }
    payload["receipt_sha256"] = canonical_json_sha256(payload)
    return payload


def _read_receipt(path: Path) -> dict[str, Any]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ExposureReceiptError(f"cannot read receipt {path}: {exc}") from exc
    value = _strict_json_loads(data, label="b58 exposure receipt")
    if not isinstance(value, dict):
        raise ExposureReceiptError("b58 exposure receipt is not an object")
    return value


def _require_canonical_receipt_path(path: Path, *, must_exist: bool) -> Path:
    canonical = CANONICAL_RECEIPT_PATH.expanduser().resolve(strict=False)
    resolved = path.expanduser().resolve(strict=False)
    if resolved != canonical:
        raise ExposureReceiptError(
            f"receipt path is not canonical: expected {canonical}, observed {resolved}"
        )
    if must_exist:
        try:
            resolved = path.expanduser().resolve(strict=True)
        except OSError as exc:
            raise ExposureReceiptError(f"canonical receipt is missing: {path}") from exc
        if resolved != canonical:
            raise ExposureReceiptError(
                f"canonical receipt resolves elsewhere: expected {canonical}, "
                f"observed {resolved}"
            )
    return resolved


def verify_receipt(path: Path | None = None) -> dict[str, Any]:
    receipt_path = _require_canonical_receipt_path(
        CANONICAL_RECEIPT_PATH if path is None else path,
        must_exist=True,
    )
    observed = _read_receipt(receipt_path)
    recorded_hash = observed.get("receipt_sha256")
    if not isinstance(recorded_hash, str):
        raise ExposureReceiptError("receipt_sha256 is missing")
    without_hash = dict(observed)
    without_hash.pop("receipt_sha256", None)
    expected_hash = canonical_json_sha256(without_hash)
    if recorded_hash != expected_hash:
        raise ExposureReceiptError(
            f"receipt self-hash drifted: expected {expected_hash}, "
            f"observed {recorded_hash}"
        )
    expected = _derive_receipt_payload()
    if observed != expected:
        raise ExposureReceiptError(
            "receipt payload differs from a fresh replay of locked inputs"
        )
    return observed


def _write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path = _require_canonical_receipt_path(path, must_exist=False)
    if path.exists():
        raise ExposureReceiptError(f"receipt already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    data = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    try:
        with temporary.open("x", encoding="ascii") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        _rename_noreplace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def build_receipt() -> dict[str, Any]:
    path = _require_canonical_receipt_path(
        CANONICAL_RECEIPT_PATH,
        must_exist=False,
    )
    if path.exists():
        raise ExposureReceiptError(f"receipt already exists: {path}")
    payload = _derive_receipt_payload()
    _write_json_exclusive(path, payload)
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("build", "dry-run", "verify"),
        help=(
            "build writes the fresh canonical receipt; dry-run replays all "
            "evidence without writing; verify replays an existing canonical receipt"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build":
            payload = build_receipt()
        elif args.command == "dry-run":
            payload = _derive_receipt_payload()
        else:
            payload = verify_receipt()
    except (ExposureReceiptError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
