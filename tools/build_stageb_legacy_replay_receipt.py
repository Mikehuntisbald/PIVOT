#!/usr/bin/env python3
"""Build or verify an auditable receipt for the legacy Stage-B replay.

The receipt binds the b58 baseline, independent R100 rank replay, independent
P50 confidence replay, and an optional eval-only merged checkpoint.  It uses
the same tensor-state hashing implementation as the Stage-B adapter audits.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.stageb_gdino_adapter_probe_audit import (  # noqa: E402
    ProbeAuditError,
    checkpoint_args,
    file_record,
    load_checkpoint,
    model_hash_record,
    tensor_state_sha256,
)


SCHEMA = "pivot.stageb.legacy_replay_receipt/v1"
ROLES = ("b58", "rank_r100", "confidence_p50", "merged")
REQUIRED_ROLES = ROLES[:3]
HASH_FIELDS = frozenset(
    {
        "file_sha256",
        "base_tensor_sha256",
        "adapter_tensor_sha256",
        "rank_tensor_sha256",
        "confidence_tensor_sha256",
        "full_model_tensor_sha256",
    }
)
ARGS_FIELDS = (
    "config_file",
    "datasets",
    "seed",
    "batch_size",
    "world_size",
    "distributed",
    "amp",
    "num_workers",
    "max_train_iters",
    "checkpoint_interval_iters",
    "gradient_accumulation_steps",
    "pretrain_model_path",
    "resume",
    "stage_b_gdino_adapter_train_mode",
    "stage_b_gdino_tn_scope",
    "stage_b_gdino_rank_weight",
    "stage_b_gdino_confidence_weight",
    "stage_b_gdino_confidence_objective",
    "stage_b_gdino_queue_size",
    "stage_b_gdino_queue_min_count",
)
MERGED_CONTRACT_HASH_FIELDS = {
    "base_tensor_sha256": "base_tensor_sha256",
    "adapter_tensor_sha256": "adapter_tensor_sha256",
    "rank_tensor_sha256": "rank_tensor_sha256",
    "confidence_tensor_sha256": "confidence_tensor_sha256",
    "full_model_tensor_sha256": "full_model_tensor_sha256",
}


class LegacyReplayReceiptError(RuntimeError):
    """A replay checkpoint or receipt failed its closed-world contract."""


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
        raise LegacyReplayReceiptError(
            f"receipt is not canonical JSON: {error}"
        ) from error
    return rendered.encode("ascii")


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _resolve_checkpoint(path: Path, *, role: str) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except OSError as error:
        raise LegacyReplayReceiptError(
            f"{role} checkpoint cannot be resolved: {path}: {error}"
        ) from error
    if not resolved.is_file():
        raise LegacyReplayReceiptError(f"{role} checkpoint is not a file: {resolved}")
    return resolved


def _strict_optional_int(value: Any, *, label: str) -> int | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise LegacyReplayReceiptError(f"{label} must be scalar")
        value = value.detach().cpu().item()
    if isinstance(value, bool):
        raise LegacyReplayReceiptError(f"{label} is bool, not an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return int(value)
    raise LegacyReplayReceiptError(f"{label} must be an integer or null")


def _json_value(value: Any, *, label: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LegacyReplayReceiptError(f"{label} is not finite")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [
            _json_value(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise LegacyReplayReceiptError(
                    f"{label} contains non-string mapping key {key!r}"
                )
            normalized[key] = _json_value(item, label=f"{label}.{key}")
        return normalized
    raise LegacyReplayReceiptError(
        f"{label} has unsupported value type {type(value).__name__}"
    )


def _model_record(state: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    if not state:
        raise LegacyReplayReceiptError(f"{role} checkpoint has an empty model state")
    invalid = [
        key
        for key, value in state.items()
        if not isinstance(key, str) or not torch.is_tensor(value)
    ]
    if invalid:
        raise LegacyReplayReceiptError(
            f"{role} model contains non-tensor entries: {invalid[:8]}"
        )
    try:
        hashes = model_hash_record(state)
        full_hash = tensor_state_sha256(state, state.keys())
    except ProbeAuditError as error:
        raise LegacyReplayReceiptError(f"{role} model audit failed: {error}") from error
    record: dict[str, Any] = {
        "model_state_keys": int(hashes["model_state_keys"]),
        "base_state_keys": int(hashes["base_state_keys"]),
        "adapter_state_keys": int(hashes["adapter_state_keys"]),
        "base_tensor_sha256": hashes["base_model_sha256"],
        "full_model_tensor_sha256": full_hash,
    }
    if hashes["adapter_state_keys"]:
        record.update(
            {
                "adapter_tensor_sha256": hashes["adapter_sha256"],
                "rank_tensor_sha256": hashes["rank_sha256"],
                "confidence_tensor_sha256": hashes["confidence_sha256"],
                "rank_final_zero": hashes["rank_final_zero"],
                "confidence_final_zero": hashes["confidence_final_zero"],
            }
        )
    return record


def _training_record(payload: Mapping[str, Any], *, role: str) -> dict[str, Any]:
    args = checkpoint_args(payload)
    summary = {
        key: _json_value(args[key], label=f"{role}.args.{key}")
        for key in ARGS_FIELDS
        if key in args
    }
    epoch_finished = payload.get("epoch_finished")
    if epoch_finished is not None and type(epoch_finished) is not bool:
        raise LegacyReplayReceiptError(
            f"{role}.epoch_finished must be boolean or null"
        )
    checkpoint_reason = payload.get("checkpoint_reason")
    if checkpoint_reason is not None and not isinstance(checkpoint_reason, str):
        raise LegacyReplayReceiptError(
            f"{role}.checkpoint_reason must be a string or null"
        )
    return {
        "epoch": _strict_optional_int(payload.get("epoch"), label=f"{role}.epoch"),
        "iteration": _strict_optional_int(
            payload.get("iteration"), label=f"{role}.iteration"
        ),
        "optimizer_updates": _strict_optional_int(
            payload.get("optimizer_updates"), label=f"{role}.optimizer_updates"
        ),
        "epoch_finished": epoch_finished,
        "checkpoint_reason": checkpoint_reason,
        "has_optimizer": isinstance(payload.get("optimizer"), Mapping),
        "has_criterion": isinstance(payload.get("criterion"), Mapping),
        "args_present": bool(args),
        "args_summary": summary,
    }


def _merged_contract_record(
    payload: Mapping[str, Any], model: Mapping[str, Any]
) -> dict[str, Any]:
    contract = payload.get("contract")
    if not isinstance(contract, Mapping):
        raise LegacyReplayReceiptError("merged checkpoint has no contract mapping")
    schema = contract.get("schema")
    if not isinstance(schema, str) or not schema:
        raise LegacyReplayReceiptError("merged checkpoint contract schema is missing")
    if contract.get("eval_only") is not True or contract.get("resumable") is not False:
        raise LegacyReplayReceiptError(
            "merged checkpoint must remain eval-only and non-resumable"
        )
    for observed_field, contract_field in MERGED_CONTRACT_HASH_FIELDS.items():
        expected = model.get(observed_field)
        observed = contract.get(contract_field)
        if observed != expected:
            raise LegacyReplayReceiptError(
                "merged contract hash mismatch for "
                f"{contract_field}: recorded {observed!r}, computed {expected!r}"
            )
    if contract.get("model_state_keys") != model["model_state_keys"]:
        raise LegacyReplayReceiptError("merged contract model_state_keys mismatch")
    if contract.get("base_state_keys") != model["base_state_keys"]:
        raise LegacyReplayReceiptError("merged contract base_state_keys mismatch")
    if contract.get("adapter_state_keys") != model["adapter_state_keys"]:
        raise LegacyReplayReceiptError("merged contract adapter_state_keys mismatch")
    return {
        "schema": schema,
        "eval_only": True,
        "resumable": False,
        "hashes_recomputed_equal": True,
    }


def inspect_checkpoint(path: Path, *, role: str) -> dict[str, Any]:
    if role not in ROLES:
        raise LegacyReplayReceiptError(f"unknown checkpoint role: {role}")
    resolved = _resolve_checkpoint(path, role=role)
    try:
        before = resolved.stat()
        checkpoint_file = file_record(resolved)
        after_hash = resolved.stat()
    except (OSError, ProbeAuditError) as error:
        raise LegacyReplayReceiptError(f"could not hash {role} checkpoint: {error}") from error
    if _stat_identity(before) != _stat_identity(after_hash):
        raise LegacyReplayReceiptError(
            f"{role} checkpoint changed while computing its file hash"
        )
    try:
        payload = load_checkpoint(resolved)
    except ProbeAuditError as error:
        raise LegacyReplayReceiptError(f"could not load {role} checkpoint: {error}") from error
    try:
        state = payload.get("model")
        if not isinstance(state, Mapping):
            raise LegacyReplayReceiptError(
                f"{role} checkpoint has no model mapping"
            )
        model = _model_record(state, role=role)
        record = {
            "file": checkpoint_file,
            "model": model,
            "training": _training_record(payload, role=role),
        }
        if role == "merged":
            record["contract"] = _merged_contract_record(payload, model)
    finally:
        del payload
        gc.collect()
    try:
        after_load = resolved.stat()
    except OSError as error:
        raise LegacyReplayReceiptError(
            f"could not restat {role} checkpoint: {error}"
        ) from error
    if _stat_identity(after_hash) != _stat_identity(after_load):
        raise LegacyReplayReceiptError(f"{role} checkpoint changed during CPU audit")
    return record


def _require_role_contracts(checkpoints: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    b58 = checkpoints["b58"]
    rank = checkpoints["rank_r100"]
    confidence = checkpoints["confidence_p50"]
    if b58["model"]["adapter_state_keys"] != 0:
        raise LegacyReplayReceiptError("b58 must not contain the GDINO score adapter")
    if rank["model"]["adapter_state_keys"] <= 0:
        raise LegacyReplayReceiptError("rank_r100 has no GDINO score adapter")
    if rank["model"].get("rank_final_zero") is not False:
        raise LegacyReplayReceiptError("rank_r100 rank output is still zero")
    if rank["model"].get("confidence_final_zero") is not True:
        raise LegacyReplayReceiptError("rank_r100 confidence output is not untouched")
    if confidence["model"]["adapter_state_keys"] <= 0:
        raise LegacyReplayReceiptError("confidence_p50 has no GDINO score adapter")
    if confidence["model"].get("rank_final_zero") is not True:
        raise LegacyReplayReceiptError("confidence_p50 rank output is not untouched")
    if confidence["model"].get("confidence_final_zero") is not False:
        raise LegacyReplayReceiptError("confidence_p50 confidence output is still zero")
    for role, target in (("rank_r100", 100), ("confidence_p50", 50)):
        training = checkpoints[role]["training"]
        if training["iteration"] != target or training["optimizer_updates"] != target:
            raise LegacyReplayReceiptError(
                f"{role} must be sealed at iteration/update {target}, got "
                f"{training['iteration']}/{training['optimizer_updates']}"
            )
    base_hash = b58["model"]["base_tensor_sha256"]
    for role in ("rank_r100", "confidence_p50"):
        if checkpoints[role]["model"]["base_tensor_sha256"] != base_hash:
            raise LegacyReplayReceiptError(
                f"{role} base tensors do not exactly match b58"
            )
    invariants: dict[str, Any] = {
        "b58_has_no_adapter": True,
        "rank_r100_iteration_and_updates": 100,
        "confidence_p50_iteration_and_updates": 50,
        "rank_r100_rank_trained_confidence_untouched": True,
        "confidence_p50_rank_untouched_confidence_trained": True,
        "b58_rank_confidence_shared_base_bitwise": True,
        "merged_present": "merged" in checkpoints,
    }
    if "merged" in checkpoints:
        merged = checkpoints["merged"]
        if merged["model"]["base_tensor_sha256"] != base_hash:
            raise LegacyReplayReceiptError("merged base tensors do not exactly match b58")
        if (
            merged["model"].get("rank_tensor_sha256")
            != rank["model"]["rank_tensor_sha256"]
        ):
            raise LegacyReplayReceiptError(
                "merged rank tensors do not exactly match rank_r100"
            )
        if (
            merged["model"].get("confidence_tensor_sha256")
            != confidence["model"]["confidence_tensor_sha256"]
        ):
            raise LegacyReplayReceiptError(
                "merged confidence tensors do not exactly match confidence_p50"
            )
        invariants.update(
            {
                "merged_base_matches_b58_bitwise": True,
                "merged_rank_matches_rank_r100_bitwise": True,
                "merged_confidence_matches_confidence_p50_bitwise": True,
                "merged_contract_hashes_recomputed_equal": True,
            }
        )
    return invariants


def parse_expectations(values: Sequence[str]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for raw in values:
        selector, separator, digest = raw.partition("=")
        if not separator:
            raise LegacyReplayReceiptError(
                f"expected hash must use role.field=sha256: {raw!r}"
            )
        role, dot, field = selector.partition(".")
        if role not in ROLES or not dot or field not in HASH_FIELDS:
            raise LegacyReplayReceiptError(f"unknown expected hash selector: {selector!r}")
        normalized = digest.strip().lower()
        if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
            raise LegacyReplayReceiptError(
                f"expected hash for {selector} is not a SHA256 digest"
            )
        role_values = result.setdefault(role, {})
        if field in role_values:
            raise LegacyReplayReceiptError(
                f"duplicate expected hash selector: {selector}"
            )
        role_values[field] = normalized
    return result


def _normalized_expectations(
    value: Mapping[str, Mapping[str, str]] | None,
) -> dict[str, dict[str, str]]:
    rendered = []
    for role, fields in (value or {}).items():
        if not isinstance(fields, Mapping):
            raise LegacyReplayReceiptError(
                f"expected hashes for {role} must be a mapping"
            )
        for field, digest in fields.items():
            rendered.append(f"{role}.{field}={digest}")
    return parse_expectations(rendered)


def _observed_hash(record: Mapping[str, Any], field: str) -> str | None:
    if field == "file_sha256":
        return record["file"].get("sha256")
    return record["model"].get(field)


def _check_expectations(
    checkpoints: Mapping[str, Mapping[str, Any]],
    expected: Mapping[str, Mapping[str, str]],
) -> None:
    for role, fields in expected.items():
        if role not in checkpoints:
            raise LegacyReplayReceiptError(
                f"expected hashes name absent checkpoint role {role}"
            )
        for field, wanted in fields.items():
            observed = _observed_hash(checkpoints[role], field)
            if observed != wanted:
                raise LegacyReplayReceiptError(
                    f"{role}.{field} mismatch: expected {wanted}, got {observed}"
                )


def build_receipt_payload(
    *,
    b58_checkpoint: Path,
    rank_r100_checkpoint: Path,
    confidence_p50_checkpoint: Path,
    merged_checkpoint: Path | None = None,
    expected_hashes: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    paths = {
        "b58": b58_checkpoint,
        "rank_r100": rank_r100_checkpoint,
        "confidence_p50": confidence_p50_checkpoint,
    }
    if merged_checkpoint is not None:
        paths["merged"] = merged_checkpoint
    checkpoints = {
        role: inspect_checkpoint(path, role=role) for role, path in paths.items()
    }
    invariants = _require_role_contracts(checkpoints)
    expected = _normalized_expectations(expected_hashes)
    _check_expectations(checkpoints, expected)
    return {
        "schema": SCHEMA,
        "repository_root": str(REPO_ROOT),
        "checkpoints": checkpoints,
        "invariants": invariants,
        "expected_hashes": expected,
    }


def _seal_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["receipt_sha256"] = canonical_json_sha256(result)
    return result


def _write_fresh_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise LegacyReplayReceiptError(
            f"refusing to overwrite existing receipt: {path}"
        ) from error
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        path.unlink(missing_ok=True)
        raise


def build_receipt(
    *,
    output: Path,
    b58_checkpoint: Path,
    rank_r100_checkpoint: Path,
    confidence_p50_checkpoint: Path,
    merged_checkpoint: Path | None = None,
    expected_hashes: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    payload = build_receipt_payload(
        b58_checkpoint=b58_checkpoint,
        rank_r100_checkpoint=rank_r100_checkpoint,
        confidence_p50_checkpoint=confidence_p50_checkpoint,
        merged_checkpoint=merged_checkpoint,
        expected_hashes=expected_hashes,
    )
    receipt = _seal_payload(payload)
    _write_fresh_json(output, receipt)
    return receipt


def _read_receipt(path: Path) -> MutableMapping[str, Any]:
    try:
        raw = path.expanduser().resolve(strict=True).read_text(encoding="ascii")
        value = json.loads(
            raw,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant {token}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise LegacyReplayReceiptError(f"could not read receipt {path}: {error}") from error
    if not isinstance(value, MutableMapping):
        raise LegacyReplayReceiptError("receipt must be a JSON object")
    return value


def verify_receipt(path: Path) -> dict[str, Any]:
    receipt = _read_receipt(path)
    observed_self_hash = receipt.pop("receipt_sha256", None)
    expected_self_hash = canonical_json_sha256(receipt)
    if observed_self_hash != expected_self_hash:
        raise LegacyReplayReceiptError(
            "receipt self-hash mismatch: "
            f"recorded {observed_self_hash!r}, computed {expected_self_hash}"
        )
    if receipt.get("schema") != SCHEMA:
        raise LegacyReplayReceiptError("receipt schema mismatch")
    checkpoints = receipt.get("checkpoints")
    expected_hashes = receipt.get("expected_hashes")
    if not isinstance(checkpoints, Mapping) or not isinstance(expected_hashes, Mapping):
        raise LegacyReplayReceiptError("receipt checkpoint/expectation records are malformed")
    if any(role not in checkpoints for role in REQUIRED_ROLES):
        raise LegacyReplayReceiptError("receipt is missing a required checkpoint role")
    rebuilt = build_receipt_payload(
        b58_checkpoint=Path(str(checkpoints["b58"]["file"]["path"])),
        rank_r100_checkpoint=Path(str(checkpoints["rank_r100"]["file"]["path"])),
        confidence_p50_checkpoint=Path(
            str(checkpoints["confidence_p50"]["file"]["path"])
        ),
        merged_checkpoint=(
            Path(str(checkpoints["merged"]["file"]["path"]))
            if "merged" in checkpoints
            else None
        ),
        expected_hashes=expected_hashes,
    )
    if dict(receipt) != rebuilt:
        raise LegacyReplayReceiptError(
            "receipt no longer equals a replay from its checkpoint inputs"
        )
    return {
        "schema": SCHEMA,
        "status": "verified",
        "receipt": file_record(path.expanduser().resolve(strict=True)),
        "receipt_sha256": observed_self_hash,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--output", required=True)
    build.add_argument("--b58-checkpoint", required=True)
    build.add_argument("--rank-r100-checkpoint", required=True)
    build.add_argument("--confidence-p50-checkpoint", required=True)
    build.add_argument("--merged-checkpoint")
    build.add_argument(
        "--expect",
        action="append",
        default=[],
        metavar="ROLE.FIELD=SHA256",
        help=(
            "Lock a file or tensor hash. Fields: "
            + ", ".join(sorted(HASH_FIELDS))
        ),
    )
    verify = subparsers.add_parser("verify")
    verify.add_argument("--receipt", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build":
            receipt = build_receipt(
                output=Path(args.output),
                b58_checkpoint=Path(args.b58_checkpoint),
                rank_r100_checkpoint=Path(args.rank_r100_checkpoint),
                confidence_p50_checkpoint=Path(args.confidence_p50_checkpoint),
                merged_checkpoint=(
                    Path(args.merged_checkpoint) if args.merged_checkpoint else None
                ),
                expected_hashes=parse_expectations(args.expect),
            )
        else:
            receipt = verify_receipt(Path(args.receipt))
    except (LegacyReplayReceiptError, ProbeAuditError) as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
