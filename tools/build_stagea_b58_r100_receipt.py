#!/usr/bin/env python3
"""Build and verify the sealed Stage-A-B58 -> R100 handoff receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCHEMA = "pivot.stagea_b58_r100_receipt/v1"
STAGEA_CONFIG = (
    REPO_ROOT
    / "config/ablations/cfg_stagea_b58_trunk_patch0006_realign_20260814.py"
)
STAGEA_DATASETS = REPO_ROOT / "config/datasets_patch_stage_a_lvis_coco2017_local.json"
RANK_CONFIG = REPO_ROOT / "config/ablations/cfg_stageb_gdino_score_adapter_rank_three_ref.py"
RANK_DATASETS = REPO_ROOT / "config/datasets_stageb_gdino_adapter_rank_three_ref.json"
INITIALIZER_SCHEMA = "pivot.stagea.b58_trunk_patch0006_initializer/v1"
ADAPTER_PREFIX = "stage_b_gdino_score_adapter."
PATCH_EXACT = frozenset({"patch_logit_scale"})
PATCH_PREFIXES = ("patch_encoder.", "query_proj_for_patch.")
TRAINABLE_PATCH_EXACT = frozenset({"patch_logit_scale"})
TRAINABLE_PATCH_PREFIXES = (
    "patch_encoder.input_proj.",
    "patch_encoder.norm.",
    "query_proj_for_patch.",
)
EXPECTED_STAGEA_EPOCH = 7
EXPECTED_STAGEA_UPDATES = 45_608
EXPECTED_RANK_UPDATES = 100


class StageAR100ReceiptError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    return {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("receipt_sha256", None)
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    path = path.expanduser().resolve(strict=True)
    value = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if not isinstance(value, Mapping):
        raise StageAR100ReceiptError(f"checkpoint is not a mapping: {path}")
    return value


def _state(payload: Mapping[str, Any], *, label: str) -> Mapping[str, torch.Tensor]:
    value = payload.get("model")
    if not isinstance(value, Mapping) or not value:
        raise StageAR100ReceiptError(f"{label} has no model state")
    if any(not isinstance(key, str) or not torch.is_tensor(tensor) for key, tensor in value.items()):
        raise StageAR100ReceiptError(f"{label} model state is not tensor-only")
    return value


def _args(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = payload.get("args")
    if isinstance(value, Mapping):
        return dict(value)
    if value is not None and hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def _resolve_arg(value: Any) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _tensor_hash(state: Mapping[str, torch.Tensor], keys: list[str]) -> str:
    if not keys:
        raise StageAR100ReceiptError("cannot hash an empty tensor role")
    digest = hashlib.sha256()
    for key in sorted(keys):
        tensor = state[key].detach().cpu().contiguous()
        header = json.dumps(
            [key, str(tensor.dtype), list(tensor.shape)],
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _is_patch_key(key: str) -> bool:
    return key in PATCH_EXACT or key.startswith(PATCH_PREFIXES)


def _is_trainable_patch_key(key: str) -> bool:
    return key in TRAINABLE_PATCH_EXACT or key.startswith(TRAINABLE_PATCH_PREFIXES)


def _require_metadata(
    payload: Mapping[str, Any],
    *,
    label: str,
    epoch: int,
    iteration: int,
    optimizer_updates: int,
    epoch_finished: bool,
    reason: str,
) -> None:
    expected = {
        "epoch": epoch,
        "iteration": iteration,
        "optimizer_updates": optimizer_updates,
        "epoch_finished": epoch_finished,
        "checkpoint_reason": reason,
    }
    drift = {key: (payload.get(key), value) for key, value in expected.items() if payload.get(key) != value}
    if drift:
        raise StageAR100ReceiptError(f"{label} checkpoint metadata drifted: {drift}")
    for key in ("optimizer", "lr_scheduler", "scaler", "rng_state", "epoch_rng_state"):
        if not isinstance(payload.get(key), Mapping):
            raise StageAR100ReceiptError(f"{label} checkpoint lacks resumable {key} state")


def _require_args(args: Mapping[str, Any], expected: Mapping[str, Any], *, label: str) -> None:
    drift: dict[str, tuple[Any, Any]] = {}
    for key, value in expected.items():
        observed = args.get(key)
        if key in {"config_file", "datasets", "pretrain_model_path", "resume"}:
            observed = _resolve_arg(observed)
        if observed != value:
            drift[key] = (observed, value)
    if drift:
        raise StageAR100ReceiptError(f"{label} args drifted: {drift}")


def _all_zero(state: Mapping[str, torch.Tensor], keys: tuple[str, ...]) -> bool:
    return all(key in state and int(torch.count_nonzero(state[key]).item()) == 0 for key in keys)


def build_receipt(*, stagea: Path, initializer: Path, rank: Path) -> dict[str, Any]:
    stagea = stagea.expanduser().resolve(strict=True)
    initializer = initializer.expanduser().resolve(strict=True)
    rank = rank.expanduser().resolve(strict=True)
    stagea_payload = _load(stagea)
    initializer_payload = _load(initializer)
    rank_payload = _load(rank)
    stagea_state = _state(stagea_payload, label="Stage A")
    initializer_state = _state(initializer_payload, label="Stage A initializer")
    rank_state = _state(rank_payload, label="R100")

    initializer_contract = initializer_payload.get("stage_a_b58_patch0006_initializer")
    if not isinstance(initializer_contract, Mapping) or initializer_contract.get("schema") != INITIALIZER_SCHEMA:
        raise StageAR100ReceiptError("initializer provenance schema is invalid")
    from tools.build_stagea_b58_patch0006_initializer import validate_payload

    try:
        validate_payload(initializer_payload)
    except RuntimeError as error:
        raise StageAR100ReceiptError(f"initializer payload verification failed: {error}") from error
    sources = initializer_contract.get("sources")
    b58_source = sources.get("b58_trunk") if isinstance(sources, Mapping) else None
    if not isinstance(b58_source, Mapping) or not b58_source.get("path"):
        raise StageAR100ReceiptError("initializer has no B58 source record")
    b58_path = Path(str(b58_source["path"])).expanduser().resolve(strict=True)
    b58_record = _file_record(b58_path)
    if (
        b58_record["sha256"] != b58_source.get("sha256")
        or b58_record["size_bytes"] != b58_source.get("bytes")
    ):
        raise StageAR100ReceiptError("initializer B58 source file identity drifted")
    b58_state = _state(_load(b58_path), label="B58 source")
    if set(stagea_state) != set(initializer_state):
        raise StageAR100ReceiptError("Stage A model key set drifted from its initializer")

    trainable = sorted(key for key in stagea_state if _is_trainable_patch_key(key))
    frozen = sorted(set(stagea_state).difference(trainable))
    patch_only = sorted(key for key in stagea_state if _is_patch_key(key))
    nonpatch = sorted(set(stagea_state).difference(patch_only))
    if len(trainable) != 9 or len(nonpatch) != 938:
        raise StageAR100ReceiptError(
            f"Stage A role cardinality drifted: trainable={len(trainable)}, nonpatch={len(nonpatch)}"
        )
    changed = [key for key in trainable if not torch.equal(stagea_state[key], initializer_state[key])]
    if changed != trainable:
        missing = sorted(set(trainable).difference(changed))
        raise StageAR100ReceiptError(f"Stage A trainable patch tensors did not all update: {missing}")
    frozen_changed = [key for key in frozen if not torch.equal(stagea_state[key], initializer_state[key])]
    if frozen_changed:
        raise StageAR100ReceiptError(f"Stage A changed frozen tensors: {frozen_changed[:8]}")
    if set(b58_state) != set(nonpatch):
        raise StageAR100ReceiptError("Stage A non-patch trunk key set differs from B58")
    b58_changed = [key for key in nonpatch if not torch.equal(stagea_state[key], b58_state[key])]
    if b58_changed:
        raise StageAR100ReceiptError(f"Stage A non-patch trunk differs from B58: {b58_changed[:8]}")

    _require_metadata(
        stagea_payload,
        label="Stage A",
        epoch=EXPECTED_STAGEA_EPOCH,
        iteration=0,
        optimizer_updates=EXPECTED_STAGEA_UPDATES,
        epoch_finished=True,
        reason="epoch",
    )
    _require_args(
        _args(stagea_payload),
        {
            "config_file": STAGEA_CONFIG.resolve(),
            "datasets": STAGEA_DATASETS.resolve(),
            "pretrain_model_path": initializer,
            "resume": None,
            "seed": 42,
            "batch_size": 38,
            "epochs": 8,
            "amp": True,
            "stage_a_b58_patch_realign": True,
            "patch_dn_num_queries": 0,
            "unfreeze_decoder_last_n_layers": 0,
        },
        label="Stage A",
    )

    adapter = sorted(key for key in rank_state if key.startswith(ADAPTER_PREFIX))
    rank_base = sorted(set(rank_state).difference(adapter))
    if rank_base != nonpatch or len(adapter) != 20:
        raise StageAR100ReceiptError(
            "R100 architecture is not the exact Stage A non-patch trunk plus 20 adapter tensors"
        )
    base_changed = [key for key in nonpatch if not torch.equal(stagea_state[key], rank_state[key])]
    if base_changed:
        raise StageAR100ReceiptError(f"R100 changed the Stage A non-patch trunk: {base_changed[:8]}")
    rank_keys = [key for key in adapter if ".rank_" in key]
    confidence_keys = sorted(set(adapter).difference(rank_keys))
    rank_final = (
        ADAPTER_PREFIX + "rank_output.weight",
        ADAPTER_PREFIX + "rank_output.bias",
    )
    confidence_final = (
        ADAPTER_PREFIX + "confidence_gate.4.weight",
        ADAPTER_PREFIX + "confidence_gate.4.bias",
    )
    if len(rank_keys) != 8 or len(confidence_keys) != 12:
        raise StageAR100ReceiptError(
            f"R100 adapter ownership drifted: rank={len(rank_keys)}, confidence={len(confidence_keys)}"
        )
    if _all_zero(rank_state, rank_final) or not _all_zero(rank_state, confidence_final):
        raise StageAR100ReceiptError("R100 rank/confidence branch-isolation contract failed")

    _require_metadata(
        rank_payload,
        label="R100",
        epoch=0,
        iteration=EXPECTED_RANK_UPDATES,
        optimizer_updates=EXPECTED_RANK_UPDATES,
        epoch_finished=False,
        reason="max_train_iters",
    )
    _require_args(
        _args(rank_payload),
        {
            "config_file": RANK_CONFIG.resolve(),
            "datasets": RANK_DATASETS.resolve(),
            "pretrain_model_path": stagea,
            "resume": None,
            "seed": 42,
            "batch_size": 32,
            "world_size": 1,
            "distributed": False,
            "amp": True,
            "max_train_iters": 100,
            "stage_b_gdino_adapter_train_mode": "rank_only",
            "stage_b_gdino_tn_scope": "",
            "stage_b_gdino_rank_weight": 1.0,
            "stage_b_gdino_confidence_weight": 0.0,
            "stage_b_gdino_queue_size": 0,
            "stage_b_gdino_queue_min_count": 0,
        },
        label="R100",
    )

    code_paths = (
        STAGEA_CONFIG,
        STAGEA_DATASETS,
        RANK_CONFIG,
        RANK_DATASETS,
        REPO_ROOT / "tools/run_stagea_b58_r100_c100.sh",
        REPO_ROOT / "tools/build_stagea_b58_r100_receipt.py",
        REPO_ROOT / "main.py",
        REPO_ROOT / "engine.py",
        REPO_ROOT / "models/GroundingDINO/groundingdino.py",
        REPO_ROOT / "models/GroundingDINO/stage_b_gdino_score_adapter.py",
    )
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "stagea": {
            "checkpoint": _file_record(stagea),
            "initializer": _file_record(initializer),
            "b58_source": b58_record,
            "epoch": EXPECTED_STAGEA_EPOCH,
            "optimizer_updates": EXPECTED_STAGEA_UPDATES,
            "model_state_keys": len(stagea_state),
            "trainable_patch_keys": trainable,
            "trainable_patch_tensor_sha256": _tensor_hash(stagea_state, trainable),
            "frozen_tensor_sha256": _tensor_hash(stagea_state, frozen),
            "nonpatch_tensor_sha256": _tensor_hash(stagea_state, nonpatch),
            "patch_tensor_sha256": _tensor_hash(stagea_state, patch_only),
        },
        "rank_r100": {
            "checkpoint": _file_record(rank),
            "iteration": EXPECTED_RANK_UPDATES,
            "model_state_keys": len(rank_state),
            "base_state_keys": len(rank_base),
            "adapter_state_keys": len(adapter),
            "base_tensor_sha256": _tensor_hash(rank_state, rank_base),
            "rank_tensor_sha256": _tensor_hash(rank_state, rank_keys),
            "confidence_tensor_sha256": _tensor_hash(rank_state, confidence_keys),
        },
        "code": [_file_record(path) for path in code_paths],
        "invariants": {
            "stagea_all_nine_patch_tensors_updated": True,
            "stagea_frozen_tensors_bitwise_equal_initializer": True,
            "stagea_nonpatch_trunk_bitwise_equal_b58": True,
            "r100_base_bitwise_equal_stagea_nonpatch_trunk": True,
            "stagea_patch_state_excluded_only_by_ordinary_gdino_architecture": True,
            "r100_rank_trained_confidence_zero_initialized": True,
            "r100_exactly_100_optimizer_updates": True,
        },
    }
    receipt["receipt_sha256"] = _canonical_sha256(receipt)
    return receipt


def verify_receipt(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StageAR100ReceiptError(f"could not read receipt: {error}") from error
    if not isinstance(stored, dict) or stored.get("schema") != SCHEMA:
        raise StageAR100ReceiptError("receipt schema is invalid")
    if stored.get("receipt_sha256") != _canonical_sha256(stored):
        raise StageAR100ReceiptError("receipt self-hash mismatch")
    rebuilt = build_receipt(
        stagea=Path(stored["stagea"]["checkpoint"]["path"]),
        initializer=Path(stored["stagea"]["initializer"]["path"]),
        rank=Path(stored["rank_r100"]["checkpoint"]["path"]),
    )
    if rebuilt != stored:
        raise StageAR100ReceiptError("receipt no longer matches current artifacts")
    return stored


def _write_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        raise StageAR100ReceiptError(f"refusing to overwrite receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    try:
        temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--stagea-checkpoint", type=Path, required=True)
    build.add_argument("--initializer", type=Path, required=True)
    build.add_argument("--rank-checkpoint", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        payload = build_receipt(
            stagea=args.stagea_checkpoint,
            initializer=args.initializer,
            rank=args.rank_checkpoint,
        )
        _write_atomic(args.output, payload)
        receipt = args.output.resolve(strict=True)
    else:
        payload = verify_receipt(args.receipt)
        receipt = args.receipt.resolve(strict=True)
    print(json.dumps({"status": "verified", "schema": SCHEMA, "receipt": _file_record(receipt), "receipt_sha256": payload["receipt_sha256"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
