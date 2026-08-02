#!/usr/bin/env python3
"""Seal the clean DD1 U1000 model as a role-routed model-only initializer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.GroundingDINO.stage_b_data_driven_score import (  # noqa: E402
    DATA_DRIVEN_ROLE_ROUTED_INITIALIZER_SCHEMA,
    data_driven_tensor_state_sha256,
)
from tools.build_stageb_data_driven_role_routed_clean_assignment import (  # noqa: E402
    _rename_directory_noreplace,
)


SOURCE_CHECKPOINT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/data_driven_new_head_20260723/"
    "lr_probe_u1000/lr3e4/checkpoint_iter.pth"
)
SOURCE_CHECKPOINT_SHA256 = (
    "dcfd1bf29668b7190f509587f1c9664345da168a9ee874bd97a1a032c01a1aa6"
)
A0_INITIALIZER = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/data_driven_initializers/fair_v2_seed42/"
    "checkpoint_dd_a0_absolute_v2_init.pth"
)
A0_INITIALIZER_SHA256 = (
    "c2c4ba71656054d3afc3d219ca2f6d56839396d6258bac0201878566b1937034"
)
SOURCE_DATASET_CONFIG = (
    REPO_ROOT / "config/datasets_stageb_data_driven_dd1_new_head_train_20260723.json"
)
SOURCE_DATASET_CONFIG_SHA256 = (
    "76de77705b897bcd3d6bf6fa4cc2a6baa82b7499812f13fe316572df2e194b77"
)
PARTITION_RECEIPT_SHA256 = (
    "56de31d883ed137f3f9332c34de846839d82c0724120f42a49c5c1c302f38506"
)
SUPPORT_RECEIPT_SHA256 = (
    "a0e6632182bc7c01ac6e6997b15f1f96e0fbb0bf6dd9d1e3fd8485ad39a6da62"
)
OUTPUT_ROOT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/data_driven_role_routed_20260727/initializers/"
    "clean_dd1_u1000_lr3e4"
)
CHECKPOINT_NAME = "checkpoint_model_only.pth"
RECEIPT_NAME = "receipt.json"
RECEIPT_SCHEMA = (
    "pivot.stageb.data_driven.role_routed_initializer_receipt/v1"
)

RANK_PREFIX = "stage_b_data_driven_score_heads.rank_branch."
CONFIDENCE_PREFIXES = (
    "stage_b_data_driven_score_heads.confidence_branch.",
    "stage_b_data_driven_score_heads.confidence_gate.",
)
PATCH_KEYS = {
    "patch_logit_scale",
    "patch_encoder.input_proj.0.weight",
    "patch_encoder.input_proj.0.bias",
    "patch_encoder.input_proj.1.weight",
    "patch_encoder.input_proj.1.bias",
    "patch_encoder.norm.weight",
    "patch_encoder.norm.bias",
    "query_proj_for_patch.weight",
    "query_proj_for_patch.bias",
}


class RoleRoutedInitializerError(RuntimeError):
    pass


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, *, reported_path: Path | None = None) -> dict[str, Any]:
    requested = path.expanduser()
    if requested.is_symlink():
        raise RoleRoutedInitializerError(f"symlinks are forbidden: {requested}")
    resolved = requested.resolve(strict=True)
    if not resolved.is_file():
        raise RoleRoutedInitializerError(f"not a regular file: {resolved}")
    before = resolved.stat()
    digest = _sha256_file(resolved)
    after = resolved.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RoleRoutedInitializerError(f"file changed while hashing: {resolved}")
    return {
        "path": str((reported_path or resolved).expanduser().resolve()),
        "size_bytes": int(after.st_size),
        "sha256": digest,
    }


def _torch_load(path: Path) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise RoleRoutedInitializerError(f"checkpoint is not a mapping: {path}")
    return payload


def _validate_source(
    *,
    source_checkpoint: Path,
    source_checkpoint_sha256: str,
    a0_initializer: Path,
    a0_initializer_sha256: str,
    source_dataset_config: Path,
    source_dataset_config_sha256: str,
) -> tuple[Mapping[str, torch.Tensor], dict[str, Any]]:
    source_record = _file_record(source_checkpoint)
    a0_record = _file_record(a0_initializer)
    dataset_record = _file_record(source_dataset_config)
    expected_records = (
        (source_record, source_checkpoint_sha256, "source checkpoint"),
        (a0_record, a0_initializer_sha256, "A0 initializer"),
        (dataset_record, source_dataset_config_sha256, "source dataset config"),
    )
    for record, expected, label in expected_records:
        if record["sha256"] != expected:
            raise RoleRoutedInitializerError(
                f"{label} SHA drifted: {record['sha256']} != {expected}"
            )

    source = _torch_load(source_checkpoint)
    a0 = _torch_load(a0_initializer)
    state = source.get("model")
    a0_state = a0.get("model")
    a0_contract = a0.get("data_driven_initializer")
    if not (
        isinstance(state, Mapping)
        and isinstance(a0_state, Mapping)
        and isinstance(a0_contract, Mapping)
        and set(state) == set(a0_state)
        and source.get("optimizer_updates") == 1000
        and source.get("iteration") == 1000
        and source.get("epoch") == 0
        and source.get("epoch_finished") is False
        and source.get("checkpoint_reason") == "max_train_iters"
    ):
        raise RoleRoutedInitializerError("source checkpoint terminal contract drifted")
    for key, value in state.items():
        base = a0_state.get(key)
        if not (
            torch.is_tensor(value)
            and torch.is_tensor(base)
            and value.dtype == base.dtype
            and tuple(value.shape) == tuple(base.shape)
        ):
            raise RoleRoutedInitializerError(f"model tensor contract drifted: {key}")

    args = source.get("args")
    criterion = source.get("criterion")
    sampling = source.get("stage_b_data_driven_sampling_state")
    if not (
        isinstance(args, Mapping)
        and isinstance(criterion, Mapping)
        and torch.equal(criterion.get("criterion_contract_version"), torch.tensor(4))
        and torch.equal(criterion.get("rank_supervision_contract_id"), torch.tensor(1))
        and isinstance(sampling, Mapping)
        and sampling.get("schema") == "deterministic_epoch_ledger_v1"
        and sampling.get("dataset_size") == 263661
        and sampling.get("num_samples") == 263661
        and sampling.get("replacement") is True
        and sampling.get("weighted") is True
    ):
        raise RoleRoutedInitializerError("source supervision/sampling contract drifted")

    required_args = {
        "stage_b_data_driven_score": True,
        "stage_b_data_driven_experiment_id": "DD1",
        "stage_b_data_driven_train_mode": "rank_patch_only",
        "stage_b_data_driven_category_complete": True,
        "stage_b_data_driven_confidence_trained": False,
        "stage_b_data_driven_rank_supervision": "all_nonpositive_negative_v1",
        "stage_b_data_driven_rank_architecture": "absolute_token",
        "stage_b_data_driven_strict_sample_identity": True,
        "stage_b_data_driven_new_head_partition_receipt_sha256": (
            PARTITION_RECEIPT_SHA256
        ),
        "stage_b_data_driven_new_head_support_receipt_sha256": (
            SUPPORT_RECEIPT_SHA256
        ),
        "stage_b_data_driven_base_initializer_sha256": a0_initializer_sha256,
        "max_train_iters": 1000,
        "lr": 3e-4,
        "stage_b_data_driven_rank_lr": 3e-4,
        "stage_b_data_driven_patch_lr": 3e-4,
        "seed": 42,
    }
    drifted = {
        key: (args.get(key), expected)
        for key, expected in required_args.items()
        if args.get(key) != expected
    }
    if drifted:
        raise RoleRoutedInitializerError(f"source training args drifted: {drifted}")
    for route in (
        "stage_b",
        "stage_b_gdino_score_adapter",
        "stage_b_u0_patch_rank",
        "stage_b_v7",
        "stage_b_v11_fixed_text",
        "stage_b_legacy_global_gate",
    ):
        if args.get(route) is not False:
            raise RoleRoutedInitializerError(
                f"source checkpoint enabled forbidden teacher/legacy route: {route}"
            )
    if Path(str(args.get("datasets"))).resolve(strict=True) != Path(
        dataset_record["path"]
    ):
        raise RoleRoutedInitializerError("source dataset path drifted")
    provenance = args.get("stage_b_data_driven_training_provenance")
    assets = (
        provenance.get("dataset_asset_files")
        if isinstance(provenance, Mapping)
        else None
    )
    asset_shas = {
        item.get("sha256")
        for item in assets or []
        if isinstance(item, Mapping)
    }
    if not (
        isinstance(provenance, Mapping)
        and provenance.get("schema")
        == "pivot.stageb.data_driven_training_provenance/v1"
        and PARTITION_RECEIPT_SHA256 in asset_shas
        and SUPPORT_RECEIPT_SHA256 in asset_shas
    ):
        raise RoleRoutedInitializerError("source data provenance drifted")

    rank_keys = sorted(key for key in state if key.startswith(RANK_PREFIX))
    confidence_keys = sorted(
        key for key in state if key.startswith(CONFIDENCE_PREFIXES)
    )
    patch_keys = sorted(PATCH_KEYS)
    if set(patch_keys) - set(state) or not rank_keys or not confidence_keys:
        raise RoleRoutedInitializerError("initializer role key discovery failed")
    changed = sorted(key for key in state if not torch.equal(state[key], a0_state[key]))
    expected_changed = sorted({*rank_keys, *patch_keys})
    if changed != expected_changed:
        raise RoleRoutedInitializerError(
            "source changed tensors outside rank/patch surface: "
            f"{sorted(set(changed).symmetric_difference(expected_changed))}"
        )
    if any(not torch.equal(state[key], a0_state[key]) for key in confidence_keys):
        raise RoleRoutedInitializerError("source confidence branch changed from A0")

    unchanged_keys = sorted(set(state) - set(changed) - set(confidence_keys))
    role_keys = {
        "source_changed_rank": rank_keys,
        "source_changed_patch": patch_keys,
        "source_frozen_confidence": confidence_keys,
        "source_unchanged_other": unchanged_keys,
    }
    contract: dict[str, Any] = {
        "schema": DATA_DRIVEN_ROLE_ROUTED_INITIALIZER_SCHEMA,
        "architecture": {
            "rank_architecture": "absolute_token",
            "head_init_seed": 42,
            "enable_patch_branch": True,
        },
        "source_checkpoint": source_record,
        "source_a0_initializer": a0_record,
        "source_dataset_config": dataset_record,
        "source_optimizer_updates": 1000,
        "source_checkpoint_reason": "max_train_iters",
        "source_criterion_contract_version": 4,
        "source_rank_supervision_contract_id": 1,
        "source_rank_supervision": "all_nonpositive_negative_v1",
        "source_partition_receipt_sha256": PARTITION_RECEIPT_SHA256,
        "source_support_receipt_sha256": SUPPORT_RECEIPT_SHA256,
        "source_sampling_ledger_sha256": sampling.get("ledger_sha256"),
        "source_training_args_sha256": hashlib.sha256(_canonical_bytes(args)).hexdigest(),
        "source_training_provenance_sha256": hashlib.sha256(
            _canonical_bytes(provenance)
        ).hexdigest(),
        "role_keys": role_keys,
        "role_key_counts": {key: len(value) for key, value in role_keys.items()},
        "full_model_tensor_sha256": data_driven_tensor_state_sha256(
            state, sorted(state)
        ),
        "a0_full_model_tensor_sha256": data_driven_tensor_state_sha256(
            a0_state, sorted(a0_state)
        ),
        "invariants": {
            "source_is_clean_DD1_stage_b_data_only_u1000": True,
            "source_has_no_teacher_adapter_or_old_winner_tensor_route": True,
            "output_copies_only_source_model_tensors": True,
            "optimizer_scheduler_scaler_rng_and_old_criterion_are_excluded": True,
            "only_rank_patch_and_deployment_inert_scale_changed_from_A0": True,
            "confidence_and_backbone_remain_bitwise_A0": True,
            "role_routed_training_starts_with_fresh_optimizer_and_v5_criterion": True,
        },
    }
    for role, keys in role_keys.items():
        contract[f"{role}_tensor_sha256"] = data_driven_tensor_state_sha256(
            state, keys
        )
    return state, contract


def _receipt_bytes(receipt: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            receipt,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def build(
    *,
    source_checkpoint: Path = SOURCE_CHECKPOINT,
    source_checkpoint_sha256: str = SOURCE_CHECKPOINT_SHA256,
    a0_initializer: Path = A0_INITIALIZER,
    a0_initializer_sha256: str = A0_INITIALIZER_SHA256,
    source_dataset_config: Path = SOURCE_DATASET_CONFIG,
    source_dataset_config_sha256: str = SOURCE_DATASET_CONFIG_SHA256,
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    if os.path.lexists(output_root):
        raise RoleRoutedInitializerError(
            f"refusing to replace existing output root: {output_root}"
        )
    state, contract = _validate_source(
        source_checkpoint=source_checkpoint,
        source_checkpoint_sha256=source_checkpoint_sha256,
        a0_initializer=a0_initializer,
        a0_initializer_sha256=a0_initializer_sha256,
        source_dataset_config=source_dataset_config,
        source_dataset_config_sha256=source_dataset_config_sha256,
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    )
    committed = False
    try:
        checkpoint_path = temporary / CHECKPOINT_NAME
        torch.save({"model": state, "data_driven_role_routed_initializer": contract}, checkpoint_path)
        checkpoint_record = _file_record(
            checkpoint_path, reported_path=output_root / CHECKPOINT_NAME
        )
        receipt: dict[str, Any] = {
            "schema": RECEIPT_SCHEMA,
            "builder": _file_record(Path(__file__)),
            "checkpoint": checkpoint_record,
            "initializer_schema": DATA_DRIVEN_ROLE_ROUTED_INITIALIZER_SCHEMA,
            "initializer_contract_sha256": hashlib.sha256(
                _canonical_bytes(contract)
            ).hexdigest(),
            "source_checkpoint": contract["source_checkpoint"],
            "source_a0_initializer": contract["source_a0_initializer"],
            "source_dataset_config": contract["source_dataset_config"],
            "full_model_tensor_sha256": contract["full_model_tensor_sha256"],
            "invariants": dict(contract["invariants"]),
        }
        receipt["canonical_payload_sha256"] = hashlib.sha256(
            _canonical_bytes(receipt)
        ).hexdigest()
        receipt_path = temporary / RECEIPT_NAME
        receipt_path.write_bytes(_receipt_bytes(receipt))
        for path in (checkpoint_path, receipt_path):
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        _rename_directory_noreplace(temporary, output_root)
        committed = True
        return receipt
    finally:
        if not committed and temporary.exists():
            shutil.rmtree(temporary)


def verify(*, output_root: Path = OUTPUT_ROOT, **kwargs: Any) -> dict[str, Any]:
    requested = output_root.expanduser()
    if requested.is_symlink():
        raise RoleRoutedInitializerError("output root must not be a symlink")
    root = requested.resolve(strict=True)
    if {path.name for path in root.iterdir()} != {CHECKPOINT_NAME, RECEIPT_NAME}:
        raise RoleRoutedInitializerError("initializer output file set drifted")
    receipt_path = root / RECEIPT_NAME
    receipt = json.loads(receipt_path.read_text(encoding="ascii"))
    claimed = receipt.get("canonical_payload_sha256")
    canonical = dict(receipt)
    canonical.pop("canonical_payload_sha256", None)
    if hashlib.sha256(_canonical_bytes(canonical)).hexdigest() != claimed:
        raise RoleRoutedInitializerError("initializer receipt canonical hash drifted")
    source_kwargs = {
        "source_checkpoint": SOURCE_CHECKPOINT,
        "source_checkpoint_sha256": SOURCE_CHECKPOINT_SHA256,
        "a0_initializer": A0_INITIALIZER,
        "a0_initializer_sha256": A0_INITIALIZER_SHA256,
        "source_dataset_config": SOURCE_DATASET_CONFIG,
        "source_dataset_config_sha256": SOURCE_DATASET_CONFIG_SHA256,
    }
    source_kwargs.update(kwargs)
    state, expected_contract = _validate_source(**source_kwargs)
    payload = _torch_load(root / CHECKPOINT_NAME)
    if set(payload) != {"model", "data_driven_role_routed_initializer"}:
        raise RoleRoutedInitializerError("initializer payload keys drifted")
    observed_state = payload["model"]
    if set(observed_state) != set(state) or any(
        not torch.equal(observed_state[key], state[key]) for key in state
    ):
        raise RoleRoutedInitializerError("initializer model is not a bitwise source copy")
    if payload["data_driven_role_routed_initializer"] != expected_contract:
        raise RoleRoutedInitializerError("initializer embedded contract drifted")
    checkpoint_record = _file_record(root / CHECKPOINT_NAME)
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("checkpoint", {}).get("sha256")
        != checkpoint_record["sha256"]
        or receipt.get("checkpoint", {}).get("size_bytes")
        != checkpoint_record["size_bytes"]
        or receipt.get("initializer_contract_sha256")
        != hashlib.sha256(_canonical_bytes(expected_contract)).hexdigest()
    ):
        raise RoleRoutedInitializerError("initializer receipt binding drifted")
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = (
        verify(output_root=args.output_root)
        if args.verify
        else build(output_root=args.output_root)
    )
    print(json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
