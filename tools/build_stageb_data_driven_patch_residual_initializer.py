#!/usr/bin/env python3
"""Add a deterministic zero-output patch residual to the clean DD1 initializer."""

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

from models.GroundingDINO.stage_b_data_driven_patch_residual import (  # noqa: E402
    DATA_DRIVEN_PATCH_RESIDUAL_INITIALIZER_SCHEMA,
    DATA_DRIVEN_PATCH_TOPK_SEMANTIC_INITIALIZER_SCHEMA,
    StageBDataDrivenPatchResidualMatcher,
    StageBDataDrivenTopKPatchResidualMatcher,
    _tensor_state_sha256,
)
from tools.build_stageb_data_driven_role_routed_clean_assignment import (  # noqa: E402
    _rename_directory_noreplace,
)


BASE_INITIALIZER = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/data_driven_role_routed_20260727/initializers/"
    "clean_dd1_u1000_lr3e4/checkpoint_model_only.pth"
)
BASE_INITIALIZER_SHA256 = (
    "5ae688008cf56130c69c152197911fa61fecb6a24956f425fdd5a7ac42e97bd1"
)
BASE_RECEIPT = BASE_INITIALIZER.parent / "receipt.json"
BASE_RECEIPT_SHA256 = (
    "5e4ed2e0730e3710300dd3dfdec44e5f56bc5082aba49e1c6f471039caba3f32"
)
OUTPUT_ROOT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/data_driven_role_routed_20260727/initializers/"
    "clean_dd1_u1000_lr3e4_patch_residual128_seed42_v2"
)
CHECKPOINT_NAME = "checkpoint_model_only.pth"
RECEIPT_NAME = "receipt.json"
RECEIPT_SCHEMA = (
    "pivot.stageb.data_driven.patch_residual_initializer_receipt/v1"
)
TOPK_RECEIPT_SCHEMA = (
    "pivot.stageb.data_driven.patch_residual_initializer_receipt/v2"
)
FEATURE_DIM = 256
HIDDEN_DIM = 128
RESIDUAL_LIMIT = 0.25
INIT_SEED = 42
RESIDUAL_PREFIX = "stage_b_data_driven_patch_residual."


class PatchResidualInitializerError(RuntimeError):
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
        raise PatchResidualInitializerError(f"symlinks are forbidden: {requested}")
    resolved = requested.resolve(strict=True)
    if not resolved.is_file():
        raise PatchResidualInitializerError(f"not a regular file: {resolved}")
    before = resolved.stat()
    digest = _sha256_file(resolved)
    after = resolved.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise PatchResidualInitializerError(f"file changed while hashing: {resolved}")
    return {
        "path": str((reported_path or resolved).expanduser().resolve()),
        "size_bytes": int(after.st_size),
        "sha256": digest,
    }


def _torch_load(path: Path) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise PatchResidualInitializerError(f"checkpoint is not a mapping: {path}")
    return payload


def _build_state(
    *, base_initializer: Path, base_initializer_sha256: str, base_receipt: Path,
    base_receipt_sha256: str,
    center_raw: bool,
    topk_semantic: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    base_record = _file_record(base_initializer)
    receipt_record = _file_record(base_receipt)
    if base_record["sha256"] != base_initializer_sha256:
        raise PatchResidualInitializerError("base initializer SHA drifted")
    if receipt_record["sha256"] != base_receipt_sha256:
        raise PatchResidualInitializerError("base receipt SHA drifted")
    payload = _torch_load(base_initializer)
    if set(payload) != {"model", "data_driven_role_routed_initializer"}:
        raise PatchResidualInitializerError("base initializer payload drifted")
    base_state = payload["model"]
    base_contract = payload["data_driven_role_routed_initializer"]
    if not isinstance(base_state, Mapping) or not isinstance(base_contract, Mapping):
        raise PatchResidualInitializerError("base initializer is malformed")
    if any(not torch.is_tensor(value) for value in base_state.values()):
        raise PatchResidualInitializerError("base model contains a non-tensor value")

    receipt = json.loads(base_receipt.read_text(encoding="ascii"))
    claimed = receipt.get("canonical_payload_sha256")
    canonical = dict(receipt)
    canonical.pop("canonical_payload_sha256", None)
    if hashlib.sha256(_canonical_bytes(canonical)).hexdigest() != claimed:
        raise PatchResidualInitializerError("base receipt canonical hash drifted")
    if receipt.get("checkpoint", {}).get("sha256") != base_record["sha256"]:
        raise PatchResidualInitializerError("base receipt/checkpoint binding drifted")

    if topk_semantic:
        if not center_raw:
            raise PatchResidualInitializerError(
                "top-k semantic residual requires raw query centering"
            )
        matcher = StageBDataDrivenTopKPatchResidualMatcher(
            feature_dim=FEATURE_DIM,
            hidden_dim=HIDDEN_DIM,
            context_dim=16,
            topk=10,
            residual_limit=RESIDUAL_LIMIT,
            init_seed=INIT_SEED,
        )
    else:
        matcher = StageBDataDrivenPatchResidualMatcher(
            feature_dim=FEATURE_DIM,
            hidden_dim=HIDDEN_DIM,
            residual_limit=RESIDUAL_LIMIT,
            init_seed=INIT_SEED,
            center_raw=center_raw,
        )
    residual_state = {
        RESIDUAL_PREFIX + key: value.detach().cpu().clone()
        for key, value in matcher.state_dict().items()
    }
    architecture = matcher.architecture()
    if (
        len(residual_state) != int(architecture["trainable_tensors"])
        or sum(value.numel() for value in residual_state.values())
        != int(architecture["trainable_parameters"])
    ):
        raise PatchResidualInitializerError("residual parameter surface drifted")
    zero_output_keys = [RESIDUAL_PREFIX + "output.weight"]
    if topk_semantic:
        zero_output_keys.append(RESIDUAL_PREFIX + "context_output.weight")
    if not all(
        bool((residual_state[key] == 0).all()) for key in zero_output_keys
    ):
        raise PatchResidualInitializerError("residual output is not exactly zero")
    if set(base_state) & set(residual_state):
        raise PatchResidualInitializerError("residual keys collide with base model")
    # Preserve the source checkpoint's shared-backbone storage aliases so the
    # additive initializer does not duplicate hundreds of megabytes on disk.
    state = dict(base_state)
    state.update(residual_state)
    base_keys = sorted(base_state)
    residual_keys = sorted(residual_state)
    contract = {
        "schema": (
            DATA_DRIVEN_PATCH_TOPK_SEMANTIC_INITIALIZER_SCHEMA
            if topk_semantic
            else DATA_DRIVEN_PATCH_RESIDUAL_INITIALIZER_SCHEMA
        ),
        "architecture": architecture,
        "source_role_routed_initializer": base_record,
        "source_role_routed_receipt": receipt_record,
        "source_checkpoint": base_contract["source_checkpoint"],
        "source_a0_initializer": base_contract["source_a0_initializer"],
        "source_optimizer_updates": base_contract["source_optimizer_updates"],
        "base_key_count": len(base_keys),
        "residual_keys": residual_keys,
        "base_tensor_sha256": _tensor_state_sha256(state, base_keys),
        "residual_tensor_sha256": _tensor_state_sha256(state, residual_keys),
        "full_model_tensor_sha256": _tensor_state_sha256(state, sorted(state)),
        "invariants": {
            "base_model_tensors_are_bitwise_source_copy": True,
            "residual_output_is_exactly_zero_initialized": True,
            "initializer_contains_no_optimizer_criterion_scaler_or_rng": True,
            "no_teacher_or_old_winner_tensor_added": True,
            "formal_load_requires_exact_model_key_coverage": True,
            **(
                {"residual_raw_is_query_mean_centered_before_tanh": True}
                if center_raw
                else {}
            ),
            **(
                {
                    "context_output_is_exactly_zero_initialized": True,
                    "topk_context_uses_only_inference_available_detached_inputs": True,
                    "single_and_multi_patch_share_the_same_query_set_contract": True,
                }
                if topk_semantic
                else {}
            ),
        },
    }
    return state, contract


def _receipt_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")


def build(
    *,
    base_initializer: Path = BASE_INITIALIZER,
    base_initializer_sha256: str = BASE_INITIALIZER_SHA256,
    base_receipt: Path = BASE_RECEIPT,
    base_receipt_sha256: str = BASE_RECEIPT_SHA256,
    output_root: Path = OUTPUT_ROOT,
    center_raw: bool = False,
    topk_semantic: bool = False,
) -> dict[str, Any]:
    output_root = output_root.expanduser().resolve()
    if os.path.lexists(output_root):
        raise PatchResidualInitializerError(
            f"refusing to replace existing output root: {output_root}"
        )
    state, contract = _build_state(
        base_initializer=base_initializer,
        base_initializer_sha256=base_initializer_sha256,
        base_receipt=base_receipt,
        base_receipt_sha256=base_receipt_sha256,
        center_raw=center_raw,
        topk_semantic=topk_semantic,
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    )
    committed = False
    try:
        checkpoint_path = temporary / CHECKPOINT_NAME
        torch.save(
            {"model": state, "data_driven_patch_residual_initializer": contract},
            checkpoint_path,
        )
        checkpoint_record = _file_record(
            checkpoint_path, reported_path=output_root / CHECKPOINT_NAME
        )
        receipt = {
            "schema": TOPK_RECEIPT_SCHEMA if topk_semantic else RECEIPT_SCHEMA,
            "builder": _file_record(Path(__file__)),
            "checkpoint": checkpoint_record,
            "architecture": contract["architecture"],
            "source_role_routed_initializer": contract[
                "source_role_routed_initializer"
            ],
            "source_role_routed_receipt": contract["source_role_routed_receipt"],
            "source_checkpoint": contract["source_checkpoint"],
            "source_a0_initializer": contract["source_a0_initializer"],
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


def verify(
    *,
    output_root: Path = OUTPUT_ROOT,
    center_raw: bool = False,
    topk_semantic: bool = False,
) -> dict[str, Any]:
    requested = output_root.expanduser()
    if requested.is_symlink():
        raise PatchResidualInitializerError("output root must not be a symlink")
    root = requested.resolve(strict=True)
    if {path.name for path in root.iterdir()} != {CHECKPOINT_NAME, RECEIPT_NAME}:
        raise PatchResidualInitializerError("initializer output files drifted")
    receipt = json.loads((root / RECEIPT_NAME).read_text(encoding="ascii"))
    claimed = receipt.get("canonical_payload_sha256")
    canonical = dict(receipt)
    canonical.pop("canonical_payload_sha256", None)
    if hashlib.sha256(_canonical_bytes(canonical)).hexdigest() != claimed:
        raise PatchResidualInitializerError("receipt canonical hash drifted")
    expected_state, expected_contract = _build_state(
        base_initializer=BASE_INITIALIZER,
        base_initializer_sha256=BASE_INITIALIZER_SHA256,
        base_receipt=BASE_RECEIPT,
        base_receipt_sha256=BASE_RECEIPT_SHA256,
        center_raw=center_raw,
        topk_semantic=topk_semantic,
    )
    payload = _torch_load(root / CHECKPOINT_NAME)
    if set(payload) != {"model", "data_driven_patch_residual_initializer"}:
        raise PatchResidualInitializerError("initializer payload keys drifted")
    state = payload["model"]
    if set(state) != set(expected_state) or any(
        not torch.equal(state[key], expected_state[key]) for key in expected_state
    ):
        raise PatchResidualInitializerError("initializer tensors drifted")
    if payload["data_driven_patch_residual_initializer"] != expected_contract:
        raise PatchResidualInitializerError("embedded initializer contract drifted")
    checkpoint_record = _file_record(root / CHECKPOINT_NAME)
    if (
        receipt.get("schema")
        != (TOPK_RECEIPT_SCHEMA if topk_semantic else RECEIPT_SCHEMA)
        or receipt.get("checkpoint", {}).get("sha256")
        != checkpoint_record["sha256"]
    ):
        raise PatchResidualInitializerError("receipt checkpoint binding drifted")
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--center-raw", action="store_true")
    parser.add_argument("--topk-semantic", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = (
        verify(
            output_root=args.output_root,
            center_raw=args.center_raw,
            topk_semantic=args.topk_semantic,
        )
        if args.verify
        else build(
            output_root=args.output_root,
            center_raw=args.center_raw,
            topk_semantic=args.topk_semantic,
        )
    )
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
