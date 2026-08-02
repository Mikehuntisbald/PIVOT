#!/usr/bin/env python3
"""Build and verify one D9-patch + R100-rank + P50-confidence checkpoint.

The artifact uses no Stage-A patch tensor.  It starts from the sealed U0
single-model layout only because that layout already contains the independently
Stage-B-trained R100/P50 adapters, then replaces all nine historical Stage-A
patch-specific tensors with the b58-native, Stage-B-trained D9 tensors.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.GroundingDINO.stage_b_u0_patch_rank import (  # noqa: E402
    U0_PATCH_BACKBONE_PREFIX,
    U0_PATCH_SOURCE_KEYS,
    stage_b_u0_tensor_state_sha256,
    validate_stage_b_u0_initializer_payload,
)
from tools.build_stageb_legacy_replay_receipt import (  # noqa: E402
    SCHEMA as LEGACY_RECEIPT_SCHEMA,
    canonical_json_sha256,
    verify_receipt,
)
from tools.build_stageb_native_patch_category_initializer import (  # noqa: E402
    NATIVE_PATCH_CATEGORY_INITIALIZER_SCHEMA,
    RANDOM_TRAINABLE_PATCH_KEYS,
    _safe_load_checkpoint,
    stable_file_record,
    validate_native_patch_category_initializer_payload,
)
from tools.stageb_dependency_audit import config_import_chain  # noqa: E402
from tools.stageb_native_patch_category_d2_contract import (  # noqa: E402
    audit_d2_source_transition,
)
from util.slconfig import SLConfig  # noqa: E402


SCHEMA = "pivot.stageb.data_only_composite/v1"
CONTRACT_KEY = "data_only_composite"
D9_GATE_SELECTION_SCHEMA = "pivot.stageb.data_only_category_gate_selection/v1"
PATCH_KEYS = frozenset(U0_PATCH_SOURCE_KEYS)
PATCH_BACKBONE_PREFIX = U0_PATCH_BACKBONE_PREFIX
U0_ADAPTER_PREFIX = "stage_b_u0_patch_rank_adapter."

DEFAULT_CONFIG = (
    REPO_ROOT
    / "config/ablations/cfg_stageb_data_only_composite_d9_r100p50_gap3.py"
)
DEFAULT_U0 = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/u0_single_network_seed42_b56_v1/initializer/"
    "checkpoint_u0_init.pth"
)
DEFAULT_D9 = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/"
    "native_patch_category_d9_seed42_s43_b36a2_lr5e5_amp8_u100_v1/"
    "checkpoint_iter.pth"
)
DEFAULT_D1 = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/"
    "native_patch_category_d1_seed42_b36a2_lr3e4_u500_v1/"
    "checkpoint_iter.pth"
)
DEFAULT_D1_INITIALIZER = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/native_patch_category_initializers/d1_seed42/"
    "checkpoint_d1_init.pth"
)
DEFAULT_LEGACY_RECEIPT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/legacy_replay/"
    "legacy_r100_p50_exact_replay_receipt.json"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/data_only_composite_d9_r100p50_gap3_v1/"
    "checkpoint_eval_only.pth"
)

EXPECTED_FILE_SHA256 = {
    "u0_initializer": "c89e5dfba795fd8074a044f0c09d81c871705c20a1dbf819b9f16c770a2cba43",
    "d9_patch": "604c9b1977449badc6e5fe4eb7dc1cfd80e53959ddbc6c6e5f8b7809aab84278",
    "d1_source": "ac8b29a8d8a5e5bb8877a7c21769ff08c0b5ca805c522f80549dfc99f55c5dc5",
    "d1_initializer": "addec47338c2e36a3121d999370349d2351535f6ff7334729424aeb1bcd880b4",
    "legacy_receipt": "f8bd5104960eca19b90353168577a126a5577ea9aac511c6b0e3ab01b4bf2bfc",
}
EXPECTED_LEGACY_SELF_SHA256 = (
    "7979cea6caf551889b0fd7cf2bc334294cb4152ec0ef3b5f4e069acb7ce606a6"
)
EXPECTED_LEGACY_FILES = {
    "b58": "b58e5209dc07dbffb2e5ed3d792e0db70c3306ec2ed24389693b4aeeebab1157",
    "rank_r100": "a933725aab2226d35aa1cd94992c154b332bbbbc3406060bbda061cef4959dd5",
    "confidence_p50": "3e7bcb976b3dc569b69eb20ad2b71ffac25c626a6cb6ff127e7f8a7eb4c87ac5",
    "merged": "d5f133b8926470e4a21fbb34bb8e19863947107f38cf9d8cf56ca4530422ba44",
}
EXPECTED_MERGED_TENSOR_SHA256 = (
    "7c5c25a4c52b4fc469b9ecb3301713bb1383b0af6ab03fe9cbd46fe5af221555"
)


class DataOnlyCompositeError(RuntimeError):
    """The requested artifact violates the closed data-only contract."""


def _require_record(path: Path, expected_sha256: str, *, label: str) -> dict[str, Any]:
    record = stable_file_record(path, label=label)
    if record["sha256"] != expected_sha256:
        raise DataOnlyCompositeError(
            f"{label} SHA256 mismatch: expected={expected_sha256}, "
            f"observed={record['sha256']}"
        )
    return record


def _model_state(
    payload: Mapping[str, Any], *, label: str
) -> Mapping[str, torch.Tensor]:
    state = payload.get("model")
    if not isinstance(state, Mapping) or not state:
        raise DataOnlyCompositeError(f"{label} has no model mapping")
    invalid = [
        key
        for key, value in state.items()
        if not isinstance(key, str) or not torch.is_tensor(value)
    ]
    if invalid:
        raise DataOnlyCompositeError(
            f"{label} contains non-tensor model values: {invalid[:8]}"
        )
    return state


def _args(payload: Mapping[str, Any], *, label: str) -> Mapping[str, Any]:
    value = payload.get("args")
    if hasattr(value, "__dict__"):
        value = vars(value)
    if not isinstance(value, Mapping):
        raise DataOnlyCompositeError(f"{label} has no saved argument mapping")
    return value


def _finite_state(state: Mapping[str, torch.Tensor], *, label: str) -> None:
    for key, value in state.items():
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all().item()
        ):
            raise DataOnlyCompositeError(f"{label} has a non-finite tensor at {key}")


def _same_shape_dtype(
    value: torch.Tensor, wanted: torch.Tensor, *, key: str
) -> None:
    if value.dtype != wanted.dtype or tuple(value.shape) != tuple(wanted.shape):
        raise DataOnlyCompositeError(
            f"shape/dtype mismatch at {key}: "
            f"{tuple(value.shape)}/{value.dtype} != "
            f"{tuple(wanted.shape)}/{wanted.dtype}"
        )


def compose_data_only_model_state(
    u0_state: Mapping[str, torch.Tensor],
    d9_state: Mapping[str, torch.Tensor],
) -> tuple[OrderedDict[str, torch.Tensor], dict[str, list[str]]]:
    """Replace exactly U0's nine Stage-A patch tensors with D9 tensors."""
    u0_keys = set(u0_state)
    d9_keys = set(d9_state)
    if not d9_keys or not d9_keys.issubset(u0_keys):
        raise DataOnlyCompositeError(
            "D9 coverage is not a non-empty exact subset of U0 "
            f"(D9-only={sorted(d9_keys - u0_keys)[:8]})"
        )
    if not PATCH_KEYS.issubset(d9_keys):
        raise DataOnlyCompositeError(
            f"D9 is missing patch keys: {sorted(PATCH_KEYS - d9_keys)}"
        )
    shared_differences: list[str] = []
    for key in sorted(d9_keys):
        _same_shape_dtype(d9_state[key], u0_state[key], key=key)
        if not torch.equal(d9_state[key], u0_state[key]):
            shared_differences.append(key)
    if set(shared_differences) != set(PATCH_KEYS):
        raise DataOnlyCompositeError(
            "U0/D9 shared-state difference must be exactly the nine patch keys; "
            f"observed={shared_differences}"
        )

    result: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in u0_state.items():
        source = d9_state[key] if key in PATCH_KEYS else value
        result[key] = source.detach().cpu().clone()

    roles = {
        "d9_patch": sorted(PATCH_KEYS),
        "u0_preserved": sorted(u0_keys - PATCH_KEYS),
    }
    return result, roles


def _read_json(path: Path, *, label: str) -> MutableMapping[str, Any]:
    try:
        value = json.loads(path.resolve(strict=True).read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DataOnlyCompositeError(f"could not read {label}: {error}") from error
    if not isinstance(value, MutableMapping):
        raise DataOnlyCompositeError(f"{label} must be a JSON object")
    return value


def _validate_legacy_receipt(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    record = _require_record(
        path, EXPECTED_FILE_SHA256["legacy_receipt"], label="legacy replay receipt"
    )
    try:
        replay = verify_receipt(path)
    except Exception as error:
        raise DataOnlyCompositeError(f"legacy replay verification failed: {error}") from error
    if replay.get("receipt_sha256") != EXPECTED_LEGACY_SELF_SHA256:
        raise DataOnlyCompositeError("legacy replay receipt self-hash drifted")

    receipt = _read_json(path, label="legacy replay receipt")
    sealed = dict(receipt)
    observed_self_hash = sealed.pop("receipt_sha256", None)
    if (
        receipt.get("schema") != LEGACY_RECEIPT_SCHEMA
        or observed_self_hash != canonical_json_sha256(sealed)
    ):
        raise DataOnlyCompositeError("legacy replay receipt seal drifted")
    checkpoints = receipt.get("checkpoints")
    invariants = receipt.get("invariants")
    if not isinstance(checkpoints, Mapping) or not isinstance(invariants, Mapping):
        raise DataOnlyCompositeError("legacy replay receipt is malformed")
    for role, expected in EXPECTED_LEGACY_FILES.items():
        try:
            observed = checkpoints[role]["file"]["sha256"]
        except (KeyError, TypeError) as error:
            raise DataOnlyCompositeError(
                f"legacy receipt role {role} is malformed"
            ) from error
        if observed != expected:
            raise DataOnlyCompositeError(f"legacy receipt {role} file hash drifted")
    required_invariants = {
        "b58_has_no_adapter": True,
        "rank_r100_iteration_and_updates": 100,
        "confidence_p50_iteration_and_updates": 50,
        "rank_r100_rank_trained_confidence_untouched": True,
        "confidence_p50_rank_untouched_confidence_trained": True,
        "b58_rank_confidence_shared_base_bitwise": True,
        "merged_present": True,
        "merged_base_matches_b58_bitwise": True,
        "merged_rank_matches_rank_r100_bitwise": True,
        "merged_confidence_matches_confidence_p50_bitwise": True,
        "merged_contract_hashes_recomputed_equal": True,
    }
    if dict(invariants) != required_invariants:
        raise DataOnlyCompositeError("legacy replay invariant contract drifted")
    if (
        checkpoints["merged"]["model"].get("full_model_tensor_sha256")
        != EXPECTED_MERGED_TENSOR_SHA256
    ):
        raise DataOnlyCompositeError("legacy merged tensor hash drifted")

    rank_args = checkpoints["rank_r100"]["training"].get("args_summary")
    confidence_args = checkpoints["confidence_p50"]["training"].get(
        "args_summary"
    )
    required_rank = {
        "config_file": "config/ablations/cfg_stageb_gdino_score_adapter_rank_three_ref.py",
        "datasets": "config/datasets_stageb_gdino_adapter_rank_three_ref.json",
        "stage_b_gdino_adapter_train_mode": "rank_only",
        "stage_b_gdino_rank_weight": 1.0,
        "stage_b_gdino_confidence_weight": 0.0,
        "max_train_iters": 100,
        "pretrain_model_path": None,
    }
    required_confidence = {
        "config_file": "config/ablations/cfg_stageb_gdino_score_adapter_semantic_verified.py",
        "datasets": "config/datasets_stageb_gdino_adapter_semantic_verified_pairs.json",
        "stage_b_gdino_adapter_train_mode": "confidence_only",
        "stage_b_gdino_rank_weight": 0.0,
        "stage_b_gdino_confidence_weight": 1.0,
        "stage_b_gdino_tn_scope": "image_global_topk_verified",
        "max_train_iters": 50,
        "pretrain_model_path": None,
    }
    for label, observed, required in (
        ("R100", rank_args, required_rank),
        ("P50", confidence_args, required_confidence),
    ):
        if not isinstance(observed, Mapping) or any(
            observed.get(key) != value for key, value in required.items()
        ):
            raise DataOnlyCompositeError(f"{label} Stage-B training contract drifted")
    return record, {
        "receipt_self_sha256": observed_self_hash,
        "merged_tensor_sha256": EXPECTED_MERGED_TENSOR_SHA256,
        "rank_optimizer_updates": 100,
        "confidence_optimizer_updates": 50,
        "rank_supervision": "stage_b_three_ref",
        "confidence_supervision": "stage_b_traceable_semantic_tn",
        "confidence_tn_scope": "image_global_topk_verified",
        "no_external_teacher_logits_or_scores": True,
    }


def _validate_d9_lineage(
    *,
    d9_payload: Mapping[str, Any],
    d1_payload: Mapping[str, Any],
    initializer_payload: Mapping[str, Any],
) -> dict[str, Any]:
    d9_state = _model_state(d9_payload, label="D9 checkpoint")
    d1_state = _model_state(d1_payload, label="D1 checkpoint")
    initializer_state = _model_state(
        initializer_payload, label="D1 b58-native initializer"
    )
    validate_native_patch_category_initializer_payload(
        d9_state,
        initializer_payload,
        checkpoint_label="D9 base D1 initializer",
    )
    source_audit = audit_d2_source_transition(
        d9_state, initializer_payload, d1_payload, expected_optimizer_updates=500
    )
    if set(d9_state) != set(d1_state) or set(d1_state) != set(initializer_state):
        raise DataOnlyCompositeError("D9/D1/initializer model coverage drifted")

    changed = []
    for key in sorted(d9_state):
        _same_shape_dtype(d9_state[key], d1_state[key], key=key)
        if not torch.equal(d9_state[key], d1_state[key]):
            changed.append(key)
    if set(changed) != set(RANDOM_TRAINABLE_PATCH_KEYS):
        raise DataOnlyCompositeError(
            "D9 must change exactly the eight trainable patch projections from D1; "
            f"observed={changed}"
        )
    _finite_state(d9_state, label="D9 checkpoint")

    args = _args(d9_payload, label="D9 checkpoint")
    required_args = {
        "stage_b_native_patch_contract_version": 9,
        "stage_b_native_patch_objective": "d9_loss_gradient_localized",
        "stage_b_native_patch_d9_detach_row_stats": True,
        "stage_b_native_patch_execution_scope": "native_patch_category_d9_u100_v1",
        "max_train_iters": 100,
        "gradient_accumulation_steps": 2,
        "batch_size": 36,
        "seed": 42,
        "stage_b_data_driven_sampler_seed": 43,
        "stage_b_native_patch_d9_base_initializer_sha256": EXPECTED_FILE_SHA256[
            "d1_initializer"
        ],
        "stage_b_native_patch_initializer_sha256": EXPECTED_FILE_SHA256[
            "d1_source"
        ],
    }
    if any(args.get(key) != value for key, value in required_args.items()):
        drift = {
            key: args.get(key)
            for key, value in required_args.items()
            if args.get(key) != value
        }
        raise DataOnlyCompositeError(f"D9 saved-argument contract drifted: {drift}")
    required_position = {
        "optimizer_updates": 100,
        "iteration": 200,
        "epoch": 0,
        "epoch_finished": False,
        "checkpoint_reason": "max_train_iters",
    }
    if any(d9_payload.get(key) != value for key, value in required_position.items()):
        raise DataOnlyCompositeError("D9 training-position contract drifted")
    saved_source_audit = args.get("stage_b_native_patch_d9_source_audit")
    if not isinstance(saved_source_audit, Mapping) or any(
        saved_source_audit.get(key) != value
        for key, value in source_audit.items()
        if key != "schema"
    ):
        raise DataOnlyCompositeError("D9 embedded D1 source audit drifted")
    if saved_source_audit.get("no_teacher_u2_r100_p50_stagea_adapter_tensors") is not True:
        raise DataOnlyCompositeError("D9 source audit permits a forbidden teacher")
    return {
        "base_initializer_schema": NATIVE_PATCH_CATEGORY_INITIALIZER_SCHEMA,
        "d1_optimizer_updates": 500,
        "d9_optimizer_updates": 100,
        "d9_iterations": 200,
        "changed_tensor_keys_from_d1": changed,
        "frozen_tensor_count_bitwise_d1": len(d9_state) - len(changed),
        "detach_row_stats": True,
        "objective": "d9_loss_gradient_localized",
        "no_teacher_u2_r100_p50_stagea_adapter_tensors": True,
    }


def _config_binding(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = SLConfig.fromfile(str(path.resolve(strict=True)))
    required = {
        "stage_b_data_only_composite": True,
        "stage_b_data_only_composite_contract_version": 1,
        "stage_b_u0_patch_rank": True,
        "stage_b_gdino_score_adapter": True,
        "enable_patch_branch": True,
        "patch_gate_with_text": False,
        "stage_b_u0_category_preserving_patch_gate": True,
        "stage_b_u0_category_gate_max_gap": 3.0,
        "stage_b_u2_category_complete_supervision": False,
    }
    if any(getattr(cfg, key, None) != value for key, value in required.items()):
        raise DataOnlyCompositeError("data-only composite config contract drifted")
    chain = config_import_chain(path.resolve(), root=REPO_ROOT)
    forbidden = [item for item in chain if "u2_" in item.name or "stagea" in item.name]
    if forbidden:
        raise DataOnlyCompositeError(
            f"data-only config imports a U2/Stage-A dependency: {forbidden}"
        )
    binding = {
        "leaf": stable_file_record(path, label="data-only composite config"),
        "import_chain": [
            stable_file_record(item, label=f"config dependency {item.name}")
            for item in chain
        ],
    }
    return binding, required


def _source_payloads(
    *,
    u0_path: Path,
    d9_path: Path,
    d1_path: Path,
    d1_initializer_path: Path,
    legacy_receipt_path: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, Mapping[str, Any]], dict[str, Any]]:
    records = {
        "u0_initializer": _require_record(
            u0_path, EXPECTED_FILE_SHA256["u0_initializer"], label="U0 initializer"
        ),
        "d9_patch": _require_record(
            d9_path, EXPECTED_FILE_SHA256["d9_patch"], label="D9 patch checkpoint"
        ),
        "d1_source": _require_record(
            d1_path, EXPECTED_FILE_SHA256["d1_source"], label="D1 source checkpoint"
        ),
        "d1_initializer": _require_record(
            d1_initializer_path,
            EXPECTED_FILE_SHA256["d1_initializer"],
            label="D1 b58-native initializer",
        ),
    }
    legacy_record, legacy_lineage = _validate_legacy_receipt(legacy_receipt_path)
    records["legacy_receipt"] = legacy_record
    payloads = {
        "u0_initializer": _safe_load_checkpoint(u0_path, label="U0 initializer"),
        "d9_patch": _safe_load_checkpoint(d9_path, label="D9 patch checkpoint"),
        "d1_source": _safe_load_checkpoint(d1_path, label="D1 source checkpoint"),
        "d1_initializer": _safe_load_checkpoint(
            d1_initializer_path, label="D1 b58-native initializer"
        ),
    }
    return records, payloads, legacy_lineage


def build_payload(
    *,
    config: Path = DEFAULT_CONFIG,
    u0_path: Path = DEFAULT_U0,
    d9_path: Path = DEFAULT_D9,
    d1_path: Path = DEFAULT_D1,
    d1_initializer_path: Path = DEFAULT_D1_INITIALIZER,
    legacy_receipt_path: Path = DEFAULT_LEGACY_RECEIPT,
) -> dict[str, Any]:
    config_binding, config_contract = _config_binding(config)
    records, payloads, legacy_lineage = _source_payloads(
        u0_path=u0_path,
        d9_path=d9_path,
        d1_path=d1_path,
        d1_initializer_path=d1_initializer_path,
        legacy_receipt_path=legacy_receipt_path,
    )
    u0_payload = payloads["u0_initializer"]
    d9_payload = payloads["d9_patch"]
    u0_state = _model_state(u0_payload, label="U0 initializer")
    d9_state = _model_state(d9_payload, label="D9 checkpoint")
    validate_stage_b_u0_initializer_payload(
        u0_state, u0_payload, checkpoint_label="data-only U0 layout source"
    )
    u0_contract = u0_payload["u0_initializer"]
    if (
        u0_contract.get("merged_teacher_tensor_sha256")
        != legacy_lineage["merged_tensor_sha256"]
    ):
        raise DataOnlyCompositeError("U0 R100/P50 tensor lineage drifted")
    d9_lineage = _validate_d9_lineage(
        d9_payload=d9_payload,
        d1_payload=payloads["d1_source"],
        initializer_payload=payloads["d1_initializer"],
    )
    state, simple_roles = compose_data_only_model_state(u0_state, d9_state)

    u0_roles = u0_contract.get("role_keys")
    if not isinstance(u0_roles, Mapping):
        raise DataOnlyCompositeError("U0 role contract is missing")
    roles = {
        "legacy_r100_p50": list(u0_roles["merged"]),
        "shared_backbone_alias": list(u0_roles["shared_backbone_alias"]),
        "d9_patch": list(u0_roles["stagea_patch"]),
        "u0_adapter": list(u0_roles["u0_zero"]),
    }
    if set(roles["d9_patch"]) != set(simple_roles["d9_patch"]):
        raise DataOnlyCompositeError("U0/D9 patch role mismatch")
    contract: dict[str, Any] = {
        "schema": SCHEMA,
        "eval_only": True,
        "resumable": False,
        "single_checkpoint": True,
        "single_model_root": True,
        "external_score_source_required": False,
        "sources": records,
        "config": config_binding,
        "config_contract": config_contract,
        "model_state_keys": len(state),
        "role_key_counts": {key: len(value) for key, value in roles.items()},
        "role_keys": roles,
        "lineage": {"legacy_r100_p50": legacy_lineage, "d9_patch": d9_lineage},
        "discarded_u0_stagea_patch_tensor_sha256": u0_contract[
            "stagea_patch_tensor_sha256"
        ],
        "invariants": {
            "all_nine_u0_stagea_patch_tensors_replaced": True,
            "no_stagea_patch_tensor_remains": True,
            "legacy_r100_p50_preserved_bitwise": True,
            "shared_b58_patch_backbone_preserved_bitwise": True,
            "u0_adapter_preserved_bitwise": True,
            "d9_patch_copied_bitwise": True,
            "d9_is_b58_native_and_stageb_trained": True,
            "r100_p50_are_independently_stageb_trained": True,
            "rank_and_confidence_remain_decoupled": True,
            "no_external_teacher_logits_or_scores": True,
        },
    }
    for role, keys in roles.items():
        contract[f"{role}_tensor_sha256"] = stage_b_u0_tensor_state_sha256(
            state, keys
        )
    contract["full_model_tensor_sha256"] = stage_b_u0_tensor_state_sha256(
        state, state.keys()
    )
    payload = {"model": state, CONTRACT_KEY: contract}
    validate_data_only_composite_payload(
        state, payload, checkpoint_label="in-memory data-only composite"
    )
    del payloads
    gc.collect()
    return payload


def validate_data_only_composite_payload(
    expected_model_or_state: nn.Module | Mapping[str, torch.Tensor],
    payload: Mapping[str, Any],
    *,
    checkpoint_label: str,
) -> None:
    """Fast, closed-world validator used by the official evaluators."""
    expected_state = (
        expected_model_or_state.state_dict()
        if isinstance(expected_model_or_state, nn.Module)
        else expected_model_or_state
    )
    if not isinstance(payload, Mapping) or set(payload) != {"model", CONTRACT_KEY}:
        raise ValueError(f"{checkpoint_label}: composite top-level keys drifted")
    state = payload.get("model")
    contract = payload.get(CONTRACT_KEY)
    if not isinstance(state, Mapping) or not isinstance(contract, Mapping):
        raise ValueError(f"{checkpoint_label}: composite payload is malformed")
    if contract.get("schema") != SCHEMA:
        raise ValueError(f"{checkpoint_label}: composite schema drifted")
    if set(state) != set(expected_state):
        raise ValueError(f"{checkpoint_label}: full-model key coverage drifted")
    for key, wanted in expected_state.items():
        value = state.get(key)
        if (
            not torch.is_tensor(value)
            or not torch.is_tensor(wanted)
            or value.dtype != wanted.dtype
            or tuple(value.shape) != tuple(wanted.shape)
        ):
            raise ValueError(f"{checkpoint_label}: tensor shape/dtype drift at {key}")

    expected_roles = {
        "legacy_r100_p50",
        "shared_backbone_alias",
        "d9_patch",
        "u0_adapter",
    }
    roles = contract.get("role_keys")
    counts = contract.get("role_key_counts")
    if (
        not isinstance(roles, Mapping)
        or set(roles) != expected_roles
        or not isinstance(counts, Mapping)
        or set(counts) != expected_roles
    ):
        raise ValueError(f"{checkpoint_label}: composite roles drifted")
    normalized: dict[str, list[str]] = {}
    for role in expected_roles:
        keys = roles.get(role)
        if (
            not isinstance(keys, list)
            or not keys
            or len(keys) != counts.get(role)
            or len(keys) != len(set(keys))
        ):
            raise ValueError(f"{checkpoint_label}: composite role {role} drifted")
        normalized[role] = list(keys)
    flattened = [key for keys in normalized.values() for key in keys]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(state):
        raise ValueError(f"{checkpoint_label}: roles are not an exact partition")
    if set(normalized["d9_patch"]) != set(PATCH_KEYS):
        raise ValueError(f"{checkpoint_label}: D9 patch role drifted")
    if not all(
        key.startswith(PATCH_BACKBONE_PREFIX)
        for key in normalized["shared_backbone_alias"]
    ):
        raise ValueError(f"{checkpoint_label}: shared-backbone role drifted")
    if not all(key.startswith(U0_ADAPTER_PREFIX) for key in normalized["u0_adapter"]):
        raise ValueError(f"{checkpoint_label}: U0 adapter role drifted")
    for role, keys in normalized.items():
        if contract.get(f"{role}_tensor_sha256") != stage_b_u0_tensor_state_sha256(
            state, keys
        ):
            raise ValueError(f"{checkpoint_label}: {role} tensor hash drifted")
    if contract.get("full_model_tensor_sha256") != stage_b_u0_tensor_state_sha256(
        state, state.keys()
    ):
        raise ValueError(f"{checkpoint_label}: full-model tensor hash drifted")
    for alias in normalized["shared_backbone_alias"]:
        base_key = alias.removeprefix("patch_encoder.")
        if base_key not in state or not torch.equal(state[alias], state[base_key]):
            raise ValueError(f"{checkpoint_label}: shared alias differs at {alias}")
    for key in (U0_ADAPTER_PREFIX + "output.weight", U0_ADAPTER_PREFIX + "output.bias"):
        if key not in state or int(torch.count_nonzero(state[key]).item()):
            raise ValueError(f"{checkpoint_label}: U0 continuous residual is not zero")
    required_top = {
        "eval_only": True,
        "resumable": False,
        "single_checkpoint": True,
        "single_model_root": True,
        "external_score_source_required": False,
        "model_state_keys": len(state),
    }
    if any(contract.get(key) != value for key, value in required_top.items()):
        raise ValueError(f"{checkpoint_label}: deployment contract drifted")
    invariants = contract.get("invariants")
    if not isinstance(invariants, Mapping) or not invariants or any(
        value is not True for value in invariants.values()
    ):
        raise ValueError(f"{checkpoint_label}: data-only invariants drifted")
    config_contract = contract.get("config_contract")
    required_config = {
        "stage_b_data_only_composite": True,
        "stage_b_data_only_composite_contract_version": 1,
        "stage_b_u0_patch_rank": True,
        "stage_b_gdino_score_adapter": True,
        "enable_patch_branch": True,
        "patch_gate_with_text": False,
        "stage_b_u0_category_preserving_patch_gate": True,
        "stage_b_u0_category_gate_max_gap": 3.0,
        "stage_b_u2_category_complete_supervision": False,
    }
    if config_contract != required_config:
        raise ValueError(f"{checkpoint_label}: config contract drifted")


def validate_data_only_composite_runtime_config(
    cfg: Any,
    payload: Mapping[str, Any],
    *,
    checkpoint_path: str | Path,
    checkpoint_label: str,
) -> None:
    """Bind any runtime gap override to the sealed canonical-val receipt."""
    contract = payload.get(CONTRACT_KEY) if isinstance(payload, Mapping) else None
    if not isinstance(contract, Mapping):
        raise ValueError(f"{checkpoint_label}: composite contract is missing")
    build_config = contract.get("config_contract")
    if not isinstance(build_config, Mapping):
        raise ValueError(f"{checkpoint_label}: build config contract is missing")
    runtime_gap = getattr(cfg, "stage_b_u0_category_gate_max_gap", None)
    if isinstance(runtime_gap, bool) or not isinstance(runtime_gap, (int, float)):
        raise ValueError(f"{checkpoint_label}: runtime category gap is invalid")
    runtime_gap = float(runtime_gap)
    build_gap = float(build_config.get("stage_b_u0_category_gate_max_gap"))
    if runtime_gap == build_gap:
        return

    expected_checkpoint_sha = str(
        getattr(cfg, "stage_b_data_only_checkpoint_sha256", "")
    )
    checkpoint_record = stable_file_record(
        Path(checkpoint_path), label="runtime data-only checkpoint"
    )
    if checkpoint_record["sha256"] != expected_checkpoint_sha:
        raise ValueError(f"{checkpoint_label}: selected checkpoint SHA256 drifted")
    receipt_value = str(
        getattr(cfg, "stage_b_data_only_gate_selection_receipt", "")
    ).strip()
    if not receipt_value:
        raise ValueError(f"{checkpoint_label}: selected gap has no receipt")
    receipt_path = Path(receipt_value).expanduser()
    if not receipt_path.is_absolute():
        receipt_path = REPO_ROOT / receipt_path
    expected_receipt_sha = str(
        getattr(cfg, "stage_b_data_only_gate_selection_receipt_sha256", "")
    )
    receipt_record = stable_file_record(
        receipt_path, label="data-only gate selection receipt"
    )
    if receipt_record["sha256"] != expected_receipt_sha:
        raise ValueError(f"{checkpoint_label}: gate selection receipt SHA256 drifted")
    receipt = _read_json(receipt_path, label="data-only gate selection receipt")
    recorded_payload_hash = receipt.get("payload_sha256")
    unsealed = dict(receipt)
    unsealed.pop("payload_sha256", None)
    expected_payload_hash = str(
        getattr(cfg, "stage_b_data_only_gate_selection_payload_sha256", "")
    )
    if (
        receipt.get("schema") != D9_GATE_SELECTION_SCHEMA
        or receipt.get("selection_frozen") is not True
        or recorded_payload_hash != canonical_json_sha256(unsealed)
        or recorded_payload_hash != expected_payload_hash
    ):
        raise ValueError(f"{checkpoint_label}: gate selection receipt seal drifted")
    selection = receipt.get("selection")
    if (
        not isinstance(selection, Mapping)
        or float(selection.get("max_gap", float("nan"))) != runtime_gap
        or receipt.get("feasible_gaps") != [runtime_gap]
    ):
        raise ValueError(f"{checkpoint_label}: runtime gap was not uniquely selected")
    split_counts = selection.get("split_counts")
    if not isinstance(split_counts, Mapping) or set(split_counts) != {
        "refcoco_val",
        "refcocop_val",
        "refcocog_val",
    }:
        raise ValueError(f"{checkpoint_label}: selected val split coverage drifted")
    for split, row in split_counts.items():
        if (
            not isinstance(row, Mapping)
            or type(row.get("gate_correct")) is not int
            or type(row.get("baseline_correct")) is not int
            or row["gate_correct"] <= row["baseline_correct"]
        ):
            raise ValueError(
                f"{checkpoint_label}: selected gap does not beat U2 on {split}"
            )
    inputs = receipt.get("inputs")
    sweep_artifact = (
        inputs.get("sweep_summary") if isinstance(inputs, Mapping) else None
    )
    if not isinstance(sweep_artifact, Mapping):
        raise ValueError(f"{checkpoint_label}: sweep artifact binding is missing")
    sweep_path = Path(str(sweep_artifact.get("path", "")))
    if stable_file_record(
        sweep_path, label="selected data-only gap sweep summary"
    ) != dict(sweep_artifact):
        raise ValueError(f"{checkpoint_label}: selected sweep summary drifted")
    sweep = _read_json(sweep_path, label="selected data-only gap sweep summary")
    rows = sweep.get("refcoco")
    if not isinstance(rows, list) or len(rows) != 33 or any(
        not isinstance(row, Mapping)
        or row.get("checkpoint_sha256") != expected_checkpoint_sha
        for row in rows
    ):
        raise ValueError(f"{checkpoint_label}: sweep checkpoint provenance drifted")


def _validate_source_copy(
    payload: Mapping[str, Any],
    *,
    source_payloads: Mapping[str, Mapping[str, Any]],
) -> None:
    state = _model_state(payload, label="serialized data-only composite")
    contract = payload[CONTRACT_KEY]
    roles = contract["role_keys"]
    u0_state = _model_state(source_payloads["u0_initializer"], label="U0 initializer")
    d9_state = _model_state(source_payloads["d9_patch"], label="D9 checkpoint")
    for key in roles["d9_patch"]:
        if not torch.equal(state[key], d9_state[key]):
            raise DataOnlyCompositeError(f"serialized D9 patch copy drifted at {key}")
    for role in ("legacy_r100_p50", "shared_backbone_alias", "u0_adapter"):
        for key in roles[role]:
            if not torch.equal(state[key], u0_state[key]):
                raise DataOnlyCompositeError(f"serialized U0 preservation drifted at {key}")


def _write_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        raise DataOnlyCompositeError(f"refusing to overwrite composite: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    if temporary.exists():
        raise DataOnlyCompositeError(f"stale composite temporary exists: {temporary}")
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def verify_checkpoint(path: Path, *, replay_sources: bool = True) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    payload = _safe_load_checkpoint(path, label="data-only composite")
    state = _model_state(payload, label="data-only composite")
    validate_data_only_composite_payload(
        state, payload, checkpoint_label="serialized data-only composite"
    )
    if replay_sources:
        sources = payload[CONTRACT_KEY].get("sources")
        if not isinstance(sources, Mapping):
            raise DataOnlyCompositeError("serialized source records are missing")
        for role, expected in EXPECTED_FILE_SHA256.items():
            record = sources.get(role)
            if not isinstance(record, Mapping) or record.get("sha256") != expected:
                raise DataOnlyCompositeError(f"serialized source record drifted: {role}")
        records, source_payloads, legacy_lineage = _source_payloads(
            u0_path=Path(sources["u0_initializer"]["path"]),
            d9_path=Path(sources["d9_patch"]["path"]),
            d1_path=Path(sources["d1_source"]["path"]),
            d1_initializer_path=Path(sources["d1_initializer"]["path"]),
            legacy_receipt_path=Path(sources["legacy_receipt"]["path"]),
        )
        del records
        validate_stage_b_u0_initializer_payload(
            _model_state(source_payloads["u0_initializer"], label="U0 initializer"),
            source_payloads["u0_initializer"],
            checkpoint_label="replayed U0 initializer",
        )
        _validate_d9_lineage(
            d9_payload=source_payloads["d9_patch"],
            d1_payload=source_payloads["d1_source"],
            initializer_payload=source_payloads["d1_initializer"],
        )
        if (
            payload[CONTRACT_KEY]["lineage"]["legacy_r100_p50"]
            != legacy_lineage
        ):
            raise DataOnlyCompositeError("serialized legacy lineage drifted")
        _validate_source_copy(payload, source_payloads=source_payloads)
        del source_payloads
        gc.collect()
    return {
        "schema": SCHEMA,
        "status": "verified",
        "source_replay": bool(replay_sources),
        "checkpoint": stable_file_record(path, label="data-only composite"),
        "full_model_tensor_sha256": payload[CONTRACT_KEY][
            "full_model_tensor_sha256"
        ],
        "model_state_keys": len(state),
        "role_key_counts": payload[CONTRACT_KEY]["role_key_counts"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    create.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    create.add_argument("--u0", type=Path, default=DEFAULT_U0)
    create.add_argument("--d9", type=Path, default=DEFAULT_D9)
    create.add_argument("--d1", type=Path, default=DEFAULT_D1)
    create.add_argument("--d1-initializer", type=Path, default=DEFAULT_D1_INITIALIZER)
    create.add_argument("--legacy-receipt", type=Path, default=DEFAULT_LEGACY_RECEIPT)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--checkpoint", type=Path, required=True)
    verify.add_argument(
        "--fast",
        action="store_true",
        help="validate the sealed artifact without replaying all source checkpoints",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "create":
            payload = build_payload(
                config=args.config,
                u0_path=args.u0,
                d9_path=args.d9,
                d1_path=args.d1,
                d1_initializer_path=args.d1_initializer,
                legacy_receipt_path=args.legacy_receipt,
            )
            _write_checkpoint(args.output, payload)
            result = verify_checkpoint(args.output, replay_sources=False)
            result["source_replay"] = True
            result["source_replay_stage"] = "pre_serialization_build"
        else:
            result = verify_checkpoint(args.checkpoint, replay_sources=not args.fast)
    except (DataOnlyCompositeError, OSError, ValueError) as error:
        print(f"[FAIL] {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
