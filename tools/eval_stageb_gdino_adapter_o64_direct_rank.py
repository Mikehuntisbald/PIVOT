#!/usr/bin/env python3
"""Audit and evaluate the b58-only native-residual O64 rank probe."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import build_dataset  # noqa: E402
from engine import _build_stage_b_gdino_adapter_rank_captions  # noqa: E402
from groundingdino.util import box_ops  # noqa: E402
from groundingdino.util.utils import clean_state_dict  # noqa: E402
from main import build_model_main  # noqa: E402
from models.GroundingDINO.stage_b_gdino_score_adapter import (  # noqa: E402
    validate_stage_b_gdino_score_adapter_checkpoint,
)
from tools.build_stageb_gdino_adapter_o64_direct_rank import (  # noqa: E402
    EXPECTED_SOURCE_PAIRS,
    OUTPUT_MANIFEST,
    OUTPUT_ROW_SCHEMA,
    verify as verify_direct_rank_artifact,
)
from tools.build_stageb_native_residual_initializer import (  # noqa: E402
    ADAPTER_PREFIX,
    CONFIDENCE_PARTS,
    EXPECTED_B58_SHA256,
    RANK_PARTS,
    validate_initializer_payload,
)
from tools.build_stageb_u0_training_receipt import (  # noqa: E402
    _safe_load_checkpoint,
    stable_file_record,
)
from util import misc as utils  # noqa: E402
from util.slconfig import SLConfig  # noqa: E402


SCHEMA = "pivot.stageb.gdino_adapter.o64_direct_rank_eval/v1"
EXPECTED_ROWS = 128
EXPECTED_PAIRS = 64
EXPECTED_QUERIES = 900
EXPECTED_BASE_TENSORS = 938
EXPECTED_RANK_TENSORS = 8
EXPECTED_CONFIDENCE_TENSORS = 12
EXPECTED_UPDATES = 500
EXPECTED_BATCH_SIZE = 32
EXPECTED_GRADIENT_ACCUMULATION_STEPS = 2
EXPECTED_EFFECTIVE_BATCH_SIZE = (
    EXPECTED_BATCH_SIZE * EXPECTED_GRADIENT_ACCUMULATION_STEPS
)
EXPECTED_EPOCHS = 250
EXPECTED_RANK_LR = 3.0e-4


class O64DirectRankAuditError(RuntimeError):
    pass


def _checkpoint_args(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = payload.get("args")
    if isinstance(value, Mapping):
        return dict(value)
    if value is not None and hasattr(value, "__dict__"):
        return dict(vars(value))
    return {}


def _model_state(payload: Mapping[str, Any], *, label: str) -> dict[str, torch.Tensor]:
    value = payload.get("model")
    if not isinstance(value, Mapping) or not value:
        raise O64DirectRankAuditError(f"{label} has no model mapping")
    state = clean_state_dict(value)
    invalid = [
        str(key)
        for key, tensor in state.items()
        if not isinstance(key, str) or not torch.is_tensor(tensor)
    ]
    if invalid:
        raise O64DirectRankAuditError(
            f"{label} contains non-tensor model values: {invalid[:8]}"
        )
    return dict(state)


def _adapter_groups(
    initializer_payload: Mapping[str, Any],
) -> tuple[list[str], list[str], list[str]]:
    state = _model_state(initializer_payload, label="native-residual initializer")
    contract = initializer_payload.get("native_residual_initializer")
    roles = contract.get("role_keys") if isinstance(contract, Mapping) else None
    if not isinstance(roles, Mapping) or set(roles) != {
        "b58_base",
        "random_identity_adapter",
    }:
        raise O64DirectRankAuditError("initializer role partition is missing or drifted")
    base = sorted(str(key) for key in roles["b58_base"])
    adapter = sorted(str(key) for key in roles["random_identity_adapter"])
    if (
        len(base) != len(set(base))
        or len(adapter) != len(set(adapter))
        or set(base) & set(adapter)
        or set(base) | set(adapter) != set(state)
    ):
        raise O64DirectRankAuditError(
            "initializer role partition is not disjoint and exhaustive"
        )
    rank = sorted(
        key
        for key in adapter
        if key.removeprefix(ADAPTER_PREFIX).startswith(RANK_PARTS)
    )
    confidence = sorted(
        key
        for key in adapter
        if key.removeprefix(ADAPTER_PREFIX).startswith(CONFIDENCE_PARTS)
    )
    unknown = sorted(set(adapter).difference(rank).difference(confidence))
    observed = (len(base), len(rank), len(confidence))
    expected = (
        EXPECTED_BASE_TENSORS,
        EXPECTED_RANK_TENSORS,
        EXPECTED_CONFIDENCE_TENSORS,
    )
    if observed != expected or unknown:
        raise O64DirectRankAuditError(
            "initializer tensor geometry drifted: "
            f"base/rank/confidence={observed}, expected={expected}, unknown={unknown[:8]}"
        )
    return base, rank, confidence


def audit_b58_lineage(
    b58_payload: Mapping[str, Any], initializer_payload: Mapping[str, Any]
) -> dict[str, Any]:
    b58 = _model_state(b58_payload, label="b58 checkpoint")
    initializer = _model_state(
        initializer_payload, label="native-residual initializer"
    )
    base, _rank, _confidence = _adapter_groups(initializer_payload)
    if set(b58) != set(base):
        raise O64DirectRankAuditError(
            "b58 model keys do not exactly equal the initializer base role"
        )
    drift = [key for key in base if not torch.equal(initializer[key], b58[key])]
    if drift:
        raise O64DirectRankAuditError(
            f"initializer base tensors differ from b58: {drift[:8]}"
        )
    return {
        "b58_tensor_count": len(b58),
        "initializer_base_tensor_count": len(base),
        "all_base_tensors_bitwise_equal_b58": True,
    }


def audit_tensor_isolation(
    initializer_payload: Mapping[str, Any],
    checkpoint_payload: Mapping[str, Any],
    *,
    identity: bool,
) -> dict[str, Any]:
    initializer = _model_state(
        initializer_payload, label="native-residual initializer"
    )
    checkpoint = _model_state(checkpoint_payload, label="O64 checkpoint")
    if set(initializer) != set(checkpoint):
        missing = sorted(set(initializer).difference(checkpoint))
        unexpected = sorted(set(checkpoint).difference(initializer))
        raise O64DirectRankAuditError(
            "checkpoint model key set differs from initializer: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    for key, expected in initializer.items():
        observed = checkpoint[key]
        if observed.dtype != expected.dtype or tuple(observed.shape) != tuple(
            expected.shape
        ):
            raise O64DirectRankAuditError(
                f"checkpoint tensor shape/dtype differs at {key}"
            )

    base, rank, confidence = _adapter_groups(initializer_payload)
    changed_base = [key for key in base if not torch.equal(initializer[key], checkpoint[key])]
    changed_rank = [key for key in rank if not torch.equal(initializer[key], checkpoint[key])]
    changed_confidence = [
        key
        for key in confidence
        if not torch.equal(initializer[key], checkpoint[key])
    ]
    if changed_base:
        raise O64DirectRankAuditError(
            f"O64 checkpoint changed frozen b58 tensors: {changed_base[:8]}"
        )
    if changed_confidence:
        raise O64DirectRankAuditError(
            "O64 checkpoint changed frozen confidence tensors: "
            f"{changed_confidence[:8]}"
        )
    expected_changed_rank = 0 if identity else EXPECTED_RANK_TENSORS
    if len(changed_rank) != expected_changed_rank:
        raise O64DirectRankAuditError(
            "O64 rank tensor change count drifted: "
            f"observed={len(changed_rank)}, expected={expected_changed_rank}"
        )
    return {
        "base_tensor_count": len(base),
        "rank_tensor_count": len(rank),
        "confidence_tensor_count": len(confidence),
        "changed_base_tensors": len(changed_base),
        "changed_rank_tensors": len(changed_rank),
        "changed_confidence_tensors": len(changed_confidence),
        "changed_rank_keys": changed_rank,
        "base_bitwise_equal_initializer": True,
        "confidence_bitwise_equal_initializer": True,
        "only_rank_tensors_changed": bool(not identity),
        "full_model_bitwise_equal_initializer": bool(identity),
    }


def _exact_int(value: Any, *, label: str) -> int:
    if type(value) is not int:
        raise O64DirectRankAuditError(f"{label} must be an exact integer")
    return int(value)


def _finite_scalar(value: Any, *, label: str) -> float:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise O64DirectRankAuditError(f"{label} must be scalar")
        value = value.detach().cpu().item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise O64DirectRankAuditError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise O64DirectRankAuditError(f"{label} must be finite")
    return result


def _saved_path(value: Any, *, label: str, allow_empty: bool = False) -> Path | None:
    if value in (None, "") and allow_empty:
        return None
    if not isinstance(value, (str, os.PathLike)) or not os.fspath(value):
        raise O64DirectRankAuditError(f"saved {label} path is missing")
    path = Path(os.fspath(value)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve(strict=True)


def _audit_optimizer(payload: Mapping[str, Any]) -> dict[str, Any]:
    optimizer = payload.get("optimizer")
    if not isinstance(optimizer, Mapping):
        raise O64DirectRankAuditError("O64 checkpoint has no optimizer state")
    groups = optimizer.get("param_groups")
    state = optimizer.get("state")
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(state, Mapping):
        raise O64DirectRankAuditError(
            "O64 optimizer must have exactly one parameter group and state mapping"
        )
    group = groups[0]
    if not isinstance(group, Mapping) or group.get("stage_b_gdino_branch") != "rank":
        raise O64DirectRankAuditError("O64 optimizer does not exclusively own rank")
    params = group.get("params")
    if (
        not isinstance(params, list)
        or len(params) != EXPECTED_RANK_TENSORS
        or len(set(params)) != EXPECTED_RANK_TENSORS
        or any(type(value) is not int for value in params)
    ):
        raise O64DirectRankAuditError("O64 optimizer rank parameter coverage drifted")
    if set(state) != set(params):
        raise O64DirectRankAuditError(
            "O64 optimizer state does not exactly cover all eight rank tensors"
        )
    if not math.isclose(
        _finite_scalar(group.get("lr"), label="optimizer rank lr"),
        EXPECTED_RANK_LR,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise O64DirectRankAuditError("O64 optimizer rank LR drifted")
    for parameter_id in params:
        parameter_state = state[parameter_id]
        if not isinstance(parameter_state, Mapping):
            raise O64DirectRankAuditError(
                f"optimizer state {parameter_id} is malformed"
            )
        step = _finite_scalar(
            parameter_state.get("step"), label=f"optimizer state {parameter_id} step"
        )
        if step != float(EXPECTED_UPDATES):
            raise O64DirectRankAuditError(
                f"optimizer state {parameter_id} is at step {step}, "
                f"expected {EXPECTED_UPDATES}"
            )
        for name in ("exp_avg", "exp_avg_sq"):
            tensor = parameter_state.get(name)
            if (
                not torch.is_tensor(tensor)
                or not tensor.is_floating_point()
                or tensor.numel() == 0
                or not bool(torch.isfinite(tensor).all().item())
            ):
                raise O64DirectRankAuditError(
                    f"optimizer state {parameter_id} {name} is invalid"
                )
        if tuple(parameter_state["exp_avg"].shape) != tuple(
            parameter_state["exp_avg_sq"].shape
        ):
            raise O64DirectRankAuditError(
                f"optimizer state {parameter_id} moment shapes differ"
            )
    return {
        "groups": 1,
        "branch": "rank",
        "parameter_states": len(state),
        "all_parameter_steps": EXPECTED_UPDATES,
        "all_moments_finite": True,
    }


def audit_training_checkpoint(
    payload: Mapping[str, Any],
    *,
    config_path: Path,
    dataset_path: Path,
    initializer_path: Path,
    checkpoint_path: Path,
    loader_batches: int,
) -> dict[str, Any]:
    if loader_batches <= 0:
        raise O64DirectRankAuditError("loader_batches must be positive")
    updates = _exact_int(payload.get("optimizer_updates"), label="optimizer_updates")
    if updates != EXPECTED_UPDATES or payload.get("checkpoint_reason") != "max_train_iters":
        raise O64DirectRankAuditError(
            "O64 terminal checkpoint must record exactly 500 successful updates "
            "and reason=max_train_iters"
        )
    epoch = _exact_int(payload.get("epoch"), label="epoch")
    iteration = _exact_int(payload.get("iteration"), label="iteration")
    updates_per_epoch = math.ceil(
        loader_batches / EXPECTED_GRADIENT_ACCUMULATION_STEPS
    )
    expected_epoch = (EXPECTED_UPDATES - 1) // updates_per_epoch
    if epoch != expected_epoch or payload.get("epoch_finished") is not True:
        raise O64DirectRankAuditError(
            "O64 terminal checkpoint does not match the derived final epoch boundary"
        )
    allowed_iterations = {0, loader_batches - 1, loader_batches}
    if iteration not in allowed_iterations:
        raise O64DirectRankAuditError(
            "O64 terminal checkpoint iteration is inconsistent with its loader length: "
            f"observed={iteration}, allowed={sorted(allowed_iterations)}"
        )
    for key in ("criterion", "optimizer", "lr_scheduler", "scaler", "rng_state"):
        if not isinstance(payload.get(key), Mapping):
            raise O64DirectRankAuditError(f"O64 checkpoint is missing {key} state")

    args = _checkpoint_args(payload)
    if not args:
        raise O64DirectRankAuditError("O64 checkpoint saved args are missing")
    expected_args = {
        "stage_b_native_residual_data_only": True,
        "stage_b_native_residual_contract_version": 1,
        "stage_b_gdino_score_adapter": True,
        "stage_b_gdino_adapter_train_mode": "rank_only",
        "stage_b_gdino_tn_scope": "",
        "stage_b_gdino_rank_weight": 1.0,
        "stage_b_gdino_confidence_weight": 0.0,
        "stage_b_gdino_paired_margin_weight": 0.0,
        "stage_b_gdino_queue_size": 0,
        "stage_b_gdino_queue_min_count": 0,
        "stage_b_gdino_rank_lr": EXPECTED_RANK_LR,
        "lr": EXPECTED_RANK_LR,
        "batch_size": EXPECTED_BATCH_SIZE,
        "epochs": EXPECTED_EPOCHS,
        "max_train_iters": EXPECTED_UPDATES,
        "lr_drop": 1000,
        "fix_size": True,
        "data_aug_hflip_prob": 0.0,
        "data_aug_scales": [800],
        "data_aug_max_size": 1333,
        "gradient_accumulation_steps": EXPECTED_GRADIENT_ACCUMULATION_STEPS,
        "amp": True,
        "enable_patch_branch": False,
    }
    drift = {
        key: {"observed": args.get(key), "expected": expected}
        for key, expected in expected_args.items()
        if args.get(key) != expected
    }
    if drift:
        raise O64DirectRankAuditError(f"O64 saved args drifted: {drift}")
    forbidden_modes = (
        "stage_b_u0_patch_rank",
        "stage_b_data_driven_score",
        "stage_b_v7",
        "stage_b_v11_fixed_text",
        "stage_b_legacy_global_gate",
    )
    enabled_forbidden = [key for key in forbidden_modes if bool(args.get(key, False))]
    if enabled_forbidden:
        raise O64DirectRankAuditError(
            f"O64 saved args enable forbidden score paths: {enabled_forbidden}"
        )
    if _saved_path(args.get("config_file"), label="config_file") != config_path:
        raise O64DirectRankAuditError("O64 saved config path drifted")
    if _saved_path(args.get("datasets"), label="datasets") != dataset_path:
        raise O64DirectRankAuditError("O64 saved dataset path drifted")
    if (
        _saved_path(args.get("pretrain_model_path"), label="pretrain_model_path")
        != initializer_path
    ):
        raise O64DirectRankAuditError("O64 saved initializer path drifted")
    if _saved_path(args.get("resume"), label="resume", allow_empty=True) is not None:
        raise O64DirectRankAuditError("O64 must initialize with pretrain, not resume")
    if _saved_path(args.get("output_dir"), label="output_dir") != checkpoint_path.parent:
        raise O64DirectRankAuditError("O64 saved output directory drifted")

    criterion = payload["criterion"]
    for key, expected in (
        ("criterion_train_mode_code", 1),
        ("criterion_scope_code", 0),
        ("criterion_queue_size", 0),
        ("criterion_queue_min_count", 0),
    ):
        if key not in criterion or _finite_scalar(
            criterion[key], label=f"criterion.{key}"
        ) != float(expected):
            raise O64DirectRankAuditError(f"O64 criterion contract drifted at {key}")
    optimizer = _audit_optimizer(payload)
    return {
        "optimizer_updates": updates,
        "checkpoint_reason": "max_train_iters",
        "epoch": epoch,
        "iteration": iteration,
        "epoch_finished": True,
        "loader_batches": loader_batches,
        "optimizer_updates_per_epoch": updates_per_epoch,
        "derived_terminal_epoch": expected_epoch,
        "train_micro_batch_size": EXPECTED_BATCH_SIZE,
        "train_gradient_accumulation_steps": (
            EXPECTED_GRADIENT_ACCUMULATION_STEPS
        ),
        "train_effective_batch_size": EXPECTED_EFFECTIVE_BATCH_SIZE,
        "saved_args_verified": sorted(expected_args),
        "lineage_uses_pretrain_not_resume": True,
        "optimizer": optimizer,
    }


def validate_o64_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if len(rows) != EXPECTED_ROWS:
        raise O64DirectRankAuditError(
            f"O64 direct-rank manifest has {len(rows)} rows, expected {EXPECTED_ROWS}"
        )
    normalized = []
    pair_ids: dict[int, str] = {}
    target_ids: set[int] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise O64DirectRankAuditError(f"O64 row {index} is not an object")
        pair_index = row.get("pair_index")
        direction = row.get("direction")
        expected_pair = index // 2
        expected_direction = "anchor" if index % 2 == 0 else "partner"
        if (
            row.get("row_schema") != OUTPUT_ROW_SCHEMA
            or type(pair_index) is not int
            or pair_index != expected_pair
            or direction != expected_direction
        ):
            raise O64DirectRankAuditError(
                f"O64 row {index} violates anchor/partner pair order"
            )
        pair_id = row.get("source_member_pair_id")
        target_id = row.get("target_coco_ann_id")
        if not isinstance(pair_id, str) or len(pair_id) != 64 or type(target_id) is not int:
            raise O64DirectRankAuditError(f"O64 row {index} identity is malformed")
        previous = pair_ids.setdefault(pair_index, pair_id)
        if previous != pair_id:
            raise O64DirectRankAuditError(
                f"O64 pair {pair_index} directions have different source IDs"
            )
        if target_id in target_ids:
            raise O64DirectRankAuditError(
                f"O64 target annotation {target_id} is repeated"
            )
        target_ids.add(target_id)
        normalized.append(
            {
                "row_index": index,
                "pair_index": pair_index,
                "direction": direction,
                "source_member_pair_id": pair_id,
                "target_coco_ann_id": target_id,
            }
        )
    if set(pair_ids) != set(range(EXPECTED_PAIRS)) or len(target_ids) != EXPECTED_ROWS:
        raise O64DirectRankAuditError("O64 pair/target coverage is incomplete")
    return normalized


def _candidate_iou(
    candidate_boxes: torch.Tensor, targets: Sequence[Mapping[str, Any]]
) -> torch.Tensor:
    if candidate_boxes.dim() != 3 or int(candidate_boxes.shape[-1]) != 4:
        raise O64DirectRankAuditError("candidate boxes must have shape (B,Q,4)")
    if len(targets) != int(candidate_boxes.shape[0]):
        raise O64DirectRankAuditError("targets do not align with candidate boxes")
    candidate_xyxy = box_ops.box_cxcywh_to_xyxy(candidate_boxes.detach().float())
    result = candidate_xyxy.new_zeros(candidate_xyxy.shape[:2])
    for index, target in enumerate(targets):
        boxes = target.get("boxes")
        if (
            not torch.is_tensor(boxes)
            or boxes.dim() != 2
            or int(boxes.shape[-1]) != 4
            or int(boxes.shape[0]) != 1
        ):
            raise O64DirectRankAuditError(
                f"O64 directed row {index} must contain exactly one target box"
            )
        target_xyxy = box_ops.box_cxcywh_to_xyxy(
            boxes.detach().to(device=result.device, dtype=torch.float32)
        )
        iou, _union = box_ops.box_iou(candidate_xyxy[index], target_xyxy)
        result[index] = iou[:, 0]
    return result


def audit_batch_outputs(
    outputs: Mapping[str, Any],
    targets: Sequence[Mapping[str, Any]],
    metadata: Sequence[Mapping[str, Any]],
    *,
    identity: bool,
) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    tensors = {
        key: outputs.get(key)
        for key in (
            "stage_b_gdino_base_score",
            "stage_b_gdino_rank_residual",
            "stage_b_gdino_rank_score",
            "pred_boxes",
        )
    }
    if any(not torch.is_tensor(value) for value in tensors.values()):
        raise O64DirectRankAuditError("O64 forward lacks native score/box tensors")
    base = tensors["stage_b_gdino_base_score"]
    residual = tensors["stage_b_gdino_rank_residual"]
    rank = tensors["stage_b_gdino_rank_score"]
    boxes = tensors["pred_boxes"]
    if (
        base.dim() != 2
        or tuple(rank.shape) != tuple(base.shape)
        or tuple(residual.shape) != tuple(base.shape)
        or tuple(boxes.shape[:2]) != tuple(base.shape)
        or int(base.shape[1]) != EXPECTED_QUERIES
        or len(targets) != int(base.shape[0])
        or len(metadata) != int(base.shape[0])
    ):
        raise O64DirectRankAuditError("O64 forward tensor geometry drifted")
    if not all(
        bool(torch.isfinite(value).all().item())
        for value in (base, residual, rank, boxes)
    ):
        raise O64DirectRankAuditError("O64 forward contains non-finite tensors")
    base_winner = base.argmax(dim=1)
    rank_winner = rank.argmax(dim=1)
    identity_checks = {
        "rank_residual_exact_zero": bool(torch.count_nonzero(residual).item() == 0),
        "rank_score_bitwise_equals_base": bool(torch.equal(rank, base)),
        "winner_query_equals_base": bool(torch.equal(rank_winner, base_winner)),
    }
    if identity and not all(identity_checks.values()):
        raise O64DirectRankAuditError(
            f"identity forward changed native b58 ranking: {identity_checks}"
        )

    iou = _candidate_iou(boxes, targets)
    row_index = torch.arange(int(base.shape[0]), device=base.device)
    base_top_iou = iou[row_index, base_winner]
    rank_top_iou = iou[row_index, rank_winner]
    base_correct = base_top_iou >= 0.5
    rank_correct = rank_top_iou >= 0.5
    records = []
    for index, meta in enumerate(metadata):
        records.append(
            {
                **dict(meta),
                "base_winner_query": int(base_winner[index].item()),
                "adapted_winner_query": int(rank_winner[index].item()),
                "base_top1_iou": float(base_top_iou[index].item()),
                "adapted_top1_iou": float(rank_top_iou[index].item()),
                "base_correct50": bool(base_correct[index].item()),
                "adapted_correct50": bool(rank_correct[index].item()),
            }
        )
    return records, identity_checks


def aggregate_o64_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(records) != EXPECTED_ROWS:
        raise O64DirectRankAuditError(
            f"O64 evaluation produced {len(records)} records, expected {EXPECTED_ROWS}"
        )
    pairs: dict[int, list[Mapping[str, Any]]] = {}
    base_correct = 0
    adapted_correct = 0
    wrong_fixed = 0
    correct_regressed = 0
    for index, record in enumerate(records):
        if record.get("row_index") != index:
            raise O64DirectRankAuditError("O64 evaluation record order drifted")
        base = record.get("base_correct50")
        adapted = record.get("adapted_correct50")
        if type(base) is not bool or type(adapted) is not bool:
            raise O64DirectRankAuditError("O64 correctness record is malformed")
        base_correct += int(base)
        adapted_correct += int(adapted)
        wrong_fixed += int((not base) and adapted)
        correct_regressed += int(base and (not adapted))
        pair_index = record.get("pair_index")
        if type(pair_index) is not int:
            raise O64DirectRankAuditError("O64 record pair index is malformed")
        pairs.setdefault(pair_index, []).append(record)
    if set(pairs) != set(range(EXPECTED_PAIRS)):
        raise O64DirectRankAuditError("O64 evaluation does not cover all 64 pairs")
    base_bidirectional = 0
    adapted_bidirectional = 0
    for pair_index, rows in pairs.items():
        if len(rows) != 2 or [row.get("direction") for row in rows] != [
            "anchor",
            "partner",
        ]:
            raise O64DirectRankAuditError(
                f"O64 pair {pair_index} does not contain ordered bidirectional rows"
            )
        base_bidirectional += int(all(bool(row["base_correct50"]) for row in rows))
        adapted_bidirectional += int(
            all(bool(row["adapted_correct50"]) for row in rows)
        )
    return {
        "rows": EXPECTED_ROWS,
        "pairs": EXPECTED_PAIRS,
        "base_correct50": base_correct,
        "adapted_correct50": adapted_correct,
        "wrong_fixed": wrong_fixed,
        "correct_regressed": correct_regressed,
        "base_bidirectional_correct_pairs": base_bidirectional,
        "adapted_bidirectional_correct_pairs": adapted_bidirectional,
        "base_acc50": base_correct / EXPECTED_ROWS,
        "adapted_acc50": adapted_correct / EXPECTED_ROWS,
        "base_bidirectional_pair_acc": base_bidirectional / EXPECTED_PAIRS,
        "adapted_bidirectional_pair_acc": adapted_bidirectional / EXPECTED_PAIRS,
    }


def _load_dataset_config(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise O64DirectRankAuditError(f"could not read O64 dataset config: {error}") from error
    if not isinstance(value, dict) or set(value) != {"train", "val"}:
        raise O64DirectRankAuditError("O64 dataset config must contain only train/val")
    train = value.get("train")
    if not isinstance(train, list) or len(train) != 1 or value.get("val") != []:
        raise O64DirectRankAuditError(
            "O64 dataset config requires one train entry and empty val"
        )
    entry = train[0]
    if (
        not isinstance(entry, dict)
        or entry.get("dataset_mode") != "odvg"
        or entry.get("root") != "/"
        or entry.get("mix_weight") != 1.0
    ):
        raise O64DirectRankAuditError("O64 dataset entry drifted")
    anno = _saved_path(entry.get("anno"), label="dataset annotation")
    if anno.name != OUTPUT_MANIFEST:
        raise O64DirectRankAuditError("O64 dataset does not point at the sealed manifest")
    return value


def _validate_config(cfg: SLConfig) -> None:
    required = {
        "stage_b_native_residual_data_only": True,
        "stage_b_native_residual_contract_version": 1,
        "stage_b_gdino_score_adapter": True,
        "stage_b_gdino_adapter_train_mode": "rank_only",
        "stage_b_gdino_tn_scope": "",
        "stage_b_gdino_rank_weight": 1.0,
        "stage_b_gdino_confidence_weight": 0.0,
        "stage_b_gdino_paired_margin_weight": 0.0,
        "stage_b_gdino_queue_size": 0,
        "stage_b_gdino_queue_min_count": 0,
        "stage_b_gdino_rank_lr": EXPECTED_RANK_LR,
        "lr": EXPECTED_RANK_LR,
        "batch_size": EXPECTED_BATCH_SIZE,
        "epochs": EXPECTED_EPOCHS,
        "lr_drop": 1000,
        "fix_size": True,
        "data_aug_hflip_prob": 0.0,
        "data_aug_scales": [800],
        "data_aug_max_size": 1333,
        "enable_patch_branch": False,
    }
    drift = {
        key: {"observed": getattr(cfg, key, None), "expected": expected}
        for key, expected in required.items()
        if getattr(cfg, key, None) != expected
    }
    if drift:
        raise O64DirectRankAuditError(f"O64 config drifted: {drift}")


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        payload, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True
    ) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="ascii",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(serialized)
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    if "GFLOPS_DEBUG_SHILONG" in os.environ:
        raise O64DirectRankAuditError("GFLOPS_DEBUG_SHILONG is forbidden")
    config_path = args.config.expanduser().resolve(strict=True)
    dataset_path = args.datasets.expanduser().resolve(strict=True)
    b58_path = args.b58.expanduser().resolve(strict=True)
    initializer_path = args.initializer.expanduser().resolve(strict=True)
    checkpoint_path = args.checkpoint.expanduser().resolve(strict=True)
    output_path = args.output_json.expanduser().resolve()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise O64DirectRankAuditError("CUDA was requested but is unavailable")
    if stable_file_record(b58_path, label="b58 checkpoint")["sha256"] != EXPECTED_B58_SHA256:
        raise O64DirectRankAuditError("b58 checkpoint SHA-256 drifted")

    cfg = SLConfig.fromfile(str(config_path))
    _validate_config(cfg)
    dataset_config = _load_dataset_config(dataset_path)
    dataset_entry = dataset_config["train"][0]
    annotation_path = _saved_path(
        dataset_entry["anno"], label="dataset annotation"
    )
    artifact_receipt = verify_direct_rank_artifact(
        output_root=annotation_path.parent, output_manifest=annotation_path.name
    )
    if (
        artifact_receipt.get("rows") != EXPECTED_ROWS
        or artifact_receipt.get("pairs") != EXPECTED_PAIRS
    ):
        raise O64DirectRankAuditError("O64 direct-rank receipt counts drifted")

    seed = int(getattr(cfg, "seed", 42))
    _seed_everything(seed)
    cfg.device = str(device)
    cfg.distributed = False
    model, _criterion, _postprocessors = build_model_main(cfg)
    initializer_payload = _safe_load_checkpoint(
        initializer_path, label="native-residual initializer"
    )
    validate_initializer_payload(
        model,
        initializer_payload,
        checkpoint_label=f"native-residual initializer {initializer_path}",
    )
    checkpoint_payload = _safe_load_checkpoint(
        checkpoint_path, label="O64 checkpoint"
    )
    b58_payload = _safe_load_checkpoint(b58_path, label="b58 checkpoint")
    b58_lineage = audit_b58_lineage(b58_payload, initializer_payload)
    isolation = audit_tensor_isolation(
        initializer_payload, checkpoint_payload, identity=bool(args.identity)
    )
    checkpoint_state = _model_state(checkpoint_payload, label="O64 checkpoint")
    validate_stage_b_gdino_score_adapter_checkpoint(
        model, checkpoint_state, checkpoint_label=f"O64 checkpoint {checkpoint_path}"
    )
    model.load_state_dict(checkpoint_state, strict=True)

    # Build the training manifest with the deterministic validation transform.
    dataset = build_dataset(image_set="val", args=cfg, datasetinfo=dataset_entry)
    metas = getattr(dataset, "metas", None)
    if not isinstance(metas, list):
        raise O64DirectRankAuditError("O64 ODVG dataset does not expose ordered metadata")
    metadata = validate_o64_rows(metas)
    if len(dataset) != EXPECTED_ROWS or getattr(dataset, "sample_weights", None) is not None:
        raise O64DirectRankAuditError("O64 evaluation dataset geometry drifted")
    loader = DataLoader(
        dataset,
        batch_size=EXPECTED_BATCH_SIZE,
        sampler=SequentialSampler(dataset),
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=0,
        pin_memory=False,
    )
    expected_loader_batches = math.ceil(EXPECTED_ROWS / EXPECTED_BATCH_SIZE)
    if len(loader) != expected_loader_batches:
        raise O64DirectRankAuditError("O64 sequential loader length drifted")
    training_audit = None
    if not args.identity:
        training_audit = audit_training_checkpoint(
            checkpoint_payload,
            config_path=config_path,
            dataset_path=dataset_path,
            initializer_path=initializer_path,
            checkpoint_path=checkpoint_path,
            loader_batches=len(loader),
        )

    model.to(device).eval()
    amp = not bool(args.no_amp)
    records: list[dict[str, Any]] = []
    identity_checks = {
        "rank_residual_exact_zero": True,
        "rank_score_bitwise_equals_base": True,
        "winner_query_equals_base": True,
    }
    cursor = 0
    with torch.inference_mode():
        for samples, raw_targets in loader:
            raw_targets = list(raw_targets)
            batch_metadata = metadata[cursor : cursor + len(raw_targets)]
            cursor += len(raw_targets)
            captions = _build_stage_b_gdino_adapter_rank_captions(raw_targets)
            samples = samples.to(device)
            targets = [
                {
                    key: value.to(device)
                    for key, value in target.items()
                    if torch.is_tensor(value)
                }
                for target in raw_targets
            ]
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=bool(amp and device.type == "cuda"),
            ):
                outputs = model(samples, captions=captions)
            batch_records, batch_identity = audit_batch_outputs(
                outputs,
                targets,
                batch_metadata,
                identity=bool(args.identity),
            )
            records.extend(batch_records)
            identity_checks = {
                key: identity_checks[key] and batch_identity[key]
                for key in identity_checks
            }
    if cursor != EXPECTED_ROWS:
        raise O64DirectRankAuditError("O64 forward did not consume exactly 128 rows")
    metrics = aggregate_o64_records(records)
    result = {
        "schema": SCHEMA,
        "mode": "identity" if args.identity else "trained",
        "inputs": {
            "config": stable_file_record(config_path, label="O64 config"),
            "datasets": stable_file_record(dataset_path, label="O64 datasets"),
            "b58": stable_file_record(b58_path, label="b58 checkpoint"),
            "initializer": stable_file_record(
                initializer_path, label="native-residual initializer"
            ),
            "checkpoint": stable_file_record(checkpoint_path, label="O64 checkpoint"),
        },
        "runtime": {
            "device": str(device),
            "amp": bool(amp and device.type == "cuda"),
            "dataset_transform": "val",
            "sampler": "sequential",
            "batch_size": EXPECTED_BATCH_SIZE,
            "eval_batch_size": EXPECTED_BATCH_SIZE,
            "batches": len(loader),
            "rows": EXPECTED_ROWS,
            "updates": 0,
            "train_micro_batch_size": EXPECTED_BATCH_SIZE,
            "train_gradient_accumulation_steps": (
                EXPECTED_GRADIENT_ACCUMULATION_STEPS
            ),
            "train_effective_batch_size": EXPECTED_EFFECTIVE_BATCH_SIZE,
        },
        "dataset_artifact": {
            "schema": artifact_receipt.get("schema"),
            "canonical_payload_sha256": artifact_receipt.get(
                "canonical_payload_sha256"
            ),
            "rows": artifact_receipt.get("rows"),
            "pairs": artifact_receipt.get("pairs"),
            "ordered_anchor_then_partner": True,
        },
        "b58_lineage": b58_lineage,
        "tensor_isolation": isolation,
        "training_audit": training_audit,
        "identity_checks": identity_checks if args.identity else None,
        "metrics": metrics,
        "passed": True,
    }
    _atomic_write_json(output_path, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--datasets", required=True, type=Path)
    parser.add_argument("--b58", required=True, type=Path)
    parser.add_argument("--initializer", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument(
        "--identity",
        action="store_true",
        help="require checkpoint model tensors and every forward rank output to be b58 identity",
    )
    return parser


def main() -> None:
    result = run_evaluation(_parser().parse_args())
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
