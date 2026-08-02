#!/usr/bin/env python3
"""Evaluate sealed DD1 PairTop1-family Overfit64 gates without updating state."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets import build_dataset  # noqa: E402
from engine import _build_stage_b_data_driven_assignment_captions  # noqa: E402
from groundingdino.util import box_ops  # noqa: E402
from groundingdino.util.utils import clean_state_dict  # noqa: E402
from main import (  # noqa: E402
    DeterministicEpochSampler,
    _torch_load_compat,
    _validate_stage_b_data_driven_assignment_training_contract,
    _validate_stage_b_data_driven_eval_update_gate,
    _validate_stage_b_data_driven_sampling_resume_state,
    build_model_main,
)
from models.GroundingDINO.stage_b_data_driven_score import (  # noqa: E402
    validate_data_driven_relational_initializer_payload,
    validate_stage_b_data_driven_score_checkpoint,
)
from util import misc as utils  # noqa: E402
from util.slconfig import SLConfig  # noqa: E402


SCHEMA = "pivot.stageb.data_driven.pairtop1_overfit_eval/v4"
VARIANT = "DD1-PairTop1"
HARDGAP3_VARIANT = "DD1-PairTop1-HardGap3"
SUPPORTED_VARIANTS = {
    VARIANT: 0.0,
    HARDGAP3_VARIANT: 1.0,
}
EXPECTED_ROWS = 64
EXPECTED_UPDATES = 500
MIN_RECIPROCAL_ROWS = 61
MIN_MARGIN_DIRECTIONS = 122
EXPECTED_AMP_SCALE = 8192.0
EXPECTED_OPTIMIZER_BRANCH_STATES = {"rank": 39, "patch": 9}
COUNT_KEYS = (
    "stage_b_data_driven_assignment_data_rows",
    "stage_b_data_driven_assignment_runtime_rows",
    "stage_b_data_driven_assignment_unreachable_rows",
    "stage_b_data_driven_assignment_correct_rows",
    "stage_b_data_driven_assignment_margin_rows",
    "stage_b_data_driven_assignment_correct_directions",
    "stage_b_data_driven_assignment_margin_directions",
    "stage_b_data_driven_assignment_deployment_correct_rows",
    "stage_b_data_driven_assignment_deployment_correct_directions",
    "stage_b_data_driven_assignment_query_collision_rows",
    "stage_b_data_driven_assignment_role0_queries",
    "stage_b_data_driven_assignment_role1_queries",
    "stage_b_data_driven_assignment_gap3_queries",
    "stage_b_data_driven_patch_valid_instances",
    "stage_b_data_driven_patch_skipped_instances",
)
MEAN_KEYS = (
    "loss_stage_b_data_driven_assignment",
    "loss_stage_b_data_driven_patch",
    "stage_b_data_driven_assignment_delta_mean",
    "stage_b_data_driven_assignment_direction_delta_mean",
    "stage_b_data_driven_assignment_selected_own_iou_mean",
    "stage_b_data_driven_assignment_selected_cross_iou_mean",
)
HARDGAP3_COUNT_KEYS = (
    "stage_b_data_driven_deployment_hard_valid_directions",
    "stage_b_data_driven_deployment_hard_margin_directions",
)
HARDGAP3_MEAN_KEYS = (
    "loss_stage_b_data_driven_deployment_hard",
    "stage_b_data_driven_deployment_hard_delta_mean",
)
HARDGAP3_DEPLOYMENT_COUNT_KEYS = (
    "deployment_gate_enabled",
    "deployment_eligible_mask_exact",
    "deployment_patch_score_exact",
    "deployment_top1_wiring_directions",
    "deployment_actual_raw_correct_rows",
    "deployment_actual_raw_correct_directions",
    "deployment_actual_clamped_correct_rows",
    "deployment_actual_clamped_correct_directions",
    "deployment_iou_correct_mask_disagreements",
    "deployment_top1_iou_correct_disagreements",
)


def _config_variant(cfg: SLConfig) -> str:
    variant = str(cfg.get("stage_b_data_driven_variant_id", VARIANT))
    if variant not in SUPPORTED_VARIANTS:
        raise RuntimeError(
            f"unsupported PairTop1-family evaluator variant: {variant!r}"
        )
    return variant


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _audit_file_records(
    records: Any,
    *,
    label: str,
    required_suffixes: Sequence[str] = (),
) -> dict[str, Any]:
    if not isinstance(records, list) or not records:
        raise RuntimeError(f"PairTop1 {label} records are missing")
    audited: list[dict[str, Any]] = []
    observed_paths: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise RuntimeError(f"PairTop1 {label}[{index}] is malformed")
        path_value = record.get("path")
        expected_sha = record.get("sha256")
        if (
            not isinstance(path_value, str)
            or not path_value
            or not isinstance(expected_sha, str)
            or len(expected_sha) != 64
        ):
            raise RuntimeError(f"PairTop1 {label}[{index}] has no path/SHA")
        path = Path(path_value).expanduser().resolve(strict=True)
        observed_sha = _sha256_file(path)
        if observed_sha != expected_sha:
            raise RuntimeError(
                f"PairTop1 {label} changed after launch: {path}"
            )
        size = int(path.stat().st_size)
        saved_size = record.get("size_bytes")
        if saved_size is not None and saved_size != size:
            raise RuntimeError(f"PairTop1 {label} size drifted: {path}")
        observed_paths.add(str(path))
        audited.append(
            {"path": str(path), "size_bytes": size, "sha256": observed_sha}
        )
    missing_suffixes = [
        suffix
        for suffix in required_suffixes
        if not any(path.endswith(suffix) for path in observed_paths)
    ]
    if missing_suffixes:
        raise RuntimeError(
            f"PairTop1 {label} lacks required files: {missing_suffixes}"
        )
    return {
        "record_count": len(audited),
        "all_current_sha256_match_launch": True,
        "records": audited,
    }


def _audit_saved_training_provenance(saved_args: Mapping[str, Any]) -> dict[str, Any]:
    provenance = saved_args.get("stage_b_data_driven_training_provenance")
    if (
        not isinstance(provenance, Mapping)
        or provenance.get("schema")
        != "pivot.stageb.data_driven_training_provenance/v1"
    ):
        raise RuntimeError("PairTop1 training provenance is missing")
    code = _audit_file_records(
        provenance.get("code_files"),
        label="launch code",
        required_suffixes=(
            "/main.py",
            "/engine.py",
            "/models/GroundingDINO/groundingdino.py",
            "/models/GroundingDINO/stage_b_data_driven_score.py",
        ),
    )
    assets = _audit_file_records(
        provenance.get("dataset_asset_files"), label="launch dataset assets"
    )
    config_chain = _audit_file_records(
        saved_args.get("stage_b_data_driven_config_import_chain"),
        label="launch config import chain",
    )
    dataset_record = saved_args.get("stage_b_data_driven_dataset_config")
    if not isinstance(dataset_record, Mapping):
        raise RuntimeError("PairTop1 launch dataset-config record is missing")
    dataset = _audit_file_records(
        [dataset_record], label="launch dataset config"
    )
    return {
        "schema": provenance["schema"],
        "code": code,
        "dataset_assets": assets,
        "config_import_chain": config_chain,
        "dataset_config": dataset,
        "verified_against_current_files_before_gate": True,
    }


def _finite_scalar(value: Any, *, key: str) -> float:
    if not torch.is_tensor(value) or value.numel() != 1:
        raise RuntimeError(f"PairTop1 metric {key!r} is not a scalar tensor")
    scalar = float(value.detach().float().cpu().item())
    if not math.isfinite(scalar):
        raise RuntimeError(f"PairTop1 metric {key!r} is not finite")
    return scalar


def _integer_metric(value: Any, *, key: str) -> int:
    scalar = _finite_scalar(value, key=key)
    rounded = int(round(scalar))
    if abs(scalar - rounded) > 1e-4:
        raise RuntimeError(f"PairTop1 count {key!r} is not integral: {scalar}")
    return rounded


def _exact_finite_scalar(value: Any, *, label: str) -> float:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise RuntimeError(f"{label} must be one scalar")
        value = value.detach().cpu().item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"{label} must be finite")
    return result


def _validate_finite_tensors(value: Any, *, label: str) -> int:
    if torch.is_tensor(value):
        if (torch.is_floating_point(value) or value.is_complex()) and not bool(
            torch.isfinite(value).all().item()
        ):
            raise RuntimeError(f"{label} contains a non-finite tensor")
        return 1
    if isinstance(value, Mapping):
        return sum(
            _validate_finite_tensors(item, label=f"{label}.{key}")
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return sum(
            _validate_finite_tensors(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        )
    return 0


def _audit_optimizer_state(payload: Mapping[str, Any]) -> dict[str, Any]:
    optimizer = payload.get("optimizer")
    if not isinstance(optimizer, Mapping):
        raise RuntimeError("PairTop1 terminal checkpoint is missing optimizer state")
    groups = optimizer.get("param_groups")
    state = optimizer.get("state")
    if not isinstance(groups, list) or len(groups) != 2 or not isinstance(state, Mapping):
        raise RuntimeError("PairTop1 optimizer must contain two groups and state")

    branch_ids: dict[str, set[int]] = {}
    for group in groups:
        if not isinstance(group, Mapping):
            raise RuntimeError("PairTop1 optimizer group is malformed")
        branch = group.get("stage_b_data_driven_branch")
        params = group.get("params")
        if branch not in {"rank", "patch"} or branch in branch_ids:
            raise RuntimeError("PairTop1 optimizer branches must be rank and patch")
        if (
            not isinstance(params, list)
            or not params
            or any(isinstance(item, bool) or not isinstance(item, int) for item in params)
            or len(params) != len(set(params))
        ):
            raise RuntimeError(f"PairTop1 optimizer {branch} parameter IDs drifted")
        branch_ids[str(branch)] = set(params)
    if set(branch_ids) != {"rank", "patch"} or branch_ids["rank"] & branch_ids["patch"]:
        raise RuntimeError("PairTop1 optimizer branch parameter IDs overlap")
    observed_counts = {branch: len(param_ids) for branch, param_ids in branch_ids.items()}
    if observed_counts != EXPECTED_OPTIMIZER_BRANCH_STATES:
        raise RuntimeError(
            "PairTop1 optimizer branch parameter coverage drifted: "
            f"{observed_counts}"
        )

    expected_ids = branch_ids["rank"] | branch_ids["patch"]
    if set(state) != expected_ids:
        raise RuntimeError("PairTop1 optimizer state does not exactly cover both branches")
    for param_id, param_state in state.items():
        if not isinstance(param_state, Mapping) or not {
            "step",
            "exp_avg",
            "exp_avg_sq",
        }.issubset(param_state):
            raise RuntimeError(f"PairTop1 optimizer state {param_id} is incomplete")
        step = _exact_finite_scalar(
            param_state["step"], label=f"optimizer.state[{param_id}].step"
        )
        if step != float(EXPECTED_UPDATES):
            raise RuntimeError(
                f"PairTop1 optimizer state {param_id} is at step {step}, "
                f"expected {EXPECTED_UPDATES}"
            )
        first = param_state["exp_avg"]
        second = param_state["exp_avg_sq"]
        if (
            not torch.is_tensor(first)
            or not torch.is_tensor(second)
            or not torch.is_floating_point(first)
            or not torch.is_floating_point(second)
            or first.numel() == 0
            or tuple(first.shape) != tuple(second.shape)
        ):
            raise RuntimeError(f"PairTop1 optimizer moments {param_id} drifted")
        _validate_finite_tensors(param_state, label=f"optimizer.state[{param_id}]")

    return {
        "branches": {
            branch: len(param_ids) for branch, param_ids in sorted(branch_ids.items())
        },
        "state_count": len(state),
        "all_state_steps": EXPECTED_UPDATES,
        "all_moments_finite": True,
        "parameter_ids_disjoint": True,
    }


def _audit_training_log(
    path: Path, *, require_hardgap3: bool = False
) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    content = path.read_text(encoding="utf-8")

    def values(key: str) -> list[float]:
        observed = [
            float(match)
            for match in re.findall(rf"\b{re.escape(key)}:\s+([^\s]+)", content)
        ]
        if any(not math.isfinite(item) for item in observed):
            raise RuntimeError(f"PairTop1 training log has non-finite {key}")
        return observed

    skips = values("amp_step_skipped")
    scales = values("amp_scale")
    steps = values("optimizer_step")
    if len(skips) not in {EXPECTED_UPDATES - 1, EXPECTED_UPDATES}:
        raise RuntimeError(
            "PairTop1 training log does not cover the 500-update one-batch run"
        )
    if len(scales) != len(skips) or len(steps) != len(skips):
        raise RuntimeError("PairTop1 training log metric counts drifted")
    if any(item != 0.0 for item in skips):
        raise RuntimeError("PairTop1 training log records an AMP skipped step")
    if any(item != EXPECTED_AMP_SCALE for item in scales):
        raise RuntimeError("PairTop1 training log AMP scale drifted")
    if any(item != 1.0 for item in steps):
        raise RuntimeError("PairTop1 training log records a failed optimizer step")
    if "optimizer_updates=500, reason=max_train_iters" not in content:
        raise RuntimeError("PairTop1 training log lacks its terminal U500 save record")
    hard_loss_rows = len(
        re.findall(
            r"\bloss_stage_b_data_driven_deployment_hard:\s+[^\s]+",
            content,
        )
    )
    if require_hardgap3 and hard_loss_rows not in {
        EXPECTED_UPDATES - 1,
        EXPECTED_UPDATES,
    }:
        raise RuntimeError(
            "HardGap3 training log does not cover the deployment-hard objective"
        )
    peak_memory = [int(item) for item in re.findall(r"\bmax mem:\s+(\d+)", content)]
    if not peak_memory:
        raise RuntimeError("PairTop1 training log lacks CUDA peak-memory evidence")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "metric_rows": len(skips),
        "amp_skipped_steps": 0,
        "amp_scale": EXPECTED_AMP_SCALE,
        "optimizer_step_success_rows": len(steps),
        "deployment_hard_loss_rows": hard_loss_rows,
        "peak_memory_mib": max(peak_memory),
    }


def _resolve_saved_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"PairTop1 saved {label} path is missing")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve(strict=True)


def _audit_terminal_checkpoint(
    payload: Mapping[str, Any],
    *,
    cfg: SLConfig,
    config_path: Path,
    dataset_path: Path,
    checkpoint_path: Path,
    base_path: Path,
) -> dict[str, Any]:
    cfg.stage_b_data_driven_eval_expected_optimizer_updates = EXPECTED_UPDATES
    _validate_stage_b_data_driven_eval_update_gate(
        cfg, payload, checkpoint_label=str(checkpoint_path)
    )
    if (
        payload.get("epoch") != EXPECTED_UPDATES - 1
        or payload.get("iteration") != 0
        or payload.get("epoch_finished") is not True
    ):
        raise RuntimeError("PairTop1 terminal checkpoint epoch boundary drifted")

    saved_args = payload.get("args")
    if not isinstance(saved_args, Mapping):
        raise RuntimeError("PairTop1 terminal checkpoint saved args are missing")
    variant = _config_variant(cfg)
    expected_args = {
        "seed": 42,
        "batch_size": EXPECTED_ROWS,
        "epochs": EXPECTED_UPDATES,
        "lr_drop": EXPECTED_UPDATES,
        "stage_b_data_driven_epoch_checkpoint_interval": EXPECTED_UPDATES,
        "max_train_iters": EXPECTED_UPDATES,
        "iter_checkpoint_interval": EXPECTED_UPDATES,
        "save_checkpoint_interval": EXPECTED_UPDATES,
        "num_workers": 0,
        "prefetch_factor": 1,
        "pin_memory": False,
        "persistent_workers": False,
        "gradient_accumulation_steps": 1,
        "amp": True,
        "amp_init_scale": EXPECTED_AMP_SCALE,
        "resume": "",
        "stage_b_data_driven_variant_id": variant,
        "stage_b_data_driven_no_teacher_contract": (
            "b58_only_random_independent_heads_v1"
        ),
        "stage_b_data_driven_assignment_weight": 1.0,
        "stage_b_data_driven_category_gate": False,
        "stage_b_data_driven_category_gate_max_gap": 3.0,
        "stage_b_data_driven_patch_score_clip": 5.0,
        "stage_b_data_driven_positive_iou_threshold": 0.5,
        "stage_b_data_driven_rank_negative_iou_threshold": 0.3,
        "stage_b_data_driven_rank_weight": 0.0,
        "stage_b_data_driven_patch_weight": 1.0,
        "stage_b_gdino_score_adapter": False,
        "stage_b_u0_patch_rank": False,
        "stage_b_v7": False,
        "stage_b_v11_fixed_text": False,
        "stage_b_legacy_global_gate": False,
    }
    drift = {
        key: (saved_args.get(key), expected)
        for key, expected in expected_args.items()
        if saved_args.get(key) != expected
    }
    if drift:
        raise RuntimeError(f"PairTop1 terminal saved runtime drifted: {drift}")
    deployment_weight = saved_args.get(
        "stage_b_data_driven_deployment_weight", 0.0
    )
    if deployment_weight != SUPPORTED_VARIANTS[variant]:
        raise RuntimeError(
            "PairTop1 terminal saved runtime drifted: "
            f"stage_b_data_driven_deployment_weight="
            f"{deployment_weight!r}, expected={SUPPORTED_VARIANTS[variant]!r}"
        )
    if _resolve_saved_path(
        saved_args.get("pretrain_model_path"), label="pretrain_model_path"
    ) != base_path:
        raise RuntimeError("PairTop1 terminal initializer path drifted")
    if _resolve_saved_path(saved_args.get("config_file"), label="config_file") != config_path:
        raise RuntimeError("PairTop1 terminal config path drifted")
    if _resolve_saved_path(saved_args.get("datasets"), label="datasets") != dataset_path:
        raise RuntimeError("PairTop1 terminal dataset path drifted")
    if _resolve_saved_path(saved_args.get("output_dir"), label="output_dir") != checkpoint_path.parent:
        raise RuntimeError("PairTop1 terminal output directory drifted")

    model_state = payload.get("model")
    criterion_state = payload.get("criterion")
    if not isinstance(model_state, Mapping) or not isinstance(criterion_state, Mapping):
        raise RuntimeError("PairTop1 terminal model/criterion state is missing")
    model_tensor_count = _validate_finite_tensors(model_state, label="model")
    criterion_tensor_count = _validate_finite_tensors(
        criterion_state, label="criterion"
    )
    optimizer_audit = _audit_optimizer_state(payload)
    provenance_audit = _audit_saved_training_provenance(saved_args)

    scaler = payload.get("scaler")
    if not isinstance(scaler, Mapping):
        raise RuntimeError("PairTop1 terminal scaler state is missing")
    scaler_expected = {
        "scale": EXPECTED_AMP_SCALE,
        "growth_factor": 2.0,
        "backoff_factor": 0.5,
        "growth_interval": 2000,
        "_growth_tracker": EXPECTED_UPDATES,
    }
    if set(scaler) != set(scaler_expected) or any(
        _exact_finite_scalar(scaler.get(key), label=f"scaler.{key}")
        != float(expected)
        for key, expected in scaler_expected.items()
    ):
        raise RuntimeError("PairTop1 terminal scaler cannot prove zero AMP skips")

    sampler = DeterministicEpochSampler(EXPECTED_ROWS, seed=42)
    sampling = _validate_stage_b_data_driven_sampling_resume_state(
        sampler, payload, loader_seed=1042
    )
    log_audit = _audit_training_log(
        checkpoint_path.parent / "info.txt",
        require_hardgap3=variant == HARDGAP3_VARIANT,
    )
    return {
        "status": "passed",
        "optimizer_updates": EXPECTED_UPDATES,
        "checkpoint_reason": "max_train_iters",
        "epoch": EXPECTED_UPDATES - 1,
        "epoch_finished": True,
        "model_tensor_count": model_tensor_count,
        "criterion_tensor_count": criterion_tensor_count,
        "all_model_and_criterion_tensors_finite": True,
        "optimizer": optimizer_audit,
        "scaler": dict(scaler_expected),
        "zero_amp_skips_proven": True,
        "sampling_state": sampling,
        "launch_provenance": provenance_audit,
        "training_log": log_audit,
    }


def _load_dataset_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or set(payload) != {"train", "val"}:
        raise RuntimeError("PairTop1 dataset config must contain only train/val")
    train = payload.get("train")
    if not isinstance(train, list) or len(train) != 1 or payload.get("val") != []:
        raise RuntimeError("PairTop1 Overfit64 requires exactly one train dataset and no val")
    if not isinstance(train[0], dict):
        raise RuntimeError("PairTop1 train dataset entry is malformed")
    return payload


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _validate_config(cfg: SLConfig, dataset_path: Path) -> Path:
    variant = _config_variant(cfg)
    base_path = Path(
        str(cfg.get("stage_b_data_driven_base_initializer_path", ""))
    ).expanduser().resolve(strict=True)
    expected_base_sha = str(
        cfg.get("stage_b_data_driven_base_initializer_sha256", "")
    )
    if _sha256_file(base_path) != expected_base_sha:
        raise RuntimeError("canonical A1 initializer SHA drifted")

    # Reuse the fail-closed training contract, but stop at its read-only eval gate.
    cfg.eval = True
    cfg.datasets = str(dataset_path)
    _validate_stage_b_data_driven_assignment_training_contract(
        cfg,
        base_path=base_path,
        variant_id=variant,
        dataset_path=dataset_path,
    )
    return base_path


def _load_state(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    checkpoint_path: Path,
) -> tuple[str, Mapping[str, Any]]:
    payload = _torch_load_compat(str(checkpoint_path), map_location="cpu")
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model"), Mapping):
        raise RuntimeError("PairTop1 checkpoint must contain a model mapping")
    if "data_driven_relational_initializer" in payload:
        checkpoint_kind = "a1_initializer"
        validate_data_driven_relational_initializer_payload(
            model,
            payload,
            checkpoint_label=str(checkpoint_path),
        )
    else:
        checkpoint_kind = "trained_checkpoint"
        validate_stage_b_data_driven_score_checkpoint(
            model,
            payload["model"],
            checkpoint_label=str(checkpoint_path),
        )
    model.load_state_dict(clean_state_dict(payload["model"]), strict=True)
    criterion_state = payload.get("criterion")
    if criterion_state is not None:
        if not isinstance(criterion_state, Mapping):
            raise RuntimeError("checkpoint criterion state is malformed")
        criterion.load_state_dict(criterion_state, strict=True)
    return checkpoint_kind, payload


def _enable_hardgap3_deployment_gate(model: torch.nn.Module) -> dict[str, Any]:
    heads = getattr(model, "stage_b_data_driven_score_heads", None)
    if heads is None:
        raise RuntimeError("HardGap3 model has no data-driven score heads")
    if bool(getattr(heads, "category_gate", False)):
        raise RuntimeError("HardGap3 training config unexpectedly enabled its gate")
    if float(getattr(heads, "category_gate_max_gap", float("nan"))) != 3.0:
        raise RuntimeError("HardGap3 deployment gap drifted")
    if float(getattr(heads, "patch_score_clip", float("nan"))) != 5.0:
        raise RuntimeError("HardGap3 deployment patch clip drifted")
    heads.category_gate = True
    return {
        "training_flag": False,
        "evaluation_flag": bool(heads.category_gate),
        "inference_only_override": True,
        "max_gap": 3.0,
        "patch_score_clip": 5.0,
    }


def _independent_assignment_iou(
    candidate_boxes: torch.Tensor,
    targets: Sequence[Mapping[str, Any]],
    *,
    clamp: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    if candidate_boxes.dim() != 3 or int(candidate_boxes.shape[-1]) != 4:
        raise RuntimeError("deployment audit candidate boxes are malformed")
    if len(targets) != int(candidate_boxes.shape[0]):
        raise RuntimeError("deployment audit targets do not align with boxes")
    candidates = box_ops.box_cxcywh_to_xyxy(
        candidate_boxes.detach().float()
    )
    if clamp:
        candidates = candidates.clamp(0.0, 1.0)
    result = candidates.new_zeros((*candidates.shape[:2], 2))
    valid = torch.zeros(
        (int(candidates.shape[0]),), dtype=torch.bool, device=candidates.device
    )
    for row, target in enumerate(targets):
        boxes = target.get("boxes")
        roles = target.get("stage_b_data_driven_assignment_role")
        pair_valid = target.get("stage_b_data_driven_assignment_valid")
        if (
            not torch.is_tensor(boxes)
            or boxes.dim() != 2
            or int(boxes.shape[-1]) != 4
            or not torch.is_tensor(roles)
            or roles.dtype != torch.int64
            or int(roles.numel()) != int(boxes.shape[0])
            or not torch.is_tensor(pair_valid)
            or pair_valid.dtype != torch.bool
            or pair_valid.numel() != 1
        ):
            raise RuntimeError("deployment audit assignment target is malformed")
        if not bool(pair_valid.reshape(-1)[0].item()):
            continue
        roles = roles.reshape(-1)
        if any(int((roles == role).sum().item()) != 1 for role in (0, 1)):
            raise RuntimeError("deployment audit requires one box per role")
        target_xyxy = box_ops.box_cxcywh_to_xyxy(
            boxes.detach().to(device=candidates.device, dtype=torch.float32)
        )
        if clamp:
            target_xyxy = target_xyxy.clamp(0.0, 1.0)
        iou, _ = box_ops.box_iou(candidates[row], target_xyxy)
        for role in (0, 1):
            result[row, :, role] = iou[:, roles == role].squeeze(1)
        valid[row] = True
    return result, valid


def _audit_hardgap3_deployment_outputs(
    outputs: Mapping[str, Any],
    targets: Sequence[Mapping[str, Any]],
    *,
    gate_runtime: Mapping[str, Any],
) -> dict[str, Any]:
    required = {
        "pred_boxes",
        "pred_logits_patch",
        "stage_b_data_driven_text_rank_score",
        "stage_b_data_driven_rank_score",
        "stage_b_data_driven_candidate_mask",
        "stage_b_data_driven_category_gate_eligible_mask",
        "stage_b_data_driven_category_gate_patch_score",
    }
    missing = sorted(required.difference(outputs))
    if missing:
        raise RuntimeError(f"HardGap3 deployment outputs are incomplete: {missing}")
    boxes = outputs["pred_boxes"]
    patch = outputs["pred_logits_patch"]
    raw_rank = outputs["stage_b_data_driven_text_rank_score"]
    deployed_rank = outputs["stage_b_data_driven_rank_score"]
    candidate = outputs["stage_b_data_driven_candidate_mask"]
    actual_eligible = outputs[
        "stage_b_data_driven_category_gate_eligible_mask"
    ]
    actual_patch = outputs["stage_b_data_driven_category_gate_patch_score"]
    if not all(
        torch.is_tensor(value)
        for value in (
            boxes,
            patch,
            raw_rank,
            deployed_rank,
            candidate,
            actual_eligible,
            actual_patch,
        )
    ):
        raise RuntimeError("HardGap3 deployment outputs must all be tensors")
    batch_size, query_count, expression_count = map(int, raw_rank.shape)
    if expression_count != 2:
        raise RuntimeError("HardGap3 deployment requires two expressions per row")
    expected_query_shape = (batch_size, query_count, expression_count)
    if any(
        tuple(value.shape) != expected_query_shape
        for value in (deployed_rank, candidate, actual_eligible, actual_patch)
    ):
        raise RuntimeError("HardGap3 deployment query outputs are misaligned")
    if candidate.dtype != torch.bool or actual_eligible.dtype != torch.bool:
        raise RuntimeError("HardGap3 deployment masks must be boolean")
    if not torch.equal(candidate[..., 0], candidate[..., 1]):
        raise RuntimeError("HardGap3 expressions changed the candidate mask")
    if patch.dim() == 3:
        if tuple(patch.shape) != (batch_size, query_count, 1):
            raise RuntimeError("HardGap3 patch logits are misaligned")
        patch_2d = patch[..., 0].float()
    elif patch.dim() == 2 and tuple(patch.shape) == (batch_size, query_count):
        patch_2d = patch.float()
    else:
        raise RuntimeError("HardGap3 patch logits are misaligned")
    if not bool(
        torch.isfinite(raw_rank).all().item()
        and torch.isfinite(deployed_rank).all().item()
        and torch.isfinite(patch).all().item()
        and torch.isfinite(actual_patch).all().item()
    ):
        raise RuntimeError("HardGap3 deployment outputs contain non-finite values")

    candidate_2d = candidate[..., 0]
    count = candidate_2d.sum(dim=1).clamp_min(1).float()
    safe = patch_2d.masked_fill(~candidate_2d, 0.0)
    mean = safe.sum(dim=1) / count
    centered = (patch_2d - mean[:, None]).masked_fill(~candidate_2d, 0.0)
    std = (centered.square().sum(dim=1) / count).clamp_min(1e-6).sqrt()
    normalized_patch = (centered / std[:, None]).clamp(-5.0, 5.0)
    best = normalized_patch.masked_fill(~candidate_2d, -torch.inf).amax(
        dim=1, keepdim=True
    )
    expected_eligible_2d = candidate_2d & (
        best - normalized_patch <= 3.0
    )
    expected_eligible = expected_eligible_2d[:, :, None].expand(
        -1, -1, expression_count
    )
    expected_patch = normalized_patch[:, :, None].expand(
        -1, -1, expression_count
    )
    eligible_exact = torch.equal(actual_eligible, expected_eligible)
    patch_max_abs_error = float(
        (actual_patch.float() - expected_patch).abs().amax().item()
    )
    patch_exact = patch_max_abs_error <= 1e-7

    expected_top = raw_rank.masked_fill(
        ~expected_eligible, -torch.inf
    ).argmax(dim=1)
    actual_top = deployed_rank.argmax(dim=1)
    top1_wiring = actual_top == expected_top

    raw_iou, raw_valid = _independent_assignment_iou(
        boxes, targets, clamp=False
    )
    clamped_iou, clamped_valid = _independent_assignment_iou(
        boxes, targets, clamp=True
    )
    if not torch.equal(raw_valid, clamped_valid):
        raise RuntimeError("raw/clamped assignment validity drifted")
    row_index = torch.arange(batch_size, device=raw_rank.device)[:, None]
    expression_index = torch.arange(
        expression_count, device=raw_rank.device
    )[None, :]

    def top_correct(iou: torch.Tensor) -> torch.Tensor:
        own = iou[row_index, actual_top, expression_index]
        cross = iou[row_index, actual_top, 1 - expression_index]
        return (
            (own >= 0.5)
            & (cross < 0.3)
            & raw_valid[:, None]
        )

    raw_correct = top_correct(raw_iou)
    clamped_correct = top_correct(clamped_iou)
    raw_query_correct = (
        (raw_iou >= 0.5)
        & (raw_iou.flip(-1) < 0.3)
        & raw_valid[:, None, None]
    )
    clamped_query_correct = (
        (clamped_iou >= 0.5)
        & (clamped_iou.flip(-1) < 0.3)
        & raw_valid[:, None, None]
    )
    direction_count = batch_size * expression_count
    counts = {
        "deployment_gate_enabled": int(bool(gate_runtime.get("evaluation_flag"))),
        "deployment_eligible_mask_exact": int(eligible_exact),
        "deployment_patch_score_exact": int(patch_exact),
        "deployment_top1_wiring_directions": int(top1_wiring.sum().item()),
        "deployment_actual_raw_correct_rows": int(
            (raw_correct.all(dim=1) & raw_valid).sum().item()
        ),
        "deployment_actual_raw_correct_directions": int(raw_correct.sum().item()),
        "deployment_actual_clamped_correct_rows": int(
            (clamped_correct.all(dim=1) & raw_valid).sum().item()
        ),
        "deployment_actual_clamped_correct_directions": int(
            clamped_correct.sum().item()
        ),
        "deployment_iou_correct_mask_disagreements": int(
            (raw_query_correct != clamped_query_correct).sum().item()
        ),
        "deployment_top1_iou_correct_disagreements": int(
            (raw_correct != clamped_correct).sum().item()
        ),
    }
    return {
        "gate_runtime": dict(gate_runtime),
        "rows": batch_size,
        "directions": direction_count,
        "eligible_queries": int(expected_eligible_2d.sum().item()),
        "patch_score_max_abs_error": patch_max_abs_error,
        "counts": counts,
    }


def _forward_batch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    batch: Sequence[Any],
    *,
    device: torch.device,
    amp: bool,
) -> tuple[dict[str, torch.Tensor], dict[str, Any], list[dict[str, torch.Tensor]]]:
    samples, raw_targets = batch
    samples = samples.to(device)
    raw_targets = list(raw_targets)
    captions, expression_captions = _build_stage_b_data_driven_assignment_captions(
        raw_targets
    )

    patches = None
    if all("patch" in target for target in raw_targets):
        patches = torch.stack(
            [target["patch"] for target in raw_targets], dim=0
        ).to(device, non_blocking=True)
    patch_global = None
    if all("patch_global" in target for target in raw_targets):
        patch_global = torch.stack(
            [target["patch_global"] for target in raw_targets], dim=0
        ).to(device, non_blocking=True)
    if patches is None and patch_global is None:
        raise RuntimeError("PairTop1 evaluation requires one support patch per row")

    targets = [
        {
            key: value.to(device)
            for key, value in target.items()
            if torch.is_tensor(value) and key not in {"patch", "patch_global"}
        }
        for target in raw_targets
    ]
    with torch.autocast(
        device_type=device.type,
        dtype=torch.float16,
        enabled=bool(amp and device.type == "cuda"),
    ):
        outputs = model(
            samples,
            captions=captions,
            patches=patches,
            patch_global=patch_global,
            stage_b_data_driven_expression_captions=expression_captions,
        )
        return criterion(outputs, targets), outputs, targets


def _pairtop1_gates(
    counts: Mapping[str, int], *, rows: int, variant: str = VARIANT
) -> dict[str, bool]:
    if variant not in SUPPORTED_VARIANTS:
        raise RuntimeError(f"unsupported gate variant: {variant!r}")
    data_rows = int(counts["data_rows"])
    runtime_rows = int(counts["runtime_rows"])
    gates = {
        "exact_64_data_rows": data_rows == EXPECTED_ROWS == rows,
        "all_rows_runtime_reachable": runtime_rows == data_rows,
        "no_query_collisions": int(counts["query_collision_rows"]) == 0,
        "reciprocal_correct_at_least_61_of_64": (
            int(counts["correct_rows"]) >= MIN_RECIPROCAL_ROWS
        ),
        "reciprocal_margin_at_least_61_of_64": (
            int(counts["margin_rows"]) >= MIN_RECIPROCAL_ROWS
        ),
        "direction_margin_at_least_122_of_128": (
            int(counts["margin_directions"]) >= MIN_MARGIN_DIRECTIONS
        ),
        "deployment_gap3_correct_at_least_61_of_64": (
            int(counts["deployment_correct_rows"]) >= MIN_RECIPROCAL_ROWS
        ),
    }
    if variant == HARDGAP3_VARIANT:
        gates.update(
            {
                "hardgap3_competitor_available_128_of_128": (
                    int(counts["deployment_hard_valid_directions"])
                    == 2 * runtime_rows
                ),
                "hardgap3_margin_at_least_122_of_128": (
                    int(counts["deployment_hard_margin_directions"])
                    >= MIN_MARGIN_DIRECTIONS
                ),
                "deployment_gap3_directions_at_least_122_of_128": (
                    int(counts["deployment_correct_directions"])
                    >= MIN_MARGIN_DIRECTIONS
                ),
                "actual_deployment_gate_enabled": (
                    int(counts["deployment_gate_enabled"]) == 1
                ),
                "actual_deployment_eligible_mask_exact": (
                    int(counts["deployment_eligible_mask_exact"]) == 1
                ),
                "actual_deployment_patch_score_exact": (
                    int(counts["deployment_patch_score_exact"]) == 1
                ),
                "actual_deployment_top1_wiring_128_of_128": (
                    int(counts["deployment_top1_wiring_directions"])
                    == 2 * runtime_rows
                ),
                "actual_deployment_raw_rows_at_least_61_of_64": (
                    int(counts["deployment_actual_raw_correct_rows"])
                    >= MIN_RECIPROCAL_ROWS
                ),
                "actual_deployment_raw_directions_at_least_122_of_128": (
                    int(counts["deployment_actual_raw_correct_directions"])
                    >= MIN_MARGIN_DIRECTIONS
                ),
                "actual_deployment_clamped_rows_at_least_61_of_64": (
                    int(counts["deployment_actual_clamped_correct_rows"])
                    >= MIN_RECIPROCAL_ROWS
                ),
                "actual_deployment_clamped_directions_at_least_122_of_128": (
                    int(counts[
                        "deployment_actual_clamped_correct_directions"
                    ])
                    >= MIN_MARGIN_DIRECTIONS
                ),
            }
        )
    return gates


def _build_result(
    loss_dict: Mapping[str, Any],
    *,
    config_path: Path,
    dataset_path: Path,
    checkpoint_path: Path,
    checkpoint_kind: str,
    training_audit: Mapping[str, Any],
    rows: int,
    amp: bool,
    variant: str,
    deployment_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if variant not in SUPPORTED_VARIANTS:
        raise RuntimeError(f"unsupported result variant: {variant!r}")
    count_keys = COUNT_KEYS + (
        HARDGAP3_COUNT_KEYS if variant == HARDGAP3_VARIANT else ()
    )
    mean_keys = MEAN_KEYS + (
        HARDGAP3_MEAN_KEYS if variant == HARDGAP3_VARIANT else ()
    )
    missing = sorted(set(count_keys + mean_keys).difference(loss_dict))
    if missing:
        raise RuntimeError(f"PairTop1 criterion metrics are incomplete: {missing}")
    counts = {
        key.removeprefix("stage_b_data_driven_assignment_").removeprefix(
            "stage_b_data_driven_"
        ): _integer_metric(loss_dict[key], key=key)
        for key in count_keys
    }
    if variant == HARDGAP3_VARIANT:
        if not isinstance(deployment_audit, Mapping):
            raise RuntimeError("HardGap3 result lacks an independent deployment audit")
        deployment_counts = deployment_audit.get("counts")
        if not isinstance(deployment_counts, Mapping):
            raise RuntimeError("HardGap3 deployment audit counts are missing")
        missing_deployment = sorted(
            set(HARDGAP3_DEPLOYMENT_COUNT_KEYS).difference(deployment_counts)
        )
        if missing_deployment:
            raise RuntimeError(
                "HardGap3 deployment audit is incomplete: "
                f"{missing_deployment}"
            )
        for key in HARDGAP3_DEPLOYMENT_COUNT_KEYS:
            value = deployment_counts[key]
            if isinstance(value, bool) or not isinstance(value, int):
                raise RuntimeError(
                    f"HardGap3 deployment count {key!r} is not integral"
                )
            counts[key] = int(value)
    means = {
        key.removeprefix("stage_b_data_driven_assignment_").removeprefix(
            "loss_stage_b_data_driven_"
        ): _finite_scalar(loss_dict[key], key=key)
        for key in mean_keys
    }
    data_rows = counts["data_rows"]
    runtime_rows = counts["runtime_rows"]
    directions = 2 * runtime_rows
    rates = {
        "runtime_reachable": runtime_rows / data_rows if data_rows else 0.0,
        "reciprocal_correct": counts["correct_rows"] / runtime_rows
        if runtime_rows
        else 0.0,
        "reciprocal_margin": counts["margin_rows"] / runtime_rows
        if runtime_rows
        else 0.0,
        "deployment_reciprocal_correct": (
            counts["deployment_correct_rows"] / runtime_rows
            if runtime_rows
            else 0.0
        ),
        "direction_correct": counts["correct_directions"] / directions
        if directions
        else 0.0,
        "direction_margin": counts["margin_directions"] / directions
        if directions
        else 0.0,
        "deployment_direction_correct": (
            counts["deployment_correct_directions"] / directions
            if directions
            else 0.0
        ),
    }
    if variant == HARDGAP3_VARIANT:
        rates.update(
            {
                "actual_deployment_raw_reciprocal_correct": (
                    counts["deployment_actual_raw_correct_rows"] / runtime_rows
                    if runtime_rows
                    else 0.0
                ),
                "actual_deployment_raw_direction_correct": (
                    counts["deployment_actual_raw_correct_directions"] / directions
                    if directions
                    else 0.0
                ),
                "actual_deployment_clamped_reciprocal_correct": (
                    counts["deployment_actual_clamped_correct_rows"] / runtime_rows
                    if runtime_rows
                    else 0.0
                ),
                "actual_deployment_clamped_direction_correct": (
                    counts["deployment_actual_clamped_correct_directions"]
                    / directions
                    if directions
                    else 0.0
                ),
            }
        )
    gates = _pairtop1_gates(counts, rows=rows, variant=variant)
    return {
        "schema": SCHEMA,
        "variant": variant,
        "checkpoint_kind": checkpoint_kind,
        "inputs": {
            "config": {
                "path": str(config_path),
                "sha256": _sha256_file(config_path),
            },
            "dataset_config": {
                "path": str(dataset_path),
                "sha256": _sha256_file(dataset_path),
            },
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": _sha256_file(checkpoint_path),
            },
        },
        "runtime": {
            "rows": rows,
            "batch_size": rows,
            "sequential": True,
            "model_mode": "eval",
            "amp": bool(amp),
            "updates": 0,
        },
        "counts": counts,
        "means": means,
        "rates": rates,
        "training_audit": dict(training_audit),
        "independent_deployment_audit": (
            dict(deployment_audit)
            if isinstance(deployment_audit, Mapping)
            else None
        ),
        "gates": gates,
        "passed": all(gates.values()),
    }


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="ascii",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as handle:
        handle.write(serialized)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--datasets", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-amp", action="store_true")
    return parser


def main() -> None:
    cli = _parser().parse_args()
    config_path = cli.config.expanduser().resolve(strict=True)
    dataset_path = cli.datasets.expanduser().resolve(strict=True)
    checkpoint_path = cli.checkpoint.expanduser().resolve(strict=True)
    output_path = cli.output_json.expanduser().resolve()
    device = torch.device(cli.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    cfg = SLConfig.fromfile(str(config_path))
    variant = _config_variant(cfg)
    base_path = _validate_config(cfg, dataset_path)
    cfg.device = str(device)
    cfg.amp = not bool(cli.no_amp)
    cfg.distributed = False
    dataset_meta = _load_dataset_config(dataset_path)
    _seed_everything(int(cfg.get("stage_b_data_driven_sampler_seed", 42)))
    dataset = build_dataset(
        image_set="train", args=cfg, datasetinfo=dataset_meta["train"][0]
    )
    if len(dataset) != EXPECTED_ROWS:
        raise RuntimeError(
            f"PairTop1 Overfit64 dataset has {len(dataset)} rows, expected 64"
        )
    if getattr(dataset, "sample_weights", None) is not None:
        raise RuntimeError("PairTop1 Overfit64 evaluation forbids weighted sampling")
    loader = DataLoader(
        dataset,
        batch_size=EXPECTED_ROWS,
        sampler=SequentialSampler(dataset),
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=0,
        pin_memory=False,
    )
    if len(loader) != 1:
        raise RuntimeError("PairTop1 Overfit64 evaluator requires one exact batch")

    model, criterion, _ = build_model_main(cfg)
    checkpoint_kind, payload = _load_state(
        model, criterion, checkpoint_path
    )
    if checkpoint_kind != "trained_checkpoint":
        raise RuntimeError("PairTop1 gate rejects an untrained A1 initializer")
    training_audit = _audit_terminal_checkpoint(
        payload,
        cfg=cfg,
        config_path=config_path,
        dataset_path=dataset_path,
        checkpoint_path=checkpoint_path,
        base_path=base_path,
    )
    model.to(device).eval()
    criterion.to(device).eval()
    gate_runtime = None
    if variant == HARDGAP3_VARIANT:
        gate_runtime = _enable_hardgap3_deployment_gate(model)
    amp = not bool(cli.no_amp)
    with torch.inference_mode():
        loss_dict, outputs, targets = _forward_batch(
            model,
            criterion,
            next(iter(loader)),
            device=device,
            amp=amp,
        )
        deployment_audit = (
            _audit_hardgap3_deployment_outputs(
                outputs,
                targets,
                gate_runtime=gate_runtime,
            )
            if variant == HARDGAP3_VARIANT
            else None
        )
    result = _build_result(
        loss_dict,
        config_path=config_path,
        dataset_path=dataset_path,
        checkpoint_path=checkpoint_path,
        checkpoint_kind=checkpoint_kind,
        training_audit=training_audit,
        rows=len(dataset),
        amp=amp,
        variant=variant,
        deployment_audit=deployment_audit,
    )
    _atomic_write_json(output_path, result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
