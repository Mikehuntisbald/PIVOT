#!/usr/bin/env python3
"""Seal the formal U2 B56/U100 category-complete training evidence.

This builder is deliberately tied to the one formal U2 artifact. It safely
loads checkpoints with ``weights_only=True``, replays the U0 transition audit,
checks optimizer/AMP state, replays the category-complete data receipt, and
records the vanished ``/tmp`` training origin separately from the durable
checkpoint location.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.GroundingDINO.stage_b_u0_patch_rank import (  # noqa: E402
    stage_b_u0_tensor_state_sha256,
)
from tools.audit_stageb_u0_transition import audit_u0_transition  # noqa: E402
from tools.build_stageb_u0_training_receipt import (  # noqa: E402
    TRANSITION_SCHEMA,
    U0TrainingReceiptError,
    _ALLOWED_CHECKPOINT_GLOBALS,
    _atomic_publish_fresh_json,
    _checkpoint_args,
    _config_binding,
    _dataset_binding,
    _initializer_binding,
    _json_safe,
    _resolve_file,
    _resolve_runtime_path,
    _safe_load_checkpoint,
    _same_path,
    _seal_payload,
    _strict_json_load,
    canonical_json_sha256,
    stable_file_record,
)
from util.path_compat import default_data_root  # noqa: E402


SCHEMA = "pivot.stageb.u2_training_receipt/v1"
DATA_RECEIPT_SCHEMA = "pivot.stageb.u2_category_complete_receipt/v1"
DATA_ROW_SCHEMA = "pivot.stageb.u2_category_complete_ref/v1"
EXPECTED_BATCH_SIZE = 56
EXPECTED_SEED = 42
EXPECTED_UPDATES = 100
EXPECTED_AMP_SCALE = 8192.0
EXPECTED_ORIGIN_OUTPUT_DIR = "/tmp/pivot_u2_category_b56_u100_scale8192_v1"

FORMAL_ROOT = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/u2_category_complete_seed42_b56_scale8192_v2"
)
CHECKPOINT = FORMAL_ROOT / "checkpoint_iter.pth"
TRANSITION_AUDIT = FORMAL_ROOT / "audits/checkpoint_iter_000100.transition.json"
INITIALIZER = (
    REPO_ROOT
    / "outputs/paper_cvpr_v1/u0_single_network_seed42_b56_v1/initializer/checkpoint_u0_init.pth"
)
CONFIG = REPO_ROOT / "config/ablations/cfg_stageb_u2_category_complete_patch_rank.py"
DATASETS = REPO_ROOT / "config/datasets_stageb_u2_category_complete_three_ref.json"
DATA_RECEIPT = (
    REPO_ROOT
    / "data/ablations/stageb_refexp_three_train_category_complete_20260720/receipt.json"
)
CONFIG_ARGS_ALL = FORMAL_ROOT / "config_args_all.json"
CONFIG_ARGS_RAW = FORMAL_ROOT / "config_args_raw.json"
CONFIG_SNAPSHOT = FORMAL_ROOT / "config_cfg.py"
TRAIN_LOG = FORMAL_ROOT / "info.txt"
DEFAULT_OUTPUT = FORMAL_ROOT / "training_receipt.json"

EXPECTED_FILE_SHA256 = {
    "checkpoint": "44e3d70b164eff2bcefacc37081b7cbab184a9373720ef69713d47949d449b90",
    "initializer": "c89e5dfba795fd8074a044f0c09d81c871705c20a1dbf819b9f16c770a2cba43",
    "transition_audit": "ba2e2c18774918f9b2beffd6246e4bfd77493b15c5243efd05ac3251c34545e4",
    "data_receipt": "fab09c61a8f53f05d75eedff25039a843ff27cb2d491d6c6576fe2b1e8aedd74",
    "config_args_all": "1a62dc75f198c83a114f25d10cfd5ca014ec8e5e7be1098021bb5c5fd10e5bbf",
    "config_args_raw": "2901af2e6615ca73f109bef67635342a43e27c21150ca00ea1a31009626b3f14",
    "config_snapshot": "9e308567d99a1e8367d402b03b7050f1d79d0eb53aa6ab932df41f5469d8b3cc",
    "train_log": "af748e81237d7e06d849fe08fb57a6c853c48580a5d3e64139cdb2ca4c94a0b8",
}

EXPECTED_MANIFEST_TOTALS = {
    "rows": 321_327,
    "instances": 1_361_554,
    "auxiliary_instances": 1_040_227,
    "multi_instance_rows": 321_327,
}

CORE_SOURCE_PATHS = (
    "main.py",
    "engine.py",
    "datasets/__init__.py",
    "datasets/patch_episode.py",
    "datasets/transforms.py",
    "models/__init__.py",
    "models/GroundingDINO/__init__.py",
    "models/GroundingDINO/groundingdino.py",
    "models/GroundingDINO/stage_b_gdino_score_adapter.py",
    "models/GroundingDINO/stage_b_u0_patch_rank.py",
    "tools/audit_stageb_u0_transition.py",
    "tools/build_stageb_u0_initializer.py",
    "tools/build_stageb_u0_training_receipt.py",
    "tools/build_stageb_u2_category_complete_ref.py",
    "tools/build_stageb_u2_training_receipt.py",
    "tools/stageb_dependency_audit.py",
    "util/path_compat.py",
)

_LOG_METRIC_RE = re.compile(
    r"Epoch: \[0\]\s+\[\s*(?P<iteration>\d+)/(?P<epoch_size>\d+)\].*?"
    r"amp_step_skipped:\s*(?P<skip>[0-9.]+)\s*\((?P<skip_avg>[0-9.]+)\).*?"
    r"amp_scale:\s*(?P<scale>[0-9.]+)\s*\((?P<scale_avg>[0-9.]+)\).*?"
    r"optimizer_step:\s*(?P<step>[0-9.]+)\s*\((?P<step_avg>[0-9.]+)\).*?"
    r"max mem:\s*(?P<max_mem>\d+)"
)
_SAVED_RE = re.compile(
    r"Saved iteration checkpoint to (?P<path>\S+) "
    r"\(epoch=0, next_iter=(?P<iteration>\d+), "
    r"optimizer_updates=(?P<updates>\d+), reason=(?P<reason>[^)]+)\)\."
)


class U2TrainingReceiptError(U0TrainingReceiptError):
    """The formal U2 evidence does not satisfy its sealed-run contract."""


def _expect_file(path: Path, *, role: str) -> dict[str, Any]:
    record = stable_file_record(path, label=role)
    expected = EXPECTED_FILE_SHA256[role]
    if record["sha256"] != expected:
        raise U2TrainingReceiptError(
            f"{role} SHA256 drifted: expected {expected}, got {record['sha256']}"
        )
    return record


def _core_source_binding() -> list[dict[str, Any]]:
    result = []
    for relative in CORE_SOURCE_PATHS:
        record = stable_file_record(
            REPO_ROOT / relative, label=f"core source {relative}"
        )
        result.append({"relative_path": relative, **record})
    return result


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise U2TrainingReceiptError(f"{label} is not an object")
    return value


def _strict_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise U2TrainingReceiptError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise U2TrainingReceiptError(f"{label} is not finite")
    return result


def _scalar_int(value: Any, *, label: str) -> int:
    if torch.is_tensor(value):
        if value.numel() != 1:
            raise U2TrainingReceiptError(f"{label} is not scalar")
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise U2TrainingReceiptError(f"{label} is not an integer scalar")
    if not math.isfinite(float(value)) or int(value) != float(value):
        raise U2TrainingReceiptError(f"{label} is not an integer scalar")
    return int(value)


def _assert_selected_args(args: Mapping[str, Any], *, data_root: Path) -> None:
    exact = {
        "batch_size": EXPECTED_BATCH_SIZE,
        "seed": EXPECTED_SEED,
        "max_train_iters": EXPECTED_UPDATES,
        "iter_checkpoint_interval": EXPECTED_UPDATES,
        "gradient_accumulation_steps": 1,
        "world_size": 1,
        "num_workers": 8,
        "prefetch_factor": 1,
        "amp_init_scale": EXPECTED_AMP_SCALE,
        "stage_b_u0_patch_rank_lr": 3e-4,
        "stage_b_u0_patch_projection_lr": 3e-4,
        "stage_b_u2_category_loss_weight": 1.0,
        "stage_b_u2_category_negative_iou_threshold": 0.3,
        "stage_b_u2_category_margin": 0.1,
        "stage_b_u2_target_preserve_weight": 1.0,
        "stage_b_gdino_adapter_train_mode": "rank_only",
        "stage_b_gdino_tn_scope": "image_global_topk_verified",
    }
    for field, expected in exact.items():
        observed = args.get(field)
        if isinstance(expected, float):
            valid = (
                not isinstance(observed, bool)
                and isinstance(observed, (int, float))
                and float(observed) == expected
            )
        else:
            valid = observed == expected and not (
                isinstance(expected, int) and isinstance(observed, bool)
            )
        if not valid:
            raise U2TrainingReceiptError(
                f"checkpoint args.{field} must be {expected!r}, got {observed!r}"
            )
    for field in (
        "amp",
        "stage_b_u0_patch_rank",
        "stage_b_u2_category_complete_supervision",
        "enable_patch_branch",
        "stage_b_gdino_score_adapter",
        "skip_eval",
    ):
        if args.get(field) is not True:
            raise U2TrainingReceiptError(f"checkpoint args.{field} must be true")
    if args.get("distributed") is not False:
        raise U2TrainingReceiptError("formal U2 must be single-process")
    if args.get("resume") not in (None, ""):
        raise U2TrainingReceiptError("formal U2 must start directly from its initializer")
    if args.get("output_dir") != EXPECTED_ORIGIN_OUTPUT_DIR:
        raise U2TrainingReceiptError(
            "checkpoint must retain the exact original /tmp output_dir"
        )
    _same_path(
        args.get("config_file"), CONFIG.resolve(), label="U2 config", data_root=data_root
    )
    _same_path(
        args.get("datasets"), DATASETS.resolve(), label="U2 datasets", data_root=data_root
    )
    _same_path(
        args.get("pretrain_model_path"),
        INITIALIZER.resolve(),
        label="U2 initializer",
        data_root=data_root,
    )


def _optimizer_binding(payload: Mapping[str, Any]) -> dict[str, Any]:
    optimizer = _mapping(payload.get("optimizer"), label="optimizer")
    groups = optimizer.get("param_groups")
    states = optimizer.get("state")
    if not isinstance(groups, list) or len(groups) != 2:
        raise U2TrainingReceiptError("formal U2 optimizer must contain exactly 2 groups")
    states = _mapping(states, label="optimizer.state")

    expected_groups = (
        ("patch_rank_residual", 8),
        ("patch_projection", 8),
    )
    summaries = []
    all_parameter_ids: list[Any] = []
    for index, ((expected_branch, expected_count), raw_group) in enumerate(
        zip(expected_groups, groups)
    ):
        group = _mapping(raw_group, label=f"optimizer.param_groups[{index}]")
        parameters = group.get("params")
        if not isinstance(parameters, list) or len(parameters) != expected_count:
            raise U2TrainingReceiptError(
                f"optimizer group {expected_branch} must contain {expected_count} parameters"
            )
        if group.get("stage_b_u0_branch") != expected_branch:
            raise U2TrainingReceiptError(
                f"optimizer group {index} is not {expected_branch}"
            )
        for field, expected in (
            ("lr", 3e-4),
            ("initial_lr", 3e-4),
            ("weight_decay", 1e-4),
        ):
            if _strict_number(group.get(field), label=f"group {index}.{field}") != expected:
                raise U2TrainingReceiptError(
                    f"optimizer group {expected_branch} {field} drifted"
                )
        if list(group.get("betas", [])) != [0.9, 0.999] or group.get("eps") != 1e-8:
            raise U2TrainingReceiptError(
                f"optimizer group {expected_branch} AdamW hyperparameters drifted"
            )
        all_parameter_ids.extend(parameters)
        summaries.append(
            {
                "index": index,
                "branch": expected_branch,
                "parameter_count": len(parameters),
                "lr": float(group["lr"]),
                "initial_lr": float(group["initial_lr"]),
                "weight_decay": float(group["weight_decay"]),
                "betas": list(group["betas"]),
                "eps": float(group["eps"]),
            }
        )
    if len(set(all_parameter_ids)) != len(all_parameter_ids):
        raise U2TrainingReceiptError("optimizer parameter groups overlap")
    if set(states) != set(all_parameter_ids):
        raise U2TrainingReceiptError(
            "optimizer states do not cover the two parameter groups exactly"
        )
    step_values = []
    for parameter_id in all_parameter_ids:
        state = _mapping(states[parameter_id], label=f"optimizer.state[{parameter_id}]")
        if not torch.is_tensor(state.get("exp_avg")) or not torch.is_tensor(
            state.get("exp_avg_sq")
        ):
            raise U2TrainingReceiptError("optimizer moment state is incomplete")
        step_values.append(
            _scalar_int(state.get("step"), label=f"optimizer.state[{parameter_id}].step")
        )
    if set(step_values) != {EXPECTED_UPDATES}:
        raise U2TrainingReceiptError(
            f"optimizer states are not all at step {EXPECTED_UPDATES}: {sorted(set(step_values))}"
        )
    return {
        "type": "AdamW",
        "group_count": len(groups),
        "groups": summaries,
        "state_count": len(states),
        "state_step_values": sorted(set(step_values)),
        "groups_disjoint": True,
        "states_cover_groups_exactly": True,
    }


def _amp_binding(payload: Mapping[str, Any]) -> dict[str, Any]:
    scaler = _mapping(payload.get("scaler"), label="AMP scaler")
    expected = {
        "scale": EXPECTED_AMP_SCALE,
        "_growth_tracker": EXPECTED_UPDATES,
        "growth_factor": 2.0,
        "backoff_factor": 0.5,
        "growth_interval": 2000,
    }
    for field, value in expected.items():
        observed = _strict_number(scaler.get(field), label=f"scaler.{field}")
        if observed != float(value):
            raise U2TrainingReceiptError(
                f"scaler.{field} must be {value!r}, got {observed!r}"
            )
    iteration = payload.get("iteration")
    updates = payload.get("optimizer_updates")
    if type(iteration) is not int or type(updates) is not int:
        raise U2TrainingReceiptError("checkpoint iteration/update counters are invalid")
    if iteration != EXPECTED_UPDATES or updates != EXPECTED_UPDATES:
        raise U2TrainingReceiptError("formal U2 checkpoint is not exact U100")
    return {
        "enabled": True,
        "initial_scale": EXPECTED_AMP_SCALE,
        "final_scale": float(scaler["scale"]),
        "growth_tracker": int(scaler["_growth_tracker"]),
        "growth_interval": int(scaler["growth_interval"]),
        "backoff_factor": float(scaler["backoff_factor"]),
        "optimizer_updates": updates,
        "micro_iterations": iteration,
        "gradient_accumulation_steps": 1,
        "amp_skipped_steps": 0,
        "zero_skip_derivation": {
            "iteration_equals_successful_optimizer_updates": True,
            "scale_never_backed_off_before_growth_interval": True,
            "growth_tracker_equals_all_100_updates": True,
        },
    }


def _train_log_binding(path: Path) -> dict[str, Any]:
    record = _expect_file(path, role="train_log")
    try:
        text = Path(record["path"]).read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise U2TrainingReceiptError(f"could not read training log: {error}") from error
    command_lines = [line for line in text.splitlines() if " | Command: " in line]
    if len(command_lines) != 1:
        raise U2TrainingReceiptError("training log must contain exactly one command")
    command = shlex.split(command_lines[0].split(" | Command: ", 1)[1])
    required_tokens = {
        "main.py",
        "--amp",
        "--save_log",
        "batch_size=56",
        "--max_train_iters",
        "100",
        EXPECTED_ORIGIN_OUTPUT_DIR,
    }
    if not required_tokens.issubset(command):
        raise U2TrainingReceiptError("training command contract drifted")

    metrics = [match.groupdict() for match in _LOG_METRIC_RE.finditer(text)]
    expected_iterations = list(range(0, EXPECTED_UPDATES, 10))
    if [int(item["iteration"]) for item in metrics] != expected_iterations:
        raise U2TrainingReceiptError("training log metric cadence is incomplete")
    for item in metrics:
        if any(float(item[field]) != 0.0 for field in ("skip", "skip_avg")):
            raise U2TrainingReceiptError("training log contains an AMP skip")
        if any(
            float(item[field]) != EXPECTED_AMP_SCALE
            for field in ("scale", "scale_avg")
        ):
            raise U2TrainingReceiptError("training log AMP scale drifted")
        if any(float(item[field]) != 1.0 for field in ("step", "step_avg")):
            raise U2TrainingReceiptError("training log contains a failed optimizer step")

    saved = list(_SAVED_RE.finditer(text))
    if len(saved) != 1:
        raise U2TrainingReceiptError("training log must contain one final checkpoint save")
    saved_value = saved[0].groupdict()
    expected_origin_checkpoint = f"{EXPECTED_ORIGIN_OUTPUT_DIR}/checkpoint_iter.pth"
    if (
        saved_value["path"] != expected_origin_checkpoint
        or int(saved_value["iteration"]) != EXPECTED_UPDATES
        or int(saved_value["updates"]) != EXPECTED_UPDATES
        or saved_value["reason"] != "max_train_iters"
    ):
        raise U2TrainingReceiptError("training log final save lineage drifted")
    if f"Reached max_train_iters={EXPECTED_UPDATES} optimizer updates" not in text:
        raise U2TrainingReceiptError("training log lacks the U100 completion marker")
    return {
        "file": record,
        "command": command,
        "logged_metric_iterations": expected_iterations,
        "logged_amp_step_skipped_max": 0.0,
        "logged_amp_scale_min": EXPECTED_AMP_SCALE,
        "logged_amp_scale_max": EXPECTED_AMP_SCALE,
        "logged_optimizer_step_min": 1.0,
        "logged_optimizer_step_max": 1.0,
        "peak_allocated_mib": max(int(item["max_mem"]) for item in metrics),
        "saved_origin_checkpoint": expected_origin_checkpoint,
        "completion_marker_verified": True,
    }


def _bind_declared_file(
    declared: Any, *, label: str, data_root: Path
) -> dict[str, Any]:
    value = _mapping(declared, label=label)
    if not isinstance(value.get("sha256"), str) or type(value.get("size_bytes")) is not int:
        raise U2TrainingReceiptError(f"{label} lacks a strict file identity")
    path = _resolve_runtime_path(value.get("path"), label=label, data_root=data_root)
    record = stable_file_record(path, label=label)
    if (
        record["sha256"] != value["sha256"]
        or record["size_bytes"] != value["size_bytes"]
    ):
        raise U2TrainingReceiptError(f"{label} no longer matches its data receipt")
    return record


def _data_receipt_binding(
    path: Path, *, datasets: Mapping[str, Any], data_root: Path
) -> dict[str, Any]:
    record = _expect_file(path, role="data_receipt")
    payload = _strict_json_load(Path(record["path"]), label="U2 data receipt")
    payload = _mapping(payload, label="U2 data receipt")
    if payload.get("schema") != DATA_RECEIPT_SCHEMA or payload.get("row_schema") != DATA_ROW_SCHEMA:
        raise U2TrainingReceiptError("U2 data receipt schema drifted")
    declared_canonical = payload.get("canonical_payload_sha256")
    canonical_payload = {
        key: value
        for key, value in payload.items()
        if key != "canonical_payload_sha256"
    }
    recomputed_canonical = canonical_json_sha256(canonical_payload)
    if declared_canonical != recomputed_canonical:
        raise U2TrainingReceiptError("U2 data receipt canonical payload hash drifted")
    expected_invariants = {
        "all_noncrowd_same_category_coco_instances_included": True,
        "primary_support_instance_index_zero": True,
        "source_expression_instance_preserved_at_index_zero": True,
        "target_annotation_matched_by_ann_id_and_iou_at_least_0_99": True,
    }
    if payload.get("invariants") != expected_invariants:
        raise U2TrainingReceiptError("U2 data receipt invariants drifted")

    manifests = _mapping(payload.get("manifests"), label="data receipt manifests")
    annotations = {
        Path(item["annotation"]["path"]).name: item["annotation"]
        for item in datasets["train"]
    }
    if set(manifests) != set(annotations):
        raise U2TrainingReceiptError("dataset JSON and data receipt manifest sets differ")
    manifest_bindings = {}
    totals = {key: 0 for key in EXPECTED_MANIFEST_TOTALS}
    for name in sorted(manifests):
        manifest = _mapping(manifests[name], label=f"manifest {name}")
        output = annotations[name]
        declared_output = _bind_declared_file(
            manifest.get("output"), label=f"manifest {name} output", data_root=data_root
        )
        if (
            declared_output["path"] != output["path"]
            or declared_output["sha256"] != output["sha256"]
            or declared_output["size_bytes"] != output["size_bytes"]
            or manifest.get("rows") != output["parsed_rows"]
        ):
            raise U2TrainingReceiptError(
                f"manifest {name} does not match the fully parsed dataset input"
            )
        source = _bind_declared_file(
            manifest.get("source"), label=f"manifest {name} source", data_root=data_root
        )
        for key in totals:
            value = manifest.get(key)
            if type(value) is not int or value < 0:
                raise U2TrainingReceiptError(f"manifest {name}.{key} is invalid")
            totals[key] += value
        manifest_bindings[name] = {
            "rows": manifest["rows"],
            "instances": manifest["instances"],
            "auxiliary_instances": manifest["auxiliary_instances"],
            "multi_instance_rows": manifest["multi_instance_rows"],
            "max_instances_per_row": manifest.get("max_instances_per_row"),
            "output": declared_output,
            "source": source,
        }
    if totals != EXPECTED_MANIFEST_TOTALS:
        raise U2TrainingReceiptError(
            f"category-complete manifest totals drifted: {totals}"
        )

    coco = _mapping(payload.get("coco_annotations"), label="COCO annotations")
    if set(coco) != {"train2014", "val2014"}:
        raise U2TrainingReceiptError("COCO annotation sources drifted")
    coco_bindings = {}
    for split in sorted(coco):
        value = _mapping(coco[split], label=f"COCO {split}")
        count = value.get("noncrowd_annotations")
        if type(count) is not int or count <= 0:
            raise U2TrainingReceiptError(f"COCO {split} count is invalid")
        coco_bindings[split] = {
            "noncrowd_annotations": count,
            "file": _bind_declared_file(
                value, label=f"COCO {split} annotations", data_root=data_root
            ),
        }
    return {
        "file": record,
        "schema": payload["schema"],
        "row_schema": payload["row_schema"],
        "canonical_payload_sha256": recomputed_canonical,
        "canonical_payload_recomputed_equal": True,
        "invariants": dict(expected_invariants),
        "manifest_totals": totals,
        "manifests": manifest_bindings,
        "coco_annotations": coco_bindings,
    }


def _transition_binding(
    initializer_payload: Mapping[str, Any],
    checkpoint_payload: Mapping[str, Any],
    initializer_record: Mapping[str, Any],
    checkpoint_record: Mapping[str, Any],
) -> dict[str, Any]:
    audit_record = _expect_file(TRANSITION_AUDIT, role="transition_audit")
    observed = _strict_json_load(Path(audit_record["path"]), label="U2 transition audit")
    observed = _mapping(observed, label="U2 transition audit")
    try:
        recomputed = audit_u0_transition(initializer_payload, checkpoint_payload)
    except (TypeError, ValueError, RuntimeError) as error:
        raise U2TrainingReceiptError(f"U2 transition replay failed: {error}") from error
    recomputed["initializer"] = dict(initializer_record)
    recomputed["checkpoint"] = dict(checkpoint_record)
    if dict(observed) != recomputed:
        raise U2TrainingReceiptError(
            "formal transition audit does not equal a fresh safe-load replay"
        )
    if (
        recomputed.get("schema") != TRANSITION_SCHEMA
        or recomputed.get("status") != "verified"
        or recomputed.get("iteration") != EXPECTED_UPDATES
        or recomputed.get("optimizer_updates") != EXPECTED_UPDATES
        or recomputed.get("changed_key_count") != 16
        or recomputed.get("frozen_key_count") != 1149
    ):
        raise U2TrainingReceiptError("formal U2 transition contract drifted")
    return {
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


def _snapshot_binding(checkpoint_args: Mapping[str, Any]) -> dict[str, Any]:
    all_record = _expect_file(CONFIG_ARGS_ALL, role="config_args_all")
    all_args = _strict_json_load(Path(all_record["path"]), label="effective args snapshot")
    if not isinstance(all_args, Mapping) or dict(all_args) != dict(checkpoint_args):
        raise U2TrainingReceiptError(
            "checkpoint args do not equal the durable effective-args snapshot"
        )
    raw_record = _expect_file(CONFIG_ARGS_RAW, role="config_args_raw")
    raw = _strict_json_load(Path(raw_record["path"]), label="raw args snapshot")
    raw = _mapping(raw, label="raw args snapshot")
    expected_raw = {
        "config_file": str(CONFIG.relative_to(REPO_ROOT)),
        "datasets": str(DATASETS.relative_to(REPO_ROOT)),
        "output_dir": EXPECTED_ORIGIN_OUTPUT_DIR,
        "seed": EXPECTED_SEED,
        "resume": "",
        "pretrain_model_path": str(INITIALIZER.relative_to(REPO_ROOT)),
        "max_train_iters": EXPECTED_UPDATES,
        "iter_checkpoint_interval": EXPECTED_UPDATES,
        "gradient_accumulation_steps": 1,
        "world_size": 1,
        "amp": True,
        "distributed": False,
        "options": {"batch_size": EXPECTED_BATCH_SIZE, "epochs": 1},
    }
    for field, expected in expected_raw.items():
        if raw.get(field) != expected:
            raise U2TrainingReceiptError(f"raw args snapshot {field} drifted")
    return {
        "effective_args": all_record,
        "effective_args_equal_checkpoint_args": True,
        "raw_args": raw_record,
        "raw_cli_contract": expected_raw,
        "flattened_config": _expect_file(CONFIG_SNAPSHOT, role="config_snapshot"),
    }


def build_receipt_payload(*, data_root: Path | None = None) -> dict[str, Any]:
    data_root = (data_root or default_data_root()).expanduser().resolve()
    checkpoint = _resolve_file(CHECKPOINT, label="formal U2 checkpoint")
    initializer = _resolve_file(INITIALIZER, label="formal U0 initializer")
    checkpoint_record = _expect_file(checkpoint, role="checkpoint")
    initializer_record = _expect_file(initializer, role="initializer")

    config_binding = _config_binding(CONFIG)
    datasets_binding = _dataset_binding(DATASETS, data_root=data_root)
    data_binding = _data_receipt_binding(
        DATA_RECEIPT, datasets=datasets_binding, data_root=data_root
    )
    log_binding = _train_log_binding(TRAIN_LOG)

    initializer_payload = _safe_load_checkpoint(initializer, label="formal U0 initializer")
    initializer_summary, frozen_keys, initializer_frozen_hash = _initializer_binding(
        initializer_payload, initializer_record
    )
    checkpoint_payload = _safe_load_checkpoint(checkpoint, label="formal U2 checkpoint")
    if checkpoint_payload.get("checkpoint_reason") != "max_train_iters":
        raise U2TrainingReceiptError("formal U2 checkpoint reason drifted")
    args = _checkpoint_args(checkpoint_payload, label="formal U2 checkpoint")
    _assert_selected_args(args, data_root=data_root)
    optimizer = _optimizer_binding(checkpoint_payload)
    amp = _amp_binding(checkpoint_payload)
    model = checkpoint_payload.get("model")
    if not isinstance(model, MutableMapping) or not model:
        raise U2TrainingReceiptError("formal U2 checkpoint lacks model state")
    if any(not isinstance(key, str) or not torch.is_tensor(value) for key, value in model.items()):
        raise U2TrainingReceiptError("formal U2 model state is not tensor-only")
    transition = _transition_binding(
        initializer_payload,
        checkpoint_payload,
        initializer_record,
        checkpoint_record,
    )
    checkpoint_frozen_hash = stage_b_u0_tensor_state_sha256(model, frozen_keys)
    if checkpoint_frozen_hash != initializer_frozen_hash:
        raise U2TrainingReceiptError("formal U2 frozen tensors differ from initializer")
    snapshots = _snapshot_binding(args)

    origin_checkpoint = Path(EXPECTED_ORIGIN_OUTPUT_DIR) / checkpoint.name
    result = {
        "schema": SCHEMA,
        "repository_root": str(REPO_ROOT),
        "formal_experiment_root": str(FORMAL_ROOT),
        "data_root": str(data_root),
        "checkpoint_load_policy": {
            "weights_only": True,
            "mmap": True,
            "fallback_to_weights_only_false": False,
            "allowed_pickle_globals": sorted(_ALLOWED_CHECKPOINT_GLOBALS),
        },
        "checkpoint": {
            "file": checkpoint_record,
            "epoch": checkpoint_payload.get("epoch"),
            "epoch_finished": checkpoint_payload.get("epoch_finished"),
            "iteration": checkpoint_payload.get("iteration"),
            "optimizer_updates": checkpoint_payload.get("optimizer_updates"),
            "checkpoint_reason": checkpoint_payload.get("checkpoint_reason"),
            "model_state_keys": len(model),
            "frozen_key_count": len(frozen_keys),
            "frozen_tensor_sha256": checkpoint_frozen_hash,
            "args": args,
            "optimizer": optimizer,
            "amp": amp,
        },
        "initializer": initializer_summary,
        "transition_audit": transition,
        "config": config_binding,
        "config_snapshots": snapshots,
        "datasets": datasets_binding,
        "category_complete_data_receipt": data_binding,
        "training_log": log_binding,
        "lineage": {
            "origin_output_dir_declared_by_checkpoint": EXPECTED_ORIGIN_OUTPUT_DIR,
            "origin_checkpoint_declared_by_training_log": str(origin_checkpoint),
            "origin_output_dir_exists_at_sealing": Path(
                EXPECTED_ORIGIN_OUTPUT_DIR
            ).exists(),
            "origin_checkpoint_exists_at_sealing": origin_checkpoint.is_file(),
            "durable_experiment_root": str(FORMAL_ROOT),
            "durable_checkpoint": checkpoint_record,
            "durable_checkpoint_matches_transition_audit": True,
            "origin_and_durable_locations_recorded_separately": True,
        },
        "core_sources_at_sealing": _core_source_binding(),
        "runtime_at_sealing": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "invariants": {
            "formal_checkpoint_sha256_exact": True,
            "formal_u0_initializer_sha256_exact": True,
            "transition_audit_recomputed_equal": True,
            "frozen_tensor_hash_equal_initializer_to_u100": True,
            "merged_r100_p50_teacher_frozen": True,
            "shared_patch_backbone_frozen": True,
            "batch_size": EXPECTED_BATCH_SIZE,
            "seed": EXPECTED_SEED,
            "optimizer_updates": EXPECTED_UPDATES,
            "optimizer_group_count": 2,
            "optimizer_state_steps_all_100": True,
            "amp_initial_scale": EXPECTED_AMP_SCALE,
            "amp_skipped_steps": 0,
            "effective_args_equal_checkpoint_args": True,
            "category_complete_data_receipt_replayed": True,
            "all_three_annotation_jsonl_files_fully_parsed": True,
            "origin_tmp_path_preserved_without_relabeling_durable_copy": True,
        },
    }
    del checkpoint_payload, initializer_payload
    gc.collect()
    return result


def build_receipt(*, output: Path, data_root: Path | None = None) -> dict[str, Any]:
    output = output.expanduser().resolve()
    if output.exists():
        raise U2TrainingReceiptError(f"refusing to overwrite existing receipt: {output}")
    payload = build_receipt_payload(data_root=data_root)
    receipt = _seal_payload(payload)
    _atomic_publish_fresh_json(output, receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--data-root", default=str(default_data_root()))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = build_receipt(
            output=Path(args.output), data_root=Path(args.data_root)
        )
        output_record = stable_file_record(Path(args.output), label="published U2 receipt")
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
